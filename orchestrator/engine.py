"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ORCHESTRATOR ENGINE                                  ║
║                                                                              ║
║  The brain of the multiagent SDS.  Responsibilities:                        ║
║  ① Build the dependency DAG of agent tasks                                  ║
║  ② Resolve execution order (topological sort – Kahn's algorithm)            ║
║  ③ Schedule agents concurrently where dependencies allow                    ║
║  ④ Drive the project through the lifecycle state machine                    ║
║  ⑤ Handle retries, failures, and cascade-skip on downstream tasks           ║
║  ⑥ Emit structured events to the EventBus for real-time SSE streaming       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type

from multiagent_sds.agents.base_agent import BaseAgent
from multiagent_sds.agents.coder_agent import CoderAgent
from multiagent_sds.agents.designer_agent import DesignerAgent
from multiagent_sds.agents.planner_agent import PlannerAgent
from multiagent_sds.agents.reviewer_agent import ReviewerAgent
from multiagent_sds.agents.tester_agent import TesterAgent
from multiagent_sds.core.event_bus import bus
from multiagent_sds.core.state_store import store
from multiagent_sds.models.domain import (
    AgentRole,
    ProjectContext,
    ProjectStatus,
    TaskNode,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# ── Dependency declarations ────────────────────────────────────────────────────
# Maps each AgentRole to the roles it must wait for before starting.
TASK_DEPENDENCIES: Dict[AgentRole, List[AgentRole]] = {
    AgentRole.PLANNER:  [],
    AgentRole.DESIGNER: [AgentRole.PLANNER],
    AgentRole.CODER:    [AgentRole.DESIGNER],
    AgentRole.TESTER:   [AgentRole.CODER],
    AgentRole.REVIEWER: [AgentRole.CODER, AgentRole.TESTER],
}

# Which project status becomes active when an agent starts running
STATUS_PIPELINE: Dict[AgentRole, ProjectStatus] = {
    AgentRole.PLANNER:  ProjectStatus.PLANNING,
    AgentRole.DESIGNER: ProjectStatus.DESIGNING,
    AgentRole.CODER:    ProjectStatus.CODING,
    AgentRole.TESTER:   ProjectStatus.TESTING,
    AgentRole.REVIEWER: ProjectStatus.REVIEWING,
}

AGENT_CLASSES: Dict[AgentRole, Type[BaseAgent]] = {
    AgentRole.PLANNER:  PlannerAgent,
    AgentRole.DESIGNER: DesignerAgent,
    AgentRole.CODER:    CoderAgent,
    AgentRole.TESTER:   TesterAgent,
    AgentRole.REVIEWER: ReviewerAgent,
}


# ── Orchestrator ───────────────────────────────────────────────────────────────

class OrchestratorEngine:
    """
    Async orchestrator that manages the full software development lifecycle.

    Usage
    -----
    engine = OrchestratorEngine(ctx)
    completed_ctx = await engine.run()
    """

    def __init__(self, ctx: ProjectContext) -> None:
        self.ctx = ctx
        self._agents: Dict[AgentRole, BaseAgent] = {}
        self._role_to_task: Dict[AgentRole, TaskNode] = {}

    # ── Public entry point ─────────────────────────────────────────────────────

    async def run(self) -> ProjectContext:
        """
        Execute the full agent pipeline.
        Returns the ProjectContext in its final state (DONE or FAILED).
        """
        ctx = self.ctx
        try:
            await self._emit("orchestrator", "Pipeline started", "info")
            ctx.status = ProjectStatus.PENDING
            await store.save(ctx)

            self._init_agents()
            self._build_task_graph()
            await store.save(ctx)

            await self._execute_pipeline()

            ctx.status = ProjectStatus.DONE
            await self._emit("orchestrator", "Pipeline completed successfully ✓")

        except Exception as exc:  # noqa: BLE001
            ctx.status = ProjectStatus.FAILED
            ctx.log_event("orchestrator", f"Pipeline failed: {exc}", level="error")
            logger.exception("Orchestrator pipeline failed for project %s", ctx.project_id)
            await bus.publish(ctx.project_id, "pipeline_failed", {"error": str(exc)})

        finally:
            ctx.updated_at = datetime.utcnow()
            await store.save(ctx)
            await bus.publish(
                ctx.project_id,
                "pipeline_finished",
                {"status": ctx.status.value},
            )

        return ctx

    # ── Agent initialisation ───────────────────────────────────────────────────

    def _init_agents(self) -> None:
        """
        Instantiate each specialist agent with the project's LLM config.
        Agents are created fresh per project so their model clients and
        AssistantAgent state are fully isolated.
        """
        for role, agent_cls in AGENT_CLASSES.items():
            self._agents[role] = agent_cls(llm_config=self.ctx.llm_config)
        logger.info("Initialised %d agents for project %s", len(self._agents), self.ctx.project_id)

    # ── Task graph construction ────────────────────────────────────────────────

    def _build_task_graph(self) -> None:
        """
        Create TaskNode objects and wire dependency edges into the project context.

        Each node's `depends_on` list holds *task_ids* (not role names) so the
        scheduler can look up readiness with a simple set-membership check.
        """
        # Pass 1: create nodes
        for role in AgentRole:
            node = TaskNode(agent_role=role)
            self._role_to_task[role] = node
            self.ctx.tasks[node.task_id] = node

        # Pass 2: wire edges
        for role, dep_roles in TASK_DEPENDENCIES.items():
            node = self._role_to_task[role]
            node.depends_on = [
                self._role_to_task[dep_role].task_id for dep_role in dep_roles
            ]

        await_count = sum(len(n.depends_on) for n in self.ctx.tasks.values())
        logger.debug(
            "Task graph: %d nodes, %d dependency edges",
            len(self.ctx.tasks),
            await_count,
        )
        # Note: bus.publish is called from _execute_pipeline once the event loop is running

    # ── Pipeline execution ─────────────────────────────────────────────────────

    async def _execute_pipeline(self) -> None:
        """
        Topologically sorted execution with parallel scheduling.

        Algorithm
        ---------
        Each iteration of the main loop:
          1. Find all PENDING tasks whose `depends_on` are fully in `completed`.
          2. Launch them as asyncio.Tasks (running concurrently).
          3. Wait for the next completion via asyncio.wait(FIRST_COMPLETED).
          4. Update completed/failed sets and repeat.

        Termination: when every task is in a terminal state
        (DONE | FAILED | SKIPPED) or no progress can be made.
        """
        ctx = self.ctx
        completed: Set[str] = set()
        failed: Set[str] = set()
        in_flight: Dict[str, asyncio.Task] = {}   # task_id → asyncio coroutine task

        all_task_ids = set(ctx.tasks.keys())

        # Emit the graph now that we're inside the running event loop
        await bus.publish(
            ctx.project_id,
            "graph_ready",
            {
                "total_tasks": len(ctx.tasks),
                "nodes": [
                    {"task_id": t.task_id, "role": t.agent_role.value, "depends_on": t.depends_on}
                    for t in ctx.tasks.values()
                ],
            },
        )

        while True:
            # ── Identify tasks that are now unblocked ───────────────────────
            ready: List[TaskNode] = [
                t
                for t in ctx.tasks.values()
                if t.status == TaskStatus.PENDING
                and all(dep in completed for dep in t.depends_on)
                and t.task_id not in in_flight
            ]

            # ── Launch all ready tasks concurrently ─────────────────────────
            for task_node in ready:
                role = task_node.agent_role
                # Advance the project status to reflect the current phase
                ctx.status = STATUS_PIPELINE.get(role, ctx.status)
                await store.save(ctx)

                await self._emit(
                    "orchestrator",
                    f"Scheduling {role.value} agent (task {task_node.task_id[:8]}…)",
                )

                coro_task = asyncio.create_task(
                    self._run_single_task(task_node),
                    name=f"agent-{role.value}",
                )
                in_flight[task_node.task_id] = coro_task

            # ── Nothing running and nothing ready → we're done or stuck ─────
            if not in_flight:
                break

            # ── Await the soonest completion ────────────────────────────────
            done_set, _ = await asyncio.wait(
                in_flight.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for done_coro in done_set:
                # Reverse-lookup: which task_id does this asyncio.Task belong to?
                tid = next(
                    tid for tid, ct in in_flight.items() if ct is done_coro
                )
                in_flight.pop(tid)

                exc = done_coro.exception()
                if exc:
                    logger.error("Task %s raised: %s", tid, exc)
                    failed.add(tid)
                    # Cancel anything in-flight that transitively depends on this task
                    self._cascade_skip(tid)
                    await store.save(ctx)
                else:
                    completed.add(tid)

            # ── Check for full termination ───────────────────────────────────
            terminal = completed | failed | {
                t.task_id
                for t in ctx.tasks.values()
                if t.status == TaskStatus.SKIPPED
            }
            if terminal >= all_task_ids:
                break

            # Safety: if nothing is in-flight and no task became ready this
            # iteration, we have an unresolvable dependency cycle → break.
            if not in_flight and not ready:
                logger.error("Pipeline stalled — possible dependency cycle detected")
                break

        if failed:
            failed_roles = ", ".join(
                ctx.tasks[tid].agent_role.value for tid in failed
            )
            raise RuntimeError(f"Pipeline failed at: {failed_roles}")

    async def _run_single_task(self, task: TaskNode) -> None:
        """Delegate a single task to its specialist agent."""
        agent = self._agents[task.agent_role]
        await agent.run(self.ctx, task)
        await store.save(self.ctx)

    # ── Cascade skip ───────────────────────────────────────────────────────────

    def _cascade_skip(self, failed_tid: str) -> None:
        """
        BFS over the reverse dependency graph.
        Marks every task that (transitively) depends on `failed_tid` as SKIPPED
        so they never enter a running state.
        """
        # Build reverse adjacency: child → {parents that depend on it}
        reverse: Dict[str, Set[str]] = defaultdict(set)
        for t in self.ctx.tasks.values():
            for dep in t.depends_on:
                reverse[dep].add(t.task_id)

        queue: deque[str] = deque([failed_tid])
        visited: Set[str] = {failed_tid}

        while queue:
            current = queue.popleft()
            for child_id in reverse[current]:
                if child_id not in visited:
                    visited.add(child_id)
                    queue.append(child_id)
                    node = self.ctx.tasks[child_id]
                    if node.status == TaskStatus.PENDING:
                        node.status = TaskStatus.SKIPPED
                        self.ctx.log_event(
                            "orchestrator",
                            f"Skipped {node.agent_role.value} — upstream task failed",
                            level="warning",
                        )

    # ── Utility ────────────────────────────────────────────────────────────────

    async def _emit(self, agent: str, message: str, level: str = "info") -> None:
        self.ctx.log_event(agent, message, level=level)
        await bus.publish(self.ctx.project_id, "log", {
            "agent": agent,
            "level": level,
            "message": message,
        })

    # ── Dry-run introspection ──────────────────────────────────────────────────

    def get_execution_plan(self) -> List[Dict[str, Any]]:
        """
        Return the topologically sorted execution plan with parallelism hints.
        Safe to call before `run()` — no agents are instantiated.
        """
        order = self._topo_sort()
        return [
            {
                "step": i + 1,
                "role": role.value,
                "depends_on": [d.value for d in TASK_DEPENDENCIES[role]],
                "parallel_with": self._parallel_candidates(role, order),
            }
            for i, role in enumerate(order)
        ]

    def _topo_sort(self) -> List[AgentRole]:
        """Kahn's algorithm over TASK_DEPENDENCIES."""
        in_degree: Dict[AgentRole, int] = {
            r: len(deps) for r, deps in TASK_DEPENDENCIES.items()
        }
        graph: Dict[AgentRole, List[AgentRole]] = defaultdict(list)
        for role, deps in TASK_DEPENDENCIES.items():
            for dep in deps:
                graph[dep].append(role)

        queue: deque[AgentRole] = deque(
            [r for r, d in in_degree.items() if d == 0]
        )
        result: List[AgentRole] = []
        while queue:
            role = queue.popleft()
            result.append(role)
            for neighbour in graph[role]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)
        return result

    def _parallel_candidates(
        self, role: AgentRole, order: List[AgentRole]
    ) -> List[str]:
        """Return roles with the same dependency set — they can run together."""
        deps = frozenset(TASK_DEPENDENCIES[role])
        return [
            other.value
            for other in order
            if other != role and frozenset(TASK_DEPENDENCIES[other]) == deps
        ]
