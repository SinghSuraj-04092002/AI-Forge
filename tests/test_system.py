"""
Tests for the Multiagent SDS (autogen-agentchat 0.4.x version).

Run with:
    pytest tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from multiagent_sds.api.app import app
from multiagent_sds.core.event_bus import EventBus
from multiagent_sds.core.state_store import StateStore
from multiagent_sds.models.domain import (
    AgentRole,
    CodeArtifact,
    CreateProjectRequest,
    DesignArtifact,
    PlanArtifact,
    ProjectContext,
    ProjectStatus,
    ReviewArtifact,
    TaskNode,
    TaskStatus,
    TestArtifact,
)
from multiagent_sds.orchestrator.engine import (
    OrchestratorEngine,
    TASK_DEPENDENCIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def llm_config() -> Dict[str, Any]:
    return {"model": "gpt-4o", "api_key": "test-key", "temperature": 0.3}


@pytest.fixture
def sample_ctx(llm_config) -> ProjectContext:
    return ProjectContext(
        title="Todo API",
        description="A simple RESTful todo list API with auth",
        requirements=["CRUD endpoints", "SQLite storage", "JWT auth"],
        llm_config=llm_config,
    )


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def make_artifacts() -> Dict[AgentRole, Any]:
    return {
        AgentRole.PLANNER: PlanArtifact(
            name="plan", content="{}",
            requirements=["req1"], milestones=["m1"], tech_stack=["Python"],
        ),
        AgentRole.DESIGNER: DesignArtifact(
            name="design", content="{}",
            architecture_diagram="[A]→[B]", api_schema="openapi: 3.0.0",
            data_models="class Item(BaseModel): ...", component_breakdown=["api"],
        ),
        AgentRole.CODER: CodeArtifact(
            name="code", content="{}",
            files={"main.py": "print('hello')"}, entry_point="main.py",
        ),
        AgentRole.TESTER: TestArtifact(
            name="tests", content="{}",
            test_files={"test_main.py": "def test_ok(): pass"},
            test_cases=["test ok"], coverage_estimate=0.85,
        ),
        AgentRole.REVIEWER: ReviewArtifact(
            name="review", content="{}",
            issues=[], suggestions=[], approved=True,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Domain model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainModels:
    def test_project_context_defaults(self, sample_ctx):
        assert sample_ctx.status == ProjectStatus.PENDING
        assert sample_ctx.project_id is not None
        assert len(sample_ctx.tasks) == 0
        assert len(sample_ctx.artifacts) == 0

    def test_log_event_appends(self, sample_ctx):
        sample_ctx.log_event("planner", "Started planning")
        assert len(sample_ctx.events) == 1
        evt = sample_ctx.events[0]
        assert evt["agent"] == "planner"
        assert evt["message"] == "Started planning"
        assert evt["level"] == "info"

    def test_log_event_updates_timestamp(self, sample_ctx):
        before = sample_ctx.updated_at
        sample_ctx.log_event("test", "msg")
        assert sample_ctx.updated_at >= before

    def test_task_node_duration_none_when_not_started(self):
        node = TaskNode(agent_role=AgentRole.PLANNER)
        assert node.duration_seconds() is None

    def test_task_node_duration_computed(self):
        node = TaskNode(agent_role=AgentRole.PLANNER)
        node.started_at = datetime(2024, 1, 1, 12, 0, 0)
        node.finished_at = datetime(2024, 1, 1, 12, 0, 45)
        assert node.duration_seconds() == 45.0

    def test_plan_artifact_role(self):
        art = PlanArtifact(name="p", content="{}", requirements=["r"])
        assert art.agent_role == AgentRole.PLANNER

    def test_review_artifact_approved_default_false(self):
        art = ReviewArtifact(name="r", content="{}")
        assert art.approved is False

    def test_code_artifact_files(self):
        art = CodeArtifact(
            name="c", content="{}",
            files={"a.py": "x=1", "b.py": "y=2"},
        )
        assert len(art.files) == 2
        assert art.files["a.py"] == "x=1"


# ─────────────────────────────────────────────────────────────────────────────
# State store tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStateStore:
    @pytest.mark.asyncio
    async def test_save_and_get(self, sample_ctx):
        s = StateStore()
        await s.save(sample_ctx)
        result = await s.get(sample_ctx.project_id)
        assert result is not None
        assert result.project_id == sample_ctx.project_id
        assert result.title == sample_ctx.title

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        s = StateStore()
        assert await s.get("does-not-exist") is None

    @pytest.mark.asyncio
    async def test_delete_existing(self, sample_ctx):
        s = StateStore()
        await s.save(sample_ctx)
        assert await s.delete(sample_ctx.project_id) is True
        assert await s.get(sample_ctx.project_id) is None

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self):
        s = StateStore()
        assert await s.delete("ghost-id") is False

    @pytest.mark.asyncio
    async def test_list_all_returns_all(self, llm_config):
        s = StateStore()
        projects = [
            ProjectContext(title=f"P{i}", description="d", llm_config=llm_config)
            for i in range(5)
        ]
        for p in projects:
            await s.save(p)
        all_p = await s.list_all()
        assert len(all_p) == 5

    @pytest.mark.asyncio
    async def test_concurrent_saves_are_safe(self, llm_config):
        s = StateStore()
        projects = [
            ProjectContext(title=f"P{i}", description="d", llm_config=llm_config)
            for i in range(30)
        ]
        await asyncio.gather(*[s.save(p) for p in projects])
        assert len(await s.list_all()) == 30

    @pytest.mark.asyncio
    async def test_overwrite_preserves_latest(self, sample_ctx):
        s = StateStore()
        await s.save(sample_ctx)
        sample_ctx.status = ProjectStatus.CODING
        await s.save(sample_ctx)
        result = await s.get(sample_ctx.project_id)
        assert result.status == ProjectStatus.CODING


# ─────────────────────────────────────────────────────────────────────────────
# Event bus tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEventBus:
    @pytest.mark.asyncio
    async def test_published_event_received(self):
        eb = EventBus()
        received = []

        async def collect():
            async for msg in eb.subscribe("proj-1"):
                received.append(msg)
                break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.02)
        await eb.publish("proj-1", "test_event", {"x": 1})
        await asyncio.wait_for(task, timeout=2)

        assert len(received) == 1
        data = json.loads(received[0].replace("data: ", "").strip())
        assert data["type"] == "test_event"
        assert data["data"]["x"] == 1

    @pytest.mark.asyncio
    async def test_event_not_leaked_to_other_project(self):
        eb = EventBus()
        received = []

        async def collect():
            async for msg in eb.subscribe("proj-A"):
                received.append(msg)
                break

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.02)
        await eb.publish("proj-B", "should_not_arrive", {})
        await eb.publish("proj-A", "correct_event", {})
        await asyncio.wait_for(task, timeout=2)

        assert len(received) == 1
        data = json.loads(received[0].replace("data: ", "").strip())
        assert data["type"] == "correct_event"

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_receive(self):
        eb = EventBus()
        buckets: list[list] = [[], [], []]

        async def collect(i):
            async for msg in eb.subscribe("multi"):
                buckets[i].append(msg)
                break

        tasks = [asyncio.create_task(collect(i)) for i in range(3)]
        await asyncio.sleep(0.02)
        await eb.publish("multi", "broadcast", {})
        await asyncio.gather(*[asyncio.wait_for(t, timeout=2) for t in tasks])

        for bucket in buckets:
            assert len(bucket) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_all_clears_subscribers(self):
        eb = EventBus()
        async with asyncio.timeout(0.5):
            # Subscribe but don't consume — just check cleanup doesn't raise
            q_added = False

            async def fake_sub():
                nonlocal q_added
                async for _ in eb.subscribe("cleanup-test"):
                    q_added = True
                    break

            t = asyncio.create_task(fake_sub())
            await asyncio.sleep(0.01)
            await eb.unsubscribe_all("cleanup-test")
            # After unsubscribe, publish should not raise
            await eb.publish("cleanup-test", "after_unsub", {})
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorEngine:
    def test_topo_sort_planner_first(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        order = engine._topo_sort()
        assert order[0] == AgentRole.PLANNER

    def test_topo_sort_reviewer_last(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        order = engine._topo_sort()
        assert order[-1] == AgentRole.REVIEWER

    def test_topo_sort_respects_all_dependencies(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        order = engine._topo_sort()
        idx = {role: i for i, role in enumerate(order)}
        for role, deps in TASK_DEPENDENCIES.items():
            for dep in deps:
                assert idx[dep] < idx[role], (
                    f"{dep.value} must precede {role.value} in topo sort"
                )

    def test_build_task_graph_creates_all_nodes(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        engine._build_task_graph()
        assert len(sample_ctx.tasks) == len(AgentRole)

    def test_build_task_graph_covers_all_roles(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        engine._build_task_graph()
        roles = {t.agent_role for t in sample_ctx.tasks.values()}
        assert roles == set(AgentRole)

    def test_dependency_edges_wired_correctly(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        engine._build_task_graph()
        designer = engine._role_to_task[AgentRole.DESIGNER]
        planner = engine._role_to_task[AgentRole.PLANNER]
        assert planner.task_id in designer.depends_on

    def test_reviewer_depends_on_coder_and_tester(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        engine._build_task_graph()
        reviewer = engine._role_to_task[AgentRole.REVIEWER]
        coder_id = engine._role_to_task[AgentRole.CODER].task_id
        tester_id = engine._role_to_task[AgentRole.TESTER].task_id
        assert coder_id in reviewer.depends_on
        assert tester_id in reviewer.depends_on

    def test_get_execution_plan_length(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        plan = engine.get_execution_plan()
        assert len(plan) == len(AgentRole)

    def test_get_execution_plan_steps_sequential(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        plan = engine.get_execution_plan()
        assert [p["step"] for p in plan] == list(range(1, len(AgentRole) + 1))

    def test_cascade_skip_marks_downstream(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        engine._build_task_graph()
        designer_tid = engine._role_to_task[AgentRole.DESIGNER].task_id
        engine._cascade_skip(designer_tid)
        skipped = {t.agent_role for t in sample_ctx.tasks.values() if t.status == TaskStatus.SKIPPED}
        assert AgentRole.CODER in skipped
        assert AgentRole.TESTER in skipped
        assert AgentRole.REVIEWER in skipped

    def test_cascade_skip_does_not_mark_upstream(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        engine._build_task_graph()
        designer_tid = engine._role_to_task[AgentRole.DESIGNER].task_id
        engine._cascade_skip(designer_tid)
        planner = sample_ctx.tasks[engine._role_to_task[AgentRole.PLANNER].task_id]
        assert planner.status == TaskStatus.PENDING

    def test_cascade_skip_planner_skips_all_others(self, sample_ctx):
        engine = OrchestratorEngine(sample_ctx)
        engine._build_task_graph()
        planner_tid = engine._role_to_task[AgentRole.PLANNER].task_id
        engine._cascade_skip(planner_tid)
        skipped = {t.agent_role for t in sample_ctx.tasks.values() if t.status == TaskStatus.SKIPPED}
        assert skipped == {AgentRole.DESIGNER, AgentRole.CODER, AgentRole.TESTER, AgentRole.REVIEWER}

    @pytest.mark.asyncio
    async def test_full_pipeline_mocked_agents_succeeds(self, sample_ctx):
        artifacts = make_artifacts()

        async def mock_run(ctx, task):
            art = artifacts[task.agent_role]
            task.status = TaskStatus.DONE
            task.artifact = art
            ctx.artifacts[task.agent_role] = art
            return art

        engine = OrchestratorEngine(sample_ctx)
        with patch.object(engine, '_init_agents'):
            engine._agents = {
                role: MagicMock(**{"run": AsyncMock(side_effect=mock_run)})
                for role in AgentRole
            }
            result = await engine.run()

        assert result.status == ProjectStatus.DONE
        assert len(result.artifacts) == len(AgentRole)

    @pytest.mark.asyncio
    async def test_pipeline_all_tasks_done_on_success(self, sample_ctx):
        artifacts = make_artifacts()

        async def mock_run(ctx, task):
            art = artifacts[task.agent_role]
            task.status = TaskStatus.DONE
            task.artifact = art
            ctx.artifacts[task.agent_role] = art
            return art

        engine = OrchestratorEngine(sample_ctx)
        with patch.object(engine, '_init_agents'):
            engine._agents = {
                role: MagicMock(**{"run": AsyncMock(side_effect=mock_run)})
                for role in AgentRole
            }
            result = await engine.run()

        done_tasks = [t for t in result.tasks.values() if t.status == TaskStatus.DONE]
        assert len(done_tasks) == len(AgentRole)

    @pytest.mark.asyncio
    async def test_pipeline_planner_failure_cascades(self, sample_ctx):
        artifacts = make_artifacts()

        async def fail_planner(ctx, task):
            if task.agent_role == AgentRole.PLANNER:
                task.status = TaskStatus.FAILED
                raise RuntimeError("LLM timeout")
            art = artifacts[task.agent_role]
            task.status = TaskStatus.DONE
            task.artifact = art
            ctx.artifacts[task.agent_role] = art
            return art

        engine = OrchestratorEngine(sample_ctx)
        with patch.object(engine, '_init_agents'):
            engine._agents = {
                role: MagicMock(**{"run": AsyncMock(side_effect=fail_planner)})
                for role in AgentRole
            }
            result = await engine.run()

        assert result.status == ProjectStatus.FAILED
        skipped = [t for t in result.tasks.values() if t.status == TaskStatus.SKIPPED]
        assert len(skipped) >= 4

    @pytest.mark.asyncio
    async def test_pipeline_emits_events(self, sample_ctx):
        """Check that events are logged on the context during a successful run."""
        artifacts = make_artifacts()

        async def mock_run(ctx, task):
            art = artifacts[task.agent_role]
            task.status = TaskStatus.DONE
            task.artifact = art
            ctx.artifacts[task.agent_role] = art
            return art

        engine = OrchestratorEngine(sample_ctx)
        with patch.object(engine, '_init_agents'):
            engine._agents = {
                role: MagicMock(**{"run": AsyncMock(side_effect=mock_run)})
                for role in AgentRole
            }
            result = await engine.run()

        assert len(result.events) > 0
        messages = [e["message"] for e in result.events]
        assert any("Pipeline started" in m for m in messages)
        assert any("successfully" in m for m in messages)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["autogen"] == "agentchat-0.4.x"

    def test_list_projects_empty(self, client):
        resp = client.get("/projects")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_project_returns_202(self, client):
        with patch("multiagent_sds.api.app._run_pipeline", new_callable=AsyncMock):
            resp = client.post(
                "/projects",
                json={
                    "title": "My App",
                    "description": "A test project",
                    "requirements": ["feature A", "feature B"],
                },
            )
        assert resp.status_code == 202
        data = resp.json()
        assert "project_id" in data
        assert "execution_plan" in data
        assert "stream_url" in data
        assert "status_url" in data
        assert data["message"] == "Pipeline started in background"

    def test_create_project_returns_execution_plan(self, client):
        with patch("multiagent_sds.api.app._run_pipeline", new_callable=AsyncMock):
            resp = client.post(
                "/projects",
                json={"title": "T", "description": "D"},
            )
        plan = resp.json()["execution_plan"]
        assert len(plan) == len(AgentRole)
        roles_in_plan = [p["role"] for p in plan]
        assert "planner" in roles_in_plan
        assert "reviewer" in roles_in_plan

    def test_get_project_not_found(self, client):
        resp = client.get("/projects/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_project_not_found(self, client):
        resp = client.delete("/projects/nonexistent-id")
        assert resp.status_code == 404

    def test_execution_plan_no_project(self, client):
        """Plan endpoint returns generic plan even for unknown project IDs."""
        resp = client.get("/projects/any-id/plan")
        assert resp.status_code == 200
        assert "execution_plan" in resp.json()

    def test_artifacts_not_found(self, client):
        resp = client.get("/projects/nonexistent/artifacts")
        assert resp.status_code == 404

    def test_create_then_get_project(self, client):
        with patch("multiagent_sds.api.app._run_pipeline", new_callable=AsyncMock):
            create_resp = client.post(
                "/projects",
                json={"title": "Get Me", "description": "desc"},
            )
        project_id = create_resp.json()["project_id"]
        get_resp = client.get(f"/projects/{project_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["project_id"] == project_id
        assert data["title"] == "Get Me"

    def test_create_then_delete_project(self, client):
        with patch("multiagent_sds.api.app._run_pipeline", new_callable=AsyncMock):
            create_resp = client.post(
                "/projects",
                json={"title": "Delete Me", "description": "desc"},
            )
        project_id = create_resp.json()["project_id"]
        del_resp = client.delete(f"/projects/{project_id}")
        assert del_resp.status_code == 204
        get_resp = client.get(f"/projects/{project_id}")
        assert get_resp.status_code == 404

    def test_create_multiple_projects_listed(self, client):
        with patch("multiagent_sds.api.app._run_pipeline", new_callable=AsyncMock):
            for i in range(3):
                client.post("/projects", json={"title": f"P{i}", "description": "d"})
        resp = client.get("/projects")
        assert resp.status_code == 200
        assert len(resp.json()) >= 3

    def test_custom_llm_config_accepted(self, client):
        with patch("multiagent_sds.api.app._run_pipeline", new_callable=AsyncMock):
            resp = client.post(
                "/projects",
                json={
                    "title": "Custom LLM",
                    "description": "Using custom config",
                    "llm_config": {
                        "model": "gpt-4o-mini",
                        "api_key": "custom-key",
                        "temperature": 0.1,
                    },
                },
            )
        assert resp.status_code == 202
