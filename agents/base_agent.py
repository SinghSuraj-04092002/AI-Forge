"""
Base agent — wraps autogen_agentchat.agents.AssistantAgent.

Key autogen-agentchat 0.4.x patterns used here:
  • OpenAIChatCompletionClient  (autogen_ext.models.openai)
  • AssistantAgent              (autogen_agentchat.agents)
  • TextMessage                 (autogen_agentchat.messages)
  • CancellationToken           (autogen_core)
  • MaxMessageTermination       (autogen_agentchat.conditions)

Each agent creates its own AssistantAgent + model client so agents can run
concurrently without sharing mutable state.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from multiagent_sds.core.event_bus import bus
from multiagent_sds.models.domain import (
    Artifact,
    AgentRole,
    ProjectContext,
    TaskNode,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def build_model_client(llm_config: Dict[str, Any]) -> OpenAIChatCompletionClient:
    """
    Construct an OpenAIChatCompletionClient from the project's llm_config dict.

    Expected llm_config shape:
        {
            "model": "gpt-4o",          # required
            "api_key": "sk-...",        # required
            "temperature": 0.3,         # optional
            "base_url": "...",          # optional (Azure / local)
            "api_version": "...",       # optional (Azure)
        }
    The config_list convention from old pyautogen is NOT used — pass fields flat.
    """
    cfg = dict(llm_config)

    # Support legacy config_list format transparently
    if "config_list" in cfg and isinstance(cfg["config_list"], list):
        first = cfg["config_list"][0]
        cfg = {**first, **{k: v for k, v in cfg.items() if k != "config_list"}}

    model = cfg.get("model", "gpt-4o")
    api_key = cfg.get("api_key", "")
    temperature = float(cfg.get("temperature", 0.3))

    kwargs: Dict[str, Any] = dict(model=model, api_key=api_key, temperature=temperature)
    if "base_url" in cfg:
        kwargs["base_url"] = cfg["base_url"]
    if "api_version" in cfg:
        kwargs["api_version"] = cfg["api_version"]

    return OpenAIChatCompletionClient(**kwargs)


class BaseAgent(ABC):
    """
    Async base class for all SDS agents.

    Subclasses must implement:
        role         – AgentRole class variable
        _system_prompt() – str
        execute()    – async method returning Artifact
    """

    role: AgentRole  # must be set by each subclass

    def __init__(self, llm_config: Dict[str, Any]) -> None:
        self.llm_config = llm_config
        self._model_client: Optional[OpenAIChatCompletionClient] = None
        self._assistant: Optional[AssistantAgent] = None

    # ── Lazy agent construction ───────────────────────────────────────────────

    def _get_assistant(self) -> AssistantAgent:
        """Build (or return cached) AssistantAgent for this role."""
        if self._assistant is None:
            self._model_client = build_model_client(self.llm_config)
            self._assistant = AssistantAgent(
                name=f"{self.role.value}_agent",
                model_client=self._model_client,
                system_message=self._system_prompt(),
            )
        return self._assistant

    # ── Core async chat helper ────────────────────────────────────────────────

    async def _chat(self, prompt: str) -> str:
        """
        Send a single prompt to the AssistantAgent and return its last text reply.

        Uses a one-turn RoundRobinGroupChat capped at 2 messages
        (user message → assistant reply) so we always get a clean single response.
        """
        assistant = self._get_assistant()

        # Fresh termination condition per call
        termination = MaxMessageTermination(max_messages=2)
        team = RoundRobinGroupChat(
            participants=[assistant],
            termination_condition=termination,
        )

        result = await team.run(
            task=prompt,
            cancellation_token=CancellationToken(),
        )

        # Extract last text content from the result messages
        for msg in reversed(result.messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                content = msg.content.strip()
                # Skip the echoed task prompt (source == "user")
                if hasattr(msg, "source") and msg.source == "user":
                    continue
                if content:
                    return content

        # Fallback: join all non-user message content
        parts = [
            msg.content
            for msg in result.messages
            if hasattr(msg, "content")
            and isinstance(msg.content, str)
            and getattr(msg, "source", "") != "user"
        ]
        return "\n".join(parts)

    # ── Retry-aware public runner ─────────────────────────────────────────────

    async def run(self, ctx: ProjectContext, task: TaskNode) -> Artifact:
        """
        Called by the Orchestrator. Handles lifecycle, retries, and event emission.
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        await bus.publish(
            ctx.project_id,
            "task_started",
            {"role": self.role.value, "task_id": task.task_id},
        )

        last_exc: Optional[Exception] = None
        for attempt in range(task.max_retries + 1):
            try:
                artifact = await self.execute(ctx, task)
                task.status = TaskStatus.DONE
                task.finished_at = datetime.utcnow()
                task.artifact = artifact
                ctx.artifacts[self.role] = artifact
                ctx.log_event(self.role.value, f"Done (attempt {attempt + 1})")
                await bus.publish(
                    ctx.project_id,
                    "task_done",
                    {
                        "role": self.role.value,
                        "task_id": task.task_id,
                        "artifact_id": artifact.artifact_id,
                    },
                )
                return artifact
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                task.retries = attempt + 1
                logger.warning("[%s] attempt %d failed: %s", self.role.value, attempt + 1, exc)
                ctx.log_event(self.role.value, f"Attempt {attempt + 1} failed: {exc}", level="warning")
                await bus.publish(
                    ctx.project_id,
                    "task_retry",
                    {"role": self.role.value, "attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < task.max_retries:
                    await asyncio.sleep(2 ** attempt)

        task.status = TaskStatus.FAILED
        task.finished_at = datetime.utcnow()
        task.error = str(last_exc)
        await bus.publish(
            ctx.project_id,
            "task_failed",
            {"role": self.role.value, "task_id": task.task_id, "error": str(last_exc)},
        )
        raise RuntimeError(
            f"Agent {self.role.value} failed after {task.max_retries + 1} attempts"
        ) from last_exc

    # ── Subclass contract ─────────────────────────────────────────────────────

    @abstractmethod
    def _system_prompt(self) -> str:
        """Return the system-level instruction string for this agent."""

    @abstractmethod
    async def execute(self, ctx: ProjectContext, task: TaskNode) -> Artifact:
        """Perform the agent's work and return the typed Artifact."""
