"""
Coder Agent
───────────
Generates implementation files from the system design.

Uses AssistantAgent.on_messages_stream() to stream tokens as they arrive —
these are forwarded to the EventBus so the SSE stream shows real-time
code generation progress.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken

from multiagent_sds.agents.base_agent import BaseAgent, build_model_client
from multiagent_sds.agents.planner_agent import PlannerAgent
from multiagent_sds.core.event_bus import bus
from multiagent_sds.models.domain import (
    AgentRole,
    Artifact,
    CodeArtifact,
    ProjectContext,
    TaskNode,
)

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    def _system_prompt(self) -> str:
        return (
            "You are an expert Python engineer. "
            "Given a system design, produce production-quality async Python code. "
            "Return a JSON object with keys: "
            "\"files\" (dict[str, str] mapping filename → full file content), "
            "\"language\" (str, e.g. \"python\"), "
            "\"entry_point\" (str, e.g. \"main.py\"), "
            "\"implementation_notes\" (str – key decisions). "
            "Write clean, well-commented, fully async code using FastAPI + Pydantic v2. "
            "Reply with ONLY valid JSON."
        )

    async def execute(self, ctx: ProjectContext, task: TaskNode) -> Artifact:
        design_artifact = ctx.artifacts.get(AgentRole.DESIGNER)
        plan_artifact = ctx.artifacts.get(AgentRole.PLANNER)
        design_content = design_artifact.content if design_artifact else "{}"
        plan_content = plan_artifact.content if plan_artifact else "{}"

        model_client = build_model_client(self.llm_config)

        # Lead coder drafts; senior reviewer asks for one round of improvements
        lead_coder = AssistantAgent(
            name="lead_coder",
            model_client=model_client,
            system_message=self._system_prompt(),
        )
        senior_reviewer = AssistantAgent(
            name="senior_coder_reviewer",
            model_client=model_client,
            system_message=(
                "You are a senior engineer reviewing code output. "
                "Check that: files are complete (not truncated), async is used correctly, "
                "error handling is present, and types are correct. "
                "If everything looks good, reply: LGTM\n<the_json>. "
                "Otherwise ask for one specific fix."
            ),
        )

        from autogen_agentchat.conditions import TextMentionTermination
        termination = TextMentionTermination("LGTM") | MaxMessageTermination(6)
        team = RoundRobinGroupChat(
            participants=[lead_coder, senior_reviewer],
            termination_condition=termination,
        )

        task_prompt = (
            f"Project: {ctx.title}\nDescription: {ctx.description}\n\n"
            f"Plan:\n{plan_content}\n\n"
            f"Design:\n{design_content}\n\n"
            "Implement the full codebase."
        )

        # Stream tokens → EventBus for real-time SSE progress
        raw_chunks: list[str] = []
        async for message in team.run_stream(
            task=task_prompt, cancellation_token=CancellationToken()
        ):
            from autogen_agentchat.base import TaskResult
            if isinstance(message, TaskResult):
                break
            content = getattr(message, "content", "")
            if isinstance(content, str) and content:
                raw_chunks.append(content)
                # Emit progress chunk to SSE stream
                await bus.publish(
                    ctx.project_id,
                    "code_stream_chunk",
                    {"role": self.role.value, "chunk": content[:200]},
                )

        raw = PlannerAgent._extract_json_content_from_texts(raw_chunks)

        try:
            clean = raw.replace("LGTM", "").strip()
            data: Dict[str, Any] = json.loads(clean)
        except json.JSONDecodeError:
            data = {
                "files": {"main.py": raw},
                "language": "python",
                "entry_point": "main.py",
                "implementation_notes": "Raw output stored in main.py",
            }

        return CodeArtifact(
            name="implementation",
            content=json.dumps(data, indent=2),
            files=data.get("files", {}),
            language=data.get("language", "python"),
            entry_point=data.get("entry_point", "main.py"),
            metadata={"file_count": len(data.get("files", {}))},
        )
