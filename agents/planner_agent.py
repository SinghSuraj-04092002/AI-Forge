"""
Planner Agent
─────────────
Converts raw project requirements into a structured plan using a
multi-turn AutoGen conversation between a planner and a critic.

autogen_agentchat 0.4.x used:
  • AssistantAgent  – planner + critic roles
  • RoundRobinGroupChat  – drives the conversation
  • TextMentionTermination – stops when critic says APPROVED
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from multiagent_sds.agents.base_agent import BaseAgent, build_model_client
from multiagent_sds.models.domain import (
    AgentRole,
    Artifact,
    PlanArtifact,
    ProjectContext,
    TaskNode,
)

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    def _system_prompt(self) -> str:
        return (
            "You are an expert software project planner. "
            "Given a project description and requirements, produce a detailed plan "
            "as a JSON object with exactly these keys: "
            "\"requirements\" (list[str] – refined list), "
            "\"milestones\" (list[str] – ordered delivery milestones), "
            "\"tech_stack\" (list[str] – recommended technologies), "
            "\"summary\" (str – 2-3 sentence overview). "
            "Reply with ONLY valid JSON — no markdown fences, no extra text."
        )

    async def execute(self, ctx: ProjectContext, task: TaskNode) -> Artifact:
        # ── Two-agent planning conversation ───────────────────────────────────
        # Planner drafts the plan; Critic validates and either requests
        # improvements or approves with the keyword APPROVED.
        model_client = build_model_client(self.llm_config)

        planner = AssistantAgent(
            name="planner",
            model_client=model_client,
            system_message=self._system_prompt(),
        )
        critic = AssistantAgent(
            name="planning_critic",
            model_client=model_client,
            system_message=(
                "You are a critical technical reviewer. "
                "Review the project plan JSON produced by the planner. "
                "If it is complete and correct, reply only with: APPROVED\n{the_json}\n"
                "If it needs improvement, give specific feedback in one sentence "
                "and ask the planner to revise. "
                "Do NOT rewrite the JSON yourself."
            ),
        )

        termination = TextMentionTermination("APPROVED") | MaxMessageTermination(6)
        team = RoundRobinGroupChat(
            participants=[planner, critic],
            termination_condition=termination,
        )

        task_prompt = (
            f"Project title: {ctx.title}\n"
            f"Description: {ctx.description}\n"
            f"Raw requirements:\n" + "\n".join(f"- {r}" for r in ctx.requirements)
        )

        result = await team.run(task=task_prompt, cancellation_token=CancellationToken())

        # Extract the last planner message that looks like JSON
        raw = self._extract_json_content(result.messages)
        logger.debug("[planner] raw output: %s", raw[:300])

        try:
            # Strip APPROVED prefix if critic echoed it
            clean = raw.replace("APPROVED", "").strip()
            data: Dict[str, Any] = json.loads(clean)
        except json.JSONDecodeError:
            data = {
                "requirements": ctx.requirements,
                "milestones": ["Phase 1: Setup", "Phase 2: Core features", "Phase 3: Polish & deploy"],
                "tech_stack": ["Python", "FastAPI"],
                "summary": raw[:300],
            }

        return PlanArtifact(
            name="project_plan",
            content=json.dumps(data, indent=2),
            requirements=data.get("requirements", ctx.requirements),
            milestones=data.get("milestones", []),
            tech_stack=data.get("tech_stack", []),
            metadata={"conversation_turns": len(result.messages)},
        )

    @staticmethod
    def _extract_json_content(messages) -> str:
        """Find the last message that contains a JSON object."""
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue
            stripped = content.strip()
            if stripped.startswith("{"):
                return stripped
            if "```json" in stripped:
                start = stripped.find("```json") + 7
                end = stripped.find("```", start)
                return stripped[start:end].strip()
            if "```" in stripped:
                start = stripped.find("```") + 3
                end = stripped.find("```", start)
                candidate = stripped[start:end].strip()
                if candidate.startswith("{"):
                    return candidate
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return "{}"

    @staticmethod
    def _extract_json_content_from_texts(chunks: list[str]) -> str:
        """Find JSON content from a list of text chunks (for streaming)."""
        # Try joining all chunks and finding a JSON object
        full = "".join(chunks)
        stripped = full.strip()
        if stripped.startswith("{"):
            return stripped
        if "```json" in stripped:
            start = stripped.find("```json") + 7
            end = stripped.find("```", start)
            if end > start:
                return stripped[start:end].strip()
        if "```" in stripped:
            start = stripped.find("```") + 3
            end = stripped.find("```", start)
            candidate = stripped[start:end].strip()
            if candidate.startswith("{"):
                return candidate
        # Try each chunk individually in reverse
        for chunk in reversed(chunks):
            c = chunk.strip()
            if c.startswith("{"):
                return c
        return full or "{}"
