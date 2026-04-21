"""
Designer Agent
──────────────
Translates the project plan into system architecture, API schema, and data models.

Uses a SelectorGroupChat so a selector LLM can route between the architect
(who drafts) and the api_designer (who refines the schema detail) before
the output is finalised.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken

from multiagent_sds.agents.base_agent import BaseAgent, build_model_client
from multiagent_sds.agents.planner_agent import PlannerAgent
from multiagent_sds.models.domain import (
    AgentRole,
    Artifact,
    DesignArtifact,
    ProjectContext,
    TaskNode,
)

logger = logging.getLogger(__name__)


class DesignerAgent(BaseAgent):
    role = AgentRole.DESIGNER

    def _system_prompt(self) -> str:
        return (
            "You are a senior software architect. "
            "Produce a system design as JSON with keys: "
            "\"architecture_diagram\" (ASCII art), "
            "\"api_schema\" (OpenAPI 3.0 YAML stub as a string), "
            "\"data_models\" (Pydantic model definitions as a string), "
            "\"component_breakdown\" (list[str] of service/module names), "
            "\"design_notes\" (str). "
            "Reply with ONLY valid JSON."
        )

    async def execute(self, ctx: ProjectContext, task: TaskNode) -> Artifact:
        plan_artifact = ctx.artifacts.get(AgentRole.PLANNER)
        plan_content = plan_artifact.content if plan_artifact else "{}"

        model_client = build_model_client(self.llm_config)

        # Three-role design conversation:
        #   architect   → high-level structure
        #   api_designer → refines API schema detail
        #   design_critic → approves or requests changes
        architect = AssistantAgent(
            name="architect",
            model_client=model_client,
            description="Produces high-level system architecture and component breakdown.",
            system_message=(
                "You are a software architect. Draft the system architecture, "
                "component breakdown, and ASCII architecture diagram as part of a "
                "full design JSON. Focus on structure, not API detail."
            ),
        )
        api_designer = AssistantAgent(
            name="api_designer",
            model_client=model_client,
            description="Refines and completes the API schema and data models.",
            system_message=(
                "You are an API design specialist. Given an architectural draft, "
                "refine the api_schema (OpenAPI 3.0 YAML) and data_models (Pydantic). "
                "Produce the complete merged JSON."
            ),
        )
        design_critic = AssistantAgent(
            name="design_critic",
            model_client=model_client,
            description="Reviews design completeness and approves with APPROVED.",
            system_message=(
                "You are a design reviewer. If the design JSON is complete and coherent, "
                "reply: APPROVED\n<the_json>. Otherwise give one-line feedback."
            ),
        )

        termination = TextMentionTermination("APPROVED") | MaxMessageTermination(8)
        team = SelectorGroupChat(
            participants=[architect, api_designer, design_critic],
            model_client=model_client,
            termination_condition=termination,
        )

        task_prompt = (
            f"Project: {ctx.title}\nDescription: {ctx.description}\n\n"
            f"Approved project plan:\n{plan_content}\n\n"
            "Produce the full system design JSON."
        )

        result = await team.run(task=task_prompt, cancellation_token=CancellationToken())
        raw = PlannerAgent._extract_json_content(result.messages)

        try:
            clean = raw.replace("APPROVED", "").strip()
            data: Dict[str, Any] = json.loads(clean)
        except json.JSONDecodeError:
            data = {
                "architecture_diagram": "[ Client ] --> [ API ] --> [ DB ]",
                "api_schema": "openapi: 3.0.0\ninfo:\n  title: API\n  version: 1.0.0",
                "data_models": "# See implementation",
                "component_breakdown": ["api", "services", "models", "db"],
                "design_notes": raw[:400],
            }

        return DesignArtifact(
            name="system_design",
            content=json.dumps(data, indent=2),
            architecture_diagram=data.get("architecture_diagram", ""),
            api_schema=data.get("api_schema", ""),
            data_models=data.get("data_models", ""),
            component_breakdown=data.get("component_breakdown", []),
            metadata={"conversation_turns": len(result.messages)},
        )
