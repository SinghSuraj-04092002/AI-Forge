"""
Reviewer Agent
──────────────
Security + quality code review of the generated code and tests.
Uses a SelectorGroupChat with three reviewers whose expertise is selected
by the model: security_reviewer, quality_reviewer, final_approver.
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
    ProjectContext,
    ReviewArtifact,
    TaskNode,
)

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    role = AgentRole.REVIEWER

    def _system_prompt(self) -> str:
        return (
            "You are a senior code reviewer with security expertise. "
            "Produce a review JSON with keys: "
            "\"issues\" (list[str] – blocking issues), "
            "\"suggestions\" (list[str] – non-blocking improvements), "
            "\"security_notes\" (list[str]), "
            "\"approved\" (bool), "
            "\"review_summary\" (str). "
            "Reply with ONLY valid JSON."
        )

    async def execute(self, ctx: ProjectContext, task: TaskNode) -> Artifact:
        code_artifact = ctx.artifacts.get(AgentRole.CODER)
        test_artifact = ctx.artifacts.get(AgentRole.TESTER)
        code_content = code_artifact.content if code_artifact else "{}"
        test_content = test_artifact.content if test_artifact else "{}"

        model_client = build_model_client(self.llm_config)

        security_reviewer = AssistantAgent(
            name="security_reviewer",
            model_client=model_client,
            description="Identifies security vulnerabilities, injection risks, auth flaws.",
            system_message=(
                "You are a security engineer. Review for: SQL injection, "
                "auth bypass, secrets in code, input validation, dependency risks. "
                "Report as JSON fragment with 'security_notes' and 'issues'."
            ),
        )
        quality_reviewer = AssistantAgent(
            name="quality_reviewer",
            model_client=model_client,
            description="Reviews code quality, async correctness, error handling, types.",
            system_message=(
                "You are a senior Python engineer. Review for: async/await correctness, "
                "error handling, type annotations, code duplication, test coverage gaps. "
                "Report as JSON fragment with 'issues' and 'suggestions'."
            ),
        )
        final_approver = AssistantAgent(
            name="final_approver",
            model_client=model_client,
            description="Consolidates feedback and issues final approval decision.",
            system_message=(
                "You consolidate security and quality reviews into a final review JSON: "
                "\"issues\", \"suggestions\", \"security_notes\", \"approved\" (bool), "
                "\"review_summary\" (str). "
                "Set approved=true only if there are no blocking issues. "
                "Reply: FINAL_REVIEW\n<the_json>"
            ),
        )

        termination = TextMentionTermination("FINAL_REVIEW") | MaxMessageTermination(8)
        team = SelectorGroupChat(
            participants=[security_reviewer, quality_reviewer, final_approver],
            model_client=model_client,
            termination_condition=termination,
        )

        task_prompt = (
            f"Project: {ctx.title}\n\n"
            f"Source code:\n{code_content}\n\n"
            f"Test suite:\n{test_content}\n\n"
            "Conduct a full review. Security reviewer goes first, "
            "then quality reviewer, then final_approver consolidates."
        )

        result = await team.run(task=task_prompt, cancellation_token=CancellationToken())
        raw = PlannerAgent._extract_json_content(result.messages)

        try:
            clean = raw.replace("FINAL_REVIEW", "").strip()
            data: Dict[str, Any] = json.loads(clean)
        except json.JSONDecodeError:
            data = {
                "issues": [],
                "suggestions": ["Review output unparseable — manual review needed"],
                "security_notes": [],
                "approved": False,
                "review_summary": raw[:400],
            }

        return ReviewArtifact(
            name="code_review",
            content=json.dumps(data, indent=2),
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            security_notes=data.get("security_notes", []),
            approved=bool(data.get("approved", False)),
            metadata={"conversation_turns": len(result.messages)},
        )
