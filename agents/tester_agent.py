"""
Tester Agent
────────────
Writes pytest test suites for the generated codebase.
Uses a two-agent pair: test_writer drafts, coverage_analyst checks completeness.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken

from multiagent_sds.agents.base_agent import BaseAgent, build_model_client
from multiagent_sds.agents.planner_agent import PlannerAgent
from multiagent_sds.models.domain import (
    AgentRole,
    Artifact,
    ProjectContext,
    TaskNode,
    TestArtifact,
)

logger = logging.getLogger(__name__)


class TesterAgent(BaseAgent):
    role = AgentRole.TESTER

    def _system_prompt(self) -> str:
        return (
            "You are a QA engineer specialising in Python testing. "
            "Write comprehensive pytest test suites for the given code. "
            "Return JSON with keys: "
            "\"test_files\" (dict[str,str] filename → test code), "
            "\"test_cases\" (list[str] – one-line description per test), "
            "\"coverage_estimate\" (float 0.0–1.0), "
            "\"testing_notes\" (str). "
            "Tests must use pytest and pytest-asyncio for async code. "
            "Reply with ONLY valid JSON."
        )

    async def execute(self, ctx: ProjectContext, task: TaskNode) -> Artifact:
        code_artifact = ctx.artifacts.get(AgentRole.CODER)
        code_content = code_artifact.content if code_artifact else "{}"

        model_client = build_model_client(self.llm_config)

        test_writer = AssistantAgent(
            name="test_writer",
            model_client=model_client,
            system_message=self._system_prompt(),
        )
        coverage_analyst = AssistantAgent(
            name="coverage_analyst",
            model_client=model_client,
            system_message=(
                "You review test suites for completeness. Check: "
                "happy paths, edge cases, error handling, async correctness. "
                "If coverage looks ≥80%, reply: COVERAGE_OK\n<the_json>. "
                "Otherwise ask for specific missing tests."
            ),
        )

        termination = TextMentionTermination("COVERAGE_OK") | MaxMessageTermination(6)
        team = RoundRobinGroupChat(
            participants=[test_writer, coverage_analyst],
            termination_condition=termination,
        )

        task_prompt = (
            f"Project: {ctx.title}\n\n"
            f"Source code to test:\n{code_content}\n\n"
            "Generate a comprehensive test suite."
        )

        result = await team.run(task=task_prompt, cancellation_token=CancellationToken())
        raw = PlannerAgent._extract_json_content(result.messages)

        try:
            clean = raw.replace("COVERAGE_OK", "").strip()
            data: Dict[str, Any] = json.loads(clean)
        except json.JSONDecodeError:
            data = {
                "test_files": {"test_main.py": raw},
                "test_cases": ["smoke test"],
                "coverage_estimate": 0.5,
                "testing_notes": "Auto-generated",
            }

        return TestArtifact(
            name="test_suite",
            content=json.dumps(data, indent=2),
            test_files=data.get("test_files", {}),
            test_cases=data.get("test_cases", []),
            coverage_estimate=float(data.get("coverage_estimate", 0.5)),
            metadata={"conversation_turns": len(result.messages)},
        )
