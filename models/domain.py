from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    PENDING    = "PENDING"
    PLANNING   = "PLANNING"
    DESIGNING  = "DESIGNING"
    CODING     = "CODING"
    TESTING    = "TESTING"
    REVIEWING  = "REVIEWING"
    DONE       = "DONE"
    FAILED     = "FAILED"


class AgentRole(str, Enum):
    PLANNER  = "planner"
    DESIGNER = "designer"
    CODER    = "coder"
    TESTER   = "tester"
    REVIEWER = "reviewer"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE    = "DONE"
    FAILED  = "FAILED"
    SKIPPED = "SKIPPED"


# ── Artifacts ─────────────────────────────────────────────────────────────────

class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_role: AgentRole
    name: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PlanArtifact(Artifact):
    agent_role: AgentRole = AgentRole.PLANNER
    requirements: List[str] = Field(default_factory=list)
    milestones: List[str] = Field(default_factory=list)
    tech_stack: List[str] = Field(default_factory=list)


class DesignArtifact(Artifact):
    agent_role: AgentRole = AgentRole.DESIGNER
    architecture_diagram: str = ""
    api_schema: str = ""
    data_models: str = ""
    component_breakdown: List[str] = Field(default_factory=list)


class CodeArtifact(Artifact):
    agent_role: AgentRole = AgentRole.CODER
    files: Dict[str, str] = Field(default_factory=dict)
    language: str = "python"
    entry_point: str = ""


class TestArtifact(Artifact):
    agent_role: AgentRole = AgentRole.TESTER
    test_files: Dict[str, str] = Field(default_factory=dict)
    coverage_estimate: float = 0.0
    test_cases: List[str] = Field(default_factory=list)


class ReviewArtifact(Artifact):
    agent_role: AgentRole = AgentRole.REVIEWER
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    approved: bool = False
    security_notes: List[str] = Field(default_factory=list)


# ── Task node ─────────────────────────────────────────────────────────────────

class TaskNode(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_role: AgentRole
    status: TaskStatus = TaskStatus.PENDING
    depends_on: List[str] = Field(default_factory=list)
    retries: int = 0
    max_retries: int = 2
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    artifact: Optional[Artifact] = None
    error: Optional[str] = None

    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


# ── Project context ───────────────────────────────────────────────────────────

class ProjectContext(BaseModel):
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    status: ProjectStatus = ProjectStatus.PENDING
    tasks: Dict[str, TaskNode] = Field(default_factory=dict)
    artifacts: Dict[AgentRole, Artifact] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    llm_config: Dict[str, Any] = Field(default_factory=dict)

    def log_event(self, agent: str, message: str, level: str = "info") -> None:
        self.events.append({
            "ts": datetime.utcnow().isoformat(),
            "agent": agent,
            "level": level,
            "message": message,
        })
        self.updated_at = datetime.utcnow()


# ── API shapes ────────────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    title: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    llm_config: Optional[Dict[str, Any]] = None


class ProjectSummary(BaseModel):
    project_id: str
    title: str
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    task_count: int
    completed_tasks: int


class ProjectStatusResponse(BaseModel):
    project_id: str
    title: str
    status: ProjectStatus
    tasks: Dict[str, TaskNode]
    artifacts: Dict[str, Any]
    events: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
