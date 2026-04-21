"""
FastAPI Async API
─────────────────
POST   /projects                – create project & start pipeline (202)
GET    /projects                – list all projects
GET    /projects/{id}           – full status + tasks + last 100 events
GET    /projects/{id}/plan      – execution plan (dry-run, no LLM calls)
GET    /projects/{id}/artifacts – all produced artifacts
DELETE /projects/{id}           – remove from store
GET    /projects/{id}/stream    – SSE real-time event stream
GET    /health                  – health check
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from multiagent_sds.core.event_bus import bus
from multiagent_sds.core.state_store import store
from multiagent_sds.models.domain import (
    AgentRole,
    CreateProjectRequest,
    ProjectContext,
    ProjectStatus,
    ProjectStatusResponse,
    ProjectSummary,
)
from multiagent_sds.orchestrator.engine import OrchestratorEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Default LLM config ─────────────────────────────────────────────────────────
# Fields are passed flat to build_model_client() which constructs
# an OpenAIChatCompletionClient from autogen_ext.models.openai
DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "model": os.getenv("LLM_MODEL", "gpt-4o"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
}


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Multiagent SDS booting up (autogen-agentchat 0.4.x)…")
    yield
    logger.info("Multiagent SDS shutting down.")


app = FastAPI(
    title="Multiagent Software Development System",
    description=(
        "An async AI-powered software factory using AutoGen 0.4.x. "
        "Submit a description + requirements → get back a full implementation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Background pipeline runner ─────────────────────────────────────────────────

async def _run_pipeline(project_id: str) -> None:
    ctx = await store.get(project_id)
    if not ctx:
        logger.error("Pipeline aborted — project %s not found in store", project_id)
        return
    engine = OrchestratorEngine(ctx)
    await engine.run()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/projects", status_code=202)
async def create_project(
    request: CreateProjectRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Create a new project and immediately launch the pipeline as a background task.

    Returns the `project_id`, a dry-run execution plan, and stream/status URLs.
    Monitor progress via `GET /projects/{id}` or stream via `GET /projects/{id}/stream`.
    """
    llm_config = request.llm_config or DEFAULT_LLM_CONFIG

    ctx = ProjectContext(
        title=request.title,
        description=request.description,
        requirements=request.requirements,
        llm_config=llm_config,
    )

    # Compute static execution plan before any LLM calls
    engine = OrchestratorEngine(ctx)
    plan = engine.get_execution_plan()

    await store.save(ctx)
    background_tasks.add_task(_run_pipeline, ctx.project_id)
    logger.info("Project %s created: '%s'", ctx.project_id, ctx.title)

    return {
        "project_id": ctx.project_id,
        "status": ctx.status.value,
        "message": "Pipeline started in background",
        "execution_plan": plan,
        "stream_url": f"/projects/{ctx.project_id}/stream",
        "status_url": f"/projects/{ctx.project_id}",
    }


@app.get("/projects", response_model=List[ProjectSummary])
async def list_projects() -> List[ProjectSummary]:
    """List all projects with summary statistics."""
    all_projects = await store.list_all()
    return [
        ProjectSummary(
            project_id=p.project_id,
            title=p.title,
            status=p.status,
            created_at=p.created_at,
            updated_at=p.updated_at,
            task_count=len(p.tasks),
            completed_tasks=sum(
                1 for t in p.tasks.values() if t.status.value == "DONE"
            ),
        )
        for p in all_projects
    ]


@app.get("/projects/{project_id}", response_model=ProjectStatusResponse)
async def get_project(project_id: str) -> ProjectStatusResponse:
    """Full project status including all tasks, artifacts, and recent events."""
    ctx = await store.get(project_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectStatusResponse(
        project_id=ctx.project_id,
        title=ctx.title,
        status=ctx.status,
        tasks=ctx.tasks,
        artifacts={role.value: art.model_dump() for role, art in ctx.artifacts.items()},
        events=ctx.events[-100:],
        created_at=ctx.created_at,
        updated_at=ctx.updated_at,
    )


@app.get("/projects/{project_id}/plan")
async def get_execution_plan(project_id: str) -> Dict[str, Any]:
    """
    Return the static execution plan showing dependency order and parallelism.
    Works before the pipeline starts — no LLM calls made.
    """
    ctx = await store.get(project_id)
    if ctx is None:
        # Return a generic plan using a dummy context
        dummy_ctx = ProjectContext(
            title="preview",
            description="",
            llm_config=DEFAULT_LLM_CONFIG,
        )
        engine = OrchestratorEngine(dummy_ctx)
    else:
        engine = OrchestratorEngine(ctx)
    return {"execution_plan": engine.get_execution_plan()}


@app.get("/projects/{project_id}/artifacts")
async def get_artifacts(project_id: str) -> Dict[str, Any]:
    """Return all produced artifacts in full detail."""
    ctx = await store.get(project_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        role.value: artifact.model_dump()
        for role, artifact in ctx.artifacts.items()
    }


@app.delete("/projects/{project_id}", status_code=204)
async def delete_project(project_id: str) -> None:
    """Remove a project from the store and clear its event subscriptions."""
    deleted = await store.delete(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    await bus.unsubscribe_all(project_id)


@app.get("/projects/{project_id}/stream")
async def stream_events(project_id: str, request: Request) -> StreamingResponse:
    """
    Server-Sent Events stream for real-time pipeline progress.

    Connect with:
        curl -N http://localhost:8000/projects/{id}/stream
    Or use EventSource in the browser:
        const es = new EventSource('/projects/{id}/stream');
        es.onmessage = e => console.log(JSON.parse(e.data));

    Events emitted:
        connected        – immediately on connect
        graph_ready      – dependency graph built
        task_started     – agent begins work
        task_done        – agent completes, artifact produced
        task_retry       – agent retrying after failure
        task_failed      – agent exhausted retries
        code_stream_chunk – live code generation tokens (coder agent only)
        log              – general orchestrator log messages
        pipeline_finished – pipeline reached terminal state
    """
    ctx = await store.get(project_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Project not found")

    async def event_generator():
        # Immediately yield current state so client doesn't wait
        yield (
            f"data: {{\"type\":\"connected\","
            f"\"project_id\":\"{project_id}\","
            f"\"status\":\"{ctx.status.value}\"}}\n\n"
        )
        async for chunk in bus.subscribe(project_id):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.0.0",
        "autogen": "agentchat-0.4.x",
        "projects_in_store": len(store),
    }
