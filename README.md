# Multiagent Software Development System

An **async, AI-powered software factory** built with **AutoGen 0.4.x** (`autogen-agentchat` + `autogen-ext`) and **FastAPI**.

Submit a title, description, and requirements → get back a fully planned, designed, implemented, tested, and reviewed codebase — with real-time SSE streaming of every step.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FastAPI Async Gateway                             │
│  POST /projects · GET /projects/{id} · GET /projects/{id}/stream    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ BackgroundTask
┌───────────────────────────────▼──────────────────────────────────────┐
│                     ORCHESTRATOR ENGINE  ◄── The Brain ──           │
│                                                                      │
│  ① Build dependency DAG        ② Kahn's topological sort           │
│  ③ asyncio.wait(FIRST_COMPLETED) concurrent scheduling              │
│  ④ Lifecycle state machine     ⑤ Retry + cascade-skip BFS          │
│  ⑥ Structured event emission → EventBus → SSE                      │
└──────┬────────┬────────┬────────┬────────┬───────────────────────────┘
       │        │        │        │        │
 ┌─────▼──┐ ┌──▼───┐ ┌──▼──┐ ┌──▼───┐ ┌──▼──────┐
 │Planner │ │Desig-│ │Coder│ │Tester│ │Reviewer │
 │        │ │ner   │ │     │ │      │ │         │
 │Round   │ │Selec-│ │Round│ │Round │ │Selector │
 │Robin   │ │tor   │ │Robin│ │Robin │ │Group    │
 │Group   │ │Group │ │+    │ │Group │ │Chat     │
 │Chat    │ │Chat  │ │SSE  │ │Chat  │ │         │
 └────────┘ └──────┘ └─────┘ └──────┘ └─────────┘
          (autogen_agentchat AssistantAgent per role)

              Shared Infrastructure
   ┌─────────────────┬─────────────────┬──────────────┐
   │ autogen_ext     │   StateStore    │  EventBus    │
   │ OpenAI client   │  (async KV)     │  (SSE pub/sub│
   └─────────────────┴─────────────────┴──────────────┘
```

### Pipeline Lifecycle

```
PENDING → PLANNING → DESIGNING → CODING → TESTING → REVIEWING → DONE
                                                               ↘ FAILED
```

### Agent Team Patterns

| Agent    | Team type              | Termination                          | Why                                              |
|----------|------------------------|--------------------------------------|--------------------------------------------------|
| Planner  | RoundRobinGroupChat    | TextMentionTermination("APPROVED")   | Planner drafts, critic reviews                   |
| Designer | SelectorGroupChat      | TextMentionTermination("APPROVED")   | LLM-selected routing: architect→api_designer→critic |
| Coder    | RoundRobinGroupChat    | TextMentionTermination("LGTM")       | Lead coder + senior reviewer; tokens streamed to SSE |
| Tester   | RoundRobinGroupChat    | TextMentionTermination("COVERAGE_OK")| test_writer + coverage_analyst                   |
| Reviewer | SelectorGroupChat      | TextMentionTermination("FINAL_REVIEW")| security_reviewer → quality_reviewer → final_approver |

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and optionally LLM_MODEL
```

### 3. Run

```bash
python main.py
# or
uvicorn multiagent_sds.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Create a project

```bash
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Todo REST API",
    "description": "An async REST API for managing todos with user accounts",
    "requirements": [
      "CRUD endpoints for todos",
      "SQLite storage via SQLAlchemy async",
      "JWT authentication",
      "Pagination support"
    ]
  }'
```

Response (202):
```json
{
  "project_id": "abc123...",
  "status": "PENDING",
  "message": "Pipeline started in background",
  "execution_plan": [
    {"step": 1, "role": "planner",  "depends_on": [],                    "parallel_with": []},
    {"step": 2, "role": "designer", "depends_on": ["planner"],           "parallel_with": []},
    {"step": 3, "role": "coder",    "depends_on": ["designer"],          "parallel_with": []},
    {"step": 4, "role": "tester",   "depends_on": ["coder"],             "parallel_with": []},
    {"step": 5, "role": "reviewer", "depends_on": ["coder", "tester"],   "parallel_with": []}
  ],
  "stream_url": "/projects/abc123.../stream",
  "status_url": "/projects/abc123..."
}
```

### 5. Stream real-time progress (SSE)

```bash
curl -N http://localhost:8000/projects/abc123.../stream
```

Events you'll see:
```json
{"type":"connected",     "project_id":"abc123","status":"PENDING"}
{"type":"graph_ready",   "data":{"total_tasks":5,"nodes":[...]}}
{"type":"task_started",  "data":{"role":"planner","task_id":"..."}}
{"type":"task_done",     "data":{"role":"planner","artifact_id":"..."}}
{"type":"task_started",  "data":{"role":"designer","task_id":"..."}}
...
{"type":"code_stream_chunk","data":{"role":"coder","chunk":"..."}}
...
{"type":"pipeline_finished","data":{"status":"DONE"}}
```

### 6. Retrieve artifacts

```bash
# Full status
curl http://localhost:8000/projects/abc123...

# Just the artifacts (plan, design, code, tests, review)
curl http://localhost:8000/projects/abc123.../artifacts
```

---

## Project Structure

```
multiagent_sds/
├── main.py                      # Uvicorn entry point
├── requirements.txt             # autogen-agentchat>=0.4.9, autogen-ext[openai]>=0.4.9
├── .env.example
│
├── models/
│   └── domain.py                # Pydantic v2 models: ProjectContext, Artifacts, TaskNode
│
├── core/
│   ├── state_store.py           # Async lock-protected in-memory KV store
│   └── event_bus.py             # Async pub/sub → SSE chunks
│
├── agents/
│   ├── base_agent.py            # BaseAgent: OpenAIChatCompletionClient + AssistantAgent wrapper
│   ├── planner_agent.py         # RoundRobinGroupChat: planner + critic
│   ├── designer_agent.py        # SelectorGroupChat:  architect + api_designer + critic
│   ├── coder_agent.py           # RoundRobinGroupChat: lead_coder + reviewer, run_stream() → SSE
│   ├── tester_agent.py          # RoundRobinGroupChat: test_writer + coverage_analyst
│   └── reviewer_agent.py        # SelectorGroupChat:  security + quality + final_approver
│
├── orchestrator/
│   └── engine.py                # ⭐ OrchestratorEngine — DAG scheduler, state machine, cascade-skip
│
├── api/
│   └── app.py                   # FastAPI routes + BackgroundTasks pipeline runner
│
└── tests/
    └── test_system.py           # pytest suite: models, store, bus, orchestrator, API (35+ tests)
```

---

## Key Design Decisions

### Orchestrator Engine (the heart)

The `OrchestratorEngine` owns the full lifecycle:

- **Dependency DAG** — `TASK_DEPENDENCIES` maps each `AgentRole` to its prerequisites. `_build_task_graph()` converts these to `TaskNode.depends_on` lists keyed by task ID.
- **Concurrent scheduling** — `asyncio.wait(FIRST_COMPLETED)` loop. All unblocked tasks are launched as `asyncio.Task`s simultaneously. Tester and Reviewer share `[CODER]` as a dependency — if you added an additional agent at the same level, it would run in parallel automatically.
- **Cascade skip** — `_cascade_skip()` does a BFS over the reverse adjacency map. When a task fails, every downstream task is immediately marked `SKIPPED`.
- **State machine** — `ProjectStatus` transitions are owned by the orchestrator, not agents. Every agent completing its task advances the status.

### autogen-agentchat 0.4.x patterns

| Old pyautogen | New autogen-agentchat 0.4.x |
|---|---|
| `ConversableAgent(llm_config={...})` | `AssistantAgent(name=..., model_client=OpenAIChatCompletionClient(...))` |
| `UserProxyAgent.initiate_chat()` | `team.run(task=..., cancellation_token=CancellationToken())` |
| Manual termination logic | `TextMentionTermination("KEYWORD") \| MaxMessageTermination(N)` |
| `GroupChatManager` | `RoundRobinGroupChat` / `SelectorGroupChat` |
| Blocking call in thread pool | Native `async/await` throughout |
| No streaming | `team.run_stream()` yields messages + `TaskResult` |

### SSE Streaming

The Coder agent uses `team.run_stream()` and forwards content chunks to the `EventBus` as `code_stream_chunk` events. The `/projects/{id}/stream` endpoint subscribes to these and streams them via SSE. No polling needed.

---

## Running Tests

```bash
pytest tests/ -v
```

All 35+ tests mock the actual LLM calls — no API key needed to run the test suite.

---

## Production Swap Guide

| Component | Dev (current) | Production |
|-----------|--------------|------------|
| State store | `asyncio.Lock` + `dict` | Redis (`aioredis`) or PostgreSQL |
| Event bus | `asyncio.Queue` per subscriber | Redis Pub/Sub or Kafka |
| LLM config | env vars in `.env` | Secrets manager (AWS SSM / Vault) |
| Task runner | FastAPI `BackgroundTasks` | Celery / ARQ / Cloud Tasks |
| Auth | None | OAuth2 Bearer / API keys |
