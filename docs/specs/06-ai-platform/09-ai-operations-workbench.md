# 09 — AI Operations Workbench

**Status:** Implemented

**Owner:** Operations / AI Platform

**Primary UI:** **Workflows**

**API:** `POST /jobs/workflow-plan`

## Problem

Loading files, clustering SKUs, refreshing models, publishing forecasts,
archiving snapshots, and recomputing inventory are one dependency chain, but
they previously appeared as separate Integration and Jobs experiences. An
operator had to know which screen, job type, and order to use. That made a
powerful system difficult to operate and made stale or duplicate work too easy.

## Product outcome

The **Workflow Command Center** provides one guarded operating surface with
three progressively disclosed views:

1. **Plan & Run** — scan inputs and database readiness, resolve meaningful
   questions, and run the next safe registered workflow.
2. **Workflow Library** — monitor active/history jobs, schedules, job groups,
   and custom pipelines.
3. **Manual Load** — advanced full-pipeline and single-domain controls.

The default path is simple: **Analyze workflows → answer if required → Run
next → Analyze again**. Re-analysis after every stage prevents a stale plan
from blindly launching downstream work.

## Safety model

This is a system-first, AI-verified planner. The LLM is not the scheduler and
cannot invent executable work.

- Deterministic readiness rules inspect source hashes, active jobs, promoted
  cluster assignments, stale tuning profiles, sales/champion timestamps,
  production-release coverage, snapshot roster/archive state, and inventory
  refresh timestamps.
- The resulting order is a safety property. AI may explain risks and ask
  questions, but it cannot omit, reorder, rename, or parameterize executable
  stages.
- Every executable recommendation must match a named preset in
  `config/forecasting/pipelines.yaml`; step details are loaded server-side.
- Only the first currently safe recommendation is executable. Blockers are
  visible, and the operator must rescan after it completes.
- Existing API-key protection, JobManager concurrency, stop-on-failure
  pipelines, PID tracking, logs, and retry controls remain authoritative.
- If AI is unavailable, deterministic recommendations remain available with a
  visible `System verified` badge and a safe diagnostic flag such as
  `ai_usage_limit`, `ai_auth_required`, `ai_timeout`, or
  `ai_verification_unavailable`.

## Readiness rules and named workflows

| Evidence | Recommendation | Registered sequence |
|---|---|---|
| One or more input domains changed | `data-refresh` | ETL refresh → SKU features → materialized views |
| No promoted SKU cluster assignments, or SKU features newer than the promoted cluster | `clustering-refresh` | SKU features → cluster pipeline with promotion |
| Stale tuning profile or sales newer than champion | `model-refresh` | stale tuning → five backtests → atomic governed champion experiment/results promotion |
| No complete champion release for the planning month | `forecast-publish` | production training → immutable candidate generation → snapshot contenders |
| Champion + three contender roster ready, archive absent | `forecast-snapshot-bundle` | freeze contenders → archive lags 0–5 → reconciled cleanup |
| Inventory outputs predate the current champion forecast | `inventory-refresh` | safety stock → EOQ → replenishment → policies → health → exceptions |

Changed source data is always first. Missing clustering stops later model
recommendations until clusters exist. Model freshness precedes publishing.
Archive and inventory work are recommended only when their upstream evidence
is ready.

## Clarification policy

Questions are deliberately rare. The planner asks only when the answer changes
safety or ordering, currently including active queued/running work. Answers are
returned to the same endpoint as stable `question_id` / `answer` pairs. A model
response that claims it needs clarification but supplies no question is not
allowed to dead-end the UI; the plan continues with a risk flag.

## Runtime and cost model

The workbench reuses `config/ai/integration_scan_config.yaml`:

- **Laptop development:** `runtime.provider: codex` runs `codex exec` with the
  saved Codex/ChatGPT sign-in and configured `models.codex` model (currently
  `gpt-5.5`). It requires no separate OpenAI API key or API credit balance.
- **Production:** set `INTEGRATION_SCAN_AI_RUNTIME=openai` and provide
  `OPENAI_API_KEY`. The configured `models.openai` model is called through the
  metered OpenAI API.
- The response always reports effective provider, model, confidence, and
  whether AI verification completed.

Both runtime paths receive the same bounded evidence and JSON contract. The AI
call is read-only and never receives an arbitrary SQL or shell execution tool.

## API contract

`POST /jobs/workflow-plan` is write-protected because it can expose an
execution-ready plan and invokes a paid production model when configured.

Request:

```json
{
  "answers": [
    {"question_id": "active_job_handling", "answer": "Wait for active jobs"}
  ]
}
```

Response fields include `provider`, `model`, `ai_verified`, `status`,
`confidence`, `questions`, ordered `recommendations`, grounded `steps`,
`blockers`, and the readiness `evidence`. Execution continues through the
existing key-protected `POST /jobs/pipelines/named/{name}` endpoint.

## UX and compatibility

- The sidebar, command palette, keyboard shortcut `6`, active-job status pill,
  and forecast-release remediation links all open **Workflows**.
- The former standalone Jobs tab is embedded as **Workflow Library**.
- Legacy `?tab=jobs` links redirect to `?tab=integration` so bookmarks remain
  safe without presenting two competing destinations.
- Advanced controls are available but visually separated from the recommended
  guided path.

## Acceptance criteria

- A non-expert can determine and start the next safe operation without knowing
  internal job names.
- AI cannot create an executable job or violate deterministic dependencies.
- Clustering, forecasting, snapshot archive, and inventory readiness are
  checked alongside input integration.
- Active work causes clarification instead of accidental duplicate execution.
- AI failure never removes deterministic safety checks or blocks manual tools.
- Every recommendation resolves to a tested, registered JobManager pipeline.
- Provider/model/runtime status and prerequisites are visible before execution.

## Key implementation paths

| Concern | Path |
|---|---|
| Planner and readiness rules | `common/ai/workflow_planner/planner.py` |
| Endpoint | `api/routers/core/jobs.py` |
| Named pipelines | `config/forecasting/pipelines.yaml` |
| Unified workspace | `frontend/src/tabs/OperationsTab.tsx` |
| Guided recommendation UI | `frontend/src/components/workflows/WorkflowScanPanel.tsx` |
| Query contract | `frontend/src/api/queries/jobs.ts` |
| Backend tests | `tests/unit/test_workflow_planner.py`, `tests/api/test_workflow_planner.py` |
| Frontend tests | `frontend/src/components/workflows/__tests__/WorkflowScanPanel.test.tsx`, `frontend/src/tabs/__tests__/OperationsTab.test.tsx` |
