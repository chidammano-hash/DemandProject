# Data Ingestion Streamlining — Delivery Status

Branch: `feat/unified-data-ingestion`. All work TDD'd and committed story-by-story.

## Shipped (20 of 21 stories)

| Story | Status | Summary |
|---|---|---|
| US1 | ✅ | Characterization tests for the untested ETL loaders (safety net) |
| US2 | ✅ | `make perf-ingestion` baseline harness + RUNBOOK section |
| US3 | ✅ | `common/core/etl_helpers.py` — shared index/constraint mgmt (killed 3 copies) |
| US4 | ✅ | Shared staging-name + monthly-partition manager; removed `_SLICE_DELETE_TABLES` |
| US5 | ✅ | Shared DFU/FK filters + `record_load_batch`; customer_demand records lineage |
| US6 | ✅ | (reframed) load.py confirmed canonical mode dispatcher; converged on shared partition helpers |
| US7 | ✅ | (reframed) verified IntegrationRunner targets the unified engine; onetime→bulk delegation |
| US8 | ✅ | Normalize-time DFU filtering for sales/forecast (skips post-COPY scan on refresh) |
| US9 | ✅ | (reframed) removed dead `_load_forecast_archive`; pinned the real loader's batch streaming |
| US10 | ✅ | Size-based index drop/recreate (small dims keep indexes) |
| US11 | ✅ | `executemany` batching for load_open_pos (killed row-by-row N+1) |
| US12 | ✅ | Loader magic numbers → `etl_config.yaml` `performance:` |
| US13 | ✅ | Transaction isolation for the open-PO multi-step load (atomic + failed-batch audit) |
| US14 | ✅ | Logging, specific exceptions, path-hack cleanup; rule gate green for scripts/etl |
| US15 | ✅ | customer_demand wired into `pipeline-refresh` (change detection + CA MVs) |
| US16 | ✅ | `etl_pipeline` JobManager job type (full/refresh, schedulable) |
| US18 | ✅ | `POST /integration/pipeline` (202 + job_id; status via /jobs) |
| US19 | ✅ | UI "Run Pipeline" control (full/refresh + parallel, live status) |
| US20 | ✅ | Unified load lineage with domain/status filters + sanitized errors |
| US21 | ✅ | Docs + final verification (this) |

## Deferred

- **US17 — converge the two job backends** (`integration_jobs` ↔ `job_history`).
  Explicitly HIGH-risk; the plan said ship it alone. Today the IntegrationRunner
  (single/chain loads) and the new `etl_pipeline` JobManager job coexist: the new
  pipeline endpoint uses JobManager (`/jobs`), the per-domain loads use
  IntegrationRunner (`/integration/jobs`). Both work; merging the stores into one
  is a dedicated follow-up. Recommended approach unchanged: a backward-compatible
  `integration_jobs` view over the unified store so the UI shape stays stable.

## Known issues (pre-existing, not from this work)

- `frontend/src/components/__tests__/DemandReferencePanel.test.tsx > shows KPI cards`
  fails ("1,200" KPI text) — unrelated to ingestion (no integration refs); present
  before this branch.
- DataQualityTab.tsx is 676 lines (was 719; trimmed here) — still above the 600
  guideline; further extraction is a separate cleanup.

## Verification

- Backend: full pytest suite green (`~/.local/bin/uv run pytest tests/ -q`).
- Frontend: vitest green except the one pre-existing failure above.
- Rule gate (`scripts/ai_checks/check_unenforced_rules.sh`): no new violations.
- `make audit-routers`: `/integration` proxy parity holds (no new prefix added).
