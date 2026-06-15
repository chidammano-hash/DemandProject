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

## US17 — converge the two job backends (in progress, risk-laddered split)

Split into 17a–17e (see `README.md`). Shipping bottom-up, lowest risk first.

| Story | Status | Summary |
|---|---|---|
| US17a | ✅ | `common/services/job_shape.py` — pure row→`Job` shape + `completed↔success` status map (read-only, zero behavior change) |
| US17b | ✅ | `integration_job_unified` view (sql/188); `IntegrationRunner.list/get` read it — merges `integration_job` + `job_history` ETL jobs, `completed→success` in SQL |
| US17c | ✅ | `load_domain` JobManager job type (group `etl`); `POST /integration/jobs` submits it — gates (allowlist/slice/sandbox/cascade) enforced pre-submission; new loads land in `job_history`, legacy rows still readable |
| US17d | ✅ | Chains run as JobManager pipelines of `load_domain` steps (`chain_shape` adapter + `ChainJobRunner`); `/integration/chains[/{id}]` shape unchanged, legacy `integration_chain` rows still readable via fallback |
| US17e | ⏳ | UI convergence + legacy retirement |

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
