# MASTER PLAN — Data Ingestion Pipeline Streamlining (All Phases)

> Single-file reference for the entire epic. Per-story detail lives in `UserStory1.md … UserStory21.md`; the index is in `README.md`.

## Goal

Unify the data ingestion pipeline along **five axes simultaneously**:
1. **Unify full + incremental** — one engine, one orchestrator.
2. **Code consolidation** — kill duplicated index/partition/filter logic.
3. **Performance** — remove scans, dual-loads, needless index churn, row-by-row inserts.
4. **Reliability/observability** — transactions, full audit coverage, logging, tests.
5. **UI integration** — run + monitor full load and incremental refresh from the browser.

Scope: **all 13 scripts in `scripts/etl/`** + `common/core`, `common/services`, API ingestion routers, UI tabs.

---

## The core problem (why this epic exists)

Two divergent load engines and three job/exec systems:

| | CLI / Make path | UI path |
|---|---|---|
| **Load engine** | `load_dataset_postgres.py` | `load.py` (separate impl) |
| **Orchestrator** | `run_pipeline.py` (`--mode full`/`refresh`) | `IntegrationRunner` (subprocess) |
| **Job backend** | none (shell) | `integration_jobs` table |
| **Other jobs** | — | `JobManager`/APScheduler → `job_history` |

Target end-state: **one load engine** (mode-parameterized), **one orchestrator** (importable, job-wrapped), **one job backend** (JobManager / pg-queue for long runs), exposed through the existing UI tabs.

---

## Current-state map (anchors for the work)

### Full-load path
- Make: `normalize-all` (L386) → `load-all` (L453–465) → `refresh-mvs-tiered` (L1656–1672); `fresh-load` (L1683) chains them.
- Orchestrator: `scripts/etl/run_pipeline.py --mode full --parallel` (2-wave: dims then facts).
- Config: `config/etl/etl_config.yaml` — `domain_order` (L12–23), `parallel` (L31–35), `mv_refresh` (L40–62), `always_refresh` (L66–68).
- Domains: `common/core/domain_specs.py` (`DomainSpec` L6–39; registrations L662–675).
- 11 domains: item, location, customer, time, sku, sales, forecast, inventory, customer_demand, sourcing, purchase_order (+ computed customer_features).

### Incremental path
- Make: `pipeline-refresh`, `pipeline-inventory-refresh` (L1451–1458); customer-demand `--month`/`--replace` (L441–450).
- Change detection: SHA-256 vs `audit_load_batch.source_hash` — `run_pipeline.detect_changes` (L60–84), `detect_inventory_changes` (L87–106), `build_incremental_delete` (L127–138); hashing in `common/engines/medallion.py` (L17–33).
- 4 phases: detect → normalize changed → load changed (incremental DELETE by month range) → refresh affected MVs.
- Forecast archive: `_load_forecast_archive` (`load_dataset_postgres.py` L299–368), archive-before-staging ordering; lag resolved from `dim_sku`.
- Promotion: `POST /backtest-management/{model_id}/promote` (candidate→production).

### UI / API
- Tabs: `IntegrationTab.tsx` (submit + monitor single/chain loads), `DataQualityTab.tsx` (lineage via `audit_load_batch`), `JobsTab.tsx` (APScheduler jobs). (`ControlTowerTab.tsx`, `AIPlannerTab.tsx` deleted.)
- Routers: `platform/integration.py` (`/integration/jobs`, `/domains`, `/health`), `platform/integration_chain.py` (`/integration/scan`,`/chains`), `core/jobs.py` (`/jobs*`), `platform/data_quality.py` (`/data-quality/batches`).
- Execution: `IntegrationRunner` (subprocess → `load.py`) and `JobManager` (APScheduler → `job_history`); `run_pipeline.py` is CLI-only.

### Pain points feeding the work
- HIGH: row-by-row inserts (`load_open_pos.py` L107/157/217); bare `except Exception` (7 sites); `print()` (28+ in `load_backtest_forecasts.py`); 3 copies of index/constraint code; path hack (`normalize_inventory_csv.py` L28).
- MEDIUM: divergent partition strategies (3); inconsistent staging names (5+); test gaps (6 scripts); transaction isolation.
- LOW: missing perf docs; magic numbers (`BATCH_SIZE`, `_PG_WORK_MEM`, workers); hardcoded partition-field map (`load.py` L67–72).

---

## Phases → stories → outcomes

### Phase 0 — Baseline & Safety Net  *(US1, US2)*
Pin current behavior and timing before touching anything.
- **US1** Characterization tests for the 6 untested loaders (golden outputs; must pass against current code).
- **US2** Baseline timing benchmarks recorded in `docs/RUNBOOK.md` (before-state for "streamline").
- **Exit:** safety net in place; `make test-all` green with zero behavior change.

### Phase 1 — Shared ETL Core  *(US3, US4, US5)* → `common/core/etl_helpers.py`
One implementation of every duplicated mechanic.
- **US3** Index/constraint drop→load→recreate (replaces 3 copies).
- **US4** Staging naming + partition manager (one convention; partition metadata in `DomainSpec`/config; removes `_SLICE_DELETE_TABLES`).
- **US5** DFU/FK filter + `audit_load_batch` writer (one filter; every domain records lineage incl. customer_demand).
- **Exit:** duplication removed; US1 parity tests still green.

### Phase 2 — Single Load Engine  *(US6, US7)*  **(highest leverage)**
Collapse the two engines.
- **US6** Mode-parameterized engine (`full`|`delta`|`file`) in `load_dataset_postgres.py`; full = delta w/ empty watermark; folds in `load.py` logic. (load.py still present → reversible.)
- **US7** Repoint `IntegrationRunner` to unified engine; **delete `load.py`**; preserve cascade/file-sandbox safety. **(HIGH risk — ship alone.)**
- **Exit:** one load engine; CLI + UI share it.

### Phase 3 — Performance  *(US8–US12)*
- **US8** Push DFU filtering to normalize time (kill post-COPY DELETE scan).
- **US9** Conditional/streamed forecast archive load (delta writes scoped slice only).
- **US10** Size-based index drop/recreate (small dims keep indexes).
- **US11** COPY/executemany for `load_open_pos.py` (kill N+1).
- **US12** Magic numbers → `etl_config.yaml performance:`.
- **Exit:** measurable speedups recorded vs US2 baseline; parity preserved.

### Phase 4 — Reliability & Observability  *(US13–US15)*
- **US13** Transaction isolation for multi-step loads (atomic + rollback + audit-failed).
- **US14** Logging + specific exceptions + path-hack cleanup (rule gate clean for `scripts/etl/`).
- **US15** `customer_demand` fully in change detection (skips when unchanged; MVs gated).
- **Exit:** debuggable, atomic, rule-compliant pipeline.

### Phase 5 — Unified Orchestration  *(US16–US18)*
- **US16** Register `etl_pipeline` job type (full+refresh) on JobManager; route long full-loads to pg-queue per CLAUDE.md. (orchestrator becomes importable, not CLI-only.)
- **US17** Converge `integration_jobs` + `job_history` onto one backend (JobManager canonical; legacy readable via view). **(HIGH risk — ship alone.)**
- **US18** `POST /integration/pipeline` (full|refresh, domains?, parallel?) + Vite proxy + barrel; status via unified job endpoints.
- **Exit:** one job backend; whole pipeline operable over HTTP.

### Phase 6 — UI Integration  *(US19, US20)*
- **US19** IntegrationTab "Run Pipeline" control (full/refresh, per-domain/all, live status + logs, destructive confirm). Tab < 600 lines.
- **US20** Unified load history & lineage (incl. customer_demand + run mode; filter by domain/status).
- **Exit:** full load + incremental refresh fully runnable and monitorable from the UI.

### Phase 7 — Verification  *(US21)*
- **US21** Docs in same commits; full gate green (`make test-all`, `audit-routers`, lint, type-check, rule gate, targeted E2E); cross-cutting self-review; no `load.py` references remain.
- **Exit:** epic Definition of Done satisfied.

---

## Dependency graph

```
US1 ─┐
US2  │
     ├─> US3 ─> US4 ─┐
     │        └> US5 ─┴─> US6 ─> US7
US1 ─┘                      │
                 US8 (US5,US6) ─┐
                 US9 (US6)      │
                 US10 (US3,US6) ├─ Phase 3 (parallel after US6)
                 US11 (US1) ─> US13
                 US12 ──────────┘
                 US14 (—)        ─ Phase 4
                 US15 (US5)
                 US16 (US6) ─> US17
                          └─> US18 ─> US19
                                  └─> US20
   all ─> US21
```
- Strictly sequential foundation: **US1/US2 → US3 → US4/US5 → US6 → US7**.
- After US6, Phase 3 stories parallelize.
- US17 and US7 are the two HIGH-risk cutovers — ship each alone, fully tested.

---

## Cross-cutting rules (apply to every story)

- **TDD**: tests first (red) → minimal impl (green) → refactor + CLAUDE.md self-review.
- **Backend hard rules**: `%s` placeholders; `psycopg.sql.Identifier` for identifiers; no bare `except Exception`; no `print()` in scripts; `get_planning_date()`; `common.core.paths`; config in YAML; `get_conn()` for inv_planning routers; write endpoints `Depends(require_api_key)`; `domains.py` mounted last; 5xx details never interpolate exception text.
- **No backward-compat shims** — rewrite importers in the same change.
- **Frontend**: HTTP via `src/api/queries/*` (`fetchJson`, no raw fetch); no `: any` in queries; tab files < 600 lines; new prefix → Vite config + barrel together.
- **Docs in same commit**; tests mandatory per the integration checklist.

## Test commands (from MEMORY.md)
- Backend: `~/.local/bin/uv run pytest tests/ -q` (project root).
- Frontend (from `frontend/`): `PATH="/opt/homebrew/bin:$PATH" /opt/homebrew/bin/node node_modules/.bin/vitest run --reporter=dot`.
- `make` may not find `uv` on PATH — run uv/npx directly.

## Epic Definition of Done
- One load engine, one orchestrator, one job backend.
- Full load + incremental refresh triggerable & monitorable from the UI for any/all domains.
- `audit_load_batch` covers every domain incl. `customer_demand`.
- Before/after benchmarks in `docs/RUNBOOK.md`.
- All `scripts/etl/` rule violations resolved; `make test-all`, `make audit-routers`, lint, type-check, rule gate green.

## Suggested shipping cadence
Ship **Phases 0–2 first** as a standalone deliverable (safety net + the highest-value unification). Then reassess before the HIGH-risk job-backend convergence (Phase 5 / US17). Each story is its own PR behind `make test-all`.
