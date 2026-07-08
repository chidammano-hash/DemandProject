# AGENTS.md — Supply Chain Command Center

## Project Overview

**Supply Chain Command Center** is a unified supply chain planning and execution platform. It ingests sales, forecast, and inventory data, stores it in PostgreSQL, and serves a React UI for interactive analytics. **All code lives at the project root** with domain-based organization inside each top-level directory.

## Tech Stack

| Layer | Technology |
|---|---|
| API | Python + FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| DB Driver | psycopg v3 |
| Frontend | React + Vite + TypeScript |
| Styling | Tailwind CSS + shadcn/ui |
| Charts | Recharts + ECharts |
| Database | PostgreSQL 16 |
| Cache | Redis 7 (shared across workers, single-flight stampede protection); falls back to per-process in-memory when Redis is unreachable. `make up` starts Redis alongside Postgres. |
| ML / Clustering | scikit-learn, pandas, scipy, matplotlib, seaborn |
| ML Tracking | MLflow |
| Job Scheduling | APScheduler 3.11 (BackgroundScheduler + ThreadPoolExecutor) |
| E2E Testing | Playwright |
| Python packaging | uv |
| Build | Make |
| Containers | Docker Compose |
| Deployment | Docker (API + Nginx frontend) |

---

## Top Critical Rules

Hard constraints. Violations cause bugs, test failures, or silent data corruption.

### Mechanically enforced (the pre-commit gate hard-blocks NEW violations)
These 6 are checked by `scripts/ai_checks/check_unenforced_rules.sh` (allowlist-pinned to
existing files) and hard-block a commit that adds a new one — so they need no prose vigilance,
just don't add them:
- `date.today()` only in `common/core/planning_date.py` → use `get_planning_date()`.
- No `Path(__file__).resolve().parents[N]` at module level → `from common.core.paths import …`
  (allowed only in an `if __name__ == "__main__"` bootstrap).
- No bare `except Exception` → catch specific + `logger.exception()`; escape hatch
  `# noqa: BLE001 — <reason>`.
- No `print()` in `scripts/{etl,ml,inventory,forecasting,ops,ai}` → `logging.getLogger(__name__)`.
- psycopg3 `%s` placeholders, never `$1`/`$2`; no f-string SQL inside `cur.execute()`.
- No `: any` / `<any>` / `as any` in `frontend/src/api/queries/` → generate types via `npm run gen:types`.

### Workflow (applies to every change)
- **Self-review + refactor at each step** before reporting done (also a global habit). In
  multi-step plans, each step ends with this self-pass — don't defer to the end.
- **Docs updated in the same commit as the code.** Architecture/API/schema/convention/infra
  changes update the relevant docs (`docs/ARCHITECTURE.md`, `docs/ENTERPRISE_ARCHITECTURE.md`,
  `docs/operations-manual/`, `docs/specs/<domain>/`, this file if rules changed). The
  Feature Integration Checklist (step 5, below) maps each kind of change to its doc.
- **Commit automatically after development + testing.** After implementing a requested change,
  self-reviewing, and running the relevant tests/gates successfully, stage only the files changed
  for that task and create a concise git commit without waiting for an extra prompt. Never include
  unrelated user changes; if unrelated edits overlap the same files and cannot be separated safely,
  stop and ask how to proceed.

### Backend / Python (architectural — not mechanically checkable, hold these yourself)
- **`get_conn()` not `Depends(_get_pool)`** in `inv_planning_*.py` routers (`Depends` → 422 in
  tests: FastAPI inspects the MagicMock signature).
- **`domains.py` mounted LAST** in `api/main.py` — its `{domain}` catch-all shadows other routes.
- **`APIRouter(prefix="/...")` + short decorator paths.** Never the full path in the decorator.
- **5xx never interpolate exception text** → `logger.exception(...)` then
  `HTTPException(500, detail="<verb-phrase>")`. No `str(exc)`/`f"...{exc}"` in a 500.
- **Every write endpoint** has `dependencies=[Depends(require_api_key)]`.
- **Identifiers via `psycopg.sql.Identifier`** — never f-string; values via `%s`.
- **Pydantic v2 only** (`model_config = ConfigDict(...)`, never `class Config:`).
- **No backward-compat shims** — rewrite all importers in the same change
  (`from common.core.db import get_db_params`).
- **Modules/routers > 800 LoC split** by sub-feature into a domain folder.
- **No `_row_to_dict` outside `common/core/sql_helpers.py`** → `row_to_dict_from_cursor` /
  `row_to_dict_from_cols`.
- **Read-only analytics opt into `get_async_read_only_conn()`** / `get_read_only_conn()`
  (read replica when `READ_REPLICA_URL` set; never for read-after-write). 7 CA endpoints use it.

### ML / Forecasting → see skill `forecasting-patterns` for the full pattern catalog
- **All tree `.fit()`/instantiation goes through `common/ml/model_registry.py`.** Direct
  `LGBMRegressor()`/`CatBoostRegressor()`/`XGBRegressor()` elsewhere is a defect.
- **All ML hyperparameters live in `forecast_pipeline_config.yaml`** — no `kwargs.get()` defaults.
- **`ml_cluster` is metadata, NOT a feature** (in `METADATA_COLS`, merged for partitioning only).
- **`FORECAST_QTY_COL`** constant, never the literal `"basefcst_pref"`.

### Testing
- **All API tests use `make_pool` / `make_async_pool` from `tests/api/conftest.py`.** No
  hand-rolled `MagicMock` on `psycopg.connect`. `httpx.AsyncClient(transport=ASGITransport(app))`
  + `patch("api.core._get_pool")` (async handlers: `_get_async_pool`).

### Frontend
- **Tab files < 600 lines** → split into `frontend/src/tabs/<tab-name>/<Subpanel>.tsx`.
- **All HTTP via `src/api/queries/<module>.ts` `fetchJson`** — never raw `fetch(` in tabs/components.
- **Charts read theme from `useThemeContext()`/`useChartColors()`** — no `theme` prop, no inline hex.
- **New API prefix → BOTH `frontend/vite.config.ts` `API_PATH_PREFIXES` AND the
  `frontend/src/api/queries/index.ts` barrel**, same change (else frontend gets HTML not JSON).

### Data Pipeline
- **New data sources extend the standard pipeline** — register `DomainSpec` in
  `common/core/domain_specs.py`, add to `etl_config.yaml` `domain_order` + `normalize-all` +
  `load-all`. Never standalone.
- **Forecast promotion**: `fact_production_forecast_staging` → `fact_production_forecast` via
  `POST /backtest-management/{model_id}/promote` (champion routes per-DFU, per-month via the
  promoted experiment's `data/champion/experiment_{id}_winners.csv`, keyed on `(item_id, loc,
  startdate)`; `dfu_assignments.csv` is dead code, unreferenced). `generate` accepts `horizon` +
  `confidence_intervals` query params → script flags. Per-DFU reads: `/forecast/production/staging`
  (future) + `/forecast/candidate` (past backtest, from `fact_candidate_forecast`) drive the Item
  Analysis overlay. Detail in skill `forecasting-patterns`.

---

## File Placement

Place new files in the correct domain subdirectory. Never add to a flat parent when a domain subdir exists.

| New code type | Place in | Example |
|---|---|---|
| Inventory/supply router | `api/routers/inventory/` | `inv_planning_new_feature.py` |
| Forecast/accuracy/cluster router | `api/routers/forecasting/` | `forecast_ensemble.py` |
| SOP/control tower router | `api/routers/operations/` | `supply_risk.py` |
| Auth/config/admin router | `api/routers/platform/` | `audit_log.py` |
| AI/chat/analysis router | `api/routers/intelligence/` | `anomaly_detect.py` |
| Dashboard/jobs router | `api/routers/core/` | `system_metrics.py` |
| Generic domain endpoint | `api/routers/domains.py` | extend the catch-all |
| DB/config/date utility | `common/core/` | `new_helper.py` |
| ML algorithm/backtest | `common/ml/` | `ensemble_strategy.py` |
| Clustering library | `common/ml/clustering/` | `features.py`, `training.py` |
| SKU feature computation | `common/ml/sku_features/` | `compute.py`, `classifiers.py` |
| Inventory math (safety stock, ROP, EOQ) | `common/inventory/` | `safety_stock.py` |
| DQ/exception engine | `common/engines/` | `alert_engine.py` |
| Scheduler/cache/webhook | `common/services/` | `event_bus.py`, `perf_profiler.py` |
| AI/LLM utility | `common/ai/` | `prompt_builder.py` |
| ETL script | `scripts/etl/` | `load_new_source.py` |
| ML training/tuning | `scripts/ml/` | `train_new_model.py` |
| Forecast generation | `scripts/forecasting/` | `run_ensemble.py` |
| Inventory planning | `scripts/inventory/` | `calculate_new_metric.py` |
| Operations/SOP | `scripts/ops/` | `generate_report.py` |
| AI/embeddings | `scripts/ai/` | `retrain_embeddings.py` |
| Backend unit test | `tests/unit/test_<module>.py` | `test_new_helper.py` |
| API endpoint test | `tests/api/test_<feature>.py` | `test_forecast_ensemble.py` |

When adding a router: (1) `app.include_router()` in `api/main.py` BEFORE `domains.py`, (2) add prefix to `frontend/vite.config.ts` proxy, (3) add to `frontend/src/api/queries/index.ts` barrel.

---

## Patterns & Pitfalls

The detailed symptom → cause → fix catalogs now live in **on-demand skills** (they load when
the matching work happens, instead of taxing every turn). Reach for:

| Domain | Skill | Covers |
|---|---|---|
| Python idioms + hard rules | `python-patterns` | planning date, no bare except, psycopg3, Pydantic v2, canonical imports, YAML config |
| FastAPI routers | `api-design` | 422/`get_conn`, `domains.py` last, safe 5xx, read-replica, register-a-router checklist |
| SQL / ETL / MV / pools | `postgres-patterns` | query patterns, COPY `write_row`, null/sales-filter rules, MV tiered refresh, 3-pool invariant |
| Forecasting engine | `forecasting-patterns` | backtest/champion/production lifecycle, per-cluster training, leakage guards, cold-start/intermittent, SHAP stages, dim_sku grain, formulas |
| Tests | `tdd-workflow` | `make_pool`/`make_async_pool`, `TestQueryWrapper`, echarts/react-virtual mocks, test placement |
| Security | `security-review` | secrets, SQL injection, API-key/role guards, PII in logs |
| Pre-PR verification | `verification-loop` | type/lint/test/security phases |

Still-here quick facts that don't belong to one skill:
- **Frontend "HTML instead of JSON"** → missing Vite proxy entry → add to `frontend/vite.config.ts`
  + `frontend/src/api/queries/index.ts`; `make audit-routers` to verify. (Full checklist in `api-design`.)
- **Customer-demand load**: `site` → `location_id` via `dim_location.site_id`; `posting_prd`
  (YYYYMM) → `startdate`; `demand_qty = MAX(0, demand_cases)`;
  `sales_qty = MAX(0, demand_cases - oos_cases)`. Supports `--replace`, `--month YYYY-MM`.
- **GPU**: `DEMAND_GPU=on|off|auto` (default `auto`) — `cupy` (Monte Carlo), `numba` (seasonality JIT); all fall back.
- **Admin** (`api/routers/platform/admin.py`, `require_api_key`): `POST /admin/llm/reset` clears
  LLM singletons; `POST /admin/tuning/invalidate-stale` clears `cluster_tuning_profile.stale`.

---

## Commands Cheatsheet

The targets Codex actually invokes. For the full operator command list (~130 targets — pipelines, ML, fresh-load, perf, setup-*), see `docs/operations-manual/` and `Makefile`.

```bash
# Run services
make api               # FastAPI on :8000
make ui                # React dev server on :5173 (npm install via `make ui-init`)
make dev               # Docker + API + UI

# Test
make test              # Backend pytest (DB mocked, ~0.7s)
make ui-test           # Frontend vitest (~1.5s)
make test-all          # Backend + frontend
make e2e               # Playwright E2E (needs API on :8000)

# Quality
make lint              # Ruff lint check + fix
make format            # Ruff format
make type-check        # Mypy
make ai-sync-check     # Verify hooks/skills/config wiring

# Routes
make audit-routers     # Verify main.py ↔ vite.config.ts parity
make new-router DOMAIN=forecasting NAME=my_feature  # Scaffold

# Data (incremental)
make pipeline-refresh  # Detect changes, reload deltas
make pipeline-inventory-refresh

# Health
make health            # DB row counts + API health

# Performance
make perf-script SCRIPT=<name>  # Profile a script (read-only)
```

OpenAPI types: from `frontend/`, `npm run gen:types` regenerates `src/api/generated/schema.ts` from `/openapi.json` (API must be running).

---

## Project Structure

```
api/                  # FastAPI — main.py, core.py, pool.py, llm.py, routers/{inventory,forecasting,operations,platform,intelligence,core}/, domains.py (mounted LAST)
common/               # Shared Python — core/, ml/{clustering,sku_features,…}, engines/, services/, ai/, auth.py
scripts/              # Pipelines — etl/, ml/, forecasting/, inventory/, ops/, ai/, algorithm_testing/, tools/, db/
frontend/             # React + Vite — src/tabs/, src/api/queries/, src/components/, vite.config.ts, nginx.conf
config/               # ~42 YAML files; master forecast config: forecasting/forecast_pipeline_config.yaml
sql/                  # 130+ DDL migrations (numeric prefix)
tests/                # api/, unit/  (3900+ backend tests)
docs/                 # ARCHITECTURE (incl. platform overview + feature catalog), ENTERPRISE_ARCHITECTURE, operations-manual/ (operator runbook), specs/
data/                 # Generated artifacts (gitignored) — input/, staged/, backtest/, champion/, clustering/, models/, tuning/
```

For full architecture, data flow, dimension/fact tables, and MV catalog, see `docs/ARCHITECTURE.md`.

---

## Workflow & Hooks

### Review + refactor at each step (workflow contract)
Before declaring any change "done":
1. **Re-read your diff.** Imagine someone else wrote it — what would you flag?
2. **Fix what you'd flag.** Naming, dead code, duplicated helpers, leaky abstractions, unclear branches.
3. **Refactor for clarity if messy.** Don't ship a working-but-ugly diff just because tests pass.
4. **Then verify** (tests + lint + gates) and only then report complete.

Multi-step plans: this pass happens at **each** step, not only the end. Multi-agent batches: each agent reviews + refactors its own diff before reporting; the orchestrator does a cross-cutting review of the merged result. The auto-trigger `code-reviewer` agent (below) backstops this for commits — but Codex must self-review before invoking the reviewer; otherwise the reviewer becomes a crutch.

### Automation and shared gates
- **Claude auto-hooks** live in `.claude/settings.json` and delegate to `scripts/ai_checks/`.
- **Codex equivalent**: run the same shared gates directly when relevant, especially
  `make ai-sync-check`, `make audit-routers`, and `make test-all`.
- **Pre-commit protection** blocks `git commit` if new AGENTS.md/CLAUDE.md rule violations are detected.

### How hooks, skills, and agents work together

**Hooks/gates** (`.claude/hooks/`, `.pre-commit-config.yaml`, and `scripts/ai_checks/`) are purely mechanical quality gates (type check, lint, secret scan, test run).
**Skills** (`.agents/skills` symlinked to `.claude/skills/`) are on-demand reference guides loaded when work context matches (e.g., "Writing a Python router" → `api-design` skill loads automatically).
**Reviewer roles** map to Codex subagents/tools when available; otherwise, apply the same checklists manually. Each reviewer role loads relevant skills.

**Recommended agents by task** (manual invocation):
- Python/FastAPI changes: `python-reviewer` (loads `python-patterns`, `api-design` for routers)
- SQL/database changes: `database-reviewer` (loads `postgres-patterns`)
- New feature/test-first: `tdd-guide` (loads `tdd-workflow`)
- Forecasting/ML: `forecasting-developer` / `forecasting-qa` (load `forecasting-patterns`)
- Pre-PR comprehensive check: `code-reviewer` (loads `security-review`, `verification-loop`)
- Complex architecture: `planner` (requires user CONFIRM before editing)
- Pipeline scripts: run `make perf-script SCRIPT=<name>` for profiling

### Tests are mandatory (no exceptions)
- New `common/` module → `tests/unit/test_<module>.py`
- New API endpoint → `tests/api/test_<feature>.py` (httpx AsyncClient + ASGITransport, `make_pool` factory)
- New React component → `src/components/__tests__/<Component>.test.tsx`
- New React hook → `src/hooks/__tests__/<hook>.test.ts`
- New tab → `src/tabs/__tests__/<Tab>.test.tsx`
- New sidebar tab → `frontend/e2e/tests/navigation.spec.ts` (semantic selectors only — `getByRole`/`getByText`, never CSS classes; use `navigateToTab()` fixture)
- Removed feature → remove its tests in the same change
- Run `make test-all` after every change

### Feature integration checklist
1. **DB**: DDL in `sql/` (next sequence); add to `db-truncate-data` + `refresh-mvs-tiered` Make targets and to `docs/operations-manual/11-maintenance-troubleshooting.md` cleanup section. New input source → register `DomainSpec`, add to `etl_config.yaml` `domain_order`, add to `normalize-all` + `load-all`. New forecast → use candidate→production promotion.
2. **Backend**: router in correct `api/routers/<domain>/`, `get_conn()`, `%s`, `app.include_router()` BEFORE `domains.py`, `Depends(require_api_key)` on writes, config in YAML.
3. **Frontend**: query module in `src/api/queries/`, Vite proxy entry, queries barrel entry, theme via context.
4. **Tests**: backend + frontend per the table above, then `make test-all`.
5. **Docs**: update `docs/ARCHITECTURE.md` (incl. its Feature Catalog §26), the relevant `docs/specs/<domain>/<spec>.md`, `docs/specs/01-foundation/01-infrastructure.md` "Implemented Features", and `docs/operations-manual/` when operational procedures change. `docs/ENTERPRISE_ARCHITECTURE.md` carries inline self-update rules — follow them. Update this `AGENTS.md` only if a new critical rule applies.
6. **Verify**: `make audit-routers`, then `make test-all`.

What does **not** require doc updates: bug fixes without interface changes, internal refactors, typo fixes.

---

## Do Not

### Files & directories
- Do not commit `__pycache__/`, `.pyc`, `.venv/`, `.env`, `credentials.*`, `*secret*`, `*.key`.
- Do not modify `data/*.csv` manually — they are generated by normalize scripts.

### Code drift
- Do not add backward-compat shims when moving modules — rewrite all importers in the same change.
- Do not recreate deleted configs (`clustering_config.yaml`, `model_competition.yaml`, `lgbm_tuning_config.yaml`, `production_forecast_config.yaml`, `backtest_sampling_config.yaml`, `algorithm_config.yaml`).
- Do not put hardcoded API keys, tokens, passwords, or hex chart colors anywhere.
- Do not use string interpolation for SQL values — always parameterized.

### Don't fabricate
- **Don't fabricate Make targets** — verify against `Makefile` before invoking.
- **Don't fabricate router paths** — verify against `api/main.py` `include_router` calls.
- **Don't fabricate column or table names** — query the schema before referencing.
- **Don't fabricate Vite proxy prefixes** — read `frontend/vite.config.ts`.
- **When a test fails, fix the code or honestly adjust expected values.** Never delete or weaken assertions to make a test pass.
- **Don't invent config keys** — read the YAML before referencing.

---

## Detailed Reference (moved out)

- **Architecture, data flow, dimension/fact tables, MV catalog** → `docs/ARCHITECTURE.md`
- **Operator runbook (setup → ops → maintenance), Make command list, DB cleanup/fresh-recreate** → `docs/operations-manual/`
- **Platform overview, quick start, feature catalog** → `docs/ARCHITECTURE.md` §25–27
- **Design specs (8 domains, 80+ files)** → `docs/specs/README.md`
- **Enterprise architecture (TOGAF-style, ADRs, transition roadmap)** → `docs/ENTERPRISE_ARCHITECTURE.md`
- **Vite proxy authoritative list** → `frontend/vite.config.ts` (run `make audit-routers` for parity)
- **Config inline comments** → each YAML in `config/` documents itself
