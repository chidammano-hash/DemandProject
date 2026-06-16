# CLAUDE.md — Supply Chain Command Center

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

### Workflow (applies to every change)
- **Review + refactor at each step.** Before reporting any change as complete: re-read your own diff, fix anything you'd flag in code review, refactor for clarity if the diff is messy. In multi-step plans (parallel agents, multi-file refactors), each step ends with this self-pass — don't defer it to the end. **[FREQUENTLY VIOLATED]**
- **Docs updated in the same commit as the code.** When a change affects architecture, APIs, schemas, conventions, infra, or operational procedures, update the relevant docs (`docs/ARCHITECTURE.md`, `docs/ENTERPRISE_ARCHITECTURE.md`, `docs/operations-manual/`, `docs/specs/<domain>/`, this file if rules changed) in the same commit. Don't ship code today and document tomorrow — they drift permanently. The "Documentation Update Rules" mapping in the Workflow & Hooks section tells you which doc maps to which kind of change. **[FREQUENTLY VIOLATED]**

### Backend / Python
- **`date.today()` forbidden outside `common/core/planning_date.py`.** Use `get_planning_date()`. Env overrides: `PLANNING_DATE`, `USE_SYSTEM_DATE`. **[FREQUENTLY VIOLATED]**
- **`Path(__file__).resolve().parents[N]` forbidden at module level.** Allowed only in `if __name__ == "__main__"` bootstrap. Use `from common.core.paths import PROJECT_ROOT, CONFIG_DIR, DATA_DIR, SQL_DIR, SCRIPTS_DIR`. **[FREQUENTLY VIOLATED]**
- **`except Exception` forbidden.** Catch specific (`psycopg.Error`, `ValueError`, …) and log via `logger.exception()`. `# noqa: BLE001 — <reason>` requires inline justification. **[FREQUENTLY VIOLATED]**
- **`get_conn()` not `Depends(_get_pool)`** in `inv_planning_*.py` routers. `Depends(_get_pool)` causes 422 in tests because FastAPI inspects MagicMock signatures. **[FREQUENTLY VIOLATED]**
- **psycopg3 uses `%s` placeholders** — never `$1`/`$2`. All SQL in scripts and routers. **[FREQUENTLY VIOLATED]**
- **`domains.py` mounted LAST** in `api/main.py` — its catch-all `{domain}` shadows other routes.
- **All routers MUST use `APIRouter(prefix="/...")`** plus short paths in `@router.get`/`.post`/etc. Never the full path inside the decorator.
- **5xx responses never interpolate exception text.** Pattern: `logger.exception("...")` then `raise HTTPException(status_code=500, detail="<short verb-phrase>")`. No `f"...{exc}"`, no `str(exc)` in 500 details.
- **Every write endpoint** (`@router.post`/`put`/`delete`/`patch`) **MUST have `dependencies=[Depends(require_api_key)]`** at router or per-route level.
- **Identifier interpolation requires `psycopg.sql.Identifier`** — never f-string identifier interpolation in SQL. Values use `%s`.
- **Pydantic v2 only** — `model_config = ConfigDict(...)`, never `class Config:`.
- **No backward-compat shims.** When moving a module, rewrite all importers in the same change. Canonical: `from common.core.db import get_db_params`.
- **Routers/modules > 800 LoC must split** by sub-feature into a domain folder.
- **No `_row_to_dict` outside `common/core/sql_helpers.py`.** Import `row_to_dict_from_cursor` / `row_to_dict_from_cols`.
- **Read-only analytics endpoints opt into `get_async_read_only_conn()`** (or sync sibling `get_read_only_conn()`). Routes to a Postgres read replica when `READ_REPLICA_URL` is set; otherwise falls back to the primary pool with no behaviour change. Use ONLY for queries that tolerate replica lag — never for read-after-write flows. Currently used by 7 customer-analytics endpoints. See `docs/operations-manual/11-maintenance-troubleshooting.md` "Read Replica Deployment".

### ML / Forecasting
- **All tree-model `.fit()` and instantiation goes through `common/ml/model_registry.py`** (`fit_model()`, `build_model()`). Direct `LGBMRegressor()` / `CatBoostRegressor()` / `XGBRegressor()` outside `model_registry.py` is a defect — applies to tuning, training, backtest, production, meta-learner.
- **All ML hyperparameters live in `forecast_pipeline_config.yaml` `algorithms.<id>.params`.** No defaults like `kwargs.get("n_estimators", 200)` in Python.
- **Forecast quantity column constant: `from common.core.constants import FORECAST_QTY_COL`.** Never the string literal `"basefcst_pref"`.
- **Foundation-model loaders live in `common/ml/foundation_backtest.py`.** Never import from `scripts/algorithm_testing/`.
- **`ml_cluster` is metadata, NOT a model feature.** Listed in `METADATA_COLS`, excluded by `get_feature_columns()`. Still merged via `build_feature_matrix()` for per-cluster partitioning.

### Testing
- **All API tests use `make_pool` from `tests/api/conftest.py`.** Hand-rolled `MagicMock` chains on `psycopg.connect` are forbidden. Use `httpx.AsyncClient(transport=ASGITransport(app))` with `patch("api.core._get_pool")`.

### Frontend
- **Tab files MUST be < 600 lines.** Split into `frontend/src/tabs/<tab-name>/<Subpanel>.tsx`.
- **All HTTP from frontend goes through `src/api/queries/<module>.ts` using `fetchJson`.** Never raw `fetch(` in tabs/components.
- **No `: any`, `<any>`, or `as any` in `src/api/queries/`.** Mirror the backend Pydantic schema as a TS interface.
- **Charts: never accept `theme` as a prop.** Read from `useThemeContext()` / `useChartColors()`. Inline hex colors are forbidden in `tabs/` and `components/`.
- **New API path prefixes: add to BOTH `frontend/vite.config.ts` `API_PATH_PREFIXES` AND the barrel `frontend/src/api/queries/index.ts`** in the same change. Otherwise frontend gets HTML instead of JSON.

### Data Pipeline
- **New data sources MUST extend the standard pipeline.** Add to `normalize-all` + `load-all` Make targets, register `DomainSpec` in `common/core/domain_specs.py`, add to `etl_config.yaml` `domain_order`. Never standalone.
- **Forecast promotion**: predictions land in `fact_candidate_forecast`, are promoted to `fact_production_forecast` via `POST /backtest-management/{model_id}/promote`. Champion uses `data/champion/dfu_assignments.csv`.

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

Each entry: **symptom → cause → fix**. File paths are anchors.

### API & Routers
- **422 in tests on `inv_planning_*` endpoints** → `Depends(_get_pool)` parameter inspected by FastAPI as MagicMock → use `get_conn()` directly.
- **Mock tuples reject the response** → mock column count != SQL select column count → align mock tuple to exact column order/count of the query.
- **Catch-all swallows a new route** → `domains.py` mounted before the new router → move `app.include_router(domains.router)` to be last in `api/main.py`.
- **`POST /{model_id}/train` 400** → only `type === "tree"` models support production training; foundation/`deep_learning` rejected by design.
- **Import error after restructure** → using a pre-restructure path (e.g. `common.<old>` instead of `common.core.<…>`) → use the canonical path: `from common.core.db import get_db_params`. Run `grep -rln "from common\.<old> " --include="*.py"` to find stragglers.
- **Admin endpoints**: `api/routers/platform/admin.py`, `require_api_key` guarded. `POST /admin/llm/reset` clears OpenAI/Anthropic singletons. `POST /admin/tuning/invalidate-stale` clears `cluster_tuning_profile.stale`; logs `noop` if column absent or psycopg error.

### Frontend
- **"HTML instead of JSON"** → missing Vite proxy entry for new API prefix → add to `frontend/vite.config.ts` and `frontend/src/api/queries/index.ts` barrel; run `make audit-routers` to verify parity. See `frontend/vite.config.ts` for the authoritative proxy list.
- **Test wrappers**: wrap with `TestQueryWrapper` from `src/tabs/__tests__/test-utils.tsx`. Mock API with `vi.mock("../api/queries")`. Mock `echarts-for-react` for charts, `@tanstack/react-virtual` for virtualized rows.

### ML & Forecasting
- **Dimension mismatch in SHAP** → feature stripped from `feature_cols` but model trained with full set → pre-SHAP stages (duplicate/variance/correlation in `shap_selector.py`) prune the *selection pool* only; SHAP input must match the trained set.
- **Cold-start DFU has no forecast** → < `cold_start_min_months` (3) months history → DFUs with 3–11 months route to `cold_start_model_id` (rolling_mean); < 3 months are skipped (absolute floor). Config: `forecast_pipeline_config.yaml` `production_forecast`.
- **Multi-stage feature selection** (`shap_selector.py`): 4 stages per timeframe — (0) duplicate alias removal, (1) near-zero variance, (2) correlation pre-filter, (3) SHAP cumulative. Config keys: `correlation_filter`, `variance_filter`. **Per-cluster SHAP**: `compute_timeframe_shap_per_cluster()` returns `dict[str, list[str]]`. Sparse clusters skip SHAP. Stratified 50/50 sampling for >50% zeros.
- **Cluster strategy resolution**: `algorithms.<name>.cluster_strategy` (default `"per_cluster"`). Tree/statistical only — foundation/DL always global.
- **Clustering master switch**: `clustering.enabled` in `forecast_pipeline_config.yaml`. When `false`, all backtests fall back to `global` regardless of per-algorithm setting. Check via `is_clustering_enabled()` from `common.core.utils`.
- **Per-cluster tuning profiles** (`config/forecasting/cluster_tuning_profiles.yaml`): Phase 1 matches by `match_criteria.cluster_name`, Phase 2 by statistical criteria (mean_demand, cv_demand, zero_demand_pct). First match wins per `_PROFILE_PRIORITY`. Resolved by `resolve_cluster_params()` in `backtest_framework.py`.
- **Intermittent routing**: clusters with > 70% zero-demand rows (`backtest.intermittent_threshold`) → rolling mean baseline (`backtest.baseline_intermittent`). Early stopping uses 5% patience standard, 10% sparse/intermittent (`compute_early_stop_patience()`).
- **Clustering library** lives in `common/ml/clustering/` — `features.py`, `training.py`, `labeling.py`, `scenario.py`. Params stored in `cluster_experiment` table (promoted row), NOT YAML. Operates on **SKUs** (item+location), not DFUs.
- **SKU features** computed once in `common/ml/sku_features/` and stored in `dim_sku`; clustering reads pre-computed values. Config: `config/forecasting/sku_features_config.yaml`.
- **`model_registry.build_model(algorithm_id, params=None)`**: reads `algorithms.<id>` entry, translates via `to_native_params()`, returns configured tree estimator. Foundation/DL/statistical → `_FoundationStub`. Unknown id → `UnknownAlgorithm` (subclass of `ValueError`).
- **Chunked fact-table reads**: production ML/forecasting scripts use `read_sql_chunked()` / `stream_query_in_chunks()` from `common.core.sql_helpers` for any query that scans a fact table (sales, forecast, inventory snapshots). Bare `pd.read_sql(...)` without `chunksize` over a fact table is a defect at 40× scale — it OOMs the worker. See `scripts/ml/train_meta_learner.py` and `scripts/forecasting/generate_production_forecasts.py` for the pattern.

### Data Loading
- **Staged CSV convention**: normalized CSVs land in `data/staged/`. `DomainSpec.clean_file` embeds the prefix (`"staged/itemdata_clean.csv"`); resolution is `ROOT / "data" / spec.clean_file`. ML intermediates (`clustering_features.csv`, `seasonality_results.csv`) also live under `data/staged/`. No new normalized output at `data/` root.
- **Null normalization**: `''`, `'null'`, `'none'`, `'NA'` → NULL during load.
- **Type casting**: integer/float/date fields auto-cast with null coercion.
- **Sales filtering**: only `TYPE=1` rows enter `fact_sales_monthly`.
- **Time dimension**: auto-generated 2020–2035, not from a file.
- **Forecast `model_id`**: default `'external'` for source-system feeds. `UNIQUE(forecast_ck, model_id)` prevents duplicates.
- **Execution-lag loading**: dual-path insert with phase ordering — archive loaded BEFORE staging mutation. See `docs/specs/02-forecasting/03-backtest-framework.md`.
- **Customer demand**: `site` → `location_id` via `dim_location.site_id`; `posting_prd` (YYYYMM) → `startdate` (YYYY-MM-01); `demand_qty = MAX(0, demand_cases)`; `sales_qty = MAX(0, demand_cases - oos_cases)`. Supports `--replace` and `--month YYYY-MM`.

### MV & Performance
- **Stale MV after refresh** → wrong order (dependent before source) → use tiered refresh; see `Makefile` `refresh-mvs-tiered`.
- **Stub table pattern**: an MV depending on a future feature's table → create with `CREATE TABLE IF NOT EXISTS` and minimum columns. LEFT JOIN yields NULL → neutral scores until real data flows.
- **Connection pool exhaustion** → too many concurrent requests vs `max_size` (now 20) → set `POOL_MAX_SIZE`; check unclosed connections.
- **Profiling**: wrap major stages with `profiled_section()` from `common/services/perf_profiler.py` instead of raw `time.time()`. `wrap_connection()` enforces `default_transaction_read_only = true` and rolls back. Thresholds in `config/platform/perf_config.yaml` — never hardcoded.
- **GPU**: `DEMAND_GPU=on|off|auto` (default `auto`). `cupy` for Monte Carlo; `numba` for seasonality JIT. All scripts fall back gracefully.

### Config
- **All config in YAML**, loaded via `load_config(name)` from `common.core.utils`. No magic numbers. Forecast pipeline master config: `config/forecasting/forecast_pipeline_config.yaml` (algorithm roster + params, backtest, tuning, champion, forecast, run tracking). Helpers: `load_forecast_pipeline_config()`, `get_algorithm_roster(stage=…)`, `get_competing_model_ids()`, `get_forecastable_model_ids()`, `get_algorithm_params(model_id)`. Old configs (`model_competition.yaml`, `lgbm_tuning_config.yaml`, `production_forecast_config.yaml`, `backtest_sampling_config.yaml`, `algorithm_config.yaml`, `clustering_config.yaml`) deleted.
- **`_includes` directive** at top of any YAML loads listed files first as defaults — used for `shared_constants.yaml` inheritance.
- **Inline comments required** on every key in every config YAML — explanation, valid options, default. Use `# ═══ SECTION ═══` headers. Update comments when values change.

### Code Quality
- **DB params**: `from common.core.db import get_db_params`. No inline connection helpers.
- **Timestamp helper**: `from common.core.utils import _ts`. No per-file `_ts()`.
- **Structured logging**: `logging.getLogger(__name__)` — never raw `print()`. `basicConfig()` only in `__main__`.

### Formulas
- **Accuracy**: `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- **Bias**: `(SUM(Forecast) / SUM(History)) - 1`
- **WAPE**: `SUM(|F-A|) / |SUM(A)|`

---

## Commands Cheatsheet

The targets Claude actually invokes. For the full operator command list (~130 targets — pipelines, ML, fresh-load, perf, setup-*), see `docs/operations-manual/` and `Makefile`.

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

Multi-step plans: this pass happens at **each** step, not only the end. Multi-agent batches: each agent reviews + refactors its own diff before reporting; the orchestrator does a cross-cutting review of the merged result. The auto-trigger `code-reviewer` agent (below) backstops this for commits — but Claude must self-review before invoking the reviewer; otherwise the reviewer becomes a crutch.

### Always-active automation (from `.claude/settings.json`)
- **PostToolUse (Write/Edit)**: ruff on Python, anti-pattern checks on SQL, auto-runs edited test files.
- **PreToolUse (Bash)**: blocks `git commit` if ruff or pytest fails.

### Auto-trigger agents
| Trigger | Agent | Skills |
|---|---|---|
| After Python changes | `python-reviewer` | `python-patterns`; routers also `api-design` + `backend-patterns`; SQL-heavy also `postgres-patterns` |
| SQL/schema change | `database-reviewer` | verify `%s`, explicit columns, indexes |
| New feature / bug fix | `tdd-guide` | `tdd-workflow` — write test FIRST (failing), then implement |
| Before commit | `code-reviewer` | `security-review` (no secrets/injection/keys), `verification-loop` (build/types/lint/tests) |
| Complex multi-file | `planner` | wait for user CONFIRM before editing |
| New/modified pipeline script | — | run `make perf-script SCRIPT=<name>` |

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
5. **Docs**: update `docs/ARCHITECTURE.md` (incl. its Feature Catalog §26), the relevant `docs/specs/<domain>/<spec>.md`, `docs/specs/01-foundation/01-infrastructure.md` "Implemented Features", and `docs/operations-manual/` when operational procedures change. `docs/ENTERPRISE_ARCHITECTURE.md` carries inline self-update rules — follow them. Update this `CLAUDE.md` only if a new critical rule applies.
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
