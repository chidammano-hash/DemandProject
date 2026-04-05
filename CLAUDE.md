# CLAUDE.md — Supply Chain Command Center

## Project Overview

**Supply Chain Command Center** is a unified supply chain planning and execution platform. It ingests sales, forecast, and inventory data, stores it in PostgreSQL, and serves a React UI for interactive analytics.

**All code lives at the project root** with domain-based organization inside each top-level directory.

---

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
| ML / Clustering | scikit-learn, pandas, scipy, matplotlib, seaborn |
| ML Tracking | MLflow |
| Job Scheduling | APScheduler 3.11 (BackgroundScheduler + ThreadPoolExecutor) |
| E2E Testing | Playwright |
| Python packaging | uv |
| Build | Make |
| Containers | Docker Compose |
| Deployment | Docker (API + Nginx frontend) |

---

## Project Structure

```
api/                         # FastAPI backend
├── main.py                  # App entry point — mounts 69 routers
├── core.py                  # SQL helpers, filtering, pagination
├── pool.py                  # Connection pool management
├── llm.py                   # OpenAI client management
└── routers/
    ├── inventory/           # 22 routers — inv_planning_*, inventory, supply, fill_rate
    ├── forecasting/         # 10 routers — accuracy, shap, competition, blended, fva, unified_model_tuning, cluster_experiments
    ├── operations/          # 10 routers — sop, control_tower, storyboard, events
    ├── platform/            # 10 routers — auth, users, config, notifications, webhooks
    ├── intelligence/        # 4 routers — ai_planner, chat, intel, analysis
    ├── core/                # 2 routers — dashboard, jobs
    └── domains.py           # Catch-all generic domain endpoint (mounted LAST)

common/                      # Shared Python modules (backward-compat shims at root)
├── core/                    # db, utils, planning_date, constants, sql_helpers, domain_specs
├── ml/                      # backtest_framework, model_registry, champion_strategies, tuning, shap, features
├── engines/                 # dq_engine, exception_engine
├── services/                # job_registry, job_scheduler, notifications, webhooks, cache
├── ai/                      # ai_planner
└── auth.py                  # Authentication helpers

scripts/                     # Pipeline scripts
├── etl/                     # normalize, load (7 scripts)
├── ml/                      # backtest, clustering, tuning, champion (16 scripts)
├── forecasting/             # production, quantile, blended, consensus (7 scripts)
├── inventory/               # safety stock, eoq, replenishment, rebalancing (18 scripts)
├── ops/                     # sop, health, dq fixes (7 scripts)
└── ai/                      # insights, embeddings (2 scripts)

frontend/                    # React + Vite + TypeScript
├── src/tabs/                # 21 tab components + sub-panels
├── src/api/queries/         # 39 domain API modules
├── src/components/          # Shared UI components
├── Dockerfile               # Nginx multi-stage build
└── nginx.conf               # SPA fallback + API reverse proxy

config/                      # ~45 YAML config files organized by concern:
│   ├── etl_config.yaml                # ETL pipeline: domain load order, MV refresh, parallel workers
│   ├── forecast_pipeline_config.yaml  # ML pipeline: algorithm roster + params, backtest, tuning, champion, forecast
│   ├── shared_constants.yaml          # Shared constants (service levels, z-table, financial defaults, guard rails)
│   ├── inventory_planning_config.yaml # Merged inventory planning (lead time, simulation, projection)
│   ├── tune_strategies.yaml           # Merged tune strategies (LGBM, CatBoost, XGBoost)
│   └── ...                            # ~40 more configs (clustering, inventory, ops, etc.)
sql/                         # 87 DDL migration files
tests/                       # 2762+ backend tests (api/ + unit/)
docs/                        # ARCHITECTURE, PLATFORM_GUIDE, RUNBOOK, specs/
data/                        # Generated ML artifacts + input CSVs (gitignored)
archive/                     # Archived reference materials
```

All paths relative to the project root. Use Glob/Grep to discover specific files.

---

## Commands

All run from the project root.

```bash
# Setup
make init              # Create .venv, install uv, sync deps
make up                # Start Docker (Postgres, MLflow)
make down              # Stop services
make db-apply-sql      # Apply DDL schemas

# Data Pipeline
make normalize-all     # CSV → clean CSV (all 10 datasets)
make load-all          # Clean CSV → Postgres + refresh views
make load-forecast-replace  # Reload external forecast only
make pipeline-full     # Full reload: normalize + load + refresh MVs
make pipeline-refresh  # Incremental: detect changes, reload only deltas
make pipeline-inventory        # Full reload inventory domain only
make pipeline-inventory-refresh # Incremental inventory refresh only
make normalize-customer-demand # Normalize customer demand CSVs
make load-customer-demand      # Load customer demand (full replace)
make load-customer-demand-month MONTH=YYYY-MM  # Reload single partition
make pipeline-customer-demand  # Normalize + load customer demand

# Run Services
make api               # FastAPI on :8000
make ui-init           # Install npm deps
make ui                # React dev server on :5173

# ML Pipelines
make cluster-all       # Full clustering pipeline
make backtest-all      # All backtests (tree + foundation models)
make backtest-load-all # Load all backtest predictions into Postgres
make backtest-load-all-bulk # Load all with single index cycle (~4x faster)
make backtest-load-bulk # Load 4 core models in bulk (lgbm, catboost, xgboost, chronos)
make backtest-load-main-only MODELS="..." # Load to main table only (skip archive)
make backtest-load-archive-only MODELS="..." # Load to archive only (skip main)
make backtest-chronos  # Chronos T5 (46M, ~2.5h)
make backtest-bolt     # Chronos Bolt (205M, ~12min)
make backtest-chronos2 # Chronos 2 zero-shot (821M, ~5.5h)
make backtest-chronos2e # Chronos 2 Enriched with 31 covariates (~6h)
make customer-features # Pre-compute 34 customer-derived features from customer demand
make backtest-cust-enriched-all # LGBM/CatBoost/XGBoost with customer features
make backtest-bolt-hier # Chronos Bolt hierarchical (customer bottom-up + reconciliation)
make champion-all      # Meta-learner + simulate + champion select
make tune-all          # Bayesian hyperparameter tuning (all models)
make tune-cust-enriched-all # Bayesian tuning for customer-enriched tree models
make expert-panel      # Expert Panel algorithm selection test (5000 DFUs, ~30 min)
make expert-panel-quick # Quick Expert Panel test (1000 DFUs, ~8 min)
make seasonality-all   # Detect + write seasonality profiles
make forecast-generate # Production forecast inference

# Testing
make test              # Backend pytest (~0.7s, DB mocked)
make ui-test           # Frontend vitest (~1.5s)
make test-all          # Backend + frontend
make test-cov          # Backend with coverage
make e2e               # Playwright E2E (needs API on :8000)

# Validation
make check-all         # DB row counts + API health
make ai-sync-check     # Verify Claude/Codex shared guidance wiring

# Performance
make perf-report           # Full system performance report with suggestions
make perf-script SCRIPT=X  # Profile specific script (read-only, zero side effects)
make perf-api              # API endpoint performance analysis
make perf-pipeline         # ETL pipeline performance analysis

# Full Pipeline (input CSVs -> ready app)
make setup-all            # Everything: data + ML + planning + ops (~4-6 hours)
make setup-data           # Data only: normalize + load all 11 domains (~30 min)
make setup-planning       # Data + inventory planning, no ML (~1 hour)
make setup-features       # Data + clustering + seasonality + variability
make setup-backtest       # Features + 3 backtests + champion selection
make setup-inv-planning   # Inventory planning (SS, EOQ, policies, exceptions)
make setup-demand-planning # Forecasts + projections + orders + replenishment
make setup-ops            # S&OP + events + financial + storyboard + DQ

# Convenience Aliases
make dev                  # Start Docker + API + UI
make test-quick           # Backend + frontend tests only
make lint                 # Ruff lint check + fix
make format               # Ruff format all Python
make type-check           # Mypy type checking
make health               # DB row counts + API health

# Developer Tooling
make audit-routers        # Check route consistency (main.py vs vite.config.ts)
make new-router DOMAIN=forecasting NAME=my_feature  # Scaffold new router
```

See `Makefile` for the full list (130+ targets including one-time schema setup, inv-planning pipelines, cleanup utilities).

---

## Architecture

### Domain-Driven Generic Design

All datasets extend a single `DomainSpec` dataclass in `common/domain_specs.py`. Scripts and API endpoints are generic — they operate on any domain via `--dataset <name>` or `/domains/{domain}/*`.

**11 Domains:** item, location, customer, time, sku (dimensions); sales, forecast, customer_demand (facts); inventory (dedicated pipeline); sourcing, purchase_order (procurement).

All domains are loaded via `make load-all` (and normalized via `make normalize-all`). New data sources MUST be added to both `normalize-all` and `load-all` targets — never as standalone pipelines.

### Data Flow

```
Source CSV → normalize_dataset_csv.py → clean CSV → load_dataset_postgres.py → PostgreSQL → FastAPI → React UI
```

### API Pattern

- Generic domains: `GET /domains/{domain}/rows`, `/search`, etc.
- Inventory: dedicated `/inventory/*` endpoints
- Pagination: offset/limit (50–1000 rows)
- Auth: `require_api_key` dependency (disabled when `API_KEY` env var unset)
- Reserved word: `class` column aliased as `class_` in responses

---

## Data Models

### Dimension Tables
- Surrogate key `sk`, composite key `ck`, `load_ts`, `modified_ts`
- Full-text search via `pg_trgm` trigram indexes

### Fact Tables
- `fact_sales_monthly`: grain = item + customer_group + location + month + type
- `fact_external_forecast_monthly`: grain = item + loc + forecast_date + actual_month; tracks lag 0–4
- `fact_inventory_snapshot`: grain = item_id + loc + snapshot_date (~198M rows); **monthly range-partitioned** by `snapshot_date`
- `fact_customer_demand_monthly`: grain = item_id + customer_no + location_id + month; **monthly range-partitioned** by `startdate`
- `fact_production_forecast`: grain = item_id + loc + plan_version + month

### Archive Tables
- `backtest_lag_archive`: All-lags (0–4) backtest predictions with `timeframe` column

### Key Materialized Views
- `agg_sales_monthly`, `agg_forecast_monthly` — pre-aggregated KPI queries
- `agg_inventory_monthly` — EOM on-hand, sales, DOS, lead time
- `mv_inventory_forecast_monthly` — inventory-forecast bridge for root cause attribution
- `mv_fill_rate_monthly`, `mv_supplier_performance`, `mv_intramonth_stockout`, `mv_control_tower_kpis`, `mv_network_balance`

See `sql/` for all DDL files.

---

## Critical Rules

These are hard constraints that cause bugs or test failures if violated.

### DB & API Patterns

- **`get_conn()` not `Depends(_get_pool)`** for all `inv_planning_*.py` routers. Using `Depends(_get_pool)` causes 422 errors in tests because FastAPI inspects MagicMock signatures.
- **psycopg3 uses `%s` placeholders** — NOT `$1`, `$2`. All SQL in scripts and routers must use `%s`.
- **Column names in fact tables**: All tables now use `item_id` (standardized from legacy `dmdunit`/`item_no`). Forecast qty column is `basefcst_pref` (not `qty`).
- **`domains.py` mounted last** in `main.py` — it has catch-all `{domain}` path params that would shadow other routes.
- **Shared test pool factory**: Import `from tests.api.conftest import make_pool as _make_pool`. For multi-fetchall endpoints use `cursor.fetchall.side_effect = [list1, list2]`; for single-call use `cursor.fetchall.return_value`.
- **API test pattern**: Use inline `httpx.AsyncClient(transport=ASGITransport(app))` with `patch("api.core._get_pool")`.

### Frontend Patterns

- **Vite proxy is CRITICAL**: `frontend/vite.config.ts` proxies API path prefixes to `:8000`. When adding a new API path prefix, you MUST add a proxy entry or the frontend gets HTML instead of JSON. Current prefixes: `/domains`, `/jobs`, `/clustering`, `/forecast`, `/inventory`, `/dashboard`, `/health`, `/chat`, `/dfu`, `/competition`, `/bench`, `/market-intelligence`, `/inv-planning`, `/fill-rate`, `/control-tower`, `/ai-planner`, `/storyboard`, `/sql-runner`, `/sourcing`, `/purchase-orders`, `/lgbm-tuning`, `/model-tuning`, `/cluster-experiments`, `/champion-experiments`, `/demand-history`.
- **Theme context, not props**: Use `useThemeContext()` or `useChartColors()` — never pass `theme` as a prop from `App.tsx`.
- **Test wrappers**: Wrap components with `TestQueryWrapper` from `src/tabs/__tests__/test-utils.tsx`. Mock API with `vi.mock("../api/queries")`. Mock `echarts-for-react` for chart tests. Mock `@tanstack/react-virtual` for virtualized row tests.

### Code Patterns

- **All config in YAML**: Every module externalizes params into `config/<name>.yaml`. No magic numbers in scripts. Load via `load_config(name)` from `common/utils.py`.
- **Forecast pipeline master config**: `config/forecast_pipeline_config.yaml` is the single source of truth for the ML forecast pipeline. It contains the algorithm roster with inline hyperparameters (under `algorithms.<model_id>.params`), backtest settings, tuning settings, champion selection, production forecast config, and run tracking. Use `load_forecast_pipeline_config()` from `common/utils.py` to load it. Use `get_algorithm_roster(stage=...)` to get algorithms filtered by lifecycle stage (tune/backtest/compete/forecast/expert). Use `get_competing_model_ids()` and `get_forecastable_model_ids()` for common queries. Use `get_algorithm_params(model_id)` to retrieve hyperparameters for a specific algorithm. The old separate configs (`model_competition.yaml`, `lgbm_tuning_config.yaml`, `production_forecast_config.yaml`, `backtest_sampling_config.yaml`, `algorithm_config.yaml`) have been deleted -- all settings now live in the master config.
- **Cold-start DFU routing**: DFUs with < `min_history_months` (12) months of sales history are routed to `cold_start_model_id` (rolling_mean) instead of the champion tree model. DFUs with < `cold_start_min_months` (3) months are skipped entirely. Configured in `config/forecast_pipeline_config.yaml` under `production_forecast`.
- **Clustering master switch**: `clustering.enabled` in `config/forecast_pipeline_config.yaml` is the master switch for the clustering pipeline. When `false`, all backtest scripts auto-fall back to `global` strategy regardless of per-algorithm `cluster_strategy` settings. Check via `is_clustering_enabled()` from `common/utils.py`.
- **Config `_includes` directive**: `load_config(name)` supports an `_includes` key at the top of any YAML file. Listed files are loaded first and merged as defaults, allowing shared constants (e.g., `shared_constants.yaml`) to be inherited without duplication.
- **`cluster_strategy` resolution order**: `forecast_pipeline_config.yaml` algorithm entry (`algorithms.<name>.cluster_strategy`) > default `"per_cluster"`. Only tree/statistical models use this field; foundation/DL models always run globally.
- **Backtest sampling config**: `forecast_pipeline_config.yaml` `backtest_sampling` section is the sole source for sampling settings (the legacy `backtest_sampling_config.yaml` has been deleted).
- **DB params**: All scripts use `from common.db import get_db_params` — no inline connection helpers.
- **Planning date**: All date-sensitive code uses `get_planning_date()` from `common/planning_date.py`, not `date.today()`. Config: `config/planning_config.yaml`. Env overrides: `PLANNING_DATE` or `USE_SYSTEM_DATE`.
- **Timestamp helper**: Import `from common.utils import _ts` — no per-file `_ts()` definitions.
- **`ml_cluster` is always a hard feature** — never stripped from `feature_cols` in per_cluster or global backtest mode. This includes SHAP computation — `shap_selector.py` must NOT strip `ml_cluster` from features (causes dimension mismatch with trained models).
- **Model registry for tree backtests**: Use `common/ml/model_registry.py` for all model-specific logic — `fit_model()`, `get_best_iteration()`, `to_native_params()`. Do NOT add new if/elif/else fit blocks in backtest scripts. Early stopping uses standardized 3% patience via `compute_early_stop_patience()`.
- **Stub table pattern**: When a materialized view depends on a future feature's table, create it with `CREATE TABLE IF NOT EXISTS` and minimum columns. LEFT JOIN produces NULL → neutral scores until real data flows.
- **Backward-compatible imports**: `common/` root has shim modules that re-export from subpackages (e.g., `from common.db import get_db_params` works via shim → `common/core/db.py`). New code may use either path; existing imports remain valid.
- **Structured logging**: Scripts use `logging.getLogger(__name__)` — no raw `print()`. `basicConfig()` only in `__main__` blocks.
- **Exception handling**: Catch specific exceptions (`psycopg.Error`, `ValueError`) — never bare `except Exception`. Always log with `logger.exception()`.
- **GPU acceleration**: `DEMAND_GPU` env var controls GPU usage in backtests (`on`/`off`/`auto`, default `auto`). Optional deps: `cupy` (Monte Carlo simulation GPU arrays), `numba` (seasonality JIT kernels). All scripts fall back gracefully when these are absent.

### Performance Profiling

- **Use `profiled_section()` for major stages**: New scripts should wrap major computation stages with `profiled_section()` from `common/services/perf_profiler.py` instead of raw `time.time()`.
- **DB profiling is read-only**: `wrap_connection()` sets `default_transaction_read_only = true` and always rolls back. Safe for production.
- **Config in YAML**: All perf thresholds in `config/perf_config.yaml`. No hardcoded timing thresholds.

### File Placement Rules

**Always place new files in the correct domain subdirectory. Never add files to a flat parent directory when a domain subdirectory exists.**

| New code type | Place in | Example |
|---|---|---|
| Inventory/supply router | `api/routers/inventory/` | `inv_planning_new_feature.py` |
| Forecast/accuracy/cluster experiment router | `api/routers/forecasting/` | `forecast_ensemble.py`, `cluster_experiments.py` |
| SOP/control tower router | `api/routers/operations/` | `supply_risk.py` |
| Auth/config/admin router | `api/routers/platform/` | `audit_log.py` |
| AI/chat/analysis router | `api/routers/intelligence/` | `anomaly_detect.py` |
| Dashboard/jobs router | `api/routers/core/` | `system_metrics.py` |
| Generic domain endpoint | `api/routers/domains.py` | Add to existing catch-all |
| DB/config/date utility | `common/core/` | `new_helper.py` |
| ML algorithm/backtest | `common/ml/` | `ensemble_strategy.py` |
| DQ/exception engine | `common/engines/` | `alert_engine.py` |
| Scheduler/cache/webhook | `common/services/` | `event_bus.py` |
| AI/LLM utility | `common/ai/` | `prompt_builder.py` |
| ETL/data loading script | `scripts/etl/` | `load_new_source.py` |
| ML training/tuning script | `scripts/ml/` | `train_new_model.py` |
| Forecast generation script | `scripts/forecasting/` | `run_ensemble.py` |
| Inventory planning script | `scripts/inventory/` | `calculate_new_metric.py` |
| Operations/SOP script | `scripts/ops/` | `generate_report.py` |
| AI/embeddings script | `scripts/ai/` | `retrain_embeddings.py` |
| Performance profiling | `common/services/` | `perf_profiler.py` |
| Backend unit test | `tests/unit/test_<module>.py` | `test_new_helper.py` |
| API endpoint test | `tests/api/test_<feature>.py` | `test_forecast_ensemble.py` |

When adding a new router, also:
1. Add the import and `app.include_router()` call in `api/main.py`
2. Add the API path prefix to `frontend/vite.config.ts` proxy if it's a new prefix
3. Mount `domains.py` LAST (move its mount if needed)

### Data Loading

- **Null normalization**: `''`, `'null'`, `'none'`, `'NA'` → NULL during load
- **Type casting**: Integer/float/date fields auto-cast with null coercion
- **Sales filtering**: Only `TYPE=1` rows loaded into `fact_sales_monthly`
- **Time dimension**: Auto-generated 2020–2035, not from a file
- **Forecast `model_id`**: Default `'external'` for source-system forecasts. `UNIQUE(forecast_ck, model_id)` prevents duplicates.
- **Execution-lag loading**: Dual-path insert with phase ordering (archive loaded BEFORE staging mutation). See spec `02-forecasting/03-backtest-framework.md` for details.
- **Customer demand**: source `site` column resolved to `location_id` via `dim_location.site_id` join; `posting_prd` (YYYYMM) converted to `startdate` (YYYY-MM-01); `demand_qty = MAX(0, demand_cases)`; `sales_qty = MAX(0, demand_cases - oos_cases)`; supports `--replace` (full) and `--month YYYY-MM` (single partition drop+reload)
- **New data sources MUST be part of the standard pipeline**: Every new input file must be added to `normalize-all` and `load-all` Makefile targets, and to `etl_config.yaml` `domain_order`. Never create standalone pipelines — `make load-all` must load everything, `make db-truncate-data` must truncate everything, `make fresh-load` must reload everything from scratch.

### Formulas

- **Accuracy**: `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- **Bias**: `(SUM(Forecast) / SUM(History)) - 1`
- **WAPE**: `SUM(|F-A|) / |SUM(A)|`

---

## Feature Integration Checklist

When adding a new feature end-to-end, follow these steps in order:

### 1. Database & Data Loading
- [ ] Create DDL migration in `sql/` (next sequence number)
- [ ] Add tables/MVs to `docs/RUNBOOK.md` cleanup section
- [ ] Add `TRUNCATE` / `REFRESH` to Makefile `db-truncate-data` target
- [ ] If new input data source: add normalize target to `normalize-all`, load target to `load-all`
- [ ] If new input data source: register `DomainSpec` in `common/core/domain_specs.py`
- [ ] If new input data source: add to `etl_config.yaml` `domain_order`

### 2. Backend
- [ ] Create router in correct `api/routers/{domain}/` subdirectory
- [ ] Use `get_conn()` pattern (not `Depends(_get_pool)` for inv_planning)
- [ ] Use `%s` placeholders in all SQL
- [ ] Add `app.include_router()` in `api/main.py` — **before** `domains.py`
- [ ] Add auth guard (`dependencies=[Depends(require_api_key)]`) on write endpoints
- [ ] Externalize config to `config/<name>.yaml`

### 3. Frontend
- [ ] Add query module in `frontend/src/api/queries/`
- [ ] Add Vite proxy entry in `frontend/vite.config.ts` for new API prefix
- [ ] Create component(s) in `frontend/src/tabs/` or `frontend/src/components/`
- [ ] Use `useThemeContext()` for theme — never pass as prop

### 4. Testing
- [ ] Backend test in `tests/api/test_<feature>.py`
- [ ] Frontend test in `src/tabs/__tests__/<Tab>.test.tsx`
- [ ] Run `make test-all` to verify

### 5. Documentation
- [ ] Update `docs/ARCHITECTURE.md`
- [ ] Update `docs/PLATFORM_GUIDE.md`
- [ ] Create/update spec in `docs/specs/<domain>/`
- [ ] Update `CLAUDE.md` if new critical rules apply

### 6. Verification
- [ ] Run `make audit-routers` to check route consistency
- [ ] Verify Vite proxy works (no "HTML instead of JSON" errors)
- [ ] Run `make test-all` one final time

---

## Troubleshooting Common Errors

### 422 Validation Error in Tests
**Cause**: Using `Depends(_get_pool)` in `inv_planning_*.py` routers. FastAPI inspects MagicMock signatures and fails validation.
**Fix**: Use `get_conn()` directly instead.

### "HTML instead of JSON" in Frontend
**Cause**: Missing Vite proxy entry for the API prefix.
**Fix**: Add the prefix to `frontend/vite.config.ts` proxy config. Run `make audit-routers` to check.

### Dimension Mismatch in SHAP
**Cause**: `ml_cluster` was stripped from `feature_cols` during SHAP computation.
**Fix**: Never strip `ml_cluster` — it is a hard feature that must always be present.

### Import Errors After Restructure
**Cause**: Missing backward-compat shim in `common/` root.
**Fix**: Ensure `common/__init__.py` or shim module re-exports from the correct subpackage.

### Tests Pass But API Fails
**Cause**: Mock row tuple column count doesn't match actual table DDL.
**Fix**: Verify mock tuples match the exact column count and order of the SQL query.

### MV Refresh Stale Data
**Cause**: Materialized views refreshed in wrong order (dependent MVs refresh before their sources).
**Fix**: Follow tiered refresh order — base aggregates first, then derived MVs. See `Makefile` `refresh-mvs-tiered`.

### Connection Pool Exhaustion
**Cause**: Too many concurrent requests with `max_size=10` (now increased to 20).
**Fix**: Set `POOL_MAX_SIZE` env var. Check for unclosed connections in scripts.

### Cold-Start DFUs Get No Forecast
**Cause**: DFU has fewer than `cold_start_min_months` (3) months of sales history and is skipped by the production forecast pipeline.
**Fix**: Check `cold_start_min_months` and `min_history_months` in `config/forecast_pipeline_config.yaml` under `production_forecast`. DFUs with 3-11 months of history are routed to `cold_start_model_id` (rolling_mean). DFUs with fewer than 3 months are skipped (absolute floor).

---

## Mandatory Testing Rules

**Every new feature, endpoint, component, hook, or utility MUST include tests. Every removed feature MUST have its tests removed.**

### What to test where:
1. New Python `common/` module → `tests/unit/test_<module>.py`
2. New API endpoint → `tests/api/test_<feature>.py` (httpx AsyncClient + ASGITransport)
3. New React component → `src/components/__tests__/<Component>.test.tsx`
4. New React hook → `src/hooks/__tests__/<hook>.test.ts`
5. New tab component → `src/tabs/__tests__/<Tab>.test.tsx`
6. New sidebar tab → E2E test in `frontend/e2e/tests/navigation.spec.ts`

### Test patterns:
- **Backend**: Mock DB via `make_pool()` factory in `tests/api/conftest.py`. No running server needed.
- **Frontend**: Wrap with `QueryClientProvider` from test-utils. Mock API layer with `vi.mock`.
- **E2E**: Semantic selectors (`getByRole`, `getByText`) — never CSS classes. Use `navigateToTab()` fixture.

### Run `make test-all` after every change.

---

## Documentation Update Rules

**When code is added, changed, or deleted, update these docs if affected:**

1. `docs/ARCHITECTURE.md` — architecture, tables, data flow
2. `docs/PLATFORM_GUIDE.md` — stack, datasets, features, quick start
3. `docs/specs/<domain>/<spec>.md` — create/update the relevant design spec
4. `docs/specs/01-foundation/01-infrastructure.md` — add to "Implemented Features" list
5. `CLAUDE.md` (this file) — if critical rules, entry points, or commands change
6. `docs/ENTERPRISE_ARCHITECTURE.md` — TOGAF-style enterprise architecture document. Update the relevant section(s) when:
   - **New technology adopted** → update Section 6.1 (Technology Portfolio) and Section 12.2 (Technology Radar)
   - **New API router or domain** → update Section 5.3 (Component Architecture) router table and Section 1.3 (Quantitative Scope)
   - **Schema change (new table, MV, or fact)** → update Section 4.2 (Logical Data Model) and Section 4.4 (Data Governance)
   - **New integration or external system** → update Section 7.4 (Integration Reference Architecture) and Section 6.4 (Network Topology)
   - **Security change (auth, RBAC, rate limiting)** → update Section 8 (Security Architecture)
   - **New ML model or AI agent** → update Section 7.3 (ML/AI Reference Architecture)
   - **Architecture decision** → add new ADR in Section 9 (Architecture Decision Records)
   - **New business capability** → update Section 3.1 (Business Capability Model)
   - **Infrastructure change (Docker, deployment)** → update Section 6.2 (Infrastructure Architecture) and Section 6.3 (Docker Compose Services)
   - **Gap closed from transition roadmap** → update Section 11 (Transition Architecture)
7. `docs/RUNBOOK.md` "Database Cleanup & Fresh Recreate" + `Makefile` cleanup targets — when adding new tables, MVs, or pipeline steps:
   - **New data table** → add `TRUNCATE TABLE <name> CASCADE;` to the correct FK group in RUNBOOK SQL + Makefile `db-truncate-data`
   - **New materialized view** → add `REFRESH MATERIALIZED VIEW <name>;` to the correct tier in RUNBOOK Step 5 + Makefile `refresh-mvs-tiered`
   - **New pipeline step** → add to the appropriate RUNBOOK step + Makefile `fresh-*` dependency chain
   - **New config/preserved table** → add to the "Preserved Tables" list (do NOT add a TRUNCATE)

**What requires updates**: New features, schema changes, new dependencies, new Make targets, pipeline changes, removals/renames.

**What does NOT**: Bug fixes without interface changes, minor internal refactors, typo fixes.

---

## Design Specs

Located in `docs/specs/` — 8 domains, 56 spec files. See [docs/specs/README.md](docs/specs/README.md) for the full index with reading order.

Domains: Foundation, Forecasting, Demand Intelligence, Inventory Planning, Operations, AI Platform, User Experience, Integration.

---

## Automatic Quality Workflow

These rules are ALWAYS active. Claude MUST follow them without explicit user request.

### After Writing/Editing Python Code
- Launch `python-reviewer` agent on the changed file(s) after completing a feature or fixing a bug
- Apply `python-patterns` skill conventions (type hints, EAFP, comprehensions)
- For API routers: also apply `api-design` and `backend-patterns` skills
- For SQL-heavy code: also apply `postgres-patterns` skill

### After Writing/Editing SQL or Schema Files
- Launch `database-reviewer` agent for any SQL/schema change
- Verify `%s` placeholders, explicit column lists, index coverage

### When Implementing New Features
- Launch `tdd-guide` agent — write tests FIRST, then implement
- Apply `tdd-workflow` skill (Red-Green-Refactor cycle)
- Every new endpoint must have a corresponding test in `tests/api/`
- Every new `common/` module must have a test in `tests/unit/`
- Every new React component must have a co-located test

### Before Suggesting a Commit
- Launch `code-reviewer` agent on all uncommitted changes
- Apply `security-review` skill (no secrets, no injection, no hardcoded keys)
- Apply `verification-loop` skill (build, types, lint, tests all pass)

### When Starting Complex Multi-File Changes
- Launch `planner` agent first to create implementation plan
- Wait for user confirmation before proceeding

### When Fixing Bugs
- Write a test that reproduces the bug FIRST
- Verify the test fails, then fix the bug, then verify the test passes
- Run the full affected test suite

### After Adding or Modifying Pipeline Scripts
- Run `make perf-script SCRIPT=<name>` to profile the script
- Review suggestions for N+1 queries, memory spikes, and unbatched inserts
- Use `profiled_section()` for major computation stages in new scripts

### Security Checks (Always Active)
- Never commit files matching: `.env`, `credentials.*`, `*secret*`, `*.key`
- Flag any hardcoded API keys, tokens, or passwords
- Verify SQL queries use parameterized queries, never string interpolation

### Agent Quick Reference
| Agent | Auto-trigger |
|---|---|
| `python-reviewer` | After Python changes |
| `code-reviewer` | Before commits, after features |
| `database-reviewer` | SQL/schema changes |
| `tdd-guide` | New features, bug fixes |
| `planner` | Complex multi-file work |

### Hooks (Configured in `.claude/settings.json`)
- **PostToolUse (Write/Edit)**: Auto-runs `ruff check` on Python files, anti-pattern checks on SQL files, auto-runs edited test files
- **PreToolUse (Bash)**: Blocks `git commit` if ruff lint or pytest fails

---

## Do Not

- Do not commit `__pycache__/`, `.pyc` files, or `.venv/`
- Do not modify `data/*.csv` files manually — they are generated by normalize scripts
- Do not touch the `archive/reference/` directory — it is archived code
