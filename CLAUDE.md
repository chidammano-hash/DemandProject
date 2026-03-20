# CLAUDE.md — Supply Chain Command Center

## Project Overview

**Supply Chain Command Center** is a unified supply chain planning and execution platform. It ingests sales, forecast, and inventory data, stores it in PostgreSQL, and serves a React UI for interactive analytics.

**All code lives at the project root.** Backend (`api/`, `common/`, `scripts/`), frontend (`frontend/`), config (`config/`), SQL (`sql/`), and tests (`tests/`) are top-level directories.

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

---

## Key Entry Points

| Layer | File | Purpose |
|---|---|---|
| **API** | `api/main.py` | FastAPI app — mounts 56 routers, no inline routes |
| **API** | `api/routers/` | 58 router files (56 mounted; `domains.py` mounted last — catch-all) |
| **API** | `api/core.py` | Shared pool, OpenAI client, SQL helpers |
| **Frontend** | `frontend/src/App.tsx` | React shell — sidebar, lazy-loaded tabs |
| **Frontend** | `frontend/src/api/queries/` | All API fetch functions (24 domain modules) |
| **Frontend** | `frontend/src/tabs/` | 21 tab components + sub-panel directories |
| **Common** | `common/domain_specs.py` | Central config: all 8 datasets with columns, types, keys |
| **Common** | `common/backtest_framework.py` | Shared backtest orchestrator (`run_tree_backtest()`) |
| **Common** | `common/sql_helpers.py` | Shared SQL utilities: `qident()`, `typed_expr()`, `business_key_expr()`, `_elapsed()`, constants (`IQR_OUTLIER_MULTIPLIER`, `LEAD_TIME_MAX_DAYS`, `HASH_CHUNK_SIZE`, `EXTERNAL_MODEL_ID`, `MV_REFRESH_ARCHIVE`) |
| **Common** | `common/db.py` | Canonical DB connection params |
| **Common** | `common/utils.py` | `_ts()`, `load_config()`, `reset_config()` |
| **Common** | `common/planning_date.py` | `get_planning_date()` — replaces `date.today()` everywhere |
| **Config** | `config/algorithm_config.yaml` | Per-algorithm backtest options (cluster_strategy, SHAP, recursive, etc.) |
| **Config** | `config/` | All YAML configs — one per module/pipeline |
| **SQL** | `sql/` | All DDL: tables, indexes, materialized views |
| **Scripts** | `scripts/` | ETL, ML pipelines, computation scripts |

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
make normalize-all     # CSV → clean CSV (all 8 datasets)
make load-all          # Clean CSV → Postgres + refresh views
make load-forecast-replace  # Reload external forecast only

# Run Services
make api               # FastAPI on :8000
make ui-init           # Install npm deps
make ui                # React dev server on :5173

# ML Pipelines
make cluster-all       # Full clustering pipeline
make backtest-all      # LGBM + CatBoost + XGBoost backtests
make backtest-load-all # Load all backtest predictions into Postgres
make champion-all      # Meta-learner + simulate + champion select
make tune-all          # Bayesian hyperparameter tuning (all models)
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
```

See `Makefile` for the full list (100+ targets including one-time schema setup, inv-planning pipelines, cleanup utilities).

---

## Architecture

### Domain-Driven Generic Design

All datasets extend a single `DomainSpec` dataclass in `common/domain_specs.py`. Scripts and API endpoints are generic — they operate on any domain via `--dataset <name>` or `/domains/{domain}/*`.

**8 Domains:** item, location, customer, time, dfu (dimensions); sales, forecast (facts); inventory (dedicated pipeline).

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
- `fact_inventory_snapshot`: grain = item_no + loc + snapshot_date (~190M rows)
- `fact_production_forecast`: grain = item_no + loc + plan_version + month

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
- **Column names in fact tables**: `fact_sales_monthly` and `fact_external_forecast_monthly` use `dmdunit` (not `item_no`). Forecast qty column is `basefcst_pref` (not `qty`).
- **`domains.py` mounted last** in `main.py` — it has catch-all `{domain}` path params that would shadow other routes.
- **Shared test pool factory**: Import `from tests.api.conftest import make_pool as _make_pool`. For multi-fetchall endpoints use `cursor.fetchall.side_effect = [list1, list2]`; for single-call use `cursor.fetchall.return_value`.
- **API test pattern**: Use inline `httpx.AsyncClient(transport=ASGITransport(app))` with `patch("api.core._get_pool")`.

### Frontend Patterns

- **Vite proxy is CRITICAL**: `frontend/vite.config.ts` proxies API path prefixes to `:8000`. When adding a new API path prefix, you MUST add a proxy entry or the frontend gets HTML instead of JSON. Current prefixes: `/domains`, `/jobs`, `/clustering`, `/forecast`, `/inventory`, `/dashboard`, `/health`, `/chat`, `/dfu`, `/competition`, `/bench`, `/market-intelligence`, `/inv-planning`, `/fill-rate`, `/control-tower`, `/ai-planner`, `/storyboard`.
- **Theme context, not props**: Use `useThemeContext()` or `useChartColors()` — never pass `theme` as a prop from `App.tsx`.
- **Test wrappers**: Wrap components with `TestQueryWrapper` from `src/tabs/__tests__/test-utils.tsx`. Mock API with `vi.mock("../api/queries")`. Mock `echarts-for-react` for chart tests. Mock `@tanstack/react-virtual` for virtualized row tests.

### Code Patterns

- **All config in YAML**: Every module externalizes params into `config/<name>.yaml`. No magic numbers in scripts. Load via `load_config(name)` from `common/utils.py`.
- **DB params**: All scripts use `from common.db import get_db_params` — no inline connection helpers.
- **Planning date**: All date-sensitive code uses `get_planning_date()` from `common/planning_date.py`, not `date.today()`. Config: `config/planning_config.yaml`. Env overrides: `PLANNING_DATE` or `USE_SYSTEM_DATE`.
- **Timestamp helper**: Import `from common.utils import _ts` — no per-file `_ts()` definitions.
- **`ml_cluster` is always a hard feature** — never stripped from `feature_cols` in per_cluster or global backtest mode.
- **Stub table pattern**: When a materialized view depends on a future feature's table, create it with `CREATE TABLE IF NOT EXISTS` and minimum columns. LEFT JOIN produces NULL → neutral scores until real data flows.

### Data Loading

- **Null normalization**: `''`, `'null'`, `'none'`, `'NA'` → NULL during load
- **Type casting**: Integer/float/date fields auto-cast with null coercion
- **Sales filtering**: Only `TYPE=1` rows loaded into `fact_sales_monthly`
- **Time dimension**: Auto-generated 2020–2035, not from a file
- **Forecast `model_id`**: Default `'external'` for source-system forecasts. `UNIQUE(forecast_ck, model_id)` prevents duplicates.
- **Execution-lag loading**: Dual-path insert with phase ordering (archive loaded BEFORE staging mutation). See spec `02-forecasting/03-backtest-framework.md` for details.

### Formulas

- **Accuracy**: `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- **Bias**: `(SUM(Forecast) / SUM(History)) - 1`
- **WAPE**: `SUM(|F-A|) / |SUM(A)|`

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
2. `docs/BACKEND_README.md` — stack, datasets, quick start
3. `docs/specs/<domain>/<spec>.md` — create/update the relevant design spec
4. `docs/specs/01-foundation/01-infrastructure.md` — add to "Implemented Features" list
5. `CLAUDE.md` (this file) — if critical rules, entry points, or commands change

**What requires updates**: New features, schema changes, new dependencies, new Make targets, pipeline changes, removals/renames.

**What does NOT**: Bug fixes without interface changes, minor internal refactors, typo fixes.

---

## Design Specs

Located in `docs/specs/` — 8 domains, 52 spec files. See [docs/specs/README.md](docs/specs/README.md) for the full index with reading order.

Domains: Foundation, Forecasting, Demand Intelligence, Inventory Planning, Operations, AI Platform, User Experience, Integration.

---

## Do Not

- Do not commit `__pycache__/`, `.pyc` files, or `.venv/`
- Do not modify `data/*.csv` files manually — they are generated by normalize scripts
- Do not touch the `archive/reference/` directory — it is archived code
