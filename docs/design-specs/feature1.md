# Feature 1: Infrastructure & Platform Setup

## Objective
Build a scalable forecasting platform for:
- traditional statistical models and ML models
- fast UI analytics across item, location, and customer attributes
- 500M+ records with reliable performance and governance

## Recommended Stack (Simple and Right-Sized)
- `Apache Iceberg` on local object storage (`MinIO`) as the single source of truth
- `Apache Spark` for ingestion, data quality, feature prep, batch training, and batch scoring
- `Trino` for low-latency SQL analytics powering the UI
- `MLflow` for experiment tracking, model registry, and run comparison
- `Postgres` (via `pgvector/pgvector:pg16`) for app metadata, workflow, and vector embeddings for NL queries
- `FastAPI` backend and `React/Vite` frontend
- `Pydantic` for canonical data structures, schema contracts, and validation
- `uv` as the Python package/environment manager (`pyproject.toml` + `uv.lock`)

## Why This Works
- `Iceberg` gives ACID tables, schema evolution, partition evolution, and time travel for large datasets.
- `Spark` handles heavy ETL and model pipelines at scale.
- `Trino` queries Iceberg tables interactively, enabling fast slice-and-dice in UI.
- `MLflow` gives model lineage, reproducibility, and governance for both traditional and ML models.
- `Pydantic` enforces typed, validated data contracts between ingestion, API, and modeling layers.
- Clear separation: Spark writes, Trino reads, Iceberg stores.

## High-Level Architecture
1. Land raw files (`CSV/Parquet`) into object storage.
2. Spark validates and standardizes data into Iceberg bronze/silver/gold tables.
3. Spark runs training and forecasting jobs and writes forecast outputs to Iceberg.
4. Trino serves UI queries from curated Iceberg gold tables.
5. App writes business actions (override, approval, comments, audit) to Postgres.
6. ML training/scoring jobs log params, metrics, and artifacts to MLflow.

## Data Model (Minimum)
- Grain: `item_id`, `location_id`, `customer_id` (nullable), `date`
- Facts:
  - `demand_actuals`
  - `demand_forecast`
  - `forecast_accuracy`
- Dimensions:
  - `dim_item`
  - `dim_location`
  - `dim_customer`
  - `dim_calendar`

## Performance Design for 500M+ Rows
- Partition Iceberg fact tables by date (weekly/monthly) and optionally region.
- Sort/cluster by high-filter columns (`location_id`, `item_id`) during Spark writes.
- Keep Trino queries on curated "gold" tables, not raw bronze data.
- Create pre-aggregated tables for UI:
  - `agg_loc_week`
  - `agg_loc_item_week`
  - `agg_loc_customer_week`
  - `exceptions_topn`
- Enforce bounded queries in API:
  - always require time window
  - default top-N for exception screens
  - server-side pagination only

## Forecasting Approach
- Traditional models: seasonal naive, ETS/ARIMA (where stable and interpretable).
- ML models: LightGBM/XGBoost with lag, rolling stats, calendar, and promo features.
- Champion/challenger selection by segment using `WMAPE`, bias, and service impact.
- Version each run (`run_id`, model version, training window) in Iceberg + MLflow + Postgres metadata.

## Simple MVP Scope
1. Ingest `actuals` and master dimensions.
2. Build Iceberg silver/gold tables.
3. Train baseline + one ML model in Spark.
4. Write forecasts + metrics to Iceberg.
5. Expose Trino-backed APIs for portfolio, exceptions, and item drilldown.
6. Add overrides and approvals in Postgres with full audit.

## Implemented Features (MVP)
- **Feature 1:** Infrastructure setup — Docker Compose, MinIO, Spark, Trino, MLflow, Postgres, FastAPI, React
- **Feature 2:** Internal data architecture & data contracts — canonical keys, lakehouse standards, SCD2, ERD
- **Feature 3:** Dimension tables — Item, Location, Customer, Time, DFU
- **Feature 4:** Fact tables — Sales (`fact_sales_monthly`), External Forecast (`fact_external_forecast_monthly`)
- **Feature 5:** Forecast accuracy KPIs — Accuracy %, WAPE, MAPE, Bias, window selector, trend charts
- **Feature 6:** Multi-model forecast support — `model_id` column, model selector, per-model analytics
- **Feature 7:** DFU clustering framework — KMeans, feature engineering, automated labeling, MLflow
- **Feature 8:** Backtesting framework — expanding window timeframes (A-J), multi-model, lag 0-4 archive
- **Feature 9:** LGBM backtesting implementation — global + per-cluster models, lag features, rolling stats
- **Feature 10:** Multi-dimensional accuracy slicing — accuracy by cluster/supplier/lag/model, materialized views, UI panel
- **Feature 11:** Chatbot / natural language queries — OpenAI GPT-4o + pgvector, NL-to-SQL with safe execution
- **Feature 12:** CatBoost backtesting implementation — global + per-cluster models, native categorical support, same feature engineering as LGBM
- **Feature 13:** XGBoost backtesting implementation — global + per-cluster models, histogram-based with native categorical support
- **Feature 14:** Transfer learning backtest strategy — global base model → per-cluster fine-tune via warm-start for all three frameworks
- **Feature 15:** Champion model selection — per-DFU per-month best-of-models via 5 configurable strategies (expanding, rolling, decay, ensemble, meta_learner), exec-lag-aware strict causality (`shift(exec_lag+1)` per DFU-model group), fallback model for warm-up gaps (default: lgbm_cluster), strategy registry in `common/champion_strategies.py`, ceiling (oracle) model for theoretical upper bound, gap-to-ceiling analysis, meta-learner training on ceiling labels, simulation framework comparing all strategies, UI-editable competition config, FVA analysis
- **Feature 16:** Data Explorer performance & UX — type-aware SQL filtering, GIN trigram indexes, capped COUNT, column-level typeahead suggestions, chemistry-themed loading overlay, debounce stability fix
- **Feature 17:** DFU Analysis tab — unified sales vs multi-model forecast overlay chart, 3 analysis modes (item@location, all items@location, item@all locations), per-model KPI cards, toggleable measures
- **Feature 18:** Market intelligence — AI-powered market briefings combining Google web search + GPT-4o narrative synthesis for item + location pairs, with demographic context and demand insights
- **Feature 19:** PatchTST backtesting implementation — Transformer-based patched time series model with Apple MPS GPU acceleration, global/per-cluster/transfer strategies
- **Feature 20:** DeepAR backtesting implementation — LSTM-based probabilistic model (Gaussian likelihood), global/per-cluster/transfer strategies
- **Feature 21:** Prophet backtesting implementation — per-DFU individual time series models with Fourier seasonality, global/per-cluster/pooled strategies
- **Feature 22:** UI theming — dark mode and midnight theme support via CSS variable-based theming with shadcn/ui
- **Feature 23:** Backtest model cleanup utility — CLI tool to selectively remove model predictions from Postgres and refresh materialized views, with list/dry-run/bulk modes
- **Feature 24:** StatsForecast backtesting — vectorized AutoARIMA + AutoETS (~100x faster than Prophet), global/per-cluster/pooled strategies, Numba JIT compiled
- **Feature 25:** NeuralProphet backtesting — PyTorch-based Prophet successor with Apple MPS GPU acceleration, global/per-cluster/pooled strategies
- **Feature 26:** Postgres vs Trino/Iceberg benchmarking — API endpoint to run identical queries against both backends with statistical latency comparison and winner determination
- **Feature 28:** UI Architecture & Performance Refactoring — monolith decomposition (2,700→230 lines), TanStack Query caching, lazy-loaded tabs, error boundaries, virtualized data grid, keyboard shortcuts, ECharts, Vitest testing
- **Feature 31:** Comprehensive Testing Strategy — full-stack testing spec covering backend (pytest), frontend (Vitest/RTL), integration, performance, security, and mandatory testing requirements for all new development
- **Feature 34:** Inventory Planning Module — 14-month inventory snapshot pipeline (190M+ rows), `fact_inventory_snapshot` table with B-tree + GIN indexes, rebuilt `agg_inventory_monthly` with daily sales derivation (LAG CTE), EOM snapshots, proper monthly sales. `/inventory/kpis` two-query pattern (latest snapshot PIT totals + trailing-month DOS/WOC/Turns/LT Coverage). 5-metric trend chart, 7 severity-coded KPI cards, position table, item detail drill-down
- **Feature 29:** What-If / Scenario UI for Clustering — ClustersTab panel to simulate alternative KMeans parameters, view result distribution charts, and promote winning scenario to production `ml_cluster`. API router implemented (`/clustering/defaults`, `/clustering/scenario`, `/clustering/scenario/{id}/promote`) and mounted via `include_router`.
- **Feature 30:** DFU Seasonality Detection & Profile Assignment — automated pipeline to compute seasonality metrics per DFU (strength, profile label, peak/trough month, peak-to-trough ratio, yearly flag) from sales history and write to `dim_dfu`. Scripts: `detect_seasonality.py`, `update_seasonality_profiles.py`. Config: `config/seasonality_config.yaml`. DDL: `sql/015_add_seasonality_columns.sql`. Make targets: `seasonality-detect`, `seasonality-update`, `seasonality-all`. Columns added to `DFU_SPEC`.
- **Feature 35:** Configurable Multi-Theme / Motif System — 5 visual motifs (Periodic Table, Wine & Spirits, Space, Formula 1, Zen Garden) with distinct tiles, icons, and loading animations. `useMotifTheme` hook with localStorage + `?motif=` URL parameter persistence. `MotifContext` app-wide provider. `MotifSettingsPanel` opened by Ctrl+M shortcut. Motif is independent of product theme (wine/general/obsidian) and light/dark color mode.
- **Feature 36:** Product-Grade UI Overhaul — Collapsible sidebar navigation (9 nav items, 5 sections), global filter bar (brand/category/market/channel), dashboard overview landing page (KPI cards with sparklines, alert panel, heatmap, top movers, forecast trend chart), 3 product themes (Wine & Spirits, General, Obsidian) with CSS variable palettes and light/dark modes, `mv_top_movers` materialized view, 5 new API endpoints (distinct, kpis, alerts, top-movers, heatmap)
- **Feature 37:** Inventory Planning Backtesting — Bridges forecast accuracy with inventory outcomes via `mv_inventory_forecast_monthly` materialized view (joins `agg_inventory_monthly` + `fact_external_forecast_monthly` + `dim_dfu`). 4 API endpoints: summary (per-model stockout/excess/service level/WAPE), trend (monthly by model), root-cause (bias direction attribution), detail (paginated DFU-level events). New Inv. Backtest UI tab with KPI cards, model comparison chart, root cause stacked bars, trend lines, and sortable event table. Configurable excess DOS threshold.
- **Feature 38:** Clustering What-If Scenario Enhancements — Background execution with async POST (HTTP 202) and status polling, runtime estimation endpoint based on DFU count and K range, cross-tab scenario notification context with dashboard alert integration, enhanced charts (elbow with optimal K marker, silhouette with quality zone thresholds, feature importance bar chart, cluster size pie chart, gap statistic chart).
- **Feature 39:** Job Scheduler/Monitor with APScheduler — Production-grade job execution powered by APScheduler 3.11 (`BackgroundScheduler` + `ThreadPoolExecutor`). Persistent `job_history` + `job_schedule` tables, `JobManager` singleton with per-group concurrency control, 7 job types across 4 groups. REST API (12 endpoints) including cron/interval scheduling, pipeline chaining, dashboard stats, and retry logic. Professional automation dashboard UI with KPI cards, grouped job type cards, live active job monitoring, schedule dialog, recurring schedules section, expandable history. Foundation for agentic AI automation.
- **Feature 41:** Hyperparameter Tuning for Tree-Based Cluster Models — Bayesian hyperparameter optimisation via Optuna (TPESampler + MedianPruner) for LGBM, CatBoost, and XGBoost cluster models. Walk-forward CV with causal masking. WAPE stabilised with denominator floor. `n_estimators` determined by early stopping (excluded from search space). Outputs `data/tuning/best_params_<model>.json`. Backtest scripts accept `--params-file` flag. Model-scoped output dirs (`data/backtest/<model_id>/`) fix PL-001 CSV overwrite issue. **PL-002 fix:** `tune_for_timeframe()` in `common/tuning.py` performs per-timeframe causal tuning — filters the feature matrix to `months <= cutoff_date` before running an inline Optuna study (20 trials, 3 folds), eliminating temporal data leakage from backtest accuracy metrics. `TRAIN_FOLD_FNS` registry exposes fold training functions for both global and inline tuning. `run_tree_backtest()` accepts `inline_tuner_fn` parameter. All three backtest scripts support `--tune-inline` flag (mutually exclusive with `--params-file`). Make targets: `tune-lgbm`, `tune-catboost`, `tune-xgboost`, `tune-all`, `backtest-lgbm-cluster-tuned`, `backtest-catboost-cluster-tuned`, `backtest-xgboost-cluster-tuned`, `backtest-load MODEL=<id>`, `backtest-load-all`. New module `common/tuning.py`, config `config/hyperparameter_tuning.yaml` (with `inline_n_trials`, `inline_n_splits`). 39 unit tests in `tests/unit/test_tuning.py`.
- **Feature 42:** SHAP-Based Per-Timeframe Feature Selection for Tree-Based Backtests — per-timeframe automatic feature selection using SHAP values for LGBM, CatBoost, and XGBoost. For each expanding-window timeframe: train initial model on all features → compute SHAP → select features covering 95% cumulative SHAP mass → retrain final model on selected features. CatBoost uses native `get_feature_importance(type="ShapValues")`; LGBM/XGBoost use `shap.TreeExplainer`. For per-cluster/transfer strategies, SHAP is pooled across clusters weighted by size; `ml_cluster` is excluded from the effective feature set. Outputs `data/backtest/<model_id>/shap/shap_timeframe_XX.csv` (per-timeframe) + `shap_summary.csv` (cross-timeframe). 4 REST API endpoints (`GET /forecast/shap/models`, `/summary`, `/timeframes`, `/timeframe/{idx}`) served from CSVs — no DB queries. Frontend: collapsible "Feature Importance (SHAP)" card in Accuracy tab with model/timeframe selectors and indigo=selected / gray=dropped bar chart. CLI flags: `--shap-select`, `--shap-top-n`, `--shap-threshold`, `--shap-sample-size`; composable with `--tune-inline` and `--params-file`. Make targets: `backtest-lgbm-shap`, `backtest-catboost-shap`, `backtest-xgboost-shap`. New module `common/shap_selector.py`, router `api/routers/shap.py`, types `frontend/src/types/shap.ts`. 30 tests (22 unit + 8 API). Dependency: `shap>=0.43.0`.
- **Feature 43:** Recursive Multi-Step Forecasting for Tree-Based Backtests — adds `--recursive` CLI flag to LGBM, CatBoost, and XGBoost backtest scripts. In recursive mode, each month in the prediction window is forecast one step at a time, and the model's own prediction for month T is fed back as `qty_lag_1` (and higher lags) for month T+1 via `update_grid_with_predictions()`. Solves the lag_1=0 problem in direct multi-output mode where masked future sales produce zero lag features for months 2+ of the prediction window. New functions: `update_grid_with_predictions()` in `common/feature_engineering.py`, `_fill_predict_nans()` and `_predict_single_month()` in `common/backtest_framework.py`. `run_tree_backtest()` accepts new `recursive: bool = False` parameter. Fully composable with `--shap-select` (SHAP retrain updates inference model and first-month preds), `--tune-inline` (PL-002), and `--params-file`. `"recursive": true` written to `backtest_metadata.json` for traceability. No API/frontend/DB changes — compute-side only. 9 Makefile targets: `backtest-{lgbm,catboost,xgboost}-{recursive,cluster-recursive,transfer-recursive}`. Backend tests: 13 unit tests (`test_backtest_recursive.py`) + 6 new tests in `TestUpdateGridWithPredictions` (`test_feature_engineering.py`). Backend test count: 514 passed.
- **Feature 44:** Algorithm Configuration & Simplification — consolidated all backtest configuration into a single declarative YAML file (`config/algorithm_config.yaml`). Eliminated Prophet, StatsForecast, NeuralProphet, PatchTST, and DeepAR backtest scripts. Removed global and transfer strategies from LGBM, CatBoost, and XGBoost — only per-cluster strategy remains. All CLI flags (`--recursive`, `--shap-select`, `--tune-inline`, `--params-file`, SHAP thresholds, hyperparameter defaults) are now config keys under algorithm sections (`lgbm`, `catboost`, `xgboost`). Simplified `run_tree_backtest()`: removed `train_fn_global`, `train_fn_transfer`, `transfer_kwargs` params; `_predict_single_month()` no longer takes `cluster_strategy`. Makefile reduced to 4 targets: `backtest-lgbm`, `backtest-catboost`, `backtest-xgboost`, `backtest-all`. Backend test count: 512 passed.
- **IPfeature4:** EOQ & Cycle Stock Calculator — per-item EOQ computation from `agg_inventory_monthly` using configurable ordering cost, holding cost %, and MOQ. Functions: `compute_eoq()` (Wilson formula), `compute_effective_eoq()` (MOQ + max-months-supply cap), `compute_eoq_metrics()` (cycle stock, annual ordering cost, annual holding cost, total cost), `sensitivity_curve()` (cost vs order quantity). DDL: `sql/024_create_eoq_targets.sql` (`fact_eoq_targets` table). Config: `config/eoq_config.yaml` (ordering_cost: 50, holding_cost_pct: 0.25, moq: 1, max_eoq_months_supply: 6). Script: `scripts/compute_eoq.py`. API router: `api/routers/inv_planning.py` — 3 new endpoints (`GET /inv-planning/eoq/summary`, `GET /inv-planning/eoq/detail`, `GET /inv-planning/eoq/sensitivity`). New frontend tab: `InvPlanningTab.tsx` with KPI cards, EOQ sensitivity chart, and paginated detail table. `AppSidebar.tsx` updated with "Inv. Planning" nav item (12 items total). Make targets: `eoq-schema`, `eoq-compute`, `eoq-all`. Tests: 23 backend unit tests (`test_eoq.py`), 10 API tests (`test_inv_planning_eoq.py`), 6 frontend tests (`InvPlanningTab.test.tsx`). Backend test count: 630 passed. Frontend test count: 238 passed.

## Deployment Notes
- Run everything on a single MacBook using Docker Compose (no cloud services):
  - Iceberg catalog + MinIO + Spark + Trino + MLflow + Postgres + API + UI
- Move to Kubernetes only when concurrency and SLA demand it.
- Standardize Python workflows with `uv`:
  - `uv venv`
  - `uv sync`
  - `uv run <command>`

## Local-Only Guardrails (MacBook)
- Use `MinIO` only (do not configure `S3` endpoints).
- Keep all data/artifacts local volumes on the MacBook.
- Disable external callbacks/webhooks from MLflow and app services.
- Restrict network egress from containers if strict isolation is required.

## Critical Standardization Rules
1. Canonical naming + grain:
   - enforce standard keys (`item_sk`, `location_sk`, `customer_sk`) and require explicit grain on every fact table.
2. Versioned data contracts with `Pydantic`:
   - use typed, versioned schemas at API and pipeline boundaries.
3. Quality gates before publish:
   - block publish when null checks, key uniqueness, referential integrity, or row-count drift checks fail.
4. Forecast lineage + governance:
   - require `scenario_id`, `algorithm_id`, `model_version`, `run_id`, and `planning_grain` on every forecast record; overrides require reason + approval.
5. Single config standard:
   - centralize typed, environment-driven config with `local/dev/prod` profiles; avoid hardcoded paths/endpoints.

## Final Recommendation
For your requirement, use `Iceberg + Spark + Trino + MLflow` as the core platform, fully local on your MacBook.
This gives the best balance of scale, speed, model governance, and simplicity for 500M+ record demand forecasting with both ML and traditional methods.

---

## Implementation Notes

### FastAPI Backend Configuration
- Middleware: `GZipMiddleware` (starlette), `CORSMiddleware` (fastapi)
- Connection pool: `psycopg_pool.ConnectionPool` (min_size=2, max_size=10)
- Central schema registry: `DomainSpec` frozen dataclass in `common/domain_specs.py` (not Pydantic models)
- Optional API key auth via `api/auth.py` (`require_api_key` dependency; disabled when `API_KEY` env var unset)

### Docker Compose Services (6 services)
| Service | Image | Ports |
|---------|-------|-------|
| minio | minio/minio:latest | 9200, 9201 |
| postgres | pgvector/pgvector:pg16 | 5440 |
| mlflow | ghcr.io/mlflow/mlflow:v2.16.2 | 5003 |
| iceberg-rest | tabulario/iceberg-rest:1.6.0 | 8381 |
| spark | spark:3.5.7-java17-python3 | 7277, 8280 |
| trino | trinodb/trino:451 | 8282 |

### Postgres Tuning
- `shared_buffers=512MB`, `work_mem=64MB`, `effective_cache_size=1536MB`
- `max_connections=50`, `checkpoint_completion_target=0.9`, `random_page_cost=1.1`

### Modular Router Architecture
10 router modules in `api/routers/`: domains, clusters, accuracy, analysis, benchmark, chat, competition, intel, jobs. Mounted via `app.include_router()` in `main.py`.


---

## Examples

### Example: Start all infrastructure services

```bash
make up       # Start 7 Docker services
make api      # FastAPI on :8000
make ui       # React on :5173
curl -s http://localhost:8000/health | jq .
# {"status": "ok", "db": "connected"}
```

### Example: End-to-end data pipeline

```bash
make normalize-sales   # datafiles/dfu_lvl2_hist.txt → data/dfu_lvl2_hist_clean.csv
make load-sales        # clean CSV → fact_sales_monthly
make check-db
# fact_sales_monthly   | 2,847,362 rows
# agg_sales_monthly    |    94,221 rows
```

### Example: Query actuals vs forecast in PostgreSQL

```sql
SELECT s.startdate, s.qty_shipped AS actual, f.basefcst_pref AS forecast,
       ABS(f.basefcst_pref - s.qty_shipped) AS abs_error
FROM fact_sales_monthly s
JOIN fact_external_forecast_monthly f
     ON s.dmdunit=f.dmdunit AND s.loc=f.loc AND s.startdate=f.startdate
WHERE s.dmdunit='100320' AND s.loc='1401-BULK'
  AND s.type=1 AND f.model_id='external' AND f.lag=2
ORDER BY s.startdate DESC LIMIT 3;
-- 2026-01-01 | 788 | 820 | 32
-- 2025-12-01 | 910 | 875 | 35
-- 2025-11-01 | 842 | 891 | 49
```
