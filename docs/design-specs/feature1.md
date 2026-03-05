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
- **IPfeature5:** Replenishment Policy Management — configurable policy catalog and per-DFU policy assignments. New DDL: `sql/025_create_replenishment_policy.sql` — `dim_replenishment_policy` table (policy_id, policy_type CHECK IN continuous_rop/periodic_review/min_max/manual, service_level, review_cycle_days, use_eoq, use_safety_stock) and `fact_dfu_policy_assignment` table (UNIQUE item_no+loc, policy_id FK, assigned_by). Config: `config/replenishment_policy_config.yaml` — 4 default policies (A_continuous_v1, B_periodic_v1, C_min_max_v1, lumpy_manual_v1) plus auto_assign rules (variability_class=lumpy overrides ABC class). Script: `scripts/assign_replenishment_policies.py` (upsert policies + auto-assign DFUs, `--dry-run`, `--force-overwrite`). API: 5 new endpoints in `api/routers/inv_planning.py` — `GET/POST /inv-planning/policies`, `PUT /inv-planning/policies/{id}`, `GET /inv-planning/policy-assignments`, `POST /inv-planning/policy-assignments/assign`, `GET /inv-planning/policy-assignments/compliance`. Frontend: Policy Management panel in `InvPlanningTab.tsx` with policy type cards, SVG compliance ring gauge, auto-assign-all button, compliance table, edit modal. Make targets: `policy-schema`, `policy-assign`, `policy-all`. Tests: 18 unit (`test_replenishment_policy.py`), 13 API (`test_inv_planning_policy.py`), 6 frontend. Backend: 663 passed. Frontend: 243 passed.
- **IPfeature6:** Inventory Health Score Dashboard — composite 0–100 health score per item-location computed from 4 independent components (SS Coverage 25pts, DOS Target Adherence 25pts, Stockout Risk History 25pts, Forecast Accuracy 25pts). Tiers: ≥80=healthy, ≥60=monitor, ≥40=at_risk, <40=critical. Neutral score (62) used when upstream data (IPfeature3, ss_targets) not yet available. New DDL: `sql/026_create_inventory_health_score.sql` — stub `fact_safety_stock_targets` (CREATE TABLE IF NOT EXISTS for forward compatibility) + `mv_inventory_health_score` materialized view (CTE-based scoring with 6 indexes including partial critical index). Script: `scripts/refresh_health_scores.py` (REFRESH MATERIALIZED VIEW CONCURRENTLY). API: 3 new endpoints — `GET /inv-planning/health/summary` (tier counts, component avgs, score histogram), `GET /inv-planning/health/detail` (paginated + filtered table, 12 sortable columns), `GET /inv-planning/health/heatmap` (ABC × variability avg-score grid). Frontend: Portfolio Health landing section in `InvPlanningTab.tsx` — 4 clickable tier KPI cards (filter detail table on click), health distribution donut chart, component score progress bars, ABC×variability heatmap table with color tiers, paginated detail table with tier badge, Clear filter button. Make targets: `health-schema`, `health-refresh`, `health-all`. Tests: 42 unit (`test_health_score.py` — scoring functions + tier mapping + composite bounds), 12 API (`test_inv_planning_health.py`), 7 frontend. Backend: 717 passed. Frontend: 249 passed.
- **IPfeature7:** Exception Queue & Replenishment Recommendations — automated exception detection and recommended replenishment actions per DFU. Pure-function detection: `detect_exception_type(qty, ss, rop, dos, target_dos_max, avg_daily_sls)` returns (exception_type, severity); `compute_recommendation(...)` returns (order_qty, order_by_date, receipt_date) using formula `max(eoq, gap + eoq/2)` capped at `max_months × demand`. 6 exception types: stockout (critical), below_rop_critical (critical), below_ss (high/critical), below_rop (high), excess (medium/low), zero_velocity (low). Deduplication: skip if same item+loc+exception_type open within 7 days. DDL: `sql/027_create_replenishment_exceptions.sql` — `fact_replenishment_exceptions` table with 6 indexes (incl. partial on open+critical rows). Script: `scripts/generate_replenishment_exceptions.py` — reads `agg_inventory_monthly` + `fact_dfu_policy_assignment` + `dim_replenishment_policy` + `fact_safety_stock_targets` (stub fallback), writes exceptions to DB; supports `--dry-run`. API: 5 new endpoints in `api/routers/inv_planning.py` all using `get_conn()` directly — `GET /inv-planning/exceptions` (paginated, filterable by type/severity/status/item/loc), `GET /inv-planning/exceptions/summary`, `PUT /inv-planning/exceptions/{id}/acknowledge` (auth), `PUT /inv-planning/exceptions/{id}/status` (auth), `POST /inv-planning/exceptions/generate` (auth). Frontend: Exception Queue panel at top of `InvPlanningTab.tsx` — 4 KPI cards (Total Open, Critical, High, Rec. Order Value), exception type filter pills, severity filter pills, status toggle (all/open/acknowledged/ordered/resolved), item/loc filter inputs, exception table with inline action buttons (Acknowledge→Mark Ordered→Resolve workflow), row background coloring by severity (critical=red-50, high=amber-50, medium=yellow-50), Generate Exceptions button. Make targets: `exceptions-schema`, `exceptions-generate`, `exceptions-generate-dry`. Tests: 19 unit (`test_exception_generation.py` — detect_exception_type, compute_recommendation pure functions), 13 API (`test_inv_planning_exceptions.py`), 5 frontend. Backend: 749 passed. Frontend: 253 passed.
- **IPfeature8:** Fill Rate Analytics — order fill rate computation from inventory snapshot data. New DDL: `sql/028_create_fill_rate_monthly.sql` — `mv_fill_rate_monthly` materialized view aggregating fill rate metrics by item-location-month. Router: `api/routers/fill_rate.py` — 3 endpoints (`GET /fill-rate/summary`, `GET /fill-rate/trend`, `GET /fill-rate/detail`). Frontend: FillRatePanel section in `InvPlanningTab.tsx`. Tests: `tests/api/test_fill_rate.py`. Backend: 851 passed. Frontend: 258 passed.
- **IPfeature9:** Demand Sensing & Short-Horizon Signal Integration — short-horizon demand signal computation from recent sales velocity and inventory movement. New DDL: `sql/029_create_demand_signals.sql` — `fact_demand_signals` table. Script: `scripts/compute_demand_signals.py`. API: 3 endpoints in `api/routers/inv_planning.py` — `GET /inv-planning/demand-signals/summary`, `GET /inv-planning/demand-signals/list`, `GET /inv-planning/demand-signals/item`. Tests: `tests/api/test_inv_planning_demand_signals.py`, `tests/unit/test_demand_signals.py`.
- **IPfeature10:** Safety Stock Monte Carlo Simulation — probabilistic safety stock simulation using configurable number of iterations. New DDL: `sql/030_create_ss_simulation_results.sql` — `fact_ss_simulation_results` table. Script: `scripts/run_ss_simulation.py`. Config: `config/simulation_config.yaml` (n_simulations, random_seed). API: 3 endpoints — `POST /inv-planning/simulation/run`, `GET /inv-planning/simulation/results`, `GET /inv-planning/simulation/compare`, `GET /inv-planning/simulation/{id}/status`. Tests: `tests/api/test_inv_planning_simulation.py`.
- **IPfeature11:** ABC-XYZ Policy Matrix — combined ABC volume segmentation with XYZ demand variability classification into a 3×3 policy matrix. New DDL: `sql/031_add_xyz_classification.sql` — XYZ classification columns. Script: `scripts/classify_abc_xyz.py`. API: 3 endpoints in `api/routers/inv_planning.py` — `GET /inv-planning/abc-xyz/matrix`, `GET /inv-planning/abc-xyz/summary`, `GET /inv-planning/abc-xyz/detail`. Frontend: AbcXyzPanel in `InvPlanningTab.tsx`. Tests: `tests/api/test_inv_planning_abc_xyz.py`, `tests/unit/test_abc_xyz_classification.py`.
- **IPfeature12:** Supplier Performance Analytics — supplier delivery performance tracking from inventory receipt data. New DDL: `sql/032_create_supplier_performance.sql` — `mv_supplier_performance` materialized view. API: 3 endpoints in `api/routers/inv_planning.py` — `GET /inv-planning/supplier-performance/summary`, `GET /inv-planning/supplier-performance/detail`, `GET /inv-planning/supplier-performance/items`. Frontend: SupplierPanel in `InvPlanningTab.tsx`. Tests: `tests/api/test_inv_planning_supplier.py`.
- **IPfeature13:** Capital Investment Optimization — portfolio-level inventory investment planning with efficient frontier computation using configurable budget constraints. New DDL: `sql/033_create_investment_plan.sql` — `fact_inventory_investment_plan` + `fact_efficient_frontier` tables. Script: `scripts/compute_investment_plan.py`. API: 4 endpoints in `api/routers/inv_planning.py` — `GET /inv-planning/investment/efficient-frontier`, `GET /inv-planning/investment/summary`, `GET /inv-planning/investment/detail`, `POST /inv-planning/investment/plan`. Tests: `tests/api/test_inv_planning_investment.py`, `tests/unit/test_investment_plan.py`.
- **IPfeature14:** Intra-Month Stockout Detection — daily inventory scan to detect within-month stockout events before end-of-month snapshot. New DDL: `sql/034_create_intramonth_stockout.sql` — `mv_intramonth_stockout` materialized view. Script: `scripts/refresh_intramonth_stockout.py`. API: 3 endpoints in `api/routers/inv_planning.py` — `GET /inv-planning/intramonth-stockouts/summary`, `GET /inv-planning/intramonth-stockouts/detail`, `GET /inv-planning/intramonth-stockouts/daily`. Frontend: IntramonthPanel in `InvPlanningTab.tsx`. Tests: `tests/api/test_inv_planning_intramonth.py`.
- **IPfeature15:** Unified Control Tower / Command Center — single-pane-of-glass operational dashboard aggregating KPIs, alerts, critical items, and trends across all inventory planning dimensions. New DDL: `sql/035_create_control_tower_kpis.sql` — `mv_control_tower_kpis` materialized view. Router: `api/routers/control_tower.py` — 4 endpoints (`GET /control-tower/kpis`, `GET /control-tower/alerts`, `GET /control-tower/top-critical`, `GET /control-tower/trend`). Frontend: `frontend/src/tabs/ControlTowerTab.tsx` — new dedicated tab. Router registered in `api/main.py`. Sidebar nav item and Vite proxy entry (`/control-tower`) added. Tests: `tests/api/test_control_tower.py` (backend), `src/tabs/__tests__/ControlTowerTab.test.tsx` (frontend). Backend: 851 passed. Frontend: 258 passed.

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
