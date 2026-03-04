# Unified Architecture

## Goal
Reduce dataset-by-dataset duplication and provide a reusable path for adding new dimensions and facts.

## Current pattern
1. Define dataset spec in `common/domain_specs.py`
2. Add DDL in `sql/`
3. Reuse generic scripts:
   - `normalize_dataset_csv.py`
   - `load_dataset_postgres.py`
   - `spark_dataset_to_iceberg.py`
4. Reuse generic API query paths in `api/main.py`:
   - `/domains/{domain}`
   - `/domains/{domain}/page`
   - `/domains/{domain}/meta`
   - `/domains/{domain}/analytics`
5. Reuse one shared React UI app (`frontend/src/App.tsx`)

## Current dimensions
1. `item`
2. `location`
3. `customer`
4. `time`
5. `dfu`

## Current facts
1. `sales` (`fact_sales_monthly`) from `dfu_lvl2_hist.txt` filtered to `TYPE=1`
   - grain: `dmdunit` + `dmdgroup` + `loc` + `startdate` (monthly) + `type`
   - key: `sales_ck` with `_` separator
   - rule: `startdate` must be month-start (`YYYY-MM-01`)
2. `forecast` (`fact_external_forecast_monthly`) from `dfu_stat_fcst.txt`
   - grain: `dmdunit` + `dmdgroup` + `loc` + `fcstdate` + `startdate` + `model_id`
   - key: `forecast_ck` with `_` separator; uniqueness: `UNIQUE(forecast_ck, model_id)`
   - rule: `fcstdate` and `startdate` must be month-start
   - rule: `lag = month_diff(startdate, fcstdate)` and only lags `0..4`
   - rule: `model_id` defaults to `'external'` when absent from source
   - loading: dual-path insert with **phase ordering** — archive loads FIRST from untouched staging (each row's original `lag` preserved as `execution_lag`), THEN staging is mutated from `dim_dfu`, THEN main table receives **execution-lag rows only** via `WHERE lag = execution_lag` (matched DFUs get dim_dfu value, unmatched default to 0)
   - `--replace` flag: deletes only `model_id='external'` rows instead of truncating (preserves backtest/champion/ceiling data)
   - `--skip-archive` flag: skips archive load entirely (Phase 3b) — only loads execution-lag row into main table for faster external forecast reloads
3. `inventory` (`fact_inventory_snapshot`) from 14 monthly CSV files (`Inventory_Snapshot_YYYY_MM.csv`)
   - grain: `item_no` + `loc` + `snapshot_date` (monthly)
   - key: `inventory_ck` with `_` separator
   - measures: `qty_on_hand`, `qty_on_hand_on_order`, `qty_on_order` (derived), `mtd_sales`, `lead_time_days`
   - source columns: `exec_date` → `snapshot_date`, `item` → `item_no`, `loc` → `loc`, `lead_time` → `lead_time_days`, `tot_oh` → `qty_on_hand`, `tot_oh_oo` → `qty_on_hand_on_order`, `mtd_sls` → `mtd_sales`
   - rule: `qty_on_order = qty_on_hand_on_order - qty_on_hand` (computed during normalization)
   - materialized view: `agg_inventory_monthly` aggregates to monthly grain with avg/sum metrics

## Component technologies
1. Source ingestion + normalization:
   - Python scripts + `uv` + Make
2. Relational sink:
   - PostgreSQL 16 (pgvector/pgvector:pg16) via `psycopg` copy/load
3. Lakehouse sink:
   - Spark 3.5 writing Apache Iceberg tables to MinIO (S3 API)
4. Query:
   - Trino over Iceberg catalog
5. API + UI:
   - FastAPI backend + React/Vite/shadcn UI frontend
6. NL→SQL chatbot:
   - OpenAI GPT-4o (generation) + text-embedding-3-small (embeddings)
   - pgvector for schema metadata vector search
   - Read-only SQL execution with safety guardrails (SELECT only, 5s timeout, 500-row cap)
7. Multi-model forecasting:
   - `model_id` column on forecast fact table
   - Per-model analytics and model selector in UI
8. DFU clustering:
   - Feature engineering from sales history, item, and DFU attributes
   - KMeans clustering with optimal K selection (elbow, silhouette, gap statistic)
   - Automated cluster labeling (high_volume_steady, seasonal_high_volume, etc.)
   - MLflow experiment tracking (`dfu_clustering`)
   - Cluster assignments stored in `dim_dfu.cluster_assignment`
   - Supports LGBM global models with homogeneous training segments
9. LGBM backtesting (per-cluster only — Feature 44):
   - Expanding window backtest (10 timeframes A–J) with LightGBM regressors
   - Per-cluster strategy (`lgbm_cluster`): separate model per cluster
   - Algorithm options controlled by `config/algorithm_config.yaml` (recursive, shap_select, tune_inline, params_file, hyperparameters)
   - Causal feature engineering: lag 1-12, rolling stats, calendar, DFU/item attributes
   - Execution-lag predictions loaded into `fact_external_forecast_monthly` via COPY + upsert
   - All-lag (0–4) predictions archived in `backtest_lag_archive` for accuracy at any horizon
   - MLflow experiment tracking (`demand_backtest`)
10. CatBoost backtesting (per-cluster only — Feature 44):
   - Same expanding window framework as LGBM (10 timeframes A–J) with CatBoost regressors
   - Per-cluster strategy (`catboost_cluster`)
   - Algorithm options controlled by `config/algorithm_config.yaml`
   - Native categorical feature handling via ordered target encoding (no one-hot needed)
   - GPU support via `task_type="GPU"`; auto-detected at runtime
   - MLflow experiment tracking (`demand_backtest`)
11. XGBoost backtesting (per-cluster only — Feature 44):
   - Same expanding window framework as LGBM (10 timeframes A–J) with XGBoost regressors
   - Per-cluster strategy (`xgboost_cluster`)
   - Algorithm options controlled by `config/algorithm_config.yaml`
   - Native categorical support via `enable_categorical=True` with `tree_method="hist"`
   - GPU support via `device="cuda"`; auto-detected at runtime
   - MLflow experiment tracking (`demand_backtest`)
12. Multi-dimensional accuracy slicing:
   - Pre-aggregated `agg_accuracy_by_dim` view: (model_id, lag, month, cluster, supplier, abc_vol, region, brand) grain
   - Pre-aggregated `agg_accuracy_lag_archive` view: same grain for archive table + timeframe
   - `/forecast/accuracy/slice` endpoint: compare WAPE, Accuracy %, Bias across models by any DFU attribute
   - `/forecast/accuracy/lag-curve` endpoint: accuracy degradation by lag horizon (0–4) per model
   - UI Accuracy Comparison panel: model comparison pivot table + lag curve chart
   - Views refreshed automatically by `backtest-load`; also manually via `make accuracy-slice-refresh`
17. Champion model selection (feature15):
   - Per-DFU per-month best-model selection using Forecast Value Added (FVA) approach
   - 5 configurable strategies via `common/champion_strategies.py` strategy registry:
     - `expanding` — cumulative WAPE, all prior months equal weight (default)
     - `rolling` — last N months only (configurable `window_months`)
     - `decay` — exponential decay weighting recent months more (`decay_factor`)
     - `ensemble` — blend top-K models by inverse-WAPE weights
     - `meta_learner` — ML classifier predicts best model from DFU features + performance stats
   - All strategies enforce **exec-lag-aware strict causality**: selection for month T with execution_lag=L uses ONLY data from `startdate < T − L` (= `startdate < fcstdate`), implemented as `shift(exec_lag + 1)` per DFU-model group — prevents using actuals not available at forecast issuance time
   - **Fallback model** (`fallback_model_id`, default `lgbm_cluster`): fills warm-up DFU-months so every DFU-month always has a champion row
   - WAPE-based DFU-level evaluation: `SUM(ABS(F-A)) / ABS(SUM(A))` per DFU per model
   - Champion composite stored as `model_id='champion'` in `fact_external_forecast_monthly` — auto-appears in all accuracy views
   - Ceiling (oracle) model: per-DFU per-month best model selection — theoretical upper bound with perfect foresight
   - Ceiling stored as `model_id='ceiling'` — provides accuracy benchmark alongside champion
   - Gap-to-ceiling metric shows how far champion is from theoretical best (in percentage points)
   - Meta-learner trained on ceiling labels as ground truth with strict temporal train/test split
   - Simulation script (`simulate_champion_strategies.py`) runs all strategies and compares accuracy vs ceiling
   - YAML config (`config/model_competition.yaml`): competing models, metric, lag mode, min DFU rows, fallback_model_id, strategy, strategy_params, meta_learner config
   - CLI: `make champion-select`, `make champion-simulate`, `make champion-train-meta`, `make champion-all`
   - API endpoints: `GET/PUT /competition/config`, `POST /competition/run`, `GET /competition/summary`
   - UI: Champion Selection panel in Accuracy tab with model checkboxes, metric/lag selectors, strategy selector, champion + ceiling KPI cards, gap indicator, and dual model wins bar charts
   - Summary saved to `data/champion/champion_summary.json` (includes `fallback_rows_inserted` count)
18. Data Explorer performance & UX (feature16):
   - Type-aware SQL filtering: `_col_type()` dispatches to native-type clauses instead of universal `::text` casts
   - GIN trigram indexes (`gin_trgm_ops`) on fact table text columns (model_id, dmdunit, loc, dmdgroup) for indexed `ILIKE` substring search
   - Capped COUNT: `pg_class.reltuples` for unfiltered; `LIMIT 100001` subquery for filtered large tables; `total_approximate` flag in response
   - Column-level typeahead suggestions: `/domains/{domain}/suggest` reused per column header with native HTML `<datalist>`
   - Chemistry-themed loading overlay: periodic table element tile with `pulse-glow` animation, frosted glass backdrop
   - Debounce stability: `useDebounce` uses `JSON.stringify` deep comparison for object values to prevent re-render loops
19. DFU Analysis tab (feature17):
   - Unified sales vs multi-model forecast overlay on a single chart
   - Three analysis modes: Item @ Location (single DFU), All Items @ Location, Item @ All Locations
   - `GET /dfu/analysis` endpoint: server-side join of `agg_sales_monthly` + `agg_forecast_monthly`, returns pre-pivoted series + per-model KPIs
   - Per-model KPI cards: Accuracy %, WAPE, Bias, Total Forecast, Total Actual
   - Toggleable measure visibility (sales line + per-model forecast lines)
   - Typeahead item/location filters with cross-filtering, auto-sample on first visit
20. Market intelligence (feature18):
   - `POST /market-intelligence` endpoint combining Google Custom Search + GPT-4o narrative
   - Item metadata lookup from `dim_item` (item_desc, brand_name, category, producer_name)
   - Location metadata lookup from `dim_location` (state_id, site_desc)
   - Google Custom Search API for product news/trends (5 results)
   - GPT-4o synthesizes search results + state demographics into 3-5 paragraph market briefing
   - Graceful degradation: if Google search fails, LLM generates from its own knowledge
   - UI: "Mi" tab with item/location typeahead, generate button, search result cards, narrative card
   - Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` env vars

## Additional tables
1. `chat_embeddings` — pgvector table storing schema metadata embeddings (1536-dim) for NL query context retrieval
2. `backtest_lag_archive` — stores all-lag (0–4) backtest predictions for accuracy reporting at any horizon; grain: `(forecast_ck, model_id, lag)`; includes `timeframe` column (A–J) for traceability; each row preserves its original `lag` as `execution_lag` (staging table is never mutated during loading)
3. `fact_inventory_snapshot` — monthly inventory position snapshots (~190M rows across 14 months); grain: `(item_no, loc, snapshot_date)`; measures: qty_on_hand, qty_on_hand_on_order, qty_on_order, mtd_sales, lead_time_days
4. `agg_inventory_monthly` — materialized view aggregating inventory to monthly grain (avg on-hand, avg on-order, avg lead time, total MTD sales)
5. `job_history` — persistent job tracking table for the APScheduler-powered job engine; includes scheduling columns (`scheduled_cron`, `retry_count`, `max_retries`, `pipeline_id`, `pipeline_step`, `triggered_by`); grain: `(job_id)` PK
6. `job_schedule` — recurring schedule definitions for APScheduler cron/interval triggers; grain: `(schedule_id)` PK; columns: `job_type`, `params`, `cron_expr`, `interval_min`, `active`

## Accuracy Slice Materialized Views (feature10)
Pre-aggregated views enabling O(1) multi-dimensional KPI slicing without raw-table joins:

1. `agg_accuracy_by_dim` — joins `fact_external_forecast_monthly` + `dim_dfu`, aggregates at (model_id, lag, month, cluster, supplier, abc_vol, region, brand, execution_lag) grain; stores `SUM(F)`, `SUM(A)`, `SUM(ABS(F-A))` for KPI derivation. Refreshed by `backtest-load`.
2. `agg_accuracy_lag_archive` — same aggregation from `backtest_lag_archive` + `dim_dfu`, adds `timeframe` grain; used for lag-horizon accuracy curves. Refreshed by `backtest-load`.

Performance impact: aggregate queries (cluster-level, supplier-level) drop from 5–30s → <300ms.

## Shared Backtest Framework (`common/`)
All tree-based backtest scripts share common logic extracted into reusable modules:

| Module | Purpose |
|--------|---------|
| `common/backtest_framework.py` | `run_tree_backtest()` orchestrator, timeframe generation, data loading, execution-lag assignment, all-lag expansion, post-processing, model-scoped output saving (`data/backtest/<model_id>/`), feature importance; `_fill_predict_nans()`, `_predict_single_month()`, `recursive` param for recursive multi-step inference (Feature 43) |
| `common/feature_engineering.py` | `build_feature_matrix()`, `get_feature_columns()`, `mask_future_sales()` with `cat_dtype` parameter for framework-specific categorical handling; `update_grid_with_predictions()` for recursive multi-step lag write-back (Feature 43) |
| `common/metrics.py` | `compute_accuracy_metrics()`: WAPE, bias, accuracy % |
| `common/mlflow_utils.py` | `log_backtest_run()`: generic MLflow experiment logging |
| `common/db.py` | `get_db_params()`: shared DB connection parameters |
| `common/constants.py` | `CAT_FEATURES`, `LAG_RANGE`, `ROLLING_WINDOWS`, output column ordering, thresholds |
| `common/tuning.py` | Shared tuning utilities: `generate_cv_month_splits`, `compute_wape_stabilised`, `suggest_params`, `save_best_params`, `load_best_params`, `best_rounds_to_n_estimators`, `tune_for_timeframe()` (per-timeframe causal tuning, PL-002), `TRAIN_FOLD_FNS` registry (`train_lgbm_fold`, `train_catboost_fold`, `train_xgboost_fold`) (Feature 41) |
| `common/shap_selector.py` | SHAP-based feature selection: `compute_shap_global` (LGBM/XGBoost via `shap.TreeExplainer`), `compute_shap_catboost` (native ShapValues), `compute_timeframe_shap` (cluster-pooled or global), `build_shap_summary`, `save_shap_outputs` (Feature 42) |

Each model script (LGBM, CatBoost, XGBoost) implements only `train_and_predict_per_cluster()`, passed as `train_fn_per_cluster` to `run_tree_backtest()`. Global and transfer strategies were removed in Feature 44 — only per-cluster training is supported. Algorithm behavior (recursive, SHAP selection, inline tuning, params file, hyperparameters) is read from `config/algorithm_config.yaml`, not from CLI flags. `run_tree_backtest()` accepts optional `feature_selector_fn` callable (Feature 42): when provided, each timeframe computes SHAP after the initial model train and retrains on the selected feature subset before generating predictions. `run_tree_backtest()` also accepts `recursive: bool = False` (Feature 43): when `True`, each predict month is scored one at a time using `_predict_single_month(models, predict_data, feature_cols)`, and predictions are written back into the feature grid via `update_grid_with_predictions()` so that `qty_lag_1` for month T+1 reflects the model's own prediction for month T rather than zero.

## ML Pipeline Components
1. **Feature Engineering** (`generate_clustering_features.py`):
   - Extracts time series features from `fact_sales_monthly` (volume, trend, seasonality, volatility, growth)
   - Joins with `dim_dfu` and `dim_item` for attribute features
   - Outputs feature matrix CSV for clustering
2. **Clustering Model** (`train_clustering_model.py`):
   - StandardScaler normalization
   - Optional PCA dimensionality reduction
   - KMeans with optimal K selection (elbow, silhouette, gap statistic)
   - Generates cluster assignments and centroids
   - Logs to MLflow with parameters, metrics, and visualization artifacts
3. **Cluster Labeling** (`label_clusters.py`):
   - Analyzes cluster centroids to assign business labels
   - Volume tiers: high/medium/low
   - Pattern types: steady, seasonal, trending, intermittent, volatile
   - Composite labels: high_volume_steady, seasonal_high_volume, etc.
4. **Assignment Update** (`update_cluster_assignments.py`):
   - Updates `dim_dfu.cluster_assignment` column in PostgreSQL
   - Validates updates and reports cluster distribution
5. **LGBM Backtest** (`run_backtest.py` → `common/backtest_framework.py` — Feature 44):
   - Uses shared `run_tree_backtest()` orchestrator from `common/backtest_framework.py`
   - Script contains only LGBM-specific `train_and_predict_per_cluster()` function
   - Algorithm options read from `config/algorithm_config.yaml` (recursive, shap_select, tune_inline, params_file, hyperparams)
   - Shared feature engineering from `common/feature_engineering.py`: lag 1-12, rolling mean/std 3/6/12m, calendar, DFU/item attributes
   - Per-cluster strategy only (`lgbm_cluster`)
   - Outputs two CSVs: execution-lag only (main table) + all lags 0–4 (archive)
   - Deduplication across timeframes (latest timeframe wins)
   - MLflow logging via `common/mlflow_utils.py` to `demand_backtest` experiment
6. **CatBoost Backtest** (`run_backtest_catboost.py` → `common/backtest_framework.py` — Feature 44):
   - Uses shared `run_tree_backtest()` orchestrator with `cat_dtype="str"` for CatBoost's index-based categoricals
   - Script contains only `train_and_predict_per_cluster()` function
   - Algorithm options read from `config/algorithm_config.yaml`
   - Per-cluster strategy only (`catboost_cluster`)
7. **XGBoost Backtest** (`run_backtest_xgboost.py` → `common/backtest_framework.py` — Feature 44):
   - Uses shared `run_tree_backtest()` orchestrator with `cat_dtype="category"` for XGBoost's native categoricals
   - Script contains only `train_and_predict_per_cluster()` function
   - Algorithm options read from `config/algorithm_config.yaml`
   - Per-cluster strategy only (`xgboost_cluster`)
8. **Backtest Loader** (`load_backtest_forecasts.py`):
   - Loads execution-lag rows into `fact_external_forecast_monthly` via COPY + staging + upsert
   - Loads all-lag rows into `backtest_lag_archive` via same pattern
   - Supports `--model MODEL_ID` (load from `data/backtest/<MODEL_ID>/`), `--all` (scan all model subdirs), `--input PATH` (legacy)
   - `--replace` scoped to `model_id` in CSV (safe for multi-model coexistence)
   - Refreshes `agg_forecast_monthly`, `agg_accuracy_by_dim`, `agg_accuracy_lag_archive` materialized views
   - Each backtest writes to `data/backtest/<model_id>/` subdirectory (prevents CSV overwrites — PL-001 fix)
15. **SHAP Feature Selection** (`common/shap_selector.py` — Feature 42):
   - Per-timeframe SHAP computation integrated into `run_tree_backtest()` via `feature_selector_fn` hook
   - LGBM/XGBoost: `shap.TreeExplainer` via `compute_shap_global`; CatBoost: native `get_feature_importance(type="ShapValues")` via `compute_shap_catboost`
   - For per_cluster/transfer strategies: SHAP pooled across cluster models weighted by cluster size via `_weighted_pool_cluster_shap`; `ml_cluster` excluded from effective feature set
   - Feature selection: cumulative importance threshold (default 95%) or exact top-N; minimum 5 features guaranteed
   - Output: `data/backtest/<model_id>/shap/shap_timeframe_XX.csv` (per-timeframe) + `shap_summary.csv` (cross-timeframe aggregated)
   - API: 4 read-only endpoints under `/forecast/shap/` served directly from CSVs (no DB queries) via `api/routers/shap.py`
   - Frontend: collapsible "Feature Importance (SHAP)" panel in Accuracy tab; indigo=selected / gray=dropped bar chart
   - Config keys in `config/algorithm_config.yaml`: `shap_select`, `shap_top_n`, `shap_threshold`, `shap_sample_size`; composable with `tune_inline` and `params_file` (Feature 44)
   - Activated by setting `shap_select: true` in the algorithm section; run via `make backtest-lgbm`, `make backtest-catboost`, or `make backtest-xgboost`
   - Graceful degradation: SHAP failures log warning and keep all features; backtest continues uninterrupted
16. **Recursive Multi-Step Inference** (`common/backtest_framework.py` + `common/feature_engineering.py` — Feature 43):
   - `--recursive` CLI flag on LGBM, CatBoost, and XGBoost backtest scripts; passes `recursive=True` to `run_tree_backtest()`
   - In direct mode (default), months 2+ of the prediction window use `qty_lag_1 = 0` (masked sales). In recursive mode, each predict month is scored individually, and the model's prediction for month T is written back via `update_grid_with_predictions()` before scoring month T+1
   - `update_grid_with_predictions(grid, month, predictions)` in `common/feature_engineering.py`: writes predicted `basefcst_pref` to `qty[month]` then recomputes all lag (1-12) and rolling (3m/6m/12m) features in a single vectorized `groupby().shift()` pass
   - `_predict_single_month(models, data, feature_cols)` in `common/backtest_framework.py`: routes one month's batch to the correct cluster model dict without retraining; per-cluster only (Feature 44)
   - `_fill_predict_nans(predict_data, feature_cols, cat_cols)`: fills numeric NaN lag features with 0 per-month (skips categorical columns)
   - Training cost unchanged: model trained once per timeframe; recursive loop is inference-only
   - Composable with `shap_select` and `tune_inline` via `config/algorithm_config.yaml` (Feature 44)
   - `"recursive": true` written to `backtest_metadata.json` for traceability
   - Enabled via `recursive: true` in algorithm config; run via `make backtest-lgbm`, `make backtest-catboost`, `make backtest-xgboost`
   - No API, frontend, or DB schema changes
14. **Hyperparameter Tuning** (`scripts/tune_hyperparams.py` + `common/tuning.py`):
   - Bayesian optimisation via Optuna (TPESampler + MedianPruner) for LGBM, CatBoost, XGBoost
   - Walk-forward expanding CV with causal masking (`mask_future_sales()` inside each fold)
   - `n_estimators` determined by early stopping (excluded from search space)
   - Per-cluster WAPE breakdown logged in output JSON and MLflow
   - Search spaces and CV settings in `config/hyperparameter_tuning.yaml` (includes `inline_n_trials: 20`, `inline_n_splits: 3`)
   - Output: `data/tuning/best_params_<model>.json` consumed via `params_file` key in `config/algorithm_config.yaml` (Feature 44)
   - MLflow experiment: `hyperparameter_tuning`
   - **Per-timeframe causal inline tuning (PL-002):** `tune_for_timeframe()` in `common/tuning.py` filters the feature matrix to `months <= cutoff_date` before running a lightweight Optuna study (20 trials, 3 folds) — eliminates future leakage into backtest accuracy metrics. Enabled via `tune_inline: true` in `config/algorithm_config.yaml`. `TRAIN_FOLD_FNS` registry (`train_lgbm_fold`, `train_catboost_fold`, `train_xgboost_fold`) shared between global tuning and inline tuner. `run_tree_backtest()` accepts optional `inline_tuner_fn` callable — each timeframe gets its own causally-valid params.
   - **Two modes:** Production (`params_file` in algorithm config — global tune once, apply everywhere) vs. Honest backtesting (`tune_inline: true` in algorithm config — 600 fits vs 250, no future leakage)
12. **Champion Selection** (`run_champion_selection.py` + `common/champion_strategies.py`):
   - 5 configurable strategies: expanding, rolling, decay, ensemble, meta_learner
   - Strategy registry in `common/champion_strategies.py` — all strategies operate on pandas DataFrames (testable without DB)
   - All strategies enforce **exec-lag-aware causality** via `shift(exec_lag + 1)` per DFU-model group — selection for month T excludes last exec_lag months whose actuals weren't available at issuance time; backward compatible with exec_lag=0
   - **Fallback model** fills warm-up DFU-months (NOT EXISTS + ON CONFLICT DO NOTHING insert) so every DFU-month has a champion row
   - Bulk inserts champion rows via temp table + COPY + INSERT...SELECT with `model_id='champion'`
   - Also computes ceiling (oracle): best model per DFU per month via `ABS(basefcst_pref - tothist_dmd)` ranking
   - Ceiling rows stored as `model_id='ceiling'` — theoretical upper bound with perfect foresight
   - Refreshes materialized views so champion + ceiling auto-appear in all accuracy comparisons
   - Config-driven via `config/model_competition.yaml`; also callable via API
   - Meta-learner (`scripts/train_meta_learner.py`): RandomForest/XGBoost classifier trained on ceiling labels with temporal split
   - Simulation (`scripts/simulate_champion_strategies.py`): runs all strategies, compares accuracy vs ceiling

21. Inventory Planning — Phase 1 (feature34):
   - Inventory position snapshots from 14 monthly CSV files (~190M rows total)
   - DDL: `fact_inventory_snapshot` with B-tree + GIN trigram indexes, `agg_inventory_monthly` materialized view
   - Custom normalize script (`normalize_inventory_csv.py`) merges multi-file CSVs with streaming (no pandas)
   - 4 API endpoints: `GET /inventory/position` (latest per item-loc via DISTINCT ON), `GET /inventory/kpis` (aggregate metrics), `GET /inventory/trend` (monthly from agg view), `GET /inventory/item-detail` (full history for item-loc pair)
   - Frontend: InventoryTab with KPI cards, filter controls (item/location debounce, months selector), trend chart (dual Y-axis), paginated position table, item detail panel
   - Makefile: `normalize-inventory`, `load-inventory`, `refresh-agg-inventory`, `db-apply-inventory`, `inventory-pipeline`
22. Backtest model cleanup (feature23):
   - CLI utility (`scripts/clean_backtest_models.py`) for selective removal of model predictions
   - Deletes from `fact_external_forecast_monthly` and `backtest_lag_archive` by `model_id`
   - Refreshes 5 materialized views: `agg_forecast_monthly`, `agg_accuracy_by_dim`, `agg_dfu_coverage`, `agg_accuracy_lag_archive`, `agg_dfu_coverage_lag_archive`
   - Modes: `--list` (inventory), `--dry-run` (preview), `--all-backtest` (bulk cleanup excluding external)
   - Makefile targets: `backtest-clean`, `backtest-list`
   - Date-range cleanup: `scripts/clean_forecasts_by_date.py` deletes by time bucket (`--before`, `--after`, `--between`, `--months`) on `startdate` or `fcstdate`, with optional `--model` filter, `--forecast-only`/`--archive-only` scope
   - Makefile targets: `forecast-clean`, `forecast-clean-list`
23. Product-grade UI overhaul (feature36):
   - Collapsible sidebar navigation replacing horizontal tab bar (9 nav items across 5 sections, 64px collapsed / 240px expanded, mobile drawer)
   - Global filter bar: brand, category, market, channel multi-select dropdowns with debounced URL sync via React context
   - Dashboard overview landing page: 6 KPI cards with sparklines/trends, AlertPanel (severity-coded), HeatmapGrid (category × time accuracy), TopMovers (period-over-period), ForecastTrendChart (ECharts)
   - Three product themes: Wine & Spirits ("The Reserve", burgundy+gold), General ("Demand Studio", blue SaaS), Obsidian ("Command", green+black dark-only)
   - CSS variable-driven theming: 35 HSL custom properties applied at runtime on `<html>`, light/dark color modes per theme
   - 5 new API endpoints: `GET /domains/{domain}/distinct`, `GET /dashboard/kpis`, `GET /dashboard/alerts`, `GET /dashboard/top-movers`, `GET /dashboard/heatmap`
   - `mv_top_movers` materialized view for period-over-period volume changes
   - New components: AppSidebar, ThemeSelector, GlobalFilterBar, WidgetGrid/WidgetCard, AlertPanel, HeatmapGrid, TopMovers, ForecastTrendChart, DashboardTab
   - Enhanced KpiCard with sparkline SVG, trend delta, severity, icon support
   - Keyboard shortcuts: `[` sidebar toggle, `t` theme cycle, `d` mode toggle, 1-9 tab switch
24. Job Scheduler/Monitor with APScheduler (feature39):
   - Production-grade job execution powered by APScheduler 3.11 (`BackgroundScheduler` + `ThreadPoolExecutor(max_workers=4)`)
   - Persistent `job_history` + `job_schedule` tables in Postgres
   - `JobManager` singleton with per-group concurrency control (one active job per group)
   - 7 job types across 4 groups: clustering (cluster_scenario, cluster_pipeline), backtest (lgbm, catboost, xgboost), seasonality (seasonality_pipeline), champion (champion_select)
   - REST API: 12 endpoints — core CRUD (types, submit, list, active, detail, cancel, delete), scheduling (create schedule, list schedules, remove schedule), pipeline (submit pipeline), stats (dashboard aggregates)
   - Cron/interval scheduling for recurring automation (e.g., daily 2AM backtest, weekly clustering refresh)
   - Job pipelines: sequential chaining of multi-step workflows (cluster → backtest → champion select)
   - Retry logic with exponential backoff (configurable max_retries per job)
   - Professional automation dashboard UI: KPI cards, grouped job type cards with category colors, live active job monitoring with animated progress bars, schedule dialog, schedules section, expandable job history
   - `JobNotificationContext` for cross-tab completion/failure alerts on Dashboard
   - Clustering What-If scenarios from ClustersTab delegate to JobManager ("Schedule Scenario Job")
   - Sidebar nav item with active job count badge, keyboard shortcut `9`
   - Foundation for agentic AI automation
   - **Vite proxy requirement:** `frontend/vite.config.ts` must include `/jobs` proxy entry to forward API calls to FastAPI backend; without it the UI receives HTML instead of JSON
   - **Scenario queueing:** When a group is busy, new jobs are queued (`status="queued"`) instead of rejected with 409. Queued jobs auto-dispatch via `_dispatch_next()` when the active job completes (FIFO order)
   - **View Results navigation:** "View Results" button in JobsTab navigates to ClustersTab with `?scenario_job=<id>` URL param for completed cluster_scenario jobs; ClustersTab auto-loads and renders ScenarioCharts
   - **Past Scenarios history:** ClustersTab What-If panel shows last 10 completed scenarios in an accordion with inline charts and promote buttons

## Testing Infrastructure

Full-stack automated testing covering backend (Python) and frontend (TypeScript):

### Backend (pytest)
| Layer | Framework | Directory |
|-------|-----------|-----------|
| Unit tests | pytest | `tests/unit/` |
| API tests | pytest + httpx AsyncClient (ASGI transport) | `tests/api/` |
| Coverage | pytest-cov | `--cov=api --cov=common` |

**Test targets:**
- `make test` — all backend tests
- `make test-unit` — unit tests only (`tests/unit/`)
- `make test-api` — API endpoint tests only (`tests/api/`)
- `make test-cov` — backend tests with coverage report

**Backend test suites:**
| Suite | Module Under Test | Tests |
|-------|-------------------|-------|
| `test_metrics.py` | `common/metrics.py` — WAPE, bias, accuracy % | 10 |
| `test_constants.py` | `common/constants.py` — LAG_RANGE, ROLLING_WINDOWS, CAT_FEATURES | 11 |
| `test_domain_specs.py` | `common/domain_specs.py` — all 8 domains, parametrized | 14+ |
| `test_backtest_framework.py` | `common/backtest_framework.py` — timeframe generation | 9 |
| `test_feature_engineering.py` | `common/feature_engineering.py` — feature matrix, mask_future_sales, update_grid_with_predictions (Feature 43) | 6+ |
| `test_mlflow_utils.py` | `common/mlflow_utils.py` — experiment logging | 3 |
| `test_db.py` | `common/db.py` — connection parameters | 5 |
| `test_load_dataset_postgres.py` | `scripts/load_dataset_postgres.py` — forecast execution-lag loading (JOIN-based filter, no staging mutation), archive loading, SQL generation | 24 |
| `test_health.py` | `api/main.py` — health endpoint | 1 |
| `test_domains.py` | `api/main.py` — domain CRUD endpoints | 6 |
| `test_forecast_accuracy.py` | `api/main.py` — accuracy/lag endpoints | 3 |
| `test_dfu_analysis.py` | `api/main.py` — DFU analysis endpoint | 2 |
| `test_competition.py` | `api/main.py` — champion selection endpoints | 2 |
| `test_clusters.py` | `api/main.py` — cluster endpoints | 2 |
| `test_inventory.py` | `api/main.py` — inventory endpoints | 18 |
| `test_inventory_domain.py` | `common/domain_specs.py` — inventory domain spec | 28 |
| `test_distinct.py` | `api/main.py` — distinct values endpoint | 12 |
| `test_dashboard.py` | `api/main.py` — dashboard endpoints (kpis, alerts, top-movers, heatmap) | 17 |
| `test_jobs.py` | `api/routers/jobs.py` — job scheduler endpoints (types, submit, list, cancel, delete, stats, schedules, pipeline) | 16 |
| `test_shap_selector.py` | `common/shap_selector.py` — SHAP extraction, feature selection, cluster pooling, CSV output, error fallback | 22 |
| `test_shap.py` | `api/routers/shap.py` — SHAP endpoints (models list, summary, timeframes, per-timeframe detail, 404 cases) | 8 |
| `test_backtest_recursive.py` | `common/backtest_framework.py` — `_fill_predict_nans`, `_predict_single_month` (global/cluster/transfer), recursive loop integration | 13 |

**API test pattern:** httpx `AsyncClient` with `ASGITransport(app)` — no running server needed. DB connections mocked via `pool` fixture in `tests/api/conftest.py`.

### Frontend (Vitest + React Testing Library)
| Layer | Framework | Directory |
|-------|-----------|-----------|
| Hook tests | Vitest + renderHook | `src/hooks/__tests__/` |
| Utility tests | Vitest | `src/lib/__tests__/`, `src/api/__tests__/` |
| Component tests | Vitest + RTL | `src/components/__tests__/` |
| Tab tests | Vitest + RTL | `src/tabs/__tests__/` |

**Test target:** `make ui-test`

**Frontend test suites:**
| Suite | Module Under Test | Tests |
|-------|-------------------|-------|
| `useTheme.test.ts` | Theme management hook (product themes + color modes) | 9 |
| `useUrlState.test.ts` | URL state synchronization (9 tabs, overview default) | 12 |
| `useKeyboardShortcuts.test.ts` | Keyboard shortcuts (1-7 tabs, sidebar, theme) | 8 |
| `useSidebar.test.ts` | Sidebar state management | 8 |
| `useGlobalFilters.test.ts` | Global filter state + URL sync | 7 |
| `export.test.ts` | CSV export (papaparse) | 4 |
| `formatters.test.ts` | Number/cell formatting | 23 |
| `queries.test.ts` | TanStack Query keys + stale times | 10 |
| `Skeleton.test.tsx` | Loading skeleton components | 7 |
| `KeyboardShortcutHelp.test.tsx` | Keyboard help dialog | 5 |
| `EChartContainer.test.tsx` | ECharts wrapper | 4 |
| `ExplorerTab.test.tsx` | Data Explorer tab | 2 |
| `AccuracyTab.test.tsx` | Accuracy tab | 1 |
| `DfuAnalysisTab.test.tsx` | DFU Analysis tab | 1 |
| `ClustersTab.test.tsx` | Clusters tab | 1 |
| `MarketIntelTab.test.tsx` | Market Intelligence tab | 1 |
| `ChatPanel.test.tsx` | Chat panel | 1 |
| `InventoryTab.test.tsx` | Inventory tab | 5 |
| `WhatIfScenarios.test.tsx` | Clustering What-If scenarios | 8 |
| `AppSidebar.test.tsx` | Sidebar navigation | 11 |
| `ThemeSelector.test.tsx` | Theme + color mode picker | 9 |
| `GlobalFilterBar.test.tsx` | Global filter bar | 7 |
| `WidgetGrid.test.tsx` | Widget grid layout | 11 |
| `AlertPanel.test.tsx` | Alert severity panel | 6 |
| `TopMovers.test.tsx` | Top movers list | 5 |
| `HeatmapGrid.test.tsx` | Heatmap grid | 13 |
| `DashboardTab.test.tsx` | Dashboard overview tab | 4 |
| `JobsTab.test.tsx` | Jobs automation dashboard tab | 7 |

**Combined:** `make test-all` runs backend + frontend.

**Mandatory rule:** Every new feature, endpoint, component, or utility must include corresponding tests. See `docs/design-specs/feature31.md` for the full testing strategy.

## How to add next dataset
1. Add `<DATASET>_SPEC` in `common/domain_specs.py`
2. Add matching DDL in `sql/`
3. Add Make targets:
   - `normalize-<dataset>`
   - `load-<dataset>`
   - `spark-<dataset>` (calls generic spark script with `--dataset <dataset>`)
4. For generic dimension/fact domains: API uses existing generic `/domains/{domain}/...` endpoints
5. For specialized domains (like inventory): Add dedicated API endpoints and frontend tab
6. Run `make generate-embeddings` to update chat context with new schema metadata

**Example: Inventory domain** — uses dedicated normalize script (`normalize_inventory_csv.py`) for multi-file merge, dedicated API endpoints (`/inventory/*`), and dedicated UI tab (`InventoryTab`). Domain spec still lives in `common/domain_specs.py` for schema metadata.
