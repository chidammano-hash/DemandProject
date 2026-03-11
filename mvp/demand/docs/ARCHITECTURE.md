# Unified Architecture

## Goal
Reduce dataset-by-dataset duplication and provide a reusable path for adding new dimensions and facts.

## Current pattern
1. Define dataset spec in `common/domain_specs.py`
2. Add DDL in `sql/`
3. Reuse generic scripts:
   - `normalize_dataset_csv.py`
   - `load_dataset_postgres.py`
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
3. API + UI:
   - FastAPI backend + React/Vite/shadcn UI frontend
6. NL→SQL chatbot:
   - OpenAI GPT-4o (generation) + text-embedding-3-small (embeddings)
   - pgvector for schema metadata vector search
   - Read-only SQL execution with safety guardrails (SELECT only, 5s timeout, 500-row cap)
6b. AI Planning Agent:
   - Anthropic Claude `claude-opus-4-6` via `anthropic>=0.40.0` SDK (tool_use API)
   - `AIPlannerAgent` in `common/ai_planner.py`: 10 tools, agentic loop, insight creation
   - Not a chatbot — proactive exception work-queue scanner writing structured insights to DB
7. Multi-model forecasting:
   - `model_id` column on forecast fact table
   - Per-model analytics and model selector in UI
8. DFU clustering:
   - Feature engineering: 14 core features across 6 dimensions (volume, trend, seasonality, periodicity, intermittency, lifecycle) from 36-month sales history
   - New features: FFT periodicity strength, OLS seasonal R-squared, Croston ADI, scale-invariant trend slope, IQR, CAGR, recency ratio, YoY correlation
   - KMeans clustering with combined Silhouette + Calinski-Harabasz scoring (0.5*sil + 0.5*CH); gap statistic removed
   - Hard 5% minimum cluster size constraint; k_range [5, 18]; post-hoc merge of small clusters
   - Priority-ordered taxonomy labeling: Intermittency -> Periodicity -> Seasonality -> Trend -> Volatility -> Volume (5 tiers)
   - Compound labels: `high_volume_seasonal_growing`, `low_volume_intermittent`, `very_high_volume_growing`, etc.
   - MLflow experiment tracking (`dfu_clustering`)
   - Cluster assignments stored in `dim_dfu.cluster_assignment`
9. LGBM backtesting (Feature 44):
   - Expanding window backtest (10 timeframes A-J) with LightGBM regressors
   - Configurable `cluster_strategy`: `per_cluster` (default, one model per ml_cluster) or `global` (one model on all data)
   - `ml_cluster` is always a hard feature — never stripped from feature_cols in either strategy
   - Algorithm options controlled by `config/algorithm_config.yaml` (cluster_strategy, recursive, shap_select, tune_inline, params_file, hyperparameters)
   - Causal feature engineering: lag 1-12, rolling stats, calendar, DFU/item attributes
   - Execution-lag predictions loaded into `fact_external_forecast_monthly` via COPY + upsert
   - All-lag (0-4) predictions archived in `backtest_lag_archive` for accuracy at any horizon
   - MLflow experiment tracking (`demand_backtest`)
10. CatBoost backtesting (Feature 44):
   - Same expanding window framework as LGBM (10 timeframes A-J) with CatBoost regressors
   - Configurable `cluster_strategy`: `per_cluster` (default) or `global`; ml_cluster always a hard feature
   - Algorithm options controlled by `config/algorithm_config.yaml`
   - Native categorical feature handling via ordered target encoding (no one-hot needed)
   - GPU support via `task_type="GPU"`; auto-detected at runtime
   - MLflow experiment tracking (`demand_backtest`)
11. XGBoost backtesting (Feature 44):
   - Same expanding window framework as LGBM (10 timeframes A-J) with XGBoost regressors
   - Configurable `cluster_strategy`: `per_cluster` (default) or `global`; ml_cluster always a hard feature
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
   - **Clickable forecast lines**: clicking any backtest model line sets `selectedModel` state in `DfuAnalysisTab`; selected line renders thicker + unselected lines fade to 30% opacity; hint text "↑ click a forecast line to explore SHAP" near toggles
   - **Per-DFU SHAP Panel** (`DfuShapPanel.tsx`): on model selection renders a stacked Recharts BarChart below the overlay chart showing signed SHAP feature contributions per month; future months rendered at 45% fill opacity; 15-color palette per feature; scrollable container (min 800px); dual-stack with `ReferenceLine y={0}` baseline; falls back to cluster-level summary SHAP (existing `/forecast/shap/{model}/summary` endpoint) with warning banner when per-DFU pkl artifacts are not available (404); placeholder card when no model selected
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
7. `fact_eoq_targets` — computed EOQ metrics per DFU (IPfeature4); grain: `(item_no, loc)`; columns: demand_mean_monthly, annual_demand, ordering_cost, holding_cost_pct, unit_cost, moq, eoq, effective_eoq, eoq_cycle_stock, order_frequency, annual_holding_cost, annual_order_cost, total_annual_cost, computed_at
8. `dim_replenishment_policy` — replenishment policy definitions (IPfeature5); grain: `(policy_id)`; columns: policy_name, policy_type, segment, review_cycle_days, service_level, use_eoq, use_safety_stock, active, dfu_count
9. `fact_dfu_policy_assignment` — DFU-to-policy assignments (IPfeature5); grain: `(item_no, loc)`; columns: policy_id (FK), assigned_at, assigned_by; `UNIQUE(item_no, loc)` prevents duplicate assignments
10. `fact_safety_stock_targets` — stub table for safety stock targets (IPfeature6); populated by IPfeature3; currently empty, causing health score SS components to return neutral scores until IPfeature3 is implemented
11. `mv_inventory_health_score` — materialized view computing composite inventory health scores (IPfeature6); grain: `(item_no, loc)`; 4 components × 25 pts = 0–100 composite; tiers: healthy (≥80), monitor (≥60), at_risk (≥40), critical (<40); components: SS Coverage (0–25), DOS Target Adherence (0–25), Stockout Risk History (0–25), Forecast Accuracy (0–25)
12. `fact_replenishment_exceptions` — exception queue table (IPfeature7); grain: `(exception_id)` UUID PK; columns: item_no, loc, exception_date, exception_type, severity, current state snapshot, recommendation (order qty, order_by date, receipt date, estimated value), workflow (status, acknowledged_by, acknowledged_ts, ordered_ts, resolved_ts, notes); 6 exception types: below_rop, below_rop_critical, below_ss, stockout, excess, zero_velocity; 4 severity levels: critical/high/medium/low; 4 workflow statuses: open/acknowledged/ordered/resolved
13. `mv_fill_rate_monthly` — materialized view aggregating order fill rate metrics by item-location-month (IPfeature8)
14. `fact_demand_signals` — short-horizon demand signals computed from recent sales velocity and inventory movement (IPfeature9); grain: `(item_no, loc, signal_date)`
15. `fact_ss_simulation_results` — Monte Carlo safety stock simulation output (IPfeature10); grain: `(simulation_id, item_no, loc)`
16. `mv_supplier_performance` — materialized view aggregating supplier delivery performance KPIs from inventory receipt data (IPfeature12)
17. `fact_inventory_investment_plan` — computed capital investment allocation plan per item-location (IPfeature13); grain: `(item_no, loc, plan_date)`
18. `fact_efficient_frontier` — efficient frontier curve data points for budget vs. service level trade-off (IPfeature13)
19. `mv_intramonth_stockout` — materialized view detecting within-month stockout events from daily inventory snapshots (IPfeature14)
20. `mv_control_tower_kpis` — materialized view aggregating cross-dimensional KPIs for the Control Tower dashboard (IPfeature15)
21. `ai_insights` — AI-generated planning exception records (IPAIfeature1); grain: `(insight_id)` PK; 5 insight types (stockout_risk, excess_inventory, forecast_bias, policy_gap, champion_degradation); 4 severity levels; open/acknowledged/resolved workflow; financial_impact_estimate, reasoning, recommendation
22. `ai_planning_memos` — AI-generated planning narrative memos (IPAIfeature1); grain: `(memo_id)` PK; scope: portfolio or DFU; narrative_text + content_json; indexed by period + scope
23. `fact_replenishment_plan` — forward-looking replenishment plan per item-location-month (CI Bands + Repl. Plan); grain: `(item_no, loc, plan_month, plan_version)`; measures: forecast_qty with P10/P90 CI bands (forecast_qty_lower, forecast_qty_upper), ss_combined (forecast-driven SS), historical_ss, ss_delta, eoq, cycle_stock, reorder_point, order_qty, order_up_to_level, is_below_ss, horizon_months, avg_daily_demand, sigma_method

## Accuracy Slice Materialized Views (feature10)
Pre-aggregated views enabling O(1) multi-dimensional KPI slicing without raw-table joins:

1. `agg_accuracy_by_dim` — joins `fact_external_forecast_monthly` + `dim_dfu`, aggregates at (model_id, lag, month, cluster, supplier, abc_vol, region, brand, execution_lag) grain; stores `SUM(F)`, `SUM(A)`, `SUM(ABS(F-A))` for KPI derivation. Refreshed by `backtest-load`.
2. `agg_accuracy_lag_archive` — same aggregation from `backtest_lag_archive` + `dim_dfu`, adds `timeframe` grain; used for lag-horizon accuracy curves. Refreshed by `backtest-load`.

Performance impact: aggregate queries (cluster-level, supplier-level) drop from 5–30s → <300ms.

## API Router Architecture

`api/main.py` is a ~65-line shell that only creates the app, adds middleware, and mounts all 30 routers via `app.include_router()`. All route handlers live in router modules under `api/routers/`. `domains.py` is mounted last (catch-all `{domain}` path parameter).

**30 active router modules** (as of CI Bands + Repl. Plan):
accuracy, ai_planner, analysis, chat, clusters, competition, control_tower, dashboard, domains, fill_rate, intel, inv_backtest, inventory, inv_planning (shim), inv_planning_abc_xyz, inv_planning_demand_signals, inv_planning_eoq, inv_planning_exceptions, inv_planning_health, inv_planning_intramonth, inv_planning_investment, inv_planning_lead_time, inv_planning_policy, inv_planning_replenishment, inv_planning_safety_stock, inv_planning_simulation, inv_planning_supplier, inv_planning_variability, jobs, shap, storyboard

**16 Vite proxy path prefixes** in `frontend/vite.config.ts`:
`/domains`, `/jobs`, `/clustering`, `/forecast`, `/inventory`, `/dashboard`, `/health`, `/chat`, `/dfu`, `/competition`, `/market-intelligence`, `/inv-planning`, `/fill-rate`, `/control-tower`, `/ai-planner`, `/storyboard`

**CRITICAL:** Every new API path prefix must be added to `frontend/vite.config.ts` or the frontend receives HTML instead of JSON. Restart the Vite dev server after changes.

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
| `common/job_state.py` | In-memory job state: `_active_jobs`, `_pending_queues`, `_cancel_flags`, state lock, status constants; extracted from `job_registry.py` for separation of concerns |
| `common/job_scheduler.py` | APScheduler wrapper: `make_scheduler()`, `make_trigger()` utilities; extracted from `job_registry.py` to isolate APScheduler-specific initialization and trigger creation |

Each model script (LGBM, CatBoost, XGBoost) implements both `train_and_predict_per_cluster()` and `train_and_predict_global()`, selecting which to pass to `run_tree_backtest()` based on the `cluster_strategy` key in `config/algorithm_config.yaml` (`per_cluster` or `global`). **`ml_cluster` is always a hard feature** — it is never stripped from `feature_cols` in either strategy. In `per_cluster` mode it provides a constant identity signal within each partition; in `global` mode it provides inter-cluster discrimination across the full dataset. Algorithm behavior (cluster_strategy, recursive, SHAP selection, inline tuning, params file, hyperparameters) is read from `config/algorithm_config.yaml`, not from CLI flags. `run_tree_backtest()` accepts optional `feature_selector_fn` callable (Feature 42): when provided, each timeframe computes SHAP after the initial model train and retrains on the selected feature subset before generating predictions. `run_tree_backtest()` also accepts `recursive: bool = False` (Feature 43): when `True`, each predict month is scored one at a time using `_predict_single_month(models, predict_data, feature_cols)`, and predictions are written back into the feature grid via `update_grid_with_predictions()` so that `qty_lag_1` for month T+1 reflects the model's own prediction for month T rather than zero.

## ML Pipeline Components
1. **Feature Engineering** (`generate_clustering_features.py`):
   - Extracts 14 core features across 6 dimensions (volume, trend, seasonality, periodicity, intermittency, lifecycle) from `fact_sales_monthly` (36-month window, min 12 months)
   - New features: FFT periodicity strength, OLS seasonal R-squared, Croston ADI, scale-invariant trend slope (`slope * n / mean`), IQR, CAGR, recency ratio, YoY correlation
   - Joins with `dim_dfu` and `dim_item` for attribute features
   - Outputs feature matrix CSV for clustering
2. **Clustering Model** (`train_clustering_model.py`):
   - Log-transforms skewed volume features (`mean_demand`, `iqr_demand`, `adi`, etc.) via `log1p` before StandardScaler
   - Uses only 14 CORE_FEATURES for clustering (not all computed features)
   - KMeans with combined Silhouette + Calinski-Harabasz scoring (`0.5 * sil_norm + 0.5 * CH_norm`); gap statistic removed
   - Hard 5% minimum cluster size constraint during K selection; k_range [5, 18]
   - Post-hoc `merge_small_clusters()` merges any cluster below threshold into nearest large neighbor
   - Optional PCA dimensionality reduction (disabled by default)
   - Generates cluster assignments and centroids
   - Logs to MLflow with parameters, metrics, and visualization artifacts
3. **Cluster Labeling** (`label_clusters.py`):
   - Priority-ordered taxonomy: Intermittency -> Periodicity -> Seasonality -> Trend -> Volatility -> Volume
   - Volume tiers: 5 levels (very_high/high/medium/low/very_low based on percentile thresholds)
   - Compound labels: `high_volume_seasonal_growing`, `low_volume_intermittent`, `medium_volume_periodic`, etc.
   - Two-pass disambiguation: base labels first, then secondary features resolve duplicates
4. **Assignment Update** (`update_cluster_assignments.py`):
   - Updates `dim_dfu.cluster_assignment` column in PostgreSQL
   - Validates updates and reports cluster distribution
5. **LGBM Backtest** (`run_backtest.py` → `common/backtest_framework.py` — Feature 44):
   - Uses shared `run_tree_backtest()` orchestrator from `common/backtest_framework.py`
   - Script implements both `train_and_predict_per_cluster()` and `train_and_predict_global()`; selects based on `cluster_strategy` config key
   - `ml_cluster` is always a hard feature (never stripped from feature_cols)
   - Algorithm options read from `config/algorithm_config.yaml` (cluster_strategy, recursive, shap_select, tune_inline, params_file, hyperparams)
   - Shared feature engineering from `common/feature_engineering.py`: lag 1-12, rolling mean/std 3/6/12m, calendar, DFU/item attributes
   - Default model IDs: `lgbm_cluster` (per_cluster) or `lgbm_global` (global)
   - Outputs two CSVs: execution-lag only (main table) + all lags 0-4 (archive)
   - Deduplication across timeframes (latest timeframe wins)
   - MLflow logging via `common/mlflow_utils.py` to `demand_backtest` experiment
6. **CatBoost Backtest** (`run_backtest_catboost.py` → `common/backtest_framework.py` — Feature 44):
   - Uses shared `run_tree_backtest()` orchestrator with `cat_dtype="str"` for CatBoost's index-based categoricals
   - Script implements both `train_and_predict_per_cluster()` and `train_and_predict_global()`
   - `ml_cluster` always a hard feature; `cluster_strategy` config key selects mode
   - Algorithm options read from `config/algorithm_config.yaml`
   - Default model IDs: `catboost_cluster` (per_cluster) or `catboost_global` (global)
7. **XGBoost Backtest** (`run_backtest_xgboost.py` → `common/backtest_framework.py` — Feature 44):
   - Uses shared `run_tree_backtest()` orchestrator with `cat_dtype="category"` for XGBoost's native categoricals
   - Script implements both `train_and_predict_per_cluster()` and `train_and_predict_global()`
   - `ml_cluster` always a hard feature; `cluster_strategy` config key selects mode
   - Algorithm options read from `config/algorithm_config.yaml`
   - Default model IDs: `xgboost_cluster` (per_cluster) or `xgboost_global` (global)
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
   - API: 4 read-only endpoints (models list, summary, timeframes, per-timeframe detail) under `/forecast/shap/` served from CSVs (no DB queries), plus **1 on-demand compute endpoint** `GET /forecast/shap/{model_id}/dfu?item_no=&loc=&top_n=` that loads persisted pkl from `data/models/{model_id}/cluster_{ml_cluster}.pkl`, rebuilds the exact feature matrix (lags 1–12, rolling mean/std with ddof=1, calendar, categoricals, item numerics), runs SHAP, and returns per-month signed contributions for both historical and future production-forecast months — all via `api/routers/shap.py`
   - Frontend: collapsible "Feature Importance (SHAP)" panel in Accuracy tab; indigo=selected / gray=dropped bar chart; **per-DFU interactive SHAP panel** (`DfuShapPanel.tsx`) in DFU Analysis tab
   - Config keys in `config/algorithm_config.yaml`: `shap_select`, `shap_top_n`, `shap_threshold`, `shap_sample_size`; composable with `tune_inline` and `params_file` (Feature 44)
   - Activated by setting `shap_select: true` in the algorithm section; run via `make backtest-lgbm`, `make backtest-catboost`, or `make backtest-xgboost`
   - Graceful degradation: SHAP failures log warning and keep all features; backtest continues uninterrupted
16. **Recursive Multi-Step Inference** (`common/backtest_framework.py` + `common/feature_engineering.py` — Feature 43):
   - `--recursive` CLI flag on LGBM, CatBoost, and XGBoost backtest scripts; passes `recursive=True` to `run_tree_backtest()`
   - In direct mode (default), months 2+ of the prediction window use `qty_lag_1 = 0` (masked sales). In recursive mode, each predict month is scored individually, and the model's prediction for month T is written back via `update_grid_with_predictions()` before scoring month T+1
   - `update_grid_with_predictions(grid, month, predictions)` in `common/feature_engineering.py`: writes predicted `basefcst_pref` to `qty[month]` then recomputes all lag (1-12) and rolling (3m/6m/12m) features in a single vectorized `groupby().shift()` pass
   - `_predict_single_month(models, data, feature_cols)` in `common/backtest_framework.py`: routes one month's batch to the correct cluster model dict (per-cluster) or single model (global) without retraining
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

25. EOQ & Cycle Stock Calculator (IPfeature4):
   - Wilson EOQ formula with MOQ rounding and months-supply cap
   - `fact_eoq_targets` table stores computed per-DFU EOQ metrics
   - Config: `config/eoq_config.yaml` (ordering_cost, holding_cost_pct, moq, max_eoq_months_supply)
   - Script: `scripts/compute_eoq.py` reads from `agg_inventory_monthly`, writes to `fact_eoq_targets`
   - 3 API endpoints: `GET /inv-planning/eoq/summary`, `GET /inv-planning/eoq/detail`, `GET /inv-planning/eoq/sensitivity`
   - Frontend: InvPlanningTab with KPI cards (avg EOQ, total cycle stock, avg order frequency, total annual cost), sensitivity curve chart, paginated detail table
   - Makefile: `eoq-schema`, `eoq-compute`, `eoq-all`
26. Replenishment Policy Management (IPfeature5):
   - 4 default policies in `config/replenishment_policy_config.yaml`: A-Class Continuous Review (ROP/EOQ), B/C Periodic Review, Lumpy/Intermittent Manual Review, Emergency/Critical Parts
   - DDL: `dim_replenishment_policy` + `fact_dfu_policy_assignment` (sql/025)
   - Script: `scripts/assign_replenishment_policies.py` — upsert policies from config + auto-assign DFUs by segment (--dry-run, --force-overwrite)
   - 5 API endpoints: `GET /inv-planning/policies`, `POST /inv-planning/policies`, `PUT /inv-planning/policies/{id}`, `GET /inv-planning/policy-assignments/compliance`, `POST /inv-planning/policy-assignments/assign`
   - Frontend: Policy Management panel in InvPlanningTab — policy cards with service level/type/config badges, ring gauge for DFU coverage, auto-assign button, compliance table, edit modal
   - Makefile: `policy-schema`, `policy-assign`, `policy-all`
27. Inventory Health Score Dashboard (IPfeature6):
   - Composite 0–100 health score per DFU from 4 components (each 0–25 pts)
   - SQL scoring via CTEs in `mv_inventory_health_score` materialized view: latest_inv, recent_stockout, recent_accuracy, ss (LEFT JOIN stub), scored
   - Stub pattern: `fact_safety_stock_targets` created empty; SS components use neutral scores (12/15) until IPfeature3 populates it
   - 3 API endpoints using `get_conn()` directly (NOT `Depends(_get_pool)` — avoids 422 MagicMock signature issue in tests): `GET /inv-planning/health/summary`, `GET /inv-planning/health/detail`, `GET /inv-planning/health/heatmap`
   - Frontend: Portfolio Health panel at top of InvPlanningTab — 4 clickable tier KPI cards, health distribution donut chart, component score progress bars, ABC×variability heatmap table, paginated detail table with severity badges
   - Makefile: `health-schema`, `health-refresh`, `health-all`
28. Exception Queue & Replenishment Recommendations (IPfeature7):
   - Automated exception detection from `agg_inventory_monthly` + policy assignments + safety stock (stub fallback)
   - 6 exception types: `stockout` (qty≤0, critical), `below_ss` (below safety stock, critical if <50% coverage, else high), `below_rop` (below reorder point, high), `excess` (DOS>1.5×target_max, medium/low), `zero_velocity` (qty>0, no sales, low)
   - Recommendation formula: `max(effective_eoq, gap + eoq/2)` capped at `max_eoq_months_supply × demand`; order_by = TODAY (critical), TODAY+review_cycle (high/medium)
   - Deduplication: skip if same item_no+loc+exception_type open within last 7 days
   - DDL: `fact_replenishment_exceptions` with 6 indexes including partial index on open+critical (sql/027)
   - Script: `scripts/generate_replenishment_exceptions.py` — pure-function detection/recommendation + DB write with --dry-run support
   - 5 API endpoints using `get_conn()` directly: `GET /inv-planning/exceptions` (paginated, filterable by type/severity/status/item/loc), `GET /inv-planning/exceptions/summary`, `PUT /inv-planning/exceptions/{id}/acknowledge` (auth), `PUT /inv-planning/exceptions/{id}/status` (auth), `POST /inv-planning/exceptions/generate` (auth)
   - Frontend: Exception Queue panel at top of InvPlanningTab — 4 KPI cards (Total Open, Critical, High, Rec. Order Value), type/severity filter pills, status toggle, item/loc filter inputs, exception table with inline action buttons (Acknowledge/Mark Ordered/Resolve), row background coloring by severity
   - Makefile: `exceptions-schema`, `exceptions-generate`, `exceptions-generate-dry`
29. Fill Rate Analytics (IPfeature8):
   - Order fill rate metrics aggregated from inventory snapshot data
   - DDL: `sql/028_create_fill_rate_monthly.sql` — `mv_fill_rate_monthly` materialized view
   - Router: `api/routers/fill_rate.py` — 3 endpoints: `GET /fill-rate/summary`, `GET /fill-rate/trend`, `GET /fill-rate/detail`
   - Frontend: FillRatePanel in InvPlanningTab
   - Makefile: `fill-rate-schema`, `fill-rate-refresh`, `fill-rate-all`
   - Tests: `tests/api/test_fill_rate.py`
30. Demand Sensing & Short-Horizon Signal Integration (IPfeature9):
   - Short-horizon demand signals computed from recent sales velocity and inventory movement patterns
   - DDL: `sql/029_create_demand_signals.sql` — `fact_demand_signals` table
   - Script: `scripts/compute_demand_signals.py`
   - 3 API endpoints in `api/routers/inv_planning.py`: `GET /inv-planning/demand-signals/summary`, `/list`, `/item`
   - Makefile: `demand-signals-schema`, `demand-signals-compute`, `demand-signals-all`
   - Tests: `tests/api/test_inv_planning_demand_signals.py`, `tests/unit/test_demand_signals.py`
31. Safety Stock Monte Carlo Simulation (IPfeature10):
   - Probabilistic safety stock simulation using configurable number of iterations and seed
   - DDL: `sql/030_create_ss_simulation_results.sql` — `fact_ss_simulation_results` table
   - Script: `scripts/run_ss_simulation.py`
   - Config: `config/simulation_config.yaml` (n_simulations, random_seed)
   - 3 API endpoints: `POST /inv-planning/simulation/run`, `GET /inv-planning/simulation/results`, `GET /inv-planning/simulation/compare`, `GET /inv-planning/simulation/{id}/status`
   - Makefile: `sim-schema`, `sim-run`
   - Tests: `tests/api/test_inv_planning_simulation.py`
32. ABC-XYZ Policy Matrix (IPfeature11):
   - Combined ABC volume segmentation × XYZ demand variability classification into 3×3 policy matrix
   - DDL: `sql/031_add_xyz_classification.sql` — XYZ classification columns on DFU dimension
   - Script: `scripts/classify_abc_xyz.py`
   - 3 API endpoints: `GET /inv-planning/abc-xyz/matrix`, `/summary`, `/detail`
   - Frontend: AbcXyzPanel in InvPlanningTab
   - Makefile: `abc-xyz-schema`, `abc-xyz-classify`, `abc-xyz-all`
   - Tests: `tests/api/test_inv_planning_abc_xyz.py`, `tests/unit/test_abc_xyz_classification.py`
33. Supplier Performance Analytics (IPfeature12):
   - Supplier delivery performance KPIs aggregated from inventory receipt data
   - DDL: `sql/032_create_supplier_performance.sql` — `mv_supplier_performance` materialized view
   - 3 API endpoints: `GET /inv-planning/supplier-performance/summary`, `/detail`, `/items`
   - Frontend: SupplierPanel in InvPlanningTab
   - Makefile: `supplier-perf-schema`, `supplier-perf-refresh`, `supplier-perf-all`
   - Tests: `tests/api/test_inv_planning_supplier.py`
34. Capital Investment Optimization (IPfeature13):
   - Portfolio-level inventory investment planning with efficient frontier computation
   - DDL: `sql/033_create_investment_plan.sql` — `fact_inventory_investment_plan` + `fact_efficient_frontier` tables
   - Script: `scripts/compute_investment_plan.py`
   - 4 API endpoints: `GET /inv-planning/investment/efficient-frontier`, `/summary`, `/detail`, `POST /inv-planning/investment/plan`
   - Makefile: `investment-schema`, `investment-plan`, `investment-all`
   - Tests: `tests/api/test_inv_planning_investment.py`, `tests/unit/test_investment_plan.py`
35. Intra-Month Stockout Detection (IPfeature14):
   - Daily inventory scan detects within-month stockout events before end-of-month snapshot
   - DDL: `sql/034_create_intramonth_stockout.sql` — `mv_intramonth_stockout` materialized view
   - Script: `scripts/refresh_intramonth_stockout.py`
   - 3 API endpoints: `GET /inv-planning/intramonth-stockouts/summary`, `/detail`, `/daily`
   - Frontend: IntramonthPanel in InvPlanningTab
   - Makefile: `intramonth-schema`, `intramonth-refresh`, `intramonth-all`
   - Tests: `tests/api/test_inv_planning_intramonth.py`
36. Unified Control Tower / Command Center (IPfeature15):
   - Single-pane-of-glass operational dashboard aggregating KPIs, alerts, critical items, and trends
   - DDL: `sql/035_create_control_tower_kpis.sql` — `mv_control_tower_kpis` materialized view
   - Router: `api/routers/control_tower.py` — 4 endpoints: `GET /control-tower/kpis`, `/alerts`, `/top-critical`, `/trend`
   - Frontend: `frontend/src/tabs/ControlTowerTab.tsx` — dedicated Control Tower tab registered in App.tsx and AppSidebar
   - Router registered in `api/main.py`; Vite proxy entry `/control-tower` added to `frontend/vite.config.ts`
   - Makefile: `control-tower-schema`, `control-tower-refresh`, `control-tower-all`
   - Tests: `tests/api/test_control_tower.py`, `src/tabs/__tests__/ControlTowerTab.test.tsx`
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

37. AI Planning Agent (IPAIfeature1):
   - Claude `tool_use` agent (`common/ai_planner.py`) with `AIPlannerAgent` class and 10 tools
   - 9 read-only PostgreSQL tools (get_dfu_full_context, get_forecast_performance, get_portfolio_exceptions, compute_bias_trend, get_inventory_trend, get_eoq_context, get_similar_dfus, check_stockout_history, get_portfolio_health_summary) + 1 write tool (create_insight)
   - Agent calls Claude `claude-opus-4-6` via Anthropic SDK; tool_use loop dispatches to Python handlers with real DB queries
   - DDL: `sql/036_create_ai_insights.sql` — `ai_insights` table (5 insight types, 4 severities, open/acknowledged/resolved workflow) + `ai_planning_memos` table (portfolio/DFU scope narratives)
   - Config: `config/ai_planner_config.yaml` (model, thresholds, severity rules, portfolio_scan_limit, schedule)
   - Script: `scripts/generate_ai_insights.py` — CLI batch scan (`--portfolio`, `--item`/`--loc`, `--dry-run`)
   - 5 API endpoints in `api/routers/ai_planner.py`: `POST /ai-planner/analyze` (single DFU, synchronous), `POST /ai-planner/portfolio-scan` (202 background), `GET /ai-planner/insights` (paginated, filterable), `PUT /ai-planner/insights/{id}/status`, `GET /ai-planner/memos`
   - Frontend: `AIPlannerTab.tsx` — insight cards with severity badges, financial impact chips, causal reasoning, acknowledge/resolve actions; portfolio health KPI bar; planning memo markdown panel
   - Types: `frontend/src/types/ai-planner.ts` — AiInsight, AiPlanningMemo, InsightSeverity, InsightStatus, InsightType
   - New UI component: `frontend/src/components/ui/select.tsx` — minimal React Context-based Select wrapper matching shadcn/ui API
   - "AI Planner" nav item (14th item) + "aiPlanner" added to VALID_TABS (13 total) + Vite proxy `/ai-planner`
   - Dependency: `anthropic>=0.40.0`
   - Makefile: `ai-insights-schema`, `ai-insights-scan`, `ai-insights-dfu`, `ai-insights-all`
   - Tests: 18 backend unit, 10 API, 7 frontend

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
| `test_shap_dfu.py` | `api/routers/shap.py` — per-DFU SHAP endpoint (200 lgbm, no model dir, DFU not found, pkl missing, top_n clamp, catboost path) | 6 |
| `test_backtest_recursive.py` | `common/backtest_framework.py` — `_fill_predict_nans`, `_predict_single_month` (global/cluster/transfer), recursive loop integration | 13 |
| `test_eoq.py` | `scripts/compute_eoq.py` — Wilson EOQ formula, effective EOQ with MOQ+cap, sensitivity curve | 23 |
| `test_inv_planning_eoq.py` | `api/routers/inv_planning.py` — EOQ summary, detail, sensitivity endpoints | 10 |
| `test_replenishment_policy.py` | Policy assignment logic — auto-assign rules, dry-run, force-overwrite, compliance calculation | 18 |
| `test_inv_planning_policy.py` | `api/routers/inv_planning.py` — policy CRUD, assign, compliance endpoints | 13 |
| `test_health_score.py` | Pure-Python replicas of SQL scoring logic — SS coverage, DOS target, stockout risk, forecast accuracy, composite, health tier | 42 |
| `test_inv_planning_health.py` | `api/routers/inv_planning.py` — health summary, detail, heatmap endpoints | 12 |
| `test_exception_generation.py` | `scripts/generate_replenishment_exceptions.py` — detect_exception_type, compute_recommendation pure functions | 19 |
| `test_inv_planning_exceptions.py` | `api/routers/inv_planning.py` — exception list, summary, acknowledge, status, generate endpoints | 13 |
| `test_fill_rate.py` | `api/routers/fill_rate.py` — fill rate summary, trend, detail endpoints | varies |
| `test_inv_planning_demand_signals.py` | `api/routers/inv_planning.py` — demand signals summary, list, item endpoints | varies |
| `test_demand_signals.py` | `scripts/compute_demand_signals.py` — demand signal computation pure functions | varies |
| `test_inv_planning_simulation.py` | `api/routers/inv_planning.py` — simulation run, results, compare, status endpoints | varies |
| `test_inv_planning_abc_xyz.py` | `api/routers/inv_planning.py` — ABC-XYZ matrix, summary, detail endpoints | varies |
| `test_abc_xyz_classification.py` | `scripts/classify_abc_xyz.py` — ABC-XYZ classification logic | varies |
| `test_inv_planning_supplier.py` | `api/routers/inv_planning.py` — supplier performance summary, detail, items endpoints | varies |
| `test_inv_planning_investment.py` | `api/routers/inv_planning.py` — investment efficient-frontier, summary, detail, plan endpoints | varies |
| `test_investment_plan.py` | `scripts/compute_investment_plan.py` — investment plan computation logic | varies |
| `test_inv_planning_intramonth.py` | `api/routers/inv_planning.py` — intramonth stockout summary, detail, daily endpoints | varies |
| `test_control_tower.py` | `api/routers/control_tower.py` — control tower kpis, alerts, top-critical, trend endpoints | varies |
| `test_ai_planner.py` | `common/ai_planner.py` — tool functions, agent loop, dry-run mode | 18 |
| `test_ai_planner_api.py` | `api/routers/ai_planner.py` — insights CRUD, portfolio scan 202, memo list | 10 |

**Total backend: 1085 tests**

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
| `DfuShapPanel.test.tsx` | Per-DFU SHAP stacked bar chart panel (placeholder, non-item_location note, skeleton, success chart, 404 fallback) | 5 |
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
| `InvPlanningTab.test.tsx` | Inventory Planning tab — EOQ KPIs, sensitivity, detail, policy management, health score, exception queue | 21 |
| `ControlTowerTab.test.tsx` | Control Tower tab — KPI cards, alerts panel, top-critical list, trend chart | varies |
| `AIPlannerTab.test.tsx` | AI Planner tab — insight cards, severity badges, generate button, acknowledge action | 7 |

**Total frontend: 372 tests**

**Combined total: 1457 tests.** `make test-all` runs backend + frontend.

**Mandatory rule:** Every new feature, endpoint, component, or utility must include corresponding tests. See `docs/design-specs/feature31.md` for the full testing strategy.

## Frontend Component Architecture

### Tab Panel Subfolder Pattern
Large tab files were refactored into shell + panel subfolder pattern for maintainability. The shell file (200–250L) handles layout and state; all panels are extracted into a co-located subfolder:

| Tab Shell | Subfolder | Extracted Panels |
|-----------|-----------|-----------------|
| `tabs/AccuracyTab.tsx` (224L) | `tabs/accuracy/` | KpiSection, TrendChartPanel, SliceTablePanel, ChampionPanel, ShapPanel |
| `tabs/DfuAnalysisTab.tsx` | `tabs/dfu-analysis/` | SelectorPanel, OverlayChartPanel, ModelKpiSection, DfuShapPanel |
| `tabs/InventoryTab.tsx` (222L) | `tabs/inventory/` | KpiSection, TrendChartPanel, PositionTablePanel, ItemDetailPanel, DemandVariabilityPanel, LeadTimeProfilePanel |
| `tabs/JobsTab.tsx` (202L) | `tabs/jobs/` | KpiSection, JobGroupsPanel, ActiveJobsPanel, SchedulesPanel, JobHistoryPanel, jobsShared.ts |
| `tabs/ClustersTab.tsx` (224L) | `tabs/clusters/` | ClusterOverviewPanel, WhatIfPanel, ScenarioResultsPanel, PastScenariosPanel |
| `tabs/InvPlanningTab.tsx` | `tabs/inv-planning/` | Two-column layout: fixed 220px grouped sidebar navigation (7 groups with colored dividers, icons, and labels — Daily Operations, Optimize, Analytics, Planning, Sensing, Strategic, Supply) + scrollable main content area with per-panel header bar (title + description). 26 panels: ExceptionQueuePanel, PortfolioHealthPanel, EoqPanel, PolicyManagementPanel, FillRatePanel, AbcXyzPanel, SupplierPanel, IntramonthPanel, SafetyStockPanel, VariabilityPanel, LeadTimePanel, DemandSignalsPanel, SimulationPanel, InvestmentPanel, ReplenishmentPlanPanel, DemandForecastPanel, BlendedDemandPanel, EchelonPanel, FinancialPlanPanel, EventCalendarPanel, ScenarioPlanningPanel, and Supply group panels |

### API Query Sub-modules
`frontend/src/api/queries.ts` is a thin re-export barrel (`export * from "./queries/index"`). All domain query modules live under `frontend/src/api/queries/`:

| Module | Purpose |
|--------|---------|
| `queries/core.ts` | Core forecast, accuracy, domain queries |
| `queries/inv-planning.ts` | Barrel re-exporting all 9 inv-planning sub-modules |
| `queries/inv-planning-eoq.ts` | EOQ query keys + fetch functions (IPfeature4) |
| `queries/inv-planning-policy.ts` | Policy CRUD query keys + fetch/mutate functions (IPfeature5) |
| `queries/inv-planning-health.ts` | Health score query keys + fetch functions (IPfeature6) |
| `queries/inv-planning-exceptions.ts` | Exception queue query keys + fetch/mutate functions (IPfeature7) |
| `queries/inv-planning-safety-stock.ts` | Safety stock query keys + fetch functions (IPfeature3) |
| `queries/inv-planning-signals.ts` | Demand signals query keys + fetch functions (IPfeature9) |
| `queries/inv-planning-abc.ts` | ABC-XYZ query keys + fetch functions (IPfeature11) |
| `queries/inv-planning-supplier.ts` | Supplier performance query keys + fetch functions (IPfeature12) |
| `queries/inv-planning-intramonth.ts` | Intramonth stockout query keys + fetch functions (IPfeature14) |
| `queries/ai-planner.ts` | AI planner query keys + fetch/mutate functions (IPAIfeature1) |
| `queries/control-tower.ts` | Control Tower query keys + fetch functions (IPfeature15) |
| `queries/fill-rate.ts` | Fill rate query keys + fetch functions (IPfeature8) |
| `queries/storyboard.ts` | Storyboard exception query keys + fetch/mutate functions (Feature 40) |

## How to add next dataset
1. Add `<DATASET>_SPEC` in `common/domain_specs.py`
2. Add matching DDL in `sql/`
3. Add Make targets:
   - `normalize-<dataset>`
   - `load-<dataset>`
4. For generic dimension/fact domains: API uses existing generic `/domains/{domain}/...` endpoints
5. For specialized domains (like inventory): Add dedicated API endpoints and frontend tab
6. Run `make generate-embeddings` to update chat context with new schema metadata

**Example: Inventory domain** — uses dedicated normalize script (`normalize_inventory_csv.py`) for multi-file merge, dedicated API endpoints (`/inventory/*`), and dedicated UI tab (`InventoryTab`). Domain spec still lives in `common/domain_specs.py` for schema metadata.
