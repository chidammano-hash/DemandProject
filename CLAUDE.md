# CLAUDE.md — Demand Studio

## Project Overview

**Demand Studio** is a unified demand forecasting analytics platform. It ingests sales and forecast data, stores it in PostgreSQL (OLTP) and Apache Iceberg (lakehouse), and serves a React UI for interactive analytics.

**Working directory for all dev work:** `mvp/demand/`

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
| Lakehouse | Apache Iceberg via MinIO + Iceberg REST |
| Big Data | Apache Spark 3.5 |
| Query Engine | Trino |
| ML / Clustering | scikit-learn, pandas, scipy, matplotlib, seaborn |
| ML Tracking | MLflow |
| Job Scheduling | APScheduler 3.11 (BackgroundScheduler + ThreadPoolExecutor) |
| Python packaging | uv |
| Build | Make |
| Containers | Docker Compose |

---

## Key Files

| File | Purpose |
|---|---|
| `mvp/demand/common/domain_specs.py` | Central config: all 8 datasets (dimensions + facts + inventory) with columns, types, keys |
| `mvp/demand/api/main.py` | FastAPI backend — primary endpoints + mounts routers (clusters, jobs, etc.) |
| `mvp/demand/frontend/src/App.tsx` | React UI — sidebar layout shell (~200 lines, lazy-loaded tabs) |
| `mvp/demand/Makefile` | All dev commands |
| `mvp/demand/docker-compose.yml` | 7-service infra cluster |
| `mvp/demand/scripts/normalize_dataset_csv.py` | Generic ETL: CSV → clean CSV |
| `mvp/demand/scripts/load_dataset_postgres.py` | Generic loader: clean CSV → PostgreSQL |
| `mvp/demand/scripts/spark_dataset_to_iceberg.py` | Spark job: clean CSV → Iceberg |
| `mvp/demand/sql/` | DDL for all tables, indexes, materialized views |
| `mvp/demand/sql/017_create_fact_inventory_snapshot.sql` | Inventory snapshot table DDL, indexes, materialized view |
| `mvp/demand/sql/019_inventory_forecast_view.sql` | Inventory-forecast bridge materialized view (Feature 37) |
| `mvp/demand/scripts/normalize_inventory_csv.py` | Inventory ETL: merge 14 monthly CSVs → single clean CSV |
| `mvp/demand/scripts/generate_clustering_features.py` | Feature engineering: sales history → clustering feature matrix |
| `mvp/demand/scripts/train_clustering_model.py` | KMeans clustering with optimal K selection + MLflow logging |
| `mvp/demand/scripts/label_clusters.py` | Assign business labels to clusters based on feature centroids |
| `mvp/demand/scripts/update_cluster_assignments.py` | Write cluster labels to `dim_dfu.cluster_assignment` in Postgres |
| `mvp/demand/config/clustering_config.yaml` | Clustering hyperparameters and labeling thresholds |
| `mvp/demand/config/model_competition.yaml` | Champion model selection: competing models, metric, lag, strategy |
| `mvp/demand/scripts/run_champion_selection.py` | Per-DFU champion selection: best-of-models via configurable strategy |
| `mvp/demand/common/champion_strategies.py` | 5 champion strategies (expanding, rolling, decay, ensemble, meta_learner) + registry + leak guards |
| `mvp/demand/scripts/simulate_champion_strategies.py` | Diagnostic: simulate all strategies, compare accuracy vs ceiling |
| `mvp/demand/scripts/train_meta_learner.py` | Meta-learner training: ceiling labels as ground truth, temporal split |
| `mvp/demand/common/backtest_framework.py` | Shared backtest orchestrator: `run_tree_backtest()` (per-cluster only), timeframes, data loading, output saving; `_fill_predict_nans()`, `_predict_single_month(models, predict_data, feature_cols)`, `recursive` param (Feature 43); config-driven via algorithm_config.yaml (Feature 44) |
| `mvp/demand/common/feature_engineering.py` | Shared feature matrix: lag/rolling features, future masking, `cat_dtype` parameter; `update_grid_with_predictions()` for recursive inference write-back (Feature 43) |
| `mvp/demand/common/metrics.py` | Shared accuracy metrics: WAPE, bias, accuracy % |
| `mvp/demand/common/mlflow_utils.py` | Shared MLflow logging wrapper for backtest runs |
| `mvp/demand/common/db.py` | Shared DB connection parameters |
| `mvp/demand/common/constants.py` | Shared constants: `CAT_FEATURES`, `LAG_RANGE`, `ROLLING_WINDOWS`, output columns, thresholds |
| `mvp/demand/config/algorithm_config.yaml` | Per-algorithm config: recursive, shap_select, shap_threshold, shap_top_n, shap_sample_size, tune_inline, params_file, default hyperparameters (Feature 44) |
| `mvp/demand/scripts/run_backtest.py` | LGBM backtest: per-cluster training function; reads config/algorithm_config.yaml (Feature 44) |
| `mvp/demand/scripts/run_backtest_catboost.py` | CatBoost backtest: per-cluster training function; reads config/algorithm_config.yaml (Feature 44) |
| `mvp/demand/scripts/run_backtest_xgboost.py` | XGBoost backtest: per-cluster training function; reads config/algorithm_config.yaml (Feature 44) |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load backtest predictions into Postgres (main + archive); supports `--model MODEL_ID`, `--all` (scan all subdirs), `--input PATH` |
| `mvp/demand/scripts/tune_hyperparams.py` | Bayesian hyperparameter tuning via Optuna: walk-forward CV, TPE sampler, early stopping, outputs `data/tuning/best_params_<model>.json` |
| `mvp/demand/common/tuning.py` | Shared tuning utilities: `generate_cv_month_splits`, `compute_wape_stabilised`, `suggest_params`, `save_best_params`, `load_best_params`, `best_rounds_to_n_estimators`, `tune_for_timeframe()` (per-timeframe causal tuning, PL-002), `TRAIN_FOLD_FNS` registry (`train_lgbm_fold`, `train_catboost_fold`, `train_xgboost_fold`) |
| `mvp/demand/config/hyperparameter_tuning.yaml` | Search spaces (LGBM 8 params, CatBoost 5 params, XGBoost 8 params), CV settings, trial budget, pruner config |
| `mvp/demand/common/shap_selector.py` | SHAP computation, cumulative-importance feature selection, cluster-pooled SHAP, CSV output helpers (Feature 42) |
| `mvp/demand/api/routers/shap.py` | SHAP REST API: 4 endpoints — models list, cross-timeframe summary, timeframes list, per-timeframe detail (Feature 42) |
| `mvp/demand/frontend/src/types/shap.ts` | TypeScript types for SHAP API responses: ShapFeatureSummary, ShapFeatureDetail, ShapTimeframeEntry, payload envelopes (Feature 42) |
| `mvp/demand/scripts/clean_backtest_models.py` | Selective cleanup of model predictions from Postgres + view refresh |
| `mvp/demand/scripts/clean_forecasts_by_date.py` | Date-range forecast cleanup: delete rows by time bucket from forecast + archive tables |
| `mvp/demand/sql/010_create_backtest_lag_archive.sql` | DDL for backtest all-lags archive table |
| `mvp/demand/sql/008_perf_indexes_and_agg.sql` | Performance indexes (B-tree, GIN trigram) + materialized views |
| `mvp/demand/frontend/src/api/queries.ts` | Centralized TanStack Query layer (all fetch functions + query keys) |
| `mvp/demand/frontend/src/tabs/` | Extracted tab components (DashboardTab, ExplorerTab, AccuracyTab, DfuAnalysisTab, ClustersTab, MarketIntelTab, InvBacktestTab, ChatPanel, JobsTab, InvPlanningTab) |
| `mvp/demand/frontend/src/hooks/useTheme.ts` | Color mode management (light/dark) for the General theme |
| `mvp/demand/frontend/src/hooks/useUrlState.ts` | URL state synchronization (12 tabs, overview default) |
| `mvp/demand/frontend/src/hooks/useKeyboardShortcuts.ts` | Keyboard shortcuts handler (1-9 tabs, sidebar, dark mode) |
| `mvp/demand/frontend/src/hooks/useSidebar.ts` | Sidebar collapse/expand state + mobile drawer |
| `mvp/demand/frontend/src/hooks/useGlobalFilters.ts` | Global filter state with debounced URL sync |
| `mvp/demand/frontend/src/context/GlobalFilterContext.tsx` | Global filter React context provider |
| `mvp/demand/frontend/src/context/ScenarioNotificationContext.tsx` | Cross-tab scenario notification context (Feature 38) |
| `mvp/demand/common/job_registry.py` | APScheduler-powered job engine: BackgroundScheduler, JobManager singleton, per-group concurrency, scheduling, pipelines (Feature 39) |
| `mvp/demand/api/routers/jobs.py` | Jobs REST API: 12 endpoints — CRUD, scheduling, pipelines, stats (Feature 39) |
| `mvp/demand/sql/020_create_job_history.sql` | DDL for persistent job_history table (Feature 39) |
| `mvp/demand/sql/021_alter_job_history_scheduling.sql` | DDL for scheduling columns + job_schedule table (Feature 39) |
| `mvp/demand/frontend/src/tabs/JobsTab.tsx` | Automation dashboard UI: KPI cards, grouped job cards, schedules, history (Feature 39) |
| `mvp/demand/frontend/src/types/jobs.ts` | TypeScript types: JobStats, JobSchedule, GROUP_CONFIG (Feature 39) |
| `mvp/demand/frontend/src/context/JobNotificationContext.tsx` | Cross-tab job notification context (Feature 39) |
| `mvp/demand/frontend/src/types/theme.ts` | TypeScript types for theme, sidebar, filters, dashboard |
| `mvp/demand/frontend/src/constants/themes/general.ts` | Single professional theme config (Demand Studio, light + dark) |
| `mvp/demand/frontend/src/components/AppSidebar.tsx` | Collapsible sidebar navigation (12 items, 5 sections) |
| `mvp/demand/frontend/src/components/ThemeSelector.tsx` | Light/dark mode toggle (sidebar footer) |
| `mvp/demand/frontend/src/components/GlobalFilterBar.tsx` | Cross-tab filter bar (brand, category, item, location, market, channel) |
| `mvp/demand/frontend/src/components/WidgetGrid.tsx` | CSS Grid dashboard layout (WidgetGrid + WidgetCard) |
| `mvp/demand/frontend/src/components/AlertPanel.tsx` | Severity-coded alert list |
| `mvp/demand/frontend/src/components/HeatmapGrid.tsx` | CSS Grid heatmap with color scale |
| `mvp/demand/frontend/src/components/TopMovers.tsx` | Period-over-period top movers list |
| `mvp/demand/frontend/src/components/ForecastTrendChart.tsx` | ECharts forecast vs actual trend chart |
| `mvp/demand/sql/018_dashboard_views.sql` | Materialized view for top movers |
| `mvp/demand/frontend/src/lib/formatters.ts` | Number/cell formatting utilities |
| `mvp/demand/frontend/src/lib/export.ts` | CSV export utility (papaparse) |
| `mvp/demand/frontend/src/components/DataTable.tsx` | Virtualized data grid (TanStack Table + Virtual) |
| `mvp/demand/frontend/src/components/Skeleton.tsx` | Loading skeleton placeholder |
| `mvp/demand/frontend/src/components/EChartContainer.tsx` | Theme-aware ECharts wrapper |
| `mvp/demand/frontend/vite.config.ts` | Vite dev server config: API proxy rules (CRITICAL — every new API path prefix must be added here) |
| `mvp/demand/frontend/vitest.config.ts` | Vitest test configuration |
| `mvp/demand/frontend/tailwind.config.ts` | Tailwind config with custom `pulse-glow` animation |
| `mvp/demand/tests/` | Backend test suite (pytest): unit/ + api/ |
| `mvp/demand/tests/conftest.py` | Shared pytest fixtures (sample DataFrames) |
| `mvp/demand/tests/api/conftest.py` | API test fixtures (mock DB pool, async httpx client) |
| `mvp/demand/frontend/src/**/__tests__/` | Frontend test suites (Vitest + RTL) |
| `docs/architecture-diagram.md` | Full-stack architecture diagram (layers, data flow, ML pipeline) |
| `docs/design-specs/` | Feature specs (feature1–feature44) |
| `mvp/demand/api/core.py` | Shared API utilities: connection pool, OpenAI client, SQL helpers used by router modules |
| `mvp/demand/api/auth.py` | Optional API key auth (`require_api_key` dependency; disabled when `API_KEY` env var unset) |
| `mvp/demand/api/routers/` | Modular FastAPI router modules: 14 routers (accuracy, analysis, benchmark, chat, clusters, competition, dashboard, domains, intel, inv_backtest, inventory, inv_planning, jobs, shap) |
| `mvp/demand/api/routers/inventory.py` | Inventory endpoints: position, KPIs, trend, item-detail |
| `mvp/demand/api/routers/inv_backtest.py` | Inventory backtest endpoints: summary, trend, root-cause, detail |
| `mvp/demand/api/routers/dashboard.py` | Dashboard endpoints: KPIs, alerts, top-movers, heatmap |
| `mvp/demand/frontend/src/context/ThemeContext.tsx` | Theme React context provider (replaces theme prop-drilling) |
| `mvp/demand/frontend/src/hooks/useChartColors.ts` | Chart color hook: returns chartColors + trendColors from theme context |
| `mvp/demand/frontend/src/components/ScenarioCharts.tsx` | Extracted scenario visualization charts (elbow, silhouette, radar, pie, gap) |
| `mvp/demand/config/seasonality_config.yaml` | Seasonality detection hyperparameters and profile labeling thresholds |
| `mvp/demand/scripts/detect_seasonality.py` | Compute seasonality metrics per DFU (strength, profile, peak/trough month) |
| `mvp/demand/scripts/update_seasonality_profiles.py` | Write seasonality profiles to `dim_dfu` in Postgres |
| `mvp/demand/scripts/run_clustering_scenario.py` | What-If clustering: run trial KMeans with custom params + promote flow |
| `mvp/demand/sql/013_add_composite_indexes.sql` | Composite B-tree indexes for multi-column query performance |
| `mvp/demand/sql/015_add_seasonality_columns.sql` | DDL: 6 seasonality columns on `dim_dfu` (Feature 30) |
| `mvp/demand/sql/016_add_seasonality_to_accuracy_views.sql` | DDL: seasonality joins in accuracy materialized views (Feature 32) |
| `mvp/demand/frontend/src/hooks/useDebounce.ts` | Generic debounce hook used by filter inputs |
| `mvp/demand/frontend/src/constants/colors.ts` | Shared color palette constants |
| `mvp/demand/frontend/src/constants/elements.ts` | Periodic table element definitions for loading overlay |
| `mvp/demand/frontend/src/components/KeyboardShortcutHelp.tsx` | Keyboard shortcut help modal (triggered by `?` shortcut) |
| `mvp/demand/frontend/src/components/KpiCard.tsx` | Reusable KPI metric card component |
| `mvp/demand/frontend/src/lib/utils.ts` | Shared utilities: `cn()` Tailwind class merger and misc helpers |
| `mvp/demand/scripts/compute_eoq.py` | EOQ computation: `compute_eoq()`, `compute_effective_eoq()`, `compute_eoq_metrics()`, `sensitivity_curve()`, `run()` (IPfeature4) |
| `mvp/demand/config/eoq_config.yaml` | EOQ config: ordering_cost, holding_cost_pct, moq, max_eoq_months_supply (IPfeature4) |
| `mvp/demand/sql/024_create_eoq_targets.sql` | DDL for `fact_eoq_targets` table (IPfeature4) |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Inventory Planning tab: KPI cards, EOQ sensitivity chart, detail table (IPfeature4) |
| `mvp/demand/api/routers/inv_planning.py` | Inventory planning endpoints: EOQ summary, detail, sensitivity (IPfeature4) |

---

## Common Commands

```bash
# One-time setup
make init              # Create .venv, install uv, sync dependencies

# Infrastructure
make up                # Start Docker services (Postgres, MinIO, Spark, Trino, MLflow)
make down              # Stop all services
make db-apply-sql      # Apply DDL schemas to Postgres

# Data pipeline
make normalize-all     # Normalize all 8 datasets (CSV → clean CSV)
make load-all          # Load cleaned data into Postgres + refresh materialized views
make load-forecast-replace  # Reload external forecast only (preserves backtest data)
make load-forecast-replace-no-archive  # Reload external forecast, skip archive (fast)
make spark-all         # Publish datasets to Iceberg (optional)

# Inventory pipeline
make db-apply-inventory     # Create inventory table + indexes + materialized view (one-time)
make normalize-inventory    # Merge 14 monthly CSVs into single clean CSV
make load-inventory         # Load into Postgres + refresh agg view
make inventory-pipeline     # normalize + load + refresh (all-in-one)

# Inventory backtest (Feature 37)
make db-apply-inv-backtest  # Create mv_inventory_forecast_monthly materialized view
make refresh-inv-backtest   # Refresh inventory-forecast bridge view with current data

# Job scheduler (Feature 39)
make db-apply-jobs          # Create job_history + job_schedule tables + indexes

# Run services
make api               # Start FastAPI on :8000
make ui-init           # Install npm deps
make ui                # Start React dev server on :5173

# Validation
make check-db          # Table row counts in Postgres
make check-api         # Curl API health + sample endpoints
make check-all         # Full check: DB + API + Trino

# Chatbot
make db-apply-chat     # Apply pgvector + embeddings table DDL
make generate-embeddings  # Generate and store schema embeddings (requires OPENAI_API_KEY)

# Benchmarking
make bench-compare DOMAIN=sales RUNS=7 ITEM=100320 LOCATION=1401-BULK

# Clustering pipeline
make cluster-features  # Generate clustering feature matrix from sales/DFU/item data
make cluster-train     # Train KMeans, select optimal K, log to MLflow
make cluster-label     # Assign business labels to clusters
make cluster-update    # Write cluster labels to dim_dfu in Postgres
make cluster-all       # Run full clustering pipeline (features → train → label → update)

# Backtesting (Feature 44 — config-driven, per-cluster only)
# Edit config/algorithm_config.yaml to enable SHAP, recursive, tune_inline, params_file, etc.
make backtest-lgbm          # Run LGBM per-cluster backtest (reads config/algorithm_config.yaml)
make backtest-catboost      # Run CatBoost per-cluster backtest
make backtest-xgboost       # Run XGBoost per-cluster backtest
make backtest-all           # Run all three sequentially (LGBM → CatBoost → XGBoost)
make backtest-all-parallel  # Run all three in parallel (logs → data/backtest/logs/)

# Hyperparameter tuning (Feature 41)
make tune-lgbm              # Tune LGBM hyperparameters via Optuna (50 trials, ~20-40 min) → data/tuning/best_params_lgbm.json
make tune-catboost          # Tune CatBoost hyperparameters (~30-60 min) → data/tuning/best_params_catboost.json
make tune-xgboost           # Tune XGBoost hyperparameters (~25-50 min) → data/tuning/best_params_xgboost.json
make tune-all               # Run all three tuning jobs sequentially

# Backtest loading (shared across all models)
make backtest-load MODEL=lgbm_cluster   # Load one model from data/backtest/lgbm_cluster/
make backtest-load MODEL=catboost_cluster
make backtest-load MODEL=xgboost_cluster
make backtest-load-all      # Load ALL models found under data/backtest/*/

# Champion model selection
make champion-select        # Run per-DFU champion selection (best-of-models via configurable strategy)
make champion-simulate      # Simulate all strategies, compare accuracy vs ceiling
make champion-train-meta    # Train meta-learner classifier for champion selection
make champion-all           # train-meta + simulate + select (full pipeline)

# Seasonality pipeline (feature 30)
make seasonality-schema     # Apply DDL for seasonality columns on dim_dfu (one-time)
make seasonality-detect     # Detect seasonality patterns per DFU from sales history
make seasonality-update     # Write seasonality profiles back to dim_dfu
make seasonality-all        # Full pipeline: schema + detect + update

# EOQ & Inventory Planning (IPfeature4)
make eoq-schema             # Apply DDL for fact_eoq_targets table (one-time)
make eoq-compute            # Compute EOQ metrics from agg_inventory_monthly → fact_eoq_targets
make eoq-all                # eoq-schema + eoq-compute (full pipeline)

# Backtest cleanup
make backtest-list          # List model_id row counts in forecast + archive tables
make backtest-clean MODELS="lgbm_cluster catboost_cluster"  # Remove specific model predictions

# Forecast date-range cleanup
make forecast-clean-list                                             # List row counts by model + month
make forecast-clean ARGS="--before 2025-04-01 --model external"     # Delete external forecasts before Apr 2025
make forecast-clean ARGS="--between 2024-01-01 2024-07-01"          # Delete all models between Jan-Jun 2024
make forecast-clean ARGS="--months 2024-03 2024-06 2024-09"         # Delete specific months
make forecast-clean ARGS="--months 2025-01 --model external"        # Delete one month for one model
make forecast-clean ARGS="--before 2025-01-01 --dry-run"            # Preview without deleting
make forecast-clean ARGS="--after 2025-06-01 --forecast-only"       # Only clean main table
make forecast-clean ARGS="--before 2025-01-01 --date-column fcstdate --archive-only"  # Archive only, by fcstdate

# Testing
make test              # Run all backend pytest tests
make test-unit         # Backend unit tests only (common/ modules)
make test-api          # Backend API endpoint tests only
make test-cov          # Backend tests with coverage report
make ui-test           # Run frontend vitest unit tests
make test-all          # Run all backend + frontend tests
```

---

## Architecture

### Domain-Driven Generic Design

All datasets extend a single `DomainSpec` dataclass in `common/domain_specs.py`. Scripts and API endpoints are generic — they operate on any domain via `--dataset <name>` or `/domains/{domain}/*`.

**8 Domains:**
- **Dimensions (read-only):** `item`, `location`, `customer`, `time`, `dfu`
- **Facts (time-series):** `sales`, `forecast`
- **Inventory:** `inventory` (dedicated normalize script + API endpoints + UI tab)

### Data Flow

```
Source CSV → normalize_dataset_csv.py → clean CSV
                                              ↓
                          ┌───────────────────┴───────────────────┐
                          ▼                                         ▼
              load_dataset_postgres.py                spark_dataset_to_iceberg.py
                          ▼                                         ▼
                    PostgreSQL 16                          Apache Iceberg (MinIO)
                          ▼                                         ▼
                      FastAPI                                    Trino SQL
                          ▼
                    React UI (:5173)
```

### API Pattern

- Generic domains served via: `GET /domains/{domain}/rows`, `GET /domains/{domain}/search`, etc.
- Inventory has dedicated endpoints: `GET /inventory/position`, `GET /inventory/kpis`, `GET /inventory/trend`, `GET /inventory/item-detail`
- Pagination: offset/limit (50–1000 rows)
- Reserved word workaround: `class` column aliased as `class_` in responses

---

## Data Models

### Dimension Tables
- Surrogate key `sk`, composite key `ck`, `load_ts`, `modified_ts`
- Full-text search on configured fields via `pg_trgm` trigram indexes

### Fact Tables
- `fact_sales_monthly`: grain = item + customer_group + location + month + type; measures = qty_shipped, qty_ordered, qty
- `fact_external_forecast_monthly`: grain = item + loc + forecast_date + actual_month; tracks lag 0–4 months; measures = base forecast + actual demand
- `fact_inventory_snapshot`: grain = item_no + loc + snapshot_date; measures = qty_on_hand, qty_on_hand_on_order, qty_on_order, mtd_sales, lead_time_days (~190M rows)

### Archive Tables
- `backtest_lag_archive`: All-lags (0–4) backtest predictions for accuracy analysis at any horizon. Grain = forecast_ck + model_id + lag. Includes `timeframe` column for traceability.

### Materialized Views
- `agg_sales_monthly`, `agg_forecast_monthly` — pre-aggregated for O(1) KPI queries
- `agg_inventory_monthly` — monthly EOM on-hand, EOM on-hand+on-order, avg on-hand, monthly sales (MAX of cumulative MTD), avg daily sales (via LAG CTE), snapshot days, latest lead time. Daily sales derived from cumulative `mtd_sales` via LAG() window function. Enables DOS, WOC, Turns, LT Coverage computation.
- `mv_inventory_forecast_monthly` — inventory-forecast bridge: joins `agg_inventory_monthly` + `fact_external_forecast_monthly` + `dim_dfu`. Grain = item_no + loc + month_start + model_id. Computed: forecast_error, abs_error, dos, is_stockout, is_excess, bias_direction. Enables stockout/excess root cause attribution by forecast model.

---

## Frontend Features

- Paginated data explorer with column filtering and sorting
- Type-aware column filters: `=exact` prefix for B-tree match, plain text for GIN trigram substring search
- Column-level typeahead suggestions via native HTML datalist (text columns only)
- Chemistry-themed loading overlay: periodic table element tile with pulse-glow animation (replaces invisible spinner)
- Approximate row count badge (`100,000+`) for large filtered queries
- KPI cards: Accuracy %, WAPE, MAPE, Bias, Total Forecast/Actual
- KPI window selector: 1–12 month rolling window
- Multi-metric trend charts (dual Y-axis: volume left, accuracy % right)
- Item/Location filter with typeahead suggestions
- Postgres vs Iceberg latency benchmarking panel
- Champion Selection panel: model competition config, run, and FVA model-wins visualization
- Market Intelligence tab: item/location selector with Google web search + GPT-4o narrative briefing
- DFU Analysis tab: unified sales vs multi-model forecast overlay chart, 3 scope modes, per-model KPI cards, toggleable measures
- Collapsible sidebar navigation (12 items, 5 sections, mobile drawer, `[` toggle)
- Dashboard overview landing page: KPI sparkline cards, alert panel, heatmap, top movers, forecast trend chart
- Global filter bar: brand, category, item (searchable), location (searchable), market, channel multi-select dropdowns — applied to dashboard, accuracy, and auto-populated into tab-local inputs
- Single professional theme (Demand Studio) with light/dark modes via CSS variable palettes
- Keyboard shortcuts (1-9 tab switch, `[` sidebar, `d` dark mode, / search, Esc close, ? help, Ctrl+E fields)
- Lazy-loaded tab components with per-tab error boundaries
- TanStack Query caching (stale-while-revalidate, instant tab switching)
- Virtualized data grid with column resize, row selection, CSV export
- Print-ready CSS (@media print rules)
- ECharts integration for canvas-based charting
- Inventory tab: KPI cards, trend chart, paginated position table, item detail drill-down
- Inventory backtest tab: model comparison (stockout/excess/service level/WAPE), root cause attribution (bias direction), monthly trend, DFU-level event detail table
- Clustering What-If Scenarios panel: parameter controls, scenario simulation, result charts, promote flow, background execution with runtime estimation, dashboard completion alerts, enhanced charts (elbow with optimal K, silhouette with quality zones, feature importance, cluster size pie, gap statistic), scenario queueing (queued status when group busy), "View Results" navigation from JobsTab, Past Scenarios history (last 10 completed runs with inline charts)
- Job Scheduler/Monitor tab (APScheduler-powered): automation dashboard with KPI cards (Total Jobs, Active Now, Success Rate, Avg Duration), grouped job type cards with category colors (blue=clustering, violet=backtest, emerald=seasonality, amber=champion), "Run Now" and schedule buttons, live active job monitoring with animated progress bars and elapsed timers, schedule dialog with presets (hourly/6h/daily 2AM/weekly), recurring schedules section with cron badges, expandable job history with params/results/errors, sidebar active job count badge, cross-tab alerts via `JobNotificationContext`, ClustersTab "Schedule Scenario Job" integration
- Feature Importance (SHAP) panel in Accuracy tab: collapsible card with model selector (populated from `/forecast/shap/models`), timeframe selector (cross-timeframe summary or individual timeframes A–J), horizontal bar chart with indigo=selected / gray=dropped feature coloring, `selected_count`/`n_timeframes` consistency indicator (Feature 42)
- Inventory Planning tab (IPfeature4): EOQ KPI cards (avg EOQ, total cycle stock, avg annual cost), sensitivity chart (total cost vs order quantity curve), paginated detail table with per-item EOQ, cycle stock, ordering cost, holding cost; sidebar nav item "Inv. Planning" (12 items total)
- Vitest testing infrastructure

---

## Mandatory Testing Rules

**Every new feature, endpoint, component, hook, or utility MUST include tests. Every removed feature MUST have its tests removed. Tests MUST pass before work is considered complete.**

### When adding functionality:
1. **New Python module in `common/`** → Add unit tests in `tests/unit/test_<module>.py`
2. **New API endpoint** → Add API tests in `tests/api/test_<feature>.py` using httpx AsyncClient with ASGI transport
3. **New React component** → Add component tests in `src/components/__tests__/<Component>.test.tsx`
4. **New React hook** → Add hook tests in `src/hooks/__tests__/<hook>.test.ts`
5. **New utility function** → Add tests in `src/lib/__tests__/<util>.test.ts`
6. **New tab component** → Add smoke tests in `src/tabs/__tests__/<Tab>.test.tsx`

### When removing functionality:
1. Delete the corresponding test files
2. Remove any fixtures that are no longer needed
3. Update `conftest.py` if shared fixtures were affected

### Test execution:
- Run `make test-all` after every change to verify no regressions
- Backend tests: `make test` (~0.7s, no infra needed — DB is mocked)
- Frontend tests: `make ui-test` (218 tests, ~1.5s)
- Coverage: `make test-cov` for backend coverage report

### Test patterns:
- **Backend API tests:** Use `httpx.AsyncClient(transport=ASGITransport(app))` — no running server needed
- **Backend mocking:** Mock `pool` fixture in `tests/api/conftest.py` for DB; use `@patch.dict("sys.modules")` for imports inside functions
- **Frontend component tests:** Wrap with `QueryClientProvider` from `src/tabs/__tests__/test-utils.tsx`
- **Frontend mocking:** Use `vi.mock("../api/queries")` for API layer; mock `echarts-for-react` for chart components

### Reference:
- Full testing strategy: `docs/design-specs/feature31.md`
- Backend test directory: `mvp/demand/tests/`
- Frontend test directories: `mvp/demand/frontend/src/**/__tests__/`

---

## Important Conventions

- **Config files for all modules:** Every module (script, job, computation pipeline) MUST externalize all configurable parameters into a YAML file under `mvp/demand/config/`. No magic numbers, thresholds, default values, or tuning knobs hardcoded inside Python scripts. Scripts load their config via `yaml.safe_load(open("config/<module>_config.yaml"))` at startup. Examples: `variability_config.yaml` (CV thresholds, history_months), `safety_stock_config.yaml` (service levels by ABC class, Z-table, guard rails), `eoq_config.yaml` (ordering cost, holding cost, MOQ source), `simulation_config.yaml` (n_simulations, random_seed), `replenishment_policy_config.yaml` (policy definitions, auto-assign rules). API endpoints that expose config values must read from the YAML file, not from hardcoded dicts.

- **Null normalization:** `''`, `'null'`, `'none'`, `'NA'` all treated as NULL during load
- **Type casting:** Integer/float/date fields auto-cast with null coercion in normalize scripts
- **Lag computation:** `month_diff` auto-computed during forecast normalization
- **Forecast accuracy formula:** `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- **Bias formula:** `(SUM(Forecast) / SUM(History)) - 1`
- **Sales filtering:** Only rows with `TYPE=1` are loaded into `fact_sales_monthly`
- **Time dimension:** Auto-generated 2020–2035, not sourced from a file
- **Forecast model_id:** Identifies the forecasting algorithm; default `'external'` for source-system forecasts. `UNIQUE(forecast_ck, model_id)` constraint prevents duplicates within a model. Not part of the business key.
- **Forecast execution-lag loading:** `load_dataset_postgres.py` uses a **dual-path insert with phase ordering** to preserve archive integrity. **Phase 3b** loads the archive FIRST from untouched staging data — each row's original `lag` is preserved as `execution_lag` in `backtest_lag_archive`. **Phase 3c** THEN mutates the staging table's `execution_lag` column from `dim_dfu` (matched DFUs get the DFU-level value, unmatched default to 0). **Phase 5** inserts into the main table with `WHERE lag = execution_lag` — only execution-lag rows enter `fact_external_forecast_monthly`. This ordering is critical: the archive must read staging before the mutation, otherwise all rows for a DFU would have the same `execution_lag` value, corrupting multi-horizon accuracy analysis. Use `--replace` flag (or `make load-forecast-replace`) to only delete+reload `model_id='external'` rows without truncating the whole table (preserves backtest/champion/ceiling data). Use `--skip-archive` flag (or `make load-forecast-replace-no-archive`) to skip the archive load entirely — only the execution-lag row is loaded into the main table, bypassing the 45M-row archive INSERT for faster external forecast reloads. See `feature2.md` § "Forecast Loading: Dual-Path with Execution-Lag Filtering" for a worked example.
- **Chat endpoint:** `POST /chat` — OpenAI-powered NL→SQL with pgvector context retrieval. Read-only execution with 5s timeout and 500-row limit. Requires `OPENAI_API_KEY` in `.env`.
- **DFU clustering:** KMeans-based clustering pipeline groups DFUs by demand patterns. Feature engineering extracts time series, item, and DFU features. Cluster labels (e.g., `high_volume_steady`, `seasonal_medium_volume`) stored in `dim_dfu.cluster_assignment`. MLflow tracks experiments under `dfu_clustering`. Config in `config/clustering_config.yaml`.
- **Champion model selection:** Configurable per-DFU per-month champion selection via 5 strategies: expanding (cumulative WAPE), rolling (last N months), decay (exponential weighting), ensemble (blend top-K models), meta_learner (ML classifier). All strategies enforce **exec-lag-aware strict causality** — selection for month T with execution_lag=L uses ONLY data from months where `startdate < T − L` (i.e. `startdate < fcstdate`), implemented as `shift(exec_lag + 1)` per DFU-model group. This prevents using actuals that weren't available when the forecast was issued. With `exec_lag=0` the behaviour is identical to `shift(1)` (backward compatible). **Fallback model** (`fallback_model_id: lgbm_cluster` by default): DFU-months in the warm-up period (first `exec_lag + min_dfu_rows` months with insufficient prior history) are filled with the fallback model's forecast so every DFU-month always has a champion row. Strategy registry in `common/champion_strategies.py`. Config in `config/model_competition.yaml` controls competing models, metric, lag, `min_dfu_rows`, `fallback_model_id`, `strategy`, and `strategy_params`. Champion rows stored as `model_id='champion'` in `fact_external_forecast_monthly`. Ceiling (oracle) picks the best model per DFU per month with perfect foresight (after-the-fact), stored as `model_id='ceiling'`. Both at DFU-month granularity with consistent WAPE formula `SUM(|F-A|) / |SUM(A)|`. Gap-to-ceiling quantifies improvement opportunity. Meta-learner uses ceiling labels as ground truth with strict temporal train/test split. Simulation script (`scripts/simulate_champion_strategies.py`) runs all strategies and compares accuracy vs ceiling. UI panel in Accuracy tab shows champion + ceiling KPI cards, gap-to-ceiling indicator, and dual model wins bar charts.
- **Shared backtest framework (Feature 44 — per-cluster only):** All tree-based backtest scripts (LGBM, CatBoost, XGBoost) use `common/backtest_framework.py` as a shared orchestrator via `run_tree_backtest()`. Each script implements only `train_fn_per_cluster` (global and transfer strategies were removed in Feature 44). Algorithm options (recursive, shap_select, tune_inline, params_file, hyperparameters) are read from `config/algorithm_config.yaml` — not from CLI flags. Shared modules in `common/`: `backtest_framework.py`, `feature_engineering.py`, `metrics.py`, `mlflow_utils.py`, `db.py`, `constants.py`, `tuning.py` (CV splits, fold functions, per-timeframe inline tuner), `shap_selector.py` (SHAP extraction, feature selection, CSV output). `run_tree_backtest()` optional parameters: `inline_tuner_fn` (PL-002, per-timeframe causal tuning), `feature_selector_fn` (Feature 42, SHAP retraining), `recursive: bool = False` (Feature 43, recursive multi-step inference via `update_grid_with_predictions()`). `_predict_single_month(models, predict_data, feature_cols)` routes each recursive inference step to the correct cluster model.
- **Backtest output paths (model-scoped subdirectories):** Each backtest run writes output to `data/backtest/<model_id>/` (e.g., `data/backtest/lgbm_cluster/backtest_predictions.csv`). Multiple models can be run sequentially without overwriting each other. Load with `make backtest-load MODEL=<model_id>` or `make backtest-load-all` (scans all subdirs). See PL-001 in `docs/PARKING_LOT.md` for history.
- **Hyperparameter tuning (Feature 41):** Bayesian Optuna tuning for LGBM, CatBoost, XGBoost cluster models. Walk-forward CV with causal masking (`mask_future_sales()` inside every fold). WAPE stabilised with denominator floor. `n_estimators` determined by early stopping (not in search space). Outputs `data/tuning/best_params_<model>.json` with `best_params` + `best_n_estimators` + per-cluster WAPEs. Apply tuned params by setting `params_file: data/tuning/best_params_lgbm.json` in `config/algorithm_config.yaml` (Feature 44). Make targets: `tune-lgbm`, `tune-catboost`, `tune-xgboost`, `tune-all`. MLflow experiment: `hyperparameter_tuning`. Config: `config/hyperparameter_tuning.yaml`. Shared utilities: `common/tuning.py`. **Two-mode workflow:** (1) Production scoring: tune once on full history (`make tune-lgbm`), apply via `params_file` in algorithm config — fastest path, no future leakage for production use. (2) Honest backtesting: per-timeframe causal tuning via `tune_inline: true` in algorithm config (PL-002 fix) — each timeframe tunes on only the data available at that point in time; no future leakage into backtest accuracy metrics. `TRAIN_FOLD_FNS` registry in `common/tuning.py` exposes shared fold functions for both global tuning and inline tuner.
- **SHAP feature selection (Feature 42):** Per-timeframe automatic feature selection using SHAP values for LGBM, CatBoost, and XGBoost backtests. Activated by `shap_select: true` in `config/algorithm_config.yaml` (Feature 44). Flow per timeframe: (1) train initial model on all features, (2) compute SHAP via `common/shap_selector.py`, (3) select features covering 95% cumulative SHAP mass (or exactly `shap_top_n` features), (4) retrain final model on selected features. CatBoost uses native `get_feature_importance(type="ShapValues")`; LGBM/XGBoost use `shap.TreeExplainer` (requires `shap>=0.43.0`). SHAP is pooled across clusters weighted by cluster size; `ml_cluster` is excluded from the effective feature set. Output written to `data/backtest/<model_id>/shap/shap_timeframe_XX.csv` (per-timeframe) and `shap_summary.csv` (cross-timeframe). Served via 4 read-only REST endpoints under `/forecast/shap/` (no DB queries — CSV-based). Composable with `tune_inline` (PL-002) and `params_file` via config keys. Config keys: `shap_select`, `shap_top_n`, `shap_threshold` (default 0.95), `shap_sample_size` (default 500). Graceful fallback: if SHAP computation fails, all features are kept and the backtest continues.
- **Recursive multi-step forecasting (Feature 43):** Enabled via `recursive: true` in `config/algorithm_config.yaml` (Feature 44). In direct mode (default), all future months are predicted from the same lag-1-zero baseline (masked sales = 0 for months 2+). In recursive mode, each month in the prediction window is forecast one at a time; the model's prediction for month T is written back into the feature grid via `update_grid_with_predictions()` in `common/feature_engineering.py`, recomputing all lag and rolling features before month T+1 is scored. This gives `qty_lag_1` a real signal (model's own prior prediction) instead of zero. `_predict_single_month(models, predict_data, feature_cols)` in `common/backtest_framework.py` routes inference to the correct cluster model dict during the recursive loop (inference-only, no retraining). `_fill_predict_nans()` fills numeric NaNs per-month. `recursive: bool = False` is the parameter on `run_tree_backtest()`. Composable with `shap_select` and `tune_inline` via config keys. `"recursive": true` written to `backtest_metadata.json` for traceability. No API, frontend, or DB changes. Trade-off: richer near-horizon signal vs potential error compounding across months.
- **Market intelligence:** `POST /market-intelligence` — combines Google Custom Search API (product news/trends) + GPT-4o narrative synthesis for item + location pairs. Looks up item metadata (description, brand, category) from `dim_item` and location state from `dim_location`. Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in `.env`.
- **Backtest cleanup:** `scripts/clean_backtest_models.py` selectively removes model predictions from `fact_external_forecast_monthly` and `backtest_lag_archive` by `model_id`, then refreshes 5 materialized views. Supports `--list`, `--dry-run`, `--all-backtest` (excludes `external`). Make targets: `backtest-clean`, `backtest-list`.
- **Forecast date-range cleanup:** `scripts/clean_forecasts_by_date.py` deletes rows from `fact_external_forecast_monthly` and/or `backtest_lag_archive` by time bucket. Supports `--before`, `--after`, `--between` date range filters and `--months` for specific month(s) on `startdate` (default) or `fcstdate`, optional `--model` filter, `--forecast-only`/`--archive-only` scope, `--dry-run` preview, and `--list` for row counts by model+month. All dates normalized to month-start. Refreshes same 5 materialized views as `clean_backtest_models.py`. Make targets: `forecast-clean`, `forecast-clean-list`.
- **Benchmarking:** `GET /bench/compare` runs identical queries (count, page, trend) against Postgres and Trino/Iceberg, returning per-query latency stats (min/max/avg/p50/p95) with winner determination and speedup factor. Requires Docker services running. Make target: `bench-compare`.
- **Inventory snapshots:** 14 monthly CSV files (`datafiles/Inventory_Snapshot_YYYY_MM.csv`, ~190M rows total) merged by `scripts/normalize_inventory_csv.py` into a single clean CSV. Loaded into `fact_inventory_snapshot` via generic loader. `qty_on_order` derived as `qty_on_hand_on_order - qty_on_hand` during normalization. Dedicated API endpoints (`/inventory/*`) and frontend InventoryTab. `agg_inventory_monthly` materialized view with daily sales derivation (LAG CTE), EOM snapshots, and proper monthly sales (MAX not SUM). `/inventory/kpis` uses two-query pattern: point-in-time totals from latest snapshot + trailing-month aggregates for supply chain KPIs (DOS, WOC, Inventory Turns, LT Coverage). KPI cards use severity color-coding (green/yellow/red thresholds). Trend chart renders 5 lines: On Hand, On Order, Monthly Sales, Lead Time, Days of Supply.
- **DFU seasonality detection:** Pipeline in `scripts/detect_seasonality.py` + `update_seasonality_profiles.py` computes seasonality metrics (strength, profile label, peak/trough month, peak-to-trough ratio, is_yearly_seasonal flag) from sales history and writes them to `dim_dfu`. Config in `config/seasonality_config.yaml`. DDL in `sql/015_add_seasonality_columns.sql`. Make targets: `seasonality-detect`, `seasonality-update`, `seasonality-all`. These 6 columns (`seasonality_profile`, `seasonality_strength`, `is_yearly_seasonal`, `peak_month`, `trough_month`, `peak_trough_ratio`) are now part of `DFU_SPEC` and are exposed by the generic Data Explorer.
- **What-If clustering scenarios:** `POST /clustering/scenario` runs a trial KMeans pipeline with custom `feature_params`, `model_params`, and `label_params` without overwriting production clustering. Returns HTTP 202 immediately and runs in background thread; `GET /clustering/scenario/{id}/status` polls for running/completed/failed. `GET /clustering/scenario/estimate` returns runtime estimate based on DFU count, K range, and gap flag. `POST /clustering/scenario/{id}/promote` applies the winning scenario to `dim_dfu.ml_cluster`. `ScenarioNotificationContext` tracks running/completed state across tabs; Dashboard injects completion alert. Enhanced charts: elbow with optimal K ReferenceLine, silhouette bar chart with quality zone thresholds (Strong/Reasonable/Weak/No structure), feature importance horizontal bars, cluster size pie chart, conditional gap statistic line chart. Requires `API_KEY` env var to be set for auth (disabled when unset). **Scenario queueing:** When a clustering job is already running, new scenarios are queued (`status="queued"`) instead of rejected with 409; queued jobs auto-dispatch via `_dispatch_next()` when the active job completes. **View Results:** "View Results" button in JobsTab navigates to ClustersTab with `?scenario_job=<id>` URL param; ClustersTab auto-loads result and renders ScenarioCharts. **Past Scenarios:** ClustersTab What-If panel shows last 10 completed scenario runs in an accordion with inline charts and promote buttons.
- **Modular API router architecture:** `api/routers/` contains 14 FastAPI `APIRouter` modules (accuracy, analysis, benchmark, chat, clusters, competition, dashboard, domains, intel, inv_backtest, inventory, inv_planning, jobs, shap). `main.py` is a ~65-line shell that only creates the app, adds middleware, and mounts all 14 routers via `app.include_router()`. All route handlers live in router modules — no inline routes in main.py. `domains.py` is mounted last because it has catch-all `{domain}` path parameters. All mutation endpoints require `require_api_key` auth when `API_KEY` env var is set.
- **Job scheduler (APScheduler):** `common/job_registry.py` provides `JobManager` singleton powered by APScheduler 3.11 (`BackgroundScheduler` + `ThreadPoolExecutor(max_workers=4)`). Thread-safe: `_state_lock` guards `_active_jobs`, `_pending_queues`, `_cancel_flags`; `_init_lock` with double-checked locking protects `_ensure_init()`. `JOB_TYPE_REGISTRY` maps 7 job types across 4 groups. Per-group concurrency control with FIFO queueing (one active job per group: clustering, backtest, seasonality, champion; busy groups queue jobs instead of rejecting). Job callables wrap existing scripts via `subprocess.run()`. Progress updates written to `job_history` table. `recover_stale_jobs()` re-enqueues queued jobs from DB on restart and marks running jobs as failed. Supports cron/interval scheduling (`POST /jobs/schedule`, `GET /jobs/schedules`), job pipelines (`POST /jobs/pipeline` — sequential chaining), retry logic with exponential backoff (`max_retries`), and dashboard stats (`GET /jobs/stats`). 12 REST API endpoints total. Route ordering in `jobs.py`: literal paths (`/jobs/schedules`, `/jobs/pipeline`) must come before parameterized `{job_id}` paths. Frontend polls `GET /jobs/active` every 2s, stats every 5s, history every 10s. `JobNotificationContext` provides cross-tab completion alerts. Sidebar shows active job count badge. ClustersTab uses "Schedule Scenario Job" button. Dependencies: `apscheduler>=3.10`, `tzlocal>=5.0`.
- **API key authentication:** `api/auth.py` provides `require_api_key` FastAPI dependency. Auth is disabled when the `API_KEY` env var is unset (development default). When set, mutation endpoints (`POST /clustering/scenario`, `PUT /competition/config`, `POST /competition/run`, `POST /chat`, `POST /market-intelligence`) require `X-API-Key` header.
- **Vite dev server proxy:** `frontend/vite.config.ts` proxies all API path prefixes (`/domains`, `/jobs`, `/clustering`, `/forecast`, `/inventory`, `/dashboard`, `/health`, `/chat`, `/dfu`, `/competition`, `/bench`, `/market-intelligence`) to the FastAPI backend at `http://127.0.0.1:8000`. **CRITICAL:** When adding a new API path prefix, you MUST add a corresponding proxy entry in `vite.config.ts` or the frontend will receive HTML instead of JSON. Restart the Vite dev server (`make ui`) after changes.
- **Single theme with light/dark modes:** Only the "General" (Demand Studio) product theme remains. `useTheme()` manages light/dark color mode. `ThemeSelector` in sidebar footer provides light/dark toggle. No theme cycling, no motifs.
- **Theme context (no prop-drilling):** Tab components access the current theme via `useThemeContext()` from `context/ThemeContext.tsx` or `useChartColors()` from `hooks/useChartColors.ts` — NOT via a `theme` prop from `App.tsx`. `ThemeProvider` wraps the app tree in `App.tsx`. `useChartColors()` returns `{ theme, chartColors, trendColors }` for Recharts styling. `ScenarioCharts` component extracted to `components/ScenarioCharts.tsx` (elbow, silhouette, radar, pie, gap charts).
- **Algorithm configuration (Feature 44):** All backtest algorithm options are controlled by `config/algorithm_config.yaml`, not CLI flags. Backtest scripts for LGBM, CatBoost, and XGBoost accept only `--config`, `--model-id`, and `--n-timeframes`. Features (recursive, shap_select, tune_inline, params_file, default hyperparameters) are set per-algorithm in the YAML file. Only per-cluster strategy (`lgbm_cluster`, `catboost_cluster`, `xgboost_cluster`) is supported — global and transfer strategies were removed. Prophet, StatsForecast, NeuralProphet, PatchTST, and DeepAR scripts were deleted. `run_tree_backtest()` only accepts `train_fn_per_cluster`; `_predict_single_month(models, predict_data, feature_cols)` no longer takes a `cluster_strategy` argument.

---

## Design Specs

Located in `docs/design-specs/`:
- `feature1.md` — Infrastructure & platform setup
- `feature2.md` — Internal data architecture & data contracts (includes ERD)
- `feature3.md` — Dimension tables (Item, Location, Customer, Time, DFU)
- `feature4.md` — Fact tables (Sales, External Forecast)
- `feature5.md` — Forecast accuracy KPIs
- `feature6.md` — Multi-model forecast support
- `feature7.md` — DFU clustering framework
- `feature8.md` — Backtesting framework (expanding window timeframes)
- `feature9.md` — LGBM backtesting implementation
- `feature10.md` — Multi-dimensional accuracy slicing
- `feature11.md` — Chatbot / natural language queries
- `feature12.md` — CatBoost backtesting implementation
- `feature13.md` — XGBoost backtesting implementation
- `feature14.md` — Transfer learning backtest strategy
- `feature15.md` — Champion model selection (exec-lag-aware best-of-models per DFU per month, fallback for warm-up gaps)
- `feature16.md` — Data Explorer performance & UX (type-aware filtering, GIN indexes, column typeahead, loading overlay)
- `feature17.md` — DFU Analysis tab (sales vs multi-model forecast overlay)
- `feature18.md` — Market intelligence (web search + LLM narrative briefings)
- `feature19.md` — PatchTST backtesting implementation (deep learning, Apple MPS GPU)
- `feature20.md` — DeepAR backtesting implementation (LSTM probabilistic forecasting)
- `feature21.md` — Prophet backtesting implementation (per-DFU time series)
- `feature22.md` — UI theming (dark mode + midnight theme)
- `feature23.md` — Backtest model cleanup utility (selective model removal + view refresh)
- `feature24.md` — StatsForecast backtesting implementation (vectorized AutoARIMA + AutoETS)
- `feature25.md` — NeuralProphet backtesting implementation (PyTorch-based Prophet with GPU)
- `feature26.md` — Postgres vs Trino/Iceberg benchmarking (latency comparison API)
- `feature27.md` — Figma MCP Integration: Design-to-Code & Code-to-Design Workflow *(not started)*
- `feature28.md` — UI Architecture & Performance (component decomposition, TanStack Query, lazy loading, error boundaries, keyboard shortcuts, testing)
- `feature29.md` — What-If / Scenario UI for Clustering (UI implemented; backend routes mounted via include_router)
- `feature30.md` — DFU Seasonality Detection & Profile Assignment (scripts + DDL implemented; Makefile targets added)
- `feature31.md` — Comprehensive Testing Strategy (full-stack testing spec, mandatory test requirements)
- `feature32.md` — Seasonality Profile Filtering (backend router written; frontend UI pending)
- `feature33.md` — Inventory Overlay in DFU Analysis *(not implemented)*
- `feature34.md` — Inventory Planning Module Phase 1 (snapshot pipeline, API, UI tab)
- `feature35.md` — Configurable Multi-Theme / Motif System (motifs removed; simplified to single professional theme with light/dark modes)
- `feature36.md` — Product-Grade UI Overhaul (sidebar, themes, dashboard, global filters, widgets)
- `feature37.md` — Inventory Planning Backtesting (forecast-inventory bridge, model comparison, root cause attribution)
- `feature38.md` — Clustering What-If Scenario Enhancements (background execution, runtime estimation, dashboard alerts, enhanced charts)
- `feature39.md` — Job Scheduler/Monitor with APScheduler (APScheduler 3.11 engine, 12 API endpoints, cron/interval scheduling, job pipelines, retry logic, automation dashboard UI, agentic AI foundation, scenario queueing, View Results navigation, Past Scenarios history)
- `feature41.md` — Hyperparameter Tuning for Tree-Based Cluster Models (Optuna Bayesian optimization, walk-forward CV, early stopping, model-scoped output dirs, --params-file integration)
- `feature42.md` — SHAP-Based Per-Timeframe Feature Selection (LGBM/CatBoost/XGBoost, cumulative importance threshold, cluster-pooled SHAP, CSV output, 4 REST endpoints, Accuracy tab UI panel)
- `feature43.md` — Recursive Multi-Step Forecasting (--recursive flag, update_grid_with_predictions, _predict_single_month, per-month lag write-back, composable with SHAP + inline tuning, 9 Makefile targets, 19 unit tests)
- `feature44.md` — Algorithm Configuration & Simplification (algorithm_config.yaml, per-cluster only, removed Prophet/StatsForecast/NeuralProphet/PatchTST/DeepAR, simplified run_tree_backtest, 4 Makefile targets, 512 backend tests)
- `IPfeature4.md` — EOQ & Cycle Stock Calculator (Wilson EOQ formula, effective EOQ with MOQ + cap, cycle stock metrics, sensitivity curve, fact_eoq_targets table, 3 API endpoints, InvPlanningTab UI, eoq_config.yaml, 39 tests total, 630 backend + 238 frontend)
- `theme-testing-strategy.md` — Multi-Theme Testing Strategy (unit tests implemented; integration/a11y/perf tests pending)
- `docs/REFACTORING_RECOMMENDATIONS.md` — Comprehensive codebase refactoring roadmap

---

## Documentation Update Rules

**Whenever ANY code is added, changed, or deleted in the codebase, you MUST update ALL of the following documentation files to keep them in sync:**

1. **`docs/architecture-diagram.md`** — Update the architecture diagram (layers, components, data flow, ML pipeline) to reflect any structural changes — new/removed modules, routers, tabs, scripts, services, tables, or data flows
2. **`mvp/demand/docs/ARCHITECTURE.md`** — Update architecture, component technologies, tables, or data flow if affected
3. **`mvp/demand/docs/README.md`** — Update stack, datasets, analytics behavior, quick start, or key paths if affected
4. **`mvp/demand/docs/RUNBOOK.md`** — Update setup steps, notes, or troubleshooting if affected
5. **`docs/design-specs/feature<N>.md`** — Create or update the design spec for the feature
6. **`docs/design-specs/feature1.md`** — Add the feature to the "Implemented Features (MVP)" list
7. **`CLAUDE.md`** (this file) — Update Key Files, Common Commands, Data Models, Frontend Features, Important Conventions, or Design Specs list if affected

**This applies to ALL changes — additions, modifications, AND deletions. When code is removed, the corresponding references in ALL documentation files above must also be removed or updated.**

**Additionally, you MUST write tests for every change and run `make test-all` to verify they pass:**

8. **`mvp/demand/tests/`** — Add or update backend tests for any new/modified Python modules or API endpoints
9. **`mvp/demand/frontend/src/**/__tests__/`** — Add or update frontend tests for any new/modified components, hooks, or utilities
10. **Run `make test-all`** — Verify all 485+ tests pass (both backend and frontend) before considering the work complete

**What counts as changes requiring doc updates:**
- New feature implementation (new endpoints, UI panels, tables, scripts)
- Schema changes (new columns, tables, indexes, materialized views)
- New dependencies or infrastructure changes (docker images, pyproject.toml)
- New Make targets or CLI commands
- Changes to data flow or pipeline behavior
- Removal or renaming of any module, endpoint, component, table, or script
- Refactors that change architecture, file structure, or public interfaces

**What does NOT require doc updates:**
- Bug fixes that don't change behavior or interfaces
- Minor internal code refactors that don't change architecture or file structure
- Typo corrections within code (not in docs)

**What ALWAYS requires tests (even for bug fixes):**
- Any new Python function or class
- Any new API endpoint or modification to existing endpoint behavior
- Any new React component, hook, or utility
- Bug fixes that change behavior (add a regression test)

---

## Do Not

- Do not commit `__pycache__/`, `.pyc` files, or `.venv/`
- Do not modify `mvp/demand/data/*.csv` files manually — they are generated by normalize scripts
- Do not touch the `reference/` directory — it is archived code
- Do not run `make spark-all` unless Iceberg/MinIO is needed; Postgres path is sufficient for most dev work
