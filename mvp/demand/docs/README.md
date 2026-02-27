# Demand Unified MVP (Item + Location + Customer + Time + DFU + Sales + Forecast)

Unified codebase for demand datasets.

## Datasets
Dimensions:
- `dim_item`
- `dim_location`
- `dim_customer`
- `dim_time`
- `dim_dfu`

Facts:
- `fact_sales_monthly`
- `fact_external_forecast_monthly`
- `fact_inventory_snapshot`

Sales source details (MVP):
- input file: `datafiles/dfu_lvl2_hist.txt`
- load rule: only `TYPE=1`
- monthly grain date: `STARTDATE` in `YYYYMMDD`, day must be `01`

Forecast source details (MVP):
- input file: `datafiles/dfu_stat_fcst.txt`
- monthly grain dates: `fcstdate`, `startdate` (both month-start)
- lag rule: `lag = month_diff(startdate, fcstdate)` with allowed range `0..4`
- `model_id`: identifies forecasting algorithm (default `'external'`); uniqueness is `(forecast_ck, model_id)`

Inventory source details (MVP):
- input files: 14 monthly CSVs `datafiles/Inventory_Snapshot_YYYY_MM.csv` (~190M rows total)
- columns: `exec_date,item,loc,lead_time,tot_oh,tot_oh_oo,mtd_sls`
- grain: `item_no` + `loc` + `snapshot_date` (monthly)
- `qty_on_order` derived as `qty_on_hand_on_order - qty_on_hand` during normalization
- materialized view: `agg_inventory_monthly` for monthly trend aggregation

AI/Chatbot:
- `chat_embeddings` table (pgvector) — stores schema metadata embeddings for NL query context

Clustering:
- DFU clustering framework — groups DFUs by historical demand patterns for improved LGBM model performance
- Cluster assignments stored in `dim_dfu.cluster_assignment` column
- Features: time series (volume, trend, seasonality, volatility), item attributes, DFU attributes
- Clustering: KMeans with optimal K selection (elbow, silhouette, gap statistic)
- Automated labeling: high_volume_steady, seasonal_high_volume, intermittent_low_volume, etc.
- MLflow integration for experiment tracking

## Why this refactor
- One backend and one UI app for all datasets
- Shared normalize/load/API paging/filter/sort patterns
- Facts (`sales`, `forecast`) handled as facts in schema and docs

## Stack
- PostgreSQL 16 (pgvector/pgvector:pg16 — includes vector extension)
- FastAPI + Uvicorn
- React + Vite + Tailwind + shadcn/ui + Recharts + ECharts
- Spark + Iceberg + MinIO
- Trino
- MLflow (experiment tracking, model registry)
- LightGBM (demand forecasting models)
- CatBoost (demand forecasting models)
- XGBoost (demand forecasting models)
- Prophet (per-DFU time series forecasting)
- PyTorch (PatchTST Transformer + DeepAR LSTM + NeuralProphet deep learning models)
- StatsForecast (vectorized AutoARIMA + AutoETS statistical models)
- scikit-learn (clustering algorithms)
- OpenAI (GPT-4o + text-embedding-3-small) for NL→SQL chatbot
- APScheduler 3.11 (job scheduling engine — BackgroundScheduler + ThreadPoolExecutor)
- Docker Compose

## Performance defaults
- Fact indexes on `(dmdunit, loc, startdate/fcstdate)` are applied via `sql/008_perf_indexes_and_agg.sql`.
- Trigram (`pg_trgm`) indexes are applied for common `ILIKE` search fields.
- Monthly materialized views are maintained for faster trend analytics:
  - `agg_sales_monthly`
  - `agg_forecast_monthly`
- Accuracy slice views for O(1) aggregate KPI queries (feature10):
  - `agg_accuracy_by_dim` — pre-joins forecast + DFU attributes at (model, lag, month, cluster, supplier, abc_vol, region, brand) grain
  - `agg_accuracy_lag_archive` — same from archive table; powers lag-curve analysis
- Inventory aggregate view for O(1) trend queries:
  - `agg_inventory_monthly` — monthly avg on-hand, avg on-order, avg lead time, total MTD sales
- `load-sales`, `load-forecast`, `load-inventory`, and `load-all` refresh `agg_*` aggregates automatically.
- `backtest-load` refreshes accuracy slice views automatically.

## Quick Start
```bash
cd mvp/demand
make init
make up
make normalize-all
make load-all
make generate-embeddings   # populate chat embeddings (requires OPENAI_API_KEY in .env)
make cluster-all            # optional: run DFU clustering (requires sales data)
make inventory-pipeline     # optional: normalize + load inventory snapshots
```

Run API:
```bash
cd mvp/demand
make api
```

Run UI:
```bash
cd mvp/demand
make ui-init
make ui
```

Open UI:
- `http://127.0.0.1:5173`

UI prerequisites:
- Node.js + npm installed
- Internet access to pull npm packages the first time (`make ui-init`)

Analytics behavior:
- analytics is enabled only for `sales` and `forecast` (dimensions are table-only)
- `sales` and `forecast`: item/location analytics filters on `dmdunit` and `loc`
- item/location filters use exact-match behavior
- on initial load for `sales`/`forecast`, UI auto-fills a sampled item+location pair to avoid full-table trend scans
- Item/location filters provide typeahead suggestions while typing
- Trend chart supports multiple measures using `Trend Measures` checkboxes
- Forecast domain includes a **Model selector** to filter analytics by `model_id`

Data Explorer (feature16):
- Column-level filters with two modes: plain text for substring search (`ILIKE`), prefix `=` for exact match
- Column-level typeahead suggestions (text columns only) — dropdown shows matching values as you type
- Type-aware SQL filtering avoids `::text` casts to leverage B-tree and GIN trigram indexes
- Approximate row count badge (`100,000+`) for filtered queries on large tables (66M+ rows)
- Chemistry-themed loading overlay: periodic table element tile with pulse-glow animation over frosted glass backdrop
- GIN trigram indexes on fact table text columns — apply once via `make db-apply-sql`

Chatbot:
- Collapsible chat panel below the analytics grid
- Ask questions in plain English (e.g., "What's the accuracy for item 100320?")
- Returns answer + generated SQL + result data table
- Requires `OPENAI_API_KEY` in `.env`

Clustering:
- Run full pipeline: `make cluster-all`
- Individual steps: `cluster-features`, `cluster-train`, `cluster-label`, `cluster-update`
- Feature generation extracts time series patterns from sales history (default: 24 months, min 12 months)
- Optimal K selection via elbow method, silhouette score, and gap statistic
- Cluster labels assigned automatically based on volume and pattern characteristics
- Results logged to MLflow experiment `dfu_clustering`
- Cluster assignments updated in `dim_dfu.cluster_assignment` column
- API endpoint: `GET /domains/dfu/clusters` returns cluster summary statistics
- Filter DFUs by cluster: use `cluster_assignment` filter in `/domains/dfu/page` endpoint

LGBM Backtesting:
- Run global backtest: `make backtest-lgbm` (trains LightGBM across 10 expanding windows)
- Run per-cluster backtest: `make backtest-lgbm-cluster` (separate model per cluster)
- Run transfer backtest: `make backtest-lgbm-transfer` (global base → per-cluster fine-tune)
- Load predictions: `make backtest-load` (loads execution-lag rows into `fact_external_forecast_monthly`, all-lag rows into `backtest_lag_archive`, refreshes accuracy slice views)
- Or all at once: `make backtest-all`
- Models appear as `lgbm_global` / `lgbm_cluster` / `lgbm_transfer` in the forecast model selector
- Existing accuracy KPIs and trend charts work automatically for LGBM models
- `backtest_lag_archive` stores lag 0–4 predictions for accuracy reporting at any horizon
- `make backtest-load` only replaces rows for the model_id in the CSV (safe to run per-cluster after global)

CatBoost Backtesting:
- Run global backtest: `make backtest-catboost` (trains CatBoost across 10 expanding windows)
- Run per-cluster backtest: `make backtest-catboost-cluster` (separate model per cluster)
- Run transfer backtest: `make backtest-catboost-transfer` (global base → per-cluster fine-tune)
- Load predictions: `make backtest-load` (same shared loader as LGBM)
- Models appear as `catboost_global` / `catboost_cluster` / `catboost_transfer` in the forecast model selector
- Same feature engineering, lag strategy, and output format as LGBM

XGBoost Backtesting:
- Run global backtest: `make backtest-xgboost` (trains XGBoost across 10 expanding windows)
- Run per-cluster backtest: `make backtest-xgboost-cluster` (separate model per cluster)
- Run transfer backtest: `make backtest-xgboost-transfer` (global base → per-cluster fine-tune)
- Load predictions: `make backtest-load` (same shared loader as LGBM)
- Models appear as `xgboost_global` / `xgboost_cluster` / `xgboost_transfer` in the forecast model selector
- Same feature engineering, lag strategy, and output format as LGBM

Prophet Backtesting (feature21):
- Run global backtest: `make backtest-prophet` (fits individual Prophet model per DFU)
- Run per-cluster backtest: `make backtest-prophet-cluster` (only clustered DFUs)
- Run pooled backtest: `make backtest-prophet-pooled` (aggregate by cluster → fit → disaggregate)
- Load predictions: `make backtest-load` (same shared loader)
- Models appear as `prophet_global` / `prophet_cluster` / `prophet_pooled` in the forecast model selector
- Native Fourier seasonality decomposition — no hand-engineered lag features
- Per-DFU fitting with multiprocessing parallelism (4 workers)

PatchTST Backtesting (feature19):
- Run global backtest: `make backtest-patchtst` (Transformer-based model, Apple MPS GPU)
- Run per-cluster backtest: `make backtest-patchtst-cluster` (separate model per cluster)
- Run transfer backtest: `make backtest-patchtst-transfer` (global base → per-cluster fine-tune)
- Load predictions: `make backtest-load` (same shared loader)
- Models appear as `patchtst_global` / `patchtst_cluster` / `patchtst_transfer` in the forecast model selector
- Patched time series input: 12-month lookback → overlapping 3-month patches → Transformer encoder
- ~60K parameters, HuberLoss, AdamW + CosineAnnealing + early stopping

DeepAR Backtesting (feature20):
- Run global backtest: `make backtest-deepar` (LSTM-based probabilistic model)
- Run per-cluster backtest: `make backtest-deepar-cluster` (separate model per cluster)
- Run transfer backtest: `make backtest-deepar-transfer` (global base → per-cluster fine-tune)
- Load predictions: `make backtest-load` (same shared loader)
- Models appear as `deepar_global` / `deepar_cluster` / `deepar_transfer` in the forecast model selector
- Gaussian likelihood output (mu + sigma) — produces point forecasts and prediction intervals
- ~67K parameters, GaussianNLLLoss, AdamW + CosineAnnealing + early stopping

StatsForecast Backtesting (feature24):
- Run global backtest: `make backtest-statsforecast` (vectorized AutoARIMA + AutoETS, ~100x faster than Prophet)
- Run per-cluster backtest: `make backtest-statsforecast-cluster` (only clustered DFUs)
- Run pooled backtest: `make backtest-statsforecast-pooled` (aggregate by cluster -> fit -> disaggregate)
- Load predictions: `make backtest-load` (same shared loader)
- Models appear as `statsforecast_global` / `statsforecast_cluster` / `statsforecast_pooled` in the forecast model selector
- Batch vectorized fitting — no per-DFU loop, Numba JIT compiled

NeuralProphet Backtesting (feature25):
- Run global backtest: `make backtest-neuralprophet` (PyTorch-based, Apple MPS GPU)
- Run per-cluster backtest: `make backtest-neuralprophet-cluster` (only clustered DFUs)
- Run pooled backtest: `make backtest-neuralprophet-pooled` (aggregate by cluster -> fit -> disaggregate)
- Load predictions: `make backtest-load` (same shared loader)
- Models appear as `neuralprophet_global` / `neuralprophet_cluster` / `neuralprophet_pooled` in the forecast model selector
- Per-DFU fitting with multiprocessing (like Prophet) but with PyTorch GPU acceleration

Transfer Learning Backtesting (feature14):
- All tree-based frameworks (LGBM, CatBoost, XGBoost) and deep learning models (PatchTST, DeepAR) support `--cluster-strategy transfer`
- Tree models: Phase 1 trains base on all data, Phase 2 fine-tunes per cluster with warm-start (100 extra trees/iterations, min 20 rows)
- Deep learning: Phase 1 trains global base, Phase 2 fine-tunes per cluster with frozen lower layers (0.1× learning rate)
- Small clusters and unassigned DFUs use base model predictions (not zeroed out)
- Improves accuracy for small clusters compared to `per_cluster` strategy

Accuracy Comparison (feature10):
- Collapsible "Accuracy Comparison" panel in the Forecast analytics page
- Slice by: Cluster, ML Cluster, Supplier, ABC Volume, Region, Brand, Execution Lag, Month
- Filter by lag: execution lag (per DFU) or specific lag 0–4
- Model comparison pivot table: side-by-side Accuracy %, WAPE, Bias per model — best model highlighted in teal
- Lag curve chart: accuracy degradation from lag 0 → lag 4, one line per model
- API endpoints: `GET /forecast/accuracy/slice`, `GET /forecast/accuracy/lag-curve`
- Data source: pre-aggregated `agg_accuracy_by_dim` and `agg_accuracy_lag_archive` views
- Refresh manually: `make accuracy-slice-refresh`

Champion Model Selection (feature15):
- Automatically selects the best-performing model per DFU using industry-standard Forecast Value Added (FVA)
- Per-DFU WAPE evaluation: picks the lowest-WAPE model for each DFU (dmdunit + dmdgroup + loc)
- Champion composite stored as `model_id='champion'` — auto-appears in all accuracy views
- **Ceiling (oracle) model**: picks the best model per DFU **per month** — theoretical upper bound with perfect foresight
- Ceiling stored as `model_id='ceiling'` — benchmarks how close champion gets to the theoretical best
- Gap-to-ceiling metric in the UI shows improvement opportunity (in percentage points)
- Configurable via YAML (`config/model_competition.yaml`) or UI panel in Accuracy tab
- UI: model checkboxes, metric/lag selectors, Run Competition button, champion + ceiling KPI cards, gap indicator, dual model wins bar charts
- CLI: `make champion-select` (runs both champion + ceiling)
- API: `GET/PUT /competition/config`, `POST /competition/run`, `GET /competition/summary`

DFU Analysis (feature17):
- **DFU Analysis tab** overlays sales history and multi-model forecast predictions on a single chart
- Three scope modes: Item @ Location (single DFU), All Items @ Location, Item @ All Locations
- Per-model KPI cards with Accuracy %, WAPE, Bias, Total Forecast, Total Actual
- Toggleable measure visibility — select/deselect sales and individual forecast models
- API: `GET /dfu/analysis?mode=&item=&location=&points=&kpi_months=&sales_metric=`

Market Intelligence (feature18):
- AI-powered market briefings for any product + location pair
- Select an item and location, click "Generate Briefing" to get:
  - Web search results from Google Custom Search (product news, market trends)
  - GPT-4o narrative with market context, state demographics, and demand insights
- Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in `.env`
- API: `POST /market-intelligence`
- UI: "Mi" tab in the navigation bar

UI Overhaul (feature36):
- Collapsible sidebar navigation (9 items, 5 sections) replacing horizontal tabs
- Dashboard overview landing page with KPI cards (sparklines), alerts, heatmap, top movers, forecast trend chart
- Global filter bar: brand, category, market, channel multi-select dropdowns
- Three product themes: Wine & Spirits (burgundy+gold), General (blue SaaS), Obsidian (green+black dark-only)
- Light/dark color modes per theme via CSS variable palettes
- Keyboard: `[` sidebar, `t` theme, `d` dark mode, `1-7` tabs
- API: `GET /domains/{domain}/distinct`, `GET /dashboard/kpis`, `GET /dashboard/alerts`, `GET /dashboard/top-movers`, `GET /dashboard/heatmap`

Inventory Planning (feature34):
- Inventory tab with KPI cards (Total On-Hand, Total On-Order, Avg Lead Time), trend charts, and position table
- Item/location filters with debounce, months selector for date range
- Paginated, sortable position table with clickable rows for item detail drill-down
- Dual Y-axis trend chart showing on-hand, on-order, lead time, and MTD sales over time
- Pipeline: `make inventory-pipeline` (normalize 14 CSVs → load → refresh aggregates)
- Individual steps: `make normalize-inventory`, `make load-inventory`, `make refresh-agg-inventory`
- DDL: `make db-apply-inventory` (creates table + indexes + materialized view)
- API: `GET /inventory/position`, `GET /inventory/kpis`, `GET /inventory/trend`, `GET /inventory/item-detail`

Backtest Cleanup (feature23):
- List model row counts: `make backtest-list`
- Preview deletions: `make backtest-clean MODELS="--dry-run lgbm_global deepar_global"`
- Delete specific models: `make backtest-clean MODELS="lgbm_global deepar_global"`
- Delete all non-external models: `make backtest-clean MODELS="--all-backtest"`
- Removes from `fact_external_forecast_monthly` + `backtest_lag_archive`, refreshes all materialized views
- `--all-backtest` protects `model_id='external'` (source-system forecasts)
- Always `--dry-run` first to preview row counts before deleting

Benchmark Postgres vs Iceberg/Trino:
- endpoint: `GET /bench/compare`
- compares the same query shapes (`count`, `page`, `trend`) for one domain
- returns per-run timings and p50/p95 stats for both backends

Example:
```bash
cd mvp/demand
make api
make bench-compare DOMAIN=sales RUNS=7 ITEM=100320 LOCATION=1401-BULK START_DATE=2023-01-01 END_DATE=2025-01-01
```

If your Trino catalog/schema differs from `iceberg.silver`, override:
```bash
make bench-compare DOMAIN=sales TRINO_CATALOG=iceberg TRINO_SCHEMA=silver
```

Optional analytics path:
```bash
make spark-item
make spark-location
make spark-customer
make spark-time
make spark-dfu
make spark-sales
make spark-forecast
make check-all
```

Optional inventory pipeline:
```bash
make db-apply-inventory     # Create table + indexes + materialized view
make inventory-pipeline     # Normalize 14 monthly CSVs + load into Postgres + refresh agg
```

Optional clustering path (for LGBM model support):
```bash
make cluster-all  # Full pipeline: features -> train -> label -> update
```

Optional job scheduler setup:
```bash
make db-apply-jobs         # Create job_history + job_schedule tables (one-time)
```

Job Scheduler/Monitor with APScheduler (feature39):
- Powered by **APScheduler 3.11** (`BackgroundScheduler` + `ThreadPoolExecutor`)
- Jobs tab (keyboard shortcut `9`) — professional automation dashboard
- 7 job types: Clustering What-If, Full Clustering Pipeline, Seasonality Detection, LGBM/CatBoost/XGBoost Backtest, Champion Selection
- Per-group concurrency control (one active job per group: clustering, backtest, seasonality, champion)
- **Cron/interval scheduling**: create recurring schedules (e.g., daily 2AM backtest, weekly clustering refresh)
- **Job pipelines**: chain multi-step workflows (cluster → backtest → champion select)
- **Retry logic**: configurable max_retries with exponential backoff
- KPI dashboard: Total Jobs, Active Now, Success Rate, Avg Duration
- Grouped job type cards with category colors and "Run Now" / schedule buttons
- Live active job monitoring with animated progress bars and elapsed timers
- Schedule dialog with presets (hourly, 6h, daily 2AM, weekly Mon 2AM)
- Recurring schedules section with cron expression badges
- Expandable job history with params, results, and error detail
- Cross-tab notifications: completed/failed jobs appear as Dashboard alerts
- Sidebar active job count badge
- ClustersTab: "Schedule Scenario Job" button delegates to job system
- API: 12 endpoints — `POST /jobs` → 202, `GET /jobs`, `GET /jobs/active`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`, `DELETE /jobs/{id}`, `GET /jobs/types`, `GET /jobs/stats`, `POST /jobs/schedule`, `GET /jobs/schedules`, `DELETE /jobs/schedules/{id}`, `POST /jobs/pipeline`
- Foundation for agentic AI automation
- **CRITICAL:** `frontend/vite.config.ts` must include a `/jobs` proxy entry — without it, all job API calls from the UI return HTML instead of JSON. Restart `make ui` after adding proxy entries.

## Testing

Full-stack automated testing (485+ tests, <3s total):

Backend (pytest):
```bash
cd mvp/demand
make test              # All backend tests
make test-unit         # Unit tests only (common/ modules)
make test-api          # API endpoint tests only
make test-cov          # With coverage report
```

Frontend (Vitest + React Testing Library):
```bash
cd mvp/demand
make ui-test           # All frontend tests (218 tests)
```

Both:
```bash
cd mvp/demand
make test-all          # Backend + frontend
```

Backend tests cover: `common/metrics.py`, `common/constants.py`, `common/domain_specs.py`, `common/backtest_framework.py`, `common/mlflow_utils.py`, `common/db.py`, and all API endpoints (health, domains, accuracy, DFU analysis, competition, clusters, dashboard, distinct values, jobs).

Frontend tests cover: hooks (`useTheme`, `useUrlState`, `useKeyboardShortcuts`, `useSidebar`, `useGlobalFilters`), utilities (`formatters`, `export`, `queries`), contexts (`JobNotificationContext`, `ScenarioNotificationContext`), components (`Skeleton`, `KeyboardShortcutHelp`, `EChartContainer`, `AppSidebar`, `ThemeSelector`, `GlobalFilterBar`, `WidgetGrid`, `AlertPanel`, `TopMovers`, `HeatmapGrid`), and all tab components (including DashboardTab, JobsTab).

**Rule:** Every new feature must include tests. See `docs/design-specs/feature31.md`.

## Key paths
- Dataset config: `mvp/demand/common/domain_specs.py`
- API: `mvp/demand/api/main.py`
- Frontend app: `mvp/demand/frontend/src/App.tsx`
- Generic normalize script: `mvp/demand/scripts/normalize_dataset_csv.py`
- Generic load script: `mvp/demand/scripts/load_dataset_postgres.py`
- Generic Spark writer: `mvp/demand/scripts/spark_dataset_to_iceberg.py`
- Embeddings generator: `mvp/demand/scripts/generate_embeddings.py`
- Clustering scripts: `mvp/demand/scripts/generate_clustering_features.py`, `train_clustering_model.py`, `label_clusters.py`, `update_cluster_assignments.py`
- Shared backtest framework: `mvp/demand/common/backtest_framework.py`, `feature_engineering.py`, `metrics.py`, `mlflow_utils.py`, `db.py`, `constants.py`
- Backtest scripts: `mvp/demand/scripts/run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`, `run_backtest_prophet.py`, `run_backtest_patchtst.py`, `run_backtest_deepar.py`, `run_backtest_statsforecast.py`, `run_backtest_neuralprophet.py`, `load_backtest_forecasts.py`
- Deep learning models: `mvp/demand/scripts/patchtst_model.py`, `deepar_model.py`
- Champion selection script: `mvp/demand/scripts/run_champion_selection.py`
- Job engine: `mvp/demand/common/job_registry.py`
- Jobs router: `mvp/demand/api/routers/jobs.py`
- Jobs tab: `mvp/demand/frontend/src/tabs/JobsTab.tsx`
- Clustering config: `mvp/demand/config/clustering_config.yaml`
- Competition config: `mvp/demand/config/model_competition.yaml`
- Inventory normalize: `mvp/demand/scripts/normalize_inventory_csv.py`
- Inventory DDL: `mvp/demand/sql/017_create_fact_inventory_snapshot.sql`
- DDL: `mvp/demand/sql/` (001–008 dataset DDL, 009 chat embeddings, 010 backtest lag archive, 011 accuracy slice views, 017 inventory snapshot, 018 dashboard views)
- Backtest cleanup: `mvp/demand/scripts/clean_backtest_models.py`
- Theme configs: `mvp/demand/frontend/src/constants/themes/` (wineSpirits, general, obsidian)
- Sidebar + filters: `mvp/demand/frontend/src/components/AppSidebar.tsx`, `GlobalFilterBar.tsx`
- Dashboard: `mvp/demand/frontend/src/tabs/DashboardTab.tsx`
- Backend tests: `mvp/demand/tests/` (unit/ + api/)
- Frontend tests: `mvp/demand/frontend/src/**/__tests__/`
- Test config: `mvp/demand/frontend/vitest.config.ts`
- Design specs: `docs/design-specs/` (feature1–feature39)
