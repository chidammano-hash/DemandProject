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
9. LGBM backtesting:
   - Expanding window backtest (10 timeframes A–J) with LightGBM regressors
   - Global strategy (`lgbm_global`): one model, `ml_cluster` as categorical feature
   - Per-cluster strategy (`lgbm_cluster`): separate model per cluster
   - Causal feature engineering: lag 1-12, rolling stats, calendar, DFU/item attributes
   - Execution-lag predictions loaded into `fact_external_forecast_monthly` via COPY + upsert
   - All-lag (0–4) predictions archived in `backtest_lag_archive` for accuracy at any horizon
   - MLflow experiment tracking (`demand_backtest`)
10. Multi-dimensional accuracy slicing:
   - Pre-aggregated `agg_accuracy_by_dim` view: (model_id, lag, month, cluster, supplier, abc_vol, region, brand) grain
   - Pre-aggregated `agg_accuracy_lag_archive` view: same grain for archive table + timeframe
   - `/forecast/accuracy/slice` endpoint: compare WAPE, Accuracy %, Bias across models by any DFU attribute
   - `/forecast/accuracy/lag-curve` endpoint: accuracy degradation by lag horizon (0–4) per model
   - UI Accuracy Comparison panel: model comparison pivot table + lag curve chart
   - Views refreshed automatically by `backtest-load`; also manually via `make accuracy-slice-refresh`

## Additional tables
1. `chat_embeddings` — pgvector table storing schema metadata embeddings (1536-dim) for NL query context retrieval
2. `backtest_lag_archive` — stores all-lag (0–4) backtest predictions for accuracy reporting at any horizon; grain: `(forecast_ck, model_id, lag)`; includes `timeframe` column (A–J) for traceability

## Accuracy Slice Materialized Views (feature16)
Pre-aggregated views enabling O(1) multi-dimensional KPI slicing without raw-table joins:

1. `agg_accuracy_by_dim` — joins `fact_external_forecast_monthly` + `dim_dfu`, aggregates at (model_id, lag, month, cluster, supplier, abc_vol, region, brand, execution_lag) grain; stores `SUM(F)`, `SUM(A)`, `SUM(ABS(F-A))` for KPI derivation. Refreshed by `backtest-load`.
2. `agg_accuracy_lag_archive` — same aggregation from `backtest_lag_archive` + `dim_dfu`, adds `timeframe` grain; used for lag-horizon accuracy curves. Refreshed by `backtest-load`.

Performance impact: aggregate queries (cluster-level, supplier-level) drop from 5–30s → <300ms.

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
5. **LGBM Backtest** (`run_backtest.py`):
   - Expanding window training with 10 timeframes (A–J)
   - Feature engineering: lag 1-12, rolling mean/std 3/6/12m, calendar, DFU/item attributes
   - Global and per-cluster training strategies
   - Outputs two CSVs: execution-lag only (main table) + all lags 0–4 (archive)
   - Deduplication across timeframes (latest timeframe wins)
   - MLflow logging to `demand_backtest` experiment
6. **Backtest Loader** (`load_backtest_forecasts.py`):
   - Loads execution-lag rows into `fact_external_forecast_monthly` via COPY + staging + upsert
   - Loads all-lag rows into `backtest_lag_archive` via same pattern
   - `--replace` scoped to `model_id` in CSV (safe for multi-model coexistence)
   - Refreshes `agg_forecast_monthly`, `agg_accuracy_by_dim`, `agg_accuracy_lag_archive` materialized views

## How to add next dataset
1. Add `<DATASET>_SPEC` in `common/domain_specs.py`
2. Add matching DDL in `sql/`
3. Add Make targets:
   - `normalize-<dataset>`
   - `load-<dataset>`
   - `spark-<dataset>` (calls generic spark script with `--dataset <dataset>`)
4. API uses existing generic `/domains/{domain}/...` endpoints
5. UI uses existing shared frontend tabs from `/domains`
6. Run `make generate-embeddings` to update chat context with new schema metadata
