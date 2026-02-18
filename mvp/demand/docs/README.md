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

Sales source details (MVP):
- input file: `datafiles/dfu_lvl2_hist.txt`
- load rule: only `TYPE=1`
- monthly grain date: `STARTDATE` in `YYYYMMDD`, day must be `01`

Forecast source details (MVP):
- input file: `datafiles/dfu_stat_fcst.txt`
- monthly grain dates: `fcstdate`, `startdate` (both month-start)
- lag rule: `lag = month_diff(startdate, fcstdate)` with allowed range `0..4`
- `model_id`: identifies forecasting algorithm (default `'external'`); uniqueness is `(forecast_ck, model_id)`

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
- React + Vite + Tailwind + shadcn/ui + Recharts
- Spark + Iceberg + MinIO
- Trino
- MLflow (experiment tracking, model registry)
- LightGBM (demand forecasting models)
- scikit-learn (clustering algorithms)
- OpenAI (GPT-4o + text-embedding-3-small) for NL→SQL chatbot
- Docker Compose

## Performance defaults
- Fact indexes on `(dmdunit, loc, startdate/fcstdate)` are applied via `sql/008_perf_indexes_and_agg.sql`.
- Trigram (`pg_trgm`) indexes are applied for common `ILIKE` search fields.
- Monthly materialized views are maintained for faster trend analytics:
  - `agg_sales_monthly`
  - `agg_forecast_monthly`
- Accuracy slice views for O(1) aggregate KPI queries (feature16):
  - `agg_accuracy_by_dim` — pre-joins forecast + DFU attributes at (model, lag, month, cluster, supplier, abc_vol, region, brand) grain
  - `agg_accuracy_lag_archive` — same from archive table; powers lag-curve analysis
- `load-sales`, `load-forecast`, and `load-all` refresh `agg_*` aggregates automatically.
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
- Load predictions: `make backtest-load` (loads execution-lag rows into `fact_external_forecast_monthly`, all-lag rows into `backtest_lag_archive`, refreshes accuracy slice views)
- Or all at once: `make backtest-all`
- Models appear as `lgbm_global` / `lgbm_cluster` in the forecast model selector
- Existing accuracy KPIs and trend charts work automatically for LGBM models
- `backtest_lag_archive` stores lag 0–4 predictions for accuracy reporting at any horizon
- `make backtest-load` only replaces rows for the model_id in the CSV (safe to run per-cluster after global)

Accuracy Comparison (feature16):
- Collapsible "Accuracy Comparison" panel in the Forecast analytics page
- Slice by: Cluster, ML Cluster, Supplier, ABC Volume, Region, Brand, Execution Lag, Month
- Filter by lag: execution lag (per DFU) or specific lag 0–4
- Model comparison pivot table: side-by-side Accuracy %, WAPE, Bias per model — best model highlighted in teal
- Lag curve chart: accuracy degradation from lag 0 → lag 4, one line per model
- API endpoints: `GET /forecast/accuracy/slice`, `GET /forecast/accuracy/lag-curve`
- Data source: pre-aggregated `agg_accuracy_by_dim` and `agg_accuracy_lag_archive` views
- Refresh manually: `make accuracy-slice-refresh`

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

Optional clustering path (for LGBM model support):
```bash
make cluster-all  # Full pipeline: features -> train -> label -> update
```

## Key paths
- Dataset config: `mvp/demand/common/domain_specs.py`
- API: `mvp/demand/api/main.py`
- Frontend app: `mvp/demand/frontend/src/App.tsx`
- Generic normalize script: `mvp/demand/scripts/normalize_dataset_csv.py`
- Generic load script: `mvp/demand/scripts/load_dataset_postgres.py`
- Generic Spark writer: `mvp/demand/scripts/spark_dataset_to_iceberg.py`
- Embeddings generator: `mvp/demand/scripts/generate_embeddings.py`
- Clustering scripts: `mvp/demand/scripts/generate_clustering_features.py`, `train_clustering_model.py`, `label_clusters.py`, `update_cluster_assignments.py`
- Backtest scripts: `mvp/demand/scripts/run_backtest.py`, `load_backtest_forecasts.py`
- Clustering config: `mvp/demand/config/clustering_config.yaml`
- DDL: `mvp/demand/sql/` (001–008 dataset DDL, 009 chat embeddings, 010 backtest lag archive, 011 accuracy slice views)
- Design specs: `docs/design-specs/` (feature0–feature16)
