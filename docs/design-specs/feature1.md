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
- `FastAPI` backend and `React/Next.js` frontend
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
- **Feature 15:** Champion model selection — per-DFU best-of-models pick via WAPE, ceiling (oracle) model for theoretical upper bound, gap-to-ceiling analysis, UI-editable competition config, FVA analysis
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
