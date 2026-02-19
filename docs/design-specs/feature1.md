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
- **Feature 18:** Market intelligence — AI-powered market briefings combining Google web search + GPT-4o narrative synthesis for item + location pairs, with demographic context and demand insights

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
