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
| Charts | Recharts |
| Database | PostgreSQL 16 |
| Lakehouse | Apache Iceberg via MinIO + Iceberg REST |
| Big Data | Apache Spark 3.5 |
| Query Engine | Trino |
| ML / Clustering | scikit-learn, pandas, scipy, matplotlib, seaborn |
| ML Tracking | MLflow |
| Python packaging | uv |
| Build | Make |
| Containers | Docker Compose |

---

## Key Files

| File | Purpose |
|---|---|
| `mvp/demand/common/domain_specs.py` | Central config: all 7 datasets (dimensions + facts) with columns, types, keys |
| `mvp/demand/api/main.py` | FastAPI backend — all endpoints live here |
| `mvp/demand/frontend/src/App.tsx` | React UI — full data explorer + analytics |
| `mvp/demand/Makefile` | All dev commands |
| `mvp/demand/docker-compose.yml` | 7-service infra cluster |
| `mvp/demand/scripts/normalize_dataset_csv.py` | Generic ETL: CSV → clean CSV |
| `mvp/demand/scripts/load_dataset_postgres.py` | Generic loader: clean CSV → PostgreSQL |
| `mvp/demand/scripts/spark_dataset_to_iceberg.py` | Spark job: clean CSV → Iceberg |
| `mvp/demand/sql/` | DDL for all tables, indexes, materialized views |
| `mvp/demand/scripts/generate_clustering_features.py` | Feature engineering: sales history → clustering feature matrix |
| `mvp/demand/scripts/train_clustering_model.py` | KMeans clustering with optimal K selection + MLflow logging |
| `mvp/demand/scripts/label_clusters.py` | Assign business labels to clusters based on feature centroids |
| `mvp/demand/scripts/update_cluster_assignments.py` | Write cluster labels to `dim_dfu.cluster_assignment` in Postgres |
| `mvp/demand/config/clustering_config.yaml` | Clustering hyperparameters and labeling thresholds |
| `mvp/demand/scripts/run_backtest.py` | LGBM backtest: expanding-window training + prediction |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load backtest predictions into Postgres (main + archive) |
| `mvp/demand/sql/010_create_backtest_lag_archive.sql` | DDL for backtest all-lags archive table |
| `docs/design-specs/` | Feature specs (feature1b–feature1g, feature10–feature15) |

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
make normalize-all     # Normalize all 7 datasets (CSV → clean CSV)
make load-all          # Load cleaned data into Postgres + refresh materialized views
make spark-all         # Publish datasets to Iceberg (optional)

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

# Backtesting (LGBM)
make backtest-lgbm          # Run global LGBM backtest (10 expanding timeframes)
make backtest-lgbm-cluster  # Run per-cluster LGBM backtest
make backtest-load          # Load backtest predictions into Postgres + refresh agg
make backtest-all           # backtest-lgbm + backtest-load
```

---

## Architecture

### Domain-Driven Generic Design

All datasets extend a single `DomainSpec` dataclass in `common/domain_specs.py`. Scripts and API endpoints are generic — they operate on any domain via `--dataset <name>` or `/domains/{domain}/*`.

**7 Domains:**
- **Dimensions (read-only):** `item`, `location`, `customer`, `time`, `dfu`
- **Facts (time-series):** `sales`, `forecast`

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

- All domains served via: `GET /domains/{domain}/rows`, `GET /domains/{domain}/search`, etc.
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

### Archive Tables
- `backtest_lag_archive`: All-lags (0–4) backtest predictions for accuracy analysis at any horizon. Grain = forecast_ck + model_id + lag. Includes `timeframe` column for traceability.

### Materialized Views
- `agg_sales_monthly`, `agg_forecast_monthly` — pre-aggregated for O(1) KPI queries

---

## Frontend Features

- Paginated data explorer with column filtering and sorting
- KPI cards: Accuracy %, WAPE, MAPE, Bias, Total Forecast/Actual
- KPI window selector: 1–12 month rolling window
- Multi-metric trend charts (dual Y-axis: volume left, accuracy % right)
- Item/Location filter with typeahead suggestions
- Postgres vs Iceberg latency benchmarking panel

---

## Important Conventions

- **Null normalization:** `''`, `'null'`, `'none'`, `'NA'` all treated as NULL during load
- **Type casting:** Integer/float/date fields auto-cast with null coercion in normalize scripts
- **Lag computation:** `month_diff` auto-computed during forecast normalization
- **Forecast accuracy formula:** `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- **Bias formula:** `(SUM(Forecast) / SUM(History)) - 1`
- **Sales filtering:** Only rows with `TYPE=1` are loaded into `fact_sales_monthly`
- **Time dimension:** Auto-generated 2020–2035, not sourced from a file
- **Forecast model_id:** Identifies the forecasting algorithm; default `'external'` for source-system forecasts. `UNIQUE(forecast_ck, model_id)` constraint prevents duplicates within a model. Not part of the business key.
- **Chat endpoint:** `POST /chat` — OpenAI-powered NL→SQL with pgvector context retrieval. Read-only execution with 5s timeout and 500-row limit. Requires `OPENAI_API_KEY` in `.env`.
- **DFU clustering:** KMeans-based clustering pipeline groups DFUs by demand patterns. Feature engineering extracts time series, item, and DFU features. Cluster labels (e.g., `high_volume_steady`, `seasonal_medium_volume`) stored in `dim_dfu.cluster_assignment`. MLflow tracks experiments under `dfu_clustering`. Config in `config/clustering_config.yaml`.

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

---

## Documentation Update Rules

**Whenever you implement a new feature or make significant changes, you MUST update the following files:**

1. **`docs/design-specs/feature<N>.md`** — Create or update the design spec for the feature
2. **`docs/design-specs/feature1.md`** — Add the feature to the "Implemented Features (MVP)" list
3. **`mvp/demand/docs/ARCHITECTURE.md`** — Update architecture, component technologies, tables, or data flow if affected
4. **`mvp/demand/docs/README.md`** — Update stack, datasets, analytics behavior, quick start, or key paths if affected
5. **`mvp/demand/docs/RUNBOOK.md`** — Update setup steps, notes, or troubleshooting if affected
6. **`CLAUDE.md`** (this file) — Update Key Files, Common Commands, Data Models, Frontend Features, Important Conventions, or Design Specs list if affected

**What counts as "significant changes":**
- New feature implementation (new endpoints, UI panels, tables, scripts)
- Schema changes (new columns, tables, indexes, materialized views)
- New dependencies or infrastructure changes (docker images, pyproject.toml)
- New Make targets or CLI commands
- Changes to data flow or pipeline behavior

**What does NOT require doc updates:**
- Bug fixes that don't change behavior or interfaces
- Minor code refactors that don't change architecture
- Typo corrections

---

## Do Not

- Do not commit `__pycache__/`, `.pyc` files, or `.venv/`
- Do not modify `mvp/demand/data/*.csv` files manually — they are generated by normalize scripts
- Do not touch the `reference/` directory — it is archived code
- Do not run `make spark-all` unless Iceberg/MinIO is needed; Postgres path is sufficient for most dev work
