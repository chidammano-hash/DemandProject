# Databricks Migration Plan — Demand Studio

## Executive Summary

Demand Studio currently runs on PostgreSQL 16 + 7 Docker services (Postgres, MLflow, MinIO, Iceberg REST, Spark, Trino, app). After exhaustive analysis of every SQL file, API router, data pipeline script, and infrastructure component, **the migration to Databricks is fully feasible**. The codebase has a clean architecture that makes an incremental, dual-run approach possible.

**60-70% of the codebase requires zero changes** (all frontend, all ML training, feature engineering, metrics, configs). The remaining 30-40% is concentrated in 5 specific areas: the DB connection layer, SQL dialect, bulk loading, materialized views, and PostgreSQL extensions (pg_trgm, pgvector).

---

## Table of Contents

1. [Migration Scope Summary](#1-migration-scope-summary)
2. [Current Architecture Audit](#2-current-architecture-audit)
3. [What Stays Untouched (Zero Changes)](#3-what-stays-untouched-zero-changes)
4. [What Must Change](#4-what-must-change)
5. [Two Migration Paths](#5-two-migration-paths)
6. [Option A — Databricks Lakebase (Lowest Effort)](#6-option-a--databricks-lakebase-lowest-effort)
7. [Option B — Full Delta Lake Migration](#7-option-b--full-delta-lake-migration)
8. [DB Abstraction Layer Design](#8-db-abstraction-layer-design)
9. [SQL Dialect Translation Reference](#9-sql-dialect-translation-reference)
10. [PostgreSQL-Specific Constructs by File](#10-postgresql-specific-constructs-by-file)
11. [Infrastructure Mapping](#11-infrastructure-mapping)
12. [Test Strategy](#12-test-strategy)
13. [Risk Assessment](#13-risk-assessment)
14. [Effort Estimates](#14-effort-estimates)
15. [Verification Plan](#15-verification-plan)

---

## 1. Migration Scope Summary

| Category | Files Affected | Effort |
|---|---|---|
| DB connection layer | `api/core.py`, `common/db.py` | High |
| SQL dialect (`::casts`, `ILIKE`, `pg_class`, etc.) | All 8 routers, 16 DDL files | Medium (mechanical) |
| Bulk load (`COPY FROM STDIN`) | 5 scripts + `competition.py` router | High |
| Materialized views (6 views) | DDL + refresh calls in 3 files | High |
| pgvector (chat embeddings) | `chat.py` + `generate_embeddings.py` | Medium |
| pg_trgm (GIN trigram search) | `core.py` search helpers | Medium |
| ML training code | None | **Zero changes** |
| Frontend | None | **Zero changes** |
| Config YAMLs | None | **Zero changes** |

---

## 2. Current Architecture Audit

### 2.1 Database Connection Layer

**`common/db.py`** — Returns psycopg connection kwargs (`host`, `port`, `dbname`, `user`, `password`) from env vars.

**`api/core.py`** — Uses `psycopg_pool.ConnectionPool` with `min_size=2, max_size=10`. Every API query flows through `get_conn()` which checks out from this pool. All routers use the pattern:

```python
with get_conn() as conn, conn.cursor() as cur:
    cur.execute(sql, params)
    rows = cur.fetchall()
```

### 2.2 psycopg-Specific Features Used

| Feature | Where Used | Databricks Impact |
|---|---|---|
| `ConnectionPool` (psycopg_pool) | `api/core.py` | Replace with `databricks-sql-connector` |
| `cur.copy("COPY ... FROM STDIN")` | 5 bulk load scripts, `competition.py` | Replace with Delta MERGE/append |
| `pd.read_sql(sql, psycopg_conn)` | All backtest/clustering scripts | Replace connection object |
| `%s` positional params | Every SQL query in the codebase | Change to `?` placeholders |
| `SET LOCAL statement_timeout` | `chat.py` | Remove (use connector timeout) |
| `SET TRANSACTION READ ONLY` | `chat.py` | Remove (enforce in app layer) |
| `SET synchronous_commit = off` | Bulk load scripts | Remove (not applicable to Delta) |
| `SET work_mem` / `maintenance_work_mem` | Bulk load scripts | Remove (not applicable) |

### 2.3 PostgreSQL Extensions

| Extension | Usage | Databricks Equivalent |
|---|---|---|
| `pg_trgm` | 12 GIN trigram indexes powering `ILIKE` substring search | Z-ORDER + `LIKE` / app-layer cache |
| `pgvector` | `vector(1536)` column + `<=>` cosine similarity in chat | Databricks Vector Search |

### 2.4 Materialized Views (6 total)

| View | Purpose | Refresh Trigger |
|---|---|---|
| `agg_sales_monthly` | Monthly aggregate of `fact_sales_monthly` | After data loads |
| `agg_forecast_monthly` | Monthly aggregate of `fact_external_forecast_monthly` | After backtest loads |
| `agg_accuracy_by_dim` | Accuracy by DFU attributes + seasonality | After champion selection |
| `agg_accuracy_lag_archive` | Lag-horizon accuracy from archive | After backtest loads |
| `agg_dfu_coverage` | Distinct DFU coverage per model/lag | After backtest loads |
| `agg_dfu_coverage_lag_archive` | Same, archive source | After backtest loads |

Refreshed via `REFRESH MATERIALIZED VIEW` in `competition.py`, `load_backtest_forecasts.py`, and `clean_backtest_models.py`.

### 2.5 API Routers — SQL Pattern Analysis

| Router | Key PG-Specific Patterns |
|---|---|
| `domains.py` | `ILIKE`, `TABLESAMPLE SYSTEM(1)`, `random()`, `pg_class` catalog query, `to_regclass()` |
| `accuracy.py` | `::date` / `::text` / `::bigint` casts, reads from materialized views |
| `analysis.py` | `::date` casts, reads `agg_sales_monthly` / `agg_forecast_monthly` |
| `competition.py` | `CREATE TEMP TABLE ON COMMIT DROP`, `COPY FROM STDIN`, `REFRESH MATERIALIZED VIEW`, `ROW_NUMBER()` window functions |
| `clusters.py` | `STDDEV()` (supported in Spark SQL), reads `dim_dfu` |
| `chat.py` | pgvector `<=>` operator, `SET LOCAL statement_timeout`, `SET TRANSACTION READ ONLY` |
| `benchmark.py` | `docker exec trino` CLI invocation for Trino queries |
| `intel.py` | Simple SELECTs from `dim_item`, `dim_location`, `agg_sales_monthly` |

**No ORM used.** Every query is hand-written raw SQL with `%s` parameterized placeholders.

### 2.6 Data Pipeline Scripts

**Pure pandas + psycopg (no Spark):**
- `normalize_dataset_csv.py` — CSV clean (pandas). **Portable as-is.**
- `load_dataset_postgres.py` — COPY bulk load. **Must rewrite.**
- `load_backtest_forecasts.py` — COPY + staging + batched INSERT. **Must rewrite.**
- `generate_clustering_features.py` — `pd.read_sql` from Postgres. **Change connection only.**
- `detect_seasonality.py` — `pd.read_sql` from Postgres. **Change connection only.**
- `update_cluster_assignments.py` — COPY into temp table, UPDATE via JOIN. **Must rewrite.**
- `update_seasonality_profiles.py` — Same pattern. **Must rewrite.**
- `run_champion_selection.py` — Complex CTEs, COPY + temp table. **Must rewrite.**
- `generate_embeddings.py` — OpenAI embeddings + psycopg write. **Change connection only.**

**Uses Spark (already Databricks-compatible):**
- `spark_dataset_to_iceberg.py` — PySpark job. Change SparkSession config and catalog references.

**ML Backtest Scripts (change data loading only):**
- All 6 backtest scripts (`run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`, `run_backtest_statsforecast.py`, `run_backtest_prophet.py`, `run_backtest_neuralprophet.py`) load data via `common/backtest_framework.py`'s `load_backtest_data()` which uses `psycopg.connect(**db)` + `pd.read_sql()`. **Update `load_backtest_data()` once → all 6 scripts fixed.**

### 2.7 Python Dependencies

**PostgreSQL-specific (must replace):**

| Package | Replacement |
|---|---|
| `psycopg[binary,pool]>=3.2.0` | `databricks-sql-connector` |

**All other packages are database-agnostic:** fastapi, uvicorn, pydantic, openai, scikit-learn, numpy, pandas, scipy, mlflow, lightgbm, catboost, xgboost, prophet, torch, statsforecast, neuralprophet.

---

## 3. What Stays Untouched (Zero Changes)

| Component | Why |
|---|---|
| **All frontend code** (React/Vite/TypeScript) | Calls FastAPI over HTTP — backend-agnostic |
| **All ML training code** (LightGBM, CatBoost, XGBoost, StatsForecast, Prophet, NeuralProphet) | Pure pandas once data is loaded |
| **`common/feature_engineering.py`** | Pure pandas/numpy |
| **`common/metrics.py`** | Pure pandas (WAPE, bias, accuracy) |
| **`common/constants.py`** | Pure Python constants |
| **`common/domain_specs.py`** | Pure Python dataclasses — no DB dependency |
| **Config YAMLs** (`clustering_config.yaml`, `seasonality_config.yaml`, `model_competition.yaml`) | No DB dependency |
| **MLflow Python SDK calls** | Only tracking URI changes (`"databricks"` instead of `localhost:5003`) |
| **FastAPI app structure** | Routers, middleware, CORS, Pydantic models — all framework-level |
| **Backtest orchestration logic** | `generate_timeframes()`, `postprocess_predictions()`, `save_backtest_output()` — all pandas |
| **`normalize_dataset_csv.py`** | Pure pandas CSV processing |
| **Auth module** (`api/auth.py`) | No DB dependency |

---

## 4. What Must Change

### 4.1 High-Impact Changes

#### 4.1.1 Connection Layer (`api/core.py`, `common/db.py`)
- Replace `psycopg_pool.ConnectionPool` with `databricks.sql.connect()`
- Replace `get_conn()` context manager to yield Databricks connection
- Replace `%s` params with `?` in every SQL query

#### 4.1.2 COPY FROM STDIN → Delta MERGE/Append
Used in 5+ scripts. The `COPY ... FROM STDIN` streaming bulk load has no Databricks equivalent. Replace with:
- `spark.createDataFrame(df).write.format("delta").mode("append").saveAsTable(...)`
- Or `MERGE INTO` for upsert semantics

#### 4.1.3 Materialized Views → Databricks MVs or DLT
6 PostgreSQL materialized views need Databricks equivalents. Options:
- **Databricks Materialized Views** (GA) — closest analog, supports incremental refresh
- **Delta Live Tables** (DLT) — streaming/triggered pipelines
- **Pre-computed Delta tables** with scheduled refresh jobs

#### 4.1.4 pg_trgm GIN Trigram Indexes → Search Strategy
12 GIN trigram indexes power `ILIKE` substring search. Replace with:
- Z-ORDER on searched columns + `UPPER(col) LIKE UPPER(?)`
- Application-layer typeahead cache for suggest endpoint

#### 4.1.5 pgvector → Databricks Vector Search
Chat endpoint uses `embedding <=> %s::vector` cosine similarity. Replace with:
- Delta table with `ARRAY<FLOAT>` embedding column
- Databricks Vector Search endpoint + Delta Sync Index

### 4.2 Medium-Impact Changes

#### 4.2.1 SQL Dialect Translation
Pervasive throughout all router SQL:
- `::text`, `::bigint`, `::date`, `::double precision` → `CAST(... AS ...)` or remove
- `random()` → `rand()`
- `TABLESAMPLE SYSTEM (1)` → `TABLESAMPLE (1 PERCENT)`
- `date_trunc('month', x)::date` → `CAST(DATE_TRUNC('MONTH', x) AS DATE)`
- `DISTINCT ON (col)` → `ROW_NUMBER() OVER (PARTITION BY col ORDER BY ...) = 1`

#### 4.2.2 System Catalog Queries
- `pg_class.reltuples` (approximate row count) → `DESCRIBE DETAIL` or `COUNT(*)`
- `to_regclass()` (table existence check) → `SHOW TABLES` or try/except
- `pg_constraint` queries → not needed (enforce at app layer)

#### 4.2.3 Session-Level Settings
- `SET LOCAL statement_timeout = '5000'` → connector `operation_timeout` parameter
- `SET TRANSACTION READ ONLY` → enforce in app layer (existing `_is_safe_sql()` already validates)
- `SET synchronous_commit = off` / `SET work_mem` → not applicable to Delta

#### 4.2.4 Temp Tables
- `CREATE TEMP TABLE ... ON COMMIT DROP` → Spark session-scoped temp views or staging Delta tables

### 4.3 Low-Impact Changes

| Item | Change |
|---|---|
| `BIGSERIAL PRIMARY KEY` | `BIGINT GENERATED ALWAYS AS IDENTITY` |
| `TIMESTAMPTZ` | `TIMESTAMP` |
| `UNIQUE` constraints | Enforce via `MERGE INTO` dedup logic (Delta has no native UNIQUE) |
| `CHECK` constraints (computed) | Enforce at ETL/app layer |
| `ON CONFLICT DO UPDATE` | `MERGE INTO ... WHEN MATCHED THEN UPDATE` |
| `COMMENT ON COLUMN` | `ALTER TABLE ... ALTER COLUMN ... COMMENT` |
| `random()` | `rand()` |
| `NULLS LAST` | Supported in Spark SQL |
| MLflow tracking URI | Change from `http://localhost:5003` to `"databricks"` |

---

## 5. Two Migration Paths

| Criterion | Option A: Lakebase | Option B: Full Delta Lake |
|---|---|---|
| **Code changes** | Near-zero (connection string only) | Significant (DB layer rewrite) |
| **Timeline** | 1-2 weeks | 12-16 weeks (full) / 5-7 weeks (read-only first) |
| **PostgreSQL compatibility** | Full (psycopg3, pg_trgm, pgvector, MVs) | None (must translate everything) |
| **Delta Lake benefits** | No | Yes (time travel, Z-ORDER, Photon, DLT) |
| **Spark processing** | No | Yes |
| **Unity Catalog governance** | Yes | Yes |
| **Managed MLflow** | Yes | Yes |
| **Managed infrastructure** | Yes (eliminates Docker) | Yes (eliminates Docker) |
| **Cost model** | Lakebase pricing | SQL Warehouse DBU pricing |

---

## 6. Option A — Databricks Lakebase (Lowest Effort)

Databricks Lakebase is a PostgreSQL-compatible OLTP engine inside Databricks. It supports:
- `psycopg3` + `psycopg_pool.ConnectionPool` natively
- All `::cast` syntax
- `ILIKE` + trigram-style search
- `pgvector` for embeddings
- Materialized views with `REFRESH`
- `COPY FROM STDIN` bulk loading

### Migration Steps

1. **Provision Lakebase instance** in Databricks workspace
2. **Update connection string** in `common/db.py` (host, port, credentials)
3. **Apply DDL** — run all 16 SQL files against Lakebase (they work as-is)
4. **Migrate data** — `pg_dump` from existing Postgres → `pg_restore` into Lakebase (or re-run `make load-all`)
5. **Update MLflow** — set tracking URI to `"databricks"`
6. **Retire Docker services** — no more self-managed Postgres, MLflow, MinIO, Spark, Trino
7. **Test** — run `make test-all` (297+ tests)

### Trade-offs
- **Pro:** Minimal risk, fastest time to production, preserves all existing code
- **Con:** You're still on a PostgreSQL-compatible engine — no Delta Lake time travel, no Photon acceleration, no DLT pipelines, no Spark-native processing

---

## 7. Option B — Full Delta Lake Migration

### Phase 0: Prerequisites (1-2 weeks, no code changes)

1. Provision Databricks workspace
2. Create Unity Catalog: `demand_studio` catalog with `silver` schema (dim/fact tables) and `gold` schema (aggregates)
3. Create Serverless SQL Warehouse (best for API-serving latency)
4. Set environment variables:
   ```bash
   DATABRICKS_HOST=<workspace-url>
   DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>
   DATABRICKS_TOKEN=<access-token>
   ```
5. Add `databricks-sql-connector` to `pyproject.toml`
6. Switch MLflow tracking URI to `"databricks"` in `common/mlflow_utils.py`

### Phase 1: DB Abstraction Layer + Dual-Run (2-3 weeks)

**Goal:** Both Postgres and Databricks can serve the API. A single env var switches backends.

Create `api/backends/` with a backend interface:

```
mvp/demand/api/backends/
    __init__.py
    base.py          # AbstractBackend: connection(), execute(), translate_sql()
    postgres.py      # Wraps existing psycopg_pool (extracted from core.py)
    databricks.py    # Wraps databricks-sql-connector + SQL dialect translator
```

Key design: `get_conn()` in `core.py` delegates to the active backend selected by `DB_BACKEND` env var. All 8 routers continue calling `with get_conn() as conn, conn.cursor() as cur:` unchanged.

The `DatabricksBackend.translate_sql()` handles mechanical SQL rewrites:
- `%s` → `?` (parameter placeholders)
- `::text/::bigint/::date/::double precision` → remove or `CAST()`
- `random()` → `rand()`
- `TABLESAMPLE SYSTEM (1)` → `TABLESAMPLE (1 PERCENT)`
- `pg_class` queries → `COUNT(*)` or `DESCRIBE DETAIL`
- `to_regclass()` → try/except or `SHOW TABLES`

**Files to modify:** `api/core.py`, `common/db.py`
**Files to create:** `api/backends/base.py`, `api/backends/postgres.py`, `api/backends/databricks.py`

### Phase 2: Delta Lake Schema + Data Migration (2-3 weeks, parallel with Phase 1)

Translate 16 SQL DDL files to Databricks SQL:

| PostgreSQL | Databricks SQL |
|---|---|
| `BIGSERIAL PRIMARY KEY` | `BIGINT GENERATED ALWAYS AS IDENTITY` |
| `TIMESTAMPTZ DEFAULT NOW()` | `TIMESTAMP DEFAULT CURRENT_TIMESTAMP()` |
| `NUMERIC(10,4)` | `DECIMAL(10,4)` |
| `CREATE EXTENSION IF NOT EXISTS pg_trgm` | Remove (not needed) |
| `CREATE EXTENSION IF NOT EXISTS vector` | Remove (use Vector Search) |
| `USING gin (col gin_trgm_ops)` | Remove (use Z-ORDER) |
| `DO $$ BEGIN ... END $$` PL/pgSQL | Rewrite as Python migrations |
| `::regclass` | Remove |
| `CONSTRAINT ... UNIQUE (ck, model_id)` | Enforce via MERGE dedup |
| `embedding vector(1536)` | `embedding ARRAY<FLOAT>` |

Load data pipeline:
```
CSV → pandas.read_csv() → spark.createDataFrame(df) → df.write.format("delta").saveAsTable("demand_studio.silver.<table>")
```

Apply Z-ORDER on fact tables:
```sql
OPTIMIZE demand_studio.silver.fact_sales_monthly ZORDER BY (dmdunit, loc, startdate);
OPTIMIZE demand_studio.silver.fact_external_forecast_monthly ZORDER BY (dmdunit, loc, fcstdate, model_id);
```

**Files to create:** `sql/databricks/` (16 translated DDL files), `scripts/load_dataset_databricks.py`

### Phase 3: Search Strategy (1-2 weeks)

Replace pg_trgm GIN trigram indexes with:

1. **For `build_where()` / `fetch_page()`** (general column filtering):
   - Convert `ILIKE` to `UPPER(col) LIKE UPPER(?)` (or use Databricks native `ILIKE` support)
   - Z-ORDER on frequently searched columns
   - Delta data skipping provides adequate performance for current data volumes

2. **For suggest endpoint** (typeahead):
   - Application-layer cache: pre-load top distinct values per text field per domain at API startup
   - Refresh cache every 5 minutes via background task
   - Eliminates DB round-trip for common typeahead queries

### Phase 4: Materialized Views → Databricks MVs or DLT (2-3 weeks)

Replace 6 PostgreSQL materialized views. **Three options:**

**Option 4A — Databricks Materialized Views (Recommended):**
```sql
CREATE MATERIALIZED VIEW demand_studio.gold.agg_sales_monthly AS
SELECT
  CAST(DATE_TRUNC('MONTH', startdate) AS DATE) AS month_start,
  dmdunit, loc,
  CAST(COUNT(*) AS BIGINT) AS row_count,
  COALESCE(SUM(qty_shipped), 0.0) AS qty_shipped,
  COALESCE(SUM(qty_ordered), 0.0) AS qty_ordered,
  COALESCE(SUM(qty), 0.0) AS qty
FROM demand_studio.silver.fact_sales_monthly
GROUP BY 1, 2, 3;
```
Supports incremental refresh (better than PG's full recompute).

**Option 4B — Delta Live Tables (DLT):**
Define each view as a DLT table in a pipeline. Triggered refresh after each data load. Best for the 4 accuracy views that join fact + dim tables.

**Option 4C — Pre-computed Delta Tables:**
Regular Delta tables overwritten by scheduled Databricks Jobs. Simplest but no automatic consistency.

Remove `REFRESH MATERIALIZED VIEW` calls from Python code (Databricks MVs refresh automatically, or trigger DLT pipeline via API).

### Phase 5: Vector Search + Chat Migration (1 week)

1. Create Delta table for embeddings:
   ```sql
   CREATE TABLE demand_studio.gold.chat_embeddings (
     id BIGINT GENERATED ALWAYS AS IDENTITY,
     domain_name STRING,
     content_type STRING,
     source_text STRING,
     embedding ARRAY<FLOAT>
   );
   ```

2. Create Databricks Vector Search endpoint + Delta Sync Index on `chat_embeddings`

3. Rewrite `chat.py` `_vector_search()`:
   ```python
   from databricks.vector_search.client import VectorSearchClient
   client = VectorSearchClient()
   results = client.get_index("demand_studio.gold.chat_embeddings_idx").similarity_search(
       query_vector=question_embedding,
       num_results=10,
   )
   ```

4. Remove `SET LOCAL statement_timeout` / `SET TRANSACTION READ ONLY` — use connector timeout instead

### Phase 6: Backtest Pipeline Rewrite (2-3 weeks)

**Rewrite `load_backtest_forecasts.py`:**
```python
# Current: COPY + staging + batched INSERT
# New: pandas → Delta write
import pandas as pd
from databricks import sql

def load_backtest_to_delta(csv_path, model_id, replace=False):
    df = pd.read_csv(csv_path)
    if replace:
        with sql.connect(**conn_params) as conn:
            conn.cursor().execute(
                "DELETE FROM demand_studio.silver.fact_external_forecast_monthly WHERE model_id = ?",
                [model_id]
            )
    # Write via Spark
    spark_df = spark.createDataFrame(df)
    spark_df.write.format("delta").mode("append").saveAsTable(
        "demand_studio.silver.fact_external_forecast_monthly"
    )
```

**Rewrite `competition.py` champion/ceiling flow:**
```python
# Current: CREATE TEMP TABLE + COPY FROM STDIN + INSERT ... SELECT
# New: DataFrame → staging Delta table → MERGE INTO
winners_df = pd.DataFrame(winners, columns=[...])
spark.createDataFrame(winners_df).write.format("delta").mode("overwrite").saveAsTable("_stg_champion_winners")
spark.sql("""
    MERGE INTO demand_studio.silver.fact_external_forecast_monthly AS t
    USING _stg_champion_winners AS s
    ON t.forecast_ck = s.forecast_ck AND t.model_id = s.model_id
    WHEN NOT MATCHED THEN INSERT *
""")
```

**Update `common/backtest_framework.py`:**
```python
# Current: psycopg.connect(**db) + pd.read_sql()
# New: databricks-sql-connector + pd.read_sql()
from databricks import sql
conn = sql.connect(server_hostname=..., http_path=..., access_token=...)
sales_df = pd.read_sql("SELECT ... FROM demand_studio.silver.fact_sales_monthly ...", conn)
```

### Phase 7: Cut Over + Retire Docker (1 week)

1. Set `DB_BACKEND=databricks` in production
2. Run full test suite against Databricks backend
3. Retire 7 Docker services (Postgres, MLflow, MinIO, Iceberg REST, Spark, Trino, app)
4. Update Makefile: replace `docker exec ... psql` targets with Databricks CLI / SQL
5. Keep Postgres running 2 weeks as rollback option

---

## 8. DB Abstraction Layer Design

### Interface Contract

```python
# api/backends/base.py
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator

class AbstractBackend(ABC):
    @contextmanager
    @abstractmethod
    def connection(self) -> Generator:
        """Yield a connection-like object with cursor() method."""
        ...

    @abstractmethod
    def execute(self, sql: str, params: list[Any]) -> list[tuple]:
        """Execute SQL and return all rows."""
        ...

    @abstractmethod
    def execute_one(self, sql: str, params: list[Any]) -> tuple | None:
        """Execute and fetch one row."""
        ...

    @abstractmethod
    def translate_sql(self, sql: str) -> str:
        """Dialect translation: PG-specific syntax → backend SQL."""
        ...
```

### PostgresBackend (wraps existing pool)

```python
# api/backends/postgres.py
class PostgresBackend(AbstractBackend):
    def __init__(self):
        self._pool = None  # lazy init, existing ConnectionPool logic

    @contextmanager
    def connection(self):
        pool = self._get_pool()
        with pool.connection() as conn:
            yield conn

    def translate_sql(self, sql: str) -> str:
        return sql  # No-op: SQL is already PG dialect
```

### DatabricksBackend

```python
# api/backends/databricks.py
import databricks.sql

class DatabricksBackend(AbstractBackend):
    def __init__(self):
        self._conn_params = {
            "server_hostname": os.getenv("DATABRICKS_HOST"),
            "http_path": os.getenv("DATABRICKS_HTTP_PATH"),
            "access_token": os.getenv("DATABRICKS_TOKEN"),
        }

    @contextmanager
    def connection(self):
        with databricks.sql.connect(**self._conn_params) as conn:
            yield conn

    def translate_sql(self, sql: str) -> str:
        return _translate_pg_to_databricks(sql)
```

### Backend Selection

```python
# api/core.py
_BACKEND_MODE = os.getenv("DB_BACKEND", "postgres")  # or "databricks"

def _get_backend() -> AbstractBackend:
    global _backend
    if _backend is None:
        if _BACKEND_MODE == "databricks":
            from api.backends.databricks import DatabricksBackend
            _backend = DatabricksBackend()
        else:
            from api.backends.postgres import PostgresBackend
            _backend = PostgresBackend()
    return _backend

def get_conn():
    return _get_backend().connection()
```

---

## 9. SQL Dialect Translation Reference

### Automatic (Regex-Based) Translations

| PostgreSQL | Databricks SQL | Regex Pattern |
|---|---|---|
| `%s` | `?` | `%s` → `?` |
| `::text` | (remove) | `::\s*text\b` → `` |
| `::bigint` | (remove or `CAST`) | `::\s*bigint\b` → `` |
| `::date` | (remove or `CAST`) | `::\s*date\b` → `` |
| `::double precision` | (remove or `CAST`) | `::\s*double precision\b` → `` |
| `::integer` | (remove or `CAST`) | `::\s*integer\b` → `` |
| `::numeric` | (remove or `CAST`) | `::\s*numeric\b` → `` |
| `ILIKE` | `ILIKE` (supported) or `UPPER() LIKE UPPER()` | Direct or rewrite |
| `random()` | `rand()` | `\brandom\(\)` → `rand()` |
| `TABLESAMPLE SYSTEM (n)` | `TABLESAMPLE (n PERCENT)` | Pattern replacement |
| `date_trunc('month', x)` | `DATE_TRUNC('MONTH', x)` | Case change (compatible) |

### Manual Rewrites Required (Cannot Be Auto-Translated)

| PostgreSQL Pattern | Databricks Replacement | Location |
|---|---|---|
| `SELECT reltuples FROM pg_class WHERE relname = %s` | `SELECT COUNT(*) FROM <table>` or `DESCRIBE DETAIL` | `core.py` `fetch_page()` |
| `SELECT to_regclass(%s)` | Try/except or `SHOW TABLES LIKE '<table>'` | `core.py` `build_agg_trend_source()` |
| `CREATE TEMP TABLE ... ON COMMIT DROP` | `CREATE OR REPLACE TEMPORARY VIEW` or staging Delta table | `competition.py` |
| `COPY ... FROM STDIN` | `spark.write.format("delta").mode("append")` or bulk INSERT | Load scripts |
| `ON CONFLICT (...) DO UPDATE` | `MERGE INTO ... WHEN MATCHED THEN UPDATE` | `load_backtest_forecasts.py` |
| `REFRESH MATERIALIZED VIEW` | Auto-refresh (Databricks MVs) or DLT trigger | Multiple files |
| `embedding <=> %s::vector` | Databricks Vector Search SDK | `chat.py` |
| `SET LOCAL statement_timeout = '5000'` | Connector `operation_timeout` param | `chat.py` |
| `DISTINCT ON (col)` | `ROW_NUMBER() OVER (PARTITION BY col ORDER BY ...) = 1` | `load_dataset_postgres.py` |
| `ALTER TABLE DROP/ADD CONSTRAINT` | Not needed (Delta manages internally) | Load scripts |
| `DO $$ BEGIN ... END $$` PL/pgSQL | Rewrite as Python migration | DDL files |

---

## 10. PostgreSQL-Specific Constructs by File

### SQL DDL Files

| File | PG-Specific Constructs |
|---|---|
| `001_create_dim_item.sql` | `BIGSERIAL`, `TIMESTAMPTZ` |
| `002_create_dim_location.sql` | `BIGSERIAL`, `TIMESTAMPTZ` |
| `003_create_dim_customer.sql` | `BIGSERIAL`, `TIMESTAMPTZ` |
| `004_create_dim_time.sql` | `BIGSERIAL`, `TIMESTAMPTZ` |
| `005_create_dim_dfu.sql` | `BIGSERIAL`, `TIMESTAMPTZ`, `DO $$...$$` PL/pgSQL, `pg_constraint`, `::regclass` |
| `006_create_fact_sales_monthly.sql` | `BIGSERIAL`, `::date` in CHECK, `DO $$...$$`, `pg_constraint` |
| `007_create_fact_external_forecast_monthly.sql` | `BIGSERIAL`, `::date`/`::int` in CHECK, `UNIQUE` constraint |
| `008_perf_indexes_and_agg.sql` | `CREATE EXTENSION pg_trgm`, 12 `GIN gin_trgm_ops` indexes, 2 `MATERIALIZED VIEW WITH NO DATA`, `::bigint`/`::double precision`/`::date` casts |
| `009_create_chat_embeddings.sql` | `CREATE EXTENSION vector`, `vector(1536)` type |
| `010_create_backtest_lag_archive.sql` | `BIGSERIAL`, `TIMESTAMPTZ`, `::int`/`::date` in CHECK, `UNIQUE` constraint |
| `011_create_accuracy_slice_views.sql` | 2 `MATERIALIZED VIEW WITH NO DATA`, pervasive `::` casts |
| `012_create_dfu_coverage_view.sql` | 2 `MATERIALIZED VIEW WITH NO DATA`, `::` casts |
| `013_add_composite_indexes.sql` | B-tree composite indexes (conceptually → Z-ORDER) |
| `015_add_seasonality_columns.sql` | `ADD COLUMN IF NOT EXISTS`, `COMMENT ON COLUMN` |
| `016_add_seasonality_to_accuracy_views.sql` | `DROP MATERIALIZED VIEW CASCADE`, recreate 4 MVs, pervasive `::` casts |

### API Router Files

| File | PG-Specific Constructs |
|---|---|
| `api/core.py` | `pg_class.reltuples`, `to_regclass()`, `::date`/`::text`, `ILIKE`, `%s` params, `build_where()` with `gin_trgm_ops` awareness |
| `api/routers/domains.py` | `TABLESAMPLE SYSTEM(1)`, `random()`, `to_regclass()`, `::text`, `ILIKE` |
| `api/routers/accuracy.py` | `::date`/`::text`/`::bigint`/`::double precision` casts, reads materialized views |
| `api/routers/analysis.py` | `::date` casts, reads materialized views |
| `api/routers/competition.py` | `CREATE TEMP TABLE ON COMMIT DROP`, `COPY FROM STDIN`, `REFRESH MATERIALIZED VIEW`, `::text`/`::date` |
| `api/routers/clusters.py` | `STDDEV()` (Spark-compatible), minor `::` casts |
| `api/routers/chat.py` | `embedding <=> %s::vector`, `SET LOCAL statement_timeout`, `SET TRANSACTION READ ONLY` |
| `api/routers/benchmark.py` | `docker exec demand-mvp-trino trino --execute` (hardcoded Docker CLI) |
| `api/routers/intel.py` | Minor `::` casts |

### Pipeline Scripts

| File | PG-Specific Constructs |
|---|---|
| `scripts/load_dataset_postgres.py` | `COPY FROM STDIN WITH (FORMAT CSV, HEADER TRUE)`, `DISTINCT ON`, temp table, `SET synchronous_commit = off` |
| `scripts/load_backtest_forecasts.py` | `COPY FROM STDIN`, staging table, `ON CONFLICT DO UPDATE`, index drop/recreate, `REFRESH MATERIALIZED VIEW`, `SET work_mem`/`maintenance_work_mem` |
| `scripts/update_cluster_assignments.py` | `COPY FROM STDIN` → temp table → `UPDATE ... FROM` |
| `scripts/update_seasonality_profiles.py` | Same COPY → temp table → UPDATE pattern |
| `scripts/run_champion_selection.py` | Complex CTEs, `ROW_NUMBER()`, COPY → temp table → INSERT |
| `scripts/clean_backtest_models.py` | `DELETE FROM`, `REFRESH MATERIALIZED VIEW` |
| `scripts/generate_clustering_features.py` | `pd.read_sql()` with psycopg connection |
| `scripts/detect_seasonality.py` | `pd.read_sql()` with psycopg connection |
| `scripts/generate_embeddings.py` | psycopg INSERT for embeddings |
| `common/backtest_framework.py` | `psycopg.connect(**db)` + `pd.read_sql()` |

---

## 11. Infrastructure Mapping

| Current (Docker Compose) | Databricks Equivalent | Notes |
|---|---|---|
| `postgres` (pgvector/pgvector:pg16) | Delta Lake on Unity Catalog (or Lakebase) | Primary OLTP → lakehouse |
| `mlflow` (self-hosted) | Databricks Managed MLflow | Built-in, zero config |
| `minio` (object storage) | Databricks managed storage (S3/ADLS/GCS) | Cloud-native |
| `iceberg-rest` (Iceberg catalog) | Unity Catalog (Iceberg-compatible) | Unified governance |
| `spark` (PySpark) | Databricks Runtime Spark | Native Spark |
| `trino` (query engine) | Databricks SQL Warehouse | Photon-accelerated |
| Docker Compose (7 services) | Fully managed (0 self-managed services) | Ops eliminated |

### Makefile Target Mapping

| Current Target | Databricks Equivalent |
|---|---|
| `make up` / `make down` | N/A (managed infrastructure) |
| `make db-apply-sql` | `databricks sql --file <ddl>.sql` or Python migration script |
| `make refresh-agg-*` | `REFRESH MATERIALIZED VIEW` (Databricks SQL) or DLT trigger |
| `make check-db` | `databricks sql "SELECT count(*) FROM ..."` |
| `make spark-*` | Databricks Job or notebook |
| `make bench-compare` | Rewrite benchmark router for Databricks SQL Warehouse |

---

## 12. Test Strategy

### Current Test Infrastructure

- **Backend:** 189 pytest tests, DB mocked via `patch("api.core._get_pool")` in `tests/api/conftest.py`
- **Frontend:** 108 Vitest tests, API layer mocked via `vi.mock("../api/queries")`

### What Changes for Databricks

1. **Mock fixture update** — `tests/api/conftest.py` mock chain must match the new backend interface:
   - Current: mocks `psycopg_pool.ConnectionPool` with `pool.connection()` → `conn.cursor()` → `cursor.fetchall()`
   - New: mock `_get_backend()` → `AbstractBackend` with same cursor protocol
   - The mock object chain is similar enough that most test assertions remain unchanged

2. **Parameterization check** — tests that verify specific SQL strings will need updating if `%s` → `?`

3. **Frontend tests** — zero changes (they mock the HTTP layer, not the DB)

4. **Integration test suite (new)** — add tests that run against a real Databricks SQL Warehouse to validate SQL dialect translation

### Dual-Run Testing
- `DB_BACKEND=postgres make test-api` — existing tests pass unchanged
- `DB_BACKEND=databricks make test-api` — same tests pass with new backend
- Compare API JSON responses between backends for parity validation

---

## 13. Risk Assessment

| Phase | Risk | Impact | Mitigation |
|---|---|---|---|
| **Phase 0** (Infrastructure) | Low | None (no code changes) | Standard Databricks provisioning |
| **Phase 1** (Abstraction layer) | Medium | Regression in existing Postgres path | Postgres tests continue to pass; Databricks is additive |
| **Phase 2** (Schema/data) | Medium | Data type mismatches, NULL handling differences | Validate row counts; re-run null normalization |
| **Phase 3** (Search) | Medium | Slower typeahead without pg_trgm | Z-ORDER + app cache; set latency acceptance threshold |
| **Phase 4** (Materialized views) | **High** | Stale KPIs if refresh timing is wrong | Use DLT event-triggered refresh; test staleness window |
| **Phase 5** (Vector search) | Low | Chat is non-critical, behind auth | Can A/B test independently |
| **Phase 6** (Backtest pipeline) | **High** | MERGE semantics differ from ON CONFLICT; throughput gap | Benchmark Delta write vs PG COPY with real data |
| **Phase 7** (Cut over) | Medium | Unforeseen edge cases in production traffic | Keep Postgres running 2 weeks as rollback |

### Critical Risk: Cold Start Latency

Databricks SQL Serverless Warehouses have a 2-6 second cold start. The first API call after idle will be slow. Mitigations:
- Use keep-warm configuration on the SQL Warehouse
- Add a health check endpoint that pings the warehouse periodically
- Use Databricks SQL Pro (always-on) if latency is critical

### Critical Risk: COPY → MERGE Performance

PostgreSQL `COPY FROM STDIN` can ingest ~500K rows/second. Delta `MERGE INTO` is typically 10-50x slower for equivalent data volumes. Mitigations:
- Use `INSERT INTO` (append-only) where dedup is not needed
- Use Spark DataFrame writes for bulk loads (bypasses SQL connector overhead)
- Batch large loads into partitioned writes

---

## 14. Effort Estimates

| Approach | Timeline | Code Changes | Risk |
|---|---|---|---|
| **Option A: Lakebase** | 1-2 weeks | Connection string + data migration only | Low |
| **Option B: Full Delta** (read-only first) | 5-7 weeks | Abstraction layer + DDL + data load | Medium |
| **Option B: Full Delta** (complete) | 12-16 weeks | All 7 phases | High |

### Phase-by-Phase Breakdown (Option B)

| Phase | Duration | Dependencies |
|---|---|---|
| Phase 0: Prerequisites | 1-2 weeks | None |
| Phase 1: Abstraction layer | 2-3 weeks | Phase 0 |
| Phase 2: Schema + data | 2-3 weeks | Phase 0 (parallel with Phase 1) |
| Phase 3: Search strategy | 1-2 weeks | Phase 1 |
| Phase 4: Materialized views | 2-3 weeks | Phase 2 |
| Phase 5: Vector search | 1 week | Phase 1 |
| Phase 6: Backtest pipeline | 2-3 weeks | Phase 1 + Phase 4 |
| Phase 7: Cut over | 1 week | All phases |

### Fastest Path to Value (Read-Only First)

1. Phase 0 + Phase 1 + Phase 2 — **~5 weeks**
2. Populate 6 aggregate views once as regular Delta tables
3. Switch read-only endpoints to `DB_BACKEND=databricks`
4. Keep all writes on Postgres

This gives **90% of migration value** (production reads on Databricks, Unity Catalog governance, managed infrastructure) while deferring the hard COPY/MERGE rewrites to a later sprint.

---

## 15. Verification Plan

1. **Unit tests:** Run `make test-all` (297+ tests) after each phase — all must pass against both backends
2. **Row count parity:** Compare row counts between Postgres and Delta tables for all 9 tables
3. **API response parity:** JSON-diff API responses for key endpoints (accuracy slice, DFU analysis, explorer page) between backends
4. **Search latency benchmark:** Compare Postgres GIN trigram vs Databricks LIKE for suggest and search endpoints
5. **Bulk load benchmark:** Compare PG COPY vs Delta write throughput for backtest loading (target: 500K+ rows)
6. **MV freshness test:** Verify materialized view staleness window is <30 seconds after data loads
7. **Cold start test:** Measure first-call latency after 30 min idle on Databricks SQL Warehouse
8. **End-to-end smoke test:** Full pipeline (normalize → load → backtest → champion select → UI verify) on Databricks

---

## Appendix A: New Files to Create (Option B)

```
mvp/demand/api/backends/__init__.py
mvp/demand/api/backends/base.py              # AbstractBackend interface
mvp/demand/api/backends/postgres.py          # PostgresBackend (extracted from core.py)
mvp/demand/api/backends/databricks.py        # DatabricksBackend + SQL translator
mvp/demand/sql/databricks/                   # Databricks DDL equivalents (16 files)
mvp/demand/scripts/load_dataset_databricks.py   # Delta Lake data loader
mvp/demand/scripts/load_backtest_databricks.py  # Delta Lake backtest loader
```

## Appendix B: Files to Modify (Option B)

```
mvp/demand/api/core.py                       # Backend selection via DB_BACKEND env var
mvp/demand/common/db.py                      # Add Databricks connection params
mvp/demand/common/backtest_framework.py      # load_backtest_data(): abstract DB connection
mvp/demand/common/mlflow_utils.py            # Change tracking URI to "databricks"
mvp/demand/api/routers/competition.py        # COPY + temp table → MERGE pattern
mvp/demand/api/routers/chat.py               # pgvector → Databricks Vector Search
mvp/demand/api/routers/benchmark.py          # Docker Trino → Databricks SQL Warehouse
mvp/demand/tests/api/conftest.py             # Update mock fixture for backend interface
mvp/demand/pyproject.toml                    # Add databricks-sql-connector dependency
mvp/demand/Makefile                          # Replace Docker targets with Databricks CLI
```

## Appendix C: Files That Stay Untouched (Option B)

```
mvp/demand/frontend/                         # Entire frontend (React/Vite/TypeScript)
mvp/demand/common/feature_engineering.py     # Pure pandas/numpy
mvp/demand/common/metrics.py                 # Pure pandas
mvp/demand/common/constants.py               # Pure Python
mvp/demand/common/domain_specs.py            # Pure Python dataclasses
mvp/demand/config/*.yaml                     # All config files
mvp/demand/scripts/run_backtest*.py          # All 6 ML backtest scripts (after framework update)
mvp/demand/scripts/normalize_dataset_csv.py  # Pure pandas CSV processing
mvp/demand/scripts/train_clustering_model.py # Pure sklearn
mvp/demand/scripts/label_clusters.py         # Pure Python
mvp/demand/scripts/detect_seasonality.py     # Pure pandas (after connection update)
mvp/demand/api/auth.py                       # No DB dependency
mvp/demand/api/routers/domains.py            # Uses get_conn() — auto-translated
mvp/demand/api/routers/accuracy.py           # Uses get_conn() — auto-translated
mvp/demand/api/routers/analysis.py           # Uses get_conn() — auto-translated
mvp/demand/api/routers/clusters.py           # Uses get_conn() — auto-translated
mvp/demand/api/routers/intel.py              # Uses get_conn() — auto-translated
```
