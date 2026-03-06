# Feature 26: Postgres vs Trino/Iceberg Benchmarking

## Objective

Provide a backend benchmarking utility that runs identical SQL queries against PostgreSQL and Trino (over Iceberg) and returns per-query latency statistics with a winner determination. This enables data engineers to quantify the performance tradeoff between the OLTP path (Postgres) and the lakehouse path (Trino/Iceberg) for any domain and filter combination.

## Motivation

- The platform maintains two parallel data paths: Postgres (direct load) and Iceberg via MinIO (Spark-ingested, Trino-queried).
- Knowing which backend is faster for a given query shape (count, paginated fetch, trend aggregation) guides architectural decisions about which path to use for specific workloads.
- A structured benchmark with warmup rounds, multiple runs, and percentile statistics provides reproducible, statistically meaningful comparisons — superior to ad-hoc manual timing.

## Scope

### In Scope
- API endpoint (`GET /bench/compare`) that benchmarks Postgres vs Trino for any domain
- Three benchmark query types: count, paginated page, and monthly trend aggregation
- Configurable runs, warmup rounds, row limits, and time points
- Optional item, location, and date range filters
- Per-query timing statistics: min, max, avg, p50, p95
- Winner determination with speedup factor
- Makefile integration (`make bench-compare`)

### Out of Scope
- Frontend benchmarking panel (API-only; future enhancement)
- Persistent storage of benchmark results (returned inline only)
- Automated regression tracking across code changes

## Architecture

### Benchmark Flow

```
Operator runs: make bench-compare DOMAIN=sales ITEM=100320
                              │
                              ▼
                curl → GET /bench/compare
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
           Build SQL     Build SQL     Build SQL
           (count)       (page)        (trend)
                │             │             │
         ┌──────┴──────┐ ┌───┴───┐   ┌─────┴─────┐
         ▼             ▼ ▼       ▼   ▼           ▼
     Postgres      Trino  PG   Trino  PG      Trino
     N runs        N runs                      N runs
         │             │ │       │   │           │
         └──────┬──────┘ └───┬───┘   └─────┬─────┘
                ▼            ▼             ▼
         stats + winner  stats + winner  stats + winner
                │            │             │
                └────────────┼─────────────┘
                             ▼
                       JSON response
```

### Query Types
#### Example — Benchmark Flow in Practice

```bash
# Prerequisite: start all services
make up    # Docker Compose (Postgres, Trino, MinIO)
make api   # FastAPI on :8000

# Run benchmark: sales domain, item 100320, location 1401-BULK, 3 timed runs
curl -s "http://localhost:8000/bench/compare?domain=sales&runs=3&warmup=1&item=100320&location=1401-BULK" | python3 -m json.tool
```

### Query Types
#### Example — Benchmark Flow in Practice

```bash
# Prerequisite: start all services
make up    # Docker Compose (Postgres, Trino, MinIO)
make api   # FastAPI on :8000

# Run benchmark: sales domain, item 100320, location 1401-BULK, 3 timed runs
curl -s "http://localhost:8000/bench/compare?domain=sales&runs=3&warmup=1&item=100320&location=1401-BULK" | python3 -m json.tool
```

### Query Types

| Query | SQL Shape | Purpose |
|-------|-----------|---------|
| `count` | `SELECT count(*) FROM {table} {where}` | Full-table or filtered row count |
| `page` | `SELECT <cols> FROM {table} {where} ORDER BY <sort> LIMIT N` | Paginated data fetch (mirrors Data Explorer) |
| `trend` | `SELECT date_trunc('month', <date>) AS bucket, SUM(<metric>) FROM {table} {where} GROUP BY 1 ORDER BY 1 DESC LIMIT N` | Monthly trend aggregation (only for domains with a date field) |

### Execution Model
#### Example — Generated SQL (Postgres vs Trino)

```sql
-- count query (Postgres)
SELECT count(*) AS cnt
FROM "fact_sales_monthly"
WHERE CAST("dmdunit" AS VARCHAR) = '100320'
  AND CAST("loc" AS VARCHAR) = '1401-BULK';

-- count query (Trino/Iceberg)
SELECT count(*) AS cnt
FROM "iceberg"."silver"."fact_sales_monthly"
WHERE CAST("dmdunit" AS VARCHAR) = '100320'
  AND CAST("loc" AS VARCHAR) = '1401-BULK';

-- page query (Postgres) — first 10 columns, 200 rows
SELECT "dmdunit","dmdgroup","loc","startdate","type","qty_shipped","qty_ordered","qty","load_ts","modified_ts"
FROM "fact_sales_monthly"
WHERE CAST("dmdunit" AS VARCHAR) = '100320'
  AND CAST("loc" AS VARCHAR) = '1401-BULK'
ORDER BY "startdate" DESC
LIMIT 200;

-- trend query (Postgres)
SELECT date_trunc('month', "startdate") AS bucket, SUM("qty") AS metric
FROM "fact_sales_monthly"
WHERE CAST("dmdunit" AS VARCHAR) = '100320'
  AND CAST("loc" AS VARCHAR) = '1401-BULK'
GROUP BY 1 ORDER BY 1 DESC LIMIT 24;
```

### Execution Model
#### Example — Generated SQL (Postgres vs Trino)

```sql
-- count query (Postgres)
SELECT count(*) AS cnt
FROM "fact_sales_monthly"
WHERE CAST("dmdunit" AS VARCHAR) = '100320'
  AND CAST("loc" AS VARCHAR) = '1401-BULK';

-- count query (Trino/Iceberg)
SELECT count(*) AS cnt
FROM "iceberg"."silver"."fact_sales_monthly"
WHERE CAST("dmdunit" AS VARCHAR) = '100320'
  AND CAST("loc" AS VARCHAR) = '1401-BULK';

-- page query (Postgres) — first 10 columns, 200 rows
SELECT "dmdunit","dmdgroup","loc","startdate","type","qty_shipped","qty_ordered","qty","load_ts","modified_ts"
FROM "fact_sales_monthly"
WHERE CAST("dmdunit" AS VARCHAR) = '100320'
  AND CAST("loc" AS VARCHAR) = '1401-BULK'
ORDER BY "startdate" DESC
LIMIT 200;

-- trend query (Postgres)
SELECT date_trunc('month', "startdate") AS bucket, SUM("qty") AS metric
FROM "fact_sales_monthly"
WHERE CAST("dmdunit" AS VARCHAR) = '100320'
  AND CAST("loc" AS VARCHAR) = '1401-BULK'
GROUP BY 1 ORDER BY 1 DESC LIMIT 24;
```

### Execution Model

1. **Postgres** — queries run in-process via `psycopg` connection, `time.perf_counter()` for timing
2. **Trino** — queries run via `docker exec` into the Trino container CLI with `--output-format CSV_HEADER`, timed end-to-end including container exec overhead
3. **Warmup** — configurable warmup rounds (default 1) run before timed rounds to prime caches
4. **Runs** — configurable timed runs (default 5) collected into a timing array for statistical analysis

## API Endpoint

### `GET /bench/compare`

**Query Parameters:**
#### Example — Full curl Request with All Parameters

```bash
curl -s "http://localhost:8000/bench/compare?domain=sales&runs=5&warmup=1&limit=200&points=24&item=100320&location=1401-BULK" | python3 -m json.tool
```

**Query Parameters:**
#### Example — Full curl Request with All Parameters

```bash
curl -s "http://localhost:8000/bench/compare?domain=sales&runs=5&warmup=1&limit=200&points=24&item=100320&location=1401-BULK" | python3 -m json.tool
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domain` | string | `sales` | Domain to benchmark (any of the 7 domains) |
| `runs` | int | `5` | Number of timed runs per query (1–30) |
| `warmup` | int | `1` | Number of warmup runs before timing (0–5) |
| `limit` | int | `200` | Row limit for the `page` query (10–1000) |
| `points` | int | `24` | Month limit for the `trend` query (3–120) |
| `item` | string | `""` | Optional item filter (e.g., `100320`) |
| `location` | string | `""` | Optional location filter (e.g., `1401-BULK`) |
| `start_date` | string | `""` | Optional date range start (ISO `YYYY-MM-DD`) |
| `end_date` | string | `""` | Optional date range end (ISO `YYYY-MM-DD`) |
| `trino_container` | string | `demand-mvp-trino` | Docker container name for Trino |
| `trino_catalog` | string | `iceberg` | Trino catalog name |
| `trino_schema` | string | `silver` | Trino schema name |

**Response:**
```json
{
  "domain": "sales",
  "table": "fact_sales_monthly",
  "iceberg_table": "iceberg.silver.fact_sales_monthly",
  "filters": {
    "item_field": "dmdunit",
    "item": "100320",
    "location_field": "loc",
    "location": "1401-BULK",
    "date_field": "startdate",
    "start_date": null,
    "end_date": null
  },
  "config": {
    "runs": 5,
    "warmup": 1,
    "limit": 200,
    "points": 24,
    "trend_metric": "qty",
    "trino_container": "demand-mvp-trino",
    "trino_catalog": "iceberg",
    "trino_schema": "silver"
  },
  "results": [
    {
      "query": "count",
      "postgres_sql": "SELECT count(*) AS cnt FROM \"fact_sales_monthly\" WHERE ...",
      "trino_sql": "SELECT count(*) AS cnt FROM \"iceberg\".\"silver\".\"fact_sales_monthly\" WHERE ...",
      "postgres": {
        "runs_ms": [12.345, 11.678, 12.012, 11.890, 12.234],
        "stats": {
          "runs": 5,
          "avg_ms": 12.032,
          "min_ms": 11.678,
          "max_ms": 12.345,
          "p50_ms": 12.012,
          "p95_ms": 12.345
        }
      },
      "trino": {
        "runs_ms": [450.123, 380.456, 360.789, 355.012, 365.234],
        "stats": {
          "runs": 5,
          "avg_ms": 382.323,
          "min_ms": 355.012,
          "max_ms": 450.123,
          "p50_ms": 365.234,
          "p95_ms": 450.123
        }
      },
      "faster_backend": "postgres",
      "speedup_factor": 31.773
    }
  ]
}
```

**Error Codes:**
- `404` — Unknown domain
#### Example — Full JSON Response with Latency Statistics

```json
{
  "domain": "sales",
  "table": "fact_sales_monthly",
  "iceberg_table": "iceberg.silver.fact_sales_monthly",
  "filters": {
    "item_field": "dmdunit",
    "item": "100320",
    "location_field": "loc",
    "location": "1401-BULK",
    "date_field": "startdate",
    "start_date": null,
    "end_date": null
  },
  "config": {
    "runs": 5,
    "warmup": 1,
    "limit": 200,
    "points": 24,
    "trend_metric": "qty",
    "trino_container": "demand-mvp-trino",
    "trino_catalog": "iceberg",
    "trino_schema": "silver"
  },
  "results": [
    {
      "query": "count",
      "postgres_sql": "SELECT count(*) AS cnt FROM "fact_sales_monthly" WHERE ...",
      "trino_sql": "SELECT count(*) AS cnt FROM "iceberg"."silver"."fact_sales_monthly" WHERE ...",
      "postgres": {
        "runs_ms": [12.3, 11.7, 12.0, 11.9, 12.2],
        "stats": {
          "runs": 5, "avg_ms": 12.02, "min_ms": 11.7,
          "max_ms": 12.3, "p50_ms": 12.0, "p95_ms": 12.3
        }
      },
      "trino": {
        "runs_ms": [450.1, 380.5, 360.8, 355.0, 365.2],
        "stats": {
          "runs": 5, "avg_ms": 382.3, "min_ms": 355.0,
          "max_ms": 450.1, "p50_ms": 365.2, "p95_ms": 450.1
        }
      },
      "faster_backend": "postgres",
      "speedup_factor": 31.8
    },
    {
      "query": "page",
      "faster_backend": "postgres",
      "speedup_factor": 8.4
    },
    {
      "query": "trend",
      "faster_backend": "postgres",
      "speedup_factor": 15.2
    }
  ]
}
```

**Error Codes:**
- `404` — Unknown domain
#### Example — Full JSON Response with Latency Statistics

```json
{
  "domain": "sales",
  "table": "fact_sales_monthly",
  "iceberg_table": "iceberg.silver.fact_sales_monthly",
  "filters": {
    "item_field": "dmdunit",
    "item": "100320",
    "location_field": "loc",
    "location": "1401-BULK",
    "date_field": "startdate",
    "start_date": null,
    "end_date": null
  },
  "config": {
    "runs": 5,
    "warmup": 1,
    "limit": 200,
    "points": 24,
    "trend_metric": "qty",
    "trino_container": "demand-mvp-trino",
    "trino_catalog": "iceberg",
    "trino_schema": "silver"
  },
  "results": [
    {
      "query": "count",
      "postgres_sql": "SELECT count(*) AS cnt FROM "fact_sales_monthly" WHERE ...",
      "trino_sql": "SELECT count(*) AS cnt FROM "iceberg"."silver"."fact_sales_monthly" WHERE ...",
      "postgres": {
        "runs_ms": [12.3, 11.7, 12.0, 11.9, 12.2],
        "stats": {
          "runs": 5, "avg_ms": 12.02, "min_ms": 11.7,
          "max_ms": 12.3, "p50_ms": 12.0, "p95_ms": 12.3
        }
      },
      "trino": {
        "runs_ms": [450.1, 380.5, 360.8, 355.0, 365.2],
        "stats": {
          "runs": 5, "avg_ms": 382.3, "min_ms": 355.0,
          "max_ms": 450.1, "p50_ms": 365.2, "p95_ms": 450.1
        }
      },
      "faster_backend": "postgres",
      "speedup_factor": 31.8
    },
    {
      "query": "page",
      "faster_backend": "postgres",
      "speedup_factor": 8.4
    },
    {
      "query": "trend",
      "faster_backend": "postgres",
      "speedup_factor": 15.2
    }
  ]
}
```

**Error Codes:**
- `404` — Unknown domain
- `422` — Invalid filter (domain lacks item/location/date field, or invalid date format)
- `503` — Trino benchmark failed (container unreachable or query error)

## Statistics

Each backend returns per-query timing stats computed from the timed runs array:

| Stat | Description |
|------|-------------|
| `runs` | Number of timed runs |
| `avg_ms` | Mean latency in milliseconds |
| `min_ms` | Best-case latency |
| `max_ms` | Worst-case latency |
| `p50_ms` | Median (50th percentile) latency |
| `p95_ms` | 95th percentile latency |

Winner determination: the backend with the lower `avg_ms` is marked as `faster_backend`, with `speedup_factor` = slower / faster.


#### Example — Statistics Computation

```python
import statistics

runs_ms = [12.3, 11.7, 12.0, 11.9, 12.2]
avg_ms = sum(runs_ms) / len(runs_ms)        # 12.02
min_ms = min(runs_ms)                        # 11.7
max_ms = max(runs_ms)                        # 12.3
sorted_runs = sorted(runs_ms)
p50_ms = sorted_runs[len(sorted_runs) // 2]  # 12.0
p95_idx = int(0.95 * len(sorted_runs))
p95_ms = sorted_runs[min(p95_idx, len(sorted_runs) - 1)]  # 12.3

trino_avg_ms = 382.3
speedup_factor = trino_avg_ms / avg_ms       # 31.8x — Postgres is 31.8x faster
```


#### Example — Statistics Computation

```python
import statistics

runs_ms = [12.3, 11.7, 12.0, 11.9, 12.2]
avg_ms = sum(runs_ms) / len(runs_ms)        # 12.02
min_ms = min(runs_ms)                        # 11.7
max_ms = max(runs_ms)                        # 12.3
sorted_runs = sorted(runs_ms)
p50_ms = sorted_runs[len(sorted_runs) // 2]  # 12.0
p95_idx = int(0.95 * len(sorted_runs))
p95_ms = sorted_runs[min(p95_idx, len(sorted_runs) - 1)]  # 12.3

trino_avg_ms = 382.3
speedup_factor = trino_avg_ms / avg_ms       # 31.8x — Postgres is 31.8x faster
```

## Makefile Target

```makefile
bench-compare:
	curl -sS "http://localhost:8000/bench/compare?domain=$(DOMAIN)&runs=$(RUNS)&..." | python3 -m json.tool
```

**Usage:**
```bash
# Default: sales domain, 5 runs, 1 warmup
make bench-compare
#### Example — Make Targets

```bash
# Default: sales domain, 5 runs, 1 warmup
make bench-compare
#### Example — Make Targets

```bash
# Default: sales domain, 5 runs, 1 warmup
make bench-compare

# Custom domain + filters
make bench-compare DOMAIN=sales RUNS=7 ITEM=100320 LOCATION=1401-BULK

# Forecast domain with date range
make bench-compare DOMAIN=forecast RUNS=10 START_DATE=2023-01-01 END_DATE=2024-12-01
```

**Make Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DOMAIN` | `sales` | Domain to benchmark |
| `RUNS` | `5` | Timed runs per query |
| `WARMUP` | `1` | Warmup rounds |
| `LIMIT` | `200` | Row limit for page query |
| `POINTS` | `24` | Month limit for trend query |
| `ITEM` | _(empty)_ | Optional item filter |
| `LOCATION` | _(empty)_ | Optional location filter |
| `START_DATE` | _(empty)_ | Optional date range start |
| `END_DATE` | _(empty)_ | Optional date range end |
| `TRINO_CATALOG` | `iceberg` | Trino catalog |
| `TRINO_SCHEMA` | `silver` | Trino schema |

## Prerequisites
#### Example — Benchmark Multiple Domains

```bash
# Sales domain with date range filter
make bench-compare DOMAIN=sales RUNS=7 ITEM=100320 LOCATION=1401-BULK

# Forecast domain — compare monthly accuracy trend query performance
make bench-compare DOMAIN=forecast RUNS=5 START_DATE=2023-01-01 END_DATE=2024-12-01

# DFU domain — dimension table (smaller, both fast)
make bench-compare DOMAIN=dfu RUNS=3

# Sample output for sales domain (3 runs):
# count  | postgres avg:   11.7ms | trino avg:  382.3ms | winner: postgres (32.7x)
# page   | postgres avg:   34.2ms | trino avg:  288.4ms | winner: postgres  (8.4x)
# trend  | postgres avg:   18.9ms | trino avg:  287.5ms | winner: postgres (15.2x)
```

## Prerequisites
#### Example — Benchmark Multiple Domains

```bash
# Sales domain with date range filter
make bench-compare DOMAIN=sales RUNS=7 ITEM=100320 LOCATION=1401-BULK

# Forecast domain — compare monthly accuracy trend query performance
make bench-compare DOMAIN=forecast RUNS=5 START_DATE=2023-01-01 END_DATE=2024-12-01

# DFU domain — dimension table (smaller, both fast)
make bench-compare DOMAIN=dfu RUNS=3

# Sample output for sales domain (3 runs):
# count  | postgres avg:   11.7ms | trino avg:  382.3ms | winner: postgres (32.7x)
# page   | postgres avg:   34.2ms | trino avg:  288.4ms | winner: postgres  (8.4x)
# trend  | postgres avg:   18.9ms | trino avg:  287.5ms | winner: postgres (15.2x)
```

## Prerequisites

- FastAPI server running (`make api`)
- Docker Compose services running (`make up`) — specifically the Trino container
- Data loaded into both Postgres (`make load-all`) and Iceberg (`make spark-all`)

## Implementation Details

### Helper Functions
#### Example — Key Helper Functions

```python
# api/core.py — timing helpers used by the benchmark endpoint

import time, subprocess, statistics
from typing import list

def timed_postgres_query(conn, sql: str, params=()) -> float:
    """Run SQL via psycopg and return wall-clock ms."""
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000.0

def timed_trino_query(container: str, sql: str) -> float:
    """Run SQL via docker exec into Trino CLI and return wall-clock ms."""
    t0 = time.perf_counter()
    result = subprocess.run(
        ["docker", "exec", container, "trino", "--execute", sql,
         "--output-format", "CSV_HEADER", "--server", "localhost:8080"],
        capture_output=True, text=True, timeout=120
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if result.returncode != 0:
        raise RuntimeError(f"Trino error: {result.stderr[:200]}")
    return elapsed_ms

def summary_stats_ms(runs_ms: list[float]) -> dict:
    """Compute min/max/avg/p50/p95 from a timing list."""
    sorted_runs = sorted(runs_ms)
    n = len(sorted_runs)
    return {
        "runs": n,
        "avg_ms": round(sum(runs_ms) / n, 3),
        "min_ms": round(sorted_runs[0], 3),
        "max_ms": round(sorted_runs[-1], 3),
        "p50_ms": round(sorted_runs[n // 2], 3),
        "p95_ms": round(sorted_runs[min(int(0.95 * n), n - 1)], 3),
    }
```

### Helper Functions
#### Example — Key Helper Functions

```python
# api/core.py — timing helpers used by the benchmark endpoint

import time, subprocess, statistics
from typing import list

def timed_postgres_query(conn, sql: str, params=()) -> float:
    """Run SQL via psycopg and return wall-clock ms."""
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000.0

def timed_trino_query(container: str, sql: str) -> float:
    """Run SQL via docker exec into Trino CLI and return wall-clock ms."""
    t0 = time.perf_counter()
    result = subprocess.run(
        ["docker", "exec", container, "trino", "--execute", sql,
         "--output-format", "CSV_HEADER", "--server", "localhost:8080"],
        capture_output=True, text=True, timeout=120
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if result.returncode != 0:
        raise RuntimeError(f"Trino error: {result.stderr[:200]}")
    return elapsed_ms

def summary_stats_ms(runs_ms: list[float]) -> dict:
    """Compute min/max/avg/p50/p95 from a timing list."""
    sorted_runs = sorted(runs_ms)
    n = len(sorted_runs)
    return {
        "runs": n,
        "avg_ms": round(sum(runs_ms) / n, 3),
        "min_ms": round(sorted_runs[0], 3),
        "max_ms": round(sorted_runs[-1], 3),
        "p50_ms": round(sorted_runs[n // 2], 3),
        "p95_ms": round(sorted_runs[min(int(0.95 * n), n - 1)], 3),
    }
```

### Helper Functions

All benchmark helper functions live in `api/core.py` (shared utilities module), not in the endpoint file itself.

| Function | Location | Purpose |
|----------|----------|---------|
| `timed_postgres_query()` | `api/core.py` | Execute SQL via psycopg with `perf_counter` timing |
| `timed_trino_query()` | `api/core.py` | Execute SQL via `docker exec` into Trino CLI with timing |
| `summary_stats_ms()` | `api/core.py` | Compute min/max/avg/p50/p95 from a timing array |
| `percentile_ms()` | `api/core.py` | Linear interpolation percentile calculation |
| `item_field_for_spec()` | `api/core.py` | Resolve item column name for a domain |
| `location_field_for_spec()` | `api/core.py` | Resolve location column name for a domain |
| `default_date_field_for_spec()` | `api/core.py` | Resolve date column name for a domain |
| `default_trend_metric_for_spec()` | `api/core.py` | Choose the first non-excluded numeric column for trend aggregation |
| `get_spec_or_404()` | `api/core.py` | Look up DomainSpec by name or raise 404 |
| `qident()` | `api/core.py` | Quote a single SQL identifier |
| `dotted_qident()` | `api/core.py` | Build fully-qualified dotted identifier (catalog.schema.table) |
| `to_api_col()` | `api/core.py` | Map internal column name to API column name (e.g., `class` -> `class_`) |
| `quote_literal()` | `api/core.py` | Safely quote a SQL string literal for cross-engine compatibility |
| `parse_optional_iso_date()` | `api/core.py` | Parse and validate optional ISO date string |

### SQL Generation

- Postgres queries use `"table_name"` (quoted identifier)
- Trino queries use `"catalog"."schema"."table_name"` (fully-qualified dotted identifier)
- WHERE clauses are shared between both backends, using `CAST(col AS VARCHAR)` for item/location filters and `CAST(col AS DATE)` for date range filters for cross-engine compatibility
- Column selection for the `page` query is capped at the first 10 columns to keep output manageable

## Dependencies

- `psycopg` — PostgreSQL driver (existing)
- `subprocess` — Trino CLI invocation via Docker exec (stdlib)
- `time.perf_counter` — high-resolution timing (stdlib)
- No new dependencies required

## Testing

Backend API tests are in `mvp/demand/tests/api/test_benchmark.py`:

| Test | Description |
|------|-------------|
| `test_benchmark_invalid_domain` | `GET /bench/compare?domain=nonexistent` returns 404 |
| `test_benchmark_date_range_validation` | Inverted date range (`start_date > end_date`) returns 422 |
| `test_benchmark_success` | Mocked timing functions return 200 with valid response structure |
| `test_benchmark_trino_failure` | Trino `RuntimeError` returns 503 |

Tests mock `timed_postgres_query` and `timed_trino_query` from `api.routers.benchmark` to avoid requiring live database and Docker services.


#### Example — Backend Test Pattern

```python
# mvp/demand/tests/api/test_benchmark.py
import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.mark.asyncio
async def test_benchmark_success():
    """Mocked benchmark returns 200 with valid structure."""
    mock_timings = [10.0, 11.0, 9.5, 10.5, 10.2]

    with patch("api.routers.benchmark.timed_postgres_query", return_value=10.2),          patch("api.routers.benchmark.timed_trino_query", return_value=350.0):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/bench/compare?domain=sales&runs=3&item=100320&location=1401-BULK")

    assert resp.status_code == 200
    data = resp.json()
    assert data["domain"] == "sales"
    assert len(data["results"]) == 3  # count, page, trend
    assert data["results"][0]["faster_backend"] == "postgres"

@pytest.mark.asyncio
async def test_benchmark_invalid_domain():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/bench/compare?domain=nonexistent")
    assert resp.status_code == 404

@pytest.mark.asyncio
async def test_benchmark_trino_failure():
    with patch("api.routers.benchmark.timed_postgres_query", return_value=10.0),          patch("api.routers.benchmark.timed_trino_query", side_effect=RuntimeError("Trino unreachable")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/bench/compare?domain=sales&runs=1")
    assert resp.status_code == 503
```


#### Example — Backend Test Pattern

```python
# mvp/demand/tests/api/test_benchmark.py
import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.mark.asyncio
async def test_benchmark_success():
    """Mocked benchmark returns 200 with valid structure."""
    mock_timings = [10.0, 11.0, 9.5, 10.5, 10.2]

    with patch("api.routers.benchmark.timed_postgres_query", return_value=10.2),          patch("api.routers.benchmark.timed_trino_query", return_value=350.0):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/bench/compare?domain=sales&runs=3&item=100320&location=1401-BULK")

    assert resp.status_code == 200
    data = resp.json()
    assert data["domain"] == "sales"
    assert len(data["results"]) == 3  # count, page, trend
    assert data["results"][0]["faster_backend"] == "postgres"

@pytest.mark.asyncio
async def test_benchmark_invalid_domain():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/bench/compare?domain=nonexistent")
    assert resp.status_code == 404

@pytest.mark.asyncio
async def test_benchmark_trino_failure():
    with patch("api.routers.benchmark.timed_postgres_query", return_value=10.0),          patch("api.routers.benchmark.timed_trino_query", side_effect=RuntimeError("Trino unreachable")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/bench/compare?domain=sales&runs=1")
    assert resp.status_code == 503
```

## Related Features

- **Feature 1** — Infrastructure & platform setup (Docker Compose, Trino, Iceberg, MinIO)
- **Feature 2** — Data architecture & domain specs (DomainSpec used for column resolution)
- **Feature 4** — Fact tables (primary benchmark targets: sales and forecast)

## Files

| File | Role |
|------|------|
| `mvp/demand/api/routers/benchmark.py` | `GET /bench/compare` endpoint (modular router) |
| `mvp/demand/api/core.py` | Shared utilities: timing helpers, SQL helpers, domain spec resolution |
| `mvp/demand/api/main.py` | Also contains an inline `@app.get("/bench/compare")` route (legacy; takes precedence over router) |
| `mvp/demand/Makefile` | `bench-compare` target |
| `mvp/demand/tests/api/test_benchmark.py` | API tests: invalid domain (404), date range validation (422), success (200), Trino failure (503) |

### Dual Route Registration Note

The benchmark endpoint exists in two places:
1. **Inline route** in `api/main.py` (line ~692): `@app.get("/bench/compare")` -- registered first, takes precedence
2. **Router module** in `api/routers/benchmark.py` -- extracted as part of modular router architecture

Because FastAPI resolves routes in registration order, the inline `main.py` route handles requests. The router module (`benchmark.py`) exists as the canonical modular implementation but is not currently mounted via `app.include_router()`.


---

## Latency Stats Interpretation Guide

#### Example — Reading Benchmark Results

```bash
# Run a benchmark and interpret the output:
make bench-compare DOMAIN=sales RUNS=5 ITEM=100320 LOCATION=1401-BULK

# Interpreting the results:
#
# count query:
#   postgres avg:   11.7ms   p95:   12.3ms   — very low latency (cached, indexed count)
#   trino    avg:  382.3ms   p95:  450.1ms   — much higher (JVM cold-start + parsing overhead)
#   winner: postgres  speedup: 32.7x
#
# page query (paginated fetch, 200 rows):
#   postgres avg:   34.2ms   p95:   38.1ms
#   trino    avg:  288.4ms   p95:  340.2ms
#   winner: postgres  speedup:  8.4x
#   interpretation: Trino's overhead shrinks as query complexity grows —
#                   speedup narrows from 32x (count) to 8x (page).
#
# trend query (GROUP BY month, SUM(qty)):
#   postgres avg:   18.9ms   p95:   21.0ms
#   trino    avg:  287.5ms   p95:  320.5ms
#   winner: postgres  speedup: 15.2x
#
# Decision guide:
#   - If all 3 queries show postgres winning:
#       → Use Postgres for this workload; Iceberg adds latency without gain.
#   - If trend query shows trino winning (large aggregations):
#       → Route monthly rollup queries to Trino for analytics workloads.
#   - High p95 vs avg gap (>50%) in either backend:
#       → Cache is cold or resource contention — re-run with more warmup rounds.
#   - speedup < 2x:
#       → Both backends are comparable; choose based on ecosystem, not latency alone.
```

#### Example — When Trino May Win

```bash
# For very large unfiltered aggregations (full-table scans), Trino's columnar
# Parquet storage on Iceberg can outperform Postgres B-tree indexes:
make bench-compare DOMAIN=sales RUNS=5
# (no item/location filter → full-table scan, 8M+ rows)
#
# Expected output (at scale):
# count  | postgres avg:  180.2ms | trino avg:   84.5ms | winner: trino   (2.1x)
# trend  | postgres avg:  420.7ms | trino avg:  195.3ms | winner: trino   (2.2x)
# page   | postgres avg:   22.1ms | trino avg:  310.8ms | winner: postgres (14.1x)
#
# Insight: Trino's columnar scan excels at full-table aggregations.
#          Postgres wins page queries (B-tree index + LIMIT pushdown).
```
