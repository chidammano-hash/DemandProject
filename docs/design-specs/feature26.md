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

| Query | SQL Shape | Purpose |
|-------|-----------|---------|
| `count` | `SELECT count(*) FROM {table} {where}` | Full-table or filtered row count |
| `page` | `SELECT <cols> FROM {table} {where} ORDER BY <sort> LIMIT N` | Paginated data fetch (mirrors Data Explorer) |
| `trend` | `SELECT date_trunc('month', <date>) AS bucket, SUM(<metric>) FROM {table} {where} GROUP BY 1 ORDER BY 1 DESC LIMIT N` | Monthly trend aggregation (only for domains with a date field) |

### Execution Model

1. **Postgres** — queries run in-process via `psycopg` connection, `time.perf_counter()` for timing
2. **Trino** — queries run via `docker exec` into the Trino container CLI with `--output-format CSV_HEADER`, timed end-to-end including container exec overhead
3. **Warmup** — configurable warmup rounds (default 1) run before timed rounds to prime caches
4. **Runs** — configurable timed runs (default 5) collected into a timing array for statistical analysis

## API Endpoint

### `GET /bench/compare`

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

## Makefile Target

```makefile
bench-compare:
	curl -sS "http://localhost:8000/bench/compare?domain=$(DOMAIN)&runs=$(RUNS)&..." | python3 -m json.tool
```

**Usage:**
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

- FastAPI server running (`make api`)
- Docker Compose services running (`make up`) — specifically the Trino container
- Data loaded into both Postgres (`make load-all`) and Iceberg (`make spark-all`)

## Implementation Details

### Helper Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `timed_postgres_query()` | `api/main.py` | Execute SQL via psycopg with `perf_counter` timing |
| `timed_trino_query()` | `api/main.py` | Execute SQL via `docker exec` into Trino CLI with timing |
| `summary_stats_ms()` | `api/main.py` | Compute min/max/avg/p50/p95 from a timing array |
| `percentile_ms()` | `api/main.py` | Linear interpolation percentile calculation |
| `item_field_for_spec()` | `api/main.py` | Resolve item column name for a domain |
| `location_field_for_spec()` | `api/main.py` | Resolve location column name for a domain |
| `default_date_field_for_spec()` | `api/main.py` | Resolve date column name for a domain |
| `default_trend_metric_for_spec()` | `api/main.py` | Choose the first non-excluded numeric column for trend aggregation |

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

## Related Features

- **Feature 1** — Infrastructure & platform setup (Docker Compose, Trino, Iceberg, MinIO)
- **Feature 2** — Data architecture & domain specs (DomainSpec used for column resolution)
- **Feature 4** — Fact tables (primary benchmark targets: sales and forecast)

## Files

| File | Role |
|------|------|
| `mvp/demand/api/main.py` | `GET /bench/compare` endpoint + timing helpers |
| `mvp/demand/Makefile` | `bench-compare` target |
