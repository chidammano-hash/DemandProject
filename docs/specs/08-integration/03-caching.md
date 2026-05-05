# API Caching Layer

> Reduces database load and speeds up dashboard rendering by caching frequently-accessed query results in memory with automatic expiration.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (infrastructure layer) |
| **Key Files** | `common/services/cache.py`, `common/services/query_tracker.py`, `config/platform/cache_config.yaml` |

---

## Problem

Dashboard KPIs, accuracy summaries, and inventory metrics are queried repeatedly by multiple users. Each request hits the database, even when the underlying data has not changed. During peak usage, this creates unnecessary database load and slower response times for read-heavy endpoints.

## Solution

An in-memory LRU cache (Least Recently Used) with TTL-based expiration (Time To Live) sits between API endpoints and the database. Cached responses are served instantly; stale entries are evicted automatically. When data is loaded or materialized views are refreshed, relevant cache namespaces are invalidated. A query tracker monitors endpoint latency and cache hit rates to identify optimization opportunities.

## How It Works

1. A GET request arrives at a cached endpoint
2. The cache layer generates a key from the endpoint path, query params, and filter hash
3. If a valid (non-expired) cache entry exists, it is returned immediately -- no database query
4. If no entry exists or it has expired, the database query runs normally
5. The result is stored in the cache with the namespace-specific TTL
6. When data changes (CSV load, MV refresh), the relevant namespace is cleared
7. The query tracker logs every request's latency and cache status for performance monitoring

## Data Model

### `fact_query_performance`

| Column | Type | Description |
|---|---|---|
| `perf_id` | `BIGSERIAL PK` | Auto-increment ID |
| `endpoint` | `TEXT` | API endpoint path |
| `method` | `TEXT` | HTTP method |
| `params_hash` | `TEXT` | Hash of query parameters |
| `latency_ms` | `INTEGER` | Response time in milliseconds |
| `cache_hit` | `BOOLEAN` | Whether the response was served from cache |
| `response_size_bytes` | `INTEGER` | Response payload size |
| `recorded_at` | `TIMESTAMPTZ` | When the query was tracked |

Indexes on `(endpoint, recorded_at DESC)` and `(cache_hit)`. Rows older than 30 days are auto-purged.

## Cache Namespaces

| Namespace | TTL | Invalidated By |
|---|---|---|
| `dashboard` | 60s | Data loads |
| `accuracy` | 300s | Data loads, forecast loads |
| `inventory` | 180s | Data loads |
| `inv_planning` | 300s | MV refreshes |
| `control_tower` | 120s | MV refreshes |

## Configuration

`config/platform/cache_config.yaml`:

```yaml
cache:
  backend: in_memory
  default_ttl_seconds: 300
  max_entries_per_namespace: 1000
  namespaces:
    dashboard: { ttl: 60 }
    accuracy: { ttl: 300 }
    inventory: { ttl: 180 }
    inv_planning: { ttl: 300 }
    control_tower: { ttl: 120 }
  invalidation:
    on_data_load: [dashboard, accuracy, inventory]
    on_forecast_load: [dashboard, accuracy]
    on_mv_refresh: [inv_planning, control_tower]
query_tracker:
  enabled: true
  slow_query_threshold_ms: 2000
  flush_interval_seconds: 60
  retention_days: 30
```

## Query Tracker

The query tracker is a middleware that records per-request performance:

- Logs endpoint, params, latency, and cache hit/miss status
- Warns when query latency exceeds the slow query threshold (default 2000ms)
- Flushes to `fact_query_performance` in batches (every 60s)
- Maintains an in-memory rolling window of the last 1000 queries for real-time stats
- Stats exposed via the `/health` endpoint

## Dependencies

- No external dependencies -- uses stdlib `collections.OrderedDict`, `threading`, `time`
- Cache decorator applied only to read-only GET endpoints (never mutations)

## See Also

- [API Governance](./09-api-governance.md) -- rate limiting complements caching for resource protection
- [Integration Architecture](./01-integration-architecture.md) -- overall integration overview
