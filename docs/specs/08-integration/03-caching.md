# API Caching Layer

> Reduces database load and speeds up dashboard rendering by caching frequently-accessed query results in memory with automatic expiration.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (infrastructure layer) |
| **Key Files** | `common/services/cache.py`, `config/platform/cache_config.yaml` |

---

## Problem

Dashboard KPIs, accuracy summaries, and inventory metrics are queried repeatedly by multiple users. Each request hits the database, even when the underlying data has not changed. During peak usage, this creates unnecessary database load and slower response times for read-heavy endpoints.

## Solution

A pluggable cache layer with TTL-based expiration sits between API endpoints and the database. The default backend is Redis (shared across all gunicorn workers, with `SETNX`-based single-flight stampede protection); when Redis is unavailable or the package is missing the layer transparently falls back to an in-memory LRU per-process cache. Cached responses are served instantly; stale entries are evicted automatically. When data is loaded or materialized views are refreshed, relevant cache namespaces are invalidated. A query tracker monitors endpoint latency and cache hit rates to identify optimization opportunities.

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
| `dashboard` | 60-600s by endpoint | Data loads |
| `accuracy` | 300s | Data loads, forecast loads |
| `inventory` | 180s | Data loads |
| `inv_planning` | 300s | MV refreshes |
| `control_tower` | 120s | MV refreshes |

## Hot Analytics Pattern

Read-heavy analytical GET endpoints use two controls together:

1. `@cached_sync(...)` or `@cached_async(...)` caches repeated identical requests.
2. `get_read_only_conn()` or `get_async_read_only_conn()` routes stale-tolerant reads to the
   read replica when `READ_REPLICA_URL` is configured, falling back to primary otherwise.

The canonical sync examples are the Dashboard endpoints in `api/routers/core/dashboard.py`
and Forecast Accuracy endpoints in `api/routers/forecasting/accuracy.py`. The pre-commit
AI check enforces that these hot routers keep `@cached_sync(...)` and `get_read_only_conn()`
together, with only `/dashboard/planning-date` intentionally uncached because it exposes the
current planning/system-date comparison.

## Configuration

`config/platform/cache_config.yaml`:

```yaml
cache:
  backend: redis            # or "memory" — see "Backends" below
  redis_url: ${REDIS_URL:-redis://localhost:6379/0}
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
```

## Backends

`common/services/cache.py` ships two backends behind a common `CacheBackend`
interface:

| Backend | Class | Use case |
|---|---|---|
| `redis` (default) | `RedisBackend` | Multi-worker production. Shared across all gunicorn workers, supports single-flight stampede protection. |
| `memory` | `InMemoryBackend` | Tests, single-process dev, automatic fallback when Redis is unavailable. |

The `REDIS_URL` env var overrides the YAML `redis_url` (highest precedence).

### Single-Flight Stampede Protection (Redis)

`RedisBackend.get_or_compute()` uses a `SETNX` lock so that when a popular key
expires under concurrent load, only the lock-holding worker recomputes the
value; all other workers briefly wait and then read the freshly populated key.
The lock has its own TTL (`lock_ttl`) so a crashed lock holder cannot block the
key indefinitely.

This eliminates the cache-stampede thundering herd that can otherwise hit the
database every time a high-traffic 60-second TTL key (e.g. `dashboard:kpis`)
expires.

### Graceful Fallback

`_build_redis_backend()` catches `ImportError` (the `redis` package is missing)
and connection-time failures (`ConnectionError`, `OSError`, `redis.RedisError`)
and transparently substitutes `InMemoryBackend`. Test environments and dev
machines without a running Redis therefore degrade to per-process cache without
any code changes.

Per-call Redis errors (`GET`/`SET`/`DELETE`/`INVALIDATE`) are logged at WARNING
and treated as cache-miss / no-op — the request still completes against the
database.

### Cache Decorators

| Decorator | For | Notes |
|---|---|---|
| `@cached(ttl, group)` | Async route handlers | Original async-only decorator. |
| `@cached_sync(ttl, group)` | Sync FastAPI handlers | Strips `Response` from cache key via `skip_kwargs` and uses backend `get_or_compute()` so Redis-backed deployments get single-flight protection. |
| `@cached_async(ttl, group)` | `async def` handlers (Item 19 pilot) | Mirrors `cached_sync` but awaits the wrapped coroutine. Used by the customer-analytics package after its async migration. |

### `reset_cache()` Semantics

`reset_cache()` is the test-isolation entrypoint. It now does two things in
order:

1. Best-effort flush of the live backend's storage — `FLUSHDB` against Redis
   or `dict.clear()` against the in-memory store. Failures are logged at DEBUG
   and ignored (teardown is non-fatal).
2. Drops the singleton so the next `get_cache()` call rebuilds cleanly.

The flush is critical when `backend=redis`: the Redis server outlives the Python
singleton, so without it consecutive tests would see leaked keys.

## Query Tracker

The query tracker is a middleware that records per-request performance:

- Logs endpoint, params, latency, and cache hit/miss status
- Warns when query latency exceeds the slow query threshold (default 2000ms)
- Flushes to `fact_query_performance` in batches (every 60s)
- Maintains an in-memory rolling window of the last 1000 queries for real-time stats
- Stats exposed via the `/health` endpoint

## Dependencies

- `redis` (optional) — required only when `backend: redis`. When absent the
  layer falls back to `InMemoryBackend` automatically.
- Stdlib `collections.OrderedDict`, `threading`, `time`, `hashlib`, `json` for
  key building and the in-memory backend.
- Cache decorator applied only to read-only GET endpoints (never mutations)

## See Also

- [API Governance](./09-api-governance.md) -- rate limiting complements caching for resource protection
- [Integration Architecture](./01-integration-architecture.md) -- overall integration overview
