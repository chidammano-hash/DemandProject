# 07-04 API Caching Layer

## Overview

Multi-tier caching layer for frequently queried API endpoints. In-memory LRU cache with TTL expiration, automatic invalidation on data mutations, and query performance tracking for optimization.

## Components

| Component | Path | Purpose |
|---|---|---|
| Cache module | `common/cache.py` | `InMemoryBackend`, `CacheLayer` with TTL + LRU eviction |
| Query tracker | `common/query_tracker.py` | Track query latency, hit rates, slow query detection |
| DDL | `sql/064_create_query_performance.sql` | `fact_query_performance` for historical tracking |
| Config | `config/cache_config.yaml` | TTL settings, size limits, tracked endpoints |

## CacheLayer (`common/cache.py`)

### InMemoryBackend
- Thread-safe LRU cache using `collections.OrderedDict` with `threading.Lock`
- TTL-based expiration checked on read (lazy eviction) + periodic sweep
- Max entries configurable per cache namespace (default 1000)
- `get(key) -> Optional[Any]`, `set(key, value, ttl_seconds)`, `delete(key)`, `clear(namespace)`
- `stats() -> CacheStats` (hits, misses, evictions, size, hit_rate)

### CacheLayer
- Decorator-based API: `@cache_layer.cached(namespace, ttl=300)`
- Key generation from endpoint path + query params + filter hash
- Namespace isolation: `dashboard`, `accuracy`, `inventory`, `inv_planning`, `control_tower`
- Invalidation hooks: `cache_layer.invalidate(namespace)` called on data mutations
- Warm-up: optional pre-population of hot queries on startup

## Query Tracker (`common/query_tracker.py`)

- Middleware that records per-request latency and cache status
- `track_query(endpoint, params, latency_ms, cache_hit: bool)`
- Slow query detection: logs warning when latency exceeds threshold (default 2000ms)
- Periodic flush to `fact_query_performance` (batch insert every 60s)
- In-memory rolling window for real-time stats (last 1000 queries)

## Database Schema

### `fact_query_performance`
- `perf_id BIGSERIAL PRIMARY KEY`
- `endpoint TEXT NOT NULL`, `method TEXT NOT NULL`
- `params_hash TEXT`, `latency_ms INTEGER NOT NULL`
- `cache_hit BOOLEAN DEFAULT false`
- `response_size_bytes INTEGER`
- `recorded_at TIMESTAMPTZ DEFAULT NOW()`
- Indexes: `(endpoint, recorded_at DESC)`, `(cache_hit)`
- Retention: auto-purge rows older than 30 days via scheduled job

## Config (`config/cache_config.yaml`)

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

## Integration Points

- FastAPI middleware in `api/main.py` for query tracking
- Cache decorator applied to read-only GET endpoints (not mutations)
- `load_dataset_postgres.py` calls `cache_layer.invalidate()` after successful loads
- MV refresh scripts call `cache_layer.invalidate()` for relevant namespaces
- `/health` endpoint extended with cache stats

## Make Targets

```bash
make cache-schema       # Apply DDL (one-time)
make cache-purge        # Purge expired query performance records
```

## Dependencies

- No external dependencies (stdlib `collections.OrderedDict`, `threading`, `time`)
