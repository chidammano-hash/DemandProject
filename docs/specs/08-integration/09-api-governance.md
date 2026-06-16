# API Governance

> Protects the platform from abuse with rate limiting, tracks API usage patterns for capacity planning, and provides versioning for controlled API evolution.

| | |
|---|---|
| **Status** | ✅ Implemented |
| **UI Tab** | N/A (infrastructure layer) |
| **Key Files** | `common/services/rate_limiter.py`, `config/api_governance_config.yaml` |

---

## Problem

Without rate limiting, a single misbehaving client (or a script in a tight loop) can overwhelm the API server and degrade performance for everyone. There is no visibility into which endpoints are called most often, which clients consume the most resources, or how to plan server capacity. As the API evolves, there is no mechanism for deprecating old endpoints without breaking existing consumers.

## Solution

A token bucket rate limiter enforces per-client request limits at the middleware level. Usage tracking records request counts, latencies, and status codes by endpoint and client. API versioning via URL prefixes allows controlled rollout of breaking changes with deprecation notices.

## How It Works

1. Every API request passes through rate limit middleware before reaching the route handler
2. The middleware identifies the client (by API key, IP address, or user ID) and the endpoint group
3. The token bucket checks whether the client has remaining capacity for this endpoint group
4. If allowed, the request proceeds; response headers show remaining quota and reset time
5. If exceeded, the client receives 429 Too Many Requests with a `Retry-After` header
6. Usage counters are aggregated in memory every 5 minutes for monitoring
7. Inactive client buckets are cleaned up after 1 hour

## Rate Limit Tiers

| Tier | Requests/Minute | Applies To |
|---|---|---|
| Default | 100 | Most endpoints |
| Heavy | 10 | Data loads, report generation, AI analysis, clustering scenarios |
| Read-only | 200 | Dashboards, KPIs, inventory, forecasts |
| Unlimited | No limit | Health checks, version info |

The `burst_multiplier` (default 1.5x) allows short bursts above the sustained rate.

## Response Headers

Every response includes rate limit headers:

| Header | Description |
|---|---|
| `X-RateLimit-Limit` | Maximum requests per window |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Unix timestamp when the window resets |
| `Retry-After` | Seconds to wait (only on 429 responses) |

## API Versioning

| Aspect | Approach |
|---|---|
| Strategy | URL prefix versioning (`/v1/...`) |
| Current version | `v1` |
| Version header | `X-API-Version` in all responses |
| Deprecation notice | `Deprecation: true` + `Sunset: <date>` headers |
| Policy | 90 days deprecation notice, 180 days until sunset |

## Configuration

`config/api_governance_config.yaml`:

```yaml
rate_limiting:
  enabled: true
  default_requests_per_minute: 100
  endpoint_groups:
    heavy:
      pattern: ["/data-quality/run", "/reports/generate", "/ai-planner/portfolio-scan"]
      requests_per_minute: 10
    read_only:
      pattern: ["/dashboard/*", "/inventory/*", "/forecast/*"]
      requests_per_minute: 200
    unlimited:
      pattern: ["/health", "/api/versions"]
      requests_per_minute: 0
  burst_multiplier: 1.5
  cleanup_ttl_seconds: 3600
versioning:
  current: "v1"
  supported: ["v1"]
usage_tracking:
  enabled: true
  retention_days: 90
  aggregation_interval_minutes: 5
```

## Usage Tracking

- In-memory rolling counters aggregated every 5 minutes
- Tracked dimensions: endpoint, client_id, method, status_code
- No persistent storage required (in-memory only; optional flush to `fact_query_performance` from [Caching](./03-caching.md))
- Stats exposed via `/health` endpoint

## Dependencies

- No external dependencies -- uses stdlib `threading`, `time`, `collections`
- Rate limiter middleware added in `api/main.py` before route matching
- Existing `api/auth.py` (`require_api_key`) provides client identification
- [RBAC](./02-rbac.md) can override rate limits per role (e.g., admin = unlimited)

## See Also

- [Caching](./03-caching.md) -- caching complements rate limiting for resource protection
- [RBAC](./02-rbac.md) -- role-based rate limit overrides
- [Integration Architecture](./01-integration-architecture.md) -- API governance as Vector 2 component
