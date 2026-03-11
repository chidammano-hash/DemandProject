# 07-10 API Governance

## Overview

API governance layer providing rate limiting, request throttling, usage metering, and API versioning. Protects backend resources from abuse, tracks API consumption patterns, and enables controlled rollout of new API versions.

## Components

| Component | Path | Purpose |
|---|---|---|
| Rate limiter | `common/rate_limiter.py` | Token bucket rate limiter with per-client tracking |
| Router | `api/routers/api_governance.py` | Usage stats, rate limit status, API version info |
| Config | `config/api_governance_config.yaml` | Rate limits, quotas, version settings |

## Rate Limiter (`common/rate_limiter.py`)

### Token Bucket Algorithm
- `RateLimiter` class with per-client token buckets
- Client identification: API key, IP address, or user ID (priority order)
- `check_rate_limit(client_id: str, endpoint_group: str) -> RateLimitResult`
- `RateLimitResult(allowed: bool, remaining: int, reset_at: float, retry_after: int)`
- Thread-safe using `threading.Lock` per bucket
- Automatic bucket cleanup for inactive clients (TTL: 1 hour)

### Rate Limit Tiers
- **Default**: 100 requests/minute per client
- **Heavy endpoints** (data loads, report generation, AI analysis): 10 requests/minute
- **Read-only endpoints** (dashboards, KPIs): 200 requests/minute
- **Health/status**: unlimited

### Middleware Integration
- FastAPI middleware in `api/main.py`
- Sets response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- Returns `429 Too Many Requests` with `Retry-After` header when exceeded

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/usage` | API usage statistics (requests by endpoint, client, time) |
| GET | `/api/rate-limit-status` | Current rate limit status for calling client |
| GET | `/api/versions` | List available API versions and deprecation schedule |

## Config (`config/api_governance_config.yaml`)

```yaml
rate_limiting:
  enabled: true
  default_requests_per_minute: 100
  endpoint_groups:
    heavy:
      pattern: ["/data-quality/run", "/reports/generate", "/ai-planner/portfolio-scan",
                "/clustering/scenario", "/jobs/*/run"]
      requests_per_minute: 10
    read_only:
      pattern: ["/dashboard/*", "/inventory/*", "/forecast/*", "/inv-planning/*"]
      requests_per_minute: 200
    unlimited:
      pattern: ["/health", "/api/versions"]
      requests_per_minute: 0  # 0 = unlimited
  burst_multiplier: 1.5  # Allow 1.5x burst above sustained rate
  cleanup_ttl_seconds: 3600

versioning:
  current: "v1"
  supported: ["v1"]
  deprecation_policy:
    notice_days: 90
    sunset_days: 180

usage_tracking:
  enabled: true
  retention_days: 90
  aggregation_interval_minutes: 5
```

## Usage Tracking

- In-memory rolling counters aggregated every 5 minutes
- Tracked dimensions: endpoint, client_id, method, status_code
- Exposed via `GET /api/usage` with filters (date range, endpoint, client)
- No persistent storage required (in-memory only; optional flush to `fact_query_performance` from 07-04)

## API Versioning Strategy

- URL prefix versioning: `/v1/...` (future: `/v2/...`)
- Current implementation serves all endpoints under implicit `/v1`
- Version header: `X-API-Version` in responses
- Deprecation notices in response headers: `Deprecation: true`, `Sunset: <date>`
- Version router in `api/main.py` mounts versioned sub-applications

## Integration Points

- Rate limiter middleware added in `api/main.py` (before route matching)
- Existing `api/auth.py` (`require_api_key`) provides client identification
- RBAC (07-03) can override rate limits per role (e.g., admin = unlimited)
- Health endpoint (`/health`) extended with rate limiter stats

## Make Targets

No schema required (in-memory only). Configuration via `config/api_governance_config.yaml`.

## Dependencies

- No external dependencies (stdlib `threading`, `time`, `collections`)
