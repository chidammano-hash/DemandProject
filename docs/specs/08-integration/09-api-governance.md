# API Governance

> Protects the platform from abuse with rate limiting, tracks API usage patterns for capacity planning, and provides versioning for controlled API evolution.

| | |
|---|---|
| **Status** | ✅ Implemented |
| **UI Tab** | N/A (infrastructure layer) |
| **Key Files** | `common/services/rate_limiter.py`, `config/platform/auth_config.yaml` (`governance:` block) |

---

## Problem

Without rate limiting, a single misbehaving client (or a script in a tight loop) can overwhelm the API server and degrade performance for everyone. There is no visibility into which endpoints are called most often, which clients consume the most resources, or how to plan server capacity. As the API evolves, there is no mechanism for deprecating old endpoints without breaking existing consumers.

## Solution

A sliding-window rate limiter enforces a per-IP request limit on write requests (POST/PUT/DELETE) at the middleware level. Usage tracking records request counts, latencies, and status codes by endpoint and client. API versioning is declared via config (current version `v1`) but is not enforced at runtime.

## How It Works

1. Only POST, PUT, and DELETE requests pass through `rate_limit_middleware` in `api/main.py`; GET requests are never rate-limited
2. The middleware identifies the client by IP address (`request.client.host`)
3. The sliding-window limiter checks whether that IP has remaining capacity in the trailing 60-second window, against the hardcoded `standard` tier limit (see Rate Limit Tiers)
4. If allowed, the request proceeds to the route handler
5. If exceeded, the client receives 429 Too Many Requests with a `Retry-After: 60` header
6. Usage counters are aggregated in memory every 5 minutes for monitoring

## Rate Limit Tiers

There are no endpoint-group tiers - no "heavy", "read-only", or "unlimited" categories exist in the code. The middleware always checks a single hardcoded `standard` tier, regardless of which endpoint is called:

| Tier (config) | Requests/Minute | Wired to the middleware? |
|---|---|---|
| `free` | 60 | No - declared in config, never read at runtime |
| `standard` | 300 | Yes - the only tier `rate_limit_middleware` ever requests |
| `premium` | 1000 | No - declared in config, never read at runtime |

Every rate-limited request (POST/PUT/DELETE only) is checked against the same 300/minute ceiling, keyed by client IP address, in a rolling 60-second window.

## Response Headers

`rate_limit_middleware` does not set `X-RateLimit-*` headers on any response. The only rate-limit-related header is `Retry-After`, sent solely on 429 responses:

| Header | When | Description |
|---|---|---|
| `Retry-After` | Only on 429 responses | Hardcoded to `60` (seconds) |

## API Versioning

Versioning is config-only today and has no runtime effect:

| Aspect | Reality |
|---|---|
| Config keys | `governance.versioning.current_version` (`v1`), `governance.versioning.supported_versions` (`[v1]`), `governance.deprecation.sunset_header` (boolean) in `config/platform/auth_config.yaml` |
| Runtime enforcement | None - no router uses a `/v1` URL prefix, and no code path reads `governance.versioning.*` or `governance.deprecation.*` to reject a request, route by version, or add an `X-API-Version`, `Deprecation`, or `Sunset` response header |
| Where it's actually used | Only as editable fields in the admin Config Manager UI (`api/routers/platform/config_manager.py`) - an operator can change the value, but nothing downstream consumes it |

## Configuration

`config/platform/auth_config.yaml`, `governance:` block (merged from the former standalone `api_governance_config.yaml` - the file documents this itself: "Merged from api_governance_config.yaml (Spec 08-09)"):

```yaml
governance:
  rate_limit_tiers:
    free:
      requests_per_minute: 60
    standard:
      requests_per_minute: 300
    premium:
      requests_per_minute: 1000
  deprecation:
    sunset_header: true
  versioning:
    current_version: v1
    supported_versions:
      - v1
```

Only `rate_limit_tiers.standard.requests_per_minute` is read at runtime, via `RateLimiter.get_tier_limit("standard")`. The remaining keys are not currently consumed by any code path.

## Usage Tracking

- In-memory rolling counters aggregated every 5 minutes
- Tracked dimensions: endpoint, client_id, method, status_code
- No persistent storage required (in-memory only; optional flush to `fact_query_performance` from [Caching](./03-caching.md))
- Stats exposed via `/health` endpoint

## Dependencies

- No external dependencies -- uses stdlib `threading`, `time`, `collections`
- Rate limiter middleware added in `api/main.py` before route matching
- Client identification is the raw request IP (`request.client.host`); it does not call `api/auth.py`'s `require_api_key`

## See Also

- [Caching](./03-caching.md) -- caching complements rate limiting for resource protection
- [RBAC](./02-rbac.md) - role-based access control (rate limiting itself is IP-based and does not vary by role)
- [Integration Architecture](./01-integration-architecture.md) -- API governance as Vector 2 component
