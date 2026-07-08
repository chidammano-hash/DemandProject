# Webhook Dispatcher

> Pushes platform events (forecast published, job completed, exception created) to external systems via signed HTTP callbacks with retry logic and delivery tracking.

| | |
|---|---|
| **Status** | ✅ Implemented |
| **UI Tab** | N/A (API-driven) |
| **Key Files** | `common/services/webhook_dispatcher.py`, `api/routers/platform/webhooks.py` |

---

## Problem

External systems (ERP portals, BI dashboards, partner platforms) need to know when something happens in Supply Chain Command Center -- a new forecast is published, an exception is detected, or a job completes. Without webhooks, these systems must poll the API repeatedly, wasting resources and introducing delays between events and reactions.

## Solution

An outbound webhook system lets external consumers register callback URLs for specific event types. When an event occurs, the dispatcher fans out signed HTTP POST requests to all active subscribers. Payloads are signed with HMAC-SHA256 so consumers can verify authenticity. Failed deliveries retry up to 3 attempts total with exponential backoff; if every attempt fails, the dispatcher gives up and logs the failure - there is no circuit breaker that disables a registration.

## How It Works

1. An external system registers a webhook via `POST /webhooks/register` with a target URL and list of event types
2. The system receives a signing secret (auto-generated or provided) for payload verification
3. When a subscribed event occurs, the dispatcher sends an HTTP POST with a signed JSON payload
4. The consumer verifies the `X-Webhook-Signature` header by computing the same HMAC
5. If delivery fails, `dispatch_webhook()` retries up to 3 attempts total, with exponential backoff between attempts (~2s after the first failure, ~4s after the second) and a 10-second timeout per attempt
6. If every attempt fails, the dispatcher gives up - there is no circuit breaker, and the registration is left active
7. One row is written to `fact_webhook_delivery` per event per webhook, recording the outcome and attempt number of the last try (not one row per individual attempt)

## Supported Event Types

| Event | Trigger |
|---|---|
| `forecast.generated` | New production forecast version created |
| `job.completed` | Scheduled job finished (success or failure) |
| `exception.created` | New storyboard exception detected |
| `insight.created` | New AI planner insight generated |
| `data.loaded` | Dataset loaded into Postgres |
| `approval.required` | S&OP stage needs planner approval |
| `threshold.breached` | Monitored KPI crossed configured threshold |

## Payload Format

Every webhook delivery includes:

| Field | Description |
|---|---|
| `event_type` | Which event occurred |
| `event_id` | Unique UUID for this event instance |
| `timestamp` | ISO 8601 timestamp |
| `data` | Event-specific payload (job details, forecast version, exception info, etc.) |

## Security

| Aspect | Approach |
|---|---|
| Signing | HMAC-SHA256: `signature = HMAC(secret, json_payload, SHA256)` |
| Verification header | `X-Webhook-Signature` |
| Secret storage | Hashed in DB; raw secret returned only on registration response |
| URL requirement | Must be HTTPS in production mode |

## Data Model

### `dim_webhook_registration`

| Column | Type | Description |
|---|---|---|
| `webhook_id` | `SERIAL PK` | Auto-increment ID |
| `url` | `TEXT` | Target endpoint URL |
| `events` | `TEXT[]` | List of subscribed event types |
| `secret` | `TEXT` | HMAC signing key |
| `description` | `TEXT` | Human-readable label |
| `is_active` | `BOOLEAN` | Whether the webhook is active |

### `fact_webhook_delivery`

| Column | Type | Description |
|---|---|---|
| `delivery_id` | `BIGSERIAL PK` | Auto-increment ID |
| `webhook_id` | `INTEGER FK` | Which registration this delivery is for |
| `event_type` | `TEXT` | Event that triggered delivery |
| `event_id` | `TEXT` | Unique event instance ID |
| `payload` | `JSONB` | Full payload sent |
| `attempt` | `INTEGER` | Attempt number (1, 2, or 3) |
| `status_code` | `INTEGER` | HTTP response code from target |
| `response_body` | `TEXT` | First 1000 chars of response |
| `latency_ms` | `INTEGER` | Delivery round-trip time |
| `status` | `TEXT` | pending, success, failed, retrying |
| `error_message` | `TEXT` | Error details if failed |
| `delivered_at` | `TIMESTAMPTZ` | When the delivery attempt occurred |

## API

| Method | Path | Description |
|---|---|---|
| POST | `/webhooks/register` | Register a new webhook (URL, events, secret) |
| GET | `/webhooks` | List all registered webhooks |
| POST | `/webhooks/{id}/test` | Send a test payload to verify the endpoint |
| DELETE | `/webhooks/{id}` | Deactivate a webhook registration |
| GET | `/webhooks/deliveries?webhook_id=<id>` | Delivery history, optionally filtered by webhook (default 50 results, `limit` caps at 500) |

## Configuration

No config file required. Retry count (3 attempts total), backoff base (2.0, i.e. ~2s / ~4s between attempts), and delivery timeout (10s) are constants (`max_retries`, `backoff_base`) in `common/services/webhook_dispatcher.py`.

## Dependencies

- `httpx>=0.27` for async HTTP delivery (already in deps)
- `hmac`, `hashlib` from stdlib for payload signing
- Event emitters call `WebhookDispatcher.dispatch()` alongside notification engine calls

## See Also

- [Notifications](./04-notifications.md) -- in-app/email/Slack alerting (complementary to webhooks)
- [Integration Architecture](./01-integration-architecture.md) -- webhooks as part of Vector 2
- [API Governance](./09-api-governance.md) -- rate limiting for inbound webhook consumers
