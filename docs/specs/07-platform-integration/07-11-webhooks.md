# 07-11 Webhook Dispatcher

## Overview

Outbound webhook system for notifying external services of platform events. Consumers register webhook endpoints for specific event types, and the dispatcher delivers signed payloads with retry logic and delivery tracking.

## Components

| Component | Path | Purpose |
|---|---|---|
| Dispatcher | `common/webhook_dispatcher.py` | `WebhookDispatcher`: payload signing, delivery, retry |
| Router | `api/routers/webhooks.py` | 5 REST endpoints for webhook registration and management |
| DDL | `sql/070_create_webhook_registrations.sql` | `dim_webhook_registration`, `fact_webhook_delivery` |

## WebhookDispatcher (`common/webhook_dispatcher.py`)

- `WebhookDispatcher` singleton with async delivery via thread pool
- `dispatch(event_type: str, payload: dict)` — fans out to all active registrations for the event
- HMAC-SHA256 payload signing: `X-Webhook-Signature` header using per-registration secret
- Retry with exponential backoff: 3 attempts at 30s, 120s, 600s intervals
- Timeout: 10s per delivery attempt
- Circuit breaker: disable registration after 10 consecutive failures (auto-re-enable after 1 hour)
- Delivery tracking: all attempts logged to `fact_webhook_delivery`

### Supported Event Types
- `forecast.generated` — new production forecast version created
- `job.completed` — scheduled job finished (success or failure)
- `exception.created` — new storyboard exception detected
- `insight.created` — new AI planner insight generated
- `data.loaded` — dataset loaded into Postgres
- `approval.required` — S&OP stage needs approval
- `threshold.breached` — monitored KPI crossed threshold

### Payload Format
```json
{
  "event_type": "job.completed",
  "event_id": "uuid-v4",
  "timestamp": "2026-03-11T10:30:00Z",
  "data": {
    "job_id": "abc-123",
    "job_type": "backtest_lgbm",
    "status": "completed",
    "duration_seconds": 142
  }
}
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/webhooks` | Register a new webhook (URL, events, secret) |
| GET | `/webhooks` | List all registered webhooks |
| POST | `/webhooks/{id}/test` | Send a test payload to verify endpoint |
| DELETE | `/webhooks/{id}` | Deactivate a webhook registration |
| GET | `/webhooks/{id}/deliveries` | List delivery history for a webhook (paginated) |

## Database Schema

### `dim_webhook_registration`
- `webhook_id SERIAL PRIMARY KEY`
- `url TEXT NOT NULL` (target endpoint URL)
- `events TEXT[] NOT NULL` (list of subscribed event types)
- `secret TEXT NOT NULL` (HMAC signing key, auto-generated if not provided)
- `description TEXT`
- `is_active BOOLEAN DEFAULT true`
- `consecutive_failures INTEGER DEFAULT 0`
- `disabled_until TIMESTAMPTZ` (circuit breaker re-enable time)
- `created_at TIMESTAMPTZ DEFAULT NOW()`, `updated_at TIMESTAMPTZ`
- Index: `(is_active) WHERE is_active = true`

### `fact_webhook_delivery`
- `delivery_id BIGSERIAL PRIMARY KEY`
- `webhook_id INTEGER REFERENCES dim_webhook_registration(webhook_id)`
- `event_type TEXT NOT NULL`, `event_id TEXT NOT NULL`
- `payload JSONB NOT NULL`
- `attempt INTEGER DEFAULT 1` (1, 2, or 3)
- `status_code INTEGER` (HTTP response code from target)
- `response_body TEXT` (first 1000 chars of response)
- `latency_ms INTEGER`
- `status TEXT DEFAULT 'pending'` (pending, success, failed, retrying)
- `error_message TEXT`
- `delivered_at TIMESTAMPTZ DEFAULT NOW()`
- Indexes: `(webhook_id, delivered_at DESC)`, `(event_type, status)`, `(status) WHERE status = 'pending'`

## Integration Points

- Event emitters call `WebhookDispatcher.dispatch()` at the same points as notification engine (07-05)
- `common/job_registry.py` emits `job.completed` on job finish
- `scripts/generate_production_forecasts.py` emits `forecast.generated` after write
- `common/exception_engine.py` emits `exception.created` for new exceptions
- `common/ai_planner.py` emits `insight.created` for new insights
- Notification engine (07-05) `WebhookAdapter` delegates to `WebhookDispatcher`

## Security

- HMAC-SHA256 signing: `signature = HMAC(secret, json_payload, SHA256)`
- Consumers verify signature via `X-Webhook-Signature` header
- Secrets stored hashed in DB; raw secret returned only on registration response
- URL validation: must be HTTPS in production mode

## Make Targets

```bash
make webhook-schema     # Apply DDL (one-time)
```

## Dependencies

- `httpx>=0.27` for async HTTP delivery (already in deps)
- `hmac`, `hashlib` from stdlib for payload signing
