# 07-05 Notification Engine

## Overview

Multi-channel notification system for alerting users on job completions, exception detections, AI insights, and threshold breaches. Supports in-app, email, Slack, and webhook channels with delivery tracking.

## Components

| Component | Path | Purpose |
|---|---|---|
| Engine | `common/notification_engine.py` | `NotificationEngine`: dispatch, retry, channel routing |
| Router | `api/routers/notifications.py` | 6 REST endpoints for notification management |
| DDL | `sql/059_create_notification_log.sql` | `fact_notification_log`, `dim_notification_preference` |
| Config | `config/notification_config.yaml` | Channel configs, templates, retry policy |

## NotificationEngine (`common/notification_engine.py`)

- `NotificationEngine` singleton with channel adapters
- `send(event_type: str, payload: dict, recipients: list[str])` — routes to configured channels
- Channel adapters: `InAppAdapter`, `EmailAdapter`, `SlackAdapter`, `WebhookAdapter`
- Template rendering: Jinja2-based templates per event type
- Retry with exponential backoff (max 3 retries, base 30s)
- Delivery tracking: all sends logged to `fact_notification_log`
- Deduplication: same event + recipient within 5 minutes is suppressed

### Event Types
- `job_completed` — job finishes (success or failure)
- `exception_detected` — new critical/high exception in storyboard
- `ai_insight_created` — new AI planner insight with severity >= high
- `threshold_breach` — KPI crosses configured threshold (e.g., WAPE > 30%)
- `data_quality_alert` — DQ check fails with critical severity
- `approval_required` — S&OP stage requires planner approval

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/notifications` | List notifications for current user (paginated) |
| GET | `/notifications/unread-count` | Count of unread notifications |
| PUT | `/notifications/{id}/read` | Mark notification as read |
| PUT | `/notifications/read-all` | Mark all as read |
| GET | `/notifications/preferences` | Get user notification preferences |
| PUT | `/notifications/preferences` | Update channel preferences per event type |

## Database Schema

### `fact_notification_log`
- `notification_id BIGSERIAL PRIMARY KEY`
- `event_type TEXT NOT NULL`, `channel TEXT NOT NULL` (in_app, email, slack, webhook)
- `recipient TEXT NOT NULL` (user_id or channel identifier)
- `subject TEXT`, `body TEXT`
- `payload JSONB` (event-specific data)
- `status TEXT DEFAULT 'pending'` (pending, sent, failed, suppressed)
- `retry_count INTEGER DEFAULT 0`
- `sent_at TIMESTAMPTZ`, `created_at TIMESTAMPTZ DEFAULT NOW()`
- Indexes: `(recipient, created_at DESC)`, `(event_type, status)`, `(status) WHERE status = 'pending'`

### `dim_notification_preference`
- `pref_id SERIAL PRIMARY KEY`
- `user_id INTEGER` (nullable for defaults)
- `event_type TEXT NOT NULL`
- `channels TEXT[] NOT NULL DEFAULT '{in_app}'`
- `is_enabled BOOLEAN DEFAULT true`
- `UNIQUE(user_id, event_type)`

## Config (`config/notification_config.yaml`)

```yaml
channels:
  in_app:
    enabled: true
  email:
    enabled: false
    smtp_host_env: SMTP_HOST
    smtp_port: 587
    from_address: "demand-studio@company.com"
  slack:
    enabled: false
    webhook_url_env: SLACK_WEBHOOK_URL
  webhook:
    enabled: false
defaults:
  retry_max: 3
  retry_base_seconds: 30
  dedup_window_minutes: 5
templates:
  job_completed: "Job '{job_type}' completed with status: {status}"
  exception_detected: "New {severity} exception: {exception_type} for {item_no}/{loc}"
  ai_insight_created: "AI insight ({severity}): {summary}"
  threshold_breach: "{metric} breached threshold: {value} (limit: {threshold})"
```

## Integration Points

- `common/job_registry.py` — emits `job_completed` on job finish
- `common/exception_engine.py` — emits `exception_detected` on critical exceptions
- `common/ai_planner.py` — emits `ai_insight_created` on high-severity insights
- Frontend: poll `/notifications/unread-count` every 30s, badge in sidebar header

## Make Targets

```bash
make notification-schema    # Apply DDL (one-time)
```

## Dependencies

- `jinja2>=3.1` (template rendering, already in deps)
- Optional: `aiosmtplib` for async email, `httpx` for Slack/webhook delivery
