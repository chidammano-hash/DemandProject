# Notification Engine

> Sends alerts to planners across multiple channels (in-app, email, Slack, webhooks) when jobs complete, exceptions fire, or AI insights need attention.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (sidebar badge for unread count) |
| **Key Files** | `common/services/notification_engine.py`, `api/routers/platform/notifications.py`, `config/platform/notification_config.yaml` |

---

## Problem

When a backtest job fails at 2 AM or the AI planner flags a critical stockout risk, nobody knows until they open the app. Planners check the platform intermittently and may miss time-sensitive alerts. There is no way to route different severity levels to different channels -- a critical stockout should page on-call, while a routine job completion just needs a Slack message.

## Solution

A multi-channel notification engine dispatches alerts based on event type and severity. Each channel (in-app, email, Slack, PagerDuty, webhook) has its own adapter. Routing rules in config map event-severity combinations to specific channels. Users can set per-event channel preferences. All sends are non-fatal -- a failed Slack webhook never crashes a job. Deduplication prevents the same alert from firing repeatedly within a 5-minute window.

## How It Works

1. A platform event occurs (job completes, exception detected, AI insight created)
2. The source module calls `NotificationEngine.send(event_type, payload, recipients)`
3. The engine looks up routing rules: which channels should receive this event at this severity?
4. For each target channel, the engine renders a message from a Jinja2 template
5. The adapter delivers the message (HTTP POST for Slack/Teams/PagerDuty, SMTP for email)
6. Failed deliveries retry with exponential backoff (max 3 retries, base 30s)
7. Duplicate events for the same recipient within 5 minutes are suppressed
8. All delivery attempts are logged to `fact_notification_log`

## Event Types

| Event | Trigger Source | Typical Channels |
|---|---|---|
| `job_completed` | Job scheduler | In-app, Slack |
| `exception_detected` | Storyboard engine | Slack, PagerDuty (if critical) |
| `ai_insight_created` | AI planner | In-app, Slack, Email |
| `threshold_breach` | KPI monitoring | Slack, PagerDuty |
| `data_quality_alert` | DQ engine | In-app, Slack |
| `approval_required` | S&OP cycle | In-app, Email |

## Data Model

### `fact_notification_log`

| Column | Type | Description |
|---|---|---|
| `notification_id` | `BIGSERIAL PK` | Auto-increment ID |
| `event_type` | `TEXT` | Event that triggered the notification |
| `channel` | `TEXT` | Delivery channel (in_app, email, slack, webhook) |
| `recipient` | `TEXT` | User ID or channel identifier |
| `subject` | `TEXT` | Notification subject line |
| `body` | `TEXT` | Notification body |
| `payload` | `JSONB` | Event-specific data |
| `status` | `TEXT` | pending, sent, failed, suppressed |
| `retry_count` | `INTEGER` | Number of retry attempts |
| `sent_at` | `TIMESTAMPTZ` | When successfully delivered |
| `created_at` | `TIMESTAMPTZ` | When the notification was created |

### `dim_notification_preference`

| Column | Type | Description |
|---|---|---|
| `pref_id` | `SERIAL PK` | Auto-increment ID |
| `user_id` | `INTEGER` | User (nullable for system defaults) |
| `event_type` | `TEXT` | Event type this preference applies to |
| `channels` | `TEXT[]` | Which channels to use (default: `{in_app}`) |
| `is_enabled` | `BOOLEAN` | Whether notifications are enabled for this event |

Unique constraint on `(user_id, event_type)`.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/notifications` | List notifications for current user (paginated) |
| GET | `/notifications/unread-count` | Count of unread notifications |
| PUT | `/notifications/{id}/read` | Mark a notification as read |
| PUT | `/notifications/read-all` | Mark all notifications as read |
| GET | `/notifications/preferences` | Get user notification preferences |
| PUT | `/notifications/preferences` | Update channel preferences per event type |

## Configuration

`config/platform/notification_config.yaml`:

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
  exception_detected: "New {severity} exception: {exception_type} for {item_id}/{loc}"
  ai_insight_created: "AI insight ({severity}): {summary}"
```

Secrets use environment variable references (`SMTP_HOST`, `SLACK_WEBHOOK_URL`), never stored in YAML.

## Dependencies

- `jinja2>=3.1` for template rendering (already in deps)
- Optional: `aiosmtplib` for async email, `httpx` for Slack/webhook delivery
- Frontend polls `/notifications/unread-count` every 30s; badge shown in sidebar header

## See Also

- [Integration Architecture](./01-integration-architecture.md) -- notifications as Vector 1
- [Webhooks](./10-webhooks.md) -- outbound event delivery to external systems
- [Storyboard](../06-ai-platform/04-storyboard.md) -- exception events that trigger notifications
