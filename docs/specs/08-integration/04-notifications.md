# Notification Engine

> Dispatches alerts to Slack, Teams, email, and PagerDuty via per-channel adapters, driven by event-type/severity routing rules, and logs every delivery attempt.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A |
| **Key Files** | `common/services/notification_engine.py`, `api/routers/platform/notifications.py`, `config/platform/notification_config.yaml` |

---

## Problem

When a backtest job fails at 2 AM or the AI planner flags a critical stockout risk, nobody knows until they open the app. Planners check the platform intermittently and may miss time-sensitive alerts. There is no way to route different severity levels to different channels -- a critical stockout should page on-call, while a routine job completion just needs a Slack message.

## Solution

A notification engine dispatches alerts to Slack, Teams, email, and PagerDuty based on event type and severity. Each channel has its own adapter (webhook POST for Slack/Teams/PagerDuty, SMTP for email). Routing rules in `notification_config.yaml` map event-type + severity combinations to a channel list and a minimum-severity threshold. A cooldown window (default 5 minutes, keyed by event type + severity + subject) suppresses repeated sends of the same alert. All sends are best-effort - a failed webhook or SMTP call never crashes the caller. Every delivery attempt (success or failure) is logged to `fact_notification_log`.

In the shipped codebase, only two call sites actually invoke the engine: the `mention` event fired when an annotation `@mentions` a user (see [Collaboration](./05-collaboration.md)), and the manual `test` event fired by `POST /notifications/test`. The `notification_config.yaml` routing rules also declare channels for job/exception/DQ/inventory/S&OP event types, but no job scheduler, storyboard, AI planner, DQ, or S&OP code currently calls `NotificationEngine.send()` for those - they are configured but not yet wired to a trigger source.

## How It Works

1. A caller invokes `NotificationEngine().send(event_type, severity, subject, body, recipient)` - today that's either the collaboration router (on an `@mention`) or the `POST /notifications/test` endpoint.
2. The engine checks an in-process cooldown keyed by `event_type:severity:subject-prefix` (default 300s, from `rate_limits.cooldown_seconds`); a repeat within the window is skipped, regardless of recipient.
3. It resolves which channels should receive the event from `routing_rules` in `notification_config.yaml` (event type + severity -> channel list, with a minimum-severity threshold per rule).
4. For each target channel, the matching adapter sends the message once: `_send_slack`/`_send_teams` (webhook POST), `_send_email` (SMTP), or `_send_pagerduty` (Events API v2 POST). There is no retry or template-rendering layer - subject/body are passed through as-is from the caller.
5. Every attempt is logged to `fact_notification_log`, best-effort - a logging failure never raises.

## Channels

Declared in `dim_notification_channel.channel_type` and implemented by an adapter in `CHANNEL_SENDERS`:

| Channel | Adapter | Transport |
|---|---|---|
| `slack` | `_send_slack` | Incoming webhook (HTTP POST) |
| `teams` | `_send_teams` | Incoming webhook (HTTP POST) |
| `email` | `_send_email` | SMTP |
| `pagerduty` | `_send_pagerduty` | Events API v2 (HTTP POST) |

## Data Model

### `dim_notification_channel`

| Column | Type | Description |
|---|---|---|
| `channel_id` | `SERIAL PK` | Auto-increment ID |
| `channel_type` | `TEXT` | `slack`, `teams`, `email`, or `pagerduty` |
| `config` | `JSONB` | Channel-specific configuration (default `{}`) |
| `enabled` | `BOOLEAN` | Whether the channel is active (default `TRUE`) |
| `created_at` | `TIMESTAMPTZ` | When the channel was configured |

### `fact_notification_log`

| Column | Type | Description |
|---|---|---|
| `notification_id` | `BIGSERIAL PK` | Auto-increment ID |
| `channel_id` | `INT FK` | References `dim_notification_channel(channel_id)` |
| `event_type` | `TEXT` | Event that triggered the notification |
| `severity` | `TEXT` | Event severity |
| `recipient` | `TEXT` | User ID, email, or channel identifier |
| `subject` | `TEXT` | Notification subject line |
| `body` | `TEXT` | Notification body |
| `status` | `TEXT` | `pending`, `sent`, `delivered`, or `failed` (default `pending`) |
| `error` | `TEXT` | Error detail when delivery failed |
| `created_at` | `TIMESTAMPTZ` | When the delivery attempt was logged |
| `delivered_at` | `TIMESTAMPTZ` | When delivery succeeded |

There is no per-user preference table or unread/read state - this is a channel-config + delivery-log system, not an in-app inbox.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/notifications/history` | View notification delivery history, filterable by `event_type` / `status`, paginated via `limit` (default 50, max 500) |
| GET | `/notifications/channels` | List configured notification channels |
| POST | `/notifications/test` | Send a test notification through the engine (manager role + API key required) |

## Configuration

`config/platform/notification_config.yaml` defines per-channel settings, routing rules (event type + severity -> channels), and rate limits, e.g.:

```yaml
channels:
  slack:
    enabled: true
    webhook_url: ${SLACK_WEBHOOK_URL}
  email:
    enabled: false
    smtp_host: ${SMTP_HOST:-localhost}
    smtp_port: ${SMTP_PORT:-587}
    from_address: ${SMTP_FROM:-noreply@demandstudio.local}
  pagerduty:
    enabled: false
    routing_key: ${PAGERDUTY_ROUTING_KEY}

routing_rules:
  - event_type: stockout_alert
    severity: [critical]
    channels: [slack, pagerduty]

rate_limits:
  cooldown_seconds: 300
```

Secrets use environment variable references (`SMTP_HOST`, `SLACK_WEBHOOK_URL`, `PAGERDUTY_ROUTING_KEY`), never stored in YAML.

## Dependencies

- No external dependencies - channel adapters use stdlib `smtplib` (email) and `urllib.request` (Slack/Teams/PagerDuty webhooks)

## See Also

- [Integration Architecture](./01-integration-architecture.md) -- notifications as Vector 1
- [Webhooks](./10-webhooks.md) -- outbound event delivery to external systems
- [Collaboration](./05-collaboration.md) - the only current caller of the notification engine, via `@mention` alerts
