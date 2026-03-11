-- 065_create_notification_log.sql
-- Notification channels + delivery log tables (Spec 08-04)

CREATE TABLE IF NOT EXISTS dim_notification_channel (
    channel_id   SERIAL PRIMARY KEY,
    channel_type TEXT NOT NULL,                  -- slack, teams, email, pagerduty
    config       JSONB NOT NULL DEFAULT '{}',    -- channel-specific configuration
    enabled      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_notification_channel_type
    ON dim_notification_channel (channel_type);
CREATE INDEX IF NOT EXISTS idx_notification_channel_enabled
    ON dim_notification_channel (enabled) WHERE enabled = TRUE;

CREATE TABLE IF NOT EXISTS fact_notification_log (
    notification_id BIGSERIAL PRIMARY KEY,
    channel_id      INT REFERENCES dim_notification_channel (channel_id),
    event_type      TEXT NOT NULL,
    severity        TEXT,
    recipient       TEXT,
    subject         TEXT,
    body            TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending, sent, delivered, failed
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    delivered_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_notification_log_event_type
    ON fact_notification_log (event_type);
CREATE INDEX IF NOT EXISTS idx_notification_log_status
    ON fact_notification_log (status);
CREATE INDEX IF NOT EXISTS idx_notification_log_created
    ON fact_notification_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notification_log_channel
    ON fact_notification_log (channel_id);
CREATE INDEX IF NOT EXISTS idx_notification_log_severity
    ON fact_notification_log (severity);
