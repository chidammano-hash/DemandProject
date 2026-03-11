-- 070_create_webhook_registrations.sql
-- Spec 08-10: Webhook registrations and delivery tracking

BEGIN;

-- Webhook registration
CREATE TABLE IF NOT EXISTS dim_webhook_registration (
    webhook_id      SERIAL          PRIMARY KEY,
    url             TEXT            NOT NULL,
    secret          TEXT            NOT NULL,
    event_types     JSONB           NOT NULL DEFAULT '[]',
    is_active       BOOLEAN         DEFAULT TRUE,
    created_by      UUID,
    created_at      TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_webhook_active
    ON dim_webhook_registration (is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_webhook_creator
    ON dim_webhook_registration (created_by);

-- Webhook delivery log
CREATE TABLE IF NOT EXISTS fact_webhook_delivery (
    delivery_id     BIGSERIAL       PRIMARY KEY,
    webhook_id      INT             REFERENCES dim_webhook_registration(webhook_id),
    event_type      TEXT,
    payload         JSONB,
    status_code     INT,
    response_body   TEXT,
    attempt         INT             DEFAULT 1,
    status          TEXT            DEFAULT 'pending',
    created_at      TIMESTAMPTZ     DEFAULT now(),
    delivered_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_webhook
    ON fact_webhook_delivery (webhook_id);
CREATE INDEX IF NOT EXISTS idx_webhook_delivery_event
    ON fact_webhook_delivery (event_type);
CREATE INDEX IF NOT EXISTS idx_webhook_delivery_status
    ON fact_webhook_delivery (status);

COMMIT;
