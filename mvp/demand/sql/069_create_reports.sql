-- 069_create_reports.sql
-- Spec 08-08: Report templates, schedules, and delivery tracking

BEGIN;

-- Report template definitions
CREATE TABLE IF NOT EXISTS dim_report_template (
    template_id     SERIAL          PRIMARY KEY,
    name            TEXT            UNIQUE,
    report_type     TEXT            NOT NULL,
    query_config    JSONB,
    layout          JSONB,
    created_by      UUID,
    is_system       BOOLEAN         DEFAULT FALSE,
    created_at      TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_report_template_type
    ON dim_report_template (report_type);
CREATE INDEX IF NOT EXISTS idx_report_template_creator
    ON dim_report_template (created_by);

-- Report schedules
CREATE TABLE IF NOT EXISTS fact_report_schedule (
    schedule_id     SERIAL          PRIMARY KEY,
    template_id     INT             REFERENCES dim_report_template(template_id),
    recipients      JSONB,
    cron            TEXT,
    format          TEXT            DEFAULT 'pdf',
    enabled         BOOLEAN         DEFAULT TRUE,
    last_run_at     TIMESTAMPTZ,
    next_run_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_report_schedule_template
    ON fact_report_schedule (template_id);
CREATE INDEX IF NOT EXISTS idx_report_schedule_enabled
    ON fact_report_schedule (enabled) WHERE enabled = TRUE;
CREATE INDEX IF NOT EXISTS idx_report_schedule_next_run
    ON fact_report_schedule (next_run_at);

-- Report delivery log
CREATE TABLE IF NOT EXISTS fact_report_delivery (
    delivery_id     BIGSERIAL       PRIMARY KEY,
    schedule_id     INT             REFERENCES fact_report_schedule(schedule_id),
    status          TEXT            DEFAULT 'pending',
    file_path       TEXT,
    error           TEXT,
    created_at      TIMESTAMPTZ     DEFAULT now(),
    delivered_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_report_delivery_schedule
    ON fact_report_delivery (schedule_id);
CREATE INDEX IF NOT EXISTS idx_report_delivery_status
    ON fact_report_delivery (status);

COMMIT;
