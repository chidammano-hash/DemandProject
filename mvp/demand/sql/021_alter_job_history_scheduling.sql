-- Feature 39 Enhancement: Scheduling & automation columns for job_history
-- Adds support for cron/interval scheduling, retry logic, and pipelines.

ALTER TABLE job_history ADD COLUMN IF NOT EXISTS scheduled_cron   TEXT DEFAULT NULL;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS retry_count      SMALLINT DEFAULT 0;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS max_retries      SMALLINT DEFAULT 0;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS pipeline_id      TEXT DEFAULT NULL;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS pipeline_step    SMALLINT DEFAULT NULL;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS triggered_by     TEXT DEFAULT 'manual';

-- Schedules table: persistent record of recurring job schedules
CREATE TABLE IF NOT EXISTS job_schedule (
    schedule_id     TEXT PRIMARY KEY,
    job_type        TEXT NOT NULL,
    job_label       TEXT NOT NULL,
    cron_expr       TEXT DEFAULT NULL,
    interval_min    INTEGER DEFAULT NULL,
    params          JSONB DEFAULT '{}',
    enabled         BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_at     TIMESTAMPTZ DEFAULT NULL,
    next_run_at     TIMESTAMPTZ DEFAULT NULL,
    run_count       INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_job_history_pipeline ON job_history (pipeline_id);
CREATE INDEX IF NOT EXISTS idx_job_schedule_enabled ON job_schedule (enabled);
