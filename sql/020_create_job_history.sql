-- Feature 39: Job Scheduler / Monitor
-- Persistent job history table for tracking long-running operations.

CREATE TABLE IF NOT EXISTS job_history (
    job_id          TEXT PRIMARY KEY,
    job_type        TEXT NOT NULL,
    job_label       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    params          JSONB DEFAULT '{}',
    result          JSONB DEFAULT NULL,
    error           TEXT DEFAULT NULL,
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ DEFAULT NULL,
    completed_at    TIMESTAMPTZ DEFAULT NULL,
    progress_pct    SMALLINT DEFAULT 0,
    progress_msg    TEXT DEFAULT NULL,
    logs            JSONB DEFAULT '[]',
    -- Scheduling & automation columns
    scheduled_cron  TEXT DEFAULT NULL,
    retry_count     SMALLINT DEFAULT 0,
    max_retries     SMALLINT DEFAULT 0,
    pipeline_id     TEXT DEFAULT NULL,
    pipeline_step   SMALLINT DEFAULT NULL,
    triggered_by    TEXT DEFAULT 'manual'
);

CREATE INDEX IF NOT EXISTS idx_job_history_status    ON job_history (status);
CREATE INDEX IF NOT EXISTS idx_job_history_type      ON job_history (job_type);
CREATE INDEX IF NOT EXISTS idx_job_history_submitted ON job_history (submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_history_pipeline  ON job_history (pipeline_id);

-- Persistent record of recurring job schedules
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

CREATE INDEX IF NOT EXISTS idx_job_schedule_enabled ON job_schedule (enabled);
