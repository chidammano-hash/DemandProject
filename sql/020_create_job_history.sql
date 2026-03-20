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
    progress_msg    TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_history_status    ON job_history (status);
CREATE INDEX IF NOT EXISTS idx_job_history_type      ON job_history (job_type);
CREATE INDEX IF NOT EXISTS idx_job_history_submitted ON job_history (submitted_at DESC);
