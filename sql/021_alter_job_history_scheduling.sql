-- 021_alter_job_history_scheduling.sql
-- Idempotent migration: add scheduling columns to job_history.
-- Columns already exist in CREATE TABLE (020) for fresh installs.

ALTER TABLE job_history ADD COLUMN IF NOT EXISTS scheduled_cron TEXT DEFAULT NULL;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS retry_count SMALLINT DEFAULT 0;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS max_retries SMALLINT DEFAULT 0;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS pipeline_id TEXT DEFAULT NULL;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS pipeline_step SMALLINT DEFAULT NULL;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS triggered_by TEXT DEFAULT 'manual';

CREATE INDEX IF NOT EXISTS idx_job_history_cron ON job_history (scheduled_cron);
