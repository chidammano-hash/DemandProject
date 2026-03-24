-- Resilient Jobs: Add PID tracking and persistent execution log to job_history.
-- PID enables real kill (os.killpg) and startup recovery (re-adopt live processes).
-- log column stores streaming subprocess output for the Jobs UI.

ALTER TABLE job_history ADD COLUMN IF NOT EXISTS pid INTEGER;
ALTER TABLE job_history ADD COLUMN IF NOT EXISTS log TEXT DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_job_history_pid ON job_history (pid) WHERE pid IS NOT NULL;
