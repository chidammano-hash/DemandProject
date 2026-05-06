-- 183_create_pg_queue.sql
--
-- Postgres-backed job queue table for the pg-queue subsystem
-- (Item 22 — pilot pg-queue alongside APScheduler).
--
-- This table powers a `FOR UPDATE SKIP LOCKED` claim pattern that lets
-- multiple workers safely race for the next pending job without blocking
-- one another. The pilot migrates ONE long-running recurring job
-- (`refresh_intramonth`) onto this queue; remaining APScheduler jobs are
-- untouched. See `common/services/pg_queue.py` and
-- `scripts/ops/pg_queue_worker.py`.
--
-- Status transitions:
--   pending  → claimed   (worker picks the row up via SKIP LOCKED)
--   claimed  → running   (worker calls mark_running before executing)
--   running  → completed (mark_completed with result JSONB)
--   running  → failed    (mark_failed with last_error; may re-enqueue
--                         if attempts < max_attempts)
--
-- Indexes are tuned for the two hot query paths:
--   1. Claim:  WHERE status='pending' AND run_at <= NOW() ORDER BY priority, run_at
--   2. Status: WHERE job_type=? AND status=? for monitoring + dead-letter scans

CREATE TABLE IF NOT EXISTS job_queue (
    id              BIGSERIAL PRIMARY KEY,
    job_type        TEXT NOT NULL,
    params          JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending|claimed|running|completed|failed
    priority        INT NOT NULL DEFAULT 100,         -- lower = higher priority
    run_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_by      TEXT,
    claimed_at      TIMESTAMPTZ,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    attempts        INT NOT NULL DEFAULT 0,
    max_attempts    INT NOT NULL DEFAULT 3,
    last_error      TEXT,
    result          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: workers scanning for pending+due jobs. Partial index keeps it tight.
CREATE INDEX IF NOT EXISTS idx_job_queue_pending
    ON job_queue (status, run_at)
    WHERE status IN ('pending', 'claimed');

-- Monitoring: count by type+status, dead-letter listings.
CREATE INDEX IF NOT EXISTS idx_job_queue_type_status
    ON job_queue (job_type, status);

COMMENT ON TABLE job_queue IS
    'Postgres-backed job queue. Workers claim rows via FOR UPDATE SKIP LOCKED. '
    'See common/services/pg_queue.py for the API.';
