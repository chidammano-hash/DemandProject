-- Sequential job chain — runs N child jobs in order, halts on first failure.
--
-- Backs the IntegrationChainRunner service
-- (common/services/integration_chain_runner.py), which submits a list of
-- child loads (each backed by an integration_job row) and runs them serially
-- in a worker thread. Halts the chain at the first failed child and marks
-- any not-yet-started children as 'failed' with a "cancelled: chain halted"
-- error message.
--
-- Each child is a normal integration_job with chain_id + chain_step set so
-- the existing /integration/jobs UI continues to show them.
--
-- Lifecycle:
--   queued  -> row inserted by IntegrationChainRunner.submit_chain()
--   running -> updated when the worker thread picks the chain up
--   success -> every child reached a non-failed terminal state
--   halted  -> a child failed; remaining children were cancelled
--   failed  -> unexpected error in the chain worker itself
--
-- Indexes:
--   idx_integration_chain_started -- recent-chains UI query
--   idx_integration_job_chain     -- look up child jobs for a chain

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS integration_chain (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    status          TEXT         NOT NULL DEFAULT 'queued'
                                 CHECK (status IN ('queued','running','success','failed','halted')),
    total_steps     INTEGER      NOT NULL,
    completed_steps INTEGER      NOT NULL DEFAULT 0,
    failed_step     INTEGER,
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    duration_ms     INTEGER,
    triggered_by    TEXT
);

ALTER TABLE integration_job
    ADD COLUMN IF NOT EXISTS chain_id   UUID,
    ADD COLUMN IF NOT EXISTS chain_step INTEGER;

CREATE INDEX IF NOT EXISTS idx_integration_job_chain
    ON integration_job (chain_id, chain_step);

CREATE INDEX IF NOT EXISTS idx_integration_chain_started
    ON integration_chain (started_at DESC);

COMMENT ON TABLE integration_chain IS
    'Sequential ETL job chain ledger; child jobs live in integration_job.';
