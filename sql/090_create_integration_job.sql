-- Integration job tracking for the unified ETL load runner.
--
-- Backs the IntegrationRunner service (common/services/integration_runner.py),
-- which spawns scripts/etl/load.py as a subprocess and records every load
-- attempt (one-time backfill, delta refresh, or single-file replay) here.
--
-- Lifecycle:
--   queued    -> row inserted by IntegrationRunner.submit()
--   running   -> updated when the worker thread picks the job up
--   success   -> subprocess exited 0 (rows_loaded parsed from final JSON line)
--   failed    -> subprocess exited non-zero / crashed / timed out
--   skipped   -> subprocess exited 2 (no work to do, e.g. delta with no new files)
--
-- Indexes:
--   idx_integration_job_started  -- recent-jobs UI query
--   idx_integration_job_domain   -- per-domain history query

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS integration_job (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    domain          TEXT         NOT NULL,
    mode            TEXT         NOT NULL CHECK (mode IN ('onetime','delta','file')),
    slice           TEXT,
    file_path       TEXT,
    status          TEXT         NOT NULL DEFAULT 'queued'
                                 CHECK (status IN ('queued','running','success','failed','skipped')),
    rows_loaded     INTEGER      NOT NULL DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    duration_ms     INTEGER,
    triggered_by    TEXT
);

CREATE INDEX IF NOT EXISTS idx_integration_job_started
    ON integration_job (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_integration_job_domain
    ON integration_job (domain, started_at DESC);

COMMENT ON TABLE integration_job IS
    'Async job ledger for the unified ETL load runner (scripts/etl/load.py).';
