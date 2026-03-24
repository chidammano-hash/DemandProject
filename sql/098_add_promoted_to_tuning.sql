-- 098: Add promote-to-production support for LGBM tuning runs
-- Allows exactly one run to be marked as the "production" configuration.

ALTER TABLE lgbm_tuning_run
    ADD COLUMN IF NOT EXISTS is_promoted BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS promoted_at TIMESTAMPTZ;

-- Partial unique index: at most one promoted run at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_tuning_run_promoted
    ON lgbm_tuning_run (is_promoted) WHERE is_promoted = TRUE;
