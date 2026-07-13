-- 098: Add promote-to-production support for tuning runs
-- Allows exactly one promoted run per LightGBM tuning model ID.

ALTER TABLE lgbm_tuning_run
    ADD COLUMN IF NOT EXISTS is_promoted BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS promoted_at TIMESTAMPTZ;

-- Drop the old global unique index (only allowed 1 promoted run across ALL models)
DROP INDEX IF EXISTS idx_tuning_run_promoted;

-- Partial unique index: at most one promoted run PER model_id
CREATE UNIQUE INDEX IF NOT EXISTS idx_tuning_run_promoted_per_model
    ON lgbm_tuning_run (model_id) WHERE is_promoted = TRUE;
