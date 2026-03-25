-- Feature 47: Dual Promotion — add results promotion tracking
-- Adds columns to lgbm_tuning_run for tracking whether backtest predictions
-- have been loaded into fact_external_forecast_monthly + backtest_lag_archive.
-- Also adds promotion_type to the audit log to distinguish params vs results.

ALTER TABLE lgbm_tuning_run
    ADD COLUMN IF NOT EXISTS is_results_promoted BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS results_promoted_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS results_promote_job_id VARCHAR(100);

ALTER TABLE tuning_promotion_log
    ADD COLUMN IF NOT EXISTS promotion_type VARCHAR(20) NOT NULL DEFAULT 'params';
