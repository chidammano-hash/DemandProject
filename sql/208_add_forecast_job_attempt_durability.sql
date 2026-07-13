-- Durable forecasting job attempts.
--
-- execution_group preserves submit-time concurrency routing (including
-- group_override). attempt_token + attempt_result make child completion a
-- compare-and-set operation. Recovery leases deduplicate startup dispatch,
-- while a quarantine blocks the affected group until operator review.

ALTER TABLE job_history
    ADD COLUMN IF NOT EXISTS execution_group TEXT,
    ADD COLUMN IF NOT EXISTS attempt_token TEXT,
    ADD COLUMN IF NOT EXISTS attempt_result JSONB,
    ADD COLUMN IF NOT EXISTS attempt_failure_recorded BOOLEAN
        NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS recovery_lease_owner TEXT,
    ADD COLUMN IF NOT EXISTS recovery_lease_until TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS recovery_quarantine_reason TEXT;

-- Best-effort backfill for active forecasting rows created before submit-time
-- group persistence existed. New rows always store the exact override.
UPDATE job_history
SET execution_group = CASE
    WHEN job_type IN (
        'backtest_lgbm',
        'backtest_chronos2_enriched',
        'backtest_mstl',
        'backtest_nhits',
        'backtest_nbeats'
    ) THEN 'backtest'
    WHEN job_type IN (
        'champion_select',
        'champion_experiment',
        'champion_results_load',
        'champion_sweep'
    ) THEN 'champion'
    WHEN job_type IN (
        'train_production_model',
        'generate_production_forecast',
        'prepare_forecast_snapshot_contenders',
        'archive_forecast_snapshot',
        'cleanup_forecast_staging'
    ) THEN 'forecast'
    WHEN job_type = 'model_tuning_run'
        THEN 'tuning_' || COALESCE(params ->> 'model', 'lgbm')
    ELSE execution_group
END
WHERE execution_group IS NULL
  AND status IN ('queued', 'running')
  AND job_type IN (
      'backtest_lgbm',
      'backtest_chronos2_enriched',
      'backtest_mstl',
      'backtest_nhits',
      'backtest_nbeats',
      'champion_select',
      'champion_experiment',
      'champion_results_load',
      'champion_sweep',
      'model_tuning_run',
      'train_production_model',
      'generate_production_forecast',
      'prepare_forecast_snapshot_contenders',
      'archive_forecast_snapshot',
      'cleanup_forecast_staging'
  );

CREATE INDEX IF NOT EXISTS idx_job_history_recovery_lease
    ON job_history (status, recovery_lease_until)
    WHERE status = 'queued';

CREATE INDEX IF NOT EXISTS idx_job_history_execution_group_active
    ON job_history (execution_group, status)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS idx_job_history_recovery_quarantine
    ON job_history (execution_group)
    WHERE recovery_quarantine_reason IS NOT NULL;
