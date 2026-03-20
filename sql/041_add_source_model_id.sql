-- F1.1 enhancement: track which underlying model won champion selection per DFU.
-- This allows production forecast to load the correct model artifacts per DFU.
-- Non-breaking: nullable, no constraints changed, existing rows get NULL.

ALTER TABLE fact_external_forecast_monthly
    ADD COLUMN IF NOT EXISTS source_model_id VARCHAR(100) DEFAULT NULL;

-- Index for production forecast query: DFU -> champion row -> source_model_id
CREATE INDEX IF NOT EXISTS idx_femf_source_model
    ON fact_external_forecast_monthly (dmdunit, loc, source_model_id)
    WHERE source_model_id IS NOT NULL;

COMMENT ON COLUMN fact_external_forecast_monthly.source_model_id IS
    'For model_id=champion rows: the underlying algorithm that was selected (e.g. lgbm_cluster). '
    'Used by production forecast to route each DFU to the correct trained model artifact.';
