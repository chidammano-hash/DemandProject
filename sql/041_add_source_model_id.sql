-- 041_add_source_model_id.sql
-- Idempotent migration: add source_model_id to fact_production_forecast.
-- Tracks which champion model generated each production forecast row.

ALTER TABLE fact_production_forecast ADD COLUMN IF NOT EXISTS source_model_id VARCHAR(100) DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_prod_fcst_source_model_id
    ON fact_production_forecast (source_model_id)
    WHERE source_model_id IS NOT NULL;
