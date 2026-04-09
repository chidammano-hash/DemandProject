-- 126_add_ss_forecast_source.sql
-- Adds forecast_source and forecast_model_id columns to fact_safety_stock_targets.
-- Tracks whether safety stock demand stats came from historical dim_sku data,
-- promoted production forecast CI bands, or staging forecast CI bands.

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS forecast_source    VARCHAR(20) DEFAULT 'historical';  -- 'historical' | 'production' | 'staging'

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS forecast_model_id  VARCHAR(100);  -- model_id when forecast_source != 'historical'
