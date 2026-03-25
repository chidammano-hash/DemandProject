-- 015_add_seasonality_columns.sql
-- Adds seasonality columns to dim_sku (idempotent).
-- These columns are already in the CREATE TABLE (005), so this is a no-op
-- for fresh installs and a safe migration for older schemas.

ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS seasonality_profile TEXT;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS seasonality_strength NUMERIC(10,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS is_yearly_seasonal BOOLEAN;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS peak_month INTEGER;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS trough_month INTEGER;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS peak_trough_ratio NUMERIC(10,4);

CREATE INDEX IF NOT EXISTS idx_dim_sku_seasonality_profile ON dim_sku (seasonality_profile);
