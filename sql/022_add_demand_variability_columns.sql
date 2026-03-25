-- 022_add_demand_variability_columns.sql
-- Idempotent migration: add demand variability columns to dim_sku.
-- Columns already exist in CREATE TABLE (005) for fresh installs.

ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_mean NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_std NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_cv NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_mad NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_p50 NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_p90 NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_skewness NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_kurtosis NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS zero_demand_months INTEGER;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS total_demand_months INTEGER;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS intermittency_ratio NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS variability_class TEXT;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS demand_profile_ts TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_dim_sku_variability_class ON dim_sku (variability_class);
