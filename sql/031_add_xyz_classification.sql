-- 031_add_xyz_classification.sql
-- Idempotent migration: add ABC-XYZ classification columns to dim_sku.
-- Columns already exist in CREATE TABLE (005) for fresh installs.

ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS xyz_class TEXT;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS abc_xyz_segment TEXT;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS abc_xyz_policy_id TEXT;
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS abc_xyz_dos_min NUMERIC(10,2);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS abc_xyz_dos_max NUMERIC(10,2);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS abc_xyz_service_level NUMERIC(6,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS abc_xyz_classified_ts TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_dim_sku_xyz_class ON dim_sku (xyz_class);
CREATE INDEX IF NOT EXISTS idx_dim_sku_abc_xyz_segment ON dim_sku (abc_xyz_segment);
