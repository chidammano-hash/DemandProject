-- IPfeature11: ABC-XYZ Policy Matrix & Portfolio Segmentation
-- Adds XYZ classification columns to dim_dfu.
-- XYZ = demand variability class (X=low, Y=moderate, Z=high/lumpy)
-- abc_xyz_segment = ABC class + XYZ class (e.g. 'AX', 'BZ', 'CY')

ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS xyz_class             TEXT;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_segment       TEXT;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_policy_id     TEXT;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_dos_min       NUMERIC(10,2);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_dos_max       NUMERIC(10,2);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_service_level NUMERIC(6,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_classified_ts TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_dim_dfu_xyz     ON dim_dfu (xyz_class);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_abc_xyz ON dim_dfu (abc_xyz_segment);
