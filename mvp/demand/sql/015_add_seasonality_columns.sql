-- Feature 30: DFU Seasonality Detection
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS seasonality_profile TEXT;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS seasonality_strength NUMERIC(10,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS is_yearly_seasonal BOOLEAN;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS peak_month INTEGER;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS trough_month INTEGER;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS peak_trough_ratio NUMERIC(10,4);

CREATE INDEX IF NOT EXISTS idx_dim_dfu_seasonality_profile
    ON dim_dfu (seasonality_profile);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_is_yearly_seasonal
    ON dim_dfu (is_yearly_seasonal);

COMMENT ON COLUMN dim_dfu.seasonality_profile IS 'none | low | medium | high | insufficient_history';
COMMENT ON COLUMN dim_dfu.seasonality_strength IS 'CV of monthly means (0=flat, >1=strongly seasonal)';
COMMENT ON COLUMN dim_dfu.is_yearly_seasonal IS 'TRUE if confirmed 12-month cycle';
COMMENT ON COLUMN dim_dfu.peak_month IS 'Month with highest avg demand (1=Jan, 12=Dec)';
COMMENT ON COLUMN dim_dfu.trough_month IS 'Month with lowest avg demand (1=Jan, 12=Dec)';
COMMENT ON COLUMN dim_dfu.peak_trough_ratio IS 'Peak month avg / trough month avg';
