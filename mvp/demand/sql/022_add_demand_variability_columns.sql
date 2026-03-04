-- IPfeature1: Demand Variability & Statistical Profiling Engine
-- Adds demand variability columns to dim_dfu.
-- Run once; idempotent (ADD COLUMN IF NOT EXISTS).

ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_mean           NUMERIC(15,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_std            NUMERIC(15,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_cv             NUMERIC(10,6);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_mad            NUMERIC(15,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_p50            NUMERIC(15,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_p90            NUMERIC(15,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_skewness       NUMERIC(10,6);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_kurtosis       NUMERIC(10,6);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS zero_demand_months    INTEGER;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS total_demand_months   INTEGER;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS intermittency_ratio   NUMERIC(10,6);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS variability_class     TEXT;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS demand_profile_ts     TIMESTAMPTZ;

-- Index on variability_class for filtering
CREATE INDEX IF NOT EXISTS idx_dim_dfu_variability_class
    ON dim_dfu (variability_class);
