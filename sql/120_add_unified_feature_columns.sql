-- Migration 120: Add unified SKU feature columns to dim_sku
-- These columns persist computed demand features that were previously
-- only ephemeral in the clustering pipeline, enabling reuse across
-- backtest, champion selection, expert panel, and inventory planning.

-- Volume features (from clustering, not yet persisted)
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS iqr_demand          NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS median_demand       NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS min_demand          NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS max_demand          NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS total_demand        NUMERIC(15,4);

-- Trend features
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS trend_slope         NUMERIC(15,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS trend_slope_norm    NUMERIC(15,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS trend_r2            NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS trend_pct_change    NUMERIC(15,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS trend_direction     SMALLINT;

-- Seasonality features (beyond existing seasonality_strength, peak/trough)
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS seasonal_amplitude  NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS seasonal_r2         NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS yoy_correlation     NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS seasonal_index_std  NUMERIC(15,4);

-- Periodicity
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS periodicity_strength NUMERIC(10,6);

-- Intermittency (Croston ADI)
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS adi                 NUMERIC(10,4);

-- Lifecycle
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS cagr                NUMERIC(10,4);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS recency_ratio       NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS acceleration        NUMERIC(10,6);
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS outlier_count       INTEGER;

-- Seasonality extra
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS acf_lag12           NUMERIC(10,6);

-- Computation timestamp
ALTER TABLE dim_sku ADD COLUMN IF NOT EXISTS features_computed_ts TIMESTAMPTZ;
