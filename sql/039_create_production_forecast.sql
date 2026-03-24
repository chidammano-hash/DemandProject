-- F1.1: Production Forecast Generation Pipeline
-- Creates fact_production_forecast table.
-- Run once per environment: make forecast-prod-schema

-- ---------------------------------------------------------------------------
-- fact_production_forecast
-- Stores forward-looking ML predictions only (future months with no actuals).
-- Historical backtest rows remain in fact_external_forecast_monthly.
-- Grain: plan_version + item_id + loc + forecast_month
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_production_forecast (
    id                  BIGSERIAL PRIMARY KEY,
    plan_version        VARCHAR(30)     NOT NULL,   -- e.g. '2026-03'
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    forecast_month      DATE            NOT NULL,   -- first day of month (always future)
    forecast_qty        NUMERIC(12, 2)  NOT NULL,
    forecast_qty_lower  NUMERIC(12, 2),             -- P10 confidence interval (optional)
    forecast_qty_upper  NUMERIC(12, 2),             -- P90 confidence interval (optional)
    model_id            VARCHAR(100)    NOT NULL,   -- e.g. 'lgbm_cluster'
    cluster_id          TEXT,                       -- ml_cluster used for inference (string label)
    horizon_months      SMALLINT        NOT NULL,   -- 1=T+1, 2=T+2, ... 12=T+12
    is_recursive        BOOLEAN         NOT NULL DEFAULT FALSE,
    lag_source          VARCHAR(20),                -- 'actual' (T+1) or 'predicted' (T+2+)
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    run_id              UUID            NOT NULL    -- ties all rows from one inference run
);

-- Unique: one forecast per (plan_version, DFU, forecast_month)
CREATE UNIQUE INDEX IF NOT EXISTS uq_prod_fcst_version_dfu_month
    ON fact_production_forecast (plan_version, item_id, loc, forecast_month);

-- Lookup by DFU + month (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_prod_fcst_item_loc_month
    ON fact_production_forecast (item_id, loc, forecast_month);

-- Lookup by plan version (for version comparison queries)
CREATE INDEX IF NOT EXISTS idx_prod_fcst_plan_version
    ON fact_production_forecast (plan_version);

-- Lookup by run_id (for cleanup/promotion operations)
CREATE INDEX IF NOT EXISTS idx_prod_fcst_run_id
    ON fact_production_forecast (run_id);
