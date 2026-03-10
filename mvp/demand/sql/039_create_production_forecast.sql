-- F1.1: Production Forecast Generation Pipeline
-- Creates fact_production_forecast and fact_model_registry tables.
-- Run once per environment: make forecast-prod-schema

-- ---------------------------------------------------------------------------
-- fact_production_forecast
-- Stores forward-looking ML predictions only (future months with no actuals).
-- Historical backtest rows remain in fact_external_forecast_monthly.
-- Grain: plan_version + item_no + loc + forecast_month
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_production_forecast (
    id                  BIGSERIAL PRIMARY KEY,
    plan_version        VARCHAR(30)     NOT NULL,   -- e.g. '2026-03'
    item_no             VARCHAR(50)     NOT NULL,
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
    ON fact_production_forecast (plan_version, item_no, loc, forecast_month);

-- Lookup by DFU + month (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_prod_fcst_item_loc_month
    ON fact_production_forecast (item_no, loc, forecast_month);

-- Lookup by plan version (for version comparison queries)
CREATE INDEX IF NOT EXISTS idx_prod_fcst_plan_version
    ON fact_production_forecast (plan_version);

-- Lookup by run_id (for cleanup/promotion operations)
CREATE INDEX IF NOT EXISTS idx_prod_fcst_run_id
    ON fact_production_forecast (run_id);


-- ---------------------------------------------------------------------------
-- fact_model_registry
-- Tracks persisted .pkl model artifacts so the inference pipeline can load them.
-- Backtest scripts write here; generate_production_forecasts.py reads from here.
-- Grain: model_id + cluster_id (one active row per combo)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_model_registry (
    id               BIGSERIAL PRIMARY KEY,
    model_id         VARCHAR(100)    NOT NULL,   -- e.g. 'lgbm_cluster'
    cluster_id       TEXT            NOT NULL,   -- ml_cluster label (string)
    timeframe        VARCHAR(20)     NOT NULL,   -- backtest timeframe label ('J' = most recent)
    model_path       TEXT            NOT NULL,   -- relative path: 'data/models/lgbm_cluster/cluster_3.pkl'
    feature_cols     TEXT[],                     -- ordered feature column names used during training
    n_estimators     INTEGER,
    train_wape       NUMERIC(6, 4),
    trained_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    is_active        BOOLEAN         NOT NULL DEFAULT FALSE,
    promoted_at      TIMESTAMPTZ,
    promoted_by      VARCHAR(100)
);

-- Only one active model per (model_id, cluster_id)
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_active
    ON fact_model_registry (model_id, cluster_id)
    WHERE is_active = TRUE;

-- Latest model lookup
CREATE INDEX IF NOT EXISTS idx_model_registry_model_id
    ON fact_model_registry (model_id, cluster_id, trained_at DESC);
