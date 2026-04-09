-- 122_create_production_forecast_staging.sql
-- Production Forecast Staging — stores forward-looking forecasts from ALL algorithms.
-- Generate writes here (DELETE-before-INSERT per model_id); Promote copies to fact_production_forecast.
-- Multiple models coexist, keyed by (model_id, item_id, loc, forecast_month).

CREATE TABLE IF NOT EXISTS fact_production_forecast_staging (
    id                      BIGSERIAL PRIMARY KEY,
    model_id                VARCHAR(100)    NOT NULL,   -- algorithm that generated this forecast
    item_id                 VARCHAR(50)     NOT NULL,   -- DFU item identity
    loc                     VARCHAR(50)     NOT NULL,   -- DFU location identity
    forecast_month          DATE            NOT NULL,   -- predicted month (first day, always YYYY-MM-01)
    forecast_month_generated DATE           NOT NULL,   -- planning cycle month when forecast was created
    forecast_qty            NUMERIC(12, 2)  NOT NULL,   -- point forecast
    forecast_qty_lower      NUMERIC(12, 2),             -- P10 confidence interval lower bound
    forecast_qty_upper      NUMERIC(12, 2),             -- P90 confidence interval upper bound
    cluster_id              TEXT,                        -- ml_cluster label used during training
    horizon_months          SMALLINT        NOT NULL,   -- 1=T+1, 2=T+2, etc.
    is_recursive            BOOLEAN         NOT NULL DEFAULT FALSE,  -- TRUE if lag features from prior predictions
    lag_source              VARCHAR(20),                -- 'actual' (T+1) or 'predicted' (T+2+)
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),  -- when inference was run
    run_id                  UUID            NOT NULL    -- ties all rows from one inference run
);

-- One prediction per model per DFU per forecast_month
CREATE UNIQUE INDEX IF NOT EXISTS uq_staging_model_dfu_month
    ON fact_production_forecast_staging (model_id, item_id, loc, forecast_month);

-- Per-model queries and DELETE-before-INSERT
CREATE INDEX IF NOT EXISTS idx_staging_model_id
    ON fact_production_forecast_staging (model_id);

-- DFU lookup for cross-model comparison
CREATE INDEX IF NOT EXISTS idx_staging_item_loc
    ON fact_production_forecast_staging (item_id, loc);

-- Planning cycle lookup
CREATE INDEX IF NOT EXISTS idx_staging_month_generated
    ON fact_production_forecast_staging (forecast_month_generated);
