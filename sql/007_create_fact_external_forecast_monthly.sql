CREATE TABLE IF NOT EXISTS fact_external_forecast_monthly (
  forecast_sk BIGSERIAL PRIMARY KEY,
  forecast_ck TEXT NOT NULL,
  item_id TEXT NOT NULL,
  customer_group TEXT NOT NULL,
  loc TEXT NOT NULL,
  fcstdate DATE NOT NULL,
  startdate DATE NOT NULL,
  lag INTEGER NOT NULL,
  execution_lag INTEGER,
  basefcst_pref NUMERIC(18,4),
  tothist_dmd NUMERIC(18,4),
  model_id TEXT NOT NULL DEFAULT 'external',
  source_model_id VARCHAR(100) DEFAULT NULL,
  load_ts TIMESTAMPTZ DEFAULT NOW(),
  modified_ts TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT chk_fact_external_forecast_monthly_lag_0_4 CHECK (lag BETWEEN 0 AND 4),
  CONSTRAINT chk_fact_external_forecast_monthly_fcst_month_start CHECK (
    fcstdate = date_trunc('month', fcstdate)::date
  ),
  CONSTRAINT chk_fact_external_forecast_monthly_start_month_start CHECK (
    startdate = date_trunc('month', startdate)::date
  ),
  -- lag represents execution_lag from dim_sku (not date difference) for external forecasts
  CONSTRAINT uq_forecast_ck_model UNIQUE (forecast_ck, model_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_item ON fact_external_forecast_monthly (item_id);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_loc ON fact_external_forecast_monthly (loc);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_fcstdate ON fact_external_forecast_monthly (fcstdate);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_startdate ON fact_external_forecast_monthly (startdate);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_lag ON fact_external_forecast_monthly (lag);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_model_id ON fact_external_forecast_monthly (model_id);

-- Index for production forecast: DFU -> champion row -> source_model_id
CREATE INDEX IF NOT EXISTS idx_femf_source_model
    ON fact_external_forecast_monthly (item_id, loc, source_model_id)
    WHERE source_model_id IS NOT NULL;
