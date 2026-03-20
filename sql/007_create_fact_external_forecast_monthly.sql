CREATE TABLE IF NOT EXISTS fact_external_forecast_monthly (
  forecast_sk BIGSERIAL PRIMARY KEY,
  forecast_ck TEXT NOT NULL,
  dmdunit TEXT NOT NULL,
  dmdgroup TEXT NOT NULL,
  loc TEXT NOT NULL,
  fcstdate DATE NOT NULL,
  startdate DATE NOT NULL,
  lag INTEGER NOT NULL,
  execution_lag INTEGER,
  basefcst_pref NUMERIC(18,4),
  tothist_dmd NUMERIC(18,4),
  model_id TEXT NOT NULL DEFAULT 'external',
  load_ts TIMESTAMPTZ DEFAULT NOW(),
  modified_ts TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT chk_fact_external_forecast_monthly_lag_0_4 CHECK (lag BETWEEN 0 AND 4),
  CONSTRAINT chk_fact_external_forecast_monthly_fcst_month_start CHECK (
    fcstdate = date_trunc('month', fcstdate)::date
  ),
  CONSTRAINT chk_fact_external_forecast_monthly_start_month_start CHECK (
    startdate = date_trunc('month', startdate)::date
  ),
  CONSTRAINT chk_fact_external_forecast_monthly_lag_matches_dates CHECK (
    ((EXTRACT(YEAR FROM startdate)::int - EXTRACT(YEAR FROM fcstdate)::int) * 12
      + (EXTRACT(MONTH FROM startdate)::int - EXTRACT(MONTH FROM fcstdate)::int)) = lag
  ),
  CONSTRAINT uq_forecast_ck_model UNIQUE (forecast_ck, model_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_item ON fact_external_forecast_monthly (dmdunit);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_loc ON fact_external_forecast_monthly (loc);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_fcstdate ON fact_external_forecast_monthly (fcstdate);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_startdate ON fact_external_forecast_monthly (startdate);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_lag ON fact_external_forecast_monthly (lag);
CREATE INDEX IF NOT EXISTS idx_fact_external_forecast_monthly_model_id ON fact_external_forecast_monthly (model_id);
