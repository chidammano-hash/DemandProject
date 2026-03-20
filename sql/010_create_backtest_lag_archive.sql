-- Archive table for backtest predictions at all lags (0-4).
-- Main forecast table stores only execution-lag rows; this table
-- preserves all lags for accuracy reporting at any horizon.

CREATE TABLE IF NOT EXISTS backtest_lag_archive (
  archive_sk    BIGSERIAL PRIMARY KEY,
  forecast_ck   TEXT NOT NULL,
  dmdunit       TEXT NOT NULL,
  dmdgroup      TEXT NOT NULL,
  loc           TEXT NOT NULL,
  fcstdate      DATE NOT NULL,
  startdate     DATE NOT NULL,
  lag           INTEGER NOT NULL,
  execution_lag INTEGER,
  basefcst_pref NUMERIC(18,4),
  tothist_dmd   NUMERIC(18,4),
  model_id      TEXT NOT NULL,
  timeframe     TEXT,
  load_ts       TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT chk_backtest_lag_archive_lag_0_4 CHECK (lag BETWEEN 0 AND 4),
  CONSTRAINT chk_backtest_lag_archive_fcst_month_start CHECK (
    fcstdate = date_trunc('month', fcstdate)::date
  ),
  CONSTRAINT chk_backtest_lag_archive_start_month_start CHECK (
    startdate = date_trunc('month', startdate)::date
  ),
  CONSTRAINT chk_backtest_lag_archive_lag_matches_dates CHECK (
    ((EXTRACT(YEAR FROM startdate)::int - EXTRACT(YEAR FROM fcstdate)::int) * 12
      + (EXTRACT(MONTH FROM startdate)::int - EXTRACT(MONTH FROM fcstdate)::int)) = lag
  ),
  CONSTRAINT uq_backtest_lag_archive_ck UNIQUE (forecast_ck, model_id, lag)
);

CREATE INDEX IF NOT EXISTS idx_backtest_lag_archive_model_id ON backtest_lag_archive (model_id);
CREATE INDEX IF NOT EXISTS idx_backtest_lag_archive_dmdunit ON backtest_lag_archive (dmdunit);
CREATE INDEX IF NOT EXISTS idx_backtest_lag_archive_startdate ON backtest_lag_archive (startdate);
CREATE INDEX IF NOT EXISTS idx_backtest_lag_archive_lag ON backtest_lag_archive (lag);
