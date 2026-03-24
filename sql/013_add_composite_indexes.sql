-- Feature 28: Composite indexes for faster accuracy and champion queries
-- These indexes support common query patterns in the accuracy slice,
-- lag curve, and champion selection endpoints.

-- Composite index for champion selection + common-DFU CTEs
CREATE INDEX IF NOT EXISTS idx_fact_forecast_dfu_triple
  ON fact_external_forecast_monthly (item_id, customer_group, loc);

-- Composite index for accuracy slice + lag-curve queries
CREATE INDEX IF NOT EXISTS idx_fact_forecast_model_lag
  ON fact_external_forecast_monthly (model_id, lag);

-- Composite index for backtest lag-curve common-DFU queries
CREATE INDEX IF NOT EXISTS idx_backtest_archive_dfu_triple
  ON backtest_lag_archive (item_id, customer_group, loc);

-- Composite index for backtest lag-curve by model+lag
CREATE INDEX IF NOT EXISTS idx_backtest_archive_model_lag
  ON backtest_lag_archive (model_id, lag);
