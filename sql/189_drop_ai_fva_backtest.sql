-- 189_drop_ai_fva_backtest.sql
-- Remove the (deleted) AI Planner FVA Backtest store. The feature was replaced
-- by the forward-only AI Champion adjuster (sql/190, spec 02-27). Idempotent:
-- safe on databases that never had these objects and on a re-run.
--
-- These objects were created by the now-deleted sql/186_create_ai_fva_backtest.sql.

DROP MATERIALIZED VIEW IF EXISTS mv_ai_fva_by_dfu CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_ai_fva_by_month CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_ai_fva_by_recommendation CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_ai_fva_overall CASCADE;

DROP TABLE IF EXISTS ai_planner_backtest_audit CASCADE;
DROP TABLE IF EXISTS fact_ai_adjusted_forecast CASCADE;
DROP TABLE IF EXISTS fact_ai_forecast_recommendation CASCADE;
DROP TABLE IF EXISTS ai_fva_backtest_run CASCADE;
