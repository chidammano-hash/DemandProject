-- 193_create_accuracy_by_dfu_view.sql
-- DFU-grain accuracy aggregate for the forecast-accuracy DIAGNOSTIC layer.
--
-- The existing agg_accuracy_by_dim (sql/011) sums forecast/actual/abs-error across
-- ALL DFUs inside each dimension bucket, so it can only ever yield a *volume-weighted*
-- WAPE — the headline 72%. It cannot express:
--   • the UNWEIGHTED per-DFU mean/median accuracy (compute WAPE per DFU, then average),
--   • per-DFU ERROR CONTRIBUTION (which DFUs own most of the absolute error / Pareto).
-- Both require the individual DFU to survive aggregation, which it does not in 011.
--
-- This MV mirrors agg_dfu_coverage (sql/012) — same (model_id, lag, item_id,
-- customer_group, loc) grain and the same dim attributes — but carries the summed
-- accuracy components instead of just min/max month. One row per DFU x model x lag,
-- aggregated over all that DFU's months. From it BOTH metrics derive:
--   volume-weighted = SUM over bucket, then one WAPE      (matches agg_accuracy_by_dim)
--   unweighted      = per-DFU WAPE first, then mean/median across the bucket's DFUs
--   contribution    = bucket SUM(abs_error) / grand-total SUM(abs_error)
--
-- Refreshed alongside agg_accuracy_by_dim / agg_dfu_coverage after backtest-load
-- (load_backtest_forecasts.py) and via `make refresh-accuracy-mvs` / `accuracy-slice-refresh`.

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_accuracy_by_dfu AS
SELECT
  f.model_id,
  f.lag,
  f.item_id,
  f.customer_group,
  f.loc,
  COALESCE(d.cluster_assignment, '(unassigned)')       AS cluster_assignment,
  COALESCE(d.ml_cluster, '(unassigned)')               AS ml_cluster,
  COALESCE(d.supplier_desc, '(unknown)')               AS supplier_desc,
  COALESCE(d.abc_vol, '(unknown)')                     AS abc_vol,
  COALESCE(d.region, '(unknown)')                      AS region,
  COALESCE(d.brand_desc, '(unknown)')                  AS brand_desc,
  COALESCE(d.execution_lag::text, '(none)')            AS dfu_execution_lag,
  COALESCE(d.seasonality_profile, '(unknown)')         AS seasonality_profile,
  COUNT(*)::bigint                                      AS row_count,
  COALESCE(SUM(f.basefcst_pref), 0)::double precision  AS sum_forecast,
  COALESCE(SUM(f.tothist_dmd), 0)::double precision    AS sum_actual,
  COALESCE(SUM(ABS(f.basefcst_pref - f.tothist_dmd)), 0)::double precision AS sum_abs_error,
  MIN(date_trunc('month', f.startdate)::date)          AS min_month,
  MAX(date_trunc('month', f.startdate)::date)          AS max_month
FROM fact_external_forecast_monthly f
JOIN dim_sku d
  ON f.item_id = d.item_id
 AND f.customer_group = d.customer_group
 AND f.loc = d.loc
WHERE f.tothist_dmd IS NOT NULL
  AND f.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
WITH NO DATA;

-- Unique index backs REFRESH MATERIALIZED VIEW CONCURRENTLY. The dim columns are
-- functionally determined by the DFU (via the dim_sku join), so (model_id, lag,
-- item_id, customer_group, loc) is unique — same key as uq_dfu_coverage_model_lag_dfu.
-- Plain (non-CONCURRENTLY) build is safe here because the MV is created WITH NO DATA.
CREATE UNIQUE INDEX IF NOT EXISTS uq_agg_accuracy_by_dfu
  ON agg_accuracy_by_dfu (model_id, lag, item_id, customer_group, loc);

CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_model_lag
  ON agg_accuracy_by_dfu (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_cluster
  ON agg_accuracy_by_dfu (cluster_assignment);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_abc_vol
  ON agg_accuracy_by_dfu (abc_vol);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_region
  ON agg_accuracy_by_dfu (region);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_seasonality
  ON agg_accuracy_by_dfu (seasonality_profile);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_execution_lag
  ON agg_accuracy_by_dfu (dfu_execution_lag);
-- Partial index covering the default "execution lag (per DFU)" filter path.
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_exec_lag
  ON agg_accuracy_by_dfu (model_id) WHERE lag::text = dfu_execution_lag;
