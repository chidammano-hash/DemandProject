-- 012_create_dfu_coverage_view.sql
-- Pre-materialized distinct DFU coverage per model per lag.
--
-- Grain: (model_id, lag, item_id, customer_group, loc)
-- One row per unique DFU x model x lag combination.
-- Querying COUNT(*) from this view = COUNT(DISTINCT DFU) from the raw table,
-- but without an expensive full-table scan + DISTINCT at query time.
--
-- Refreshed alongside agg_accuracy_by_dim after backtest-load / champion-select.

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_dfu_coverage AS
SELECT
  f.model_id,
  f.lag,
  f.item_id,
  f.customer_group,
  f.loc,
  COALESCE(d.cluster_assignment, '(unassigned)')  AS cluster_assignment,
  COALESCE(d.ml_cluster, '(unassigned)')           AS ml_cluster,
  COALESCE(d.supplier_desc, '(unknown)')           AS supplier_desc,
  COALESCE(d.abc_vol, '(unknown)')                 AS abc_vol,
  COALESCE(d.region, '(unknown)')                  AS region,
  COALESCE(d.brand_desc, '(unknown)')              AS brand_desc,
  COALESCE(d.execution_lag::text, '(none)')        AS dfu_execution_lag,
  COALESCE(d.seasonality_profile, '(unknown)')     AS seasonality_profile,
  MIN(date_trunc('month', f.startdate)::date)      AS min_month,
  MAX(date_trunc('month', f.startdate)::date)      AS max_month
FROM fact_external_forecast_monthly f
JOIN dim_sku d
  ON f.item_id = d.item_id
 AND f.customer_group = d.customer_group
 AND f.loc = d.loc
WHERE f.tothist_dmd IS NOT NULL
  AND f.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_model_lag
  ON agg_dfu_coverage (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_cluster
  ON agg_dfu_coverage (cluster_assignment);
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_execution_lag
  ON agg_dfu_coverage (dfu_execution_lag);
-- Partial index: covers the default "Execution Lag (per DFU)" filter
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_exec_lag
  ON agg_dfu_coverage (cluster_assignment, model_id) WHERE lag::text = dfu_execution_lag;
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_seasonality
  ON agg_dfu_coverage (seasonality_profile);


-- Same view for backtest_lag_archive (used by lag-curve endpoint)

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_dfu_coverage_lag_archive AS
SELECT
  a.model_id,
  a.lag,
  a.item_id,
  a.customer_group,
  a.loc,
  COALESCE(d.cluster_assignment, '(unassigned)')  AS cluster_assignment,
  COALESCE(d.ml_cluster, '(unassigned)')           AS ml_cluster,
  COALESCE(d.supplier_desc, '(unknown)')           AS supplier_desc,
  COALESCE(d.abc_vol, '(unknown)')                 AS abc_vol,
  COALESCE(d.region, '(unknown)')                  AS region,
  COALESCE(d.brand_desc, '(unknown)')              AS brand_desc,
  COALESCE(d.execution_lag::text, '(none)')        AS dfu_execution_lag,
  COALESCE(d.seasonality_profile, '(unknown)')     AS seasonality_profile,
  MIN(date_trunc('month', a.startdate)::date)      AS min_month,
  MAX(date_trunc('month', a.startdate)::date)      AS max_month
FROM backtest_lag_archive a
JOIN dim_sku d
  ON a.item_id = d.item_id
 AND a.customer_group = d.customer_group
 AND a.loc = d.loc
WHERE a.tothist_dmd IS NOT NULL
  AND a.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_la_model_lag
  ON agg_dfu_coverage_lag_archive (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_la_cluster
  ON agg_dfu_coverage_lag_archive (cluster_assignment);
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_la_seasonality
  ON agg_dfu_coverage_lag_archive (seasonality_profile);
