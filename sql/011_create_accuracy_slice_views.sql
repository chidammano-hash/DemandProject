-- 011_create_accuracy_slice_views.sql
-- Pre-aggregated accuracy views for fast multi-dimensional KPI slicing (feature16).
--
-- Instead of scanning fact_external_forecast_monthly + JOIN dim_sku at query time,
-- these views pre-join and pre-aggregate accuracy components at the DFU-attribute grain.
-- Query cost drops from O(millions) to O(dim cardinality) once the views are populated.
--
-- Refresh triggered automatically by load_backtest_forecasts.py at the end of backtest-load.
-- Also callable manually via: make accuracy-slice-refresh

-- ── agg_accuracy_by_dim ────────────────────────────────────────────────────
-- Grain: (model_id, lag, month_start, cluster_assignment, ml_cluster,
--          supplier_desc, abc_vol, region, brand_desc, dfu_execution_lag,
--          seasonality_profile)
-- Stores pre-summed forecast, actual, and abs-error so all KPIs can be derived
-- at the API layer with simple arithmetic: WAPE, Bias, Accuracy %.
-- Source: fact_external_forecast_monthly JOIN dim_sku LEFT JOIN promoted sku_cluster_assignment

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_accuracy_by_dim AS
SELECT
  f.model_id,
  f.lag,
  date_trunc('month', f.startdate)::date               AS month_start,
  COALESCE(d.cluster_assignment, '(unassigned)')       AS cluster_assignment,
  COALESCE(ca.cluster_label, '(unassigned)')           AS ml_cluster,
  COALESCE(d.supplier_desc, '(unknown)')               AS supplier_desc,
  COALESCE(d.abc_vol, '(unknown)')                     AS abc_vol,
  COALESCE(d.region, '(unknown)')                      AS region,
  COALESCE(d.brand_desc, '(unknown)')                  AS brand_desc,
  COALESCE(d.execution_lag::text, '(none)')            AS dfu_execution_lag,
  COALESCE(d.seasonality_profile, '(unknown)')         AS seasonality_profile,
  COUNT(*)::bigint                                      AS row_count,
  COALESCE(SUM(f.basefcst_pref), 0)::double precision  AS sum_forecast,
  COALESCE(SUM(f.tothist_dmd), 0)::double precision    AS sum_actual,
  COALESCE(SUM(ABS(f.basefcst_pref - f.tothist_dmd)), 0)::double precision AS sum_abs_error
FROM fact_external_forecast_monthly f
JOIN dim_sku d
  ON f.item_id = d.item_id
 AND f.customer_group = d.customer_group
 AND f.loc = d.loc
LEFT JOIN (
  SELECT a.sku_ck, a.cluster_label
  FROM sku_cluster_assignment a
  JOIN cluster_experiment e
    ON e.experiment_id = a.experiment_id
   AND e.is_promoted IS TRUE
) ca
  ON ca.sku_ck = d.sku_ck
WHERE f.tothist_dmd IS NOT NULL
  AND f.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_model_lag
  ON agg_accuracy_by_dim (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_cluster
  ON agg_accuracy_by_dim (cluster_assignment);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_supplier
  ON agg_accuracy_by_dim (supplier_desc);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_month
  ON agg_accuracy_by_dim (month_start);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_abc_vol
  ON agg_accuracy_by_dim (abc_vol);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_region
  ON agg_accuracy_by_dim (region);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_seasonality
  ON agg_accuracy_by_dim (seasonality_profile);


-- ── agg_accuracy_lag_archive ───────────────────────────────────────────────
-- Grain: (model_id, lag, timeframe, month_start, cluster_assignment,
--          ml_cluster, supplier_desc, abc_vol, region, brand_desc,
--          seasonality_profile)
-- Source: backtest_lag_archive JOIN dim_sku LEFT JOIN promoted sku_cluster_assignment
-- Used for lag-horizon accuracy curves (how accuracy degrades lag 0 -> lag 4).

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_accuracy_lag_archive AS
SELECT
  a.model_id,
  a.lag,
  COALESCE(a.timeframe, '(none)')                      AS timeframe,
  date_trunc('month', a.startdate)::date               AS month_start,
  COALESCE(d.cluster_assignment, '(unassigned)')       AS cluster_assignment,
  COALESCE(ca.cluster_label, '(unassigned)')           AS ml_cluster,
  COALESCE(d.supplier_desc, '(unknown)')               AS supplier_desc,
  COALESCE(d.abc_vol, '(unknown)')                     AS abc_vol,
  COALESCE(d.region, '(unknown)')                      AS region,
  COALESCE(d.brand_desc, '(unknown)')                  AS brand_desc,
  COALESCE(d.seasonality_profile, '(unknown)')         AS seasonality_profile,
  COUNT(*)::bigint                                      AS row_count,
  COALESCE(SUM(a.basefcst_pref), 0)::double precision  AS sum_forecast,
  COALESCE(SUM(a.tothist_dmd), 0)::double precision    AS sum_actual,
  COALESCE(SUM(ABS(a.basefcst_pref - a.tothist_dmd)), 0)::double precision AS sum_abs_error
FROM backtest_lag_archive a
JOIN dim_sku d
  ON a.item_id = d.item_id
 AND a.customer_group = d.customer_group
 AND a.loc = d.loc
LEFT JOIN (
  SELECT a.sku_ck, a.cluster_label
  FROM sku_cluster_assignment a
  JOIN cluster_experiment e
    ON e.experiment_id = a.experiment_id
   AND e.is_promoted IS TRUE
) ca
  ON ca.sku_ck = d.sku_ck
WHERE a.tothist_dmd IS NOT NULL
  AND a.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
WITH NO DATA;

CREATE INDEX IF NOT EXISTS idx_agg_accuracy_lag_archive_model_lag
  ON agg_accuracy_lag_archive (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_lag_archive_cluster
  ON agg_accuracy_lag_archive (cluster_assignment);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_lag_archive_month
  ON agg_accuracy_lag_archive (month_start);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_lag_archive_seasonality
  ON agg_accuracy_lag_archive (seasonality_profile);
