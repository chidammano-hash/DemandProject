-- 016_add_seasonality_to_accuracy_views.sql
-- Add seasonality_profile as a dimension to all 4 accuracy materialized views.
-- Requires DROP + CREATE because adding a column to a materialized view is not supported.
-- After applying this DDL, run: REFRESH MATERIALIZED VIEW <view> for each view.

-- ══════════════════════════════════════════════════════════════════════════════
-- Drop existing views (reverse dependency order)
-- ══════════════════════════════════════════════════════════════════════════════
DROP MATERIALIZED VIEW IF EXISTS agg_dfu_coverage_lag_archive CASCADE;
DROP MATERIALIZED VIEW IF EXISTS agg_dfu_coverage CASCADE;
DROP MATERIALIZED VIEW IF EXISTS agg_accuracy_lag_archive CASCADE;
DROP MATERIALIZED VIEW IF EXISTS agg_accuracy_by_dim CASCADE;

-- ══════════════════════════════════════════════════════════════════════════════
-- Recreate agg_accuracy_by_dim (with seasonality_profile)
-- ══════════════════════════════════════════════════════════════════════════════
-- Grain: (model_id, lag, month_start, cluster_assignment, ml_cluster,
--          supplier_desc, abc_vol, region, brand_desc, dfu_execution_lag,
--          seasonality_profile)

CREATE MATERIALIZED VIEW agg_accuracy_by_dim AS
SELECT
  f.model_id,
  f.lag,
  date_trunc('month', f.startdate)::date               AS month_start,
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
  COALESCE(SUM(ABS(f.basefcst_pref - f.tothist_dmd)), 0)::double precision AS sum_abs_error
FROM fact_external_forecast_monthly f
JOIN dim_dfu d
  ON f.dmdunit = d.dmdunit
 AND f.dmdgroup = d.dmdgroup
 AND f.loc = d.loc
WHERE f.tothist_dmd IS NOT NULL
  AND f.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
WITH NO DATA;

CREATE INDEX idx_agg_accuracy_by_dim_model_lag
  ON agg_accuracy_by_dim (model_id, lag);
CREATE INDEX idx_agg_accuracy_by_dim_cluster
  ON agg_accuracy_by_dim (cluster_assignment);
CREATE INDEX idx_agg_accuracy_by_dim_supplier
  ON agg_accuracy_by_dim (supplier_desc);
CREATE INDEX idx_agg_accuracy_by_dim_month
  ON agg_accuracy_by_dim (month_start);
CREATE INDEX idx_agg_accuracy_by_dim_abc_vol
  ON agg_accuracy_by_dim (abc_vol);
CREATE INDEX idx_agg_accuracy_by_dim_region
  ON agg_accuracy_by_dim (region);
CREATE INDEX idx_agg_accuracy_by_dim_seasonality
  ON agg_accuracy_by_dim (seasonality_profile);

-- ══════════════════════════════════════════════════════════════════════════════
-- Recreate agg_accuracy_lag_archive (with seasonality_profile)
-- ══════════════════════════════════════════════════════════════════════════════
-- Grain: (model_id, lag, timeframe, month_start, cluster_assignment,
--          ml_cluster, supplier_desc, abc_vol, region, brand_desc,
--          seasonality_profile)

CREATE MATERIALIZED VIEW agg_accuracy_lag_archive AS
SELECT
  a.model_id,
  a.lag,
  COALESCE(a.timeframe, '(none)')                      AS timeframe,
  date_trunc('month', a.startdate)::date               AS month_start,
  COALESCE(d.cluster_assignment, '(unassigned)')       AS cluster_assignment,
  COALESCE(d.ml_cluster, '(unassigned)')               AS ml_cluster,
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
JOIN dim_dfu d
  ON a.dmdunit = d.dmdunit
 AND a.dmdgroup = d.dmdgroup
 AND a.loc = d.loc
WHERE a.tothist_dmd IS NOT NULL
  AND a.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
WITH NO DATA;

CREATE INDEX idx_agg_accuracy_lag_archive_model_lag
  ON agg_accuracy_lag_archive (model_id, lag);
CREATE INDEX idx_agg_accuracy_lag_archive_cluster
  ON agg_accuracy_lag_archive (cluster_assignment);
CREATE INDEX idx_agg_accuracy_lag_archive_month
  ON agg_accuracy_lag_archive (month_start);
CREATE INDEX idx_agg_accuracy_lag_archive_seasonality
  ON agg_accuracy_lag_archive (seasonality_profile);

-- ══════════════════════════════════════════════════════════════════════════════
-- Recreate agg_dfu_coverage (with seasonality_profile)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE MATERIALIZED VIEW agg_dfu_coverage AS
SELECT
  f.model_id,
  f.lag,
  f.dmdunit,
  f.dmdgroup,
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
JOIN dim_dfu d
  ON f.dmdunit = d.dmdunit
 AND f.dmdgroup = d.dmdgroup
 AND f.loc = d.loc
WHERE f.tothist_dmd IS NOT NULL
  AND f.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
WITH NO DATA;

CREATE INDEX idx_agg_dfu_coverage_model_lag
  ON agg_dfu_coverage (model_id, lag);
CREATE INDEX idx_agg_dfu_coverage_cluster
  ON agg_dfu_coverage (cluster_assignment);
CREATE INDEX idx_agg_dfu_coverage_execution_lag
  ON agg_dfu_coverage (dfu_execution_lag);
CREATE INDEX idx_agg_dfu_coverage_exec_lag
  ON agg_dfu_coverage (cluster_assignment, model_id) WHERE lag::text = dfu_execution_lag;
CREATE INDEX idx_agg_dfu_coverage_seasonality
  ON agg_dfu_coverage (seasonality_profile);

-- ══════════════════════════════════════════════════════════════════════════════
-- Recreate agg_dfu_coverage_lag_archive (with seasonality_profile)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE MATERIALIZED VIEW agg_dfu_coverage_lag_archive AS
SELECT
  a.model_id,
  a.lag,
  a.dmdunit,
  a.dmdgroup,
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
JOIN dim_dfu d
  ON a.dmdunit = d.dmdunit
 AND a.dmdgroup = d.dmdgroup
 AND a.loc = d.loc
WHERE a.tothist_dmd IS NOT NULL
  AND a.basefcst_pref IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
WITH NO DATA;

CREATE INDEX idx_agg_dfu_coverage_la_model_lag
  ON agg_dfu_coverage_lag_archive (model_id, lag);
CREATE INDEX idx_agg_dfu_coverage_la_cluster
  ON agg_dfu_coverage_lag_archive (cluster_assignment);
CREATE INDEX idx_agg_dfu_coverage_la_seasonality
  ON agg_dfu_coverage_lag_archive (seasonality_profile);
