-- 200: Durable SKU cluster assignments.
--
-- ML clustering output is computed lifecycle state, not source-loaded SKU
-- master data. It must survive dim_sku reloads from dfu.txt, so the promoted
-- per-SKU labels live here instead of on dim_sku.

CREATE TABLE IF NOT EXISTS sku_cluster_assignment (
    experiment_id       INTEGER NOT NULL
                            REFERENCES cluster_experiment(experiment_id)
                            ON DELETE CASCADE,
    sku_ck              TEXT NOT NULL,
    item_id             TEXT NOT NULL,
    customer_group      TEXT,
    loc                 TEXT NOT NULL,
    cluster_id          TEXT,
    cluster_label       TEXT NOT NULL,
    assigned_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (experiment_id, sku_ck)
);

CREATE INDEX IF NOT EXISTS idx_sku_cluster_assignment_sku
    ON sku_cluster_assignment (sku_ck);

CREATE INDEX IF NOT EXISTS idx_sku_cluster_assignment_label
    ON sku_cluster_assignment (cluster_label);

CREATE INDEX IF NOT EXISTS idx_sku_cluster_assignment_exp_label
    ON sku_cluster_assignment (experiment_id, cluster_label);

COMMENT ON TABLE sku_cluster_assignment IS
    'Durable per-SKU ML cluster assignments keyed by clustering experiment. '
    'This is the source of truth for promoted ML clusters.';

COMMENT ON COLUMN sku_cluster_assignment.sku_ck IS
    'Full SKU grain key (item_id, customer_group, loc) copied at assignment time.';

DROP VIEW IF EXISTS current_sku_cluster_assignment;

CREATE VIEW current_sku_cluster_assignment AS
SELECT
    a.experiment_id,
    e.scenario_id,
    a.sku_ck,
    a.item_id,
    a.customer_group,
    a.loc,
    a.cluster_id,
    a.cluster_label,
    a.cluster_label AS ml_cluster,
    a.assigned_at,
    a.modified_ts
FROM sku_cluster_assignment a
JOIN cluster_experiment e
  ON e.experiment_id = a.experiment_id
WHERE e.is_promoted IS TRUE;

COMMENT ON VIEW current_sku_cluster_assignment IS
    'Current promoted SKU ML cluster labels. Consumers should join this view '
    'for promoted ML clustering.';

-- Rebuild accuracy MVs so existing databases read promoted assignment rows.
DROP MATERIALIZED VIEW IF EXISTS agg_accuracy_by_dim;
DROP MATERIALIZED VIEW IF EXISTS agg_accuracy_lag_archive;
DROP MATERIALIZED VIEW IF EXISTS agg_dfu_coverage;
DROP MATERIALIZED VIEW IF EXISTS agg_dfu_coverage_lag_archive;
DROP MATERIALIZED VIEW IF EXISTS agg_accuracy_by_dfu;

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
  COUNT(*)::bigint                                     AS row_count,
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

CREATE UNIQUE INDEX IF NOT EXISTS uq_agg_accuracy_dim
ON agg_accuracy_by_dim (model_id, lag, month_start, cluster_assignment, ml_cluster,
                        supplier_desc, abc_vol, region, brand_desc, dfu_execution_lag,
                        seasonality_profile);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_model_lag
  ON agg_accuracy_by_dim (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_cluster
  ON agg_accuracy_by_dim (cluster_assignment);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dim_month
  ON agg_accuracy_by_dim (month_start);

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
  COUNT(*)::bigint                                     AS row_count,
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

CREATE UNIQUE INDEX IF NOT EXISTS uq_agg_accuracy_lag_archive
ON agg_accuracy_lag_archive (model_id, lag, timeframe, month_start, cluster_assignment,
                             ml_cluster, supplier_desc, abc_vol, region, brand_desc,
                             seasonality_profile);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_lag_archive_model_lag
  ON agg_accuracy_lag_archive (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_lag_archive_cluster
  ON agg_accuracy_lag_archive (cluster_assignment);

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_dfu_coverage AS
SELECT
  f.model_id,
  f.lag,
  f.item_id,
  f.customer_group,
  f.loc,
  COALESCE(d.cluster_assignment, '(unassigned)')  AS cluster_assignment,
  COALESCE(ca.cluster_label, '(unassigned)')      AS ml_cluster,
  COALESCE(d.supplier_desc, '(unknown)')          AS supplier_desc,
  COALESCE(d.abc_vol, '(unknown)')                AS abc_vol,
  COALESCE(d.region, '(unknown)')                 AS region,
  COALESCE(d.brand_desc, '(unknown)')             AS brand_desc,
  COALESCE(d.execution_lag::text, '(none)')       AS dfu_execution_lag,
  COALESCE(d.seasonality_profile, '(unknown)')    AS seasonality_profile,
  MIN(date_trunc('month', f.startdate)::date)     AS min_month,
  MAX(date_trunc('month', f.startdate)::date)     AS max_month
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
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uq_dfu_coverage_model_lag_dfu
ON agg_dfu_coverage (model_id, lag, item_id, customer_group, loc);
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_model_lag
  ON agg_dfu_coverage (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_cluster
  ON agg_dfu_coverage (cluster_assignment);

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_dfu_coverage_lag_archive AS
SELECT
  a.model_id,
  a.lag,
  a.item_id,
  a.customer_group,
  a.loc,
  COALESCE(d.cluster_assignment, '(unassigned)')  AS cluster_assignment,
  COALESCE(ca.cluster_label, '(unassigned)')      AS ml_cluster,
  COALESCE(d.supplier_desc, '(unknown)')          AS supplier_desc,
  COALESCE(d.abc_vol, '(unknown)')                AS abc_vol,
  COALESCE(d.region, '(unknown)')                 AS region,
  COALESCE(d.brand_desc, '(unknown)')             AS brand_desc,
  COALESCE(d.execution_lag::text, '(none)')       AS dfu_execution_lag,
  COALESCE(d.seasonality_profile, '(unknown)')    AS seasonality_profile,
  MIN(date_trunc('month', a.startdate)::date)     AS min_month,
  MAX(date_trunc('month', a.startdate)::date)     AS max_month
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
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uq_dfu_coverage_lag_archive
ON agg_dfu_coverage_lag_archive (model_id, lag, item_id, customer_group, loc);
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_la_model_lag
  ON agg_dfu_coverage_lag_archive (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_dfu_coverage_la_cluster
  ON agg_dfu_coverage_lag_archive (cluster_assignment);

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_accuracy_by_dfu AS
SELECT
  f.model_id,
  f.lag,
  f.item_id,
  f.customer_group,
  f.loc,
  COALESCE(d.cluster_assignment, '(unassigned)')       AS cluster_assignment,
  COALESCE(ca.cluster_label, '(unassigned)')           AS ml_cluster,
  COALESCE(d.supplier_desc, '(unknown)')               AS supplier_desc,
  COALESCE(d.abc_vol, '(unknown)')                     AS abc_vol,
  COALESCE(d.region, '(unknown)')                      AS region,
  COALESCE(d.brand_desc, '(unknown)')                  AS brand_desc,
  COALESCE(d.execution_lag::text, '(none)')            AS dfu_execution_lag,
  COALESCE(d.seasonality_profile, '(unknown)')         AS seasonality_profile,
  COUNT(*)::bigint                                     AS row_count,
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
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uq_agg_accuracy_by_dfu
  ON agg_accuracy_by_dfu (model_id, lag, item_id, customer_group, loc);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_model_lag
  ON agg_accuracy_by_dfu (model_id, lag);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_cluster
  ON agg_accuracy_by_dfu (cluster_assignment);
CREATE INDEX IF NOT EXISTS idx_agg_accuracy_by_dfu_execution_lag
  ON agg_accuracy_by_dfu (dfu_execution_lag);
