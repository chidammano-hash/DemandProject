-- IPfeature12: Supplier Performance Intelligence
-- Materialized view: mv_supplier_performance
-- Joins dim_item_lead_time_profile + dim_item + fact_safety_stock_targets
-- to compute per-supplier reliability scores and SS attribution.

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_supplier_performance AS
WITH lt_by_supplier AS (
    SELECT
        i.supplier_no,
        i.supplier_name,
        ltp.item_id,
        ltp.loc,
        ltp.lt_mean_days,
        ltp.lt_std_days,
        ltp.lt_cv,
        ltp.lt_variability_class,
        ltp.observation_months
    FROM dim_item_lead_time_profile ltp
    INNER JOIN dim_item i ON ltp.item_id = i.item_id
    WHERE i.supplier_no IS NOT NULL
),
ss_by_supplier AS (
    SELECT
        i.supplier_no,
        SUM(s.ss_combined)                                   AS total_safety_stock_units,
        NULL::NUMERIC                                        AS ss_from_lt_variance,
        AVG(s.ss_coverage)                                   AS avg_ss_coverage,
        COUNT(*)                                             AS sku_loc_count,
        SUM(CASE WHEN s.is_below_ss THEN 1 ELSE 0 END)      AS below_ss_count,
        SUM(s.ss_combined)                                   AS total_ss_value
    FROM fact_safety_stock_targets s
    INNER JOIN dim_item i ON s.item_id = i.item_id
    WHERE s.policy_version = 'v1'
    GROUP BY i.supplier_no
)
SELECT
    l.supplier_no,
    MAX(l.supplier_name)                       AS supplier_name,
    COUNT(DISTINCT l.item_id || '_' || l.loc)  AS sku_loc_count,
    COUNT(DISTINCT l.item_id)                  AS distinct_items,
    AVG(l.lt_mean_days)                        AS avg_lt_mean_days,
    STDDEV(l.lt_mean_days)                     AS stddev_lt_mean_cross_skus,
    AVG(l.lt_std_days)                         AS avg_lt_std_days,
    AVG(l.lt_cv)                               AS avg_lt_cv,
    MIN(l.lt_mean_days)                        AS min_lt_days,
    MAX(l.lt_mean_days)                        AS max_lt_days,
    SUM(CASE WHEN l.lt_variability_class = 'stable' THEN 1 ELSE 0 END)::float
        / NULLIF(COUNT(*), 0)                  AS pct_stable_lt,
    SUM(CASE WHEN l.lt_variability_class = 'volatile' THEN 1 ELSE 0 END)::float
        / NULLIF(COUNT(*), 0)                  AS pct_volatile_lt,
    -- SS attribution from IPfeature3
    s.total_safety_stock_units,
    s.ss_from_lt_variance,
    s.avg_ss_coverage,
    s.below_ss_count,
    s.total_ss_value,
    -- Reliability score (0–100): 50 * pct_stable_lt + 50 * (1 – avg_lt_cv)
    LEAST(100, GREATEST(0,
        50 * COALESCE(
            SUM(CASE WHEN l.lt_variability_class = 'stable' THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0),
        0.5)
        + 50 * GREATEST(0, 1 - COALESCE(AVG(l.lt_cv), 0))
    ))::INTEGER AS supplier_reliability_score
FROM lt_by_supplier l
LEFT JOIN ss_by_supplier s ON l.supplier_no = s.supplier_no
GROUP BY l.supplier_no,
         s.total_safety_stock_units, s.ss_from_lt_variance, s.avg_ss_coverage,
         s.below_ss_count, s.total_ss_value
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_supplier_perf_pk
    ON mv_supplier_performance (supplier_no);
CREATE INDEX IF NOT EXISTS idx_supplier_perf_score
    ON mv_supplier_performance (supplier_reliability_score);
CREATE INDEX IF NOT EXISTS idx_supplier_perf_lt_cv
    ON mv_supplier_performance (avg_lt_cv DESC);
