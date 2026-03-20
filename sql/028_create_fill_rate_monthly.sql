-- IPfeature8: Fill Rate & Demand Fulfillment Analytics
-- Materialized view: mv_fill_rate_monthly
-- Computes fill_rate = qty_shipped / qty_ordered per item-loc-month
-- from fact_sales_monthly (type=1, qty_ordered > 0 rows only).

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_fill_rate_monthly AS
SELECT
    s.dmdunit                              AS item_no,
    s.loc,
    s.startdate                            AS month_start,
    SUM(s.qty_ordered)                     AS total_ordered,
    SUM(s.qty_shipped)                     AS total_shipped,
    CASE WHEN SUM(s.qty_ordered) > 0
         THEN SUM(s.qty_shipped) / SUM(s.qty_ordered)
         ELSE NULL END                     AS fill_rate,
    GREATEST(SUM(s.qty_ordered) - SUM(s.qty_shipped), 0) AS shortage_qty,
    (SUM(s.qty_shipped) < SUM(s.qty_ordered)) AS had_partial_fulfillment,
    -- DFU attributes for slicing (from dim_dfu)
    COALESCE(d.abc_vol, '(unknown)')               AS abc_vol,
    COALESCE(d.cluster_assignment, '(unassigned)') AS cluster_assignment,
    COALESCE(d.region, '(unknown)')                AS region,
    d.is_yearly_seasonal,
    d.seasonality_profile,
    d.variability_class
FROM fact_sales_monthly s
LEFT JOIN dim_dfu d
    ON s.dmdunit = d.dmdunit
    AND s.dmdgroup = d.dmdgroup
    AND s.loc = d.loc
WHERE s.type = 1
  AND s.qty_ordered IS NOT NULL
  AND s.qty_ordered > 0
GROUP BY
    s.dmdunit, s.loc, s.startdate,
    d.abc_vol, d.cluster_assignment, d.region,
    d.is_yearly_seasonal, d.seasonality_profile, d.variability_class
WITH NO DATA;

-- Indexes for fast filtering and sorting
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_fill_rate_pk
    ON mv_fill_rate_monthly (item_no, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_month
    ON mv_fill_rate_monthly (month_start);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_abc
    ON mv_fill_rate_monthly (abc_vol);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_partial
    ON mv_fill_rate_monthly (had_partial_fulfillment)
    WHERE had_partial_fulfillment = TRUE;
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_low
    ON mv_fill_rate_monthly (fill_rate)
    WHERE fill_rate < 0.95;
