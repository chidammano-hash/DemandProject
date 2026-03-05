-- IPfeature14: Intra-Month Stockout Detection (Daily Granularity)
-- Materialized view: mv_intramonth_stockout
-- Computes daily stockout metrics from fact_inventory_snapshot (190M rows).
-- Uses LAG() to derive daily incremental sales from cumulative mtd_sales.
-- WARNING: Full REFRESH takes 10-30 minutes. Use incremental refresh script
--          (scripts/refresh_intramonth_stockout.py) for daily operations.

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_intramonth_stockout AS
WITH daily_with_lag AS (
    SELECT
        item_no,
        loc,
        DATE_TRUNC('month', snapshot_date)::DATE AS month_start,
        snapshot_date,
        qty_on_hand,
        mtd_sales,
        -- Daily incremental sales from cumulative MTD (LAG within same item-loc-month)
        GREATEST(
            mtd_sales
            - LAG(mtd_sales, 1, 0::NUMERIC) OVER (
                PARTITION BY item_no, loc, DATE_TRUNC('month', snapshot_date)
                ORDER BY snapshot_date
            ),
            0
        ) AS daily_sls
    FROM fact_inventory_snapshot
),
monthly_agg AS (
    SELECT
        item_no,
        loc,
        month_start,
        COUNT(*)                                          AS snapshot_days,
        COUNT(*) FILTER (WHERE qty_on_hand <= 0)         AS stockout_days,
        CASE WHEN COUNT(*) > 0
             THEN COUNT(*) FILTER (WHERE qty_on_hand <= 0)::NUMERIC / COUNT(*)
             ELSE NULL END                               AS stockout_day_rate,
        MIN(qty_on_hand)                                 AS min_qty_on_hand,
        MAX(qty_on_hand)                                 AS max_qty_on_hand,
        AVG(qty_on_hand)                                 AS avg_qty_on_hand,
        SUM(daily_sls) FILTER (WHERE qty_on_hand <= 0)  AS est_lost_sales,
        (COUNT(*) FILTER (WHERE qty_on_hand <= 0)) >= 1  AS had_full_stockout,
        (COUNT(*) FILTER (WHERE qty_on_hand <= 0)) >= 7  AS had_extended_stockout
    FROM daily_with_lag
    GROUP BY item_no, loc, month_start
)
SELECT
    m.item_no,
    m.loc,
    m.month_start,
    m.snapshot_days,
    m.stockout_days,
    m.stockout_day_rate,
    m.min_qty_on_hand,
    m.max_qty_on_hand,
    m.avg_qty_on_hand,
    m.est_lost_sales,
    m.had_full_stockout,
    m.had_extended_stockout,
    -- DFU attributes for slicing
    COALESCE(d.abc_vol, '(unknown)')               AS abc_vol,
    COALESCE(d.abc_xyz_segment, '(unknown)')       AS abc_xyz_segment,
    COALESCE(d.cluster_assignment, '(unassigned)') AS cluster_assignment,
    d.variability_class
FROM monthly_agg m
LEFT JOIN dim_dfu d
    ON m.item_no = d.dmdunit
    AND m.loc = d.loc
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_intramonth_pk
    ON mv_intramonth_stockout (item_no, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_intramonth_month
    ON mv_intramonth_stockout (month_start DESC);
CREATE INDEX IF NOT EXISTS idx_intramonth_abc
    ON mv_intramonth_stockout (abc_vol, month_start);
CREATE INDEX IF NOT EXISTS idx_intramonth_extended
    ON mv_intramonth_stockout (had_extended_stockout)
    WHERE had_extended_stockout = TRUE;
CREATE INDEX IF NOT EXISTS idx_intramonth_high_rate
    ON mv_intramonth_stockout (stockout_day_rate DESC)
    WHERE stockout_day_rate > 0.1;
