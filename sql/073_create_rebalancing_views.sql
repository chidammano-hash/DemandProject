-- Inventory Rebalancing: Network Balance Materialized View
-- Per-item DOS variance across locations for imbalance detection.

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_network_balance AS
SELECT
    a.item_id,
    COUNT(DISTINCT a.loc)                                                    AS location_count,
    AVG(a.eom_qty_on_hand)                                                   AS avg_on_hand,
    STDDEV(a.eom_qty_on_hand)                                                AS stddev_on_hand,
    AVG(CASE WHEN a.avg_daily_sls > 0
        THEN a.eom_qty_on_hand / a.avg_daily_sls ELSE NULL END)            AS avg_dos,
    STDDEV(CASE WHEN a.avg_daily_sls > 0
        THEN a.eom_qty_on_hand / a.avg_daily_sls ELSE NULL END)            AS stddev_dos,
    CASE WHEN AVG(CASE WHEN a.avg_daily_sls > 0
        THEN a.eom_qty_on_hand / a.avg_daily_sls ELSE NULL END) > 0
    THEN STDDEV(CASE WHEN a.avg_daily_sls > 0
        THEN a.eom_qty_on_hand / a.avg_daily_sls ELSE NULL END)
        / AVG(CASE WHEN a.avg_daily_sls > 0
            THEN a.eom_qty_on_hand / a.avg_daily_sls ELSE NULL END)
    ELSE NULL END                                                            AS dos_cv,
    SUM(CASE WHEN s.ss_combined IS NOT NULL
             AND a.eom_qty_on_hand > s.ss_combined * 1.5 THEN 1 ELSE 0 END) AS excess_loc_count,
    SUM(CASE WHEN s.ss_combined IS NOT NULL
             AND a.eom_qty_on_hand < s.ss_combined THEN 1 ELSE 0 END)       AS shortage_loc_count
FROM agg_inventory_monthly a
LEFT JOIN fact_safety_stock_targets s ON a.item_id = s.item_id AND a.loc = s.loc
WHERE a.month_start = (SELECT MAX(month_start) FROM agg_inventory_monthly)
GROUP BY a.item_id
HAVING COUNT(DISTINCT a.loc) >= 2
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_netbal_item ON mv_network_balance (item_id);
