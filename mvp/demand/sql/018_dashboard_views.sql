-- Feature 36: Dashboard materialized views
-- Top movers: period-over-period volume change by item

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_movers AS
SELECT
    s.item,
    i.item_description,
    i.brand,
    i.class_ AS category,
    SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '1 month')
             THEN s.qty END) AS current_qty,
    SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '2 months')
              AND s.month_actual < (CURRENT_DATE - INTERVAL '1 month')
             THEN s.qty END) AS prior_qty,
    COALESCE(
      SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0
    ) - COALESCE(
      SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '2 months')
                AND s.month_actual < (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0
    ) AS delta
FROM fact_sales_monthly s
JOIN dim_item i ON s.item = i.item
GROUP BY s.item, i.item_description, i.brand, i.class_
ORDER BY ABS(
    COALESCE(
      SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0
    ) - COALESCE(
      SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '2 months')
                AND s.month_actual < (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0
    )
) DESC
LIMIT 50;

CREATE INDEX IF NOT EXISTS idx_mv_top_movers_delta ON mv_top_movers (delta DESC);
