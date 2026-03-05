-- IPfeature15: Unified Inventory Control Tower
-- Materialized view: mv_control_tower_kpis
-- Thin aggregation layer pulling KPIs from all upstream IP features.

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_control_tower_kpis AS
SELECT
    NOW() AS computed_at,

    -- === Health Score KPIs ===
    COUNT(*)                                          AS total_dfus,
    COUNT(*) FILTER (WHERE health_tier = 'healthy')  AS healthy_count,
    COUNT(*) FILTER (WHERE health_tier = 'monitor')  AS monitor_count,
    COUNT(*) FILTER (WHERE health_tier = 'at_risk')  AS at_risk_count,
    COUNT(*) FILTER (WHERE health_tier = 'critical') AS critical_count,
    AVG(health_score)                                 AS avg_health_score,
    AVG(ss_coverage)                                  AS avg_ss_coverage,
    COUNT(*) FILTER (WHERE is_below_ss = TRUE)        AS below_ss_count,
    CASE WHEN COUNT(*) > 0
         THEN COUNT(*) FILTER (WHERE is_below_ss = TRUE)::NUMERIC / COUNT(*)
         ELSE 0 END                                   AS below_ss_pct,
    AVG(current_dos)                                  AS avg_portfolio_dos,

    -- === Exception KPIs (open only) ===
    (SELECT COUNT(*) FROM fact_replenishment_exceptions
     WHERE status = 'open') AS open_exceptions_total,
    (SELECT COUNT(*) FROM fact_replenishment_exceptions
     WHERE status = 'open' AND severity = 'critical') AS critical_exceptions,
    (SELECT COUNT(*) FROM fact_replenishment_exceptions
     WHERE status = 'open' AND severity = 'high') AS high_exceptions,
    (SELECT SUM(e.recommended_order_qty)
     FROM fact_replenishment_exceptions e
     WHERE e.status = 'open') AS recommended_order_value,

    -- === Fill Rate KPIs (latest 3 months) ===
    (SELECT SUM(total_shipped)::NUMERIC / NULLIF(SUM(total_ordered), 0)
     FROM mv_fill_rate_monthly
     WHERE month_start >= (SELECT MAX(month_start) FROM mv_fill_rate_monthly)
                          - INTERVAL '2 months') AS portfolio_fill_rate_3m,
    (SELECT COALESCE(SUM(shortage_qty), 0)
     FROM mv_fill_rate_monthly
     WHERE month_start >= (SELECT MAX(month_start) FROM mv_fill_rate_monthly)
                          - INTERVAL '2 months') AS total_shortage_qty_3m,

    -- === Demand Signal KPIs (today) ===
    (SELECT COUNT(*) FROM fact_demand_signals
     WHERE signal_date = CURRENT_DATE AND alert_priority = 'urgent') AS urgent_demand_signals,
    (SELECT COUNT(*) FROM fact_demand_signals
     WHERE signal_date = CURRENT_DATE AND projected_stockout = TRUE) AS projected_stockouts_today,

    -- === Intra-Month Stockout KPIs (current month) ===
    (SELECT COUNT(*) FROM mv_intramonth_stockout
     WHERE month_start = DATE_TRUNC('month', CURRENT_DATE)::DATE
       AND had_full_stockout = TRUE) AS items_with_stockout_this_month,
    (SELECT COUNT(*) FROM mv_intramonth_stockout
     WHERE month_start = DATE_TRUNC('month', CURRENT_DATE)::DATE
       AND had_extended_stockout = TRUE) AS extended_stockouts_this_month

FROM mv_inventory_health_score
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_ct_kpis_singleton
    ON mv_control_tower_kpis ((1));
