-- PO lead time trend analysis by supplier × month
-- Shows monthly lead time trends for closed POs

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_po_lead_time_analysis AS
SELECT
    supplier_id,
    MAX(supplier_name) AS supplier_name,
    DATE_TRUNC('month', delivery_date)::date AS delivery_month,
    count(*) AS line_count,
    ROUND(AVG(lead_time_actual)::numeric, 1) AS avg_lead_time_days,
    ROUND(STDDEV(lead_time_actual)::numeric, 1) AS stddev_lead_time_days,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lead_time_actual)::numeric, 1)
        AS p50_lead_time_days,
    ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY lead_time_actual)::numeric, 1)
        AS p90_lead_time_days,
    ROUND(100.0 * count(*) FILTER (WHERE delivery_date <= original_delivery_date)
          / NULLIF(count(*), 0), 1) AS otd_pct,
    ROUND(100.0 * count(*) FILTER (WHERE delivery_date > original_delivery_date)
          / NULLIF(count(*), 0), 1) AS late_pct,
    ROUND(100.0 * count(*) FILTER (WHERE delivery_date < original_delivery_date)
          / NULLIF(count(*), 0), 1) AS early_pct
FROM fact_purchase_orders
WHERE is_closed
  AND delivery_date IS NOT NULL
  AND lead_time_actual IS NOT NULL
  AND supplier_id IS NOT NULL AND supplier_id != ''
GROUP BY supplier_id, DATE_TRUNC('month', delivery_date)
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_po_lt_analysis_pk
    ON mv_po_lead_time_analysis (supplier_id, delivery_month);

CREATE INDEX IF NOT EXISTS idx_po_lt_analysis_month
    ON mv_po_lead_time_analysis (delivery_month);
