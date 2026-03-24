-- Supplier performance from actual PO data
-- Replaces proxy-based mv_supplier_performance with real delivery metrics

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_supplier_po_performance AS
WITH po_metrics AS (
    SELECT
        supplier_id,
        MAX(supplier_name)                                        AS supplier_name,
        count(*)                                                  AS total_lines,
        count(*) FILTER (WHERE is_closed)                         AS closed_lines,
        count(*) FILTER (WHERE NOT is_closed)                     AS open_lines,
        count(DISTINCT po_number)                                 AS distinct_pos,
        count(DISTINCT item_id)                                   AS distinct_items,
        count(DISTINCT loc)                                       AS distinct_locations,
        -- On-time delivery (closed POs with both dates)
        count(*) FILTER (WHERE is_closed
                         AND delivery_date IS NOT NULL
                         AND original_delivery_date IS NOT NULL)   AS delivery_evaluated,
        count(*) FILTER (WHERE is_closed
                         AND delivery_date <= original_delivery_date) AS on_time_count,
        -- Lead time stats (closed POs)
        AVG(lead_time_actual) FILTER (WHERE is_closed
                                      AND lead_time_actual IS NOT NULL) AS avg_lead_time_days,
        STDDEV(lead_time_actual) FILTER (WHERE is_closed
                                         AND lead_time_actual IS NOT NULL) AS stddev_lead_time_days,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lead_time_actual)
            FILTER (WHERE is_closed AND lead_time_actual IS NOT NULL) AS p50_lead_time_days,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY lead_time_actual)
            FILTER (WHERE is_closed AND lead_time_actual IS NOT NULL) AS p90_lead_time_days,
        -- Value metrics
        COALESCE(SUM(gross_value), 0)                             AS total_value,
        COALESCE(SUM(gross_value) FILTER (WHERE NOT is_closed), 0) AS open_value
    FROM fact_purchase_orders
    WHERE supplier_id IS NOT NULL AND supplier_id != ''
    GROUP BY supplier_id
)
SELECT
    supplier_id,
    supplier_name,
    total_lines,
    closed_lines,
    open_lines,
    distinct_pos,
    distinct_items,
    distinct_locations,
    delivery_evaluated,
    on_time_count,
    CASE WHEN delivery_evaluated > 0
         THEN ROUND(100.0 * on_time_count / delivery_evaluated, 1)
         ELSE NULL
    END AS otd_pct,
    ROUND(avg_lead_time_days::numeric, 1) AS avg_lead_time_days,
    ROUND(stddev_lead_time_days::numeric, 1) AS stddev_lead_time_days,
    ROUND(p50_lead_time_days::numeric, 1) AS p50_lead_time_days,
    ROUND(p90_lead_time_days::numeric, 1) AS p90_lead_time_days,
    total_value,
    open_value,
    -- Reliability score: 60% OTD + 40% lead time consistency
    LEAST(100, GREATEST(0,
        60 * COALESCE(on_time_count::float / NULLIF(delivery_evaluated, 0), 0.5)
        + 40 * GREATEST(0, 1.0 - COALESCE(
            stddev_lead_time_days / NULLIF(avg_lead_time_days, 0), 0.5))
    ))::INTEGER AS reliability_score
FROM po_metrics
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_supplier_po_perf_pk
    ON mv_supplier_po_performance (supplier_id);

CREATE INDEX IF NOT EXISTS idx_supplier_po_perf_score
    ON mv_supplier_po_performance (reliability_score);
