-- Gen-4 Roadmap 1.5 / 1.6
--
-- 1.5 Add OTIF (On-Time In-Full: date AND quantity) alongside existing OTD
--     to mv_supplier_po_performance.
-- 1.6 Retire duplicate mv_supplier_performance — keep mv_supplier_po_performance only.
--
-- Drops and recreates mv_supplier_po_performance to add OTIF columns derived
-- from fact_purchase_orders.
--   - in_full_count        : closed PO lines where received_qty >= ordered_qty
--   - otif_count           : closed lines that are on-time AND in-full
--   - otif_pct             : % of delivery_evaluated lines that were OTIF
--
-- Retires mv_supplier_performance (the proxy based on dim_item_lead_time_profile
-- with fabricated reliability scores).

BEGIN;

-- 1. Retire old MV
DROP MATERIALIZED VIEW IF EXISTS mv_supplier_performance CASCADE;

-- 2. Recreate mv_supplier_po_performance with OTIF metrics
DROP MATERIALIZED VIEW IF EXISTS mv_supplier_po_performance CASCADE;

CREATE MATERIALIZED VIEW mv_supplier_po_performance AS
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
        -- Delivery evaluation denominator: closed POs with both dates
        count(*) FILTER (WHERE is_closed
                         AND delivery_date IS NOT NULL
                         AND original_delivery_date IS NOT NULL)   AS delivery_evaluated,
        -- OTD: on-time delivery (date only)
        count(*) FILTER (WHERE is_closed
                         AND delivery_date <= original_delivery_date) AS on_time_count,
        -- In-full: received_qty >= ordered_qty (closed + quantities known)
        count(*) FILTER (WHERE is_closed
                         AND received_qty IS NOT NULL
                         AND ordered_qty IS NOT NULL
                         AND ordered_qty > 0
                         AND received_qty >= ordered_qty) AS in_full_count,
        -- OTIF: on-time AND in-full
        count(*) FILTER (WHERE is_closed
                         AND delivery_date IS NOT NULL
                         AND original_delivery_date IS NOT NULL
                         AND delivery_date <= original_delivery_date
                         AND received_qty IS NOT NULL
                         AND ordered_qty IS NOT NULL
                         AND ordered_qty > 0
                         AND received_qty >= ordered_qty) AS otif_count,
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
    in_full_count,
    otif_count,
    CASE WHEN delivery_evaluated > 0
         THEN ROUND(100.0 * on_time_count / delivery_evaluated, 1)
         ELSE NULL
    END AS otd_pct,
    CASE WHEN delivery_evaluated > 0
         THEN ROUND(100.0 * in_full_count / delivery_evaluated, 1)
         ELSE NULL
    END AS in_full_pct,
    CASE WHEN delivery_evaluated > 0
         THEN ROUND(100.0 * otif_count / delivery_evaluated, 1)
         ELSE NULL
    END AS otif_pct,
    ROUND(avg_lead_time_days::numeric, 1) AS avg_lead_time_days,
    ROUND(stddev_lead_time_days::numeric, 1) AS stddev_lead_time_days,
    ROUND(p50_lead_time_days::numeric, 1) AS p50_lead_time_days,
    ROUND(p90_lead_time_days::numeric, 1) AS p90_lead_time_days,
    total_value,
    open_value,
    -- Reliability score: 40% OTIF + 20% OTD + 40% lead time consistency
    -- (OTIF captures full fulfillment; OTD retained for backward compat weight)
    LEAST(100, GREATEST(0,
        40 * COALESCE(otif_count::float / NULLIF(delivery_evaluated, 0), 0.5)
        + 20 * COALESCE(on_time_count::float / NULLIF(delivery_evaluated, 0), 0.5)
        + 40 * GREATEST(0, 1.0 - COALESCE(
            stddev_lead_time_days / NULLIF(avg_lead_time_days, 0), 0.5))
    ))::INTEGER AS reliability_score
FROM po_metrics
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_supplier_po_perf_pk
    ON mv_supplier_po_performance (supplier_id);

CREATE INDEX IF NOT EXISTS idx_supplier_po_perf_score
    ON mv_supplier_po_performance (reliability_score);

CREATE INDEX IF NOT EXISTS idx_supplier_po_perf_otif
    ON mv_supplier_po_performance (otif_pct);

COMMIT;
