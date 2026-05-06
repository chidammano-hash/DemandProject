-- mv_ca_segment_trends
--
-- Pre-aggregated per-(segment_dim, segment_value, month) rollup powering
-- /customer-analytics/segment-trends when no item_id filter is set.
--
-- The endpoint groups by one of {rpt_channel_desc, store_type_desc,
-- chain_type_desc, state} and per request runs:
--    fact_customer_demand_monthly JOIN dim_customer
--    GROUP BY <segment_col>, startdate
--    + COUNT(DISTINCT customer_no)
-- COUNT(DISTINCT) over millions of rows is the single most expensive pattern
-- in this tab. Pre-materializing it as a per-segment customer set count
-- collapses each request to a small indexed lookup.
--
-- We materialize ALL FOUR segment dimensions in a single tall MV with a
-- segment_dim discriminator so we don't have to maintain four separate views.
-- The endpoint filters `WHERE segment_dim = %s` at query time.
--
-- Grain: (segment_dim, segment_value, startdate)
--   - segment_dim ∈ {'rpt_channel_desc', 'store_type_desc', 'chain_type_desc', 'state'}
--   - customer_count = exact COUNT(DISTINCT customer_no) precomputed.
--
-- Cardinality on prod: 4 dims * ~50 distinct values each * 24 months
--   = ~4800 rows. Tiny.
--
-- Refresh cadence: nightly. CONCURRENTLY safe.

DROP MATERIALIZED VIEW IF EXISTS mv_ca_segment_trends CASCADE;

CREATE MATERIALIZED VIEW mv_ca_segment_trends AS
WITH base AS (
    SELECT
        f.customer_no,
        f.startdate,
        f.demand_qty,
        f.sales_qty,
        f.oos_qty,
        c.rpt_channel_desc,
        c.store_type_desc,
        c.chain_type_desc,
        c.state
    FROM fact_customer_demand_monthly f
    JOIN dim_customer c
        ON c.customer_no = f.customer_no
       AND c.site = f.site
)
SELECT 'rpt_channel_desc' AS segment_dim,
       rpt_channel_desc   AS segment_value,
       startdate,
       COUNT(DISTINCT customer_no) AS customer_count,
       SUM(demand_qty)             AS demand_qty,
       SUM(sales_qty)              AS sales_qty,
       SUM(oos_qty)                AS oos_qty
FROM base
WHERE rpt_channel_desc IS NOT NULL AND TRIM(rpt_channel_desc) != ''
GROUP BY rpt_channel_desc, startdate

UNION ALL

SELECT 'store_type_desc',
       store_type_desc,
       startdate,
       COUNT(DISTINCT customer_no),
       SUM(demand_qty), SUM(sales_qty), SUM(oos_qty)
FROM base
WHERE store_type_desc IS NOT NULL AND TRIM(store_type_desc) != ''
GROUP BY store_type_desc, startdate

UNION ALL

SELECT 'chain_type_desc',
       chain_type_desc,
       startdate,
       COUNT(DISTINCT customer_no),
       SUM(demand_qty), SUM(sales_qty), SUM(oos_qty)
FROM base
WHERE chain_type_desc IS NOT NULL AND TRIM(chain_type_desc) != ''
GROUP BY chain_type_desc, startdate

UNION ALL

SELECT 'state',
       state,
       startdate,
       COUNT(DISTINCT customer_no),
       SUM(demand_qty), SUM(sales_qty), SUM(oos_qty)
FROM base
WHERE state IS NOT NULL AND TRIM(state) != ''
GROUP BY state, startdate;

-- UNIQUE index required for REFRESH MATERIALIZED VIEW CONCURRENTLY.
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_ca_seg_trends_pk
    ON mv_ca_segment_trends (segment_dim, segment_value, startdate);

-- Endpoint filters segment_dim + startdate range
CREATE INDEX IF NOT EXISTS idx_mv_ca_seg_trends_dim_date
    ON mv_ca_segment_trends (segment_dim, startdate);

ANALYZE mv_ca_segment_trends;
