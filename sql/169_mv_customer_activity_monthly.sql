-- Pre-aggregated customer activity for the customer-analytics lifecycle,
-- demand-at-risk, and order-patterns endpoints.
--
-- All three endpoints repeatedly compute the same shape:
--   SELECT DISTINCT f.customer_no, f.startdate
--   FROM fact_customer_demand_monthly f
--   JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
--   WHERE f.startdate BETWEEN ? AND ? [AND optional channel/store_type/item_id]
--
-- That JOIN+DISTINCT scans millions of rows per request even when the result
-- collapses to ~33K customer × 12 months = ~400K rows. Pre-materializing
-- the join with the customer-dim attributes that participate in filters
-- (channel, store_type, state) lets the endpoints filter a smaller, indexed
-- table directly.
--
-- We keep `item_id`-level granularity OUT of this MV so it stays small —
-- queries that need item filtering still hit the fact table directly.
-- Lifecycle / waterfall / churn never filter by item_id at the customer
-- aggregation step (item_id only filters fact tuples, not customer cohorts).
--
-- Refresh policy:
--   - Refresh after the customer-demand monthly load (`make load-customer-demand`)
--   - CONCURRENTLY so reads don't block during refresh
--   - Cost: ~30s on full dataset; cheap relative to the 100s+ per-request
--     savings on lifecycle endpoints

DROP MATERIALIZED VIEW IF EXISTS mv_customer_activity_monthly CASCADE;

CREATE MATERIALIZED VIEW mv_customer_activity_monthly AS
SELECT
    f.customer_no,
    f.site,
    f.startdate,
    c.rpt_channel_desc,
    c.store_type_desc,
    c.state,
    SUM(f.demand_qty) AS demand_qty,
    SUM(f.sales_qty) AS sales_qty,
    SUM(f.oos_qty) AS oos_qty
FROM fact_customer_demand_monthly f
JOIN dim_customer c
    ON c.customer_no = f.customer_no
   AND c.site = f.site
GROUP BY
    f.customer_no, f.site, f.startdate,
    c.rpt_channel_desc, c.store_type_desc, c.state;

-- UNIQUE index required for REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_cust_activity_pk
    ON mv_customer_activity_monthly (customer_no, site, startdate);

-- Time-series filter: every endpoint filters `startdate BETWEEN ? AND ?`
CREATE INDEX IF NOT EXISTS idx_mv_cust_activity_startdate
    ON mv_customer_activity_monthly (startdate);

-- Customer cohort lookups
CREATE INDEX IF NOT EXISTS idx_mv_cust_activity_customer_startdate
    ON mv_customer_activity_monthly (customer_no, startdate)
    INCLUDE (demand_qty, sales_qty, oos_qty);

-- Channel/store_type filter pushdown
CREATE INDEX IF NOT EXISTS idx_mv_cust_activity_channel_store
    ON mv_customer_activity_monthly (rpt_channel_desc, store_type_desc, startdate);

ANALYZE mv_customer_activity_monthly;
