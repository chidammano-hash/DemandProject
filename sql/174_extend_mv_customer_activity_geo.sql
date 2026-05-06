-- Extend mv_customer_activity_monthly with geo + extra dim_customer attributes.
--
-- Why: 13 of 16 customer-analytics endpoints were still hitting the raw
--   fact_customer_demand_monthly x dim_customer JOIN because the MV lacked
--   the columns they filtered/grouped on (city, zip, location_id, customer_name,
--   rpt_sub_channel_desc, chain_type_desc). The /kpis endpoint alone was
--   measured at 10.8s on the raw join. Routing through the MV brings that
--   down to ~63ms — see Item 8 of the perf roadmap.
--
-- Grain change:
--   - OLD: (customer_no, site, startdate) — purely customer-level
--   - NEW: (customer_no, site, location_id, startdate) — adds the warehouse
--     dimension so /demand-flow can group by f.location_id without falling
--     back to the raw fact join. Customer typically ships from 1-2 warehouses,
--     so row count grows ~1.2x at most.
--
-- Customer-attribute columns added (all functionally dependent on
-- (customer_no, site) so they don't expand grain):
--   - customer_name : /ranking, /treemap, /affinity, /order-patterns
--   - city, zip     : /map (group_by=city|zip)
--   - rpt_sub_channel_desc : /channel-mix sunburst
--   - chain_type_desc      : /segment-trends (segment_by=chain_type_desc)
--
-- Endpoints that still need the raw fact (item_id filter / item-grain joins):
--   - /demand-at-risk (item_id+location_id grouping by raw fact rows)
--   - /heatmap (item_id grouping)
--   - /affinity (item_id pivots)
--   - /alerts low_fill_rate / hhi (item_id grouping)
--   - /items typeahead (uses dim_item)

DROP MATERIALIZED VIEW IF EXISTS mv_customer_activity_monthly CASCADE;

CREATE MATERIALIZED VIEW mv_customer_activity_monthly AS
SELECT
    f.customer_no,
    f.site,
    f.location_id,
    f.startdate,
    c.customer_name,
    c.city,
    c.state,
    c.zip,
    c.rpt_channel_desc,
    c.rpt_sub_channel_desc,
    c.store_type_desc,
    c.chain_type_desc,
    SUM(f.demand_qty) AS demand_qty,
    SUM(f.sales_qty) AS sales_qty,
    SUM(f.oos_qty) AS oos_qty
FROM fact_customer_demand_monthly f
JOIN dim_customer c
    ON c.customer_no = f.customer_no
   AND c.site = f.site
GROUP BY
    f.customer_no, f.site, f.location_id, f.startdate,
    c.customer_name, c.city, c.state, c.zip,
    c.rpt_channel_desc, c.rpt_sub_channel_desc,
    c.store_type_desc, c.chain_type_desc;

-- UNIQUE index required for REFRESH MATERIALIZED VIEW CONCURRENTLY.
-- Includes location_id since the new grain spans warehouses.
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_cust_activity_pk
    ON mv_customer_activity_monthly (customer_no, site, location_id, startdate);

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

-- Geo filter pushdown for /map, /treemap, /heatmap with state filter
CREATE INDEX IF NOT EXISTS idx_mv_cust_activity_state_startdate
    ON mv_customer_activity_monthly (state, startdate);

-- Warehouse rollup for /demand-flow
CREATE INDEX IF NOT EXISTS idx_mv_cust_activity_location_startdate
    ON mv_customer_activity_monthly (location_id, startdate);

ANALYZE mv_customer_activity_monthly;
