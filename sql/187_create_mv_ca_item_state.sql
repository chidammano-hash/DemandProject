-- mv_ca_item_state
--
-- Pre-aggregated per-(item_id, state, month) rollup powering the
-- /customer-analytics/heatmap Item x State matrix.
--
-- F5.1: the heatmap was the sole CA panel still hitting the raw
-- fact_customer_demand_monthly JOIN dim_customer on every cold call. Its `agg`
-- CTE groups by (item_id, item_desc, state) and is then re-scanned 3 more
-- times (top_items, top_states, final join). On a year of data that cold call
-- took ~9.4 s — dominated by a Seq Scan of dim_item (~500k rows) inside the
-- CTE plus the repeated CTE materialization. Every distinct State/Channel/date
-- filter combination is a fresh ~9 s scan; the 5-minute server cache only hides
-- repeat hits.
--
-- This MV pre-joins + pre-aggregates to the heatmap's exact grain so the cold
-- query collapses to a small indexed scan over a ~490k-row view. The endpoint
-- still does the top_n-items x top-30-states reduction at query time, but over
-- the MV instead of the raw fact join.
--
-- Grain: (item_id, item_desc, state, startdate)
--   item_desc resolved once here (COALESCE to item_id) so the endpoint never
--   touches dim_item.
--
-- Cardinality on prod: ~490k rows (item x state x month). Modest.
--
-- Refresh cadence: nightly / after a customer-demand reload. CONCURRENTLY safe
-- (UNIQUE index below).

DROP MATERIALIZED VIEW IF EXISTS mv_ca_item_state CASCADE;

CREATE MATERIALIZED VIEW mv_ca_item_state AS
SELECT
    f.item_id,
    COALESCE(i.item_desc, f.item_id) AS item_desc,
    c.state,
    c.rpt_channel_desc,
    c.store_type_desc,
    f.startdate,
    COUNT(DISTINCT f.customer_no) AS customer_count,
    SUM(f.demand_qty)             AS demand_qty,
    SUM(f.sales_qty)              AS sales_qty
FROM fact_customer_demand_monthly f
JOIN dim_customer c
    ON c.customer_no = f.customer_no
   AND c.site = f.site
LEFT JOIN dim_item i
    ON i.item_id = f.item_id
WHERE c.state IS NOT NULL AND TRIM(c.state) != ''
GROUP BY f.item_id, i.item_desc, c.state, c.rpt_channel_desc, c.store_type_desc, f.startdate;

-- UNIQUE index required for REFRESH MATERIALIZED VIEW CONCURRENTLY.
-- (item_id, state, channel, store_type, startdate) is the full grain.
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_ca_item_state_pk
    ON mv_ca_item_state (item_id, state, rpt_channel_desc, store_type_desc, startdate);

-- Endpoint filters startdate range (+ optional channel / store_type) then
-- aggregates by (item_id, state). Cover the hot path.
CREATE INDEX IF NOT EXISTS idx_mv_ca_item_state_date
    ON mv_ca_item_state (startdate);

ANALYZE mv_ca_item_state;
