-- Pre-aggregate immutable customer-series history bounds for Spec 35.
--
-- Readiness and generation need each item-location-customer series' first
-- observed month. Computing that value from the full customer-demand fact on
-- every HTTP request scans all historical partitions and exceeds the API's
-- statement timeout. This refreshable profile moves that work to the existing
-- post-load materialized-view lifecycle.

CREATE MATERIALIZED VIEW mv_customer_demand_series_profile AS
SELECT
    item_id,
    location_id,
    customer_no,
    MIN(startdate) AS first_month,
    MAX(startdate) AS last_month
FROM fact_customer_demand_monthly
GROUP BY item_id, location_id, customer_no;

-- Required for concurrent refreshes and bounded series joins.
CREATE UNIQUE INDEX uq_mv_customer_demand_series_profile
    ON mv_customer_demand_series_profile (item_id, location_id, customer_no);

CREATE INDEX idx_mv_customer_demand_series_profile_bounds
    ON mv_customer_demand_series_profile (first_month, last_month);

ANALYZE mv_customer_demand_series_profile;
