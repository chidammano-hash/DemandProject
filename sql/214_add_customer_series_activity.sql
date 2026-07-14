-- Precompute recent customer sales activity for fast readiness and batching.
--
-- Rebuilding the existing profile avoids repeated scans of the partitioned
-- customer-demand fact whenever the UI checks readiness or starts a run.

BEGIN;

DROP MATERIALIZED VIEW IF EXISTS mv_customer_demand_series_profile;

CREATE MATERIALIZED VIEW mv_customer_demand_series_profile AS
SELECT
    item_id,
    location_id,
    customer_no,
    MIN(startdate) AS first_month,
    MAX(startdate) AS last_month,
    MAX(startdate) FILTER (WHERE sales_qty > 0) AS last_sales_month
FROM fact_customer_demand_monthly
GROUP BY item_id, location_id, customer_no;

CREATE UNIQUE INDEX uq_mv_customer_demand_series_profile
    ON mv_customer_demand_series_profile (item_id, location_id, customer_no);

CREATE INDEX idx_mv_customer_demand_series_profile_bounds
    ON mv_customer_demand_series_profile (first_month, last_month);

CREATE INDEX idx_mv_customer_demand_series_profile_last_sales
    ON mv_customer_demand_series_profile (last_sales_month, first_month);

ANALYZE mv_customer_demand_series_profile;

COMMIT;
