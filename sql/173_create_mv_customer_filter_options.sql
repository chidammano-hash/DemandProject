-- 173: Pre-compute customer filter dropdown options.
--
-- Background:
--   /customer-analytics/filter-options serves three dropdowns (channels,
--   store types, states) with three ARRAY_AGG(DISTINCT ...) over dim_customer
--   (~1M rows). On every tab open the planner does three sort/uniq passes,
--   which costs hundreds of milliseconds even with the analytics covering
--   index from sql/166.
--
--   dim_customer changes only when the customer dimension is reloaded
--   (~daily). Caching the distinct sets in a materialized view turns the
--   endpoint into a single 3-row scan.
--
-- Refresh:
--   `make refresh-customer-filter-options` (added in this batch). The Make
--   target is also wired into `make load-customer` so the MV stays in sync
--   with dim_customer reloads.
--
-- Schema:
--   One row per filter category, with the distinct values pre-aggregated
--   into a TEXT[] payload. The endpoint reads three rows and returns them.

DROP MATERIALIZED VIEW IF EXISTS mv_customer_filter_options CASCADE;

CREATE MATERIALIZED VIEW mv_customer_filter_options AS
SELECT
    'channels'::TEXT AS category,
    COALESCE(
        ARRAY_AGG(DISTINCT rpt_channel_desc ORDER BY rpt_channel_desc)
            FILTER (WHERE rpt_channel_desc IS NOT NULL AND TRIM(rpt_channel_desc) != ''),
        ARRAY[]::TEXT[]
    ) AS values
FROM dim_customer
UNION ALL
SELECT
    'store_types'::TEXT AS category,
    COALESCE(
        ARRAY_AGG(DISTINCT store_type_desc ORDER BY store_type_desc)
            FILTER (WHERE store_type_desc IS NOT NULL AND TRIM(store_type_desc) != ''),
        ARRAY[]::TEXT[]
    ) AS values
FROM dim_customer
UNION ALL
SELECT
    'states'::TEXT AS category,
    COALESCE(
        ARRAY_AGG(DISTINCT state ORDER BY state)
            FILTER (WHERE state IS NOT NULL AND TRIM(state) != ''),
        ARRAY[]::TEXT[]
    ) AS values
FROM dim_customer;

-- Unique index required for REFRESH MATERIALIZED VIEW CONCURRENTLY.
CREATE UNIQUE INDEX IF NOT EXISTS uq_mv_customer_filter_options_category
    ON mv_customer_filter_options (category);

COMMENT ON MATERIALIZED VIEW mv_customer_filter_options IS
    'Pre-computed distinct values for the customer-analytics filter dropdowns. '
    'Three rows: channels, store_types, states. Refresh after dim_customer reloads '
    'via `REFRESH MATERIALIZED VIEW CONCURRENTLY mv_customer_filter_options`.';
