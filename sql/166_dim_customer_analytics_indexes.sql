-- Composite indexes on dim_customer to speed customer-analytics joins.
-- Customer-analytics endpoints all join fact_customer_demand_monthly
-- on (customer_no, site) and then filter by combinations of state /
-- rpt_channel_desc / store_type_desc. Without these indexes the planner
-- often picks a seq-scan on dim_customer, which dominates query time.

CREATE INDEX IF NOT EXISTS idx_dim_customer_no_site_filters
    ON dim_customer (customer_no, site)
    INCLUDE (state, rpt_channel_desc, store_type_desc, rpt_sub_channel_desc, customer_name);

CREATE INDEX IF NOT EXISTS idx_dim_customer_state
    ON dim_customer (state)
    WHERE state IS NOT NULL AND TRIM(state) <> '';

CREATE INDEX IF NOT EXISTS idx_dim_customer_channel
    ON dim_customer (rpt_channel_desc)
    WHERE rpt_channel_desc IS NOT NULL AND TRIM(rpt_channel_desc) <> '';

CREATE INDEX IF NOT EXISTS idx_dim_customer_store_type
    ON dim_customer (store_type_desc)
    WHERE store_type_desc IS NOT NULL AND TRIM(store_type_desc) <> '';

ANALYZE dim_customer;
