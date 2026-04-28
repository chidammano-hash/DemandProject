-- Covering indexes for the customer-analytics lifecycle / demand-at-risk queries.
-- Both endpoints repeatedly compute MIN/MAX(startdate) per customer_no and
-- scan windowed (max - 6mo, max - 3mo) ranges. Existing indexes cover only
-- (customer_no) or (item_id, customer_no), forcing a heap fetch per row.
--
-- Adding (customer_no, startdate) unlocks index-only scans for first/last
-- order detection and the 3-/6-month windowing in churn calculations.
-- The (item_id, customer_no, startdate) variant covers the same pattern
-- when an item filter is active (lifecycle/at-risk both accept item_id).

CREATE INDEX IF NOT EXISTS idx_cust_demand_customer_startdate
    ON fact_customer_demand_monthly (customer_no, startdate);

CREATE INDEX IF NOT EXISTS idx_cust_demand_item_customer_startdate
    ON fact_customer_demand_monthly (item_id, customer_no, startdate);

ANALYZE fact_customer_demand_monthly;
