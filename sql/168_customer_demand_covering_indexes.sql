-- Convert the lifecycle / demand-at-risk indexes from "non-covering" to
-- "covering" by adding INCLUDE columns. Without INCLUDE, every index hit
-- still requires a heap fetch to read the aggregate columns (demand_qty,
-- sales_qty, oos_qty), which doubles I/O on the hot path.
--
-- With INCLUDE, the planner can satisfy the entire query from the index
-- (index-only scan), provided VACUUM has kept the visibility map current.
--
-- Replaces the indexes from 167_customer_demand_lifecycle_indexes.sql.
-- Safe to re-run: drops the old non-covering versions first.

DROP INDEX IF EXISTS idx_cust_demand_customer_startdate;
DROP INDEX IF EXISTS idx_cust_demand_item_customer_startdate;

CREATE INDEX IF NOT EXISTS idx_cust_demand_customer_startdate
    ON fact_customer_demand_monthly (customer_no, startdate)
    INCLUDE (demand_qty, sales_qty, oos_qty);

CREATE INDEX IF NOT EXISTS idx_cust_demand_item_customer_startdate
    ON fact_customer_demand_monthly (item_id, customer_no, startdate)
    INCLUDE (demand_qty, sales_qty, oos_qty);

-- Refresh planner stats after rebuilding indexes.
ANALYZE fact_customer_demand_monthly;
