-- Customer-level demand fact table (monthly range partitioned).
-- Grain: item_id + customer_no + location_id + startdate
-- Source: {YYYY}_customer_demand.csv or {YYYYMM}_customer_demand.csv
-- Columns: site, warehouse_no, item_no, customer_no, posting_prd, demand_cases, oos_cases
--
-- Partitioned by startdate (monthly ranges) for:
--   1. Efficient drop-and-reload per month (future incremental loads)
--   2. Partition pruning on date-filtered queries
--   3. Per-partition index maintenance
--
-- demand_qty = MAX(0, demand_cases)     -- true demand (ordered), negatives floored
-- sales_qty  = MAX(0, demand_cases - oos_cases)  -- actual shipped
-- oos_qty    = MAX(0, oos_cases)        -- unfulfilled demand

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly (
    demand_sk       BIGSERIAL,
    demand_ck       TEXT NOT NULL,               -- item_id_customer_no_location_id_startdate
    item_id         TEXT NOT NULL,               -- FK dim_item.item_id (source: item_no)
    customer_no     TEXT NOT NULL,               -- FK dim_customer.customer_no (scoped to site)
    site            TEXT NOT NULL,               -- site identifier for dim_customer FK
    location_id     TEXT NOT NULL,               -- FK dim_location.location_id (resolved from warehouse_no)
    startdate       DATE NOT NULL,               -- first day of month (partition key)
    demand_qty      NUMERIC(18,4) NOT NULL,      -- ordered qty in cases (floored at 0)
    sales_qty       NUMERIC(18,4) NOT NULL,      -- shipped qty = demand - oos (floored at 0)
    oos_qty         NUMERIC(18,4) NOT NULL DEFAULT 0, -- out-of-stock qty in cases
    load_ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_ts     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_cust_demand_month_start
        CHECK (startdate = date_trunc('month', startdate)::date),
    CONSTRAINT chk_cust_demand_qty_nonneg
        CHECK (demand_qty >= 0 AND sales_qty >= 0 AND oos_qty >= 0),
    CONSTRAINT uq_cust_demand_ck UNIQUE (demand_ck, startdate)
) PARTITION BY RANGE (startdate);

-- Default partition for out-of-range data
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_default
    PARTITION OF fact_customer_demand_monthly DEFAULT;

-- Pre-create partitions for expected data range (2024-01 through 2026-12).
-- The loader auto-creates new partitions dynamically.
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_01 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_02 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_03 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_04 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_05 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_06 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_07 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_08 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_09 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_10 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_11 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2024_12 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_01 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_02 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_03 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_04 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_05 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_06 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_07 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_08 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_09 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_10 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_11 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2025_12 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_01 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_02 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_03 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_04 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_05 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_06 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_07 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_08 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_09 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_10 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_11 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');
CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_2026_12 PARTITION OF fact_customer_demand_monthly FOR VALUES FROM ('2026-12-01') TO ('2027-01-01');

-- Indexes (auto-propagated to all partitions)
CREATE INDEX IF NOT EXISTS idx_cust_demand_item        ON fact_customer_demand_monthly (item_id);
CREATE INDEX IF NOT EXISTS idx_cust_demand_customer    ON fact_customer_demand_monthly (customer_no);
CREATE INDEX IF NOT EXISTS idx_cust_demand_location    ON fact_customer_demand_monthly (location_id);
CREATE INDEX IF NOT EXISTS idx_cust_demand_startdate   ON fact_customer_demand_monthly (startdate);
CREATE INDEX IF NOT EXISTS idx_cust_demand_item_loc    ON fact_customer_demand_monthly (item_id, location_id);
CREATE INDEX IF NOT EXISTS idx_cust_demand_item_cust   ON fact_customer_demand_monthly (item_id, customer_no);
CREATE INDEX IF NOT EXISTS idx_cust_demand_site_cust   ON fact_customer_demand_monthly (site, customer_no);
