-- 118: Add BRIN indexes for large time-series tables
-- BRIN (Block Range Index) indexes are ~100-1000x smaller than B-tree
-- for naturally ordered data. They work by storing min/max summaries
-- per range of physical pages rather than indexing every row.
--
-- Ideal for: partitioned fact tables where data arrives in date order
-- and queries typically filter by date range.

-- ==========================================================================
-- 1. fact_customer_demand_monthly — 297M rows, 77 GB
-- ==========================================================================
-- Partition pruning handles startdate filtering, but within each partition
-- BRIN helps narrow down item_id ranges (data is loaded in item order).

-- NOTE: BRIN on partitioned tables requires creating on EACH partition.
-- We create on the parent and let PG propagate to partitions.

-- Composite BRIN for range scans within partitions
DO $$
DECLARE
    part_name TEXT;
BEGIN
    FOR part_name IN
        SELECT child.relname
        FROM pg_inherits
        JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
        JOIN pg_class child ON pg_inherits.inhrelid = child.oid
        WHERE parent.relname = 'fact_customer_demand_monthly'
        ORDER BY child.relname
    LOOP
        -- BRIN on item_id within each partition (data loaded in item order)
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I USING BRIN (item_id) WITH (pages_per_range = 32)',
            'brin_' || part_name || '_item',
            part_name
        );
    END LOOP;
END $$;


-- ==========================================================================
-- 2. fact_inventory_snapshot — 87M rows, 14 GB
-- ==========================================================================
-- Partitioned by snapshot_date. Within each partition, data is ordered by
-- item_id + loc. BRIN helps range scans.

DO $$
DECLARE
    part_name TEXT;
BEGIN
    FOR part_name IN
        SELECT child.relname
        FROM pg_inherits
        JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
        JOIN pg_class child ON pg_inherits.inhrelid = child.oid
        WHERE parent.relname = 'fact_inventory_snapshot'
        ORDER BY child.relname
    LOOP
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I USING BRIN (item_id, loc) WITH (pages_per_range = 32)',
            'brin_' || part_name || '_item_loc',
            part_name
        );
    END LOOP;
END $$;


-- ==========================================================================
-- 3. fact_purchase_orders — 5.6M rows, 2.5 GB
-- ==========================================================================
-- Date-ordered data, queries filter by delivery_date ranges
CREATE INDEX CONCURRENTLY IF NOT EXISTS brin_po_delivery_date
ON fact_purchase_orders USING BRIN (delivery_date) WITH (pages_per_range = 64);

-- ==========================================================================
-- 4. backtest_lag_archive — grows with each backtest run
-- ==========================================================================
CREATE INDEX CONCURRENTLY IF NOT EXISTS brin_backtest_archive_fcstdate
ON backtest_lag_archive USING BRIN (fcstdate) WITH (pages_per_range = 32);
