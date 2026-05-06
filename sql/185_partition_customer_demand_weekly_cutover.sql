-- =============================================================================
-- Migration: fact_customer_demand_monthly — monthly → weekly partitioning cutover
-- =============================================================================
--
-- !! REVIEW BEFORE RUN — schema cutover. !!
--
-- This migration adds NEW weekly partitions to the existing monthly-partitioned
-- ``fact_customer_demand_monthly`` parent. Historical monthly partitions
-- (sql/110) are LEFT IN PLACE.
--
-- Note on grain: the table's ``startdate`` column has a CHECK constraint
-- (``chk_cust_demand_month_start``) requiring ``startdate = date_trunc('month', startdate)``.
-- Weekly partitions add ranges within that monthly grain — every weekly
-- partition's bounds will still permit only the 1st-of-month rows that pass
-- the existing CHECK. Rows for any non-first-of-month date are rejected by
-- the CHECK regardless of the partition layout.
--
-- !! IF the source data starts arriving at weekly grain, the
-- ``chk_cust_demand_month_start`` CHECK MUST be relaxed FIRST in a separate,
-- coordinated migration that also updates the loader and downstream MVs.
-- This migration only changes the partition layout, NOT the row grain. !!
--
-- Estimated impact (rough — measure on staging first):
--   * Each ``CREATE TABLE ... PARTITION OF`` takes a brief ACCESS EXCLUSIVE
--     lock on the parent. Empty-partition creation is fast (<100 ms each).
--   * Total wall-clock for 12 weekly partitions: typically 5–30 s.
--   * If the existing default partition holds rows, DETACH (Step 1) takes
--     longer. With ~5M rows expect 30–90 s.
--
-- COORDINATE WITH DBA. RUN DURING A MAINTENANCE WINDOW. PAUSE ETL FIRST.
--
-- -----------------------------------------------------------------------------
-- Strategy
-- -----------------------------------------------------------------------------
-- 1. KEEP every existing monthly partition.
-- 2. DETACH the default partition; drain or rename it.
-- 3. CREATE 12 weekly partitions (Mon–Sun, ISO numbering, naming
--    ``fact_customer_demand_monthly_YYYYwWW``).
-- 4. RE-ATTACH a fresh empty default.
-- 5. Update scripts/db/auto_create_partitions.py to set ``interval="week"``
--    for ``fact_customer_demand_monthly`` so the rolling window is weekly.
--
-- -----------------------------------------------------------------------------
-- Preflight
-- -----------------------------------------------------------------------------
--   * STOP customer-demand loader (`load_customer_demand_postgres.py`).
--   * Verify default partition is empty:
--       SELECT COUNT(*) FROM fact_customer_demand_monthly_default;
--   * Confirm the operator has decided whether to relax the
--     ``chk_cust_demand_month_start`` CHECK; if not, the weekly partitions
--     will only ever hold rows whose ``startdate`` is the 1st of the month
--     they cover (i.e. the first weekly partition of a calendar month).
--
-- -----------------------------------------------------------------------------
-- Rollback (manual — DO NOT WRITE; document only)
-- -----------------------------------------------------------------------------
--   1. ALTER TABLE fact_customer_demand_monthly DETACH PARTITION
--        fact_customer_demand_monthly_<weekly>;  -- repeat
--   2. DROP TABLE fact_customer_demand_monthly_<weekly>;
--   3. ALTER TABLE fact_customer_demand_monthly DETACH PARTITION
--        fact_customer_demand_monthly_default;
--      ALTER TABLE fact_customer_demand_monthly ATTACH PARTITION
--        fact_customer_demand_monthly_default DEFAULT;
--   4. Revert auto_create_partitions registry entry to ``interval="month"``.
--
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- Step 1: Detach the default partition.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM pg_inherits i
          JOIN pg_class c ON c.oid = i.inhrelid
         WHERE i.inhparent = 'fact_customer_demand_monthly'::regclass
           AND c.relname   = 'fact_customer_demand_monthly_default'
    ) THEN
        EXECUTE 'ALTER TABLE fact_customer_demand_monthly '
             || 'DETACH PARTITION fact_customer_demand_monthly_default';
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- Step 2: Create 12 weekly partitions.
-- -----------------------------------------------------------------------------
-- !! FILL IN THESE BOUNDS BEFORE RUNNING. !!
--
-- Rules:
--   * FROM bound MUST be a Monday (ISO weekday 1).
--   * FROM bound MUST be >= the last monthly partition's TO bound.
--   * TO bound = FROM bound + 7 days.
--   * Naming: fact_customer_demand_monthly_YYYYwWW (ISO-8601).

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_1>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_1>') TO (DATE '<YYYY-MM-DD_mon_2>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_2>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_2>') TO (DATE '<YYYY-MM-DD_mon_3>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_3>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_3>') TO (DATE '<YYYY-MM-DD_mon_4>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_4>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_4>') TO (DATE '<YYYY-MM-DD_mon_5>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_5>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_5>') TO (DATE '<YYYY-MM-DD_mon_6>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_6>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_6>') TO (DATE '<YYYY-MM-DD_mon_7>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_7>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_7>') TO (DATE '<YYYY-MM-DD_mon_8>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_8>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_8>') TO (DATE '<YYYY-MM-DD_mon_9>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_9>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_9>') TO (DATE '<YYYY-MM-DD_mon_10>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_10>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_10>') TO (DATE '<YYYY-MM-DD_mon_11>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_11>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_11>') TO (DATE '<YYYY-MM-DD_mon_12>');

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_<YYYYwWW_12>
    PARTITION OF fact_customer_demand_monthly
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_12>') TO (DATE '<YYYY-MM-DD_mon_13>');

-- -----------------------------------------------------------------------------
-- Step 3: Re-attach a fresh default partition.
-- -----------------------------------------------------------------------------
ALTER TABLE IF EXISTS fact_customer_demand_monthly_default
    RENAME TO fact_customer_demand_monthly_default_premigration;

CREATE TABLE IF NOT EXISTS fact_customer_demand_monthly_default
    PARTITION OF fact_customer_demand_monthly DEFAULT;

-- -----------------------------------------------------------------------------
-- Step 4: Post-migration steps (manual)
-- -----------------------------------------------------------------------------
--   1. Inspect fact_customer_demand_monthly_default_premigration for stragglers.
--   2. Update scripts/db/auto_create_partitions.py — change
--      fact_customer_demand_monthly's registered ``interval`` from ``"month"``
--      to ``"week"``. Run ``make auto-create-partitions`` to extend the window.
--   3. Re-enable the customer-demand loader.
--   4. Verify with:
--        SELECT relname, pg_size_pretty(pg_total_relation_size(c.oid)) AS size
--          FROM pg_inherits i
--          JOIN pg_class c ON c.oid = i.inhrelid
--         WHERE i.inhparent = 'fact_customer_demand_monthly'::regclass
--         ORDER BY relname;

COMMIT;
