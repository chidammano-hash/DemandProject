-- =============================================================================
-- Migration: fact_inventory_snapshot — monthly → weekly partitioning cutover
-- =============================================================================
--
-- !! REVIEW BEFORE RUN — schema cutover. !!
--
-- This migration adds NEW weekly partitions to the existing monthly-partitioned
-- ``fact_inventory_snapshot`` parent. Historical monthly partitions (sql/088)
-- are LEFT IN PLACE — they are populated, indexed, and don't need to move.
--
-- Estimated impact (rough — measure on staging first):
--   * Each ``CREATE TABLE ... PARTITION OF`` takes a brief ACCESS EXCLUSIVE
--     lock on ``fact_inventory_snapshot`` (the parent). Empty-partition creation
--     is fast (<100 ms each), but the lock blocks concurrent INSERT/SELECT.
--   * Total wall-clock for 12 weekly partitions: typically 5–30 s.
--   * If the existing default partition holds rows, DETACH (Step 4) takes
--     longer — Postgres must scan the default to verify no overlap. With ~30M
--     rows in the default, expect 2–5 minutes per detach. **Drain the default
--     before running this migration** (see preflight notes).
--
-- COORDINATE WITH DBA. RUN DURING A MAINTENANCE WINDOW. PAUSE ETL FIRST.
--
-- -----------------------------------------------------------------------------
-- Strategy
-- -----------------------------------------------------------------------------
-- 1. KEEP every existing monthly partition (``fact_inventory_snapshot_YYYY_MM``).
--    They cover historical and current-month data; the planner prunes them
--    correctly for queries that fall in their month range.
--
-- 2. DETACH the existing DEFAULT partition (``fact_inventory_snapshot_default``)
--    if present. Postgres requires the default to be removed before adding
--    NEW partitions whose ranges might overlap any rows currently in the
--    default. We re-attach a fresh empty default at the end so out-of-range
--    rows still land somewhere.
--
-- 3. CREATE 12 weekly partitions covering the 12 ISO weeks AFTER the last
--    fully-populated monthly partition. Bounds are Mon 00:00 → next Mon 00:00.
--    Naming: ``fact_inventory_snapshot_YYYYwWW`` (ISO-8601 week numbering).
--    The exact start week is left to the operator — fill in the placeholders
--    below before running.
--
-- 4. RE-ATTACH a fresh DEFAULT partition. After cutover, the registry in
--    ``scripts/db/auto_create_partitions.py`` should be updated to set
--    ``interval="week"`` for ``fact_inventory_snapshot`` so the rolling
--    window provisions weekly going forward.
--
-- 5. Partition pruning will continue to work correctly because monthly and
--    weekly partitions describe non-overlapping date ranges. The planner
--    picks the right partition based on the query's date predicate.
--
-- -----------------------------------------------------------------------------
-- Preflight
-- -----------------------------------------------------------------------------
--   * STOP all ETL writers to fact_inventory_snapshot (the inventory loader,
--     any backfill script).
--   * Verify the default partition is empty (or drain it):
--       SELECT COUNT(*) FROM fact_inventory_snapshot_default;
--     If non-zero, either redirect those rows into a new monthly partition
--     covering their snapshot_date, or accept that the DETACH (Step 4) will
--     scan them.
--   * Check pg_locks for long-running transactions on the parent:
--       SELECT pid, mode, granted FROM pg_locks
--        WHERE relation = 'fact_inventory_snapshot'::regclass;
--   * Ensure the latest monthly partition extends through the END of the
--     calendar month BEFORE the first weekly partition. Otherwise there is a
--     gap, and INSERTs for the gap dates will hit the (now-empty) default.
--
-- -----------------------------------------------------------------------------
-- Rollback (manual — DO NOT WRITE; document only)
-- -----------------------------------------------------------------------------
--   1. ALTER TABLE fact_inventory_snapshot DETACH PARTITION
--        fact_inventory_snapshot_<weekly>;  -- repeat for each weekly partition
--   2. DROP TABLE fact_inventory_snapshot_<weekly>;  -- if not needed
--   3. ALTER TABLE fact_inventory_snapshot DETACH PARTITION
--        fact_inventory_snapshot_default;
--      ALTER TABLE fact_inventory_snapshot ATTACH PARTITION
--        fact_inventory_snapshot_default DEFAULT;
--   4. Revert the auto_create_partitions registry entry to ``interval="month"``.
--
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- Step 1: Detach the default partition.
-- -----------------------------------------------------------------------------
-- Wrapped in DO block so re-runs after a partial failure don't error if the
-- default is already detached.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM pg_inherits i
          JOIN pg_class c ON c.oid = i.inhrelid
         WHERE i.inhparent = 'fact_inventory_snapshot'::regclass
           AND c.relname   = 'fact_inventory_snapshot_default'
    ) THEN
        EXECUTE 'ALTER TABLE fact_inventory_snapshot '
             || 'DETACH PARTITION fact_inventory_snapshot_default';
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- Step 2: Create 12 weekly partitions.
-- -----------------------------------------------------------------------------
-- !! FILL IN THESE BOUNDS BEFORE RUNNING. !!
--
-- Rules:
--   * Each FROM bound MUST be a Monday (ISO weekday 1).
--   * Each FROM bound MUST be >= the last monthly partition's TO bound, so
--     there is no overlap.
--   * Each TO bound = FROM bound + 7 days.
--   * Naming: fact_inventory_snapshot_YYYYwWW — ISO-8601 week numbering
--     (NOT the strftime %U/%W tokens, which use Sunday).
--
-- Example, assuming last monthly partition is fact_inventory_snapshot_2027_06
-- with TO bound 2027-07-01 (a Thursday). The first usable Monday is 2027-07-05
-- (ISO week 27 of 2027). Backfill the 4 days 2027-07-01..04 with a manual
-- partition or accept they sit in the (re-attached) default.

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_1>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_1>') TO (DATE '<YYYY-MM-DD_mon_2>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_2>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_2>') TO (DATE '<YYYY-MM-DD_mon_3>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_3>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_3>') TO (DATE '<YYYY-MM-DD_mon_4>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_4>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_4>') TO (DATE '<YYYY-MM-DD_mon_5>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_5>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_5>') TO (DATE '<YYYY-MM-DD_mon_6>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_6>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_6>') TO (DATE '<YYYY-MM-DD_mon_7>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_7>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_7>') TO (DATE '<YYYY-MM-DD_mon_8>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_8>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_8>') TO (DATE '<YYYY-MM-DD_mon_9>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_9>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_9>') TO (DATE '<YYYY-MM-DD_mon_10>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_10>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_10>') TO (DATE '<YYYY-MM-DD_mon_11>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_11>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_11>') TO (DATE '<YYYY-MM-DD_mon_12>');

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_<YYYYwWW_12>
    PARTITION OF fact_inventory_snapshot
    FOR VALUES FROM (DATE '<YYYY-MM-DD_mon_12>') TO (DATE '<YYYY-MM-DD_mon_13>');

-- -----------------------------------------------------------------------------
-- Step 3: Re-attach a fresh default partition.
-- -----------------------------------------------------------------------------
-- The detached default still exists as a standalone table. We give it a
-- one-shot _premigration suffix and let the operator drop or merge later.
-- A new empty default takes its place.
ALTER TABLE IF EXISTS fact_inventory_snapshot_default
    RENAME TO fact_inventory_snapshot_default_premigration;

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot_default
    PARTITION OF fact_inventory_snapshot DEFAULT;

-- -----------------------------------------------------------------------------
-- Step 4: Post-migration steps (manual)
-- -----------------------------------------------------------------------------
--   1. Inspect fact_inventory_snapshot_default_premigration for orphaned rows.
--      If empty, DROP it. If non-empty, INSERT them into the appropriate
--      monthly or weekly partition.
--   2. Update scripts/db/auto_create_partitions.py — change
--      fact_inventory_snapshot's registered ``interval`` from ``"month"`` to
--      ``"week"``. Run ``make auto-create-partitions`` to extend the rolling
--      window.
--   3. Re-enable ETL writers.
--   4. Verify with:
--        SELECT relname, pg_size_pretty(pg_total_relation_size(c.oid)) AS size
--          FROM pg_inherits i
--          JOIN pg_class c ON c.oid = i.inhrelid
--         WHERE i.inhparent = 'fact_inventory_snapshot'::regclass
--         ORDER BY relname;

COMMIT;
