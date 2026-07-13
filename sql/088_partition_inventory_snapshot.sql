-- MIGRATION: SUPERSEDED
-- Migration: Convert fact_inventory_snapshot to monthly range partitioning.
-- Grain: item_id + loc + snapshot_date (one row per item-location-day).
-- Partition key: snapshot_date (DATE) — matches source file granularity.
--
-- IMPORTANT: Run this migration ONCE on existing databases.
-- New installs should use the updated 017_create_fact_inventory_snapshot.sql directly.
-- The migration runner records this file without executing it because the
-- current base schema already creates the partitioned table.

BEGIN;

-- Step 1: Rename existing table
ALTER TABLE IF EXISTS fact_inventory_snapshot RENAME TO fact_inventory_snapshot_old;

-- Drop dependent MVs (they reference the old table name)
DROP MATERIALIZED VIEW IF EXISTS agg_inventory_monthly CASCADE;

-- Step 2: Create partitioned parent table (no BIGSERIAL surrogate key)
CREATE TABLE fact_inventory_snapshot (
  inventory_ck          TEXT NOT NULL,
  item_id               TEXT NOT NULL,
  loc                   TEXT NOT NULL,
  snapshot_date         DATE NOT NULL,
  lead_time_days        NUMERIC(10,2),
  qty_on_hand           NUMERIC(15,4),
  qty_on_hand_on_order  NUMERIC(15,4),
  qty_on_order          NUMERIC(15,4),
  mtd_sales             NUMERIC(15,4),
  load_ts               TIMESTAMPTZ DEFAULT NOW(),
  modified_ts           TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT fact_inventory_snapshot_ck_uq UNIQUE (inventory_ck, snapshot_date)
) PARTITION BY RANGE (snapshot_date);

-- Step 3: Create monthly partitions (2024-12 through 2027-06 for headroom)
CREATE TABLE fact_inventory_snapshot_2024_12 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');
CREATE TABLE fact_inventory_snapshot_2025_01 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE fact_inventory_snapshot_2025_02 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE fact_inventory_snapshot_2025_03 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE fact_inventory_snapshot_2025_04 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE fact_inventory_snapshot_2025_05 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE fact_inventory_snapshot_2025_06 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE fact_inventory_snapshot_2025_07 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE fact_inventory_snapshot_2025_08 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE fact_inventory_snapshot_2025_09 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE fact_inventory_snapshot_2025_10 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE fact_inventory_snapshot_2025_11 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE fact_inventory_snapshot_2025_12 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
CREATE TABLE fact_inventory_snapshot_2026_01 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE fact_inventory_snapshot_2026_02 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE fact_inventory_snapshot_2026_03 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE fact_inventory_snapshot_2026_04 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE fact_inventory_snapshot_2026_05 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');
CREATE TABLE fact_inventory_snapshot_2026_06 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE fact_inventory_snapshot_2026_07 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');
CREATE TABLE fact_inventory_snapshot_2026_08 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
CREATE TABLE fact_inventory_snapshot_2026_09 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
CREATE TABLE fact_inventory_snapshot_2026_10 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
CREATE TABLE fact_inventory_snapshot_2026_11 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');
CREATE TABLE fact_inventory_snapshot_2026_12 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2026-12-01') TO ('2027-01-01');
CREATE TABLE fact_inventory_snapshot_2027_01 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2027-01-01') TO ('2027-02-01');
CREATE TABLE fact_inventory_snapshot_2027_02 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2027-02-01') TO ('2027-03-01');
CREATE TABLE fact_inventory_snapshot_2027_03 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2027-03-01') TO ('2027-04-01');
CREATE TABLE fact_inventory_snapshot_2027_04 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2027-04-01') TO ('2027-05-01');
CREATE TABLE fact_inventory_snapshot_2027_05 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2027-05-01') TO ('2027-06-01');
CREATE TABLE fact_inventory_snapshot_2027_06 PARTITION OF fact_inventory_snapshot
  FOR VALUES FROM ('2027-06-01') TO ('2027-07-01');

-- Default partition catches any out-of-range data
CREATE TABLE fact_inventory_snapshot_default PARTITION OF fact_inventory_snapshot DEFAULT;

-- Step 4: Create indexes on parent (auto-propagated to all partitions)
CREATE INDEX idx_fact_inventory_snapshot_item_id
  ON fact_inventory_snapshot (item_id);
CREATE INDEX idx_fact_inventory_snapshot_loc
  ON fact_inventory_snapshot (loc);
CREATE INDEX idx_fact_inventory_snapshot_snapshot_date
  ON fact_inventory_snapshot (snapshot_date);
CREATE INDEX idx_fact_inventory_snapshot_item_loc_date
  ON fact_inventory_snapshot (item_id, loc, snapshot_date);
CREATE INDEX idx_fact_inventory_snapshot_item_id_trgm
  ON fact_inventory_snapshot USING gin (item_id gin_trgm_ops);
CREATE INDEX idx_fact_inventory_snapshot_loc_trgm
  ON fact_inventory_snapshot USING gin (loc gin_trgm_ops);

-- Step 5: Migrate data from old table
INSERT INTO fact_inventory_snapshot (
  inventory_ck, item_id, loc, snapshot_date,
  lead_time_days, qty_on_hand, qty_on_hand_on_order, qty_on_order,
  mtd_sales, load_ts, modified_ts
)
SELECT
  inventory_ck, item_id, loc, snapshot_date,
  lead_time_days, qty_on_hand, qty_on_hand_on_order, qty_on_order,
  mtd_sales, load_ts, modified_ts
FROM fact_inventory_snapshot_old;

-- Step 6: Drop old table
DROP TABLE fact_inventory_snapshot_old;

-- Step 7: Rebuild agg_inventory_monthly MV (references partitioned parent transparently)
CREATE MATERIALIZED VIEW agg_inventory_monthly AS
WITH daily AS (
    SELECT
        item_id,
        loc,
        snapshot_date,
        lead_time_days,
        qty_on_hand,
        qty_on_hand_on_order,
        qty_on_order,
        mtd_sales,
        CASE
            WHEN EXTRACT(day FROM snapshot_date) = 1 THEN mtd_sales
            ELSE mtd_sales - LAG(mtd_sales) OVER (
                PARTITION BY item_id, loc, date_trunc('month', snapshot_date)
                ORDER BY snapshot_date
            )
        END AS daily_sls
    FROM fact_inventory_snapshot
)
SELECT
    date_trunc('month', snapshot_date)::date AS month_start,
    item_id,
    loc,
    COALESCE(AVG(qty_on_hand), 0)::double precision               AS avg_qty_on_hand,
    COALESCE(AVG(qty_on_hand_on_order), 0)::double precision       AS avg_qty_on_hand_on_order,
    (ARRAY_AGG(qty_on_hand ORDER BY snapshot_date DESC))[1]::double precision     AS eom_qty_on_hand,
    (ARRAY_AGG(qty_on_hand_on_order ORDER BY snapshot_date DESC))[1]::double precision AS eom_qty_on_hand_on_order,
    COALESCE(MAX(mtd_sales), 0)::double precision                  AS monthly_sales,
    COALESCE(AVG(NULLIF(daily_sls, 0)), 0)::double precision       AS avg_daily_sls,
    COUNT(*)::integer                                               AS snapshot_days,
    (ARRAY_AGG(lead_time_days ORDER BY snapshot_date DESC))[1]::double precision AS latest_lead_time_days
FROM daily
GROUP BY 1, 2, 3
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agg_inventory_monthly_pk
  ON agg_inventory_monthly (item_id, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_agg_inventory_monthly_month
  ON agg_inventory_monthly (month_start);

COMMIT;
