-- Inventory snapshot fact table.
-- Grain: item_no + loc + snapshot_date (one row per item-location-day).
-- Source CSV columns: exec_date, item, loc, lead_time, tot_oh, tot_oh_oo, mtd_sls
-- qty_on_order is derived during load as (qty_on_hand_on_order - qty_on_hand).

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS fact_inventory_snapshot (
  inventory_sk          BIGSERIAL PRIMARY KEY,
  inventory_ck          TEXT UNIQUE NOT NULL,
  item_no               TEXT NOT NULL,
  loc                   TEXT NOT NULL,
  snapshot_date         DATE NOT NULL,
  lead_time_days        NUMERIC(10,2),
  qty_on_hand           NUMERIC(15,4),
  qty_on_hand_on_order  NUMERIC(15,4),
  qty_on_order          NUMERIC(15,4),
  mtd_sales             NUMERIC(15,4),
  load_ts               TIMESTAMPTZ DEFAULT NOW(),
  modified_ts           TIMESTAMPTZ DEFAULT NOW()
);

-- B-tree indexes for exact filters and sort paths.
CREATE INDEX IF NOT EXISTS idx_fact_inventory_snapshot_item_no
  ON fact_inventory_snapshot (item_no);
CREATE INDEX IF NOT EXISTS idx_fact_inventory_snapshot_loc
  ON fact_inventory_snapshot (loc);
CREATE INDEX IF NOT EXISTS idx_fact_inventory_snapshot_snapshot_date
  ON fact_inventory_snapshot (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_fact_inventory_snapshot_item_loc_date
  ON fact_inventory_snapshot (item_no, loc, snapshot_date);

-- GIN trigram indexes for ILIKE substring search on text columns.
CREATE INDEX IF NOT EXISTS idx_fact_inventory_snapshot_item_no_trgm
  ON fact_inventory_snapshot USING gin (item_no gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_inventory_snapshot_loc_trgm
  ON fact_inventory_snapshot USING gin (loc gin_trgm_ops);

-- Drop old view so we can rebuild with enhanced columns.
DROP MATERIALIZED VIEW IF EXISTS agg_inventory_monthly CASCADE;

-- Monthly aggregate materialized view with daily sales derivation.
-- daily_sls CTE derives actual daily sales from cumulative mtd_sales via LAG().
-- Day 1 of each month: daily_sls = mtd_sales (counter just reset).
-- Subsequent days: daily_sls = mtd_sales - prior day's mtd_sales.
CREATE MATERIALIZED VIEW agg_inventory_monthly AS
WITH daily AS (
    SELECT
        item_no,
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
                PARTITION BY item_no, loc, date_trunc('month', snapshot_date)
                ORDER BY snapshot_date
            )
        END AS daily_sls
    FROM fact_inventory_snapshot
)
SELECT
    date_trunc('month', snapshot_date)::date AS month_start,
    item_no,
    loc,
    -- Averages over the month (smoothed position)
    COALESCE(AVG(qty_on_hand), 0)::double precision               AS avg_qty_on_hand,
    COALESCE(AVG(qty_on_hand_on_order), 0)::double precision       AS avg_qty_on_hand_on_order,
    -- End-of-month snapshot (point-in-time position)
    (ARRAY_AGG(qty_on_hand ORDER BY snapshot_date DESC))[1]::double precision     AS eom_qty_on_hand,
    (ARRAY_AGG(qty_on_hand_on_order ORDER BY snapshot_date DESC))[1]::double precision AS eom_qty_on_hand_on_order,
    -- Monthly sales = MAX(mtd_sales) = last cumulative value (NOT SUM)
    COALESCE(MAX(mtd_sales), 0)::double precision                  AS monthly_sales,
    -- Average daily sales rate for DOS/WOC (excludes zero-demand days)
    COALESCE(AVG(NULLIF(daily_sls, 0)), 0)::double precision       AS avg_daily_sls,
    -- Snapshot count for partial month handling
    COUNT(*)::integer                                               AS snapshot_days,
    -- Lead time: last known value per month (not average)
    (ARRAY_AGG(lead_time_days ORDER BY snapshot_date DESC))[1]::double precision AS latest_lead_time_days
FROM daily
GROUP BY 1, 2, 3
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agg_inventory_monthly_pk
  ON agg_inventory_monthly (item_no, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_agg_inventory_monthly_month
  ON agg_inventory_monthly (month_start);
