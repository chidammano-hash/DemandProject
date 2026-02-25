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

-- Monthly aggregate materialized view for trend analytics.
CREATE MATERIALIZED VIEW IF NOT EXISTS agg_inventory_monthly AS
SELECT
  date_trunc('month', snapshot_date)::date AS month_start,
  item_no,
  loc,
  coalesce(avg(qty_on_hand), 0)::double precision       AS avg_qty_on_hand,
  coalesce(avg(qty_on_hand_on_order), 0)::double precision AS avg_qty_on_hand_on_order,
  coalesce(avg(lead_time_days), 0)::double precision     AS avg_lead_time_days,
  coalesce(sum(mtd_sales), 0)::double precision          AS total_mtd_sales,
  count(*)::bigint                                       AS snapshot_count
FROM fact_inventory_snapshot
GROUP BY 1, 2, 3
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agg_inventory_monthly_pk
  ON agg_inventory_monthly (item_no, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_agg_inventory_monthly_month
  ON agg_inventory_monthly (month_start);
