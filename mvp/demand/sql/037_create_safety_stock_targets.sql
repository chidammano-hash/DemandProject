-- IPfeature3: Safety Stock Engine
-- Creates the full fact_safety_stock_targets table.
--
-- NOTE: sql/026_create_inventory_health_score.sql created a minimal stub of this table
-- with only a few columns so that mv_inventory_health_score could be created before
-- IPfeature3 was implemented. This DDL uses CREATE TABLE IF NOT EXISTS — if the stub
-- already exists with fewer columns, the table structure stays as-is. The compute script
-- (scripts/compute_safety_stock.py) will upsert data using ON CONFLICT on (item_no, loc,
-- policy_version) once you run: make ss-schema ss-compute
--
-- If you need the full column set, drop the stub first:
--   DROP TABLE IF EXISTS fact_safety_stock_targets CASCADE;
-- then re-run: make health-schema ss-schema
--
-- Table grain: one row per (item_no, loc, policy_version).
-- Populated by scripts/compute_safety_stock.py on each SS refresh run.

CREATE TABLE IF NOT EXISTS fact_safety_stock_targets (
    -- Surrogate / composite keys
    ss_sk                  BIGSERIAL PRIMARY KEY,
    ss_ck                  TEXT UNIQUE,            -- item_no || '_' || loc || '_' || policy_version

    -- Business key
    item_no                TEXT        NOT NULL,
    loc                    TEXT        NOT NULL,
    policy_version         TEXT        NOT NULL    DEFAULT 'v1',

    -- Effective date of this computation
    effective_date         DATE,

    -- Inputs recorded for auditability
    service_level_target   NUMERIC(6,4),           -- e.g. 0.95
    z_score                NUMERIC(8,4),           -- e.g. 1.645
    demand_mean_monthly    NUMERIC(15,4),          -- avg monthly demand (units)
    demand_std_monthly     NUMERIC(15,4),          -- std dev of monthly demand
    lead_time_mean_days    NUMERIC(10,2),          -- mean lead time in days
    lead_time_std_days     NUMERIC(10,2),          -- std dev of lead time in days

    -- ABC classification (snapshot at compute time)
    abc_vol                TEXT,

    -- Safety stock components
    ss_demand_only         NUMERIC(15,4),          -- demand variability component only
    ss_lt_only             NUMERIC(15,4),          -- lead time variability component only
    ss_combined            NUMERIC(15,4),          -- recommended SS (combined formula)
    ss_method              TEXT        NOT NULL    DEFAULT 'combined',  -- 'combined' | 'demand_only' | 'manual'

    -- Derived targets
    avg_daily_demand       NUMERIC(15,4),          -- demand_mean_monthly / 30.44
    demand_cv              NUMERIC(10,6),          -- demand_std / demand_mean (CV)
    lt_mean_days           NUMERIC(10,2),          -- alias of lead_time_mean_days
    lt_std_days            NUMERIC(10,2),          -- alias of lead_time_std_days

    reorder_point          NUMERIC(15,4),          -- = avg_daily_demand * lt_mean_days + ss_combined
    target_min_qty         NUMERIC(15,4),          -- = ss_combined
    target_max_qty         NUMERIC(15,4),          -- updated by IPfeature4 (EOQ/2 + SS)
    target_dos_min         NUMERIC(10,2),          -- ss_combined / avg_daily_demand  (days)
    target_dos_max         NUMERIC(10,2),          -- (ss_combined + eoq/2) / avg_daily_demand (set by IPfeature4)

    -- Current position comparison (refreshed on each run)
    current_qty_on_hand    NUMERIC(15,4),
    current_dos            NUMERIC(10,2),          -- eom_qty / avg_daily_demand
    ss_coverage            NUMERIC(10,4),          -- current_qty / ss_combined
    ss_gap                 NUMERIC(15,4),          -- current_qty - ss_combined  (negative = shortfall)
    is_below_ss            BOOLEAN,

    -- Audit
    computed_at            TIMESTAMPTZ             DEFAULT NOW(),
    load_ts                TIMESTAMPTZ             DEFAULT NOW(),
    modified_ts            TIMESTAMPTZ             DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- Primary business key uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS idx_ss_targets_bk
    ON fact_safety_stock_targets (item_no, loc, policy_version);

-- Item / location lookups
CREATE INDEX IF NOT EXISTS idx_ss_targets_item
    ON fact_safety_stock_targets (item_no);

CREATE INDEX IF NOT EXISTS idx_ss_targets_loc
    ON fact_safety_stock_targets (loc);

-- Fast filter for below-SS items (Exception Queue, Health Score)
CREATE INDEX IF NOT EXISTS idx_ss_targets_below_ss
    ON fact_safety_stock_targets (is_below_ss)
    WHERE is_below_ss = TRUE;

-- ABC class filter
CREATE INDEX IF NOT EXISTS idx_ss_targets_abc
    ON fact_safety_stock_targets (abc_vol);

-- Policy version filter
CREATE INDEX IF NOT EXISTS idx_ss_targets_policy_version
    ON fact_safety_stock_targets (policy_version);
