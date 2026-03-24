-- IPfeature3: Safety Stock Engine
-- Creates the full fact_safety_stock_targets table.
--
-- NOTE: sql/026_create_inventory_health_score.sql created a minimal stub of this table
-- with only a few columns so that mv_inventory_health_score could be created before
-- IPfeature3 was implemented. This DDL uses CREATE TABLE IF NOT EXISTS — if the stub
-- already exists with fewer columns, the table structure stays as-is. The compute script
-- (scripts/compute_safety_stock.py) will upsert data using ON CONFLICT on (item_id, loc,
-- policy_version) once you run: make ss-schema ss-compute
--
-- If you need the full column set, drop the stub first:
--   DROP TABLE IF EXISTS fact_safety_stock_targets CASCADE;
-- then re-run: make health-schema ss-schema
--
-- Table grain: one row per (item_id, loc, policy_version).
-- Populated by scripts/compute_safety_stock.py on each SS refresh run.

-- Migrate stub table created by 026_create_inventory_health_score.sql:
-- Add any columns that are missing from the stub (idempotent).
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'fact_safety_stock_targets') THEN
        -- Add primary key surrogate if missing
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'ss_sk') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN ss_sk BIGSERIAL PRIMARY KEY;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'ss_ck') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN ss_ck TEXT UNIQUE;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'effective_date') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN effective_date DATE;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'service_level_target') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN service_level_target NUMERIC(6,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'z_score') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN z_score NUMERIC(8,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'demand_mean_monthly') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN demand_mean_monthly NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'demand_std_monthly') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN demand_std_monthly NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'lead_time_mean_days') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN lead_time_mean_days NUMERIC(10,2);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'lead_time_std_days') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN lead_time_std_days NUMERIC(10,2);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'abc_vol') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN abc_vol TEXT;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'ss_demand_only') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN ss_demand_only NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'ss_lt_only') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN ss_lt_only NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'ss_method') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN ss_method TEXT NOT NULL DEFAULT 'combined';
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'avg_daily_demand') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN avg_daily_demand NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'demand_cv') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN demand_cv NUMERIC(10,6);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'lt_mean_days') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN lt_mean_days NUMERIC(10,2);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'lt_std_days') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN lt_std_days NUMERIC(10,2);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'target_min_qty') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN target_min_qty NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'target_max_qty') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN target_max_qty NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'current_qty_on_hand') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN current_qty_on_hand NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'current_dos') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN current_dos NUMERIC(10,2);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'ss_gap') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN ss_gap NUMERIC(15,4);
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'computed_at') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN computed_at TIMESTAMPTZ DEFAULT NOW();
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'load_ts') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN load_ts TIMESTAMPTZ DEFAULT NOW();
        END IF;
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'fact_safety_stock_targets' AND column_name = 'modified_ts') THEN
            ALTER TABLE fact_safety_stock_targets ADD COLUMN modified_ts TIMESTAMPTZ DEFAULT NOW();
        END IF;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS fact_safety_stock_targets (
    -- Surrogate / composite keys
    ss_sk                  BIGSERIAL PRIMARY KEY,
    ss_ck                  TEXT UNIQUE,            -- item_id || '_' || loc || '_' || policy_version

    -- Business key
    item_id                TEXT        NOT NULL,
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
    ON fact_safety_stock_targets (item_id, loc, policy_version);

-- Item / location lookups
CREATE INDEX IF NOT EXISTS idx_ss_targets_item
    ON fact_safety_stock_targets (item_id);

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
