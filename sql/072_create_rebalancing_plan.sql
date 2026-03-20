-- Inventory Rebalancing: Plan + Transfer Recommendations
-- Stores the output of the rebalancing optimization engine.

-- Plan header: one row per computation run
CREATE TABLE IF NOT EXISTS fact_rebalancing_plan (
    plan_sk                    BIGSERIAL PRIMARY KEY,
    plan_id                    TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
    computation_date           DATE NOT NULL,
    horizon_weeks              INTEGER NOT NULL DEFAULT 4,
    solver_method              TEXT NOT NULL DEFAULT 'greedy',
    objective                  TEXT NOT NULL DEFAULT 'min_cost',
    -- Plan-level KPIs
    total_transfer_qty         NUMERIC(15,2),
    total_transfer_cost        NUMERIC(12,2),
    total_avoided_stockout_value NUMERIC(12,2),
    net_roi                    NUMERIC(10,4),
    network_balance_before     NUMERIC(6,4),
    network_balance_after      NUMERIC(6,4),
    items_rebalanced           INTEGER,
    lanes_used                 INTEGER,
    -- Workflow status
    status                     TEXT NOT NULL DEFAULT 'draft'
                               CHECK (status IN ('draft','pending_approval','approved',
                                                  'partially_approved','executing',
                                                  'completed','cancelled')),
    approved_by                TEXT,
    approved_ts                TIMESTAMPTZ,
    -- Metadata
    solver_runtime_ms          INTEGER,
    created_ts                 TIMESTAMPTZ DEFAULT NOW(),
    modified_ts                TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rebal_plan_date   ON fact_rebalancing_plan (computation_date DESC);
CREATE INDEX IF NOT EXISTS idx_rebal_plan_status ON fact_rebalancing_plan (status);

-- Individual transfer recommendations within a plan
CREATE TABLE IF NOT EXISTS fact_rebalancing_transfer (
    transfer_sk                BIGSERIAL PRIMARY KEY,
    transfer_id                TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
    plan_id                    TEXT NOT NULL REFERENCES fact_rebalancing_plan(plan_id),
    item_no                    TEXT NOT NULL,
    source_loc                 TEXT NOT NULL,
    dest_loc                   TEXT NOT NULL,
    lane_id                    TEXT,
    transfer_mode              TEXT DEFAULT 'truck',
    -- Quantities
    recommended_qty            NUMERIC(15,4) NOT NULL,
    approved_qty               NUMERIC(15,4),
    -- Source context (at computation time)
    source_on_hand             NUMERIC(15,4),
    source_dos                 NUMERIC(10,2),
    source_ss_target           NUMERIC(15,4),
    source_excess_qty          NUMERIC(15,4),
    -- Destination context
    dest_on_hand               NUMERIC(15,4),
    dest_dos                   NUMERIC(10,2),
    dest_ss_target             NUMERIC(15,4),
    dest_shortage_qty          NUMERIC(15,4),
    -- Financial
    transfer_cost              NUMERIC(12,2),
    carrying_cost_saved        NUMERIC(12,2),
    stockout_cost_avoided      NUMERIC(12,2),
    net_benefit                NUMERIC(12,2),
    roi                        NUMERIC(10,4),
    -- Scheduling
    planned_ship_date          DATE,
    expected_arrival_date      DATE,
    transfer_lt_days           INTEGER,
    -- Priority
    priority_score             NUMERIC(10,4),
    abc_class                  TEXT,
    urgency                    TEXT CHECK (urgency IN ('critical','high','medium','low')),
    -- Workflow
    status                     TEXT NOT NULL DEFAULT 'recommended'
                               CHECK (status IN ('recommended','approved','rejected',
                                                  'hold','in_transit','received','cancelled')),
    approved_by                TEXT,
    approved_ts                TIMESTAMPTZ,
    rejection_reason           TEXT,
    notes                      TEXT,
    -- Metadata
    created_ts                 TIMESTAMPTZ DEFAULT NOW(),
    modified_ts                TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rebal_transfer_plan    ON fact_rebalancing_transfer (plan_id);
CREATE INDEX IF NOT EXISTS idx_rebal_transfer_item    ON fact_rebalancing_transfer (item_no);
CREATE INDEX IF NOT EXISTS idx_rebal_transfer_source  ON fact_rebalancing_transfer (source_loc);
CREATE INDEX IF NOT EXISTS idx_rebal_transfer_dest    ON fact_rebalancing_transfer (dest_loc);
CREATE INDEX IF NOT EXISTS idx_rebal_transfer_status  ON fact_rebalancing_transfer (status);
CREATE INDEX IF NOT EXISTS idx_rebal_transfer_urgency ON fact_rebalancing_transfer (urgency, priority_score DESC)
    WHERE status = 'recommended';
