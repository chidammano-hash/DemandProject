-- IPfeature4: EOQ & Cycle Stock Calculator
-- Standalone table; extends with ss_combined when IPfeature3 is available.

CREATE TABLE IF NOT EXISTS fact_eoq_targets (
    id                   SERIAL          PRIMARY KEY,
    item_id              TEXT            NOT NULL,
    loc                  TEXT            NOT NULL,
    abc_vol              TEXT,

    -- Demand inputs (derived from dim_sku.demand_mean)
    demand_mean_monthly  NUMERIC(15, 4),
    annual_demand        NUMERIC(15, 4),

    -- Cost inputs (from config defaults)
    ordering_cost        NUMERIC(10, 2),
    holding_cost_pct     NUMERIC(6, 4),
    unit_cost            NUMERIC(10, 4),
    moq                  NUMERIC(10, 2),

    -- EOQ outputs
    eoq                  NUMERIC(15, 4),
    effective_eoq        NUMERIC(15, 4),     -- max(eoq, moq), capped at max_eoq_months_supply
    eoq_cycle_stock      NUMERIC(15, 4),     -- effective_eoq / 2
    order_frequency      NUMERIC(10, 4),     -- annual_demand / effective_eoq

    -- Cost metrics
    annual_holding_cost  NUMERIC(15, 4),
    annual_order_cost    NUMERIC(15, 4),
    total_annual_cost    NUMERIC(15, 4),

    computed_at          TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE (item_id, loc)
);

CREATE INDEX IF NOT EXISTS idx_fact_eoq_targets_abc_vol
    ON fact_eoq_targets (abc_vol);

CREATE INDEX IF NOT EXISTS idx_fact_eoq_targets_item_id
    ON fact_eoq_targets (item_id);
