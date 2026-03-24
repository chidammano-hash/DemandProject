-- IPfeature13: Capital & Space Investment Optimization
-- Tables: fact_inventory_investment_plan + fact_efficient_frontier

CREATE TABLE IF NOT EXISTS fact_inventory_investment_plan (
    plan_sk                BIGSERIAL PRIMARY KEY,
    plan_id                TEXT NOT NULL,
    item_id                TEXT NOT NULL,
    loc                    TEXT NOT NULL,
    computation_date       DATE NOT NULL,
    -- Current state
    current_ss_qty         NUMERIC(15,4),
    current_ss_value       NUMERIC(12,2),
    current_csl            NUMERIC(6,4),
    -- Recommended state
    recommended_ss_qty     NUMERIC(15,4),
    recommended_ss_value   NUMERIC(12,2),
    recommended_csl        NUMERIC(6,4),
    -- Incremental analysis
    ss_increment_qty       NUMERIC(15,4),
    investment_increment   NUMERIC(12,2),
    csl_increment          NUMERIC(6,4),
    marginal_roi           NUMERIC(10,4),
    -- Ranking
    investment_rank        INTEGER,
    cumulative_investment  NUMERIC(15,2),
    -- Metadata
    abc_vol                TEXT,
    abc_xyz_segment        TEXT,
    unit_cost              NUMERIC(12,4),
    created_ts             TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (plan_id, item_id, loc)
);

CREATE INDEX IF NOT EXISTS idx_inv_plan_plan_id
    ON fact_inventory_investment_plan (plan_id, investment_rank);
CREATE INDEX IF NOT EXISTS idx_inv_plan_item_loc
    ON fact_inventory_investment_plan (item_id, loc);
CREATE INDEX IF NOT EXISTS idx_inv_plan_marginal_roi
    ON fact_inventory_investment_plan (marginal_roi DESC);

-- Efficient frontier: pre-computed budget-to-CSL mapping for the plan
CREATE TABLE IF NOT EXISTS fact_efficient_frontier (
    frontier_sk       BIGSERIAL PRIMARY KEY,
    plan_id           TEXT NOT NULL,
    budget_point      NUMERIC(15,2),
    items_funded      INTEGER,
    achievable_csl    NUMERIC(6,4),
    marginal_item     TEXT,
    created_ts        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_frontier_plan ON fact_efficient_frontier (plan_id, budget_point);
