-- F4.1 — Financial Inventory Plan (Budget vs. Actuals)

CREATE TABLE IF NOT EXISTS dim_item_cost (
    id                  BIGSERIAL       PRIMARY KEY,
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL DEFAULT '',
    unit_cost           NUMERIC(12,4)   NOT NULL,
    cost_type           VARCHAR(30)     NOT NULL DEFAULT 'standard',  -- 'standard' | 'moving_avg' | 'last_purchase'
    currency            CHAR(3)         NOT NULL DEFAULT 'USD',
    effective_from      DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to        DATE,
    load_ts             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_item_cost_item_loc_from
    ON dim_item_cost (item_id, loc, effective_from);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_budget_periods (
    budget_id           BIGSERIAL       PRIMARY KEY,
    scope_type          VARCHAR(30)     NOT NULL,   -- 'global' | 'category' | 'buyer' | 'location'
    scope_value         VARCHAR(100)    NOT NULL,
    period_type         VARCHAR(20)     NOT NULL DEFAULT 'monthly',
    budget_start        DATE            NOT NULL,
    budget_end          DATE            NOT NULL,
    budget_cap          NUMERIC(16,2)   NOT NULL,
    carrying_cost_pct   NUMERIC(5,4)    NOT NULL DEFAULT 0.25,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_budget_periods_dates
    ON fact_budget_periods (budget_start, budget_end);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_financial_inventory_plan (
    id                          BIGSERIAL       PRIMARY KEY,
    item_id                     VARCHAR(50)     NOT NULL,
    loc                         VARCHAR(50)     NOT NULL,
    plan_month                  DATE            NOT NULL,
    plan_version                VARCHAR(50)     NOT NULL DEFAULT 'latest',
    projected_inventory_value   NUMERIC(16,2),
    planned_order_value         NUMERIC(16,2),
    carrying_cost_monthly       NUMERIC(16,2),
    excess_qty                  NUMERIC(12,2)   NOT NULL DEFAULT 0,
    excess_value                NUMERIC(16,2)   NOT NULL DEFAULT 0,
    max_stock_target            NUMERIC(12,2),
    budget_cap                  NUMERIC(16,2),
    within_budget               BOOLEAN,
    computed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_fin_plan UNIQUE (item_id, loc, plan_month, plan_version)
);

CREATE INDEX IF NOT EXISTS idx_fin_plan_item_loc_month
    ON fact_financial_inventory_plan (item_id, loc, plan_month);

CREATE INDEX IF NOT EXISTS idx_fin_plan_excess
    ON fact_financial_inventory_plan (excess_value DESC)
    WHERE excess_value > 0;
