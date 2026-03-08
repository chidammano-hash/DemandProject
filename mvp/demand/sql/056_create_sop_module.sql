-- F4.2 — Sales & Operations Planning (S&OP) Module

CREATE TABLE IF NOT EXISTS fact_sop_cycles (
    cycle_id                BIGSERIAL       PRIMARY KEY,
    cycle_month             DATE            NOT NULL UNIQUE,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'demand_review',
    demand_plan_version     VARCHAR(50),
    supply_plan_version     VARCHAR(50),
    approved_plan_version   VARCHAR(50),
    facilitated_by          VARCHAR(100),
    approved_by             VARCHAR(100),
    demand_review_at        TIMESTAMPTZ,
    supply_review_at        TIMESTAMPTZ,
    pre_sop_at              TIMESTAMPTZ,
    executive_sop_at        TIMESTAMPTZ,
    approved_at             TIMESTAMPTZ,
    run_by                  VARCHAR(100),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sop_cycles_month
    ON fact_sop_cycles (cycle_month DESC);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_sop_demand_review (
    id                      BIGSERIAL       PRIMARY KEY,
    cycle_id                BIGINT          NOT NULL REFERENCES fact_sop_cycles(cycle_id),
    item_category           VARCHAR(100)    NOT NULL,
    statistical_demand_qty  NUMERIC(14,2),
    commercial_demand_qty   NUMERIC(14,2),
    consensus_demand_qty    NUMERIC(14,2),
    statistical_demand_val  NUMERIC(16,2),
    commercial_demand_val   NUMERIC(16,2),
    consensus_demand_val    NUMERIC(16,2),
    review_status           VARCHAR(20)     NOT NULL DEFAULT 'pending',
    CONSTRAINT uq_sop_demand UNIQUE (cycle_id, item_category)
);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_sop_supply_constraints (
    constraint_id           BIGSERIAL       PRIMARY KEY,
    cycle_id                BIGINT          NOT NULL REFERENCES fact_sop_cycles(cycle_id),
    constraint_type         VARCHAR(50)     NOT NULL,
    supplier_id             VARCHAR(50),
    impact_qty              NUMERIC(14,2),
    impact_period           DATE,
    mitigation_status       VARCHAR(30)     NOT NULL DEFAULT 'open'
);

CREATE INDEX IF NOT EXISTS idx_sop_constraints_cycle
    ON fact_sop_supply_constraints (cycle_id, mitigation_status);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_sop_gaps (
    gap_id                  BIGSERIAL       PRIMARY KEY,
    cycle_id                BIGINT          NOT NULL REFERENCES fact_sop_cycles(cycle_id),
    gap_type                VARCHAR(50)     NOT NULL,
    gap_qty                 NUMERIC(14,2),
    gap_value               NUMERIC(16,2),
    severity                VARCHAR(20)     NOT NULL DEFAULT 'medium',  -- 'critical' | 'high' | 'medium' | 'low'
    resolution_options      JSONB,
    resolution_status       VARCHAR(20)     NOT NULL DEFAULT 'open'
);

CREATE INDEX IF NOT EXISTS idx_sop_gaps_cycle
    ON fact_sop_gaps (cycle_id, severity, resolution_status);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_sop_approved_plan (
    id                  BIGSERIAL       PRIMARY KEY,
    cycle_id            BIGINT          NOT NULL REFERENCES fact_sop_cycles(cycle_id),
    item_no             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    plan_month          DATE            NOT NULL,
    approved_qty        NUMERIC(12,2)   NOT NULL,
    statistical_qty     NUMERIC(12,2),
    override_qty        NUMERIC(12,2),
    source              VARCHAR(30)     NOT NULL DEFAULT 'consensus',
    locked              BOOLEAN         NOT NULL DEFAULT TRUE,
    CONSTRAINT uq_sop_approved UNIQUE (cycle_id, item_no, loc, plan_month)
);

CREATE INDEX IF NOT EXISTS idx_sop_approved_item_loc
    ON fact_sop_approved_plan (item_no, loc, plan_month);
