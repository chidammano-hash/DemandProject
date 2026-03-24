-- F4.4 — What-If Scenario Planning for Supply Chain Disruptions

CREATE TABLE IF NOT EXISTS fact_supply_scenarios (
    scenario_id         BIGSERIAL       PRIMARY KEY,
    scenario_name       VARCHAR(200)    NOT NULL,
    scenario_type       VARCHAR(50)     NOT NULL,  -- 'demand_shock' | 'lead_time_shock' | 'capacity_constraint' | 'logistics_disruption'
    description         TEXT,
    shock_parameters    JSONB           NOT NULL DEFAULT '{}',
    affected_items      JSONB,
    affected_locations  JSONB,
    affected_suppliers  JSONB,
    horizon_months      INTEGER         NOT NULL DEFAULT 3,
    status              VARCHAR(20)     NOT NULL DEFAULT 'draft',  -- 'draft' | 'running' | 'completed' | 'failed'
    created_by          VARCHAR(100)    NOT NULL DEFAULT 'api',
    run_by              VARCHAR(100),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_run_at         TIMESTAMPTZ,
    run_duration_ms     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_supply_scenarios_type_status
    ON fact_supply_scenarios (scenario_type, status);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_scenario_results (
    id                  BIGSERIAL       PRIMARY KEY,
    scenario_id         BIGINT          NOT NULL REFERENCES fact_supply_scenarios(scenario_id),
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    plan_month          DATE            NOT NULL,
    baseline_qty        NUMERIC(12,2),
    scenario_qty        NUMERIC(12,2),
    impact_qty          NUMERIC(12,2),   -- scenario - baseline
    impact_pct          NUMERIC(6,2),
    stockout_risk_days  INTEGER,
    excess_risk_qty     NUMERIC(12,2),
    mitigation_option   TEXT,
    computed_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_scenario_result UNIQUE (scenario_id, item_id, loc, plan_month)
);

CREATE INDEX IF NOT EXISTS idx_scenario_results_scenario
    ON fact_scenario_results (scenario_id, plan_month);

CREATE INDEX IF NOT EXISTS idx_scenario_results_impact
    ON fact_scenario_results (scenario_id, impact_qty);
