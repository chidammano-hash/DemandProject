-- Forward-Looking Replenishment Plan
-- Feature: Forward-Looking Replenishment Plan
--
-- Grain: (plan_version, item_no, loc, plan_month) — one row per DFU per forward month per version.
-- Computed from fact_production_forecast CI bands + fact_dfu_policy_assignment
-- by scripts/compute_replenishment_plan.py.
--
-- sigma_method tracks how demand variability was derived:
--   'ci_spread'         — sigma = (forecast_qty_upper - forecast_qty_lower) / (2 * z_ci)
--   'historical_fallback' — CI bands were NULL; fell back to dim_dfu.demand_std
--
-- Policy-type dispatch:
--   continuous_rop   — ROP + order_qty (= effective_eoq)
--   periodic_review  — order_up_to_level; order_qty NULL
--   min_max          — ROP as min; order_up_to_level as max; order_qty NULL
--   manual / JIT     — is_jit = TRUE; all replenishment parameters NULL
--
-- Populate with: make replenishment-plan-compute
-- Full pipeline:  make replenishment-plan-all

CREATE TABLE IF NOT EXISTS fact_replenishment_plan (
    id                      BIGSERIAL PRIMARY KEY,

    -- Version key (matches fact_production_forecast.plan_version, e.g. '2026-02')
    plan_version            TEXT        NOT NULL,

    -- Business key
    item_no                 TEXT        NOT NULL,
    loc                     TEXT        NOT NULL,
    plan_month              DATE        NOT NULL,    -- first day of the forward month covered

    -- Forward horizon position (1 = T+1, 2 = T+2, …)
    horizon_months          SMALLINT,

    -- Policy context (snapshot at compute time)
    policy_id               TEXT,                   -- from fact_dfu_policy_assignment
    policy_type             TEXT,                   -- continuous_rop | periodic_review | min_max | manual
    abc_vol                 TEXT,                   -- ABC classification at compute time
    review_cycle_days       INTEGER,                -- for periodic_review only

    -- Demand inputs from production forecast
    forecast_qty            NUMERIC(15,4),
    forecast_qty_lower      NUMERIC(15,4),          -- P10 CI lower bound
    forecast_qty_upper      NUMERIC(15,4),          -- P90 CI upper bound
    forecast_annual_demand  NUMERIC(15,4),          -- annualized: sum of next N months forecast_qty

    -- Demand variability derived from CI bands (or historical fallback)
    sigma_demand_monthly    NUMERIC(15,4),          -- (upper - lower) / (2 * z_ci)
    sigma_demand_daily      NUMERIC(15,4),          -- sigma_demand_monthly / sqrt(30.44)
    avg_daily_demand        NUMERIC(15,4),          -- forecast_qty / 30.44
    sigma_method            TEXT        DEFAULT 'ci_spread',  -- 'ci_spread' | 'historical_fallback'

    -- Lead time inputs
    lt_mean_days            NUMERIC(10,2),
    lt_std_days             NUMERIC(10,2),

    -- Service level
    service_level_target    NUMERIC(6,4),
    z_score                 NUMERIC(8,4),

    -- Safety stock outputs (forward-looking)
    ss_demand_only          NUMERIC(15,4),          -- demand variability component only
    ss_lt_only              NUMERIC(15,4),          -- lead time variability component only
    ss_combined             NUMERIC(15,4),          -- recommended SS (combined formula)

    -- Cycle stock outputs (EOQ from forecasted annual demand)
    eoq                     NUMERIC(15,4),
    effective_eoq           NUMERIC(15,4),          -- max(eoq, moq), capped at max_eoq_months_supply
    cycle_stock             NUMERIC(15,4),          -- effective_eoq / 2

    -- Policy-specific replenishment parameters
    reorder_point           NUMERIC(15,4),          -- avg_daily_demand * lt_mean_days + ss_combined
    order_qty               NUMERIC(15,4),          -- effective_eoq (continuous_rop only)
    order_up_to_level       NUMERIC(15,4),          -- target max qty (min_max, periodic_review)
    is_jit                  BOOLEAN     DEFAULT FALSE,  -- TRUE for manual/JIT policy

    -- Comparison vs historical SS targets (from fact_safety_stock_targets)
    historical_ss           NUMERIC(15,4),          -- fact_safety_stock_targets.ss_combined
    ss_delta                NUMERIC(15,4),          -- ss_combined - historical_ss
    ss_delta_pct            NUMERIC(10,4),          -- (ss_delta / NULLIF(historical_ss,0)) * 100

    -- Current inventory position snapshot (at compute time)
    current_qty_on_hand     NUMERIC(15,4),
    ss_gap                  NUMERIC(15,4),          -- current_qty_on_hand - ss_combined (negative = shortfall)
    is_below_ss             BOOLEAN,

    -- Audit
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- Primary uniqueness: one row per DFU per forward month per plan version
CREATE UNIQUE INDEX IF NOT EXISTS uix_replenishment_plan_key
    ON fact_replenishment_plan (plan_version, item_no, loc, plan_month);

-- DFU lookup (most common filter pattern)
CREATE INDEX IF NOT EXISTS ix_replenishment_plan_dfu
    ON fact_replenishment_plan (item_no, loc);

-- Plan version filtering (latest version queries)
CREATE INDEX IF NOT EXISTS ix_replenishment_plan_version
    ON fact_replenishment_plan (plan_version);

-- Policy type filtering (policy-breakdown analytics)
CREATE INDEX IF NOT EXISTS ix_replenishment_plan_policy_type
    ON fact_replenishment_plan (policy_type);

-- Below-SS exception queries (partial index — only rows where shortfall exists)
CREATE INDEX IF NOT EXISTS ix_replenishment_plan_below_ss
    ON fact_replenishment_plan (is_below_ss)
    WHERE is_below_ss = TRUE;

-- Month-range queries (forward horizon slicing)
CREATE INDEX IF NOT EXISTS ix_replenishment_plan_month
    ON fact_replenishment_plan (plan_month);

-- ABC class filtering
CREATE INDEX IF NOT EXISTS ix_replenishment_plan_abc_vol
    ON fact_replenishment_plan (abc_vol);
