-- Unified inventory planning targets combining SS, EOQ, and ROP
-- Provides a single row per DFU with all three planning parameters + cost metrics
-- Depends on: fact_safety_stock_targets, fact_eoq_targets

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_integrated_planning_targets AS
SELECT
    ss.item_id,
    ss.loc,
    -- Safety Stock
    ss.ss_combined                  AS safety_stock_qty,
    ss.reorder_point                AS reorder_point,
    ss.service_level_target,
    ss.abc_vol,
    ss.abc_xyz_segment,
    ss.demand_mean_monthly,
    ss.demand_std_monthly,
    ss.demand_cv,
    -- EOQ
    COALESCE(eoq.eoq, 0)           AS eoq_qty,
    COALESCE(eoq.effective_eoq, 0) AS effective_eoq,
    COALESCE(eoq.eoq_cycle_stock, 0) AS cycle_stock,
    COALESCE(eoq.order_frequency, 0) AS orders_per_year,
    COALESCE(eoq.unit_cost, 0)     AS unit_cost,
    -- Combined targets
    ss.ss_combined + COALESCE(eoq.eoq_cycle_stock, 0)  AS target_avg_inventory,
    ss.ss_combined                  AS target_min_inventory,
    ss.ss_combined + COALESCE(eoq.effective_eoq, 0)    AS target_max_inventory,
    -- Current position
    ss.current_qty_on_hand,
    ss.current_dos,
    ss.ss_gap,
    ss.is_below_ss,
    -- Cost metrics (25% annual holding cost rate, divided by 12 for monthly)
    ss.ss_combined * COALESCE(eoq.unit_cost, 0) * 0.25 / 12   AS monthly_ss_holding_cost,
    COALESCE(eoq.eoq_cycle_stock, 0) * COALESCE(eoq.unit_cost, 0) * 0.25 / 12 AS monthly_cycle_holding_cost,
    (ss.ss_combined + COALESCE(eoq.eoq_cycle_stock, 0)) * COALESCE(eoq.unit_cost, 0) * 0.25 / 12 AS monthly_total_holding_cost,
    COALESCE(eoq.annual_order_cost, 0) / 12     AS monthly_ordering_cost,
    COALESCE(eoq.total_annual_cost, 0) / 12     AS monthly_total_cost,
    -- Lead time
    ss.lead_time_mean_days,
    ss.lead_time_std_days,
    -- Metadata
    ss.policy_version,
    ss.forecast_source,
    ss.forecast_model_id,
    ss.computed_at
FROM fact_safety_stock_targets ss
LEFT JOIN fact_eoq_targets eoq
    ON ss.item_id = eoq.item_id AND ss.loc = eoq.loc
WHERE ss.policy_version = (
    SELECT MAX(policy_version) FROM fact_safety_stock_targets
);

-- Unique index enables CONCURRENTLY refresh
CREATE UNIQUE INDEX IF NOT EXISTS uq_integrated_targets_dfu
    ON mv_integrated_planning_targets (item_id, loc);

-- Filter by ABC class
CREATE INDEX IF NOT EXISTS idx_integrated_targets_abc
    ON mv_integrated_planning_targets (abc_vol);

-- Fast lookup for items below safety stock
CREATE INDEX IF NOT EXISTS idx_integrated_targets_below_ss
    ON mv_integrated_planning_targets (is_below_ss) WHERE is_below_ss = TRUE;
