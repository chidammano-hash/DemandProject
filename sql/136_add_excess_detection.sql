-- Economic excess detection columns for integrated planning targets
-- Replaces static 6-month threshold with cost-aware excess that considers
-- holding cost, safety stock, EOQ, and demand velocity.
-- Depends on: fact_safety_stock_targets, fact_eoq_targets

DROP MATERIALIZED VIEW IF EXISTS mv_integrated_planning_targets;

CREATE MATERIALIZED VIEW mv_integrated_planning_targets AS
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
    ss.computed_at,
    -- Stockout Risk Score (0-100): higher = more urgent
    -- Factor 1: Days of supply vs lead time (0-40 pts)
    -- Factor 2: Safety stock coverage (0-30 pts)
    -- Factor 3: Demand variability (0-20 pts)
    -- Factor 4: ABC class weight (0-10 pts)
    LEAST(100, GREATEST(0,
        -- Factor 1: Days of supply vs lead time (0-40 pts)
        CASE
            WHEN COALESCE(ss.current_dos, 0) <= 0 THEN 40
            WHEN ss.lead_time_mean_days > 0 AND ss.current_dos < ss.lead_time_mean_days THEN
                40 * (1 - ss.current_dos / ss.lead_time_mean_days)
            WHEN ss.lead_time_mean_days > 0 AND ss.current_dos < 2 * ss.lead_time_mean_days THEN
                20 * (1 - (ss.current_dos - ss.lead_time_mean_days) / ss.lead_time_mean_days)
            ELSE 0
        END
        +
        -- Factor 2: Safety stock coverage (0-30 pts)
        CASE
            WHEN ss.ss_combined <= 0 THEN 15
            WHEN ss.current_qty_on_hand <= 0 THEN 30
            WHEN ss.current_qty_on_hand < ss.ss_combined THEN
                30 * (1 - ss.current_qty_on_hand / ss.ss_combined)
            ELSE 0
        END
        +
        -- Factor 3: Demand variability (0-20 pts)
        CASE
            WHEN COALESCE(ss.demand_cv, 0) > 1.0 THEN 20
            WHEN COALESCE(ss.demand_cv, 0) > 0.5 THEN 15
            WHEN COALESCE(ss.demand_cv, 0) > 0.3 THEN 10
            ELSE 5
        END
        +
        -- Factor 4: ABC class weight (0-10 pts)
        CASE
            WHEN ss.abc_vol = 'A' THEN 10
            WHEN ss.abc_vol = 'B' THEN 5
            ELSE 2
        END
    ))::smallint AS stockout_risk_score,

    -- ═══ EXCESS DETECTION ═══
    -- Excess qty: on-hand above safety stock + EOQ ceiling
    CASE
        WHEN COALESCE(ss.current_qty_on_hand, 0) <= 0 THEN 0
        WHEN ss.demand_mean_monthly > 0 THEN
            GREATEST(0, ss.current_qty_on_hand - ss.ss_combined - COALESCE(eoq.effective_eoq, 0))
        ELSE ss.current_qty_on_hand  -- all inventory is excess if no demand
    END AS excess_qty,

    -- Excess value in dollars (excess qty * unit cost)
    CASE
        WHEN COALESCE(ss.current_qty_on_hand, 0) <= 0 THEN 0
        WHEN ss.demand_mean_monthly > 0 THEN
            GREATEST(0, ss.current_qty_on_hand - ss.ss_combined - COALESCE(eoq.effective_eoq, 0))
            * COALESCE(eoq.unit_cost, 0)
        ELSE ss.current_qty_on_hand * COALESCE(eoq.unit_cost, 0)
    END AS excess_value_usd,

    -- Monthly holding cost on excess inventory (25% annual rate / 12)
    CASE
        WHEN COALESCE(ss.current_qty_on_hand, 0) <= ss.ss_combined + COALESCE(eoq.effective_eoq, 0) THEN 0
        ELSE GREATEST(0, ss.current_qty_on_hand - ss.ss_combined - COALESCE(eoq.effective_eoq, 0))
             * COALESCE(eoq.unit_cost, 0) * 0.25 / 12
    END AS excess_holding_cost_monthly,

    -- Months of supply above safety stock
    CASE
        WHEN ss.demand_mean_monthly > 0 THEN
            GREATEST(0, ss.current_qty_on_hand - ss.ss_combined) / ss.demand_mean_monthly
        ELSE NULL
    END AS excess_months_supply,

    -- Excess risk score (0-100): higher = more excess, more costly
    LEAST(100, GREATEST(0,
        CASE
            WHEN ss.demand_mean_monthly <= 0 AND ss.current_qty_on_hand > 0 THEN 80  -- zero-velocity excess
            WHEN ss.demand_mean_monthly > 0 AND ss.current_qty_on_hand > (ss.ss_combined + COALESCE(eoq.effective_eoq, 0)) * 2 THEN 70  -- >2x target
            WHEN ss.demand_mean_monthly > 0 AND ss.current_qty_on_hand > ss.ss_combined + COALESCE(eoq.effective_eoq, 0) THEN 40  -- above target
            ELSE 0
        END
        + CASE WHEN ss.abc_vol = 'A' THEN 20 WHEN ss.abc_vol = 'B' THEN 10 ELSE 5 END  -- value-weighted
    ))::smallint AS excess_risk_score

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

-- Fast lookup/sort by stockout risk score (high risk first)
CREATE INDEX IF NOT EXISTS idx_integrated_targets_risk_score
    ON mv_integrated_planning_targets (stockout_risk_score DESC);

-- Fast lookup/sort by excess risk score (high excess first)
CREATE INDEX IF NOT EXISTS idx_integrated_targets_excess_risk
    ON mv_integrated_planning_targets (excess_risk_score DESC);

-- Fast lookup for SKUs with excess inventory
CREATE INDEX IF NOT EXISTS idx_integrated_targets_excess_qty
    ON mv_integrated_planning_targets (excess_qty) WHERE excess_qty > 0;
