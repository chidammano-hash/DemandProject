-- Feature 37: Inventory-Forecast bridge materialized view
-- Joins monthly inventory aggregates with forecast predictions at execution lag.
-- Grain: item_no + loc + month_start + model_id
-- Purpose: connects forecast accuracy to inventory outcomes (stockout / excess).

DROP MATERIALIZED VIEW IF EXISTS mv_inventory_forecast_monthly CASCADE;

CREATE MATERIALIZED VIEW mv_inventory_forecast_monthly AS
SELECT
    i.month_start,
    i.item_no,
    i.loc,
    -- Inventory position
    i.eom_qty_on_hand,
    i.eom_qty_on_hand_on_order,
    i.monthly_sales,
    i.avg_daily_sls,
    i.latest_lead_time_days,
    i.snapshot_days,
    -- Forecast fields
    f.model_id,
    f.basefcst_pref                                AS forecast,
    f.tothist_dmd                                  AS actual_demand,
    (f.basefcst_pref - f.tothist_dmd)             AS forecast_error,
    ABS(f.basefcst_pref - f.tothist_dmd)          AS abs_error,
    -- Inventory events
    (i.eom_qty_on_hand <= 0)                       AS is_stockout,
    CASE
        WHEN i.avg_daily_sls > 0
             AND (i.eom_qty_on_hand / i.avg_daily_sls) > 90
        THEN TRUE ELSE FALSE
    END                                            AS is_excess,
    -- Days of supply
    CASE
        WHEN i.avg_daily_sls > 0
        THEN (i.eom_qty_on_hand / i.avg_daily_sls)::double precision
        ELSE NULL
    END                                            AS dos,
    -- Bias direction
    CASE
        WHEN f.basefcst_pref > f.tothist_dmd THEN 'over'
        WHEN f.basefcst_pref < f.tothist_dmd THEN 'under'
        ELSE 'exact'
    END                                            AS bias_direction,
    -- DFU attributes for slicing
    COALESCE(d.cluster_assignment, '(unassigned)') AS cluster_assignment,
    COALESCE(d.abc_vol, '(unknown)')               AS abc_vol,
    COALESCE(d.region, '(unknown)')                AS region,
    COALESCE(d.brand, '(unknown)')                 AS brand,
    COALESCE(d.seasonality_profile, '(unknown)')   AS seasonality_profile,
    -- Zero-velocity flag: items with on-hand stock but no sales velocity cannot be
    -- classified as excess via DOS (DOS=NULL). Flag them separately.
    (i.avg_daily_sls = 0 AND i.eom_qty_on_hand > 0) AS zero_velocity_flag
FROM agg_inventory_monthly i
INNER JOIN fact_external_forecast_monthly f
    ON i.item_no = f.dmdunit
   AND i.loc    = f.loc
   AND i.month_start = f.startdate
LEFT JOIN dim_dfu d
    ON f.dmdunit = d.dmdunit
   AND f.loc     = d.loc
WHERE f.lag = COALESCE(d.execution_lag, 0)
  AND f.tothist_dmd  IS NOT NULL
  AND f.basefcst_pref IS NOT NULL
WITH NO DATA;

-- Primary key index
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_inv_fcst_pk
    ON mv_inventory_forecast_monthly (item_no, loc, month_start, model_id);

-- Query-path indexes
CREATE INDEX IF NOT EXISTS idx_mv_inv_fcst_model
    ON mv_inventory_forecast_monthly (model_id);
CREATE INDEX IF NOT EXISTS idx_mv_inv_fcst_month
    ON mv_inventory_forecast_monthly (month_start);
CREATE INDEX IF NOT EXISTS idx_mv_inv_fcst_cluster
    ON mv_inventory_forecast_monthly (cluster_assignment);
CREATE INDEX IF NOT EXISTS idx_mv_inv_fcst_abc
    ON mv_inventory_forecast_monthly (abc_vol);

-- Partial indexes for event filtering
CREATE INDEX IF NOT EXISTS idx_mv_inv_fcst_stockout
    ON mv_inventory_forecast_monthly (model_id, month_start) WHERE is_stockout = TRUE;
CREATE INDEX IF NOT EXISTS idx_mv_inv_fcst_excess
    ON mv_inventory_forecast_monthly (model_id, month_start) WHERE is_excess = TRUE;

-- Seasonality and zero-velocity indexes
CREATE INDEX IF NOT EXISTS idx_mv_inv_fcst_seasonality
    ON mv_inventory_forecast_monthly (seasonality_profile);
CREATE INDEX IF NOT EXISTS idx_mv_inv_fcst_zero_vel
    ON mv_inventory_forecast_monthly (zero_velocity_flag) WHERE zero_velocity_flag = TRUE;
