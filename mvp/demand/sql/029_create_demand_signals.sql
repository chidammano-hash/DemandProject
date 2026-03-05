-- IPfeature9: Demand Sensing & Short-Horizon Signal Integration
-- Table: fact_demand_signals
-- Stores daily intra-month demand velocity signals computed from
-- fact_inventory_snapshot (MTD sales) vs champion forecast.

CREATE TABLE IF NOT EXISTS fact_demand_signals (
    signal_sk               BIGSERIAL PRIMARY KEY,
    item_no                 TEXT NOT NULL,
    loc                     TEXT NOT NULL,
    signal_date             DATE NOT NULL,
    month_start             DATE NOT NULL,
    day_of_month            INTEGER NOT NULL,
    days_elapsed            INTEGER NOT NULL,
    days_remaining          INTEGER NOT NULL,
    -- Current month progress
    mtd_actual              NUMERIC(15,4),
    mtd_expected            NUMERIC(15,4),       -- forecast_daily × days_elapsed
    projected_monthly       NUMERIC(15,4),
    historical_avg_monthly  NUMERIC(15,4),       -- same calendar month, 3-yr avg
    forecast_monthly        NUMERIC(15,4),       -- champion forecast for this month
    -- Signal metrics
    demand_vs_forecast_pct  NUMERIC(10,2),
    demand_acceleration     NUMERIC(10,4),       -- daily rate this month vs prior 30d
    signal_type             TEXT,                -- 'above_plan' | 'below_plan' | 'on_plan'
    signal_strength         NUMERIC(10,4),
    -- Inventory implication
    current_on_hand         NUMERIC(15,4),
    ss_combined             NUMERIC(15,4),
    is_below_ss             BOOLEAN,
    projected_stockout      BOOLEAN,
    projected_excess        BOOLEAN,
    alert_priority          TEXT,                -- 'urgent' | 'watch' | 'none'
    load_ts                 TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_demand_signals_pk
    ON fact_demand_signals (item_no, loc, signal_date);
CREATE INDEX IF NOT EXISTS idx_demand_signals_month
    ON fact_demand_signals (month_start);
CREATE INDEX IF NOT EXISTS idx_demand_signals_type
    ON fact_demand_signals (signal_type);
CREATE INDEX IF NOT EXISTS idx_demand_signals_urgent
    ON fact_demand_signals (alert_priority)
    WHERE alert_priority = 'urgent';
CREATE INDEX IF NOT EXISTS idx_demand_signals_date
    ON fact_demand_signals (signal_date DESC);
