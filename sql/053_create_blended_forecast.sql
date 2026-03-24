-- F3.4 — Demand Sensing Integration (Blended Forecast)

CREATE TABLE IF NOT EXISTS fact_blended_demand_plan (
    id                      BIGSERIAL       PRIMARY KEY,
    item_id                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    week_start              DATE            NOT NULL,
    plan_version            VARCHAR(50)     NOT NULL DEFAULT 'latest',
    alpha_weight            NUMERIC(4,3)    NOT NULL,  -- sensing weight 0.0..1.0
    sensing_signal_qty      NUMERIC(12,2),
    statistical_forecast_qty NUMERIC(12,2),
    blended_qty             NUMERIC(12,2)   NOT NULL,
    velocity_spike_ratio    NUMERIC(6,3),
    is_outlier_capped       BOOLEAN         NOT NULL DEFAULT FALSE,
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_blended_plan UNIQUE (item_id, loc, week_start, plan_version)
);

CREATE INDEX IF NOT EXISTS idx_blended_item_loc
    ON fact_blended_demand_plan (item_id, loc, week_start);

CREATE INDEX IF NOT EXISTS idx_blended_sensing_active
    ON fact_blended_demand_plan (alpha_weight, week_start)
    WHERE alpha_weight > 0.3;

-- -----------------------------------------------------------------------
-- Materialized view: DFUs where sensing currently overrides statistical

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_sensing_overrides_active AS
SELECT
    item_id, loc, week_start, alpha_weight,
    sensing_signal_qty, statistical_forecast_qty, blended_qty,
    velocity_spike_ratio
FROM fact_blended_demand_plan
WHERE week_start = (
    SELECT MIN(week_start)
    FROM fact_blended_demand_plan
    WHERE week_start >= CURRENT_DATE
)
  AND alpha_weight > 0.5
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uq_mv_sensing_active
    ON mv_sensing_overrides_active (item_id, loc);
