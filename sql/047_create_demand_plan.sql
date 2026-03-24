-- F2.2 — Multi-Horizon Demand Plan (Quantile Forecasts)
-- Tables: fact_demand_plan, fact_demand_plan_weekly, fact_plan_versions

CREATE TABLE IF NOT EXISTS fact_plan_versions (
    plan_version        VARCHAR(50)     PRIMARY KEY,
    plan_date           DATE            NOT NULL,
    plan_label          VARCHAR(100),
    model_id            VARCHAR(100)    NOT NULL,
    horizon_months      INTEGER         NOT NULL,
    dfu_count           INTEGER,
    generated_by        VARCHAR(100),
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    status              VARCHAR(20)     NOT NULL DEFAULT 'draft',
    notes               TEXT,
    parent_version      VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS fact_demand_plan (
    id                  BIGSERIAL       PRIMARY KEY,
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    plan_month          DATE            NOT NULL,
    quantile            NUMERIC(4,2)    NOT NULL,
    forecast_qty        NUMERIC(12,2)   NOT NULL,
    lower_bound         NUMERIC(12,2),
    upper_bound         NUMERIC(12,2),
    model_id            VARCHAR(100)    NOT NULL,
    plan_version        VARCHAR(50)     NOT NULL,
    horizon_months      INTEGER         NOT NULL,
    sigma_forecast      NUMERIC(10,4),
    sigma_demand        NUMERIC(10,4),
    sigma_combined      NUMERIC(10,4),
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    cluster_id          INTEGER,
    abc_class           VARCHAR(5),
    seasonality_profile VARCHAR(50),
    CONSTRAINT uq_demand_plan UNIQUE (item_id, loc, plan_month, quantile, plan_version)
);

CREATE INDEX IF NOT EXISTS idx_demand_plan_item_loc_month
    ON fact_demand_plan (item_id, loc, plan_month);

CREATE INDEX IF NOT EXISTS idx_demand_plan_version
    ON fact_demand_plan (plan_version, plan_month);

CREATE INDEX IF NOT EXISTS idx_demand_plan_quantile
    ON fact_demand_plan (quantile, plan_version);

CREATE INDEX IF NOT EXISTS idx_demand_plan_horizon
    ON fact_demand_plan (horizon_months, plan_version);

CREATE TABLE IF NOT EXISTS fact_demand_plan_weekly (
    id                  BIGSERIAL       PRIMARY KEY,
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    plan_week           DATE            NOT NULL,
    iso_week            INTEGER         NOT NULL,
    iso_year            INTEGER         NOT NULL,
    plan_month          DATE            NOT NULL,
    quantile            NUMERIC(4,2)    NOT NULL,
    forecast_qty        NUMERIC(12,2)   NOT NULL,
    weekly_weight       NUMERIC(6,4)    NOT NULL,
    plan_version        VARCHAR(50)     NOT NULL,
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_demand_plan_weekly UNIQUE (item_id, loc, plan_week, quantile, plan_version)
);

CREATE INDEX IF NOT EXISTS idx_demand_plan_weekly_item_loc
    ON fact_demand_plan_weekly (item_id, loc, plan_week);

CREATE INDEX IF NOT EXISTS idx_demand_plan_weekly_version
    ON fact_demand_plan_weekly (plan_version, plan_week);
