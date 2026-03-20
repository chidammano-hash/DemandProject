-- 067_create_external_signals.sql
-- Spec 08-06: External demand signals and demand decomposition

BEGIN;

-- External signal source registry
CREATE TABLE IF NOT EXISTS dim_external_signal_source (
    source_id               SERIAL          PRIMARY KEY,
    name                    TEXT            UNIQUE,
    source_type             TEXT            NOT NULL,
    api_config              JSONB,
    refresh_interval_hours  INT             DEFAULT 24,
    enabled                 BOOLEAN         DEFAULT TRUE,
    last_refresh_at         TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     DEFAULT now()
);

-- External signal fact table
CREATE TABLE IF NOT EXISTS fact_external_signal (
    signal_id       BIGSERIAL       PRIMARY KEY,
    source_id       INT             REFERENCES dim_external_signal_source(source_id),
    signal_date     DATE,
    item_no         TEXT,
    loc             TEXT,
    signal_type     TEXT,
    signal_value    NUMERIC,
    confidence      NUMERIC,
    raw_payload     JSONB,
    created_at      TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ext_signal_date
    ON fact_external_signal (signal_date);
CREATE INDEX IF NOT EXISTS idx_ext_signal_item_loc
    ON fact_external_signal (item_no, loc);
CREATE INDEX IF NOT EXISTS idx_ext_signal_source
    ON fact_external_signal (source_id);

-- Demand decomposition placeholder table
-- Will be converted to a materialized view once the decomposition query is finalized
CREATE TABLE IF NOT EXISTS mv_demand_decomposition (
    item_no                 TEXT,
    loc                     TEXT,
    month                   DATE,
    base_demand             NUMERIC,
    trend_component         NUMERIC,
    seasonal_component      NUMERIC,
    promotional_uplift      NUMERIC,
    external_signal_effect  NUMERIC,
    residual                NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_demand_decomp_item_loc
    ON mv_demand_decomposition (item_no, loc);
CREATE INDEX IF NOT EXISTS idx_demand_decomp_month
    ON mv_demand_decomposition (month);

COMMIT;
