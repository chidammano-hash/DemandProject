-- 068_create_fva_tracking.sql
-- Spec 08-07: Forecast Value Added (FVA) intervention tracking

BEGIN;

CREATE TABLE IF NOT EXISTS fact_intervention_metrics (
    intervention_id             BIGSERIAL       PRIMARY KEY,
    user_id                     UUID,
    intervention_type           TEXT            NOT NULL,
    resource_type               TEXT,
    resource_id                 TEXT,
    metric_before               JSONB,
    metric_after                JSONB,
    financial_impact_estimate   NUMERIC,
    actual_financial_impact     NUMERIC,
    measurement_window_start    DATE,
    measurement_window_end      DATE,
    status                      TEXT            DEFAULT 'pending',
    created_at                  TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_intervention_user
    ON fact_intervention_metrics (user_id);
CREATE INDEX IF NOT EXISTS idx_intervention_status
    ON fact_intervention_metrics (status);
CREATE INDEX IF NOT EXISTS idx_intervention_type
    ON fact_intervention_metrics (intervention_type);
CREATE INDEX IF NOT EXISTS idx_intervention_window_end
    ON fact_intervention_metrics (measurement_window_end);

COMMIT;
