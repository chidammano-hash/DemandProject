-- 063_create_data_quality.sql
-- Data Quality monitoring tables + dashboard materialized view (Spec 08-01)

-- Catalog of defined checks (reusable definitions)
CREATE TABLE IF NOT EXISTS dim_dq_check_catalog (
    check_id    SERIAL PRIMARY KEY,
    check_name  TEXT NOT NULL UNIQUE,
    check_type  TEXT NOT NULL,          -- freshness, completeness, uniqueness, range, volume_delta, referential_integrity
    domain      TEXT NOT NULL,          -- item, location, customer, time, dfu, sales, forecast, inventory
    sql_template TEXT,                  -- parameterized SQL for the check
    threshold   NUMERIC,               -- numeric threshold for pass/fail evaluation
    severity    TEXT NOT NULL DEFAULT 'warning',  -- info, warning, critical
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dq_catalog_domain   ON dim_dq_check_catalog (domain);
CREATE INDEX IF NOT EXISTS idx_dq_catalog_type     ON dim_dq_check_catalog (check_type);
CREATE INDEX IF NOT EXISTS idx_dq_catalog_enabled  ON dim_dq_check_catalog (enabled) WHERE enabled = TRUE;

-- Results of each check execution
CREATE TABLE IF NOT EXISTS fact_dq_check_results (
    check_id      BIGSERIAL PRIMARY KEY,
    check_name    TEXT NOT NULL,
    domain        TEXT NOT NULL,
    table_name    TEXT NOT NULL,
    severity      TEXT NOT NULL DEFAULT 'warning',
    status        TEXT NOT NULL DEFAULT 'pass',     -- pass, fail, warn, error
    metric_value  NUMERIC,
    threshold     NUMERIC,
    details       JSONB,
    run_ts        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dq_results_run_ts     ON fact_dq_check_results (run_ts DESC);
CREATE INDEX IF NOT EXISTS idx_dq_results_domain     ON fact_dq_check_results (domain);
CREATE INDEX IF NOT EXISTS idx_dq_results_status     ON fact_dq_check_results (status);
CREATE INDEX IF NOT EXISTS idx_dq_results_check_name ON fact_dq_check_results (check_name);
CREATE INDEX IF NOT EXISTS idx_dq_results_severity   ON fact_dq_check_results (severity);
CREATE INDEX IF NOT EXISTS idx_dq_results_domain_ts  ON fact_dq_check_results (domain, run_ts DESC);

-- Dashboard materialized view: pass/fail/warn counts by domain and date
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_dq_dashboard AS
SELECT
    domain,
    date_trunc('day', run_ts)::DATE AS run_date,
    COUNT(*) FILTER (WHERE status = 'pass')  AS pass_count,
    COUNT(*) FILTER (WHERE status = 'fail')  AS fail_count,
    COUNT(*) FILTER (WHERE status = 'warn')  AS warn_count,
    COUNT(*) FILTER (WHERE status = 'error') AS error_count,
    COUNT(*)                                  AS total_count
FROM fact_dq_check_results
GROUP BY domain, date_trunc('day', run_ts)::DATE;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_dq_dashboard_pk
    ON mv_dq_dashboard (domain, run_date);
CREATE INDEX IF NOT EXISTS idx_mv_dq_dashboard_date
    ON mv_dq_dashboard (run_date DESC);
