-- 064_create_query_performance.sql
-- Query/endpoint performance tracking table (Spec 08-03)

CREATE TABLE IF NOT EXISTS fact_query_performance (
    perf_id     BIGSERIAL PRIMARY KEY,
    endpoint    TEXT NOT NULL,
    method      TEXT NOT NULL DEFAULT 'GET',
    duration_ms NUMERIC NOT NULL,
    db_queries  INT NOT NULL DEFAULT 0,
    cache_hit   BOOLEAN NOT NULL DEFAULT FALSE,
    user_id     UUID,
    params      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_perf_created_duration
    ON fact_query_performance (created_at DESC, duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_query_perf_endpoint
    ON fact_query_performance (endpoint);
CREATE INDEX IF NOT EXISTS idx_query_perf_slow
    ON fact_query_performance (duration_ms DESC) WHERE duration_ms > 1000;
