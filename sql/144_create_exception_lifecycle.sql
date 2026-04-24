-- Gen-4 Roadmap 1.9 — Exception Lifecycle Audit
--
-- Immutable audit table recording every state transition of a replenishment
-- exception (open -> acknowledged -> investigating -> resolved, or ordered).
-- Every row is append-only; do not UPDATE or DELETE.
--
-- Also provides an MTTR (mean-time-to-resolve) aggregate view keyed on
-- exception_type and severity.

CREATE TABLE IF NOT EXISTS fact_exception_lifecycle (
    lifecycle_id     BIGSERIAL PRIMARY KEY,
    exception_id     TEXT        NOT NULL,
    from_state       TEXT,                                -- NULL for initial 'open' insert
    to_state         TEXT        NOT NULL,                -- open|acknowledged|investigating|ordered|resolved|cancelled
    transitioned_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor            TEXT,                                -- user id / 'system' / 'auto'
    notes            TEXT,
    -- Optional denormalized context (cheaper to keep than re-join exception_queue later)
    exception_type   TEXT,
    severity         TEXT,
    item_id          TEXT,
    loc              TEXT,
    financial_impact NUMERIC(18, 2),
    load_ts          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exc_lifecycle_exception_id
    ON fact_exception_lifecycle (exception_id);

CREATE INDEX IF NOT EXISTS idx_exc_lifecycle_transitioned_at
    ON fact_exception_lifecycle (transitioned_at DESC);

CREATE INDEX IF NOT EXISTS idx_exc_lifecycle_to_state
    ON fact_exception_lifecycle (to_state);

-- Append-only guard: block UPDATE and DELETE.
-- Keeps the ledger tamper-evident for audit.
CREATE OR REPLACE FUNCTION fact_exception_lifecycle_block_update()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'fact_exception_lifecycle is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_exc_lifecycle_no_update ON fact_exception_lifecycle;
CREATE TRIGGER trg_exc_lifecycle_no_update
    BEFORE UPDATE OR DELETE ON fact_exception_lifecycle
    FOR EACH ROW EXECUTE FUNCTION fact_exception_lifecycle_block_update();

-- MTTR view: for each exception_id that reached a terminal state, compute
-- (resolved_ts - first_opened_ts). Aggregated at the granularity the caller
-- wants via the base view; a summary view gives per-type / per-severity MTTR.

CREATE OR REPLACE VIEW v_exception_mttr_by_exception AS
SELECT
    l.exception_id,
    MAX(l.exception_type)              AS exception_type,
    MAX(l.severity)                    AS severity,
    MIN(l.transitioned_at) FILTER (WHERE l.to_state = 'open')
                                       AS opened_at,
    MAX(l.transitioned_at) FILTER (WHERE l.to_state IN ('resolved', 'cancelled'))
                                       AS resolved_at,
    EXTRACT(EPOCH FROM (
        MAX(l.transitioned_at) FILTER (WHERE l.to_state IN ('resolved', 'cancelled'))
        - MIN(l.transitioned_at) FILTER (WHERE l.to_state = 'open')
    )) / 3600.0                        AS resolution_hours
FROM fact_exception_lifecycle l
GROUP BY l.exception_id
HAVING MIN(l.transitioned_at) FILTER (WHERE l.to_state = 'open') IS NOT NULL;

CREATE OR REPLACE VIEW v_exception_mttr_summary AS
SELECT
    exception_type,
    severity,
    COUNT(*)                                        AS resolved_count,
    ROUND(AVG(resolution_hours)::NUMERIC, 2)         AS mttr_hours_avg,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_hours)::NUMERIC, 2)
                                                    AS mttr_hours_p50,
    ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY resolution_hours)::NUMERIC, 2)
                                                    AS mttr_hours_p90
FROM v_exception_mttr_by_exception
WHERE resolution_hours IS NOT NULL
GROUP BY exception_type, severity;
