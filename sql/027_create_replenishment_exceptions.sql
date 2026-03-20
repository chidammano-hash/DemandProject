-- IPfeature7: Exception Queue & Replenishment Recommendations
-- Creates fact_replenishment_exceptions table for storing detected inventory exceptions

CREATE TABLE IF NOT EXISTS fact_replenishment_exceptions (
    exception_sk              BIGSERIAL PRIMARY KEY,
    exception_id              TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
    item_no                   TEXT NOT NULL,
    loc                       TEXT NOT NULL,
    exception_date            DATE NOT NULL,
    exception_type            TEXT NOT NULL,
    severity                  TEXT NOT NULL CHECK (severity IN ('critical','high','medium','low')),
    -- Current state snapshot (at detection time)
    current_qty_on_hand       NUMERIC(15,4),
    current_dos               NUMERIC(10,2),
    ss_combined               NUMERIC(15,4),
    reorder_point             NUMERIC(15,4),
    -- Recommendation
    recommended_order_qty     NUMERIC(15,4),
    recommended_order_by      DATE,
    expected_receipt_date     DATE,
    estimated_order_value     NUMERIC(12,2),
    -- Context
    policy_id                 TEXT,
    lead_time_mean_days       NUMERIC(10,2),
    -- Workflow
    status                    TEXT NOT NULL DEFAULT 'open'
                              CHECK (status IN ('open','acknowledged','ordered','resolved')),
    acknowledged_by           TEXT,
    acknowledged_ts           TIMESTAMPTZ,
    ordered_ts                TIMESTAMPTZ,
    resolved_ts               TIMESTAMPTZ,
    notes                     TEXT,
    load_ts                   TIMESTAMPTZ DEFAULT NOW(),
    modified_ts               TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exceptions_item_loc
    ON fact_replenishment_exceptions (item_no, loc);
CREATE INDEX IF NOT EXISTS idx_exceptions_type
    ON fact_replenishment_exceptions (exception_type);
CREATE INDEX IF NOT EXISTS idx_exceptions_severity
    ON fact_replenishment_exceptions (severity);
CREATE INDEX IF NOT EXISTS idx_exceptions_status
    ON fact_replenishment_exceptions (status);
CREATE INDEX IF NOT EXISTS idx_exceptions_open_crit
    ON fact_replenishment_exceptions (severity, exception_date)
    WHERE status = 'open' AND severity = 'critical';
CREATE INDEX IF NOT EXISTS idx_exceptions_date
    ON fact_replenishment_exceptions (exception_date DESC);
