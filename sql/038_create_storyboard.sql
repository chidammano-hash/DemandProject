-- Feature 40: Planner Storyboard — Exception-Based Value Workflow
-- DDL for exception_queue and planner_decisions tables.
--
-- exception_queue: planner's exception inbox, generated nightly.
-- planner_decisions: audit trail of every planner action taken on exceptions.

-- ---------------------------------------------------------------------------
-- exception_queue
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS exception_queue (
    exception_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exception_type   TEXT NOT NULL,        -- forecast_bias | stockout_risk | accuracy_drop | excess_risk | model_drift | new_item
    item_id          TEXT NOT NULL,
    loc              TEXT NOT NULL,
    severity         NUMERIC NOT NULL,     -- 0.0-1.0 computed score
    financial_impact NUMERIC,              -- estimated $ impact
    headline         TEXT,                 -- rule-based 1-sentence description
    supporting_data  JSONB,                -- structured evidence dict
    status           TEXT NOT NULL DEFAULT 'open',  -- open | investigating | resolved | dismissed
    assigned_to      TEXT,
    generated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at       TIMESTAMPTZ,          -- auto-close stale exceptions
    month_start      DATE,                 -- which forecast month this covers
    load_ts          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eq_status_severity  ON exception_queue(status, severity DESC);
CREATE INDEX IF NOT EXISTS idx_eq_item_loc         ON exception_queue(item_id, loc);
CREATE INDEX IF NOT EXISTS idx_eq_type             ON exception_queue(exception_type);
CREATE INDEX IF NOT EXISTS idx_eq_generated        ON exception_queue(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_eq_month_start      ON exception_queue(month_start DESC);

-- Partial index for fast open+critical lookups (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_eq_open_critical
    ON exception_queue(generated_at DESC)
    WHERE status = 'open' AND severity >= 0.75;

-- ---------------------------------------------------------------------------
-- planner_decisions  (audit trail)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS planner_decisions (
    decision_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exception_id   UUID REFERENCES exception_queue(exception_id),
    item_id        TEXT NOT NULL,
    loc            TEXT NOT NULL,
    decision_type  TEXT NOT NULL,  -- override_forecast | accept_exception | escalate | dismiss | request_info
    decision_value JSONB,          -- {"new_forecast": 500} or {"escalation_reason": "..."}
    rationale      TEXT,
    decided_by     TEXT DEFAULT 'planner',
    decided_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pd_exception  ON planner_decisions(exception_id);
CREATE INDEX IF NOT EXISTS idx_pd_item_loc   ON planner_decisions(item_id, loc);
CREATE INDEX IF NOT EXISTS idx_pd_decided_at ON planner_decisions(decided_at DESC);
