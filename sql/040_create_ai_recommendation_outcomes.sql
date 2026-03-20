-- IPAIfeature1: AI Recommendation Outcomes — closes the feedback loop
-- Records planner decisions and 30-day outcome measurements for every
-- accepted/rejected AI insight.
--
-- Run: psql $DATABASE_URL -f sql/040_create_ai_recommendation_outcomes.sql

CREATE TABLE IF NOT EXISTS ai_recommendation_outcomes (
    outcome_id              SERIAL PRIMARY KEY,
    insight_id              INTEGER NOT NULL REFERENCES ai_insights(insight_id),
    insight_type            VARCHAR(80) NOT NULL,
    item_no                 TEXT NOT NULL,
    loc                     TEXT NOT NULL,
    abc_vol                 TEXT,

    -- Planner decision
    planner_decision        VARCHAR(20) NOT NULL
                                CHECK (planner_decision IN
                                    ('accepted','rejected','snoozed','auto_accepted')),
    ai_confidence           VARCHAR(10)
                                CHECK (ai_confidence IN ('high','medium','low')),
    financial_impact_est    NUMERIC(15,2),

    -- Snapshot metrics BEFORE action (from ai_insights at decision time)
    metric_before_dos       NUMERIC(10,2),
    metric_before_wape      NUMERIC(8,4),
    metric_before_bias_pct  NUMERIC(8,4),
    lead_time_days          INTEGER,

    -- Action details
    action_taken            TEXT,       -- what the planner or AI said to do
    executed_at             TIMESTAMPTZ,

    -- Outcome measured T+30 days (populated by outcome checker job)
    outcome_check_due_at    TIMESTAMPTZ,
    outcome_label           VARCHAR(20)
                                CHECK (outcome_label IN
                                    ('improved','degraded','neutral','insufficient_data')),
    metric_after_dos        NUMERIC(10,2),
    metric_after_wape       NUMERIC(8,4),
    outcome_delta           NUMERIC(10,4),

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index: pending outcome checks (checked by outcome-checker job)
CREATE INDEX IF NOT EXISTS idx_aro_outcome_due
    ON ai_recommendation_outcomes (outcome_check_due_at)
    WHERE outcome_label IS NULL;

-- Index: look up outcomes for a specific insight
CREATE INDEX IF NOT EXISTS idx_aro_insight_id
    ON ai_recommendation_outcomes (insight_id);

-- Index: scorecard queries (by type + confidence + period)
CREATE INDEX IF NOT EXISTS idx_aro_type_decision
    ON ai_recommendation_outcomes (insight_type, planner_decision, created_at DESC);
