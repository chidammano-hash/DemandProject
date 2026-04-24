-- 156_create_fact_shadow_rollout.sql
-- Gen-4 Stream G — Governance / AI-7: Shadow / A-B champion-challenger.
--
-- Records a challenger rollout plan against the current champion. Live
-- traffic is routed to both models; challenger predictions are logged for
-- offline comparison without impacting production forecasts.

CREATE TABLE IF NOT EXISTS fact_shadow_rollout (
    id                BIGSERIAL PRIMARY KEY,

    -- Currently promoted champion this challenger is compared against
    champion_id       VARCHAR(100)    NOT NULL,

    -- Challenger model being shadow-tested
    challenger_id     VARCHAR(100)    NOT NULL,

    -- Fraction of traffic (0.0 - 1.0) that routes predictions to the
    -- challenger. 0.0 = shadow-only (logged, not served); 1.0 = full cutover.
    traffic_pct       NUMERIC(5, 4)   NOT NULL DEFAULT 0.0
                          CHECK (traffic_pct >= 0.0 AND traffic_pct <= 1.0),

    -- Rollout lifecycle timestamps
    start_ts          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    end_ts            TIMESTAMPTZ,                     -- NULL while active

    -- Rollout state: 'proposed', 'active', 'completed', 'aborted'
    status            VARCHAR(20)     NOT NULL DEFAULT 'proposed'
                          CHECK (status IN ('proposed', 'active', 'completed', 'aborted')),

    -- Aggregated live-result metrics (populated at `completed`)
    observed_metrics  JSONB,

    -- Free-text rationale / decision notes
    notes             TEXT,

    -- Audit fields
    created_by        TEXT            DEFAULT 'manual',
    created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Primary lookup: only one active rollout per challenger at a time.
CREATE UNIQUE INDEX IF NOT EXISTS uq_shadow_rollout_active
    ON fact_shadow_rollout (challenger_id)
    WHERE status = 'active';

-- History queries by champion
CREATE INDEX IF NOT EXISTS idx_shadow_rollout_champion
    ON fact_shadow_rollout (champion_id, start_ts DESC);
