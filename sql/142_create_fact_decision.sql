-- Gen-4 Roadmap Cross-cutting: Three-tier memory — Episodic memory store.
--
-- Records the *outcome* of each AI decision after it has been applied,
-- so agents can learn from past actions. Ties each outcome back to the
-- originating `ai_decision_ledger` row via FK.
--
-- Companion Python: common/ai/memory.py (EpisodicMemory class).

CREATE TABLE IF NOT EXISTS fact_decision (
    id              BIGSERIAL       PRIMARY KEY,
    decision_id     BIGINT          NOT NULL REFERENCES ai_decision_ledger(id) ON DELETE RESTRICT,
    outcome_ts      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),   -- when outcome was observed
    outcome_json    JSONB           NOT NULL DEFAULT '{}'::jsonb,  -- arbitrary observation details
    succeeded       BOOLEAN         NOT NULL,                 -- reward signal for learning
    reward          REAL,                                     -- optional scalar reward (-1..1 typical)
    notes           TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_decision_decision
    ON fact_decision (decision_id);

CREATE INDEX IF NOT EXISTS idx_fact_decision_outcome_ts
    ON fact_decision (outcome_ts DESC);

CREATE INDEX IF NOT EXISTS idx_fact_decision_succeeded
    ON fact_decision (succeeded, outcome_ts DESC);

COMMENT ON TABLE fact_decision IS
    'Episodic memory: outcomes of AI decisions. FK -> ai_decision_ledger.id. '
    'Consumed by common/ai/memory.py:EpisodicMemory.';
