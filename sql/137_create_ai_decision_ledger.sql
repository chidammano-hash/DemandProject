-- Gen-4 Roadmap AI-10 P0: Immutable AI decision ledger with hash-chained rows.
--
-- Purpose: every auto-action, suggestion, or override issued by an AI agent
-- is recorded as one immutable row. A SHA-256 chain links each row to the
-- previous one so tampering is detectable. A BEFORE UPDATE/DELETE trigger
-- blocks mutations.
--
-- Companion Python helper: common/ai/decision_ledger.py
-- Companion policy engine:  common/ai/policy_engine.py (see spec 06-ai-platform).

CREATE TABLE IF NOT EXISTS ai_decision_ledger (
    id              BIGSERIAL       PRIMARY KEY,
    ts              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    agent_id        VARCHAR(64)     NOT NULL,        -- e.g. 'demand_agent', 'exception_agent'
    action_type     VARCHAR(64)     NOT NULL,        -- e.g. 'promote_model', 'auto_resolve_exception'
    autonomy_tier   VARCHAR(32)     NOT NULL,        -- advisory | suggestive | auto_within_policy | autonomous
    subject_kind    VARCHAR(64),                     -- e.g. 'dfu', 'po', 'model_id'
    subject_id      VARCHAR(128),                    -- the specific subject identifier
    payload         JSONB           NOT NULL DEFAULT '{}'::jsonb,  -- action details (inputs, outputs)
    policy_id       VARCHAR(64),                     -- policy from agent_autonomy.yaml that authorized this
    prev_hash       CHAR(64)        NOT NULL,        -- previous row's row_hash (or 64-hex '0' for genesis)
    row_hash        CHAR(64)        NOT NULL,        -- sha256 over (id, ts, agent_id, action_type, autonomy_tier, subject_kind, subject_id, payload::text, policy_id, prev_hash)
    actor           VARCHAR(64),                     -- 'system' or user_id when a human overrode
    outcome         VARCHAR(32)                                    -- 'applied' | 'rolled_back' | 'rejected' | 'superseded'
);

CREATE INDEX IF NOT EXISTS idx_ai_ledger_ts        ON ai_decision_ledger (ts DESC);
CREATE INDEX IF NOT EXISTS idx_ai_ledger_agent     ON ai_decision_ledger (agent_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ai_ledger_subject   ON ai_decision_ledger (subject_kind, subject_id);
CREATE INDEX IF NOT EXISTS idx_ai_ledger_tier      ON ai_decision_ledger (autonomy_tier);

-- ── Append-only enforcement ─────────────────────────────────────────────
-- Blocks any UPDATE or DELETE. Rows can only be appended. Outcome changes
-- are modeled as new rows with `outcome='superseded'` on the prior row
-- (but even that update is blocked — the new row carries the effective
-- state; prior rows are immutable by design).

CREATE OR REPLACE FUNCTION ai_decision_ledger_block_mutation() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
      'ai_decision_ledger is append-only (tried % on id=%)',
      TG_OP, OLD.id;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ai_ledger_no_update ON ai_decision_ledger;
CREATE TRIGGER trg_ai_ledger_no_update
    BEFORE UPDATE OR DELETE ON ai_decision_ledger
    FOR EACH ROW EXECUTE FUNCTION ai_decision_ledger_block_mutation();

-- ── Hash-chain enforcement on insert ────────────────────────────────────
-- Verifies that each new row's prev_hash matches the latest row's row_hash
-- (first row uses GENESIS_HASH) and that row_hash matches the recomputed
-- SHA-256 over the canonical tuple. Rejects rows that fail either check.

CREATE OR REPLACE FUNCTION ai_decision_ledger_verify_chain() RETURNS TRIGGER AS $$
DECLARE
    expected_prev CHAR(64);
    computed      CHAR(64);
BEGIN
    -- Genesis sentinel: 64 zeros
    SELECT COALESCE(
        (SELECT row_hash FROM ai_decision_ledger ORDER BY id DESC LIMIT 1),
        repeat('0', 64)
    ) INTO expected_prev;

    IF NEW.prev_hash IS DISTINCT FROM expected_prev THEN
        RAISE EXCEPTION
          'ai_decision_ledger chain break: expected prev_hash=% got %',
          expected_prev, NEW.prev_hash;
    END IF;

    computed := encode(
        digest(
            COALESCE(NEW.agent_id, '') || '|' ||
            COALESCE(NEW.action_type, '') || '|' ||
            COALESCE(NEW.autonomy_tier, '') || '|' ||
            COALESCE(NEW.subject_kind, '') || '|' ||
            COALESCE(NEW.subject_id, '') || '|' ||
            COALESCE(NEW.payload::text, '{}') || '|' ||
            COALESCE(NEW.policy_id, '') || '|' ||
            NEW.prev_hash,
            'sha256'
        ),
        'hex'
    );

    IF NEW.row_hash IS DISTINCT FROM computed THEN
        RAISE EXCEPTION
          'ai_decision_ledger hash mismatch: computed=% supplied=%',
          computed, NEW.row_hash;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- pgcrypto provides `digest()` — install if missing.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DROP TRIGGER IF EXISTS trg_ai_ledger_verify ON ai_decision_ledger;
CREATE TRIGGER trg_ai_ledger_verify
    BEFORE INSERT ON ai_decision_ledger
    FOR EACH ROW EXECUTE FUNCTION ai_decision_ledger_verify_chain();
