-- Gen-4 Roadmap AI-8 (Stream H Phase 2): Reversible Action Ledger.
--
-- Every closed-loop action applied by the exception orchestrator writes a
-- row here. Rows carry a `rollback_payload` describing how to reverse the
-- action plus an `expires_at` horizon (default applied_at + 24h). A KPI
-- quiet-period sweeper scans rows with status='applied' and, on KPI
-- regression detection, invokes the rollback and flips status='rolled_back'.
--
-- Companion Python helper: common/ai/reversible.py
-- Companion orchestrator:  common/ai/orchestrator.py

CREATE TABLE IF NOT EXISTS fact_reversible_action (
    id                BIGSERIAL      PRIMARY KEY,
    action_type       VARCHAR(64)    NOT NULL,            -- e.g. 'expedite_transfer', 'emergency_po', 'reallocate'
    target_kind       VARCHAR(64)    NOT NULL,            -- e.g. 'exception_id', 'po_id', 'transfer_id'
    target_id         VARCHAR(128)   NOT NULL,            -- specific identifier for the target entity
    applied_at        TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    expires_at        TIMESTAMPTZ    NOT NULL,            -- quiet-period deadline (applied_at + 24h typical)
    rollback_payload  JSONB          NOT NULL DEFAULT '{}'::jsonb,  -- reverse-action spec (qty, target, sql, etc.)
    status            VARCHAR(32)    NOT NULL DEFAULT 'applied',    -- applied | rolled_back | expired | confirmed
    applied_by        VARCHAR(64),                                   -- agent or user id
    rolled_back_at    TIMESTAMPTZ,                                   -- when sweeper reverted (NULL unless rolled back)
    rollback_reason   TEXT,                                          -- free-text / KPI regression code
    ledger_id         BIGINT,                                        -- optional link to ai_decision_ledger.id
    load_ts           TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reversible_status_expires
    ON fact_reversible_action (status, expires_at);
CREATE INDEX IF NOT EXISTS idx_reversible_target
    ON fact_reversible_action (target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_reversible_applied_at
    ON fact_reversible_action (applied_at DESC);

-- Valid statuses enforced via check constraint (cheaper than a trigger).
ALTER TABLE fact_reversible_action
    DROP CONSTRAINT IF EXISTS chk_reversible_status;
ALTER TABLE fact_reversible_action
    ADD CONSTRAINT chk_reversible_status
    CHECK (status IN ('applied', 'rolled_back', 'expired', 'confirmed'));
