-- 197: SKU Chatbot — staged champion-forecast adjustments awaiting planner approval.
-- Spec docs/specs/06-ai-platform/07-sku-chatbot.md (agentic adjust, approval-gated).
--
-- The chatbot's `apply_champion_adjustment` tool STAGES a guardrail-validated
-- proposal here (reusing common/ai/champion_adjust_service.adjust_dfu); the actual
-- write to fact_ai_champion_forecast happens only when the planner approves
-- (POST /sku-chat/adjustment/{id}, which calls save_adjustment). The agent never
-- writes the forecast directly — its only write is a 'pending' row in this table.
-- No FK to sku_chat_session: a turn may run with persistence disabled.

CREATE TABLE IF NOT EXISTS sku_chat_pending_adjustment (
    approval_id     TEXT PRIMARY KEY,
    session_id      TEXT,
    item_id         TEXT NOT NULL,
    customer_group  TEXT NOT NULL DEFAULT '',
    loc             TEXT NOT NULL,
    preview         JSONB NOT NULL,               -- AdjustPreview.to_dict()
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    created_by      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sku_chat_pending_adj_session
    ON sku_chat_pending_adjustment (session_id, status);
