-- 196: SKU Chatbot persistence (spec docs/specs/06-ai-platform/07-sku-chatbot.md, Phase 3).
-- Session continuity + message history + per-turn call log (cost / usage / tools / latency).
-- All writes are best-effort: common/ai/sku_chat/store.py degrades gracefully if these
-- tables are absent, so the chatbot streams whether or not this migration has run.

-- One conversation, scoped to a single SKU.
CREATE TABLE IF NOT EXISTS sku_chat_session (
    session_id      TEXT PRIMARY KEY,           -- client/uuid4 correlation id
    item_id         TEXT NOT NULL,
    customer_group  TEXT NOT NULL DEFAULT '',
    loc             TEXT NOT NULL,
    created_by      TEXT,                        -- optional X-User header
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sku_chat_session_sku
    ON sku_chat_session (item_id, loc);

-- One turn (user question or assistant answer).
CREATE TABLE IF NOT EXISTS sku_chat_message (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sku_chat_session (session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,                   -- 'user' | 'assistant'
    content     TEXT NOT NULL DEFAULT '',
    model       TEXT,                            -- model that produced an assistant turn
    tier        TEXT,                            -- fast | standard | deep | custom
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sku_chat_message_session
    ON sku_chat_message (session_id, id);

-- Per-turn observability: cost, token usage, tool count, latency, truncation.
-- total_cost_usd is the Agent SDK's client-side estimate (not billing truth).
CREATE TABLE IF NOT EXISTS sku_chat_call_log (
    id                BIGSERIAL PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sku_chat_session (session_id) ON DELETE CASCADE,
    message_id        BIGINT REFERENCES sku_chat_message (id) ON DELETE SET NULL,
    model             TEXT,
    tier              TEXT,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    cache_read_tokens INTEGER,
    total_cost_usd    NUMERIC(12,6),
    tool_calls        INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER,
    truncated         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sku_chat_call_log_session
    ON sku_chat_call_log (session_id, created_at);
