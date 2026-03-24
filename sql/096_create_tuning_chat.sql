-- 096: Tuning Chat — persistent AI-powered LGBM tuning sessions
--
-- Two tables for chat-based interactive hyperparameter tuning sessions.
-- Messages store the full conversation including AI recommendations,
-- run starts, completions, and analysis results.

-- ── Sessions ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tuning_chat_session (
    session_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title         TEXT NOT NULL DEFAULT 'New Tuning Session',
    status        TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active', 'archived')),
    context       JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tuning_chat_session_status
    ON tuning_chat_session (status, updated_at DESC);

-- ── Messages ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tuning_chat_message (
    message_id    SERIAL PRIMARY KEY,
    session_id    UUID NOT NULL
                      REFERENCES tuning_chat_session(session_id) ON DELETE CASCADE,
    role          TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content       TEXT NOT NULL,
    message_type  TEXT NOT NULL DEFAULT 'text'
                      CHECK (message_type IN (
                          'text',
                          'recommendation',
                          'run_started',
                          'run_completed',
                          'run_failed',
                          'analysis',
                          'error'
                      )),
    metadata      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tuning_chat_msg_session
    ON tuning_chat_message (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_tuning_chat_msg_type
    ON tuning_chat_message (message_type);
