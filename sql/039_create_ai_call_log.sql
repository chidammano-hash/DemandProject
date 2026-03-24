-- IPAIfeature1 (enhancement): AI Planning Agent observability log
-- Records per-turn token usage, latency, and tool call outcomes.
-- Run: psql $DATABASE_URL -f sql/039_create_ai_call_log.sql

CREATE TABLE IF NOT EXISTS ai_call_log (
    log_id              BIGSERIAL PRIMARY KEY,
    scan_run_id         TEXT NOT NULL,
    dfu_key             TEXT,                          -- "item_id@loc" or NULL for portfolio turn
    provider            TEXT NOT NULL,                 -- "openai" | "anthropic"
    model               TEXT NOT NULL,
    turn_number         INTEGER NOT NULL,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    total_tokens        INTEGER,
    latency_ms          INTEGER,
    tool_name           TEXT,                          -- NULL for LLM turn rows
    tool_success        BOOLEAN,
    error_type          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partial index: only tool call rows (for failure analysis)
CREATE INDEX IF NOT EXISTS idx_ai_call_log_tool_failures
    ON ai_call_log (scan_run_id, tool_name, created_at)
    WHERE tool_success = false;

-- Index for cost / latency queries by scan
CREATE INDEX IF NOT EXISTS idx_ai_call_log_scan_run
    ON ai_call_log (scan_run_id, created_at DESC);

-- Index for model-level aggregation
CREATE INDEX IF NOT EXISTS idx_ai_call_log_model
    ON ai_call_log (model, provider, created_at DESC);
