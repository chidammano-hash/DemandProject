-- 190_create_ai_champion_forecast.sql
-- AI Champion forward adjuster (spec docs/specs/02-forecasting/27-ai-champion-forecast.md)
--
-- The repurposed AI planner reads the promoted champion production forecast
-- (fact_production_forecast, model_id='champion'), applies a per-DFU AI
-- adjustment, and writes a NEW forward forecast here with model_id='ai_champion'.
-- Forward-only: no historical backtest, no accuracy grading, no actuals.
--
-- A dedicated table (not fact_production_forecast) because that table's unique
-- index is (plan_version, item_id, loc, forecast_month) with NO model_id, so
-- champion and ai_champion rows for the same DFU-month would collide. Keeping
-- ai_champion separate also co-locates the AI rationale with the adjusted qty.

-- ── Run header ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ai_champion_run (
    run_id          UUID            PRIMARY KEY,
    plan_version    VARCHAR(30)     NOT NULL,        -- champion plan_version adjusted, e.g. '2026-04'
    provider        VARCHAR(30)     NOT NULL,        -- ollama | anthropic | openai | openai_compat
    ai_model        VARCHAR(100)    NOT NULL,        -- e.g. 'llama3.1:8b' or 'claude-opus-4-7'
    prompt_version  VARCHAR(20),                     -- champion_adjuster PROMPT_VERSION
    status          VARCHAR(20)     NOT NULL DEFAULT 'running',  -- running | succeeded | failed
    n_dfus          INTEGER         NOT NULL DEFAULT 0,          -- DFUs considered
    n_adjusted      INTEGER         NOT NULL DEFAULT 0,          -- DFUs the AI changed (non-KEEP)
    est_cost_usd    NUMERIC(10,4),                   -- pre-flight estimate (0 for ollama)
    error           TEXT,
    started_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ai_champion_run_status
    ON ai_champion_run (status, started_at DESC);

-- ── Per-DFU per-month adjusted forecast + rationale ─────────────────────────
CREATE TABLE IF NOT EXISTS fact_ai_champion_forecast (
    id                  BIGSERIAL       PRIMARY KEY,
    run_id              UUID            NOT NULL REFERENCES ai_champion_run(run_id) ON DELETE CASCADE,
    plan_version        VARCHAR(30)     NOT NULL,
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    forecast_month      DATE            NOT NULL,    -- first day of month (always future)
    horizon_months      SMALLINT,                    -- 1..H offset from the plan month
    champion_qty        NUMERIC(14,2)   NOT NULL,    -- the champion baseline (pre-adjustment)
    ai_qty              NUMERIC(14,2)   NOT NULL,    -- the AI Champion forecast (post-adjustment)
    model_id            VARCHAR(100)    NOT NULL DEFAULT 'ai_champion',
    recommendation_code VARCHAR(30)     NOT NULL,    -- KEEP | SCALE_UP | SCALE_DOWN | REPLACE | SHIFT_TIMING | OVERRIDE_TO_BASELINE
    pct_change          NUMERIC(7,2),                -- SCALE_* magnitude
    confidence          NUMERIC(4,3),                -- 0..1 LLM confidence
    rationale           TEXT,                        -- 1-3 sentence AI explanation
    evidence_keys       TEXT[],                      -- short evidence tags
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- One ai_champion forecast per (plan_version, DFU, forecast_month).
CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_champion_version_dfu_month
    ON fact_ai_champion_forecast (plan_version, item_id, loc, forecast_month);

CREATE INDEX IF NOT EXISTS idx_ai_champion_dfu_month
    ON fact_ai_champion_forecast (item_id, loc, forecast_month);

CREATE INDEX IF NOT EXISTS idx_ai_champion_run
    ON fact_ai_champion_forecast (run_id);
