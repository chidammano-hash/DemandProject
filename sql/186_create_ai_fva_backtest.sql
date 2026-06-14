-- Migration 186 — AI Planner Forecast Value Add (FVA) backtest tables.
-- Spec: docs/specs/PRD/PRD-ai-planner-fva-backtest.md (§5 Data Model)
-- Stores walk-forward backtest runs where the AI Planner emits forecast-adjustment
-- recommendations against the champion baseline, the recommendations are applied
-- deterministically, and accuracy is measured against actuals.

-- ═══════════════════════════════════════════════════════════════════════════
-- Run metadata: one row per backtest invocation
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ai_fva_backtest_run (
    run_id              UUID PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    status              TEXT NOT NULL CHECK (status IN ('running','succeeded','failed','cancelled')),
    window_months       INT  NOT NULL DEFAULT 10,
    as_of_date          DATE NOT NULL,
    horizon_months      INT  NOT NULL DEFAULT 3,
    sample_strategy     JSONB NOT NULL,         -- {mode, pct, min_dfus, max_dfus, strata, ...}
    provider            TEXT NOT NULL,          -- ollama | anthropic | openai | openai_compat
    ai_model            TEXT NOT NULL,          -- e.g. qwen2.5:32b, claude-opus-4-7
    prompt_version      TEXT NOT NULL,
    apply_guardrails    JSONB NOT NULL,
    n_dfus_sampled      INT,
    n_recommendations   INT,
    estimated_cost_usd  NUMERIC(10,2),          -- pre-flight estimate
    actual_cost_usd     NUMERIC(10,2),          -- 0.0 for ollama
    error_message       TEXT,
    created_by          TEXT,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS ix_ai_fva_run_status
    ON ai_fva_backtest_run (status, started_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════
-- Per-DFU per-month AI recommendation
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS fact_ai_forecast_recommendation (
    run_id              UUID NOT NULL REFERENCES ai_fva_backtest_run(run_id) ON DELETE CASCADE,
    item_id             TEXT NOT NULL,
    loc                 TEXT NOT NULL,
    forecast_run_month  DATE NOT NULL,
    recommendation_code TEXT NOT NULL CHECK (recommendation_code IN
        ('KEEP','SCALE_UP','SCALE_DOWN','REPLACE','SHIFT_TIMING','OVERRIDE_TO_BASELINE')),
    pct_change          NUMERIC(6,2),           -- nullable: only for SCALE_UP / SCALE_DOWN
    proposed_qty        JSONB,                  -- nullable: for REPLACE / SHIFT_TIMING
    apply_horizon_months INT NOT NULL DEFAULT 3,
    confidence          NUMERIC(4,3) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    rationale           TEXT,
    evidence_keys       JSONB,
    ai_call_ms          INT,
    ai_tokens_in        INT,
    ai_tokens_out       INT,
    PRIMARY KEY (run_id, item_id, loc, forecast_run_month)
);

CREATE INDEX IF NOT EXISTS ix_ai_rec_run_code
    ON fact_ai_forecast_recommendation (run_id, recommendation_code);

-- ═══════════════════════════════════════════════════════════════════════════
-- Per-DFU per-month-target baseline + AI-adjusted forecast quantities
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS fact_ai_adjusted_forecast (
    run_id              UUID NOT NULL REFERENCES ai_fva_backtest_run(run_id) ON DELETE CASCADE,
    item_id             TEXT NOT NULL,
    loc                 TEXT NOT NULL,
    forecast_run_month  DATE NOT NULL,          -- month T when recommendation was issued
    target_month        DATE NOT NULL,          -- month being forecast (T+1 .. T+H)
    lag                 INT  NOT NULL,          -- target_month - forecast_run_month, in months
    baseline_qty        NUMERIC NOT NULL,
    ai_qty              NUMERIC NOT NULL,
    actual_qty          NUMERIC,                -- backfilled from sales for accuracy calc
    PRIMARY KEY (run_id, item_id, loc, forecast_run_month, target_month)
);

CREATE INDEX IF NOT EXISTS ix_ai_adj_run_lag
    ON fact_ai_adjusted_forecast (run_id, lag);

-- ═══════════════════════════════════════════════════════════════════════════
-- Audit log: full AI context payload + raw response per call
-- Stored separately from recommendation table because payloads are large.
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ai_planner_backtest_audit (
    run_id              UUID NOT NULL REFERENCES ai_fva_backtest_run(run_id) ON DELETE CASCADE,
    item_id             TEXT NOT NULL,
    loc                 TEXT NOT NULL,
    forecast_run_month  DATE NOT NULL,
    context_payload     JSONB NOT NULL,         -- everything the AI saw
    ai_response_raw     JSONB NOT NULL,         -- raw LLM JSON response
    PRIMARY KEY (run_id, item_id, loc, forecast_run_month)
);

-- ═══════════════════════════════════════════════════════════════════════════
-- Materialized view: overall FVA per run (baseline vs AI WAPE + lift)
-- WAPE = 100 - 100 * sum(|F-A|) / |sum(A)|  (per CLAUDE.md "Formulas")
-- ═══════════════════════════════════════════════════════════════════════════
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ai_fva_overall AS
WITH per_dfu AS (
    SELECT
        f.run_id,
        f.item_id,
        f.loc,
        SUM(ABS(f.baseline_qty - COALESCE(f.actual_qty, 0))) AS sae_baseline,
        SUM(ABS(f.ai_qty       - COALESCE(f.actual_qty, 0))) AS sae_ai,
        ABS(SUM(COALESCE(f.actual_qty, 0)))                  AS abs_sum_actual
    FROM fact_ai_adjusted_forecast f
    WHERE f.actual_qty IS NOT NULL
    GROUP BY f.run_id, f.item_id, f.loc
)
SELECT
    p.run_id,
    100.0 - 100.0 * SUM(p.sae_baseline) / NULLIF(SUM(p.abs_sum_actual), 0) AS baseline_wape_pct,
    100.0 - 100.0 * SUM(p.sae_ai)       / NULLIF(SUM(p.abs_sum_actual), 0) AS ai_wape_pct,
    (100.0 - 100.0 * SUM(p.sae_ai)       / NULLIF(SUM(p.abs_sum_actual), 0))
      - (100.0 - 100.0 * SUM(p.sae_baseline) / NULLIF(SUM(p.abs_sum_actual), 0)) AS lift_pct,
    COUNT(DISTINCT (p.item_id, p.loc))                                     AS n_dfus,
    SUM(CASE WHEN p.sae_ai < p.sae_baseline THEN 1 ELSE 0 END)             AS n_winners,
    SUM(CASE WHEN p.sae_ai > p.sae_baseline THEN 1 ELSE 0 END)             AS n_losers,
    SUM(CASE WHEN p.sae_ai = p.sae_baseline THEN 1 ELSE 0 END)             AS n_ties,
    100.0 * SUM(CASE WHEN p.sae_ai < p.sae_baseline THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0)                                              AS win_rate_pct
FROM per_dfu p
GROUP BY p.run_id;

CREATE UNIQUE INDEX IF NOT EXISTS ix_mv_ai_fva_overall_run
    ON mv_ai_fva_overall (run_id);

-- ═══════════════════════════════════════════════════════════════════════════
-- Materialized view: FVA broken down by recommendation type
-- ═══════════════════════════════════════════════════════════════════════════
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ai_fva_by_recommendation AS
WITH joined AS (
    SELECT
        f.run_id,
        r.recommendation_code,
        f.item_id,
        f.loc,
        f.target_month,
        ABS(f.baseline_qty - COALESCE(f.actual_qty, 0)) AS ae_baseline,
        ABS(f.ai_qty       - COALESCE(f.actual_qty, 0)) AS ae_ai,
        ABS(COALESCE(f.actual_qty, 0))                  AS abs_actual,
        r.confidence,
        r.pct_change
    FROM fact_ai_adjusted_forecast f
    JOIN fact_ai_forecast_recommendation r
      ON r.run_id = f.run_id
     AND r.item_id = f.item_id
     AND r.loc = f.loc
     AND r.forecast_run_month = f.forecast_run_month
    WHERE f.actual_qty IS NOT NULL
)
SELECT
    j.run_id,
    j.recommendation_code,
    100.0 - 100.0 * SUM(j.ae_baseline) / NULLIF(SUM(j.abs_actual), 0) AS baseline_wape_pct,
    100.0 - 100.0 * SUM(j.ae_ai)       / NULLIF(SUM(j.abs_actual), 0) AS ai_wape_pct,
    (100.0 - 100.0 * SUM(j.ae_ai)       / NULLIF(SUM(j.abs_actual), 0))
      - (100.0 - 100.0 * SUM(j.ae_baseline) / NULLIF(SUM(j.abs_actual), 0)) AS lift_pct,
    COUNT(*)                                                          AS n_obs,
    AVG(j.confidence)                                                 AS avg_confidence,
    AVG(j.pct_change)                                                 AS avg_pct_change
FROM joined j
GROUP BY j.run_id, j.recommendation_code;

CREATE INDEX IF NOT EXISTS ix_mv_ai_fva_by_rec_run
    ON mv_ai_fva_by_recommendation (run_id, recommendation_code);

-- ═══════════════════════════════════════════════════════════════════════════
-- Materialized view: FVA by month (across the 10-month walk-forward window)
-- ═══════════════════════════════════════════════════════════════════════════
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ai_fva_by_month AS
SELECT
    f.run_id,
    f.forecast_run_month,
    100.0 - 100.0 * SUM(ABS(f.baseline_qty - COALESCE(f.actual_qty, 0)))
        / NULLIF(SUM(ABS(COALESCE(f.actual_qty, 0))), 0) AS baseline_wape_pct,
    100.0 - 100.0 * SUM(ABS(f.ai_qty       - COALESCE(f.actual_qty, 0)))
        / NULLIF(SUM(ABS(COALESCE(f.actual_qty, 0))), 0) AS ai_wape_pct,
    COUNT(DISTINCT (f.item_id, f.loc)) AS n_dfus
FROM fact_ai_adjusted_forecast f
WHERE f.actual_qty IS NOT NULL
GROUP BY f.run_id, f.forecast_run_month;

CREATE INDEX IF NOT EXISTS ix_mv_ai_fva_by_month_run
    ON mv_ai_fva_by_month (run_id, forecast_run_month);

-- ═══════════════════════════════════════════════════════════════════════════
-- Materialized view: per-DFU lift (for drill-down)
-- ═══════════════════════════════════════════════════════════════════════════
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ai_fva_by_dfu AS
SELECT
    f.run_id,
    f.item_id,
    f.loc,
    SUM(ABS(f.baseline_qty - COALESCE(f.actual_qty, 0))) AS sae_baseline,
    SUM(ABS(f.ai_qty       - COALESCE(f.actual_qty, 0))) AS sae_ai,
    SUM(ABS(f.baseline_qty - COALESCE(f.actual_qty, 0)))
      - SUM(ABS(f.ai_qty   - COALESCE(f.actual_qty, 0))) AS abs_error_reduction,
    COUNT(*)                                             AS n_obs,
    BOOL_OR(ABS(f.ai_qty - COALESCE(f.actual_qty, 0))
          < ABS(f.baseline_qty - COALESCE(f.actual_qty, 0))) AS any_winner_obs
FROM fact_ai_adjusted_forecast f
WHERE f.actual_qty IS NOT NULL
GROUP BY f.run_id, f.item_id, f.loc;

CREATE INDEX IF NOT EXISTS ix_mv_ai_fva_by_dfu_run
    ON mv_ai_fva_by_dfu (run_id, item_id, loc);
