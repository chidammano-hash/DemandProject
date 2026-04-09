-- 121_candidate_forecast_and_promotion.sql
-- Candidate Forecast Staging & Model Promotion Audit
--
-- Part of the Train → Generate → Load → Promote workflow.
-- All backtest/forecast predictions land in fact_candidate_forecast first.
-- Only the promoted model (single or champion-selected) gets copied to
-- fact_production_forecast.  model_promotion_log provides a full audit
-- trail of every promotion/demotion event.

-- ═══════════════════════════════════════════════════════════════════════════
-- 1. CANDIDATE FORECAST TABLE
-- ═══════════════════════════════════════════════════════════════════════════
-- Staging area for all model predictions.  Every backtest-load writes here
-- regardless of model.  Rows are flagged is_promoted = TRUE when the model
-- is promoted to production, and the corresponding rows are copied into
-- fact_production_forecast.
-- Grain: item_id + loc + model_id + forecast_month

CREATE TABLE IF NOT EXISTS fact_candidate_forecast (
    id                  BIGSERIAL PRIMARY KEY,

    -- DFU identity (item + location)
    item_id             VARCHAR(50)     NOT NULL,   -- dim_item surrogate or natural key
    loc                 VARCHAR(50)     NOT NULL,   -- dim_location key

    -- Model that generated this prediction
    model_id            VARCHAR(100)    NOT NULL,   -- e.g. 'lgbm_cluster', 'nbeats', 'chronos_bolt'

    -- Forecast period
    forecast_month      DATE            NOT NULL,   -- first day of the month (always YYYY-MM-01)

    -- Predicted quantity and confidence interval
    forecast_qty        NUMERIC(12, 2)  NOT NULL,   -- point forecast
    forecast_qty_lower  NUMERIC(12, 2),             -- P10 confidence interval lower bound
    forecast_qty_upper  NUMERIC(12, 2),             -- P90 confidence interval upper bound

    -- Actuals and accuracy metrics (populated during backtest evaluation)
    actual_qty          NUMERIC(12, 2),             -- actual demand for the period (NULL if future)
    accuracy_pct        NUMERIC(8, 4),              -- 100 - WAPE, per-row accuracy
    wape                NUMERIC(8, 4),              -- weighted absolute percentage error
    bias                NUMERIC(8, 4),              -- (forecast / actual) - 1

    -- Forecast metadata
    horizon_months      SMALLINT,                   -- 1=T+1, 2=T+2, etc.
    cluster_id          TEXT,                        -- ml_cluster label used during training

    -- Lineage: links back to the backtest run that produced this row
    backtest_run_id     INTEGER                     -- FK to backtest_run.id
                            REFERENCES backtest_run(id)
                            ON DELETE SET NULL,

    -- Load tracking
    loaded_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),   -- when this row was inserted

    -- Promotion lifecycle
    is_promoted         BOOLEAN         NOT NULL DEFAULT FALSE,   -- TRUE after promotion to production
    promoted_at         TIMESTAMPTZ                               -- timestamp of promotion event
);

-- Unique: one prediction per (DFU, model, month) — prevents duplicate loads
CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_fcst_dfu_model_month
    ON fact_candidate_forecast (item_id, loc, model_id, forecast_month);

-- Filter by model (compare models, per-model accuracy queries)
CREATE INDEX IF NOT EXISTS idx_candidate_fcst_model_id
    ON fact_candidate_forecast (model_id);

-- Lookup by DFU (item + loc) for per-SKU comparison across models
CREATE INDEX IF NOT EXISTS idx_candidate_fcst_item_loc
    ON fact_candidate_forecast (item_id, loc);

-- Quickly find promoted vs. non-promoted rows
CREATE INDEX IF NOT EXISTS idx_candidate_fcst_promoted
    ON fact_candidate_forecast (is_promoted) WHERE is_promoted;

-- Join back to backtest_run for run-level aggregation
CREATE INDEX IF NOT EXISTS idx_candidate_fcst_backtest_run
    ON fact_candidate_forecast (backtest_run_id);


-- ═══════════════════════════════════════════════════════════════════════════
-- 2. MODEL PROMOTION LOG
-- ═══════════════════════════════════════════════════════════════════════════
-- Immutable audit trail for every promotion and demotion event.
-- Each promotion creates a new row; when a newer promotion occurs the
-- previous row's demoted_at is set and is_active flipped to FALSE.
-- Supports both single-model promotions and champion-experiment-based
-- promotions (where the champion strategy selects per-DFU winners).

CREATE TABLE IF NOT EXISTS model_promotion_log (
    id                      SERIAL PRIMARY KEY,

    -- Which model was promoted (or 'champion' for champion-strategy promotions)
    model_id                VARCHAR(100)    NOT NULL,

    -- Promotion type: 'single' = one model promoted globally,
    --                 'champion' = champion experiment selected per-DFU winners
    promotion_type          VARCHAR(20)     NOT NULL DEFAULT 'single'
                                CHECK (promotion_type IN ('single', 'champion')),

    -- Optional link to champion_experiment (NULL for single-model promotions)
    champion_experiment_id  INTEGER
                                REFERENCES champion_experiment(experiment_id)
                                ON DELETE SET NULL,

    -- Version tag written to fact_production_forecast.plan_version
    plan_version            VARCHAR(30),                -- e.g. '2026-04', 'v3.1'

    -- Timestamps
    promoted_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),   -- when promotion occurred
    demoted_at              TIMESTAMPTZ,                              -- set when replaced by a newer promotion

    -- Active flag: only one promotion should be active at any time
    is_active               BOOLEAN         NOT NULL DEFAULT TRUE,

    -- Volume metrics (populated at promotion time)
    dfu_count               INTEGER,        -- number of distinct item+loc combos promoted
    total_rows              INTEGER,        -- total rows copied to fact_production_forecast

    -- Audit fields
    promoted_by             TEXT            DEFAULT 'manual',   -- 'manual', 'api', 'scheduler', user id
    notes                   TEXT,                               -- free-text description of why this promotion was done

    -- Config snapshot for reproducibility
    config_snapshot         JSONB                               -- champion strategy params, model config, etc.
);

-- Fast lookup for the currently active promotion (most queries filter on this)
CREATE INDEX IF NOT EXISTS idx_promotion_log_active
    ON model_promotion_log (is_active) WHERE is_active;

-- Time-ordered audit trail
CREATE INDEX IF NOT EXISTS idx_promotion_log_promoted_at
    ON model_promotion_log (promoted_at DESC);

-- Filter by model for model-specific promotion history
CREATE INDEX IF NOT EXISTS idx_promotion_log_model_id
    ON model_promotion_log (model_id);


-- ═══════════════════════════════════════════════════════════════════════════
-- 3. ALTER backtest_run — CANDIDATE LOAD TRACKING
-- ═══════════════════════════════════════════════════════════════════════════
-- Extend backtest_run with columns to track whether its predictions have
-- been loaded into fact_candidate_forecast (separate from the existing
-- is_loaded_to_db which tracks loading to the legacy prediction tables).

ALTER TABLE backtest_run
    ADD COLUMN IF NOT EXISTS is_loaded_to_candidate  BOOLEAN     NOT NULL DEFAULT FALSE;
    -- TRUE when this run's predictions are loaded into fact_candidate_forecast

ALTER TABLE backtest_run
    ADD COLUMN IF NOT EXISTS candidate_loaded_at     TIMESTAMPTZ;
    -- Timestamp of when candidate load completed
