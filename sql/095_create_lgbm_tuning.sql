-- LGBM Tuning Schema
-- Tracks iterative LightGBM backtest experiments: run-level results,
-- per-timeframe breakdowns, and pairwise comparisons between runs.

-- ── Run-level summary ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lgbm_tuning_run (
    run_id          SERIAL PRIMARY KEY,
    run_label       TEXT NOT NULL,           -- e.g. "baseline", "v2_ts_features", "v3_regularization"
    model_id        TEXT NOT NULL DEFAULT 'lgbm_cluster',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed')),
    params          JSONB,                   -- full LGBM params dict
    feature_count   INTEGER,
    features        JSONB,                   -- list of feature names used
    accuracy_pct    NUMERIC(6,2),
    wape            NUMERIC(6,2),
    bias            NUMERIC(8,4),
    n_predictions   BIGINT,
    n_dfus          INTEGER,
    metadata        JSONB,                   -- full backtest_metadata.json contents
    notes           TEXT,
    backup_path     TEXT                     -- filesystem path to backed-up data
);

CREATE INDEX IF NOT EXISTS idx_tuning_run_status ON lgbm_tuning_run (status);
CREATE INDEX IF NOT EXISTS idx_tuning_run_model  ON lgbm_tuning_run (model_id);

-- ── Per-timeframe breakdown ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lgbm_tuning_timeframe (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id) ON DELETE CASCADE,
    timeframe       TEXT NOT NULL,           -- A-J
    train_end       DATE,
    predict_start   DATE,
    predict_end     DATE,
    n_predictions   INTEGER,
    accuracy_pct    NUMERIC(6,2),
    wape            NUMERIC(6,2),
    bias            NUMERIC(8,4),
    UNIQUE (run_id, timeframe)
);

CREATE INDEX IF NOT EXISTS idx_tuning_tf_run ON lgbm_tuning_timeframe (run_id);

-- ── Per-cluster accuracy breakdown ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lgbm_tuning_cluster (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id) ON DELETE CASCADE,
    cluster_type    TEXT NOT NULL,            -- 'ml_cluster' or 'business_cluster'
    cluster_value   TEXT NOT NULL,
    n_predictions   INTEGER,
    n_dfus          INTEGER,
    accuracy_pct    NUMERIC(6,2),
    wape            NUMERIC(6,2),
    bias            NUMERIC(8,4),
    UNIQUE (run_id, cluster_type, cluster_value)
);

CREATE INDEX IF NOT EXISTS idx_tuning_cluster_run ON lgbm_tuning_cluster (run_id, cluster_type);

-- ── Per-month accuracy breakdown ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lgbm_tuning_month (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id) ON DELETE CASCADE,
    month_start     DATE NOT NULL,
    n_predictions   INTEGER,
    n_dfus          INTEGER,
    accuracy_pct    NUMERIC(6,2),
    wape            NUMERIC(6,2),
    bias            NUMERIC(8,4),
    UNIQUE (run_id, month_start)
);

CREATE INDEX IF NOT EXISTS idx_tuning_month_run ON lgbm_tuning_month (run_id);

-- ── Pairwise comparison between two runs ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS lgbm_tuning_comparison (
    id                      SERIAL PRIMARY KEY,
    baseline_run_id         INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id),
    candidate_run_id        INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id),
    created_at              TIMESTAMPTZ DEFAULT now(),
    delta_accuracy          NUMERIC(6,2),    -- candidate - baseline
    delta_wape              NUMERIC(6,2),    -- candidate - baseline (negative is better)
    delta_bias              NUMERIC(8,4),
    per_timeframe_detail    JSONB,           -- array of per-timeframe deltas
    verdict                 TEXT
                                CHECK (verdict IN ('improved', 'degraded', 'neutral')),
    UNIQUE (baseline_run_id, candidate_run_id)
);

-- ── Latest completed runs view ─────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_lgbm_tuning_latest AS
SELECT
    run_id,
    run_label,
    model_id,
    started_at,
    completed_at,
    accuracy_pct,
    wape,
    bias,
    n_predictions,
    n_dfus,
    feature_count,
    params,
    notes
FROM lgbm_tuning_run
WHERE status = 'completed'
ORDER BY accuracy_pct DESC NULLS LAST;
