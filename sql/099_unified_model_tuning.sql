-- 099: Unified Model Tuning Studio schema additions
-- Adds per-execution-lag accuracy breakdowns, promotion audit trail,
-- per-lag-per-cluster drill-down, and extends lgbm_tuning_run with
-- job_id, template_id, and expanded status CHECK.

-- ── Per-execution-lag accuracy breakdown ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS lgbm_tuning_lag (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id) ON DELETE CASCADE,
    exec_lag        SMALLINT NOT NULL CHECK (exec_lag BETWEEN 0 AND 4),
    n_predictions   INTEGER NOT NULL DEFAULT 0,
    n_dfus          INTEGER NOT NULL DEFAULT 0,
    accuracy_pct    NUMERIC(6, 2),
    wape            NUMERIC(6, 2),
    bias            NUMERIC(8, 4),
    UNIQUE (run_id, exec_lag)
);

CREATE INDEX IF NOT EXISTS idx_tuning_lag_run ON lgbm_tuning_lag(run_id);

-- ── Promotion audit trail ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tuning_promotion_log (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id),
    model_id        VARCHAR(50) NOT NULL,
    promoted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_by     VARCHAR(100),           -- user or 'system'
    previous_run_id INTEGER REFERENCES lgbm_tuning_run(run_id),
    params_written  JSONB NOT NULL,
    accuracy_pct    NUMERIC(6, 2),
    wape            NUMERIC(6, 2),
    bias            NUMERIC(8, 4),
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_promotion_log_model
    ON tuning_promotion_log(model_id, promoted_at DESC);

-- ── Per-lag-per-cluster accuracy breakdown ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS lgbm_tuning_lag_cluster (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id) ON DELETE CASCADE,
    exec_lag        SMALLINT NOT NULL CHECK (exec_lag BETWEEN 0 AND 4),
    cluster_type    TEXT NOT NULL,
    cluster_value   TEXT NOT NULL,
    n_predictions   INTEGER NOT NULL DEFAULT 0,
    accuracy_pct    NUMERIC(6, 2),
    wape            NUMERIC(6, 2),
    bias            NUMERIC(8, 4),
    UNIQUE (run_id, exec_lag, cluster_type, cluster_value)
);

CREATE INDEX IF NOT EXISTS idx_tuning_lag_cluster_run
    ON lgbm_tuning_lag_cluster(run_id);

-- ── Alter lgbm_tuning_run: add columns and expand status CHECK ─────────────────
ALTER TABLE lgbm_tuning_run
    ADD COLUMN IF NOT EXISTS job_id VARCHAR(100),
    ADD COLUMN IF NOT EXISTS template_id VARCHAR(100);

-- Extend status CHECK to include 'queued' and 'cancelled'
ALTER TABLE lgbm_tuning_run DROP CONSTRAINT IF EXISTS lgbm_tuning_run_status_check;
ALTER TABLE lgbm_tuning_run ADD CONSTRAINT lgbm_tuning_run_status_check
    CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'));

CREATE INDEX IF NOT EXISTS idx_tuning_run_job
    ON lgbm_tuning_run(job_id) WHERE job_id IS NOT NULL;
