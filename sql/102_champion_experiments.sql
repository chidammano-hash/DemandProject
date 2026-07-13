-- 102_champion_experiments.sql
-- Champion Selection Experimentation Studio
-- Tracks strategy experiments, per-lag/month breakdowns, comparisons, and promotion audit.

-- ---------------------------------------------------------------------------
-- 1. Main experiment table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS champion_experiment (
    experiment_id         SERIAL PRIMARY KEY,
    label                 TEXT NOT NULL,
    notes                 TEXT,
    template_id           VARCHAR(100),
    status                TEXT NOT NULL DEFAULT 'queued'
                              CHECK (status IN ('queued','running','completed','failed','cancelled')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at            TIMESTAMPTZ,
    completed_at          TIMESTAMPTZ,
    runtime_seconds       NUMERIC(8, 1),
    job_id                VARCHAR(100),

    -- Input config
    strategy              TEXT NOT NULL,
    strategy_params       JSONB,
    meta_learner_params   JSONB,
    models                JSONB NOT NULL DEFAULT '["lgbm_cluster","nhits","nbeats","mstl","chronos2_enriched"]',
    metric                TEXT NOT NULL DEFAULT 'accuracy_pct',
    lag_mode              TEXT NOT NULL DEFAULT 'execution',
    min_sku_rows          INTEGER NOT NULL DEFAULT 3,

    -- Results (populated on completion)
    champion_accuracy     NUMERIC(8, 4),
    ceiling_accuracy      NUMERIC(8, 4),
    gap_bps               NUMERIC(8, 2),
    n_champions           INTEGER,
    n_dfu_months          INTEGER,
    model_distribution    JSONB,

    -- Promotion stage 1: config written to forecast_pipeline_config.yaml
    is_promoted           BOOLEAN NOT NULL DEFAULT FALSE,
    promoted_at           TIMESTAMPTZ,

    -- Promotion stage 2: champion rows loaded into fact tables
    is_results_promoted   BOOLEAN NOT NULL DEFAULT FALSE,
    results_promoted_at   TIMESTAMPTZ,
    results_promote_job_id VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_champion_exp_status
    ON champion_experiment(status);
CREATE INDEX IF NOT EXISTS idx_champion_exp_promoted
    ON champion_experiment(is_promoted) WHERE is_promoted;
CREATE INDEX IF NOT EXISTS idx_champion_exp_created
    ON champion_experiment(created_at DESC);

-- ---------------------------------------------------------------------------
-- 2. Per-execution-lag breakdown
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS champion_experiment_lag (
    id                   SERIAL PRIMARY KEY,
    experiment_id        INTEGER NOT NULL
                             REFERENCES champion_experiment(experiment_id) ON DELETE CASCADE,
    exec_lag             INTEGER NOT NULL,
    champion_accuracy    NUMERIC(8, 4),
    ceiling_accuracy     NUMERIC(8, 4),
    gap_bps              NUMERIC(8, 2),
    n_dfu_months         INTEGER,
    model_distribution   JSONB,
    UNIQUE (experiment_id, exec_lag)
);

-- ---------------------------------------------------------------------------
-- 3. Per-month breakdown
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS champion_experiment_month (
    id                   SERIAL PRIMARY KEY,
    experiment_id        INTEGER NOT NULL
                             REFERENCES champion_experiment(experiment_id) ON DELETE CASCADE,
    month_start          DATE NOT NULL,
    champion_accuracy    NUMERIC(8, 4),
    ceiling_accuracy     NUMERIC(8, 4),
    gap_bps              NUMERIC(8, 2),
    n_champions          INTEGER,
    model_distribution   JSONB,
    UNIQUE (experiment_id, month_start)
);

-- ---------------------------------------------------------------------------
-- 4. Cached pairwise comparison
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS champion_experiment_comparison (
    id                   SERIAL PRIMARY KEY,
    experiment_a_id      INTEGER NOT NULL
                             REFERENCES champion_experiment(experiment_id) ON DELETE CASCADE,
    experiment_b_id      INTEGER NOT NULL
                             REFERENCES champion_experiment(experiment_id) ON DELETE CASCADE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    overall_comparison   JSONB,
    per_lag_comparison   JSONB,
    per_month_comparison JSONB,
    model_dist_comparison JSONB,
    config_diffs         JSONB,
    UNIQUE (experiment_a_id, experiment_b_id)
);

-- ---------------------------------------------------------------------------
-- 5. Promotion audit log
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS champion_promotion_log (
    id                     SERIAL PRIMARY KEY,
    experiment_id          INTEGER NOT NULL
                               REFERENCES champion_experiment(experiment_id) ON DELETE RESTRICT,
    promoted_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_by            TEXT DEFAULT 'manual',
    previous_experiment_id INTEGER,
    strategy               TEXT,
    champion_accuracy      NUMERIC(8, 4),
    config_snapshot        JSONB
);
