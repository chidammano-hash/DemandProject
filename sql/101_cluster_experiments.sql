-- 101: Cluster Experimentation Studio
-- Tracks cluster experiment lifecycle (create → run → compare → promote),
-- caches pairwise comparison results, and extends lgbm_tuning_run with
-- cluster source reference for cluster-aware algorithm tuning.
-- Date: 2026-03-25

-- ── Cluster experiment lifecycle ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cluster_experiment (
    experiment_id       SERIAL PRIMARY KEY,
    scenario_id         VARCHAR(30) NOT NULL UNIQUE,   -- sc_YYYYMMDD_HHMMSS_xxxx
    label               TEXT NOT NULL,
    notes               TEXT,
    template_id         VARCHAR(100),
    status              TEXT NOT NULL DEFAULT 'queued'
                            CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    runtime_seconds     NUMERIC(8, 1),
    job_id              VARCHAR(100),

    -- Input config
    feature_params      JSONB,   -- {time_window_months, min_months_history}
    model_params        JSONB,   -- {k_range, min_cluster_size_pct, use_pca, ...}
    label_params        JSONB,   -- {volume_high, volume_low, cv_steady, ...}

    -- Results (populated on completion)
    optimal_k           INTEGER,
    silhouette_score    NUMERIC(8, 6),
    inertia             NUMERIC(14, 2),
    total_dfus          INTEGER,
    n_clusters          INTEGER,
    cluster_sizes       JSONB,
    profiles            JSONB,
    k_selection_results JSONB,

    -- Promotion
    is_promoted         BOOLEAN NOT NULL DEFAULT FALSE,
    promoted_at         TIMESTAMPTZ,
    artifacts_path      TEXT     -- /tmp/clustering_scenarios/{scenario_id}
);

CREATE INDEX IF NOT EXISTS idx_cluster_exp_status
    ON cluster_experiment(status);

CREATE INDEX IF NOT EXISTS idx_cluster_exp_promoted
    ON cluster_experiment(is_promoted) WHERE is_promoted;

CREATE INDEX IF NOT EXISTS idx_cluster_exp_created
    ON cluster_experiment(created_at DESC);

-- ── Pairwise comparison cache ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cluster_experiment_comparison (
    id                  SERIAL PRIMARY KEY,
    experiment_a_id     INTEGER NOT NULL
                            REFERENCES cluster_experiment(experiment_id) ON DELETE CASCADE,
    experiment_b_id     INTEGER NOT NULL
                            REFERENCES cluster_experiment(experiment_id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    migration_matrix    JSONB,     -- {from_label: {to_label: count}}
    quality_comparison  JSONB,
    profile_comparison  JSONB,
    UNIQUE (experiment_a_id, experiment_b_id)
);

-- ── Alter lgbm_tuning_run: add cluster source reference ─────────────────────
ALTER TABLE lgbm_tuning_run
    ADD COLUMN IF NOT EXISTS cluster_source VARCHAR(20) NOT NULL DEFAULT 'production'
        CHECK (cluster_source IN ('production', 'experimental')),
    ADD COLUMN IF NOT EXISTS cluster_experiment_id INTEGER
        REFERENCES cluster_experiment(experiment_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_tuning_run_cluster_exp
    ON lgbm_tuning_run(cluster_experiment_id)
    WHERE cluster_experiment_id IS NOT NULL;
