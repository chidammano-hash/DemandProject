-- 192_champion_sweep.sql
-- Champion Strategy Sweep (Tournament) — spec docs/specs/02-forecasting/30-champion-strategy-sweep.md
--
-- A sweep is a PARENT over the existing champion_experiment machinery: it fans out a
-- curated grid of candidate configs (each a real champion_experiment row), ranks them
-- globally AND within demand segments, assembles a per-segment composite, gates everything
-- against current production, and recommends a winner. These three tables add only the
-- orchestration + per-segment-slice metadata; children reuse champion_experiment entirely.
--
-- Idempotent: IF NOT EXISTS on all objects.

-- ---------------------------------------------------------------------------
-- 1. Parent sweep record
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS champion_sweep (
    sweep_id                   SERIAL PRIMARY KEY,
    label                      TEXT NOT NULL,
    notes                      TEXT,

    -- Input config
    mode                       TEXT NOT NULL DEFAULT 'both'
                                   CHECK (mode IN ('global','per_segment','both')),
    segment_axis               TEXT NOT NULL DEFAULT 'demand_class'
                                   CHECK (segment_axis IN ('demand_class','ml_cluster','abc_xyz')),
    objective                  TEXT NOT NULL DEFAULT 'robust'
                                   CHECK (objective IN ('accuracy','gap_to_ceiling','robust')),
    grid_spec                  JSONB NOT NULL,
    parallel                   BOOLEAN NOT NULL DEFAULT FALSE,
    baseline_experiment_id     INTEGER REFERENCES champion_experiment(experiment_id) ON DELETE SET NULL,

    -- State
    status                     TEXT NOT NULL DEFAULT 'queued'
                                   CHECK (status IN ('queued','running','completed','failed','cancelled')),
    candidate_count            INTEGER,
    completed_count            INTEGER NOT NULL DEFAULT 0,
    job_id                     VARCHAR(100),
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at                 TIMESTAMPTZ,
    completed_at               TIMESTAMPTZ,
    runtime_seconds            NUMERIC(10, 1),

    -- Results (populated on completion)
    best_global_experiment_id  INTEGER REFERENCES champion_experiment(experiment_id) ON DELETE SET NULL,
    composite_experiment_id    INTEGER REFERENCES champion_experiment(experiment_id) ON DELETE SET NULL,
    recommended_experiment_id  INTEGER REFERENCES champion_experiment(experiment_id) ON DELETE SET NULL,
    recommended_score          NUMERIC(10, 4),
    recommended_gate_eligible  BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_champion_sweep_status
    ON champion_sweep(status);
CREATE INDEX IF NOT EXISTS idx_champion_sweep_created
    ON champion_sweep(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_champion_sweep_recommended
    ON champion_sweep(recommended_experiment_id) WHERE recommended_experiment_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 2. Candidate members (one row per config in the sweep)
-- ---------------------------------------------------------------------------
-- Children are ordinary champion_experiment rows; this join table carries the
-- sweep-level rank/score and the duplicate-guard hash.
CREATE TABLE IF NOT EXISTS champion_sweep_member (
    id                   SERIAL PRIMARY KEY,
    sweep_id             INTEGER NOT NULL
                             REFERENCES champion_sweep(sweep_id) ON DELETE CASCADE,
    experiment_id        INTEGER NOT NULL
                             REFERENCES champion_experiment(experiment_id) ON DELETE CASCADE,
    config_hash          TEXT NOT NULL,        -- stable hash of (strategy, params, models, metric, lag)
    is_composite         BOOLEAN NOT NULL DEFAULT FALSE,
    global_rank          INTEGER,              -- 1 = best on the global objective; NULL until complete
    global_score         NUMERIC(10, 4),
    gate_eligible        BOOLEAN,
    skipped_duplicate    BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (sweep_id, experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_champion_sweep_member_rank
    ON champion_sweep_member(sweep_id, global_rank);

-- ---------------------------------------------------------------------------
-- 3. Per-segment scores (post-hoc slicing of each candidate by segment)
-- ---------------------------------------------------------------------------
-- Computed by restricting each candidate's per-DFU results to each segment's
-- DFUs — NOT by re-running. Drives the per-segment winner map + the composite.
CREATE TABLE IF NOT EXISTS champion_sweep_segment_score (
    id                   SERIAL PRIMARY KEY,
    sweep_id             INTEGER NOT NULL
                             REFERENCES champion_sweep(sweep_id) ON DELETE CASCADE,
    experiment_id        INTEGER NOT NULL
                             REFERENCES champion_experiment(experiment_id) ON DELETE CASCADE,
    segment              TEXT NOT NULL,        -- label on segment_axis (e.g. smooth/intermittent)
    n_dfus               INTEGER,              -- DFUs in this segment for this candidate
    accuracy             NUMERIC(8, 4),        -- segment-restricted champion accuracy %
    score                NUMERIC(10, 4),       -- segment-restricted objective score
    segment_rank         INTEGER,              -- 1 = best candidate WITHIN this segment
    UNIQUE (sweep_id, experiment_id, segment)
);

CREATE INDEX IF NOT EXISTS idx_champion_sweep_segment_rank
    ON champion_sweep_segment_score(sweep_id, segment, segment_rank);
