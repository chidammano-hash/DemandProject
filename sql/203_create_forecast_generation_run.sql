-- Run-scoped forecast generation and promotion lineage.
--
-- A release candidate is promotable only when it has an explicit manifest.
-- Rows that predate this migration remain available for inspection, but are
-- classified as legacy_invalid and cannot be promoted. Snapshot-roster runs
-- are retained as snapshot_contender evidence and are also non-promotable.

BEGIN;

-- Exact historical champion-result lineage. A canonical champion rewrite that
-- is not tied to an explicitly promoted experiment remains NULL and therefore
-- cannot be used to generate a release candidate.
ALTER TABLE champion_experiment
    ADD COLUMN IF NOT EXISTS results_artifact_checksum TEXT;

ALTER TABLE champion_experiment
    ADD COLUMN IF NOT EXISTS results_forecast_checksum TEXT;

ALTER TABLE champion_experiment
    ADD COLUMN IF NOT EXISTS results_forecast_row_count INTEGER;

ALTER TABLE fact_external_forecast_monthly
    ADD COLUMN IF NOT EXISTS champion_experiment_id INTEGER;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_external_forecast_champion_experiment'
          AND conrelid = 'fact_external_forecast_monthly'::regclass
    ) THEN
        ALTER TABLE fact_external_forecast_monthly
            ADD CONSTRAINT fk_external_forecast_champion_experiment
            FOREIGN KEY (champion_experiment_id)
            REFERENCES champion_experiment (experiment_id)
            ON DELETE RESTRICT
            NOT VALID;
    END IF;
END $$;

ALTER TABLE fact_external_forecast_monthly
    VALIDATE CONSTRAINT fk_external_forecast_champion_experiment;

CREATE INDEX IF NOT EXISTS idx_external_forecast_champion_experiment
    ON fact_external_forecast_monthly (champion_experiment_id)
    WHERE champion_experiment_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 1. Immutable generation identity and mutable lifecycle manifest
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS forecast_generation_run (
    run_id                  UUID NOT NULL,
    generation_purpose      TEXT NOT NULL,
    run_status              TEXT NOT NULL,
    promotion_eligible      BOOLEAN NOT NULL DEFAULT FALSE,
    requested_model_id      VARCHAR(100) NOT NULL,
    forecast_month_generated DATE NOT NULL,
    horizon_months          SMALLINT,
    row_count               INTEGER NOT NULL DEFAULT 0,
    dfu_count               INTEGER NOT NULL DEFAULT 0,
    candidate_model_count   INTEGER NOT NULL DEFAULT 0,
    champion_experiment_id  INTEGER
                                REFERENCES champion_experiment (experiment_id)
                                ON DELETE RESTRICT,
    cluster_experiment_id   INTEGER
                                REFERENCES cluster_experiment (experiment_id)
                                ON DELETE RESTRICT,
    source_sales_batch_id   BIGINT
                                REFERENCES audit_load_batch (batch_id)
                                ON DELETE RESTRICT,
    routing_artifact_checksum TEXT,  -- SHA-256 of exact champion routing artifact bytes
    champion_results_checksum TEXT, -- SHA-256 of exact historical champion payload
    artifact_checksum       TEXT,    -- SHA-256 of canonical run-scoped staging payload
    created_by              TEXT NOT NULL DEFAULT 'forecast-generator',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMPTZ,
    invalid_reason          TEXT,
    metadata                JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT pk_forecast_generation_run PRIMARY KEY (run_id),
    CONSTRAINT uq_forecast_generation_run_id_purpose
        UNIQUE (run_id, generation_purpose, forecast_month_generated),
    CONSTRAINT chk_forecast_generation_run_purpose CHECK (
        generation_purpose IN (
            'release_candidate',
            'snapshot_contender',
            'legacy_invalid'
        )
    ),
    CONSTRAINT chk_forecast_generation_run_status CHECK (
        run_status IN ('generating', 'ready', 'invalid', 'promoted', 'archived')
    ),
    CONSTRAINT chk_forecast_generation_run_month_start CHECK (
        forecast_month_generated =
            date_trunc('month', forecast_month_generated)::date
    ),
    CONSTRAINT chk_forecast_generation_run_counts CHECK (
        row_count >= 0
        AND dfu_count >= 0
        AND candidate_model_count >= 0
        AND (horizon_months IS NULL OR horizon_months > 0)
    ),
    CONSTRAINT chk_forecast_generation_run_checksum CHECK (
        artifact_checksum IS NULL OR artifact_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_forecast_generation_run_routing_checksum CHECK (
        routing_artifact_checksum IS NULL
        OR routing_artifact_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_forecast_generation_run_results_checksum CHECK (
        champion_results_checksum IS NULL
        OR champion_results_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_forecast_generation_run_eligibility CHECK (
        NOT promotion_eligible
        OR (
            generation_purpose = 'release_candidate'
            AND run_status = 'ready'
        )
    ),
    CONSTRAINT chk_forecast_generation_run_ready_evidence CHECK (
        generation_purpose <> 'release_candidate'
        OR run_status <> 'ready'
        OR (
            row_count > 0
            AND dfu_count > 0
            AND candidate_model_count > 0
            AND horizon_months > 0
            AND artifact_checksum IS NOT NULL
            AND source_sales_batch_id IS NOT NULL
            AND completed_at IS NOT NULL
        )
    ),
    CONSTRAINT chk_forecast_generation_run_champion_lineage CHECK (
        generation_purpose <> 'release_candidate'
        OR run_status <> 'ready'
        OR requested_model_id <> 'champion'
        OR (
            champion_experiment_id IS NOT NULL
            AND cluster_experiment_id IS NOT NULL
            AND routing_artifact_checksum IS NOT NULL
            AND champion_results_checksum IS NOT NULL
        )
    ),
    CONSTRAINT chk_forecast_generation_run_legacy_invalid CHECK (
        generation_purpose <> 'legacy_invalid'
        OR (
            run_status = 'invalid'
            AND NOT promotion_eligible
            AND invalid_reason IS NOT NULL
        )
    ),
    CONSTRAINT chk_forecast_generation_run_promoted_purpose CHECK (
        run_status <> 'promoted'
        OR generation_purpose = 'release_candidate'
    ),
    CONSTRAINT chk_forecast_generation_run_archived_purpose CHECK (
        run_status <> 'archived'
        OR generation_purpose = 'snapshot_contender'
    )
);

CREATE INDEX IF NOT EXISTS idx_forecast_generation_run_purpose_status
    ON forecast_generation_run (generation_purpose, run_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_forecast_generation_run_month_generated
    ON forecast_generation_run (forecast_month_generated, created_at DESC);

-- Register frozen snapshot-contender runs first so they are never swept into
-- the generic legacy classification. These rows are intentionally not
-- promotion eligible: their purpose is bounded FVA evidence.
INSERT INTO forecast_generation_run (
    run_id,
    generation_purpose,
    run_status,
    promotion_eligible,
    requested_model_id,
    forecast_month_generated,
    candidate_model_count,
    created_by,
    created_at,
    completed_at,
    metadata
)
SELECT
    roster.generation_run_id,
    'snapshot_contender',
    'ready',
    FALSE,
    CASE
        WHEN COUNT(DISTINCT roster.model_id) = 1 THEN MIN(roster.model_id)
        ELSE 'legacy_mixed'
    END,
    MIN(roster.record_month),
    COUNT(DISTINCT roster.model_id)::integer,
    'migration-203',
    MIN(roster.selected_at),
    MAX(roster.selected_at),
    jsonb_build_object('classification', 'forecast_snapshot_roster')
FROM forecast_snapshot_roster AS roster
WHERE roster.generation_run_id IS NOT NULL
GROUP BY roster.generation_run_id
ON CONFLICT (run_id) DO NOTHING;

-- Every other pre-manifest staging run is legacy. It is retained so operators
-- can inspect or clean it, but the manifest constraint makes it ineligible for
-- release. No historical row is silently blessed as a release candidate.
INSERT INTO forecast_generation_run (
    run_id,
    generation_purpose,
    run_status,
    promotion_eligible,
    requested_model_id,
    forecast_month_generated,
    horizon_months,
    row_count,
    dfu_count,
    candidate_model_count,
    created_by,
    created_at,
    completed_at,
    invalid_reason,
    metadata
)
SELECT
    staging.run_id,
    'legacy_invalid',
    'invalid',
    FALSE,
    CASE
        WHEN COUNT(DISTINCT staging.model_id) = 1 THEN MIN(staging.model_id)
        ELSE 'legacy_mixed'
    END,
    MIN(staging.forecast_month_generated),
    MAX(staging.horizon_months),
    COUNT(*)::integer,
    COUNT(DISTINCT (staging.item_id, staging.loc))::integer,
    COUNT(DISTINCT staging.model_id)::integer,
    'migration-203',
    MIN(staging.generated_at),
    MAX(staging.generated_at),
    'legacy staging run predates the release-candidate manifest',
    jsonb_build_object('classification', 'pre_manifest_staging')
FROM fact_production_forecast_staging AS staging
LEFT JOIN forecast_snapshot_roster AS roster
  ON roster.generation_run_id = staging.run_id
WHERE roster.generation_run_id IS NULL
GROUP BY staging.run_id
ON CONFLICT (run_id) DO NOTHING;

-- Snapshot runs can still have their staging rows present. Capture their
-- actual row/model counts without changing their non-promotable purpose.
WITH staging_stats AS (
    SELECT
        run_id,
        COUNT(*)::integer AS row_count,
        COUNT(DISTINCT (item_id, loc))::integer AS dfu_count,
        COUNT(DISTINCT model_id)::integer AS candidate_model_count,
        MAX(horizon_months) AS horizon_months,
        MIN(generated_at) AS created_at,
        MAX(generated_at) AS completed_at
    FROM fact_production_forecast_staging
    GROUP BY run_id
)
UPDATE forecast_generation_run AS generation
SET row_count = stats.row_count,
    dfu_count = stats.dfu_count,
    candidate_model_count = stats.candidate_model_count,
    horizon_months = stats.horizon_months,
    created_at = LEAST(generation.created_at, stats.created_at),
    completed_at = GREATEST(generation.completed_at, stats.completed_at)
FROM staging_stats AS stats
WHERE generation.run_id = stats.run_id
  AND generation.generation_purpose = 'snapshot_contender';

-- ---------------------------------------------------------------------------
-- 2. Staging rows belong to exactly one run purpose and candidate model
-- ---------------------------------------------------------------------------

ALTER TABLE fact_production_forecast_staging
    ADD COLUMN IF NOT EXISTS generation_purpose TEXT;

ALTER TABLE fact_production_forecast_staging
    ADD COLUMN IF NOT EXISTS candidate_model_id VARCHAR(100);

UPDATE fact_production_forecast_staging AS staging
SET generation_purpose = generation.generation_purpose,
    candidate_model_id = COALESCE(staging.candidate_model_id, staging.model_id)
FROM forecast_generation_run AS generation
WHERE generation.run_id = staging.run_id
  AND (
      staging.generation_purpose IS DISTINCT FROM generation.generation_purpose
      OR staging.candidate_model_id IS NULL
  );

ALTER TABLE fact_production_forecast_staging
    ALTER COLUMN generation_purpose SET NOT NULL;

ALTER TABLE fact_production_forecast_staging
    ALTER COLUMN candidate_model_id SET NOT NULL;

DROP INDEX IF EXISTS uq_staging_model_dfu_month;

CREATE UNIQUE INDEX IF NOT EXISTS uq_staging_run_candidate_dfu_month
    ON fact_production_forecast_staging
    (run_id, generation_purpose, candidate_model_id, item_id, loc, forecast_month);

CREATE INDEX IF NOT EXISTS idx_staging_purpose_run_candidate
    ON fact_production_forecast_staging
    (generation_purpose, run_id, candidate_model_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_staging_generation_run_purpose'
          AND conrelid = 'fact_production_forecast_staging'::regclass
    ) THEN
        ALTER TABLE fact_production_forecast_staging
            ADD CONSTRAINT fk_staging_generation_run_purpose
            FOREIGN KEY (run_id, generation_purpose, forecast_month_generated)
            REFERENCES forecast_generation_run
                (run_id, generation_purpose, forecast_month_generated)
            ON DELETE RESTRICT
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_staging_forecast_months'
          AND conrelid = 'fact_production_forecast_staging'::regclass
    ) THEN
        ALTER TABLE fact_production_forecast_staging
            ADD CONSTRAINT chk_staging_forecast_months CHECK (
                forecast_month = date_trunc('month', forecast_month)::date
                AND forecast_month_generated =
                    date_trunc('month', forecast_month_generated)::date
            ) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_staging_forecast_quantities'
          AND conrelid = 'fact_production_forecast_staging'::regclass
    ) THEN
        ALTER TABLE fact_production_forecast_staging
            ADD CONSTRAINT chk_staging_forecast_quantities CHECK (
                forecast_qty >= 0
                AND (
                    forecast_qty_lower IS NULL
                    OR (
                        forecast_qty_lower >= 0
                        AND forecast_qty_lower <= forecast_qty
                    )
                )
                AND (
                    forecast_qty_upper IS NULL
                    OR (
                        forecast_qty_upper >= 0
                        AND forecast_qty <= forecast_qty_upper
                    )
                )
                AND (
                    forecast_qty_lower IS NULL
                    OR forecast_qty_upper IS NULL
                    OR forecast_qty_lower <= forecast_qty_upper
                )
                AND horizon_months > 0
            ) NOT VALID;
    END IF;
END $$;

ALTER TABLE fact_production_forecast_staging
    VALIDATE CONSTRAINT fk_staging_generation_run_purpose;

ALTER TABLE fact_production_forecast_staging
    VALIDATE CONSTRAINT chk_staging_forecast_months;

ALTER TABLE fact_production_forecast_staging
    VALIDATE CONSTRAINT chk_staging_forecast_quantities;

-- Derive (rather than accept) the roster purpose so existing snapshot writers
-- cannot accidentally relabel a release-candidate run as snapshot evidence.
ALTER TABLE forecast_snapshot_roster
    ADD COLUMN IF NOT EXISTS generation_purpose TEXT
    GENERATED ALWAYS AS (
        CASE
            WHEN generation_run_id IS NULL THEN NULL
            ELSE 'snapshot_contender'
        END
    ) STORED;

-- A frozen contender roster must point to a registered snapshot-contender run.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_snapshot_roster_generation_run'
          AND conrelid = 'forecast_snapshot_roster'::regclass
    ) THEN
        ALTER TABLE forecast_snapshot_roster
            ADD CONSTRAINT fk_snapshot_roster_generation_run
            FOREIGN KEY (generation_run_id, generation_purpose, record_month)
            REFERENCES forecast_generation_run
                (run_id, generation_purpose, forecast_month_generated)
            ON DELETE RESTRICT
            NOT VALID;
    END IF;
END $$;

ALTER TABLE forecast_snapshot_roster
    VALIDATE CONSTRAINT fk_snapshot_roster_generation_run;

-- Preserve the operational fields required for exact champion value-level
-- reconciliation. Existing snapshots remain valid with NULL legacy values;
-- every snapshot written after this migration carries them.
ALTER TABLE fact_forecast_snapshot
    ADD COLUMN IF NOT EXISTS is_recursive BOOLEAN;

ALTER TABLE fact_forecast_snapshot
    ADD COLUMN IF NOT EXISTS lag_source VARCHAR(20);

ALTER TABLE fact_forecast_snapshot
    ADD COLUMN IF NOT EXISTS source_promotion_id INTEGER;

-- ---------------------------------------------------------------------------
-- 3. Promotion audit stores exact source, release, gate, and checksum evidence
-- ---------------------------------------------------------------------------

ALTER TABLE model_promotion_log
    ADD COLUMN IF NOT EXISTS source_run_id UUID;

ALTER TABLE model_promotion_log
    ADD COLUMN IF NOT EXISTS production_run_id UUID;

ALTER TABLE model_promotion_log
    ADD COLUMN IF NOT EXISTS gate_report JSONB;

ALTER TABLE model_promotion_log
    ADD COLUMN IF NOT EXISTS candidate_checksum TEXT;

ALTER TABLE model_promotion_log
    ADD COLUMN IF NOT EXISTS production_checksum TEXT;

ALTER TABLE model_promotion_log
    ADD COLUMN IF NOT EXISTS archive_checksum TEXT;

ALTER TABLE model_promotion_log
    ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;

ALTER TABLE model_promotion_log
    ADD COLUMN IF NOT EXISTS replaces_promotion_id INTEGER;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_model_promotion_source_run'
          AND conrelid = 'model_promotion_log'::regclass
    ) THEN
        ALTER TABLE model_promotion_log
            ADD CONSTRAINT fk_model_promotion_source_run
            FOREIGN KEY (source_run_id)
            REFERENCES forecast_generation_run (run_id)
            ON DELETE RESTRICT
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_model_promotion_replaces'
          AND conrelid = 'model_promotion_log'::regclass
    ) THEN
        ALTER TABLE model_promotion_log
            ADD CONSTRAINT fk_model_promotion_replaces
            FOREIGN KEY (replaces_promotion_id)
            REFERENCES model_promotion_log (id)
            ON DELETE RESTRICT
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_model_promotion_evidence_bundle'
          AND conrelid = 'model_promotion_log'::regclass
    ) THEN
        ALTER TABLE model_promotion_log
            ADD CONSTRAINT chk_model_promotion_evidence_bundle CHECK (
                (
                    source_run_id IS NULL
                    AND gate_report IS NULL
                    AND candidate_checksum IS NULL
                    AND production_checksum IS NULL
                )
                OR (
                    source_run_id IS NOT NULL
                    AND production_run_id IS NOT NULL
                    AND gate_report IS NOT NULL
                    AND jsonb_typeof(gate_report) = 'object'
                    AND candidate_checksum IS NOT NULL
                    AND production_checksum IS NOT NULL
                    AND candidate_checksum = production_checksum
                )
            ) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_model_promotion_archive_evidence'
          AND conrelid = 'model_promotion_log'::regclass
    ) THEN
        ALTER TABLE model_promotion_log
            ADD CONSTRAINT chk_model_promotion_archive_evidence CHECK (
                (archive_checksum IS NULL AND archived_at IS NULL)
                OR (archive_checksum IS NOT NULL AND archived_at IS NOT NULL)
            ) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_model_promotion_replacement'
          AND conrelid = 'model_promotion_log'::regclass
    ) THEN
        ALTER TABLE model_promotion_log
            ADD CONSTRAINT chk_model_promotion_replacement CHECK (
                replaces_promotion_id IS NULL
                OR replaces_promotion_id <> id
            ) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_model_promotion_checksum_format'
          AND conrelid = 'model_promotion_log'::regclass
    ) THEN
        ALTER TABLE model_promotion_log
            ADD CONSTRAINT chk_model_promotion_checksum_format CHECK (
                (candidate_checksum IS NULL OR candidate_checksum ~ '^[0-9a-f]{64}$')
                AND (production_checksum IS NULL OR production_checksum ~ '^[0-9a-f]{64}$')
                AND (
                    archive_checksum IS NULL
                    OR archive_checksum ~ '^[0-9a-f]{64}$'
                )
            ) NOT VALID;
    END IF;
END $$;

ALTER TABLE model_promotion_log
    VALIDATE CONSTRAINT fk_model_promotion_source_run;

ALTER TABLE model_promotion_log
    VALIDATE CONSTRAINT fk_model_promotion_replaces;

ALTER TABLE model_promotion_log
    VALIDATE CONSTRAINT chk_model_promotion_evidence_bundle;

ALTER TABLE model_promotion_log
    VALIDATE CONSTRAINT chk_model_promotion_archive_evidence;

ALTER TABLE model_promotion_log
    VALIDATE CONSTRAINT chk_model_promotion_replacement;

ALTER TABLE model_promotion_log
    VALIDATE CONSTRAINT chk_model_promotion_checksum_format;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_forecast_snapshot_source_promotion'
          AND conrelid = 'fact_forecast_snapshot'::regclass
    ) THEN
        ALTER TABLE fact_forecast_snapshot
            ADD CONSTRAINT fk_forecast_snapshot_source_promotion
            FOREIGN KEY (source_promotion_id)
            REFERENCES model_promotion_log (id)
            ON DELETE RESTRICT
            NOT VALID;
    END IF;
END $$;

ALTER TABLE fact_forecast_snapshot
    VALIDATE CONSTRAINT fk_forecast_snapshot_source_promotion;

CREATE INDEX IF NOT EXISTS idx_forecast_snapshot_source_promotion
    ON fact_forecast_snapshot (source_promotion_id)
    WHERE source_promotion_id IS NOT NULL;

-- Repair historical duplicate-active rows deterministically before replacing
-- the old non-unique lookup index with a database-enforced singleton.
WITH ranked_active AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY promoted_at DESC, id DESC) AS active_rank,
        FIRST_VALUE(promoted_at) OVER (
            ORDER BY promoted_at DESC, id DESC
        ) AS replacement_promoted_at
    FROM model_promotion_log
    WHERE is_active
)
UPDATE model_promotion_log AS promotion
SET is_active = FALSE,
    demoted_at = GREATEST(
        promotion.promoted_at,
        COALESCE(promotion.demoted_at, ranked.replacement_promoted_at)
    )
FROM ranked_active AS ranked
WHERE promotion.id = ranked.id
  AND ranked.active_rank > 1;

-- The current legacy release can be tied to its promotion transaction only
-- when production contains one run and its audited row/DFU counts agree. A
-- mixed or mismatched release remains explicitly unverified and unlinked.
WITH production_run_summary AS (
    SELECT
        MIN(run_id::text)::uuid AS production_run_id,
        COUNT(*)::integer AS row_count,
        COUNT(DISTINCT (item_id, loc))::integer AS dfu_count
    FROM fact_production_forecast
    HAVING COUNT(DISTINCT run_id) = 1
)
UPDATE model_promotion_log AS promotion
SET production_run_id = summary.production_run_id
FROM production_run_summary AS summary
WHERE promotion.is_active
  AND promotion.source_run_id IS NULL
  AND promotion.production_run_id IS NULL
  AND promotion.total_rows = summary.row_count
  AND promotion.dfu_count = summary.dfu_count;

DROP INDEX IF EXISTS idx_promotion_log_active;

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_promotion_log_one_active
    ON model_promotion_log ((is_active))
    WHERE is_active;

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_promotion_log_source_run
    ON model_promotion_log (source_run_id)
    WHERE source_run_id IS NOT NULL;

-- This candidate key lets production rows prove that their audit id, source
-- run, and production run are the same triplet recorded by the promotion.
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_promotion_log_release_pair
    ON model_promotion_log (id, production_run_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_promotion_log_lineage_triplet
    ON model_promotion_log (id, source_run_id, production_run_id);

-- ---------------------------------------------------------------------------
-- 4. Every newly promoted production row carries verified audit lineage
-- ---------------------------------------------------------------------------

ALTER TABLE fact_production_forecast
    ADD COLUMN IF NOT EXISTS source_run_id UUID;

ALTER TABLE fact_production_forecast
    ADD COLUMN IF NOT EXISTS promotion_log_id INTEGER;

ALTER TABLE fact_production_forecast
    ADD COLUMN IF NOT EXISTS lineage_status TEXT;

UPDATE fact_production_forecast
SET lineage_status = 'legacy_unverified'
WHERE lineage_status IS NULL;

UPDATE fact_production_forecast AS forecast
SET promotion_log_id = promotion.id,
    lineage_status = 'legacy_unverified'
FROM model_promotion_log AS promotion
WHERE promotion.is_active
  AND promotion.source_run_id IS NULL
  AND promotion.production_run_id = forecast.run_id
  AND forecast.source_run_id IS NULL
  AND forecast.promotion_log_id IS NULL;

ALTER TABLE fact_production_forecast
    ALTER COLUMN lineage_status SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_production_forecast_lineage_status'
          AND conrelid = 'fact_production_forecast'::regclass
    ) THEN
        ALTER TABLE fact_production_forecast
            ADD CONSTRAINT chk_production_forecast_lineage_status CHECK (
                lineage_status IN ('legacy_unverified', 'verified')
                AND (
                    (
                        lineage_status = 'legacy_unverified'
                        AND source_run_id IS NULL
                    )
                    OR (
                        lineage_status = 'verified'
                        AND source_run_id IS NOT NULL
                        AND promotion_log_id IS NOT NULL
                    )
                )
            ) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_production_forecast_release_run'
          AND conrelid = 'fact_production_forecast'::regclass
    ) THEN
        ALTER TABLE fact_production_forecast
            ADD CONSTRAINT fk_production_forecast_release_run
            FOREIGN KEY (promotion_log_id, run_id)
            REFERENCES model_promotion_log (id, production_run_id)
            ON DELETE RESTRICT
            DEFERRABLE INITIALLY DEFERRED
            NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_production_forecast_promotion_lineage'
          AND conrelid = 'fact_production_forecast'::regclass
    ) THEN
        ALTER TABLE fact_production_forecast
            ADD CONSTRAINT fk_production_forecast_promotion_lineage
            FOREIGN KEY (promotion_log_id, source_run_id, run_id)
            REFERENCES model_promotion_log (id, source_run_id, production_run_id)
            ON DELETE RESTRICT
            DEFERRABLE INITIALLY DEFERRED
            NOT VALID;
    END IF;
END $$;

ALTER TABLE fact_production_forecast
    VALIDATE CONSTRAINT chk_production_forecast_lineage_status;

ALTER TABLE fact_production_forecast
    VALIDATE CONSTRAINT fk_production_forecast_release_run;

ALTER TABLE fact_production_forecast
    VALIDATE CONSTRAINT fk_production_forecast_promotion_lineage;

CREATE INDEX IF NOT EXISTS idx_prod_fcst_source_run_id
    ON fact_production_forecast (source_run_id)
    WHERE source_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_prod_fcst_promotion_log_id
    ON fact_production_forecast (promotion_log_id)
    WHERE promotion_log_id IS NOT NULL;

COMMIT;
