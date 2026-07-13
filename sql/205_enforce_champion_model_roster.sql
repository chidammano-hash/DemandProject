-- 205_enforce_champion_model_roster.sql
-- Keep all new champion experiments on the canonical five-model roster while
-- preserving historical experiment rows for audit and reproducibility.

ALTER TABLE champion_experiment
    ALTER COLUMN models SET DEFAULT
    '["lgbm_cluster","nhits","nbeats","mstl","chronos2_enriched"]'::jsonb;

ALTER TABLE champion_experiment
    DROP CONSTRAINT IF EXISTS ck_champion_experiment_canonical_models;

ALTER TABLE champion_experiment
    ADD CONSTRAINT ck_champion_experiment_canonical_models
    CHECK (
        jsonb_typeof(models) = 'array'
        AND jsonb_array_length(models) BETWEEN 1 AND 5
        AND models <@
            '["lgbm_cluster","nhits","nbeats","mstl","chronos2_enriched"]'::jsonb
        -- Array containment alone permits duplicates.  Match length to the
        -- number of distinct canonical members present so every model occurs
        -- at most once without rewriting historical rows.
        AND jsonb_array_length(models) = (
            (models ? 'lgbm_cluster')::integer
            + (models ? 'nhits')::integer
            + (models ? 'nbeats')::integer
            + (models ? 'mstl')::integer
            + (models ? 'chronos2_enriched')::integer
        )
    ) NOT VALID;

COMMENT ON CONSTRAINT ck_champion_experiment_canonical_models
    ON champion_experiment IS
    'New experiments may use only LightGBM, N-HiTS, N-BEATS, MSTL, and Chronos 2E; legacy rows remain auditable.';
