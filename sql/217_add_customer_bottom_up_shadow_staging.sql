-- Add a review-only staging purpose for normalized customer bottom-up forecasts.
--
-- Shadow candidates use the canonical staging fact and immutable manifest, but
-- they can never become production releases. The separately governed
-- customer_bottom_up_blend remains the only customer-derived release candidate.

BEGIN;

ALTER TABLE forecast_generation_run
    DROP CONSTRAINT IF EXISTS chk_forecast_generation_run_purpose;

ALTER TABLE forecast_generation_run
    ADD CONSTRAINT chk_forecast_generation_run_purpose CHECK (
        generation_purpose IN (
            'release_candidate',
            'snapshot_contender',
            'shadow_candidate',
            'legacy_invalid'
        )
    );

-- Re-state the promotion invariant in the migration that introduces the new
-- purpose. This deliberately excludes shadow_candidate from promoted status.
ALTER TABLE forecast_generation_run
    DROP CONSTRAINT IF EXISTS chk_forecast_generation_run_promoted_purpose;

ALTER TABLE forecast_generation_run
    ADD CONSTRAINT chk_forecast_generation_run_promoted_purpose CHECK (
        run_status <> 'promoted'
        OR generation_purpose = 'release_candidate'
    );

-- A reviewable shadow is not ready until it carries the same minimum payload
-- evidence as a release candidate. It is deliberately never eligible.
ALTER TABLE forecast_generation_run
    DROP CONSTRAINT IF EXISTS chk_forecast_generation_run_shadow_ready_evidence;

ALTER TABLE forecast_generation_run
    ADD CONSTRAINT chk_forecast_generation_run_shadow_ready_evidence CHECK (
        generation_purpose <> 'shadow_candidate'
        OR run_status <> 'ready'
        OR (
            requested_model_id = 'customer_bottom_up'
            AND NOT promotion_eligible
            AND row_count > 0
            AND dfu_count > 0
            AND candidate_model_count > 0
            AND horizon_months > 0
            AND artifact_checksum IS NOT NULL
            AND source_sales_batch_id IS NOT NULL
            AND completed_at IS NOT NULL
            AND metadata ? 'customer_bottom_up_staging'
        )
    );

-- Shadow payload rows may be built or cleaned only while their manifest is
-- generating/invalid. Once ready, inserts, updates, and deletes fail closed.
CREATE OR REPLACE FUNCTION customer_bottom_up_shadow_staging_guard()
RETURNS TRIGGER AS $$
DECLARE
    parent_status TEXT;
BEGIN
    IF TG_OP = 'INSERT' THEN
        FOR parent_status IN
            SELECT run.run_status
            FROM forecast_generation_run AS run
            JOIN (
                SELECT DISTINCT run_id
                FROM inserted_rows
                WHERE generation_purpose = 'shadow_candidate'
            ) AS changed
              ON changed.run_id = run.run_id
            FOR UPDATE OF run
        LOOP
            IF parent_status NOT IN ('generating', 'invalid') THEN
                RAISE EXCEPTION 'ready customer bottom-up shadow staging is immutable';
            END IF;
        END LOOP;
    ELSIF TG_OP = 'UPDATE' THEN
        FOR parent_status IN
            SELECT run.run_status
            FROM forecast_generation_run AS run
            JOIN (
                SELECT run_id
                FROM old_rows
                WHERE generation_purpose = 'shadow_candidate'
                UNION
                SELECT run_id
                FROM new_rows
                WHERE generation_purpose = 'shadow_candidate'
            ) AS changed
              ON changed.run_id = run.run_id
            FOR UPDATE OF run
        LOOP
            IF parent_status NOT IN ('generating', 'invalid') THEN
                RAISE EXCEPTION 'ready customer bottom-up shadow staging is immutable';
            END IF;
        END LOOP;
    ELSE
        FOR parent_status IN
            SELECT run.run_status
            FROM forecast_generation_run AS run
            JOIN (
                SELECT DISTINCT run_id
                FROM deleted_rows
                WHERE generation_purpose = 'shadow_candidate'
            ) AS changed
              ON changed.run_id = run.run_id
            FOR UPDATE OF run
        LOOP
            IF parent_status NOT IN ('generating', 'invalid') THEN
                RAISE EXCEPTION 'ready customer bottom-up shadow staging is immutable';
            END IF;
        END LOOP;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_bottom_up_shadow_staging_insert_guard
    ON fact_production_forecast_staging;
CREATE TRIGGER trg_customer_bottom_up_shadow_staging_insert_guard
    AFTER INSERT ON fact_production_forecast_staging
    REFERENCING NEW TABLE AS inserted_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION customer_bottom_up_shadow_staging_guard();

DROP TRIGGER IF EXISTS trg_customer_bottom_up_shadow_staging_update_guard
    ON fact_production_forecast_staging;
CREATE TRIGGER trg_customer_bottom_up_shadow_staging_update_guard
    AFTER UPDATE ON fact_production_forecast_staging
    REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION customer_bottom_up_shadow_staging_guard();

DROP TRIGGER IF EXISTS trg_customer_bottom_up_shadow_staging_delete_guard
    ON fact_production_forecast_staging;
CREATE TRIGGER trg_customer_bottom_up_shadow_staging_delete_guard
    AFTER DELETE ON fact_production_forecast_staging
    REFERENCING OLD TABLE AS deleted_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION customer_bottom_up_shadow_staging_guard();

-- The ready manifest seals the checksum and cardinality that the UI trusts.
-- Prevent relabeling it to a mutable state before changing its payload.
CREATE OR REPLACE FUNCTION customer_bottom_up_shadow_manifest_guard()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.generation_purpose = 'shadow_candidate'
       AND OLD.run_status = 'ready' THEN
        RAISE EXCEPTION 'ready customer bottom-up shadow manifest is immutable';
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_bottom_up_shadow_manifest_guard
    ON forecast_generation_run;
CREATE TRIGGER trg_customer_bottom_up_shadow_manifest_guard
    BEFORE UPDATE OR DELETE ON forecast_generation_run
    FOR EACH ROW
    EXECUTE FUNCTION customer_bottom_up_shadow_manifest_guard();

COMMIT;
