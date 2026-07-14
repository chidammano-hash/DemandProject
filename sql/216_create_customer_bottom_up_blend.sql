-- Governed customer bottom-up blend evidence (Spec 35).
--
-- A blend run is anchored to one immutable customer forecast run and one
-- active item-location production release.  The rows are component evidence
-- for a reviewable staging draft; they do not themselves promote or replace
-- the active production forecast.

BEGIN;

-- Bind every new customer forecast to the exact completed customer-demand
-- load it read. The column remains nullable so pre-lineage historical runs can
-- remain queryable, but all current generation entrypoints fail closed.
ALTER TABLE customer_forecast_run
    ADD COLUMN IF NOT EXISTS source_customer_demand_batch_id BIGINT;

ALTER TABLE customer_forecast_run
    DROP CONSTRAINT IF EXISTS fk_customer_forecast_source_demand_batch,
    ADD CONSTRAINT fk_customer_forecast_source_demand_batch
        FOREIGN KEY (source_customer_demand_batch_id)
        REFERENCES audit_load_batch (batch_id)
        NOT VALID;

CREATE INDEX IF NOT EXISTS idx_customer_forecast_source_demand_batch
    ON customer_forecast_run (source_customer_demand_batch_id)
    WHERE source_customer_demand_batch_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_audit_load_batch_customer_demand_completed
    ON audit_load_batch (completed_at DESC, batch_id DESC)
    WHERE domain = 'customer_demand' AND status = 'completed';

CREATE INDEX IF NOT EXISTS idx_job_history_customer_forecast_run
    ON job_history (job_type, ((params ->> 'run_id')), submitted_at DESC)
    WHERE job_type IN (
        'generate_customer_forecast',
        'generate_customer_forecast_backtest',
        'generate_customer_forecast_blend'
    );

-- One explicit proof that the customer-demand profile represents an exact
-- completed source load. This table is intentionally not backfilled: existing
-- deployments fail closed until a new customer-demand load refreshes every
-- dependent materialized view and publishes its batch atomically.
CREATE TABLE IF NOT EXISTS customer_demand_profile_refresh_state (
    singleton_id       SMALLINT PRIMARY KEY DEFAULT 1,
    source_batch_id    BIGINT NOT NULL,
    refreshed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_customer_demand_profile_refresh_singleton
        CHECK (singleton_id = 1),
    CONSTRAINT fk_customer_demand_profile_refresh_batch
        FOREIGN KEY (source_batch_id)
        REFERENCES audit_load_batch (batch_id)
        ON DELETE RESTRICT
);

COMMENT ON TABLE customer_demand_profile_refresh_state IS
    'Singleton proof of the completed customer-demand batch represented by refreshed profile MVs';

-- Existing Chronos rows remain readable. NOT VALID constraints still enforce
-- Croston-only policy for every row inserted or updated after this migration.
ALTER TABLE customer_forecast_run
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_run_croston_only,
    ADD CONSTRAINT chk_customer_forecast_run_croston_only
        CHECK (model_id = 'croston') NOT VALID;

ALTER TABLE customer_forecast_batch
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_batch_route,
    ADD CONSTRAINT chk_customer_forecast_batch_route
        CHECK (route_model_id = 'croston') NOT VALID;

ALTER TABLE fact_customer_forecast
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_fact_croston_only,
    ADD CONSTRAINT chk_customer_forecast_fact_croston_only
        CHECK (model_id = 'croston') NOT VALID;

CREATE TABLE IF NOT EXISTS customer_forecast_backtest_run (
    run_id                     UUID PRIMARY KEY,
    job_id                     TEXT UNIQUE,
    customer_run_id            UUID NOT NULL
                                   REFERENCES customer_forecast_run (run_id)
                                   ON DELETE RESTRICT,
    run_status                 TEXT NOT NULL,
    planning_month             DATE NOT NULL,
    evaluation_start           DATE NOT NULL,
    evaluation_end             DATE NOT NULL,
    lookback_months            SMALLINT NOT NULL,
    min_train_months           SMALLINT NOT NULL,
    horizon_months             SMALLINT NOT NULL,
    batch_size                 INTEGER NOT NULL,
    source_series_count        INTEGER NOT NULL,
    source_series_checksum     TEXT NOT NULL,
    customer_model_id          TEXT NOT NULL,
    blend_model_id             TEXT NOT NULL,
    source_promotion_id        INTEGER NOT NULL,
    source_production_run_id   UUID NOT NULL,
    config_checksum            TEXT,
    component_checksum         TEXT,
    total_batches              INTEGER NOT NULL DEFAULT 0,
    completed_batches          INTEGER NOT NULL DEFAULT 0,
    component_rows             INTEGER NOT NULL DEFAULT 0,
    metadata                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_summary              TEXT,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at                 TIMESTAMPTZ,
    completed_at               TIMESTAMPTZ,
    CONSTRAINT fk_customer_backtest_source_release
        FOREIGN KEY (source_promotion_id, source_production_run_id)
        REFERENCES model_promotion_log (id, production_run_id)
        ON DELETE RESTRICT,
    CONSTRAINT uq_customer_backtest_lineage
        UNIQUE (
            run_id,
            customer_run_id,
            source_promotion_id,
            source_production_run_id
        ),
    CONSTRAINT chk_customer_backtest_status CHECK (
        run_status IN ('queued', 'generating', 'completed', 'failed', 'cancelled')
    ),
    CONSTRAINT chk_customer_backtest_months CHECK (
        planning_month = date_trunc('month', planning_month)::date
        AND evaluation_start = date_trunc('month', evaluation_start)::date
        AND evaluation_end = date_trunc('month', evaluation_end)::date
        AND evaluation_start <= evaluation_end
        AND lookback_months > 0
        AND min_train_months > 0
        AND horizon_months > 0
    ),
    CONSTRAINT chk_customer_backtest_contract CHECK (
        customer_model_id = 'croston'
        AND blend_model_id = 'customer_bottom_up_blend'
        AND batch_size > 0
    ),
    CONSTRAINT chk_customer_backtest_counts CHECK (
        total_batches >= 0
        AND completed_batches >= 0
        AND completed_batches <= total_batches
        AND source_series_count >= 0
        AND component_rows >= 0
    ),
    CONSTRAINT chk_customer_backtest_config_checksum CHECK (
        config_checksum IS NULL OR config_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_customer_backtest_source_series_checksum CHECK (
        source_series_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_customer_backtest_component_checksum CHECK (
        component_checksum IS NULL OR component_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_customer_backtest_completed CHECK (
        run_status <> 'completed'
        OR (
            completed_at IS NOT NULL
            AND config_checksum IS NOT NULL
            AND component_checksum IS NOT NULL
            AND total_batches > 0
            AND completed_batches = total_batches
            AND component_rows > 0
        )
    ),
    CONSTRAINT chk_customer_backtest_metadata CHECK (
        jsonb_typeof(metadata) = 'object'
    )
);

CREATE INDEX IF NOT EXISTS idx_customer_backtest_status_created
    ON customer_forecast_backtest_run (run_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_customer_backtest_source_release
    ON customer_forecast_backtest_run
       (source_promotion_id, source_production_run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_customer_backtest_matching_evidence
    ON customer_forecast_backtest_run
       (customer_run_id, source_promotion_id, source_production_run_id,
        completed_at DESC)
    WHERE run_status = 'completed';

CREATE UNIQUE INDEX IF NOT EXISTS uq_customer_backtest_one_active
    ON customer_forecast_backtest_run ((1))
    WHERE run_status IN ('queued', 'generating');

CREATE TABLE IF NOT EXISTS customer_bottom_up_backtest_component (
    backtest_run_id              UUID NOT NULL
                                      REFERENCES customer_forecast_backtest_run (run_id)
                                      ON DELETE RESTRICT,
    item_id                      TEXT NOT NULL,
    loc                          TEXT NOT NULL,
    forecast_origin              DATE NOT NULL,
    forecast_month               DATE NOT NULL,
    raw_customer_demand_qty      NUMERIC(18,4),
    normalized_customer_qty      NUMERIC(18,4),
    champion_qty                 NUMERIC(18,4) NOT NULL,
    blended_qty                  NUMERIC(18,4) NOT NULL,
    actual_qty                   NUMERIC(18,4) NOT NULL,
    fulfillment_ratio            NUMERIC(12,8),
    customer_weight              NUMERIC(8,6) NOT NULL,
    champion_weight              NUMERIC(8,6) NOT NULL,
    effective_customer_weight    NUMERIC(8,6) NOT NULL,
    customer_series_count        INTEGER NOT NULL,
    coverage_status              TEXT NOT NULL,
    generated_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_customer_bottom_up_backtest_component
        PRIMARY KEY (
            backtest_run_id,
            item_id,
            loc,
            forecast_origin,
            forecast_month
        ),
    CONSTRAINT chk_customer_bottom_up_backtest_months CHECK (
        forecast_origin = date_trunc('month', forecast_origin)::date
        AND forecast_month = date_trunc('month', forecast_month)::date
        AND forecast_month = (forecast_origin + INTERVAL '1 month')::date
    ),
    CONSTRAINT chk_customer_bottom_up_backtest_quantities CHECK (
        champion_qty >= 0
        AND blended_qty >= 0
        AND actual_qty >= 0
        AND (raw_customer_demand_qty IS NULL OR raw_customer_demand_qty >= 0)
        AND (normalized_customer_qty IS NULL OR normalized_customer_qty >= 0)
    ),
    CONSTRAINT chk_customer_bottom_up_backtest_ratio CHECK (
        fulfillment_ratio IS NULL
        OR (fulfillment_ratio >= 0 AND fulfillment_ratio <= 1)
    ),
    CONSTRAINT chk_customer_bottom_up_backtest_weights CHECK (
        customer_weight >= 0
        AND customer_weight <= 1
        AND champion_weight >= 0
        AND champion_weight <= 1
        AND ABS(customer_weight + champion_weight - 1) <= 0.000001
        AND effective_customer_weight >= 0
        AND effective_customer_weight <= customer_weight
    ),
    CONSTRAINT chk_customer_bottom_up_backtest_normalization CHECK (
        normalized_customer_qty IS NULL
        OR (
            raw_customer_demand_qty IS NOT NULL
            AND fulfillment_ratio IS NOT NULL
            AND ABS(
                normalized_customer_qty
                - raw_customer_demand_qty * fulfillment_ratio
            ) <= 0.0001 + raw_customer_demand_qty * 0.00000001
        )
    ),
    CONSTRAINT chk_customer_bottom_up_backtest_formula CHECK (
        (
            effective_customer_weight = 0
            AND blended_qty = champion_qty
        )
        OR (
            effective_customer_weight > 0
            AND normalized_customer_qty IS NOT NULL
            AND ABS(
                blended_qty
                - (
                    effective_customer_weight * normalized_customer_qty
                    + (1 - effective_customer_weight) * champion_qty
                )
            ) <= 0.0001
        )
    ),
    CONSTRAINT chk_customer_bottom_up_backtest_coverage CHECK (
        coverage_status IN ('blended', 'champion_fallback')
        AND customer_series_count >= 0
        AND (
            (coverage_status = 'blended'
             AND effective_customer_weight = customer_weight
             AND effective_customer_weight > 0
             AND customer_series_count > 0)
            OR
            (coverage_status = 'champion_fallback'
             AND effective_customer_weight = 0)
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_customer_bottom_up_backtest_lookup
    ON customer_bottom_up_backtest_component
       (backtest_run_id, item_id, loc, forecast_month);

CREATE INDEX IF NOT EXISTS idx_customer_bottom_up_backtest_coverage
    ON customer_bottom_up_backtest_component
       (backtest_run_id, coverage_status, forecast_month);

CREATE TABLE IF NOT EXISTS customer_bottom_up_backtest_accuracy (
    backtest_run_id                    UUID PRIMARY KEY
                                            REFERENCES customer_forecast_backtest_run (run_id)
                                            ON DELETE RESTRICT,
    evaluation_start                   DATE NOT NULL,
    evaluation_end                     DATE NOT NULL,
    common_months                      INTEGER NOT NULL,
    common_dfus                        INTEGER NOT NULL,
    common_rows                        INTEGER NOT NULL,
    actual_qty                         NUMERIC(24,4) NOT NULL,
    customer_absolute_error            NUMERIC(24,4) NOT NULL,
    customer_wape_pct                  NUMERIC(14,6),
    customer_mae                       NUMERIC(18,6) NOT NULL,
    customer_bias_pct                  NUMERIC(14,6),
    customer_accuracy_pct              NUMERIC(14,6),
    champion_absolute_error            NUMERIC(24,4) NOT NULL,
    champion_wape_pct                  NUMERIC(14,6),
    champion_mae                       NUMERIC(18,6) NOT NULL,
    champion_bias_pct                  NUMERIC(14,6),
    champion_accuracy_pct              NUMERIC(14,6),
    blend_absolute_error               NUMERIC(24,4) NOT NULL,
    blend_wape_pct                     NUMERIC(14,6),
    blend_mae                          NUMERIC(18,6) NOT NULL,
    blend_bias_pct                     NUMERIC(14,6),
    blend_accuracy_pct                 NUMERIC(14,6),
    blend_wape_degradation_pct         NUMERIC(14,6),
    min_common_months                  INTEGER NOT NULL,
    min_common_dfus                    INTEGER NOT NULL,
    max_wape_degradation_pct           NUMERIC(14,6) NOT NULL,
    gate_passed                        BOOLEAN NOT NULL,
    gate_reason                        TEXT,
    generated_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_customer_bottom_up_accuracy_window CHECK (
        evaluation_start = date_trunc('month', evaluation_start)::date
        AND evaluation_end = date_trunc('month', evaluation_end)::date
        AND evaluation_start <= evaluation_end
    ),
    CONSTRAINT chk_customer_bottom_up_accuracy_counts CHECK (
        common_months >= 0
        AND common_dfus >= 0
        AND common_rows >= 0
        AND actual_qty >= 0
        AND customer_absolute_error >= 0
        AND champion_absolute_error >= 0
        AND blend_absolute_error >= 0
        AND customer_mae >= 0
        AND champion_mae >= 0
        AND blend_mae >= 0
        AND min_common_months > 0
        AND min_common_dfus > 0
        AND max_wape_degradation_pct = 0
    ),
    CONSTRAINT chk_customer_bottom_up_accuracy_metrics CHECK (
        (
            (customer_wape_pct IS NULL
             AND customer_bias_pct IS NULL
             AND customer_accuracy_pct IS NULL)
            OR
            (customer_wape_pct >= 0
             AND customer_bias_pct IS NOT NULL
             AND customer_accuracy_pct BETWEEN 0 AND 100)
        )
        AND (
            (champion_wape_pct IS NULL
             AND champion_bias_pct IS NULL
             AND champion_accuracy_pct IS NULL)
            OR
            (champion_wape_pct >= 0
             AND champion_bias_pct IS NOT NULL
             AND champion_accuracy_pct BETWEEN 0 AND 100)
        )
        AND (
            (blend_wape_pct IS NULL
             AND blend_bias_pct IS NULL
             AND blend_accuracy_pct IS NULL)
            OR
            (blend_wape_pct >= 0
             AND blend_bias_pct IS NOT NULL
             AND blend_accuracy_pct BETWEEN 0 AND 100)
        )
    ),
    CONSTRAINT chk_customer_bottom_up_customer_accuracy CHECK (
        customer_wape_pct IS NULL
        OR ABS(
            customer_accuracy_pct - GREATEST(0, 100 - customer_wape_pct)
        ) <= 0.000002
    ),
    CONSTRAINT chk_customer_bottom_up_champion_accuracy CHECK (
        champion_wape_pct IS NULL
        OR ABS(
            champion_accuracy_pct - GREATEST(0, 100 - champion_wape_pct)
        ) <= 0.000002
    ),
    CONSTRAINT chk_customer_bottom_up_blended_accuracy CHECK (
        blend_wape_pct IS NULL
        OR ABS(
            blend_accuracy_pct - GREATEST(0, 100 - blend_wape_pct)
        ) <= 0.000002
    ),
    CONSTRAINT chk_customer_bottom_up_wape_degradation CHECK (
        blend_wape_degradation_pct IS NULL
        OR (
            blend_wape_pct IS NOT NULL
            AND champion_wape_pct IS NOT NULL
            AND ABS(
                blend_wape_degradation_pct
                - (blend_wape_pct - champion_wape_pct)
            ) <= 0.000002
        )
    ),
    CONSTRAINT chk_customer_bottom_up_gate_reason CHECK (
        gate_reason IS NULL OR BTRIM(gate_reason) <> ''
    ),
    CONSTRAINT chk_customer_bottom_up_gate_pass CHECK (
        (
            gate_passed
            AND common_months >= min_common_months
            AND common_dfus >= min_common_dfus
            AND common_rows > 0
            AND customer_wape_pct IS NOT NULL
            AND champion_wape_pct IS NOT NULL
            AND blend_wape_pct IS NOT NULL
            AND blend_wape_degradation_pct IS NOT NULL
            AND blend_wape_degradation_pct
                <= max_wape_degradation_pct
        )
        OR (NOT gate_passed AND gate_reason IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_customer_bottom_up_blend_active
    ON forecast_generation_run ((1))
    WHERE run_status = 'generating'
      AND metadata ? 'customer_bottom_up_blend';

CREATE TABLE IF NOT EXISTS customer_bottom_up_blend_component (
    run_id                     UUID NOT NULL
                                   REFERENCES forecast_generation_run (run_id)
                                   ON DELETE RESTRICT,
    backtest_run_id            UUID NOT NULL
                                   REFERENCES customer_forecast_backtest_run (run_id)
                                   ON DELETE RESTRICT,
    customer_run_id            UUID NOT NULL
                                   REFERENCES customer_forecast_run (run_id)
                                   ON DELETE RESTRICT,
    source_promotion_id        INTEGER NOT NULL,
    source_production_run_id   UUID NOT NULL,
    item_id                    TEXT NOT NULL,
    loc                        TEXT NOT NULL,
    forecast_month             DATE NOT NULL,
    raw_customer_demand_qty    NUMERIC(18,4),
    normalized_customer_qty    NUMERIC(18,4),
    champion_qty               NUMERIC(18,4) NOT NULL,
    blended_qty                NUMERIC(18,4) NOT NULL,
    blended_lower              NUMERIC(18,4),
    blended_upper              NUMERIC(18,4),
    fulfillment_ratio          NUMERIC(12,8),
    customer_weight            NUMERIC(8,6) NOT NULL,
    champion_weight            NUMERIC(8,6) NOT NULL,
    effective_customer_weight  NUMERIC(8,6) NOT NULL,
    customer_series_count      INTEGER NOT NULL,
    coverage_status            TEXT NOT NULL,
    interval_method            TEXT NOT NULL,
    generated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_customer_bottom_up_blend_component
        PRIMARY KEY (run_id, item_id, loc, forecast_month),
    CONSTRAINT fk_customer_bottom_up_blend_source_release
        FOREIGN KEY (source_promotion_id, source_production_run_id)
        REFERENCES model_promotion_log (id, production_run_id)
        ON DELETE RESTRICT,
    CONSTRAINT fk_customer_bottom_up_blend_backtest_lineage
        FOREIGN KEY (
            backtest_run_id,
            customer_run_id,
            source_promotion_id,
            source_production_run_id
        )
        REFERENCES customer_forecast_backtest_run (
            run_id,
            customer_run_id,
            source_promotion_id,
            source_production_run_id
        )
        ON DELETE RESTRICT,
    CONSTRAINT chk_customer_bottom_up_blend_month_start CHECK (
        forecast_month = date_trunc('month', forecast_month)::date
    ),
    CONSTRAINT chk_customer_bottom_up_blend_quantities CHECK (
        champion_qty >= 0
        AND blended_qty >= 0
        AND (raw_customer_demand_qty IS NULL OR raw_customer_demand_qty >= 0)
        AND (normalized_customer_qty IS NULL OR normalized_customer_qty >= 0)
    ),
    CONSTRAINT chk_customer_bottom_up_blend_ratio CHECK (
        fulfillment_ratio IS NULL
        OR (fulfillment_ratio >= 0 AND fulfillment_ratio <= 1)
    ),
    CONSTRAINT chk_customer_bottom_up_blend_weights CHECK (
        customer_weight >= 0
        AND customer_weight <= 1
        AND champion_weight >= 0
        AND champion_weight <= 1
        AND ABS(customer_weight + champion_weight - 1) <= 0.000001
        AND effective_customer_weight >= 0
        AND effective_customer_weight <= customer_weight
    ),
    CONSTRAINT chk_customer_bottom_up_blend_normalization CHECK (
        normalized_customer_qty IS NULL
        OR (
            raw_customer_demand_qty IS NOT NULL
            AND fulfillment_ratio IS NOT NULL
            AND ABS(
                normalized_customer_qty
                - raw_customer_demand_qty * fulfillment_ratio
            ) <= 0.0001 + raw_customer_demand_qty * 0.00000001
        )
    ),
    CONSTRAINT chk_customer_bottom_up_blend_formula CHECK (
        (
            effective_customer_weight = 0
            AND blended_qty = champion_qty
        )
        OR (
            effective_customer_weight > 0
            AND normalized_customer_qty IS NOT NULL
            AND ABS(
                blended_qty
                - (
                    effective_customer_weight * normalized_customer_qty
                    + (1 - effective_customer_weight) * champion_qty
                )
            ) <= 0.0001
        )
    ),
    CONSTRAINT chk_customer_bottom_up_blend_interval CHECK (
        (
            interval_method = 'none'
            AND blended_lower IS NULL
            AND blended_upper IS NULL
        )
        OR (
            interval_method IN ('champion_width_shift', 'champion_passthrough')
            AND blended_lower IS NOT NULL
            AND blended_upper IS NOT NULL
            AND blended_lower >= 0
            AND blended_lower <= blended_qty
            AND blended_qty <= blended_upper
        )
    ),
    CONSTRAINT chk_customer_bottom_up_blend_lineage_labels CHECK (
        customer_series_count >= 0
        AND coverage_status IN ('blended', 'champion_fallback')
        AND interval_method IN (
            'champion_width_shift',
            'champion_passthrough',
            'none'
        )
    ),
    CONSTRAINT chk_customer_bottom_up_blend_customer_evidence CHECK (
        (
            coverage_status = 'blended'
            AND effective_customer_weight = customer_weight
            AND effective_customer_weight > 0
            AND customer_series_count > 0
            AND interval_method IN ('champion_width_shift', 'none')
        )
        OR (
            coverage_status = 'champion_fallback'
            AND effective_customer_weight = 0
            AND interval_method IN ('champion_passthrough', 'none')
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_customer_bottom_up_blend_customer_run
    ON customer_bottom_up_blend_component
       (customer_run_id, item_id, loc, forecast_month);

CREATE INDEX IF NOT EXISTS idx_customer_bottom_up_blend_backtest_run
    ON customer_bottom_up_blend_component
       (backtest_run_id, item_id, loc, forecast_month);

CREATE INDEX IF NOT EXISTS idx_customer_bottom_up_blend_source_release
    ON customer_bottom_up_blend_component
       (source_promotion_id, source_production_run_id, forecast_month);

CREATE INDEX IF NOT EXISTS idx_customer_bottom_up_blend_coverage
    ON customer_bottom_up_blend_component
       (run_id, coverage_status, forecast_month);

-- Customer forecast rows may only change while their parent is generating.
-- Failed/cancelled/completed payloads remain frozen evidence until resumed.
CREATE OR REPLACE FUNCTION fact_customer_forecast_guard_completed_run()
RETURNS TRIGGER AS $$
DECLARE
    parent_status TEXT;
BEGIN
    IF TG_OP = 'INSERT' THEN
        FOR parent_status IN
            SELECT run.run_status
            FROM customer_forecast_run AS run
            JOIN (SELECT DISTINCT run_id FROM inserted_rows) AS changed
              ON changed.run_id = run.run_id
            FOR UPDATE OF run
        LOOP
            IF parent_status <> 'generating' THEN
                RAISE EXCEPTION 'fact_customer_forecast requires a generating parent run';
            END IF;
        END LOOP;
    ELSIF TG_OP = 'UPDATE' THEN
        FOR parent_status IN
            SELECT run.run_status
            FROM customer_forecast_run AS run
            JOIN (
                SELECT run_id FROM old_rows
                UNION
                SELECT run_id FROM new_rows
            ) AS changed
              ON changed.run_id = run.run_id
            FOR UPDATE OF run
        LOOP
            IF parent_status <> 'generating' THEN
                RAISE EXCEPTION 'fact_customer_forecast requires a generating parent run';
            END IF;
        END LOOP;
    ELSE
        FOR parent_status IN
            SELECT run.run_status
            FROM customer_forecast_run AS run
            JOIN (SELECT DISTINCT run_id FROM deleted_rows) AS changed
              ON changed.run_id = run.run_id
            FOR UPDATE OF run
        LOOP
            IF parent_status <> 'generating' THEN
                RAISE EXCEPTION 'fact_customer_forecast requires a generating parent run';
            END IF;
        END LOOP;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_fact_customer_forecast_completed_insert
    ON fact_customer_forecast;
CREATE TRIGGER trg_fact_customer_forecast_completed_insert
    AFTER INSERT ON fact_customer_forecast
    REFERENCING NEW TABLE AS inserted_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION fact_customer_forecast_guard_completed_run();

DROP TRIGGER IF EXISTS trg_fact_customer_forecast_completed_update
    ON fact_customer_forecast;
CREATE TRIGGER trg_fact_customer_forecast_completed_update
    AFTER UPDATE ON fact_customer_forecast
    REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION fact_customer_forecast_guard_completed_run();

DROP TRIGGER IF EXISTS trg_fact_customer_forecast_completed_delete
    ON fact_customer_forecast;
CREATE TRIGGER trg_fact_customer_forecast_completed_delete
    AFTER DELETE ON fact_customer_forecast
    REFERENCING OLD TABLE AS deleted_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION fact_customer_forecast_guard_completed_run();

-- ON DELETE CASCADE runs after the parent row is no longer visible to a child
-- trigger. Block parent deletion when any frozen child payload exists, and
-- keep completed manifests terminal so the child guard cannot be bypassed.
CREATE OR REPLACE FUNCTION customer_forecast_run_guard_completed_terminal()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE'
       AND OLD.run_status <> 'generating'
       AND EXISTS (
           SELECT 1
           FROM fact_customer_forecast AS forecast
           WHERE forecast.run_id = OLD.run_id
       ) THEN
        RAISE EXCEPTION 'customer_forecast_run with frozen forecast rows is immutable';
    END IF;

    IF OLD.run_status = 'completed'
       AND (
           TG_OP = 'DELETE'
           OR TO_JSONB(NEW) - 'job_id' IS DISTINCT FROM TO_JSONB(OLD) - 'job_id'
       ) THEN
        RAISE EXCEPTION 'completed customer_forecast_run is immutable';
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_forecast_run_completed_terminal
    ON customer_forecast_run;
CREATE TRIGGER trg_customer_forecast_run_completed_terminal
    BEFORE UPDATE OR DELETE ON customer_forecast_run
    FOR EACH ROW
    EXECUTE FUNCTION customer_forecast_run_guard_completed_terminal();

CREATE OR REPLACE FUNCTION customer_forecast_backtest_run_guard_completed_terminal()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.run_status = 'completed'
       AND (
           TG_OP = 'DELETE'
           OR TO_JSONB(NEW) - 'job_id' IS DISTINCT FROM TO_JSONB(OLD) - 'job_id'
       ) THEN
        RAISE EXCEPTION 'completed customer_forecast_backtest_run is immutable';
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_forecast_backtest_run_completed_terminal
    ON customer_forecast_backtest_run;
CREATE TRIGGER trg_customer_forecast_backtest_run_completed_terminal
    BEFORE UPDATE OR DELETE ON customer_forecast_backtest_run
    FOR EACH ROW
    EXECUTE FUNCTION customer_forecast_backtest_run_guard_completed_terminal();

CREATE OR REPLACE FUNCTION customer_bottom_up_backtest_component_guard_insert()
RETURNS TRIGGER AS $$
DECLARE
    parent_status TEXT;
BEGIN
    FOR parent_status IN
        SELECT run.run_status
        FROM customer_forecast_backtest_run AS run
        JOIN (
            SELECT DISTINCT backtest_run_id
            FROM inserted_rows
        ) AS inserted
          ON inserted.backtest_run_id = run.run_id
        FOR UPDATE OF run
    LOOP
        IF parent_status <> 'generating' THEN
            RAISE EXCEPTION
                'customer_bottom_up_backtest_component requires a generating parent run';
        END IF;
    END LOOP;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_bottom_up_backtest_component_insert
    ON customer_bottom_up_backtest_component;
CREATE TRIGGER trg_customer_bottom_up_backtest_component_insert
    AFTER INSERT ON customer_bottom_up_backtest_component
    REFERENCING NEW TABLE AS inserted_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION customer_bottom_up_backtest_component_guard_insert();

CREATE OR REPLACE FUNCTION customer_bottom_up_backtest_component_block_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'customer_bottom_up_backtest_component is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_bottom_up_backtest_component_immutable
    ON customer_bottom_up_backtest_component;
CREATE TRIGGER trg_customer_bottom_up_backtest_component_immutable
    BEFORE UPDATE OR DELETE ON customer_bottom_up_backtest_component
    FOR EACH ROW
    EXECUTE FUNCTION customer_bottom_up_backtest_component_block_mutation();

CREATE OR REPLACE FUNCTION customer_bottom_up_backtest_accuracy_guard_insert()
RETURNS TRIGGER AS $$
DECLARE
    parent_status TEXT;
BEGIN
    FOR parent_status IN
        SELECT run.run_status
        FROM customer_forecast_backtest_run AS run
        JOIN (
            SELECT DISTINCT backtest_run_id
            FROM inserted_rows
        ) AS inserted
          ON inserted.backtest_run_id = run.run_id
        FOR UPDATE OF run
    LOOP
        IF parent_status <> 'generating' THEN
            RAISE EXCEPTION
                'customer_bottom_up_backtest_accuracy requires a generating parent run';
        END IF;
    END LOOP;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_bottom_up_backtest_accuracy_insert
    ON customer_bottom_up_backtest_accuracy;
CREATE TRIGGER trg_customer_bottom_up_backtest_accuracy_insert
    AFTER INSERT ON customer_bottom_up_backtest_accuracy
    REFERENCING NEW TABLE AS inserted_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION customer_bottom_up_backtest_accuracy_guard_insert();

CREATE OR REPLACE FUNCTION customer_bottom_up_backtest_accuracy_block_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'customer_bottom_up_backtest_accuracy is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_bottom_up_backtest_accuracy_immutable
    ON customer_bottom_up_backtest_accuracy;
CREATE TRIGGER trg_customer_bottom_up_backtest_accuracy_immutable
    BEFORE UPDATE OR DELETE ON customer_bottom_up_backtest_accuracy
    FOR EACH ROW
    EXECUTE FUNCTION customer_bottom_up_backtest_accuracy_block_mutation();

CREATE OR REPLACE FUNCTION customer_bottom_up_blend_component_guard_insert()
RETURNS TRIGGER AS $$
DECLARE
    parent_status TEXT;
BEGIN
    FOR parent_status IN
        SELECT run.run_status
        FROM forecast_generation_run AS run
        JOIN (SELECT DISTINCT run_id FROM inserted_rows) AS inserted
          ON inserted.run_id = run.run_id
        FOR UPDATE OF run
    LOOP
        IF parent_status <> 'generating' THEN
            RAISE EXCEPTION
                'customer_bottom_up_blend_component requires a generating parent run';
        END IF;
    END LOOP;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_bottom_up_blend_component_insert
    ON customer_bottom_up_blend_component;
CREATE TRIGGER trg_customer_bottom_up_blend_component_insert
    AFTER INSERT ON customer_bottom_up_blend_component
    REFERENCING NEW TABLE AS inserted_rows
    FOR EACH STATEMENT
    EXECUTE FUNCTION customer_bottom_up_blend_component_guard_insert();

CREATE OR REPLACE FUNCTION customer_bottom_up_blend_component_block_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'customer_bottom_up_blend_component is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_customer_bottom_up_blend_component_immutable
    ON customer_bottom_up_blend_component;
CREATE TRIGGER trg_customer_bottom_up_blend_component_immutable
    BEFORE UPDATE OR DELETE ON customer_bottom_up_blend_component
    FOR EACH ROW
    EXECUTE FUNCTION customer_bottom_up_blend_component_block_mutation();

COMMIT;
