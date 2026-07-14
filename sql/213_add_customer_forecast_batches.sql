-- Resumable, parallel customer-forecast batch ledger (Spec 35).
--
-- Batch rows are the durable unit of work. A forecast batch and its fact rows
-- commit together, so retrying a failed/cancelled run preserves completed work.

BEGIN;

ALTER TABLE customer_forecast_run
    ADD COLUMN IF NOT EXISTS total_series INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS completed_series INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS total_batches INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS completed_batches INTEGER NOT NULL DEFAULT 0;

ALTER TABLE customer_forecast_run
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_batch_progress;

ALTER TABLE customer_forecast_run
    ADD CONSTRAINT chk_customer_forecast_batch_progress CHECK (
        total_series >= 0
        AND completed_series >= 0
        AND completed_series <= total_series
        AND total_batches >= 0
        AND completed_batches >= 0
        AND completed_batches <= total_batches
    );

CREATE TABLE IF NOT EXISTS customer_forecast_batch (
    batch_id            BIGSERIAL PRIMARY KEY,
    run_id              UUID NOT NULL
                            REFERENCES customer_forecast_run (run_id)
                            ON DELETE CASCADE,
    route_model_id      TEXT NOT NULL,
    route_batch_no      INTEGER NOT NULL,
    batch_status        TEXT NOT NULL DEFAULT 'pending',
    series_count        INTEGER NOT NULL,
    completed_series    INTEGER NOT NULL DEFAULT 0,
    row_count           INTEGER NOT NULL DEFAULT 0,
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    source_checksum     TEXT,
    error_summary       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    CONSTRAINT uq_customer_forecast_batch_route
        UNIQUE (run_id, route_model_id, route_batch_no),
    CONSTRAINT chk_customer_forecast_batch_route CHECK (
        route_model_id IN ('chronos2_enriched', 'croston')
    ),
    CONSTRAINT chk_customer_forecast_batch_status CHECK (
        batch_status IN ('pending', 'running', 'completed', 'failed')
    ),
    CONSTRAINT chk_customer_forecast_batch_counts CHECK (
        route_batch_no >= 0
        AND series_count > 0
        AND completed_series >= 0
        AND completed_series <= series_count
        AND row_count >= 0
        AND attempt_count >= 0
    ),
    CONSTRAINT chk_customer_forecast_batch_checksum CHECK (
        source_checksum IS NULL OR source_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_customer_forecast_batch_completed CHECK (
        batch_status <> 'completed'
        OR (
            completed_at IS NOT NULL
            AND completed_series = series_count
            AND row_count > 0
            AND source_checksum IS NOT NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_customer_forecast_batch_claim
    ON customer_forecast_batch (run_id, route_model_id, batch_status, route_batch_no)
    WHERE batch_status IN ('pending', 'failed');

CREATE TABLE IF NOT EXISTS customer_forecast_batch_series (
    run_id          UUID NOT NULL
                        REFERENCES customer_forecast_run (run_id)
                        ON DELETE CASCADE,
    batch_id        BIGINT NOT NULL
                        REFERENCES customer_forecast_batch (batch_id)
                        ON DELETE CASCADE,
    item_id         TEXT NOT NULL,
    location_id     TEXT NOT NULL,
    customer_no     TEXT NOT NULL,
    CONSTRAINT pk_customer_forecast_batch_series
        PRIMARY KEY (batch_id, item_id, location_id, customer_no),
    CONSTRAINT uq_customer_forecast_run_series
        UNIQUE (run_id, item_id, location_id, customer_no)
);

CREATE INDEX IF NOT EXISTS idx_customer_forecast_batch_series_run
    ON customer_forecast_batch_series (run_id, batch_id);

ALTER TABLE fact_customer_forecast
    ADD COLUMN IF NOT EXISTS batch_id BIGINT;

ALTER TABLE fact_customer_forecast
    DROP CONSTRAINT IF EXISTS fk_customer_forecast_batch;

ALTER TABLE fact_customer_forecast
    ADD CONSTRAINT fk_customer_forecast_batch
        FOREIGN KEY (batch_id)
        REFERENCES customer_forecast_batch (batch_id)
        NOT VALID;

CREATE INDEX IF NOT EXISTS idx_customer_forecast_fact_batch
    ON fact_customer_forecast (run_id, batch_id);

COMMIT;
