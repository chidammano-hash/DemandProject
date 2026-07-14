-- Couple every customer forecast batch payload to the same run manifest.

BEGIN;

ALTER TABLE customer_forecast_batch
    ADD CONSTRAINT uq_customer_forecast_batch_run_identity
        UNIQUE (run_id, batch_id);

ALTER TABLE customer_forecast_batch_series
    DROP CONSTRAINT IF EXISTS customer_forecast_batch_series_batch_id_fkey,
    ADD CONSTRAINT fk_customer_forecast_batch_series_run
        FOREIGN KEY (run_id, batch_id)
        REFERENCES customer_forecast_batch (run_id, batch_id)
        ON DELETE CASCADE;

ALTER TABLE fact_customer_forecast
    DROP CONSTRAINT IF EXISTS fk_customer_forecast_batch,
    ADD CONSTRAINT fk_customer_forecast_batch
        FOREIGN KEY (run_id, batch_id)
        REFERENCES customer_forecast_batch (run_id, batch_id)
        NOT VALID;

COMMIT;
