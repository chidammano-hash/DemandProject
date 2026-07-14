-- Record the customer-only Chronos/Croston route composition for each run.

BEGIN;

ALTER TABLE customer_forecast_run
    ADD COLUMN IF NOT EXISTS model_route_counts JSONB NOT NULL DEFAULT '{}'::jsonb;

UPDATE customer_forecast_run
SET model_route_counts = jsonb_build_object(model_id, eligible_series)
WHERE run_status = 'completed'
  AND model_route_counts = '{}'::jsonb;

ALTER TABLE customer_forecast_run
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_model_route_counts;

ALTER TABLE customer_forecast_run
    ADD CONSTRAINT chk_customer_forecast_model_route_counts CHECK (
        jsonb_typeof(model_route_counts) = 'object'
    );

COMMIT;
