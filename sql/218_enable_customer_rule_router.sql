-- Enable the ordered customer rule router and precompute its stable route features.
--
-- Run-level lineage records the router family. Each customer series still belongs
-- to exactly one durable batch and persists one canonical per-series route ID.

BEGIN;

-- Build beside the live profile so the full fact scan does not hold an
-- exclusive lock on the API's current readiness source. The final rename is
-- a short metadata swap near the end of the transaction.
SELECT pg_advisory_xact_lock(
    hashtext('customer_demand_load_and_profile_refresh')
);

DROP MATERIALIZED VIEW IF EXISTS mv_customer_demand_series_profile_router;

CREATE MATERIALIZED VIEW mv_customer_demand_series_profile_router AS
WITH source_month AS (
    SELECT MAX(startdate) AS latest_month
    FROM fact_customer_demand_monthly
)
SELECT
    demand.item_id,
    demand.location_id,
    demand.customer_no,
    MIN(demand.startdate) AS first_month,
    MAX(demand.startdate) AS last_month,
    source_month.latest_month AS source_latest_month,
    MAX(demand.startdate) FILTER (WHERE demand.sales_qty > 0) AS last_sales_month,
    MIN(demand.startdate) FILTER (WHERE demand.demand_qty > 0) AS first_demand_month,
    COUNT(*) FILTER (
        WHERE demand.demand_qty > 0
          AND demand.startdate
              >= source_month.latest_month - INTERVAL '11 months'
    ) AS demand_months_last_12
FROM fact_customer_demand_monthly demand
CROSS JOIN source_month
GROUP BY
    demand.item_id,
    demand.location_id,
    demand.customer_no,
    source_month.latest_month;

CREATE UNIQUE INDEX uq_mv_customer_demand_series_profile_router
    ON mv_customer_demand_series_profile_router (item_id, location_id, customer_no);

CREATE INDEX idx_mv_customer_demand_series_profile_bounds_router
    ON mv_customer_demand_series_profile_router (first_month, last_month);

CREATE INDEX idx_mv_customer_demand_series_profile_last_sales_router
    ON mv_customer_demand_series_profile_router (last_sales_month, first_month);

CREATE INDEX idx_mv_customer_demand_series_profile_rules_router
    ON mv_customer_demand_series_profile_router
       (last_sales_month, first_demand_month, demand_months_last_12);

ANALYZE mv_customer_demand_series_profile_router;

DROP MATERIALIZED VIEW IF EXISTS mv_customer_demand_series_profile;
ALTER MATERIALIZED VIEW mv_customer_demand_series_profile_router
    RENAME TO mv_customer_demand_series_profile;
ALTER INDEX uq_mv_customer_demand_series_profile_router
    RENAME TO uq_mv_customer_demand_series_profile;
ALTER INDEX idx_mv_customer_demand_series_profile_bounds_router
    RENAME TO idx_mv_customer_demand_series_profile_bounds;
ALTER INDEX idx_mv_customer_demand_series_profile_last_sales_router
    RENAME TO idx_mv_customer_demand_series_profile_last_sales;
ALTER INDEX idx_mv_customer_demand_series_profile_rules_router
    RENAME TO idx_mv_customer_demand_series_profile_rules;
-- The side-build name no longer exists after the rename. Keep an idempotent
-- cleanup event so DDL inventory tooling does not treat it as a live MV.
DROP MATERIALIZED VIEW IF EXISTS mv_customer_demand_series_profile_router;

-- A pre-router active run has a different checksum and cannot be resumed by the
-- new workers. Retire it so the one-active-run guard cannot block a new request.
UPDATE customer_forecast_run
SET run_status = 'failed',
    error_summary = 'customer forecast routing configuration changed',
    completed_at = NOW()
WHERE run_status IN ('queued', 'generating')
  AND model_id <> 'customer_rule_router';

-- Backtests and queued blend manifests freeze the customer model lineage. A
-- legacy active row cannot be resumed under the router and would otherwise
-- retain its one-active-run index indefinitely after deployment.
UPDATE customer_forecast_backtest_run
SET run_status = 'failed',
    error_summary = 'customer forecast routing configuration changed',
    completed_at = NOW()
WHERE run_status IN ('queued', 'generating')
  AND customer_model_id <> 'customer_rule_router';

UPDATE forecast_generation_run AS generation
SET run_status = 'invalid',
    promotion_eligible = FALSE,
    invalid_reason = 'customer forecast routing configuration changed',
    completed_at = NOW()
FROM customer_forecast_run AS customer
WHERE generation.run_status = 'generating'
  AND generation.metadata ? 'customer_bottom_up_blend'
  AND generation.metadata
          -> 'customer_bottom_up_blend'
          ->> 'customer_run_id' = customer.run_id::text
  AND customer.model_id <> 'customer_rule_router';

ALTER TABLE customer_forecast_run
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_run_croston_only,
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_run_model,
    ADD CONSTRAINT chk_customer_forecast_run_model
        CHECK (model_id = 'customer_rule_router') NOT VALID;

ALTER TABLE customer_forecast_batch
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_batch_route,
    ADD CONSTRAINT chk_customer_forecast_batch_route
        CHECK (
            route_model_id IN (
                'croston',
                'moving_average_3',
                'seasonal_repeat_12'
            )
        ) NOT VALID;

ALTER TABLE fact_customer_forecast
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_fact_croston_only,
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_fact_route,
    ADD CONSTRAINT chk_customer_forecast_fact_route
        CHECK (
            model_id IN (
                'croston',
                'moving_average_3',
                'seasonal_repeat_12'
            )
        ) NOT VALID;

ALTER TABLE customer_forecast_backtest_run
    DROP CONSTRAINT IF EXISTS chk_customer_backtest_contract,
    ADD CONSTRAINT chk_customer_backtest_contract CHECK (
        customer_model_id = 'customer_rule_router'
        AND blend_model_id = 'customer_bottom_up_blend'
        AND batch_size > 0
    ) NOT VALID;

COMMIT;
