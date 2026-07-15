-- Enable the statistical customer rule router v2 and precompute its causal
-- routing features. Historical v1 rows remain readable; every new run and
-- per-series row must use the v2 lineage and one of its eight routes.

BEGIN;

-- Build beside the live profile while holding the same lock used by customer-
-- demand loads. The final rename is a short metadata swap after the full fact
-- scan and index builds have completed.
SELECT pg_advisory_xact_lock(
    hashtext('customer_demand_load_and_profile_refresh')
);

DROP MATERIALIZED VIEW IF EXISTS mv_customer_demand_series_profile_router_v2;

CREATE MATERIALIZED VIEW mv_customer_demand_series_profile_router_v2 AS
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
    MAX(demand.startdate) FILTER (WHERE demand.demand_qty > 0) AS last_demand_month,
    COUNT(*) FILTER (
        WHERE demand.demand_qty > 0
          AND demand.startdate
              >= source_month.latest_month - INTERVAL '11 months'
          AND demand.startdate <= source_month.latest_month
    ) AS demand_months_last_12,
    COUNT(*) FILTER (
        WHERE demand.demand_qty > 0
          AND demand.startdate
              >= source_month.latest_month - INTERVAL '17 months'
          AND demand.startdate <= source_month.latest_month
    ) AS demand_months_last_18,
    COALESCE(
        SUM(demand.demand_qty) FILTER (
            WHERE demand.demand_qty > 0
              AND demand.startdate
                  >= source_month.latest_month - INTERVAL '17 months'
              AND demand.startdate <= source_month.latest_month
        ),
        0::numeric
    ) AS demand_sum_last_18,
    COALESCE(
        SUM(demand.demand_qty * demand.demand_qty) FILTER (
            WHERE demand.demand_qty > 0
              AND demand.startdate
                  >= source_month.latest_month - INTERVAL '17 months'
              AND demand.startdate <= source_month.latest_month
        ),
        0::numeric
    ) AS demand_sumsq_last_18,
    COUNT(*) FILTER (
        WHERE demand.demand_qty > 0
          AND demand.startdate
              >= source_month.latest_month - INTERVAL '5 months'
          AND demand.startdate <= source_month.latest_month
    ) AS demand_months_recent_6,
    COUNT(*) FILTER (
        WHERE demand.demand_qty > 0
          AND demand.startdate
              >= source_month.latest_month - INTERVAL '11 months'
          AND demand.startdate
              <= source_month.latest_month - INTERVAL '6 months'
    ) AS demand_months_previous_6,
    COALESCE(
        SUM(demand.demand_qty) FILTER (
            WHERE demand.demand_qty > 0
              AND demand.startdate
                  >= source_month.latest_month - INTERVAL '5 months'
              AND demand.startdate <= source_month.latest_month
        ),
        0::numeric
    ) AS demand_sum_recent_6,
    COALESCE(
        SUM(demand.demand_qty) FILTER (
            WHERE demand.demand_qty > 0
              AND demand.startdate
                  >= source_month.latest_month - INTERVAL '11 months'
              AND demand.startdate
                  <= source_month.latest_month - INTERVAL '6 months'
        ),
        0::numeric
    ) AS demand_sum_previous_6,
    -- Eighteen source months provide only one full annual cycle, so a causal
    -- twelve-month seasonal holdout cannot yet be validated. Keep the flag
    -- non-null and false until a longer-history evidence migration replaces it.
    FALSE::boolean AS seasonal_repeat_validated
FROM fact_customer_demand_monthly demand
CROSS JOIN source_month
GROUP BY
    demand.item_id,
    demand.location_id,
    demand.customer_no,
    source_month.latest_month;

CREATE UNIQUE INDEX uq_mv_customer_demand_series_profile_router_v2
    ON mv_customer_demand_series_profile_router_v2
       (item_id, location_id, customer_no);

CREATE INDEX idx_mv_customer_demand_series_profile_bounds_router_v2
    ON mv_customer_demand_series_profile_router_v2 (first_month, last_month);

CREATE INDEX idx_mv_customer_demand_series_profile_last_sales_router_v2
    ON mv_customer_demand_series_profile_router_v2 (last_sales_month, first_month);

CREATE INDEX idx_mv_customer_demand_series_profile_rules_router_v2
    ON mv_customer_demand_series_profile_router_v2
       (last_sales_month, first_demand_month, demand_months_last_12);

ANALYZE mv_customer_demand_series_profile_router_v2;

DROP MATERIALIZED VIEW IF EXISTS mv_customer_demand_series_profile;
ALTER MATERIALIZED VIEW mv_customer_demand_series_profile_router_v2
    RENAME TO mv_customer_demand_series_profile;
ALTER INDEX uq_mv_customer_demand_series_profile_router_v2
    RENAME TO uq_mv_customer_demand_series_profile;
ALTER INDEX idx_mv_customer_demand_series_profile_bounds_router_v2
    RENAME TO idx_mv_customer_demand_series_profile_bounds;
ALTER INDEX idx_mv_customer_demand_series_profile_last_sales_router_v2
    RENAME TO idx_mv_customer_demand_series_profile_last_sales;
ALTER INDEX idx_mv_customer_demand_series_profile_rules_router_v2
    RENAME TO idx_mv_customer_demand_series_profile_rules;
-- The side-build name no longer exists after the rename. Keep an explicit
-- cleanup event so DDL inventory tooling does not classify it as a live MV.
DROP MATERIALIZED VIEW IF EXISTS mv_customer_demand_series_profile_router_v2;

-- Stop unfinished v1 batches before retiring their parent manifests. Workers
-- that wake after this commit cannot continue writing under the v2 contract.
UPDATE customer_forecast_batch AS batch
SET batch_status = 'failed',
    error_summary = 'customer forecast routing configuration changed to v2',
    completed_at = NOW()
FROM customer_forecast_run AS run
WHERE batch.run_id = run.run_id
  AND batch.batch_status IN ('pending', 'running')
  AND run.run_status IN ('queued', 'generating')
  AND run.model_id <> 'customer_rule_router_v2';

UPDATE customer_forecast_run
SET run_status = 'failed',
    error_summary = 'customer forecast routing configuration changed to v2',
    completed_at = NOW()
WHERE run_status IN ('queued', 'generating')
  AND model_id <> 'customer_rule_router_v2';

UPDATE customer_forecast_backtest_run
SET run_status = 'failed',
    error_summary = 'customer forecast routing configuration changed to v2',
    completed_at = NOW()
WHERE run_status IN ('queued', 'generating')
  AND customer_model_id <> 'customer_rule_router_v2';

-- Generating drafts and ready-but-unpromoted release candidates freeze v1
-- customer lineage. Invalidate them while preserving their evidence rows for
-- audit. Already promoted releases are intentionally untouched.
UPDATE forecast_generation_run AS generation
SET run_status = 'invalid',
    promotion_eligible = FALSE,
    invalid_reason = 'customer forecast routing configuration changed to v2',
    completed_at = NOW()
FROM customer_forecast_run AS customer
WHERE generation.run_status IN ('generating', 'ready')
  AND generation.metadata ? 'customer_bottom_up_blend'
  AND generation.metadata
          -> 'customer_bottom_up_blend'
          ->> 'customer_run_id' = customer.run_id::text
  AND customer.model_id <> 'customer_rule_router_v2';

-- NOT VALID preserves historical v1 rows while enforcing v2 lineage and route
-- IDs on every row inserted or updated after this migration.
ALTER TABLE customer_forecast_run
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_run_model,
    ADD CONSTRAINT chk_customer_forecast_run_model
        CHECK (model_id = 'customer_rule_router_v2') NOT VALID;

ALTER TABLE customer_forecast_batch
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_batch_route,
    ADD CONSTRAINT chk_customer_forecast_batch_route
        CHECK (
            route_model_id IN (
                'moving_average_3',
                'trailing_average_6',
                'seasonal_repeat_12',
                'tsb',
                'adida',
                'croston',
                'ses',
                'holt_damped'
            )
        ) NOT VALID;

ALTER TABLE fact_customer_forecast
    DROP CONSTRAINT IF EXISTS chk_customer_forecast_fact_route,
    ADD CONSTRAINT chk_customer_forecast_fact_route
        CHECK (
            model_id IN (
                'moving_average_3',
                'trailing_average_6',
                'seasonal_repeat_12',
                'tsb',
                'adida',
                'croston',
                'ses',
                'holt_damped'
            )
        ) NOT VALID;

ALTER TABLE customer_forecast_backtest_run
    DROP CONSTRAINT IF EXISTS chk_customer_backtest_contract,
    ADD CONSTRAINT chk_customer_backtest_contract CHECK (
        customer_model_id = 'customer_rule_router_v2'
        AND blend_model_id = 'customer_bottom_up_blend'
        AND batch_size > 0
    ) NOT VALID;

COMMIT;
