-- Generation-only customer-level forecasts (Spec 35).
--
-- This lifecycle is intentionally independent of fact_production_forecast:
-- completed runs are immutable, have no promotion state, and never reconcile
-- or write back to the governed item-location forecast.

BEGIN;

CREATE TABLE IF NOT EXISTS customer_forecast_run (
    run_id              UUID PRIMARY KEY,
    job_id              TEXT UNIQUE,
    run_status          TEXT NOT NULL,
    planning_month      DATE NOT NULL,
    history_start       DATE NOT NULL,
    history_end         DATE NOT NULL,
    forecast_start      DATE NOT NULL,
    forecast_end        DATE NOT NULL,
    history_months      SMALLINT NOT NULL,
    horizon_months      SMALLINT NOT NULL,
    eligible_series     INTEGER NOT NULL DEFAULT 0,
    row_count           INTEGER NOT NULL DEFAULT 0,
    skipped_series      INTEGER NOT NULL DEFAULT 0,
    skip_reason_counts  JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_id            TEXT NOT NULL,
    config_checksum     TEXT,
    source_checksum     TEXT,
    error_summary       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    CONSTRAINT chk_customer_forecast_run_status CHECK (
        run_status IN ('queued', 'generating', 'completed', 'failed', 'cancelled')
    ),
    CONSTRAINT chk_customer_forecast_run_months CHECK (
        planning_month = date_trunc('month', planning_month)::date
        AND history_months > 0
        AND horizon_months > 0
        AND history_start <= history_end
        AND forecast_start <= forecast_end
        AND history_end + 1 = forecast_start
    ),
    CONSTRAINT chk_customer_forecast_run_counts CHECK (
        eligible_series >= 0 AND row_count >= 0 AND skipped_series >= 0
    ),
    CONSTRAINT chk_customer_forecast_skip_reasons CHECK (
        jsonb_typeof(skip_reason_counts) = 'object'
    ),
    CONSTRAINT chk_customer_forecast_run_config_checksum CHECK (
        config_checksum IS NULL OR config_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_customer_forecast_run_source_checksum CHECK (
        source_checksum IS NULL OR source_checksum ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT chk_customer_forecast_run_completed CHECK (
        run_status <> 'completed'
        OR (
            completed_at IS NOT NULL
            AND eligible_series > 0
            AND row_count = eligible_series * horizon_months
            AND config_checksum IS NOT NULL
            AND source_checksum IS NOT NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_customer_forecast_run_status_created
    ON customer_forecast_run (run_status, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_customer_forecast_one_active
    ON customer_forecast_run ((1))
    WHERE run_status IN ('queued', 'generating');

CREATE TABLE IF NOT EXISTS fact_customer_forecast (
    run_id          UUID NOT NULL
                        REFERENCES customer_forecast_run (run_id)
                        ON DELETE CASCADE,
    item_id         TEXT NOT NULL,
    location_id     TEXT NOT NULL,
    customer_no     TEXT NOT NULL,
    forecast_month  DATE NOT NULL,
    forecast_qty    NUMERIC(18,4) NOT NULL,
    lower_bound     NUMERIC(18,4),
    upper_bound     NUMERIC(18,4),
    model_id        TEXT NOT NULL,
    history_end     DATE NOT NULL,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_customer_forecast_grain
        UNIQUE (run_id, item_id, location_id, customer_no, forecast_month),
    CONSTRAINT chk_customer_forecast_month_start CHECK (
        forecast_month = date_trunc('month', forecast_month)::date
    ),
    CONSTRAINT chk_customer_forecast_nonnegative CHECK (
        forecast_qty >= 0
        AND (lower_bound IS NULL OR lower_bound >= 0)
        AND (upper_bound IS NULL OR upper_bound >= 0)
    ),
    CONSTRAINT chk_customer_forecast_interval CHECK (
        (lower_bound IS NULL AND upper_bound IS NULL)
        OR (
            lower_bound IS NOT NULL
            AND upper_bound IS NOT NULL
            AND lower_bound <= forecast_qty
            AND forecast_qty <= upper_bound
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_customer_forecast_lookup
    ON fact_customer_forecast
       (item_id, location_id, customer_no, forecast_month, run_id);

CREATE INDEX IF NOT EXISTS idx_customer_forecast_run_month
    ON fact_customer_forecast (run_id, forecast_month);

COMMIT;
