-- 202_create_forecast_snapshot.sql
-- Bounded live-forward archive: champion plus three frozen contender models,
-- restricted at the database boundary to snapshot lags 0..5.

CREATE TABLE IF NOT EXISTS forecast_snapshot_roster (
    record_month          DATE NOT NULL,
    model_id              VARCHAR(100) NOT NULL,
    snapshot_role         TEXT NOT NULL CHECK (snapshot_role IN ('champion', 'contender')),
    contender_rank        SMALLINT,
    source_backtest_run_id INTEGER REFERENCES backtest_run(id) ON DELETE RESTRICT,
    rank_wape             NUMERIC,
    generation_run_id     UUID,
    selected_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (record_month, model_id),
    CONSTRAINT chk_snapshot_roster_record_month_start CHECK (EXTRACT(DAY FROM record_month) = 1),
    CONSTRAINT chk_snapshot_roster_role_fields CHECK (
        (snapshot_role = 'champion'
         AND model_id = 'champion'
         AND contender_rank IS NULL
         AND source_backtest_run_id IS NULL
         AND rank_wape IS NULL
         AND generation_run_id IS NULL)
        OR
        (snapshot_role = 'contender'
         AND model_id <> 'champion'
         AND contender_rank BETWEEN 1 AND 3
         AND source_backtest_run_id IS NOT NULL
         AND rank_wape IS NOT NULL
         AND generation_run_id IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_forecast_snapshot_roster_champion
    ON forecast_snapshot_roster (record_month)
    WHERE snapshot_role = 'champion';

CREATE UNIQUE INDEX IF NOT EXISTS uq_forecast_snapshot_roster_contender_rank
    ON forecast_snapshot_roster (record_month, contender_rank)
    WHERE snapshot_role = 'contender';

CREATE TABLE IF NOT EXISTS fact_forecast_snapshot (
    snapshot_sk          BIGSERIAL PRIMARY KEY,
    record_month         DATE NOT NULL,
    model_id             VARCHAR(100) NOT NULL,
    item_id              VARCHAR(50) NOT NULL,
    loc                  VARCHAR(50) NOT NULL,
    forecast_month       DATE NOT NULL,
    lag                  SMALLINT GENERATED ALWAYS AS (
        ((EXTRACT(YEAR FROM forecast_month) - EXTRACT(YEAR FROM record_month)) * 12
         + (EXTRACT(MONTH FROM forecast_month) - EXTRACT(MONTH FROM record_month)))::smallint
    ) STORED,
    horizon_months       SMALLINT,
    forecast_qty         NUMERIC(12, 2) NOT NULL,
    forecast_qty_lower   NUMERIC(12, 2),
    forecast_qty_upper   NUMERIC(12, 2),
    source_model_id      VARCHAR(100),
    cluster_id           TEXT,
    plan_version         VARCHAR(30),
    run_id               UUID NOT NULL,
    generated_at         TIMESTAMPTZ,
    archived_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_fact_forecast_snapshot UNIQUE (record_month, model_id, item_id, loc, forecast_month),
    CONSTRAINT fk_fact_forecast_snapshot_roster
        FOREIGN KEY (record_month, model_id)
        REFERENCES forecast_snapshot_roster (record_month, model_id)
        ON DELETE RESTRICT,
    CONSTRAINT chk_fact_forecast_snapshot_record_month_start CHECK (EXTRACT(DAY FROM record_month) = 1),
    CONSTRAINT chk_fact_forecast_snapshot_month_start CHECK (EXTRACT(DAY FROM forecast_month) = 1),
    CONSTRAINT chk_fact_forecast_snapshot_lag CHECK (lag BETWEEN 0 AND 5)
);

CREATE INDEX IF NOT EXISTS idx_forecast_snapshot_record_month_lag
    ON fact_forecast_snapshot (record_month, lag);
CREATE INDEX IF NOT EXISTS idx_forecast_snapshot_model_record_month
    ON fact_forecast_snapshot (model_id, record_month);
CREATE INDEX IF NOT EXISTS idx_forecast_snapshot_item_loc_month
    ON fact_forecast_snapshot (item_id, loc, forecast_month);

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_accuracy_snapshot AS
WITH closed_months AS (
    SELECT DISTINCT startdate
    FROM fact_sales_monthly
    WHERE type = 1
),
actuals AS (
    SELECT item_id, loc, startdate, SUM(qty) AS actual_qty
    FROM fact_sales_monthly
    WHERE type = 1
    GROUP BY item_id, loc, startdate
)
SELECT
    f.record_month,
    f.model_id,
    f.lag,
    f.horizon_months,
    f.item_id,
    f.loc,
    f.forecast_month,
    f.forecast_qty,
    COALESCE(a.actual_qty, 0)::numeric AS actual_qty,
    ABS(f.forecast_qty - COALESCE(a.actual_qty, 0))::numeric AS abs_error,
    CURRENT_TIMESTAMP AS last_refresh_at
FROM fact_forecast_snapshot f
JOIN closed_months c ON c.startdate = f.forecast_month
LEFT JOIN actuals a
  ON a.item_id = f.item_id
 AND a.loc = f.loc
 AND a.startdate = f.forecast_month
WHERE EXISTS (
    SELECT 1
    FROM fact_sales_monthly active_sales
    WHERE active_sales.type = 1
      AND active_sales.item_id = f.item_id
      AND active_sales.loc = f.loc
      AND active_sales.startdate BETWEEN f.forecast_month - INTERVAL '11 months' AND f.forecast_month
)
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uq_agg_accuracy_snapshot_dfu_month
    ON agg_accuracy_snapshot (record_month, model_id, item_id, loc, forecast_month);
