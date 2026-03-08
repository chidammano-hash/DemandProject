-- F4.3 — Promotion & Event Planning

CREATE TABLE IF NOT EXISTS fact_event_calendar (
    event_id                BIGSERIAL       PRIMARY KEY,
    event_type              VARCHAR(30)     NOT NULL,  -- 'promo' | 'new_launch' | 'phase_out' | 'holiday' | 'cannibalization'
    event_name              VARCHAR(200)    NOT NULL,
    event_start             DATE            NOT NULL,
    event_end               DATE            NOT NULL,
    uplift_pct              NUMERIC(6,3)    NOT NULL DEFAULT 0,
    ramp_profile            VARCHAR(20)     NOT NULL DEFAULT 'linear',  -- 'linear' | 's_curve' | 'immediate'
    ramp_weeks              INTEGER         NOT NULL DEFAULT 1,
    peak_qty_weekly         NUMERIC(12,2),
    decay_rate              NUMERIC(6,4),
    pantry_loading_pct      NUMERIC(6,3)    NOT NULL DEFAULT 0,
    pantry_loading_weeks    INTEGER         NOT NULL DEFAULT 0,
    last_order_date         DATE,
    cannibalized_item_no    VARCHAR(50),
    override_multiplier     NUMERIC(6,4),
    target_items            JSONB,
    target_locations        JSONB,
    target_categories       JSONB,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',  -- 'draft' | 'approved' | 'active' | 'completed'
    conflict_resolution     VARCHAR(30)     NOT NULL DEFAULT 'highest_priority',
    priority                INTEGER         NOT NULL DEFAULT 5,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_cal_dates
    ON fact_event_calendar (event_start, event_end, status);

CREATE INDEX IF NOT EXISTS idx_event_cal_type_status
    ON fact_event_calendar (event_type, status);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_event_adjusted_forecast (
    id                      BIGSERIAL       PRIMARY KEY,
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    plan_month              DATE            NOT NULL,
    event_id                BIGINT          NOT NULL REFERENCES fact_event_calendar(event_id),
    base_forecast_qty       NUMERIC(12,2)   NOT NULL,
    event_adjustment_qty    NUMERIC(12,2)   NOT NULL DEFAULT 0,
    post_promo_dip_qty      NUMERIC(12,2)   NOT NULL DEFAULT 0,
    adjusted_forecast_qty   NUMERIC(12,2)   NOT NULL,
    adjustment_type         VARCHAR(30)     NOT NULL,
    order_deadline          DATE,
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_event_adj UNIQUE (item_no, loc, plan_month, event_id)
);

CREATE INDEX IF NOT EXISTS idx_event_adj_item_loc
    ON fact_event_adjusted_forecast (item_no, loc, plan_month);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_event_performance (
    id                      BIGSERIAL       PRIMARY KEY,
    event_id                BIGINT          NOT NULL REFERENCES fact_event_calendar(event_id),
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    plan_month              DATE            NOT NULL,
    forecasted_lift_qty     NUMERIC(12,2),
    actual_sales_qty        NUMERIC(12,2),
    actual_lift_qty         NUMERIC(12,2),
    lift_accuracy_pct       NUMERIC(6,2),
    uplift_calibration_factor NUMERIC(6,4),
    CONSTRAINT uq_event_perf UNIQUE (event_id, item_no, loc, plan_month)
);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_event_conflicts (
    conflict_id         BIGSERIAL       PRIMARY KEY,
    event_id_a          BIGINT          NOT NULL REFERENCES fact_event_calendar(event_id),
    event_id_b          BIGINT          NOT NULL REFERENCES fact_event_calendar(event_id),
    overlap_start       DATE            NOT NULL,
    overlap_end         DATE            NOT NULL,
    resolution_status   VARCHAR(20)     NOT NULL DEFAULT 'open'
);
