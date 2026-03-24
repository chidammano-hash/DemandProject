-- F3.1 — Forecast Bias Correction Engine
-- Tables: fact_bias_corrections, fact_bias_correction_history

CREATE TABLE IF NOT EXISTS fact_bias_corrections (
    id                      BIGSERIAL       PRIMARY KEY,
    item_id                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    plan_month              DATE            NOT NULL,
    segment_type            VARCHAR(30)     NOT NULL,
    segment_value           VARCHAR(100)    NOT NULL,
    rolling_bias_3m         NUMERIC(8,4)    NOT NULL,
    rolling_wape_3m         NUMERIC(8,4),
    bias_month1             NUMERIC(8,4),
    bias_month2             NUMERIC(8,4),
    bias_month3             NUMERIC(8,4),
    wape_month1             NUMERIC(8,4),
    wape_month2             NUMERIC(8,4),
    wape_month3             NUMERIC(8,4),
    correction_factor_raw   NUMERIC(6,4)    NOT NULL,
    correction_factor       NUMERIC(6,4)    NOT NULL,
    correction_was_clipped  BOOLEAN         NOT NULL DEFAULT FALSE,
    raw_forecast_qty        NUMERIC(12,2),
    corrected_forecast_qty  NUMERIC(12,2),
    correction_pct          NUMERIC(6,2),
    flagged_for_review      BOOLEAN         NOT NULL DEFAULT FALSE,
    correction_applied      BOOLEAN         NOT NULL DEFAULT FALSE,
    applied_at              TIMESTAMPTZ,
    applied_to_version      VARCHAR(50),
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    months_of_data          INTEGER,
    CONSTRAINT uq_bias_correction UNIQUE (item_id, loc, plan_month, segment_type)
);

CREATE INDEX IF NOT EXISTS idx_bias_correction_item_loc_month
    ON fact_bias_corrections (item_id, loc, plan_month);

CREATE INDEX IF NOT EXISTS idx_bias_correction_segment
    ON fact_bias_corrections (segment_type, segment_value, plan_month);

CREATE INDEX IF NOT EXISTS idx_bias_correction_flagged
    ON fact_bias_corrections (flagged_for_review, plan_month)
    WHERE flagged_for_review = TRUE;

CREATE INDEX IF NOT EXISTS idx_bias_correction_applied
    ON fact_bias_corrections (correction_applied, plan_month);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_bias_correction_history (
    id                          BIGSERIAL       PRIMARY KEY,
    segment_type                VARCHAR(30)     NOT NULL,
    segment_value               VARCHAR(100)    NOT NULL,
    computation_month           DATE            NOT NULL,
    rolling_bias_3m             NUMERIC(8,4)    NOT NULL,
    correction_factor           NUMERIC(6,4)    NOT NULL,
    dfu_count_in_segment        INTEGER,
    avg_raw_wape                NUMERIC(6,4),
    avg_corrected_wape          NUMERIC(6,4),
    correction_improved_accuracy BOOLEAN,
    computed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bias_history_segment_month
    ON fact_bias_correction_history (segment_type, segment_value, computation_month DESC);
