-- F3.3 — Supplier Performance & Lead Time Learning

CREATE TABLE IF NOT EXISTS fact_lead_time_actuals (
    id                      BIGSERIAL       PRIMARY KEY,
    po_number               VARCHAR(50)     NOT NULL,
    line_number             INTEGER         NOT NULL DEFAULT 1,
    supplier_id             VARCHAR(50),
    item_id                 VARCHAR(50)     NOT NULL,
    item_category           VARCHAR(100),
    loc                     VARCHAR(50)     NOT NULL,
    promised_delivery_date  DATE,
    actual_receipt_date     DATE            NOT NULL,
    lead_time_days_promised INTEGER,
    lead_time_days_actual   INTEGER,
    lead_time_variance_days INTEGER GENERATED ALWAYS AS
        (lead_time_days_actual - COALESCE(lead_time_days_promised, lead_time_days_actual)) STORED,
    on_time                 BOOLEAN GENERATED ALWAYS AS
        (lead_time_days_actual <= COALESCE(lead_time_days_promised, lead_time_days_actual + 1)) STORED,
    partial_receipt         BOOLEAN         NOT NULL DEFAULT FALSE,
    source_file             VARCHAR(200),
    load_ts                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_lt_actuals UNIQUE (po_number, line_number)
);

CREATE INDEX IF NOT EXISTS idx_lt_actuals_supplier_cat
    ON fact_lead_time_actuals (supplier_id, item_category, actual_receipt_date DESC);

CREATE INDEX IF NOT EXISTS idx_lt_actuals_item_loc
    ON fact_lead_time_actuals (item_id, loc, actual_receipt_date DESC);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_lead_time_profile (
    id                      BIGSERIAL       PRIMARY KEY,
    supplier_id             VARCHAR(50)     NOT NULL,
    item_category           VARCHAR(100)    NOT NULL DEFAULT '',
    loc                     VARCHAR(50)     NOT NULL DEFAULT '',
    mean_lt_days            NUMERIC(6,2),
    stddev_lt_days          NUMERIC(6,2),
    p50_lt_days             NUMERIC(6,2),
    p90_lt_days             NUMERIC(6,2),
    p95_lt_days             NUMERIC(6,2),
    on_time_delivery_rate   NUMERIC(5,4),
    sample_size             INTEGER         NOT NULL DEFAULT 0,
    prior_mean_lt_days      NUMERIC(6,2),
    prior_stddev_lt_days    NUMERIC(6,2),
    prior_otdr              NUMERIC(5,4),
    flagged_for_ss_review   BOOLEAN         NOT NULL DEFAULT FALSE,
    window_months           INTEGER         NOT NULL DEFAULT 12,
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_lt_profile UNIQUE (supplier_id, item_category, loc)
);

CREATE INDEX IF NOT EXISTS idx_lt_profile_supplier
    ON dim_lead_time_profile (supplier_id);

CREATE INDEX IF NOT EXISTS idx_lt_profile_flagged
    ON dim_lead_time_profile (flagged_for_ss_review)
    WHERE flagged_for_ss_review = TRUE;

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_lt_review_triggers (
    id                  BIGSERIAL       PRIMARY KEY,
    supplier_id         VARCHAR(50)     NOT NULL,
    trigger_type        VARCHAR(50)     NOT NULL,  -- 'mean_lt_change' | 'stddev_change' | 'otdr_degradation'
    old_mean_lt_days    NUMERIC(6,2),
    new_mean_lt_days    NUMERIC(6,2),
    old_stddev_lt_days  NUMERIC(6,2),
    new_stddev_lt_days  NUMERIC(6,2),
    old_otdr            NUMERIC(5,4),
    new_otdr            NUMERIC(5,4),
    affected_dfu_count  INTEGER,
    review_status       VARCHAR(20)     NOT NULL DEFAULT 'open',  -- 'open' | 'acknowledged'
    acknowledged_at     TIMESTAMPTZ,
    triggered_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lt_triggers_status
    ON fact_lt_review_triggers (review_status, triggered_at DESC)
    WHERE review_status = 'open';
