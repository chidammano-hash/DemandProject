-- F3.2 — Service Level Actuals vs. Targets Tracking

CREATE TABLE IF NOT EXISTS fact_service_level_targets (
    id                  BIGSERIAL       PRIMARY KEY,
    abc_class           CHAR(1)         NOT NULL,
    item_id             VARCHAR(50),
    loc                 VARCHAR(50),
    target_fill_rate    NUMERIC(5,4)    NOT NULL,  -- e.g. 0.9750 = 97.5%
    effective_from      DATE,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sl_target_class_item_loc
    ON fact_service_level_targets (abc_class, COALESCE(item_id, ''), COALESCE(loc, ''));

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_service_level_performance (
    id                  BIGSERIAL       PRIMARY KEY,
    item_id             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    perf_month          DATE            NOT NULL,
    abc_class           CHAR(1),
    actual_fill_rate    NUMERIC(5,4),
    target_fill_rate    NUMERIC(5,4),
    gap                 NUMERIC(6,4),    -- actual - target (negative = miss)
    gap_direction       VARCHAR(20),     -- 'above_target' | 'below_target' | 'on_target'
    stockout_events     INTEGER         NOT NULL DEFAULT 0,
    miss_streak_months  INTEGER         NOT NULL DEFAULT 0,
    flagged_for_review  BOOLEAN         NOT NULL DEFAULT FALSE,
    primary_miss_reason VARCHAR(50),    -- 'insufficient_ss' | 'lead_time_variance' | 'demand_spike' | 'data_gap'
    computed_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_sl_perf UNIQUE (item_id, loc, perf_month)
);

CREATE INDEX IF NOT EXISTS idx_sl_perf_item_loc_month
    ON fact_service_level_performance (item_id, loc, perf_month DESC);

CREATE INDEX IF NOT EXISTS idx_sl_perf_flagged
    ON fact_service_level_performance (flagged_for_review, perf_month)
    WHERE flagged_for_review = TRUE;

CREATE INDEX IF NOT EXISTS idx_sl_perf_streak
    ON fact_service_level_performance (miss_streak_months DESC, abc_class)
    WHERE miss_streak_months >= 2;
