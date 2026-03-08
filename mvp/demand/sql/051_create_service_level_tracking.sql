-- F3.2 — Service Level Actuals vs. Targets Tracking

CREATE TABLE IF NOT EXISTS fact_service_level_targets (
    id                  BIGSERIAL       PRIMARY KEY,
    abc_class           CHAR(1)         NOT NULL,
    item_no             VARCHAR(50),
    loc                 VARCHAR(50),
    target_fill_rate    NUMERIC(5,4)    NOT NULL,  -- e.g. 0.9750 = 97.5%
    effective_from      DATE,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sl_target_class_item_loc
    ON fact_service_level_targets (abc_class, COALESCE(item_no, ''), COALESCE(loc, ''));

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_service_level_performance (
    id                  BIGSERIAL       PRIMARY KEY,
    item_no             VARCHAR(50)     NOT NULL,
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
    CONSTRAINT uq_sl_perf UNIQUE (item_no, loc, perf_month)
);

CREATE INDEX IF NOT EXISTS idx_sl_perf_item_loc_month
    ON fact_service_level_performance (item_no, loc, perf_month DESC);

CREATE INDEX IF NOT EXISTS idx_sl_perf_flagged
    ON fact_service_level_performance (flagged_for_review, perf_month)
    WHERE flagged_for_review = TRUE;

CREATE INDEX IF NOT EXISTS idx_sl_perf_streak
    ON fact_service_level_performance (miss_streak_months DESC, abc_class)
    WHERE miss_streak_months >= 2;

-- -----------------------------------------------------------------------
-- Materialized view: current 3-month SL health per DFU
-- (stubbed with empty select; populated by compute_service_level_actuals.py)

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_service_level_dashboard AS
SELECT
    item_no, loc, abc_class,
    MAX(perf_month)          AS latest_month,
    AVG(actual_fill_rate)    AS avg_fill_rate_3m,
    AVG(target_fill_rate)    AS target_fill_rate,
    MIN(gap)                 AS worst_gap,
    MAX(miss_streak_months)  AS current_streak,
    BOOL_OR(flagged_for_review) AS any_flagged,
    CASE
        WHEN MIN(gap) >= 0                   THEN 'green'
        WHEN MIN(gap) >= -0.03               THEN 'amber'
        ELSE                                      'red'
    END                      AS rag_status
FROM fact_service_level_performance
WHERE perf_month >= (CURRENT_DATE - INTERVAL '3 months')
GROUP BY item_no, loc, abc_class
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uq_mv_sl_dashboard
    ON mv_service_level_dashboard (item_no, loc);
