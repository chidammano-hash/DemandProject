-- IPfeature6: Inventory Health Score Dashboard
-- Creates materialized view mv_inventory_health_score.
--
-- Health score = sum of 4 components (25 pts each) = 0–100:
--   score_ss_coverage      — safety stock coverage ratio
--   score_dos_target       — days-of-supply target adherence
--   score_stockout_risk    — recent stockout frequency (last 3 months)
--   score_forecast_accuracy — recent WAPE
--
-- Tiers: ≥80 = healthy, ≥60 = monitor, ≥40 = at_risk, <40 = critical
--
-- Dependencies:
--   agg_inventory_monthly            (existing)
--   mv_inventory_forecast_monthly    (Feature 37)
--   dim_sku                          (existing)
--   fact_safety_stock_targets        (IPfeature3 — stub created below if missing)

-- -------------------------------------------------------------------------
-- Stub table for fact_safety_stock_targets (IPfeature3 not yet implemented).
-- Has no rows: all SS-dependent score components default to neutral values.
-- When IPfeature3 is implemented it will replace this stub.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_safety_stock_targets (
    item_id           TEXT,
    loc               TEXT,
    ss_combined       NUMERIC,
    ss_coverage       NUMERIC,
    reorder_point     NUMERIC,
    is_below_ss       BOOLEAN,
    target_dos_min    NUMERIC,
    target_dos_max    NUMERIC,
    policy_version    TEXT DEFAULT 'v1'
);

-- -------------------------------------------------------------------------
-- Materialized view
-- -------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS mv_inventory_health_score CASCADE;

CREATE MATERIALIZED VIEW mv_inventory_health_score AS
WITH latest_inv AS (
    -- Most recent month per item-location from inventory aggregates
    SELECT DISTINCT ON (item_id, loc)
        item_id, loc, month_start,
        eom_qty_on_hand,
        monthly_sales,
        avg_daily_sls,
        latest_lead_time_days
    FROM agg_inventory_monthly
    ORDER BY item_id, loc, month_start DESC
),
recent_stockout AS (
    -- Stockout events in the 3 most recent months with forecast data
    SELECT item_id, loc,
        SUM(CASE WHEN is_stockout THEN 1 ELSE 0 END) AS stockout_count_3m
    FROM mv_inventory_forecast_monthly
    WHERE month_start >= (
              SELECT MAX(month_start) FROM mv_inventory_forecast_monthly
          ) - INTERVAL '2 months'
      AND model_id IN ('champion', 'external')
    GROUP BY item_id, loc
),
recent_accuracy AS (
    -- WAPE over the 3 most recent months with forecast data
    SELECT item_id, loc,
        SUM(abs_error) / NULLIF(ABS(SUM(actual_demand)), 0) AS recent_wape
    FROM mv_inventory_forecast_monthly
    WHERE month_start >= (
              SELECT MAX(month_start) FROM mv_inventory_forecast_monthly
          ) - INTERVAL '2 months'
      AND model_id IN ('champion', 'external')
    GROUP BY item_id, loc
),
ss AS (
    SELECT item_id, loc,
        ss_combined, ss_coverage, reorder_point,
        is_below_ss, target_dos_min, target_dos_max
    FROM fact_safety_stock_targets
),
scored AS (
    SELECT
        l.item_id,
        l.loc,
        l.month_start,
        d.cluster_assignment,
        d.abc_vol,
        d.variability_class,
        d.seasonality_profile,
        d.region,
        -- Current inventory position
        l.eom_qty_on_hand,
        l.avg_daily_sls,
        CASE WHEN l.avg_daily_sls > 0
             THEN l.eom_qty_on_hand / l.avg_daily_sls
             ELSE NULL
        END AS current_dos,
        -- SS targets (NULL when IPfeature3 not populated)
        s.ss_combined,
        s.reorder_point,
        s.is_below_ss,
        s.ss_coverage,
        s.target_dos_min,
        s.target_dos_max,
        -- Recent forecast accuracy
        ra.recent_wape,
        rs.stockout_count_3m,

        -- Component 1: SS Coverage (25 pts)
        CASE
            WHEN s.ss_combined IS NULL                THEN 12  -- neutral
            WHEN COALESCE(s.ss_coverage, 0) >= 1.5   THEN 25
            WHEN COALESCE(s.ss_coverage, 0) >= 1.0   THEN 18
            WHEN COALESCE(s.ss_coverage, 0) >= 0.5   THEN 10
            ELSE 0
        END AS score_ss_coverage,

        -- Component 2: DOS Target Adherence (25 pts)
        CASE
            WHEN s.target_dos_min IS NULL             THEN 15  -- neutral
            WHEN l.avg_daily_sls = 0
                 AND l.eom_qty_on_hand = 0            THEN 0   -- stockout
            WHEN l.avg_daily_sls = 0                  THEN 5   -- no sales movement
            WHEN (l.eom_qty_on_hand / l.avg_daily_sls)
                 BETWEEN s.target_dos_min
                     AND s.target_dos_max             THEN 25
            WHEN (l.eom_qty_on_hand / l.avg_daily_sls) > s.target_dos_max
                                                      THEN 10  -- excess
            ELSE 5                                             -- below minimum
        END AS score_dos_target,

        -- Component 3: Stockout Risk History (25 pts)
        CASE
            WHEN rs.stockout_count_3m IS NULL         THEN 20  -- assume OK
            WHEN rs.stockout_count_3m = 0             THEN 25
            WHEN rs.stockout_count_3m = 1             THEN 15
            WHEN rs.stockout_count_3m = 2             THEN 8
            ELSE 0                                             -- 3 = chronic
        END AS score_stockout_risk,

        -- Component 4: Forecast Accuracy (25 pts)
        CASE
            WHEN ra.recent_wape IS NULL               THEN 15  -- neutral
            WHEN ra.recent_wape < 0.15                THEN 25  -- excellent
            WHEN ra.recent_wape < 0.25                THEN 20  -- good
            WHEN ra.recent_wape < 0.40                THEN 15  -- fair
            WHEN ra.recent_wape < 0.60                THEN 8   -- poor
            ELSE 0                                             -- very poor
        END AS score_forecast_accuracy

    FROM latest_inv l
    LEFT JOIN dim_sku d
           ON l.item_id = d.item_id AND l.loc = d.loc
    LEFT JOIN ss s
           ON l.item_id = s.item_id AND l.loc = s.loc
    LEFT JOIN recent_stockout rs
           ON l.item_id = rs.item_id AND l.loc = rs.loc
    LEFT JOIN recent_accuracy ra
           ON l.item_id = ra.item_id AND l.loc = ra.loc
)
SELECT
    item_id, loc, month_start,
    cluster_assignment, abc_vol, variability_class,
    seasonality_profile, region,
    eom_qty_on_hand, avg_daily_sls, current_dos,
    ss_combined, reorder_point, is_below_ss, ss_coverage,
    target_dos_min, target_dos_max,
    recent_wape, stockout_count_3m,
    score_ss_coverage, score_dos_target, score_stockout_risk, score_forecast_accuracy,
    -- Composite health score (0–100)
    (score_ss_coverage + score_dos_target + score_stockout_risk + score_forecast_accuracy)::INTEGER
        AS health_score,
    -- Health tier
    CASE
        WHEN (score_ss_coverage + score_dos_target + score_stockout_risk + score_forecast_accuracy) >= 80
             THEN 'healthy'
        WHEN (score_ss_coverage + score_dos_target + score_stockout_risk + score_forecast_accuracy) >= 60
             THEN 'monitor'
        WHEN (score_ss_coverage + score_dos_target + score_stockout_risk + score_forecast_accuracy) >= 40
             THEN 'at_risk'
        ELSE 'critical'
    END AS health_tier
FROM scored
WITH NO DATA;

-- -------------------------------------------------------------------------
-- Indexes
-- -------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS idx_health_score_pk
    ON mv_inventory_health_score (item_id, loc);

CREATE INDEX IF NOT EXISTS idx_health_score_tier
    ON mv_inventory_health_score (health_tier);

CREATE INDEX IF NOT EXISTS idx_health_score_critical
    ON mv_inventory_health_score (health_score)
    WHERE health_tier = 'critical';

CREATE INDEX IF NOT EXISTS idx_health_score_abc
    ON mv_inventory_health_score (abc_vol, health_tier);
