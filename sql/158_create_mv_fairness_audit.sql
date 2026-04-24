-- 158_create_mv_fairness_audit.sql
-- Gen-4 Stream G — Governance / AI-9: Fairness audit across slices.
--
-- Computes per-slice forecast WAPE and disparity ratio (slice_wape / overall_wape)
-- so governance can flag any customer-channel / region / abc-class combination
-- where the champion systematically under- or over-forecasts.
--
-- Grain: one row per (plan_version, slice_dim, slice_value).
--
-- NOTE: fact_production_forecast is DFU-grained (item+loc, no customer). The
-- customer-channel slice derives channel from the majority customer_group
-- on the matching SKU row in dim_sku -> dim_customer. If the fact layer
-- gains a customer grain, rewrite this MV to include dim_customer.tier
-- directly.

DROP MATERIALIZED VIEW IF EXISTS mv_fairness_audit CASCADE;

CREATE MATERIALIZED VIEW mv_fairness_audit AS
WITH forecast_actual AS (
    -- Pair each promoted forecast row with the realized sales for that month.
    SELECT
        pf.plan_version,
        pf.item_id,
        pf.loc,
        pf.forecast_month,
        pf.forecast_qty,
        COALESCE(sales.qty_shipped, 0.0) AS actual_qty
    FROM fact_production_forecast pf
    LEFT JOIN agg_sales_monthly sales
           ON sales.item_id = pf.item_id
          AND sales.loc = pf.loc
          AND sales.month_start = pf.forecast_month
),
overall AS (
    SELECT
        plan_version,
        CASE WHEN SUM(ABS(actual_qty)) > 0
             THEN SUM(ABS(forecast_qty - actual_qty)) / SUM(ABS(actual_qty))
             ELSE NULL END AS overall_wape,
        COUNT(*) AS overall_n
    FROM forecast_actual
    GROUP BY plan_version
),
enriched AS (
    SELECT
        fa.plan_version,
        fa.item_id,
        fa.loc,
        fa.forecast_qty,
        fa.actual_qty,
        sku.abc_vol,
        sku.region,
        loc_d.state_id,
        -- rpt_channel_desc used as tier proxy (dim_customer lacks a tier column)
        sku.supergroup AS channel
    FROM forecast_actual fa
    LEFT JOIN dim_sku sku
           ON sku.item_id = fa.item_id
          AND sku.loc = fa.loc
    LEFT JOIN dim_location loc_d
           ON loc_d.location_id = fa.loc
),
slice_abc AS (
    SELECT
        plan_version,
        'abc_vol'::text AS slice_dim,
        COALESCE(abc_vol, 'unclassified') AS slice_value,
        CASE WHEN SUM(ABS(actual_qty)) > 0
             THEN SUM(ABS(forecast_qty - actual_qty)) / SUM(ABS(actual_qty))
             ELSE NULL END AS slice_wape,
        COUNT(*) AS slice_n
    FROM enriched
    GROUP BY plan_version, COALESCE(abc_vol, 'unclassified')
),
slice_region AS (
    SELECT
        plan_version,
        'region'::text AS slice_dim,
        COALESCE(region, 'unknown') AS slice_value,
        CASE WHEN SUM(ABS(actual_qty)) > 0
             THEN SUM(ABS(forecast_qty - actual_qty)) / SUM(ABS(actual_qty))
             ELSE NULL END AS slice_wape,
        COUNT(*) AS slice_n
    FROM enriched
    GROUP BY plan_version, COALESCE(region, 'unknown')
),
slice_channel AS (
    SELECT
        plan_version,
        'channel'::text AS slice_dim,
        COALESCE(channel, 'unknown') AS slice_value,
        CASE WHEN SUM(ABS(actual_qty)) > 0
             THEN SUM(ABS(forecast_qty - actual_qty)) / SUM(ABS(actual_qty))
             ELSE NULL END AS slice_wape,
        COUNT(*) AS slice_n
    FROM enriched
    GROUP BY plan_version, COALESCE(channel, 'unknown')
),
all_slices AS (
    SELECT * FROM slice_abc
    UNION ALL
    SELECT * FROM slice_region
    UNION ALL
    SELECT * FROM slice_channel
)
SELECT
    s.plan_version,
    s.slice_dim,
    s.slice_value,
    s.slice_wape,
    o.overall_wape,
    s.slice_n,
    o.overall_n,
    CASE WHEN o.overall_wape IS NULL OR o.overall_wape = 0
         THEN NULL
         ELSE s.slice_wape / o.overall_wape END AS disparity_ratio
FROM all_slices s
LEFT JOIN overall o ON o.plan_version = s.plan_version
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uq_mv_fairness_audit_pk
    ON mv_fairness_audit (plan_version, slice_dim, slice_value);

CREATE INDEX IF NOT EXISTS idx_mv_fairness_audit_disparity
    ON mv_fairness_audit (disparity_ratio DESC);
