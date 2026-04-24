-- Gen-4 Roadmap SC-10 (P2): Weekly granularity for rolling 13-week view.
--
-- Materialized view of sales aggregated to ISO-week grain. Feeds the
-- GET /analytics/rolling-13-week endpoint and any week-level planning KPI.

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_sales_weekly AS
SELECT
    s.item_id,
    s.loc,
    DATE_TRUNC('week', s.startdate)::date      AS week_start,  -- ISO week start (Mon)
    EXTRACT(ISOYEAR FROM s.startdate)::int     AS iso_year,
    EXTRACT(WEEK    FROM s.startdate)::int     AS iso_week,
    SUM(s.qty_ordered)                          AS qty_ordered,
    SUM(s.qty_shipped)                          AS qty_shipped,
    SUM(COALESCE(s.qty_shipped, 0))             AS qty_sold,    -- alias for convenience
    COUNT(*)                                    AS row_count
FROM fact_sales_monthly s
WHERE s.type = 1
GROUP BY
    s.item_id, s.loc,
    DATE_TRUNC('week', s.startdate)::date,
    EXTRACT(ISOYEAR FROM s.startdate),
    EXTRACT(WEEK    FROM s.startdate)
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agg_sales_weekly_pk
    ON agg_sales_weekly (item_id, loc, week_start);
CREATE INDEX IF NOT EXISTS idx_agg_sales_weekly_week
    ON agg_sales_weekly (week_start DESC);
CREATE INDEX IF NOT EXISTS idx_agg_sales_weekly_iso
    ON agg_sales_weekly (iso_year, iso_week);

COMMENT ON MATERIALIZED VIEW agg_sales_weekly IS
    'Gen-4 SC-10: weekly sales rollup; feeds rolling-13-week endpoint.';
