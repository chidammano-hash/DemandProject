-- Gen-4 Roadmap SC-5: Line-fill and case-fill alongside unit-fill.
--
-- Drops and recreates mv_fill_rate_monthly with three fill-rate denominators:
--   - unit_fill_rate  = sum(shipped_qty) / sum(ordered_qty)              (existing `fill_rate`)
--   - line_fill_rate  = lines_filled / lines_ordered                      (NEW)
--   - case_fill_rate  = cases_shipped / cases_ordered                     (NULL fallback if case data absent)
--
-- Line count is derived from distinct (item_id, order_key) at the fact grain.
-- Case qty: fact_sales_monthly does not carry case units, so case_fill_rate
-- is emitted as NULL. When per-SKU case_pack_size is introduced, this MV can
-- be amended to derive cases = qty / case_pack_size.

DROP MATERIALIZED VIEW IF EXISTS mv_fill_rate_monthly CASCADE;

CREATE MATERIALIZED VIEW mv_fill_rate_monthly AS
SELECT
    s.item_id                                          AS item_id,
    s.loc,
    s.startdate                                        AS month_start,
    SUM(s.qty_ordered)                                 AS total_ordered,
    SUM(s.qty_shipped)                                 AS total_shipped,
    CASE WHEN SUM(s.qty_ordered) > 0
         THEN SUM(s.qty_shipped) / SUM(s.qty_ordered)
         ELSE NULL END                                 AS fill_rate,           -- legacy alias == unit_fill_rate
    CASE WHEN SUM(s.qty_ordered) > 0
         THEN SUM(s.qty_shipped) / SUM(s.qty_ordered)
         ELSE NULL END                                 AS unit_fill_rate,       -- NEW: explicit unit-fill column
    -- Line-fill: count distinct item_id rows that were fully filled vs rows with any order
    CASE WHEN COUNT(*) FILTER (WHERE s.qty_ordered > 0) > 0
         THEN COUNT(*) FILTER (WHERE s.qty_ordered > 0 AND s.qty_shipped >= s.qty_ordered)::numeric
              / COUNT(*) FILTER (WHERE s.qty_ordered > 0)
         ELSE NULL END                                 AS line_fill_rate,       -- NEW: line-level fill
    COUNT(*) FILTER (WHERE s.qty_ordered > 0)          AS lines_ordered,        -- NEW
    COUNT(*) FILTER (WHERE s.qty_ordered > 0
                         AND s.qty_shipped >= s.qty_ordered) AS lines_filled,   -- NEW
    -- Case-fill: case-level qty not in source fact table. Emit NULL until case_pack_size is added.
    NULL::numeric                                      AS case_fill_rate,       -- NEW: NULL until case data available
    NULL::numeric                                      AS cases_ordered,
    NULL::numeric                                      AS cases_shipped,
    GREATEST(SUM(s.qty_ordered) - SUM(s.qty_shipped), 0) AS shortage_qty,
    (SUM(s.qty_shipped) < SUM(s.qty_ordered))          AS had_partial_fulfillment,
    COALESCE(d.abc_vol, '(unknown)')                   AS abc_vol,
    COALESCE(d.cluster_assignment, '(unassigned)')     AS cluster_assignment,
    COALESCE(d.region, '(unknown)')                    AS region,
    d.is_yearly_seasonal,
    d.seasonality_profile,
    d.variability_class
FROM fact_sales_monthly s
LEFT JOIN dim_sku d
    ON s.item_id = d.item_id
    AND s.customer_group = d.customer_group
    AND s.loc = d.loc
WHERE s.type = 1
  AND s.qty_ordered IS NOT NULL
  AND s.qty_ordered > 0
GROUP BY
    s.item_id, s.loc, s.startdate,
    d.abc_vol, d.cluster_assignment, d.region,
    d.is_yearly_seasonal, d.seasonality_profile, d.variability_class
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_fill_rate_pk
    ON mv_fill_rate_monthly (item_id, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_month
    ON mv_fill_rate_monthly (month_start);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_abc
    ON mv_fill_rate_monthly (abc_vol);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_partial
    ON mv_fill_rate_monthly (had_partial_fulfillment)
    WHERE had_partial_fulfillment = TRUE;
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_low
    ON mv_fill_rate_monthly (fill_rate)
    WHERE fill_rate < 0.95;
