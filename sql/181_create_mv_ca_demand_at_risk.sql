-- mv_ca_demand_at_risk
--
-- Pre-computed per-customer risk scores for the /customer-analytics/demand-at-risk
-- waterfall. The endpoint produces a 5-bucket waterfall (total / concentration /
-- oos_loss / churn / secure) over fact_customer_demand_monthly with multiple
-- subqueries (HHI-style concentration cutoff, item-loc OOS rate, 3/6 month
-- churn cohort). At 40x scale this is the slowest single query in the tab.
--
-- Strategy: pre-compute per-customer flags for the canonical 12-month window
-- (planning_date - 12 months .. planning_date) so the endpoint becomes a
-- 4-row SELECT. Endpoints that pass a custom date range fall back to the
-- fact-table path — see _fast_path_eligible() in lifecycle.py.
--
-- Grain: (customer_no, site)
--   - demand_qty: total customer demand in the window
--   - is_concentration_risk: customer's share of total > 40% in any item-loc
--     pair (rare but high-impact)
--   - oos_loss_qty: portion of customer demand attributable to item-loc OOS
--     rate (sum-product approximation matches the endpoint formula)
--   - is_churn_risk: customer last active in [-6mo, -3mo) but not in [-3mo, 0]
--
-- Cardinality on prod: ~33K customers. Trivial.
--
-- Refresh cadence: nightly. The window slides forward each day so the MV is
-- fresh modulo planning_date drift; refresh BEFORE running the dashboard
-- nightly so the waterfall reflects the latest month.

DROP MATERIALIZED VIEW IF EXISTS mv_ca_demand_at_risk CASCADE;

CREATE MATERIALIZED VIEW mv_ca_demand_at_risk AS
WITH bounds AS (
    -- Last 12 full months relative to MAX(startdate) in fact (proxy for
    -- planning date — avoids referencing planning_date config in DDL).
    SELECT MAX(startdate) AS max_dt,
           (MAX(startdate) - INTERVAL '12 months')::date AS min_dt
    FROM fact_customer_demand_monthly
),
window_rows AS (
    SELECT
        f.customer_no,
        f.site,
        f.item_id,
        f.location_id,
        f.startdate,
        f.demand_qty,
        f.sales_qty,
        f.oos_qty
    FROM fact_customer_demand_monthly f, bounds b
    WHERE f.startdate >= b.min_dt
      AND f.startdate <  b.max_dt + INTERVAL '1 month'
),
-- Per item-loc concentration: customers with > 40% share trigger the bucket.
item_loc_totals AS (
    SELECT item_id, location_id, SUM(demand_qty) AS il_total
    FROM window_rows
    GROUP BY item_id, location_id
    HAVING SUM(demand_qty) > 0
),
cust_il_share AS (
    -- Per (customer, item, location): customer's share of that item-loc demand.
    -- Cannot nest aggregates — compute the per-cell sum first, then derive the
    -- > 40% flag in the surrounding GROUP BY.
    SELECT w.customer_no, w.site, w.item_id, w.location_id,
           SUM(w.demand_qty) / NULLIF(MAX(t.il_total), 0) > 0.4 AS is_concentration_risk,
           SUM(w.demand_qty) AS demand_qty
    FROM window_rows w
    JOIN item_loc_totals t
      ON t.item_id = w.item_id AND t.location_id = w.location_id
    GROUP BY w.customer_no, w.site, w.item_id, w.location_id
),
cust_agg AS (
    SELECT customer_no, site,
           BOOL_OR(is_concentration_risk) AS is_concentration_risk,
           SUM(demand_qty) AS demand_qty
    FROM cust_il_share
    GROUP BY customer_no, site
),
-- OOS loss: per-customer sum of (demand * item_loc_oos_rate).
item_loc_oos AS (
    SELECT item_id, location_id,
           CASE WHEN SUM(demand_qty) > 0
                THEN SUM(oos_qty)::float / SUM(demand_qty)
                ELSE 0 END AS oos_rate
    FROM window_rows
    GROUP BY item_id, location_id
),
cust_oos AS (
    SELECT w.customer_no, w.site,
           SUM(w.demand_qty * COALESCE(o.oos_rate, 0)) AS oos_loss_qty
    FROM window_rows w
    LEFT JOIN item_loc_oos o
      ON o.item_id = w.item_id AND o.location_id = w.location_id
    GROUP BY w.customer_no, w.site
),
-- Churn: last activity in [-6mo, -3mo) AND no activity in [-3mo, 0].
cust_recency AS (
    SELECT w.customer_no, w.site,
           BOOL_OR(w.startdate >  b.max_dt - INTERVAL '3 months') AS active_recent,
           BOOL_OR(w.startdate <= b.max_dt - INTERVAL '3 months'
                   AND w.startdate > b.max_dt - INTERVAL '6 months') AS active_older
    FROM window_rows w, bounds b
    GROUP BY w.customer_no, w.site
)
SELECT
    a.customer_no,
    a.site,
    a.demand_qty,
    COALESCE(a.is_concentration_risk, FALSE) AS is_concentration_risk,
    COALESCE(o.oos_loss_qty, 0)              AS oos_loss_qty,
    (COALESCE(r.active_older, FALSE) AND NOT COALESCE(r.active_recent, FALSE))
                                              AS is_churn_risk
FROM cust_agg a
LEFT JOIN cust_oos     o ON o.customer_no = a.customer_no AND o.site = a.site
LEFT JOIN cust_recency r ON r.customer_no = a.customer_no AND r.site = a.site;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_ca_dar_pk
    ON mv_ca_demand_at_risk (customer_no, site);

CREATE INDEX IF NOT EXISTS idx_mv_ca_dar_concentration
    ON mv_ca_demand_at_risk (is_concentration_risk) WHERE is_concentration_risk;

CREATE INDEX IF NOT EXISTS idx_mv_ca_dar_churn
    ON mv_ca_demand_at_risk (is_churn_risk) WHERE is_churn_risk;

ANALYZE mv_ca_demand_at_risk;
