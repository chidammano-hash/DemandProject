-- mv_ca_order_patterns
--
-- Pre-computed per-customer ordering cadence stats for /customer-analytics/order-patterns.
-- The endpoint computes LAG()-based interval gaps then AVG/STDDEV per customer
-- over fact_customer_demand_monthly. At 40x scale this window function pass
-- over millions of rows is a 10s+ query.
--
-- Strategy: pre-compute mean/CV of inter-order interval per customer for the
-- canonical 12-month window so the endpoint is a tiny LIMIT 200 scan + sort.
-- Endpoints that pass a custom date range fall back to the fact-table path.
--
-- Grain: (customer_no, site)
--   - customer_name: cached from dim_customer
--   - avg_interval_months: AVG(gap_months) where gap = months between
--     consecutive activity months for the customer
--   - interval_cv: STDDEV / AVG (coefficient of variation; bounded [0, ∞))
--   - order_count: number of activity months observed
--   - total_demand: total demand_qty over the window (used for sort + display)
--
-- Cardinality on prod: ~33K customers. Trivial.
--
-- Refresh cadence: nightly. CONCURRENTLY safe.

DROP MATERIALIZED VIEW IF EXISTS mv_ca_order_patterns CASCADE;

CREATE MATERIALIZED VIEW mv_ca_order_patterns AS
WITH bounds AS (
    SELECT MAX(startdate) AS max_dt,
           (MAX(startdate) - INTERVAL '12 months')::date AS min_dt
    FROM fact_customer_demand_monthly
),
base AS (
    SELECT
        f.customer_no,
        f.site,
        c.customer_name,
        f.startdate,
        SUM(f.demand_qty) AS demand_qty
    FROM fact_customer_demand_monthly f, bounds b
    JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
    WHERE f.startdate >= b.min_dt
      AND f.startdate <  b.max_dt + INTERVAL '1 month'
    GROUP BY f.customer_no, f.site, c.customer_name, f.startdate
),
intervals AS (
    SELECT
        customer_no,
        site,
        startdate,
        EXTRACT(YEAR FROM age(startdate, LAG(startdate)
                              OVER (PARTITION BY customer_no, site ORDER BY startdate))) * 12
          + EXTRACT(MONTH FROM age(startdate, LAG(startdate)
                                   OVER (PARTITION BY customer_no, site ORDER BY startdate))) AS gap_months
    FROM base
),
cust_stats AS (
    SELECT customer_no, site,
           AVG(gap_months) AS avg_interval,
           CASE WHEN AVG(gap_months) > 0
                THEN STDDEV(gap_months) / AVG(gap_months)
                ELSE 0 END AS interval_cv,
           COUNT(*) AS order_count
    FROM intervals
    WHERE gap_months IS NOT NULL
    GROUP BY customer_no, site
),
cust_total AS (
    SELECT customer_no, site,
           MAX(customer_name) AS customer_name,
           SUM(demand_qty)    AS total_demand
    FROM base
    GROUP BY customer_no, site
)
SELECT
    ct.customer_no,
    ct.site,
    ct.customer_name,
    cs.avg_interval AS avg_interval_months,
    COALESCE(cs.interval_cv, 0)   AS interval_cv,
    COALESCE(cs.order_count, 0)   AS order_count,
    ct.total_demand
FROM cust_total ct
LEFT JOIN cust_stats cs
    ON cs.customer_no = ct.customer_no AND cs.site = ct.site;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_ca_order_patterns_pk
    ON mv_ca_order_patterns (customer_no, site);

-- Endpoint sorts by total_demand DESC LIMIT 200
CREATE INDEX IF NOT EXISTS idx_mv_ca_order_patterns_demand
    ON mv_ca_order_patterns (total_demand DESC);

ANALYZE mv_ca_order_patterns;
