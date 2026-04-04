"""Generate customer-derived features using SQL aggregation (fast path).

Replaces the Python-loop approach with a single SQL INSERT...SELECT that
runs entirely in Postgres. Handles 600K+ item×loc groups in minutes.

Usage:
    python -m scripts.ml.generate_customer_features_sql [--months 36]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)


SQL_POPULATE = """
-- Truncate and repopulate in one transaction
TRUNCATE TABLE customer_features_monthly;

INSERT INTO customer_features_monthly (
    item_id, loc, startdate,
    n_active_cust, n_active_cust_6m, hhi_demand,
    top1_cust_share, top3_cust_share, cust_gini,
    new_cust_demand_share, churned_cust_demand_share,
    cust_count_mom, cust_retention_rate, cust_tenure_mean,
    true_demand_ratio, oos_rate, oos_cust_pct,
    demand_sales_gap_3m, oos_trend,
    demand_qty_lag1, demand_qty_lag3_mean,
    channel_entropy, dominant_channel_share,
    channel_mix_shift, on_premise_share,
    cust_demand_cv_mean, cust_demand_sync, max_cust_share_delta,
    store_type_entropy, dominant_store_type_share,
    chain_ratio, top_chain_share,
    sub_channel_entropy, active_cust_pct, avg_delivery_freq,
    on_premise_acct_share, premise_diversity
)
WITH base AS (
    -- Pre-aggregate to customer × item × loc × month grain with attributes
    SELECT
        f.item_id,
        f.location_id AS loc,
        f.startdate,
        f.customer_no,
        SUM(f.demand_qty) AS demand_qty,
        SUM(f.sales_qty) AS sales_qty,
        SUM(f.oos_qty) AS oos_qty,
        MAX(c.rpt_channel_desc) AS channel,
        MAX(c.store_type_desc) AS store_type,
        MAX(c.chain_type_desc) AS chain_type,
        MAX(c.rpt_sub_channel_desc) AS sub_channel,
        MAX(c.status) AS cust_status,
        MAX(c.premise_code) AS premise_code
    FROM fact_customer_demand_monthly f
    JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
    WHERE f.startdate >= (CURRENT_DATE - INTERVAL '{months} months')::date
    GROUP BY f.item_id, f.location_id, f.startdate, f.customer_no
),
-- Item × loc × month totals
monthly_agg AS (
    SELECT
        item_id, loc, startdate,
        COUNT(DISTINCT customer_no) AS n_cust,
        SUM(demand_qty) AS total_demand,
        SUM(sales_qty) AS total_sales,
        SUM(oos_qty) AS total_oos,
        COUNT(DISTINCT customer_no) FILTER (WHERE oos_qty > 0) AS oos_cust_count
    FROM base
    GROUP BY item_id, loc, startdate
),
-- Customer shares per item × loc × month
cust_shares AS (
    SELECT
        b.item_id, b.loc, b.startdate, b.customer_no,
        b.demand_qty,
        b.demand_qty / NULLIF(m.total_demand, 0) AS share,
        b.channel, b.store_type, b.chain_type, b.sub_channel,
        b.cust_status, b.premise_code
    FROM base b
    JOIN monthly_agg m USING (item_id, loc, startdate)
),
-- HHI and top-N shares
concentration AS (
    SELECT
        item_id, loc, startdate,
        SUM(share * share) AS hhi,
        MAX(share) AS top1_share,
        -- Gini approximation: 2*sum(rank*share)/(n*sum(share)) - (n+1)/n
        CASE WHEN COUNT(*) > 1 THEN
            GREATEST(0, (2.0 * SUM(rk * share) / (COUNT(*) * NULLIF(SUM(share), 0))) - (COUNT(*) + 1.0) / COUNT(*))
        ELSE 0 END AS gini
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY item_id, loc, startdate ORDER BY share ASC) AS rk
        FROM cust_shares
    ) ranked
    GROUP BY item_id, loc, startdate
),
top3 AS (
    SELECT item_id, loc, startdate, SUM(share) AS top3_share
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY item_id, loc, startdate ORDER BY demand_qty DESC) AS rn
        FROM cust_shares
    ) t WHERE rn <= 3
    GROUP BY item_id, loc, startdate
),
-- Channel entropy and dominant share
channel_agg AS (
    SELECT
        item_id, loc, startdate,
        -SUM(CASE WHEN ch_share > 0 THEN ch_share * LN(ch_share) ELSE 0 END) AS ch_entropy,
        MAX(ch_share) AS dom_ch_share,
        SUM(CASE WHEN channel IN ('On Premise', 'ON PREMISE', 'On-Premise') THEN ch_demand ELSE 0 END)
            / NULLIF(SUM(ch_demand), 0) AS on_prem_share
    FROM (
        SELECT item_id, loc, startdate, channel,
               SUM(demand_qty) AS ch_demand,
               SUM(demand_qty) / NULLIF(SUM(SUM(demand_qty)) OVER (PARTITION BY item_id, loc, startdate), 0) AS ch_share
        FROM base GROUP BY item_id, loc, startdate, channel
    ) ch
    GROUP BY item_id, loc, startdate
),
-- Store type entropy
store_type_agg AS (
    SELECT
        item_id, loc, startdate,
        -SUM(CASE WHEN st_share > 0 THEN st_share * LN(st_share) ELSE 0 END) AS st_entropy,
        MAX(st_share) AS dom_st_share
    FROM (
        SELECT item_id, loc, startdate, store_type,
               SUM(demand_qty) / NULLIF(SUM(SUM(demand_qty)) OVER (PARTITION BY item_id, loc, startdate), 0) AS st_share
        FROM base GROUP BY item_id, loc, startdate, store_type
    ) st
    GROUP BY item_id, loc, startdate
),
-- Sub-channel entropy
sub_ch_agg AS (
    SELECT
        item_id, loc, startdate,
        -SUM(CASE WHEN sc_share > 0 THEN sc_share * LN(sc_share) ELSE 0 END) AS sc_entropy
    FROM (
        SELECT item_id, loc, startdate, sub_channel,
               SUM(demand_qty) / NULLIF(SUM(SUM(demand_qty)) OVER (PARTITION BY item_id, loc, startdate), 0) AS sc_share
        FROM base GROUP BY item_id, loc, startdate, sub_channel
    ) sc
    GROUP BY item_id, loc, startdate
),
-- Chain ratio and top chain
chain_agg AS (
    SELECT
        item_id, loc, startdate,
        1.0 - SUM(CASE WHEN LOWER(chain_type) LIKE '%%indep%%' THEN ct_demand ELSE 0 END) / NULLIF(SUM(ct_demand), 0) AS chain_ratio,
        MAX(ct_demand) / NULLIF(SUM(ct_demand), 0) AS top_chain_share
    FROM (
        SELECT item_id, loc, startdate, chain_type, SUM(demand_qty) AS ct_demand
        FROM base GROUP BY item_id, loc, startdate, chain_type
    ) ct
    GROUP BY item_id, loc, startdate
),
-- Premise features
premise_agg AS (
    SELECT
        item_id, loc, startdate,
        SUM(CASE WHEN UPPER(premise_code) IN ('ON', 'O', 'ON-PREMISE', 'ON PREMISE') THEN demand_qty ELSE 0 END)
            / NULLIF(SUM(demand_qty), 0) AS on_prem_acct_share,
        COUNT(DISTINCT premise_code)::real / NULLIF(COUNT(DISTINCT customer_no), 0) AS premise_div
    FROM base
    GROUP BY item_id, loc, startdate
),
-- Active status pct
status_agg AS (
    SELECT
        item_id, loc, startdate,
        COUNT(DISTINCT customer_no) FILTER (WHERE LOWER(cust_status) LIKE '%%act%%' OR cust_status = 'A')::real
            / NULLIF(COUNT(DISTINCT customer_no), 0) AS active_pct
    FROM base
    GROUP BY item_id, loc, startdate
),
-- Prior month metrics for MoM and demand lags
prior_month AS (
    SELECT
        m.item_id, m.loc, m.startdate,
        -- Lag values from prior months
        LAG(m.total_demand, 1) OVER w AS demand_lag1,
        (LAG(m.total_demand, 1) OVER w + LAG(m.total_demand, 2) OVER w + LAG(m.total_demand, 3) OVER w) / 3.0 AS demand_lag3_mean,
        LAG(m.n_cust, 1) OVER w AS prev_n_cust,
        LAG(m.total_oos / NULLIF(m.total_demand, 0), 1) OVER w AS prev_oos_rate
    FROM monthly_agg m
    WINDOW w AS (PARTITION BY m.item_id, m.loc ORDER BY m.startdate)
)
SELECT
    m.item_id,
    m.loc,
    m.startdate,
    -- Concentration
    m.n_cust AS n_active_cust,
    m.n_cust AS n_active_cust_6m,  -- simplified: same as current month
    COALESCE(con.hhi, 0) AS hhi_demand,
    COALESCE(con.top1_share, 0) AS top1_cust_share,
    COALESCE(t3.top3_share, 0) AS top3_cust_share,
    COALESCE(con.gini, 0) AS cust_gini,
    -- Dynamics (simplified for SQL path)
    0 AS new_cust_demand_share,
    0 AS churned_cust_demand_share,
    CASE WHEN COALESCE(pm.prev_n_cust, 0) > 0
         THEN (m.n_cust - pm.prev_n_cust)::real / pm.prev_n_cust
         ELSE 0 END AS cust_count_mom,
    0 AS cust_retention_rate,
    0 AS cust_tenure_mean,
    -- True demand
    m.total_demand / NULLIF(m.total_sales, 0) AS true_demand_ratio,
    m.total_oos / NULLIF(m.total_demand, 0) AS oos_rate,
    m.oos_cust_count::real / NULLIF(m.n_cust, 0) AS oos_cust_pct,
    m.total_demand - m.total_sales AS demand_sales_gap_3m,
    CASE WHEN COALESCE(pm.prev_oos_rate, 0) > 0
         THEN ((m.total_oos / NULLIF(m.total_demand, 0)) - pm.prev_oos_rate) / pm.prev_oos_rate
         ELSE 0 END AS oos_trend,
    COALESCE(pm.demand_lag1, 0) AS demand_qty_lag1,
    COALESCE(pm.demand_lag3_mean, 0) AS demand_qty_lag3_mean,
    -- Channel mix
    COALESCE(ca.ch_entropy, 0) AS channel_entropy,
    COALESCE(ca.dom_ch_share, 0) AS dominant_channel_share,
    0 AS channel_mix_shift,  -- requires two periods comparison, omitted for speed
    COALESCE(ca.on_prem_share, 0) AS on_premise_share,
    -- Cross-customer (simplified)
    0 AS cust_demand_cv_mean,
    0 AS cust_demand_sync,
    0 AS max_cust_share_delta,
    -- Attribute mix
    COALESCE(sta.st_entropy, 0) AS store_type_entropy,
    COALESCE(sta.dom_st_share, 0) AS dominant_store_type_share,
    COALESCE(cha.chain_ratio, 0) AS chain_ratio,
    COALESCE(cha.top_chain_share, 0) AS top_chain_share,
    COALESCE(sca.sc_entropy, 0) AS sub_channel_entropy,
    COALESCE(sa.active_pct, 1) AS active_cust_pct,
    0 AS avg_delivery_freq,
    COALESCE(pa.on_prem_acct_share, 0) AS on_premise_acct_share,
    COALESCE(pa.premise_div, 0) AS premise_diversity
FROM monthly_agg m
LEFT JOIN concentration con USING (item_id, loc, startdate)
LEFT JOIN top3 t3 USING (item_id, loc, startdate)
LEFT JOIN channel_agg ca USING (item_id, loc, startdate)
LEFT JOIN store_type_agg sta USING (item_id, loc, startdate)
LEFT JOIN sub_ch_agg sca USING (item_id, loc, startdate)
LEFT JOIN chain_agg cha USING (item_id, loc, startdate)
LEFT JOIN premise_agg pa USING (item_id, loc, startdate)
LEFT JOIN status_agg sa USING (item_id, loc, startdate)
LEFT JOIN prior_month pm USING (item_id, loc, startdate)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate customer features (SQL fast path)")
    parser.add_argument("--months", type=int, default=36, help="Lookback months (default 36)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    import psycopg
    db = get_db_params()

    logger.info("=" * 60)
    logger.info("Customer Feature Generation (SQL) — Start")
    logger.info("=" * 60)

    sql = SQL_POPULATE.replace("{months}", str(args.months))

    logger.info("Running SQL aggregation (lookback=%d months)...", args.months)
    t0 = time.time()

    with profiled_section("sql_populate"):
        with psycopg.connect(**db) as conn:
            conn.execute("SET work_mem = '512MB'")
            conn.execute("SET statement_timeout = '3600000'")  # 1 hour max
            with conn.cursor() as cur:
                cur.execute(sql)
                n_rows = cur.rowcount
            conn.commit()

    elapsed = time.time() - t0
    logger.info("Inserted %s rows in %.1fs (%.0f rows/sec)",
                f"{n_rows:,}", elapsed, n_rows / max(elapsed, 0.01))

    # Verify
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT item_id || loc) FROM customer_features_monthly")
        total, n_il = cur.fetchone()
        cur.execute("SELECT MIN(startdate), MAX(startdate) FROM customer_features_monthly")
        d_min, d_max = cur.fetchone()

    logger.info("Verification: %s rows, %s item×loc pairs, %s → %s",
                f"{total:,}", f"{n_il:,}", d_min, d_max)
    logger.info("=" * 60)
    logger.info("Customer Feature Generation (SQL) — Done (%.1fs)", time.time() - t0)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
