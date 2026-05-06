"""Customer lifecycle and demand-at-risk endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import (
    _CA_CACHE,
    _build_where,
    _build_where_mv,
    _customer_activity_source,
)

router = APIRouter(tags=["customer-analytics"])


# ---------------------------------------------------------------------------
# 10. Customer Lifecycle
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/lifecycle")
@_CA_CACHE
def customer_analytics_lifecycle(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Cohort retention + waterfall (new vs churned)."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    # Use the pre-aggregated MV when no item filter is requested (the common
    # case) — it avoids the fact x dim_customer JOIN + DISTINCT for every
    # request. With an item filter we have to hit the raw fact table since
    # the MV is item-aggregated.
    source_from, uses_mv = _customer_activity_source(item_id)
    if uses_mv:
        where = _build_where_mv(params, date_from, date_to, None, None)
    else:
        where = _build_where(params, item_id, date_from, date_to, None, None)

    # --- cohort retention ---
    # Cap the result set: the UI only renders ~24 cohorts x ~24 months_since
    # cells. Without a cap, datasets with many cohort months produce huge
    # payloads that dominate both DB time and JSON serialization.
    cohort_sql = f"""
        WITH base AS (
            SELECT DISTINCT f.customer_no, f.startdate
            FROM {source_from}
            WHERE {where}
        ),
        first_order AS (
            SELECT customer_no, MIN(startdate) AS cohort_month
            FROM base
            GROUP BY customer_no
        ),
        cohort_activity AS (
            SELECT fo.cohort_month,
                   EXTRACT(YEAR FROM age(b.startdate, fo.cohort_month)) * 12
                     + EXTRACT(MONTH FROM age(b.startdate, fo.cohort_month)) AS months_since,
                   COUNT(DISTINCT b.customer_no) AS active_customers
            FROM base b
            JOIN first_order fo ON fo.customer_no = b.customer_no
            GROUP BY fo.cohort_month, months_since
        ),
        cohort_size AS (
            SELECT cohort_month, COUNT(*) AS size
            FROM first_order
            GROUP BY cohort_month
        )
        SELECT ca.cohort_month, ca.months_since::int, ca.active_customers, cs.size
        FROM cohort_activity ca
        JOIN cohort_size cs ON cs.cohort_month = ca.cohort_month
        WHERE ca.months_since <= 24
        ORDER BY ca.cohort_month, ca.months_since
        LIMIT 1000
    """

    # --- waterfall (new / churned) ---
    params2: list[Any] = []
    if uses_mv:
        where2 = _build_where_mv(params2, date_from, date_to, None, None)
    else:
        where2 = _build_where(params2, item_id, date_from, date_to, None, None)
    # Window-function rewrite of the churn waterfall. The previous version
    # had two `months × base` range joins (`b.startdate >= m.month - 6mo`)
    # which materialize an N×M intermediate set. Here we instead annotate each
    # base row with `last_order_per_customer` via a window function, then
    # detect churn at the customer level (last activity 3-6 mo before MAX),
    # and finally aggregate per month. One pass over `base`, no Cartesian.
    waterfall_sql = f"""
        WITH base AS (
            SELECT DISTINCT f.customer_no, f.startdate
            FROM {source_from}
            WHERE {where2}
        ),
        months AS (
            SELECT DISTINCT startdate AS month FROM base
        ),
        first_order AS (
            SELECT customer_no, MIN(startdate) AS first_month, MAX(startdate) AS last_month
            FROM base GROUP BY customer_no
        ),
        new_per_month AS (
            SELECT first_month AS month, COUNT(*) AS new_customers
            FROM first_order GROUP BY first_month
        ),
        -- For each month m, churn = customers whose last activity falls in
        -- [m - 6mo, m - 3mo) — i.e. they were active in the older window
        -- but NOT in the recent window. Computed without re-joining base.
        churned AS (
            SELECT m.month,
                   COUNT(*) AS churned_customers
            FROM months m
            JOIN first_order fo
              ON fo.last_month >= m.month - INTERVAL '6 months'
             AND fo.last_month <  m.month - INTERVAL '3 months'
            GROUP BY m.month
        )
        SELECT m.month,
               COALESCE(n.new_customers, 0) AS new_customers,
               COALESCE(ch.churned_customers, 0) AS churned_customers
        FROM months m
        LEFT JOIN new_per_month n ON n.month = m.month
        LEFT JOIN churned ch ON ch.month = m.month
        ORDER BY m.month
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(cohort_sql, params)
        cohort_rows = cur.fetchall()
        cur.execute(waterfall_sql, params2)
        waterfall_rows = cur.fetchall()

    # Build cohort response
    cohort_map: dict[str, dict[str, Any]] = {}
    for cm, ms, active, size in cohort_rows:
        key = cm.isoformat() if hasattr(cm, "isoformat") else str(cm)
        if key not in cohort_map:
            cohort_map[key] = {"cohort_month": key, "months_since": [], "retention_pct": []}
        cohort_map[key]["months_since"].append(int(ms))
        pct = round(int(active) / int(size) * 100, 1) if int(size) > 0 else 0.0
        cohort_map[key]["retention_pct"].append(pct)

    cohorts = sorted(cohort_map.values(), key=lambda x: x["cohort_month"])

    waterfall = []
    for month, new_c, churned_c in waterfall_rows:
        m_str = month.isoformat() if hasattr(month, "isoformat") else str(month)
        n = int(new_c)
        ch = int(churned_c)
        waterfall.append({
            "month": m_str,
            "new_customers": n,
            "churned_customers": ch,
            "net_change": n - ch,
        })

    return {"cohorts": cohorts, "waterfall": waterfall}


# ---------------------------------------------------------------------------
# 11. Demand at Risk
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/demand-at-risk")
@_CA_CACHE
def customer_analytics_demand_at_risk(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Waterfall breakdown of demand risk categories."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, None, None)

    sql = f"""
        WITH base AS (
            SELECT f.customer_no,
                   f.item_id,
                   f.location_id,
                   f.startdate,
                   f.demand_qty,
                   f.sales_qty,
                   f.oos_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where}
        ),
        totals AS (
            SELECT COALESCE(SUM(demand_qty), 0) AS total_demand
            FROM base
        ),
        -- concentration risk: customers whose share > 40%
        cust_share AS (
            SELECT customer_no, SUM(demand_qty) AS cust_demand
            FROM base GROUP BY customer_no
        ),
        conc_risk AS (
            SELECT COALESCE(SUM(cs.cust_demand), 0) AS concentration_risk
            FROM cust_share cs, totals t
            WHERE t.total_demand > 0
              AND cs.cust_demand / t.total_demand > 0.4
        ),
        -- oos loss: demand * oos_rate at item-loc level
        oos_agg AS (
            SELECT item_id, location_id,
                   SUM(demand_qty) AS d, SUM(oos_qty) AS o
            FROM base GROUP BY item_id, location_id
            HAVING SUM(demand_qty) > 0
        ),
        oos_loss AS (
            SELECT COALESCE(SUM(d * (o / NULLIF(d, 0))), 0) AS oos_loss
            FROM oos_agg
            WHERE o > 0
        ),
        -- churn risk: customers in [-6,-3] but not [-3,0]
        bounds AS (
            SELECT MAX(startdate) AS max_dt FROM base
        ),
        recent AS (
            SELECT DISTINCT customer_no FROM base, bounds
            WHERE startdate > bounds.max_dt - INTERVAL '3 months'
        ),
        older AS (
            SELECT DISTINCT customer_no FROM base, bounds
            WHERE startdate <= bounds.max_dt - INTERVAL '3 months'
              AND startdate > bounds.max_dt - INTERVAL '6 months'
        ),
        churned_custs AS (
            SELECT o.customer_no
            FROM older o
            LEFT JOIN recent r ON r.customer_no = o.customer_no
            WHERE r.customer_no IS NULL
        ),
        churn_risk AS (
            SELECT COALESCE(SUM(b.demand_qty), 0) AS churn_risk
            FROM base b
            JOIN churned_custs cc ON cc.customer_no = b.customer_no
        )
        SELECT t.total_demand,
               cr.concentration_risk,
               ol.oos_loss,
               chr.churn_risk
        FROM totals t, conc_risk cr, oos_loss ol, churn_risk chr
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    if not row:
        return {"waterfall": []}

    total = float(row[0] or 0)
    conc = float(row[1] or 0)
    oos = float(row[2] or 0)
    churn = float(row[3] or 0)
    secure = max(total - conc - oos - churn, 0)

    waterfall = [
        {"category": "total_demand", "value": round(total, 1)},
        {"category": "concentration_risk", "value": round(conc, 1)},
        {"category": "oos_loss", "value": round(oos, 1)},
        {"category": "churn_risk", "value": round(churn, 1)},
        {"category": "secure_demand", "value": round(secure, 1)},
    ]
    return {"waterfall": waterfall}
