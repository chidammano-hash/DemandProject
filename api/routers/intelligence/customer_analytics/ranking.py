"""Customer-level analytics — ranking, OOS impact, affinity, order patterns."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _CA_CACHE, _build_where

router = APIRouter(tags=["customer-analytics"])


# ---------------------------------------------------------------------------
# 6. Customer Ranking
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/ranking")
@_CA_CACHE
def customer_analytics_ranking(
    response: FastAPIResponse,
    sort: str = Query(default="demand_desc", pattern="^(demand_desc|fill_rate_asc)$"),
    top_n: int = Query(default=20, ge=5, le=50),
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
    min_demand: float = Query(default=0, ge=0),
):
    """Top/bottom customer ranking by demand or fill rate."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, channel, store_type)

    order = "SUM(f.demand_qty) DESC" if sort == "demand_desc" else "CASE WHEN SUM(f.demand_qty) > 0 THEN SUM(f.sales_qty)::float / SUM(f.demand_qty) ELSE 1 END ASC"

    sql = f"""
        SELECT c.customer_no,
               c.customer_name,
               c.state,
               COALESCE(c.rpt_channel_desc, 'Unknown') AS channel,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty,
               COALESCE(SUM(f.oos_qty), 0) AS oos_qty
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
        GROUP BY c.customer_no, c.customer_name, c.state, c.rpt_channel_desc
        HAVING SUM(f.demand_qty) >= %s
        ORDER BY {order}
        LIMIT %s
    """
    params.extend([min_demand, top_n])

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    customers: list[dict[str, Any]] = []
    for cno, cname, state, ch, demand, sales, oos in rows:
        d = float(demand or 0)
        s = float(sales or 0)
        o = float(oos or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0
        customers.append({
            "customer_no": cno,
            "customer_name": cname or cno,
            "state": state or "",
            "channel": ch,
            "demand_qty": round(d, 1),
            "sales_qty": round(s, 1),
            "oos_qty": round(o, 1),
            "fill_rate": fr,
        })

    return {"customers": customers, "sort": sort, "top_n": top_n}


# ---------------------------------------------------------------------------
# 7. OOS Impact Bubble Chart
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/oos-impact")
@_CA_CACHE
def customer_analytics_oos_impact(
    response: FastAPIResponse,
    grain: str = Query(default="customer", pattern="^(customer|state)$"),
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
):
    """Bubble chart data: demand vs fill rate, bubble size = OOS qty."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, channel, None)

    if grain == "customer":
        group = "c.customer_no, c.customer_name, c.state, c.rpt_channel_desc"
        select_extra = "c.customer_no, c.customer_name, c.state, COALESCE(c.rpt_channel_desc, 'Unknown') AS channel"
    else:
        group = "c.state"
        select_extra = "c.state, c.state AS label, c.state AS state_col, 'All' AS channel"

    sql = f"""
        SELECT {select_extra},
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty,
               COALESCE(SUM(f.oos_qty), 0) AS oos_qty
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
          AND c.state IS NOT NULL AND TRIM(c.state) != ''
        GROUP BY {group}
        HAVING SUM(f.demand_qty) > 0
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT 200
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    bubbles: list[dict[str, Any]] = []
    for r in rows:
        if grain == "customer":
            cno, cname, state, ch, demand, sales, oos = r
            label = cname or cno
        else:
            state, label, _sc, ch, demand, sales, oos = r
            cno = None
        d = float(demand or 0)
        s = float(sales or 0)
        o = float(oos or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0
        entry: dict[str, Any] = {
            "label": str(label).strip() if label else (state or ""),
            "state": str(state).strip() if state else "",
            "channel": str(ch).strip() if ch else "Unknown",
            "demand_qty": round(d, 1),
            "sales_qty": round(s, 1),
            "oos_qty": round(o, 1),
            "fill_rate": fr,
        }
        if cno:
            entry["customer_no"] = cno
        bubbles.append(entry)

    return {"bubbles": bubbles, "grain": grain}


# ---------------------------------------------------------------------------
# 12. Customer-Item Affinity
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/affinity")
@_CA_CACHE
def customer_analytics_affinity(
    response: FastAPIResponse,
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    top_n: int = Query(default=20, ge=5, le=50),
):
    """Heatmap of top N customers x top N items."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, None, date_from, date_to, None, None)

    sql = f"""
        WITH base AS (
            SELECT f.customer_no,
                   c.customer_name,
                   f.item_id,
                   COALESCE(i.item_desc, f.item_id) AS item_desc,
                   SUM(f.demand_qty) AS demand_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            LEFT JOIN dim_item i ON i.item_id = f.item_id
            WHERE {where}
            GROUP BY f.customer_no, c.customer_name, f.item_id, i.item_desc
        ),
        top_customers AS (
            SELECT customer_no, customer_name
            FROM base
            GROUP BY customer_no, customer_name
            ORDER BY SUM(demand_qty) DESC
            LIMIT %s
        ),
        top_items AS (
            SELECT item_id, item_desc
            FROM base
            GROUP BY item_id, item_desc
            ORDER BY SUM(demand_qty) DESC
            LIMIT %s
        )
        SELECT b.customer_no, tc.customer_name,
               b.item_id, ti.item_desc,
               b.demand_qty
        FROM base b
        JOIN top_customers tc ON tc.customer_no = b.customer_no
        JOIN top_items ti ON ti.item_id = b.item_id
        ORDER BY b.demand_qty DESC
    """
    params.extend([top_n, top_n])

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    cust_set: dict[str, str] = {}
    item_set: dict[str, str] = {}
    cells: list[dict[str, Any]] = []
    for cno, cname, iid, idesc, dq in rows:
        cust_set[cno] = cname or cno
        item_set[iid] = idesc or iid
        cells.append({
            "customer_no": cno,
            "item_id": iid,
            "demand_qty": round(float(dq or 0), 1),
        })

    return {
        "customers": [{"customer_no": k, "customer_name": v} for k, v in cust_set.items()],
        "items": [{"item_id": k, "item_desc": v} for k, v in item_set.items()],
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# 13. Order Patterns
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/order-patterns")
@_CA_CACHE
def customer_analytics_order_patterns(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Frequency histogram + regularity scatter for ordering cadence."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, None, None)

    # Two-stage shape: rank customers by total demand FIRST, keep only the
    # top ~1000, then run the expensive STDDEV/window work on that bounded
    # set. The previous version computed LAG, AVG, STDDEV across all ~33K
    # customers and only sliced the top 200 at the very end — wasted work
    # since the response only renders 200 rows anyway. Top 1000 (5x the
    # display cap) gives some headroom for ties at the boundary.
    sql = f"""
        WITH base AS (
            SELECT f.customer_no,
                   c.customer_name,
                   f.startdate,
                   SUM(f.demand_qty) AS demand_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where}
            GROUP BY f.customer_no, c.customer_name, f.startdate
        ),
        cust_total AS (
            SELECT customer_no,
                   MAX(customer_name) AS customer_name,
                   SUM(demand_qty) AS total_demand
            FROM base
            GROUP BY customer_no
            ORDER BY SUM(demand_qty) DESC
            LIMIT 1000
        ),
        base_topn AS (
            SELECT b.customer_no, b.startdate
            FROM base b
            JOIN cust_total ct ON ct.customer_no = b.customer_no
        ),
        intervals AS (
            SELECT customer_no,
                   startdate,
                   EXTRACT(YEAR FROM age(startdate, LAG(startdate) OVER (PARTITION BY customer_no ORDER BY startdate))) * 12
                     + EXTRACT(MONTH FROM age(startdate, LAG(startdate) OVER (PARTITION BY customer_no ORDER BY startdate))) AS gap_months
            FROM base_topn
        ),
        cust_stats AS (
            SELECT customer_no,
                   AVG(gap_months) AS avg_interval,
                   CASE WHEN AVG(gap_months) > 0
                        THEN STDDEV(gap_months) / AVG(gap_months)
                        ELSE 0 END AS interval_cv,
                   COUNT(*) AS order_count
            FROM intervals
            WHERE gap_months IS NOT NULL
            GROUP BY customer_no
        )
        SELECT ct.customer_no, ct.customer_name,
               cs.avg_interval, cs.interval_cv, cs.order_count,
               ct.total_demand
        FROM cust_total ct
        LEFT JOIN cust_stats cs ON cs.customer_no = ct.customer_no
        ORDER BY ct.total_demand DESC
        LIMIT 200
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    buckets = {"monthly": 0, "bimonthly": 0, "quarterly": 0, "sporadic": 0}
    scatter: list[dict[str, Any]] = []

    for cno, cname, avg_int, cv, _oc, total_d in rows:
        avg_i = float(avg_int or 0)
        cv_val = float(cv or 0)
        td_val = float(total_d or 0)
        scatter.append({
            "customer_no": cno,
            "customer_name": cname or cno,
            "avg_interval_months": round(avg_i, 2),
            "interval_cv": round(cv_val, 2),
            "total_demand": round(td_val, 1),
        })
        if avg_i <= 1.5:
            buckets["monthly"] += 1
        elif avg_i <= 2.5:
            buckets["bimonthly"] += 1
        elif avg_i <= 4.0:
            buckets["quarterly"] += 1
        else:
            buckets["sporadic"] += 1

    total_custs = max(sum(buckets.values()), 1)
    histogram = [
        {"bucket": k, "count": v, "pct": round(v / total_custs * 100, 1)}
        for k, v in buckets.items()
    ]

    return {"frequency_histogram": histogram, "regularity_scatter": scatter}
