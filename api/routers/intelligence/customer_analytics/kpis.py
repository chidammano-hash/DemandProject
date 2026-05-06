"""Customer-analytics summary endpoints — KPIs, item picker, alerts.

Item 19 pilot: handlers are ``async def`` and use ``get_async_conn`` so the
13-endpoint Customer Analytics fan-out is no longer capped by the anyio
threadpool ceiling. Each handler awaits a real psycopg ``AsyncConnection``
from the dedicated async pool.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_async_conn, get_async_read_only_conn, set_cache

from ._helpers import (
    _CA_CACHE,
    _build_where,
    _build_where_mv,
    _customer_activity_source,
    _default_date_range,
)

router = APIRouter(tags=["customer-analytics"])


# ---------------------------------------------------------------------------
# 8. Item Search (typeahead for filter picker)
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/items")
@_CA_CACHE
async def customer_analytics_items(
    response: FastAPIResponse,
    search: str = Query(default="", min_length=0),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Typeahead item search for the customer analytics filter bar.

    Filters dim_item to only items that have at least one row in
    fact_customer_demand_monthly for the selected date range. Without this
    scoping, planners would pick items that look valid in dim_item but have
    zero demand history (we observed this with item 100012 — exists in
    dim_item, zero rows in fact, panels rendered as visually-blank).

    Date range defaults to the same trailing 12 months used by the rest of
    the CA endpoints so the picker matches the default filter window.
    """
    set_cache(response, max_age=600)
    search = search.strip()
    df, dt = _default_date_range()
    actual_from = (date_from or "").strip() or df
    actual_to = (date_to or "").strip() or dt

    base_sql = """
        SELECT DISTINCT i.item_id, i.item_desc
        FROM dim_item i
        WHERE EXISTS (
            SELECT 1 FROM fact_customer_demand_monthly f
            WHERE f.item_id = i.item_id
              AND f.startdate >= %s AND f.startdate < %s
        )
    """
    query_params: list[Any] = [actual_from, actual_to]

    if search:
        base_sql += " AND (i.item_id ILIKE %s OR i.item_desc ILIKE %s)"
        pattern = f"%{search}%"
        query_params.extend([pattern, pattern])

    base_sql += " ORDER BY i.item_id LIMIT 50"

    async with get_async_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(base_sql, query_params)
            rows = await cur.fetchall()

    items = [{"item_id": r[0], "item_desc": r[1] or r[0]} for r in rows]
    return {"items": items}


# ---------------------------------------------------------------------------
# 9. KPI Summary
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/kpis")
@_CA_CACHE
async def customer_analytics_kpis(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
    state: str | None = Query(default=None),
):
    """KPI values aggregated over the selected date range, with MoM deltas
    computed from the latest two months in the range."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    # Route through mv_customer_activity_monthly when no item filter is set
    # (the common dashboard load). The MV pre-joins fact x dim_customer and
    # aggregates to (customer_no, site, location_id, startdate), which cuts
    # /kpis from ~10.8s on the raw join to ~63ms (Item 8 of perf roadmap).
    source_from, uses_mv = _customer_activity_source(item_id)
    if uses_mv:
        where = _build_where_mv(params, date_from, date_to, channel, store_type, state=state)
    else:
        where = _build_where(params, item_id, date_from, date_to, channel, store_type, state=state)

    sql = f"""
        WITH base AS (
            SELECT f.startdate,
                   f.customer_no,
                   f.demand_qty,
                   f.sales_qty,
                   f.oos_qty
            FROM {source_from}
            WHERE {where}
        ),
        bounds AS (
            SELECT MAX(startdate) AS cur_month,
                   MAX(startdate) - INTERVAL '1 month' AS prev_month
            FROM base
        ),
        per_cust AS (
            SELECT customer_no, SUM(demand_qty) AS cust_demand
            FROM base
            GROUP BY customer_no
        ),
        top10 AS (
            SELECT COALESCE(SUM(cust_demand), 0) AS top10_demand
            FROM (
                SELECT cust_demand
                FROM per_cust
                ORDER BY cust_demand DESC
                LIMIT 10
            ) t
        ),
        agg AS (
            SELECT
                COALESCE(SUM(b.demand_qty), 0)                                                       AS t_demand,
                COALESCE(SUM(b.sales_qty), 0)                                                        AS t_sales,
                COALESCE(SUM(b.oos_qty), 0)                                                          AS t_oos,
                COUNT(DISTINCT b.customer_no)                                                        AS t_cust,
                COALESCE(SUM(b.demand_qty) FILTER (WHERE b.startdate = bnd.cur_month), 0)            AS c_demand,
                COALESCE(SUM(b.sales_qty)  FILTER (WHERE b.startdate = bnd.cur_month), 0)            AS c_sales,
                COALESCE(SUM(b.oos_qty)    FILTER (WHERE b.startdate = bnd.cur_month), 0)            AS c_oos,
                COUNT(DISTINCT b.customer_no) FILTER (WHERE b.startdate = bnd.cur_month)             AS c_cust,
                COALESCE(SUM(b.demand_qty) FILTER (WHERE b.startdate = bnd.prev_month), 0)           AS p_demand,
                COALESCE(SUM(b.sales_qty)  FILTER (WHERE b.startdate = bnd.prev_month), 0)           AS p_sales,
                COALESCE(SUM(b.oos_qty)    FILTER (WHERE b.startdate = bnd.prev_month), 0)           AS p_oos,
                COUNT(DISTINCT b.customer_no) FILTER (WHERE b.startdate = bnd.prev_month)            AS p_cust
            FROM base b CROSS JOIN bounds bnd
        )
        SELECT agg.t_demand, agg.t_sales, agg.t_oos, agg.t_cust,
               agg.c_demand, agg.c_sales, agg.c_oos, agg.c_cust,
               agg.p_demand, agg.p_sales, agg.p_oos, agg.p_cust,
               top10.top10_demand
        FROM agg, top10
    """

    # Read-only KPI aggregate — replica-safe (Item 24). Falls back to the
    # primary async pool when READ_REPLICA_URL is unset.
    async with get_async_read_only_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            row = await cur.fetchone()

    if not row:
        return {"kpis": []}

    t_demand = float(row[0] or 0)
    t_sales = float(row[1] or 0)
    t_oos = float(row[2] or 0)
    t_cust = int(row[3] or 0)
    c_demand = float(row[4] or 0)
    c_sales = float(row[5] or 0)
    c_oos = float(row[6] or 0)
    c_cust = int(row[7] or 0)
    p_demand = float(row[8] or 0)
    p_sales = float(row[9] or 0)
    p_oos = float(row[10] or 0)
    p_cust = int(row[11] or 0)
    top10_d = float(row[12] or 0)

    def _delta(cur_val: float, prev_val: float) -> float:
        if prev_val == 0:
            return 0.0
        return round((cur_val - prev_val) / prev_val * 100, 1)

    t_fr = round(t_sales / t_demand * 100, 1) if t_demand > 0 else 100.0
    c_fr = round(c_sales / c_demand * 100, 1) if c_demand > 0 else 100.0
    p_fr = round(p_sales / p_demand * 100, 1) if p_demand > 0 else 100.0
    conc = round(top10_d / t_demand * 100, 1) if t_demand > 0 else 0.0
    odr = round(t_sales / t_demand, 3) if t_demand > 0 else 0.0

    kpis = [
        {"key": "total_demand", "value": round(t_demand, 1), "delta": _delta(c_demand, p_demand)},
        {"key": "fill_rate", "value": t_fr, "delta": round(c_fr - p_fr, 1)},
        {"key": "oos_volume", "value": round(t_oos, 1), "delta": _delta(c_oos, p_oos)},
        {"key": "active_customers", "value": t_cust, "delta": _delta(float(c_cust), float(p_cust))},
        {"key": "concentration_top10", "value": conc, "delta": 0.0},
        {"key": "order_demand_ratio", "value": odr, "delta": 0.0},
    ]
    return {"kpis": kpis}


# ---------------------------------------------------------------------------
# 16. Alerts
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/alerts")
@_CA_CACHE
async def customer_analytics_alerts(
    response: FastAPIResponse,
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Evaluate threshold rules and return active alerts."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, None, date_from, date_to, None, None)

    # 1) fill rate < 85% per item-loc
    fr_sql = f"""
        SELECT f.item_id, f.location_id,
               CASE WHEN SUM(f.demand_qty) > 0
                    THEN SUM(f.sales_qty)::float / SUM(f.demand_qty) * 100
                    ELSE 100 END AS fill_rate
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
        GROUP BY f.item_id, f.location_id
        HAVING SUM(f.demand_qty) > 0
           AND SUM(f.sales_qty)::float / SUM(f.demand_qty) * 100 < 85
        ORDER BY SUM(f.sales_qty)::float / SUM(f.demand_qty) ASC
        LIMIT 50
    """

    # 2) HHI > 0.6 per item-loc
    params2: list[Any] = []
    where2 = _build_where(params2, None, date_from, date_to, None, None)
    hhi_sql = f"""
        WITH item_loc AS (
            SELECT f.item_id, f.location_id, f.customer_no,
                   SUM(f.demand_qty) AS cust_demand
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where2}
            GROUP BY f.item_id, f.location_id, f.customer_no
        ),
        il_total AS (
            SELECT item_id, location_id, SUM(cust_demand) AS total_demand
            FROM item_loc GROUP BY item_id, location_id
            HAVING SUM(cust_demand) > 0
        ),
        hhi AS (
            SELECT il.item_id, il.location_id,
                   SUM(POWER(il.cust_demand / t.total_demand, 2)) AS hhi
            FROM item_loc il
            JOIN il_total t ON t.item_id = il.item_id AND t.location_id = il.location_id
            GROUP BY il.item_id, il.location_id
            HAVING SUM(POWER(il.cust_demand / t.total_demand, 2)) > 0.6
        )
        SELECT item_id, location_id, ROUND(hhi::numeric, 3)
        FROM hhi
        ORDER BY hhi DESC
        LIMIT 50
    """

    # 3) churn rate > 10% MoM + 4) demand surge > 30% MoM
    # MoM aggregates are item-agnostic and only need (customer_no, startdate,
    # demand_qty) — route through the MV to avoid the fact x dim_customer JOIN.
    params3: list[Any] = []
    where3 = _build_where_mv(params3, date_from, date_to, None, None)
    mom_sql = f"""
        WITH base AS (
            SELECT f.startdate,
                   COUNT(DISTINCT f.customer_no) AS active_cust,
                   SUM(f.demand_qty) AS demand
            FROM mv_customer_activity_monthly f
            WHERE {where3}
            GROUP BY f.startdate
            ORDER BY f.startdate
        ),
        lagged AS (
            SELECT startdate, active_cust, demand,
                   LAG(active_cust) OVER (ORDER BY startdate) AS prev_cust,
                   LAG(demand) OVER (ORDER BY startdate) AS prev_demand
            FROM base
        )
        SELECT startdate, active_cust, prev_cust, demand, prev_demand
        FROM lagged
        WHERE prev_cust IS NOT NULL
        ORDER BY startdate DESC
        LIMIT 1
    """

    async with get_async_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(fr_sql, params)
            fr_rows = await cur.fetchall()
            await cur.execute(hhi_sql, params2)
            hhi_rows = await cur.fetchall()
            await cur.execute(mom_sql, params3)
            mom_row = await cur.fetchone()

    alerts: list[dict[str, Any]] = []

    for item_id_val, loc, fill_rate_val in fr_rows:
        alerts.append({
            "alert_type": "low_fill_rate",
            "severity": "red" if float(fill_rate_val) < 70 else "amber",
            "message": f"Fill rate {round(float(fill_rate_val), 1)}% for {item_id_val} at {loc}",
            "item_id": item_id_val,
            "loc": loc,
            "value": round(float(fill_rate_val), 1),
            "threshold": 85,
        })

    for item_id_val, loc, hhi_val in hhi_rows:
        alerts.append({
            "alert_type": "high_concentration",
            "severity": "red" if float(hhi_val) > 0.8 else "amber",
            "message": f"HHI {hhi_val} for {item_id_val} at {loc}",
            "item_id": item_id_val,
            "loc": loc,
            "value": float(hhi_val),
            "threshold": 0.6,
        })

    if mom_row:
        _sd, cur_cust, prev_cust, cur_demand, prev_demand = mom_row
        cur_c = int(cur_cust or 0)
        prev_c = int(prev_cust or 0)
        if prev_c > 0:
            churn_rate = round((prev_c - cur_c) / prev_c * 100, 1)
            if churn_rate > 10:
                alerts.append({
                    "alert_type": "high_churn",
                    "severity": "red" if churn_rate > 20 else "amber",
                    "message": f"Customer churn rate {churn_rate}% MoM",
                    "item_id": None,
                    "loc": None,
                    "value": churn_rate,
                    "threshold": 10,
                })
        cur_d = float(cur_demand or 0)
        prev_d = float(prev_demand or 0)
        if prev_d > 0:
            surge = round((cur_d - prev_d) / prev_d * 100, 1)
            if surge > 30:
                alerts.append({
                    "alert_type": "demand_surge",
                    "severity": "amber",
                    "message": f"New demand surge {surge}% MoM",
                    "item_id": None,
                    "loc": None,
                    "value": surge,
                    "threshold": 30,
                })

    return {"alerts": alerts}
