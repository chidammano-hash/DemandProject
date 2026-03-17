"""Fill Rate & Demand Fulfillment Analytics — IPfeature8.

Router mounted at /fill-rate in api/main.py.
Queries mv_fill_rate_monthly materialized view for fill rate metrics.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, _s, add_cross_dim_filters, get_conn, set_cache

router = APIRouter(tags=["fill-rate"])

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# GET /fill-rate/summary
# ---------------------------------------------------------------------------

@router.get("/fill-rate/summary")
def get_fill_rate_summary(
    response: FastAPIResponse,
    month_from: Optional[str] = Query(None),
    month_to: Optional[str] = Query(None),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    cluster_assignment: Optional[str] = Query(None, max_length=120),
    region: Optional[str] = Query(None, max_length=120),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
) -> dict:
    """Portfolio fill rate summary with by_abc breakdown, worst items, and trend.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    where_parts: list[str] = []
    params: list = []

    if month_from:
        params.append(month_from)
        where_parts.append("month_start >= %s")
    if month_to:
        params.append(month_to)
        where_parts.append("month_start <= %s")
    if item:
        params.append(f"%{item}%")
        where_parts.append("item_no ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_parts.append("loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_parts.append("abc_vol = %s")
    if cluster_assignment:
        params.append(cluster_assignment)
        where_parts.append("cluster_assignment = %s")
    if region:
        params.append(region)
        where_parts.append("region = %s")
    add_cross_dim_filters(where_parts, params, brand=brand, category=category, market=market)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    totals_sql = f"""
        SELECT
            CASE WHEN SUM(total_ordered) > 0
                 THEN SUM(total_shipped) / SUM(total_ordered)
                 ELSE NULL END                              AS portfolio_fill_rate,
            COALESCE(SUM(total_ordered), 0)                AS total_ordered,
            COALESCE(SUM(total_shipped), 0)                AS total_shipped,
            COALESCE(SUM(shortage_qty), 0)                 AS total_shortage_qty,
            COUNT(*) FILTER (WHERE had_partial_fulfillment) AS partial_fulfillment_events
        FROM mv_fill_rate_monthly t
        {where_sql}
    """

    abc_sql = f"""
        SELECT
            abc_vol,
            CASE WHEN SUM(total_ordered) > 0
                 THEN SUM(total_shipped) / SUM(total_ordered) ELSE NULL END AS avg_fill_rate,
            COALESCE(SUM(shortage_qty), 0)                                  AS total_shortage_qty,
            COUNT(*) FILTER (WHERE had_partial_fulfillment)                 AS events
        FROM mv_fill_rate_monthly t
        {where_sql}
        GROUP BY abc_vol
        ORDER BY abc_vol
    """

    # If there are existing filters, we need AND, else WHERE
    if where_parts:
        worst_sql = f"""
            SELECT item_no, loc, fill_rate, shortage_qty, abc_vol
            FROM mv_fill_rate_monthly t
            WHERE {' AND '.join(where_parts)} AND shortage_qty > 0
            ORDER BY shortage_qty DESC
            LIMIT 10
        """
    else:
        worst_sql = """
            SELECT item_no, loc, fill_rate, shortage_qty, abc_vol
            FROM mv_fill_rate_monthly t
            WHERE shortage_qty > 0
            ORDER BY shortage_qty DESC
            LIMIT 10
        """

    trend_sql = f"""
        SELECT
            month_start,
            CASE WHEN SUM(total_ordered) > 0
                 THEN SUM(total_shipped) / SUM(total_ordered) ELSE NULL END AS portfolio_fill_rate,
            COALESCE(SUM(shortage_qty), 0) AS total_shortage_qty
        FROM mv_fill_rate_monthly t
        {where_sql}
        GROUP BY month_start
        ORDER BY month_start
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(totals_sql, params)
            totals_row = cur.fetchone()
            totals_cols = [d[0] for d in cur.description]
            totals = dict(zip(totals_cols, totals_row)) if totals_row else {}

            cur.execute(abc_sql, params)
            abc_rows = cur.fetchall()

            cur.execute(worst_sql, params)
            worst_rows = cur.fetchall()

            cur.execute(trend_sql, params)
            trend_rows = cur.fetchall()

    by_abc: dict = {}
    for r in abc_rows:
        seg = r[0]
        by_abc[seg] = {
            "avg_fill_rate": _f(r[1]),
            "total_shortage_qty": _f(r[2]),
            "events": int(r[3] or 0),
        }

    worst_items = [
        {
            "item_no": r[0],
            "loc": r[1],
            "fill_rate": _f(r[2]),
            "shortage_qty": _f(r[3]),
            "abc_vol": r[4],
        }
        for r in worst_rows
    ]

    trend = [
        {
            "month_start": str(r[0]),
            "portfolio_fill_rate": _f(r[1]),
            "total_shortage_qty": _f(r[2]),
        }
        for r in trend_rows
    ]

    return {
        "portfolio_fill_rate": _f(totals.get("portfolio_fill_rate")),
        "total_ordered": _f(totals.get("total_ordered")) or 0.0,
        "total_shipped": _f(totals.get("total_shipped")) or 0.0,
        "total_shortage_qty": _f(totals.get("total_shortage_qty")) or 0.0,
        "partial_fulfillment_events": int(totals.get("partial_fulfillment_events") or 0),
        "by_abc": by_abc,
        "worst_items": worst_items,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# GET /fill-rate/trend
# ---------------------------------------------------------------------------

@router.get("/fill-rate/trend")
def get_fill_rate_trend(
    response: FastAPIResponse,
    month_from: Optional[str] = Query(None),
    month_to: Optional[str] = Query(None),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
) -> dict:
    """Monthly fill rate trend.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    where_parts: list[str] = []
    params: list = []

    if month_from:
        params.append(month_from)
        where_parts.append("month_start >= %s")
    if month_to:
        params.append(month_to)
        where_parts.append("month_start <= %s")
    if item:
        params.append(f"%{item}%")
        where_parts.append("item_no ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_parts.append("loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_parts.append("abc_vol = %s")
    add_cross_dim_filters(where_parts, params, brand=brand, category=category, market=market)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        SELECT
            month_start,
            CASE WHEN SUM(total_ordered) > 0
                 THEN SUM(total_shipped) / SUM(total_ordered) ELSE NULL END AS fill_rate,
            COALESCE(SUM(total_ordered), 0)  AS total_ordered,
            COALESCE(SUM(total_shipped), 0)  AS total_shipped,
            COALESCE(SUM(shortage_qty), 0)   AS shortage_qty
        FROM mv_fill_rate_monthly t
        {where_sql}
        GROUP BY month_start
        ORDER BY month_start
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return {
        "months": [
            {
                "month_start": str(r[0]),
                "fill_rate": _f(r[1]),
                "total_ordered": _f(r[2]),
                "total_shipped": _f(r[3]),
                "shortage_qty": _f(r[4]),
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# GET /fill-rate/detail
# ---------------------------------------------------------------------------

@router.get("/fill-rate/detail")
def get_fill_rate_detail(
    response: FastAPIResponse,
    month_from: Optional[str] = Query(None),
    month_to: Optional[str] = Query(None),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    had_partial_fulfillment: Optional[bool] = Query(None),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("shortage_qty", max_length=40),
    sort_dir: str = Query("desc", max_length=4),
) -> dict:
    """Paginated fill rate detail rows.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    allowed_sort = {"fill_rate", "shortage_qty", "total_ordered"}
    order_col = sort_by if sort_by in allowed_sort else "shortage_qty"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list = []

    if month_from:
        params.append(month_from)
        where_parts.append("month_start >= %s")
    if month_to:
        params.append(month_to)
        where_parts.append("month_start <= %s")
    if item:
        params.append(f"%{item}%")
        where_parts.append("item_no ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_parts.append("loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_parts.append("abc_vol = %s")
    if had_partial_fulfillment is not None:
        params.append(had_partial_fulfillment)
        where_parts.append("had_partial_fulfillment = %s")
    add_cross_dim_filters(where_parts, params, brand=brand, category=category, market=market)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM mv_fill_rate_monthly t {where_sql}"

    filter_params = list(params)
    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT
            item_no, loc, month_start,
            total_ordered, total_shipped,
            fill_rate, shortage_qty,
            had_partial_fulfillment,
            abc_vol, cluster_assignment, region
        FROM mv_fill_rate_monthly t
        {where_sql}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, filter_params)
            total = cur.fetchone()[0] or 0

            cur.execute(data_sql, params)
            rows = cur.fetchall()

    return {
        "total": int(total),
        "rows": [
            {
                "item_no": r[0],
                "loc": r[1],
                "month_start": str(r[2]),
                "total_ordered": _f(r[3]),
                "total_shipped": _f(r[4]),
                "fill_rate": _f(r[5]),
                "shortage_qty": _f(r[6]),
                "had_partial_fulfillment": r[7],
                "abc_vol": r[8],
                "cluster_assignment": r[9],
                "region": r[10],
            }
            for r in rows
        ],
    }
