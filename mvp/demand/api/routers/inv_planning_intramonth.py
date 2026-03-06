"""Inventory Planning — IPfeature14: Intra-Month Stockout Detection endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

router = APIRouter()


def _f(v: Any) -> float | None:
    return float(v) if v is not None else None


@router.get("/inv-planning/intramonth-stockouts/summary")
def get_intramonth_stockout_summary(
    response: FastAPIResponse,
    month_from: Optional[str] = Query(None),
    month_to: Optional[str] = Query(None),
    abc_vol: Optional[str] = Query(None, max_length=10),
) -> dict:
    """Portfolio-level intra-month stockout summary."""
    set_cache(response, max_age=600)

    where_clauses: list[str] = []
    params: list = []

    if month_from:
        params.append(month_from)
        where_clauses.append(f"month_start >= ${len(params)}")
    if month_to:
        params.append(month_to)
        where_clauses.append(f"month_start <= ${len(params)}")
    if abc_vol:
        params.append(abc_vol.upper())
        where_clauses.append(f"abc_vol = ${len(params)}")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT
            COUNT(*)                                  AS total_records,
            COUNT(*) FILTER (WHERE had_full_stockout)     AS items_with_stockout,
            COUNT(*) FILTER (WHERE had_extended_stockout) AS items_with_extended_stockout,
            AVG(stockout_day_rate)                    AS avg_stockout_day_rate,
            SUM(stockout_days)                        AS total_stockout_days,
            SUM(est_lost_sales)                       AS total_est_lost_sales,
            AVG(avg_qty_on_hand)                      AS avg_qty_on_hand
        FROM mv_intramonth_stockout
        {where_sql}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]

    result = dict(zip(cols, row)) if row else {}
    return {k: (_f(v) if isinstance(v, (int, float)) else v) for k, v in result.items()}


@router.get("/inv-planning/intramonth-stockouts/detail")
def get_intramonth_stockout_detail(
    response: FastAPIResponse,
    month_from: Optional[str] = Query(None),
    month_to: Optional[str] = Query(None),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    had_stockout: Optional[bool] = Query(None),
    had_extended: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("stockout_day_rate", max_length=40),
    sort_dir: str = Query("desc", max_length=4),
) -> dict:
    """Paginated intra-month stockout detail."""
    set_cache(response, max_age=300)

    allowed_sort = {"stockout_day_rate", "stockout_days", "est_lost_sales", "avg_qty_on_hand"}
    order_col = sort_by if sort_by in allowed_sort else "stockout_day_rate"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_clauses: list[str] = []
    params: list = []

    if month_from:
        params.append(month_from)
        where_clauses.append(f"month_start >= ${len(params)}")
    if month_to:
        params.append(month_to)
        where_clauses.append(f"month_start <= ${len(params)}")
    if item:
        params.append(f"%{item}%")
        where_clauses.append(f"item_no ILIKE ${len(params)}")
    if location:
        params.append(f"%{location}%")
        where_clauses.append(f"loc ILIKE ${len(params)}")
    if abc_vol:
        params.append(abc_vol.upper())
        where_clauses.append(f"abc_vol = ${len(params)}")
    if had_stockout is not None:
        params.append(had_stockout)
        where_clauses.append(f"had_full_stockout = ${len(params)}")
    if had_extended is not None:
        params.append(had_extended)
        where_clauses.append(f"had_extended_stockout = ${len(params)}")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    count_sql = f"SELECT COUNT(*) FROM mv_intramonth_stockout {where_sql}"

    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT item_no, loc, month_start,
               snapshot_days, stockout_days, stockout_day_rate,
               min_qty_on_hand, max_qty_on_hand, avg_qty_on_hand,
               est_lost_sales, had_full_stockout, had_extended_stockout,
               abc_vol, abc_xyz_segment, cluster_assignment
        FROM mv_intramonth_stockout
        {where_sql}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params[:-2])
            total = cur.fetchone()[0] or 0
            cur.execute(data_sql, params)
            rows = cur.fetchall()

    return {
        "total": int(total),
        "rows": [
            {
                "item_no":              r[0],
                "loc":                  r[1],
                "month_start":          str(r[2]),
                "snapshot_days":        int(r[3] or 0),
                "stockout_days":        int(r[4] or 0),
                "stockout_day_rate":    _f(r[5]),
                "min_qty_on_hand":      _f(r[6]),
                "max_qty_on_hand":      _f(r[7]),
                "avg_qty_on_hand":      _f(r[8]),
                "est_lost_sales":       _f(r[9]),
                "had_full_stockout":    bool(r[10]),
                "had_extended_stockout":bool(r[11]),
                "abc_vol":              r[12],
                "abc_xyz_segment":      r[13],
                "cluster_assignment":   r[14],
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/intramonth-stockouts/daily")
def get_intramonth_daily(
    response: FastAPIResponse,
    item: str = Query(..., max_length=120),
    location: str = Query(..., max_length=120),
    month: Optional[str] = Query(None),
) -> dict:
    """Daily stockout data for a specific item-location (for drill-down chart)."""
    set_cache(response, max_age=600)

    params: list = [item, location]
    month_filter = ""
    if month:
        params.append(month)
        month_filter = "AND DATE_TRUNC('month', snapshot_date)::DATE = %s"

    sql = f"""
        SELECT snapshot_date, qty_on_hand, mtd_sales,
               GREATEST(
                   mtd_sales - LAG(mtd_sales, 1, 0::NUMERIC) OVER (
                       PARTITION BY item_no, loc, DATE_TRUNC('month', snapshot_date)
                       ORDER BY snapshot_date
                   ), 0
               ) AS daily_sls
        FROM fact_inventory_snapshot
        WHERE item_no = %s AND loc = %s
        {month_filter}
        ORDER BY snapshot_date DESC
        LIMIT 62
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return {
        "item_no": item,
        "loc": location,
        "daily": [
            {
                "snapshot_date": str(r[0]),
                "qty_on_hand":   _f(r[1]),
                "mtd_sales":     _f(r[2]),
                "daily_sls":     _f(r[3]),
            }
            for r in rows
        ],
    }
