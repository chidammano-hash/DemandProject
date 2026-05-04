"""Inventory Planning — IPfeature9: Demand Sensing endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, get_conn, set_cache

router = APIRouter(tags=["inv-planning"])




@router.get("/inv-planning/demand-signals/summary")
def get_demand_signals_summary(
    response: FastAPIResponse,
    signal_date: Optional[str] = Query(None),
) -> dict:
    """Summary of demand signals for a given date."""
    set_cache(response, max_age=3600)

    params: list = []
    date_filter = ""
    if signal_date:
        params.append(signal_date)
        date_filter = "WHERE signal_date = %s"
    else:
        date_filter = "WHERE signal_date = (SELECT MAX(signal_date) FROM fact_demand_signals)"

    sql = f"""
        SELECT
            MAX(signal_date)                                       AS signal_date,
            COUNT(*)                                               AS total_items_with_signals,
            COUNT(*) FILTER (WHERE signal_type = 'above_plan')    AS above_plan,
            COUNT(*) FILTER (WHERE signal_type = 'below_plan')    AS below_plan,
            COUNT(*) FILTER (WHERE signal_type = 'on_plan')       AS on_plan,
            COUNT(*) FILTER (WHERE alert_priority = 'urgent')     AS urgent_alerts,
            COUNT(*) FILTER (WHERE alert_priority = 'watch')      AS watch_alerts,
            COUNT(*) FILTER (WHERE projected_stockout = TRUE)     AS projected_stockouts
        FROM fact_demand_signals
        {date_filter}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]

    result = dict(zip(cols, row)) if row else {}
    return {k: (str(v) if hasattr(v, "isoformat") else int(v) if isinstance(v, (int,)) else v)
            for k, v in result.items()}


@router.get("/inv-planning/demand-signals")
def get_demand_signals(
    response: FastAPIResponse,
    signal_date: Optional[str] = Query(None),
    signal_type: Optional[str] = Query(None, max_length=20),
    alert_priority: Optional[str] = Query(None, max_length=20),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("alert_priority", max_length=40),
    sort_dir: str = Query("desc", max_length=4),
) -> dict:
    """Paginated demand signals list."""
    set_cache(response, max_age=3600)

    allowed_sort = {"alert_priority", "demand_vs_forecast_pct", "signal_strength", "projected_monthly"}
    order_col = sort_by if sort_by in allowed_sort else "alert_priority"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_clauses: list[str] = []
    params: list = []

    if signal_date:
        params.append(signal_date)
        where_clauses.append("s.signal_date = %s")
    else:
        where_clauses.append("s.signal_date = (SELECT MAX(signal_date) FROM fact_demand_signals)")

    if signal_type:
        params.append(signal_type)
        where_clauses.append("s.signal_type = %s")
    if alert_priority:
        params.append(alert_priority)
        where_clauses.append("s.alert_priority = %s")
    if item:
        params.append(f"%{item}%")
        where_clauses.append("s.item_id ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_clauses.append("s.loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_clauses.append("d.abc_vol = %s")

    where_sql = "WHERE " + " AND ".join(where_clauses)
    count_sql = f"""
        SELECT COUNT(*) FROM fact_demand_signals s
        LEFT JOIN dim_sku d ON s.item_id = d.item_id AND s.loc = d.loc
        {where_sql}
    """

    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT s.item_id, s.loc, s.signal_date, s.signal_type, s.alert_priority,
               s.mtd_actual, s.projected_monthly, s.forecast_monthly,
               s.demand_vs_forecast_pct, s.projected_stockout, s.projected_excess,
               s.current_on_hand, s.is_below_ss, s.days_remaining,
               d.abc_vol
        FROM fact_demand_signals s
        LEFT JOIN dim_sku d ON s.item_id = d.item_id AND s.loc = d.loc
        {where_sql}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params[:-2])
            total = cur.fetchone()[0] or 0
            cur.execute(data_sql, params)
            rows = cur.fetchall()
            sd = str(rows[0][2]) if rows else None

    return {
        "signal_date": sd,
        "total": int(total),
        "rows": [
            {
                "item_id":                r[0],
                "loc":                    r[1],
                "signal_date":            str(r[2]),
                "signal_type":            r[3],
                "alert_priority":         r[4],
                "mtd_actual":             _f(r[5]),
                "projected_monthly":      _f(r[6]),
                "forecast_monthly":       _f(r[7]),
                "demand_vs_forecast_pct": _f(r[8]),
                "projected_stockout":     bool(r[9]) if r[9] is not None else False,
                "projected_excess":       bool(r[10]) if r[10] is not None else False,
                "current_on_hand":        _f(r[11]),
                "is_below_ss":            bool(r[12]) if r[12] is not None else False,
                "days_remaining":         int(r[13]) if r[13] is not None else None,
                "abc_vol":                r[14],
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/demand-signals/item")
def get_demand_signal_item(
    response: FastAPIResponse,
    item: str = Query(..., max_length=120),
    location: str = Query(..., max_length=120),
) -> dict:
    """Single item-location demand signal with daily MTD series."""
    set_cache(response, max_age=3600)

    sql = """
        SELECT item_id, loc, signal_date, signal_type, alert_priority,
               mtd_actual, projected_monthly, forecast_monthly,
               demand_vs_forecast_pct, days_elapsed, days_remaining,
               current_on_hand, is_below_ss
        FROM fact_demand_signals
        WHERE item_id = %s AND loc = %s
        ORDER BY signal_date DESC
        LIMIT 1
    """

    daily_sql = """
        SELECT snapshot_date, mtd_sales,
               mtd_sales / NULLIF(EXTRACT(day FROM snapshot_date), 0) *
               EXTRACT(days IN month FROM snapshot_date) AS mtd_expected_pace
        FROM fact_inventory_snapshot
        WHERE item_id = %s AND loc = %s
          AND snapshot_date >= DATE_TRUNC('month', CURRENT_DATE)
        ORDER BY snapshot_date
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [item, location])
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="No signal found for item/location")

            cur.execute(daily_sql, [item, location])
            daily_rows = cur.fetchall()

    return {
        "item_id":                row[0],
        "loc":                    row[1],
        "signal_date":            str(row[2]),
        "signal_type":            row[3],
        "alert_priority":         row[4],
        "mtd_actual":             _f(row[5]),
        "projected_monthly":      _f(row[6]),
        "forecast_monthly":       _f(row[7]),
        "demand_vs_forecast_pct": _f(row[8]),
        "days_elapsed":           int(row[9]) if row[9] else None,
        "days_remaining":         int(row[10]) if row[10] else None,
        "current_on_hand":        _f(row[11]),
        "is_below_ss":            bool(row[12]) if row[12] is not None else False,
        "daily_series": [
            {"date": str(r[0]), "mtd_actual": _f(r[1]), "mtd_expected_pace": _f(r[2])}
            for r in daily_rows
        ],
    }
