"""Inventory backtest endpoints (feature 37)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared filter builder
# ---------------------------------------------------------------------------

def _inv_backtest_filters(
    models: str,
    month_from: str,
    month_to: str,
    item: str,
    location: str,
    cluster_assignment: str,
    abc_vol: str,
    region: str,
) -> tuple[list[str], list[Any]]:
    """Build WHERE parts and params for inventory-backtest queries."""
    parts: list[str] = []
    params: list[Any] = []
    if models.strip():
        ml = [m.strip() for m in models.split(",") if m.strip()]
        ph = ",".join(["%s"] * len(ml))
        parts.append(f"model_id IN ({ph})")
        params.extend(ml)
    if month_from.strip():
        parts.append("month_start >= %s::date")
        params.append(month_from.strip())
    if month_to.strip():
        parts.append("month_start <= %s::date")
        params.append(month_to.strip())
    if item.strip():
        parts.append("item_no ILIKE %s")
        params.append(f"%{item.strip()}%")
    if location.strip():
        parts.append("loc ILIKE %s")
        params.append(f"%{location.strip()}%")
    if cluster_assignment.strip():
        parts.append("cluster_assignment = %s")
        params.append(cluster_assignment.strip())
    if abc_vol.strip():
        parts.append("abc_vol = %s")
        params.append(abc_vol.strip())
    if region.strip():
        parts.append("region = %s")
        params.append(region.strip())
    return parts, params


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/inventory-backtest/summary")
def inv_backtest_summary(
    response: FastAPIResponse,
    models: str = Query(default="", max_length=500),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    cluster_assignment: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    excess_dos_threshold: int = Query(default=90, ge=1, le=365),
):
    """Per-model inventory-outcome metrics: stockout rate, excess rate, service level, WAPE."""
    set_cache(response, max_age=120)

    parts, params = _inv_backtest_filters(
        models, month_from, month_to, item, location,
        cluster_assignment, abc_vol, region,
    )
    threshold_idx = len(params)
    params.append(excess_dos_threshold)
    where_sql = f"WHERE {' AND '.join(parts)}" if parts else ""

    sql = f"""
        SELECT
            model_id,
            COUNT(*)::bigint                                                AS dfu_months,
            SUM(CASE WHEN eom_qty_on_hand <= 0 THEN 1 ELSE 0 END)::bigint  AS stockout_count,
            SUM(CASE WHEN dos IS NOT NULL AND dos > %s THEN 1 ELSE 0 END)::bigint AS excess_count,
            CASE WHEN ABS(SUM(actual_demand)) > 0
                 THEN (SUM(abs_error) / ABS(SUM(actual_demand)) * 100)::double precision
                 ELSE NULL END                                              AS wape,
            CASE WHEN ABS(SUM(actual_demand)) > 0
                 THEN ((SUM(forecast) / ABS(SUM(actual_demand))) - 1) * 100::double precision
                 ELSE NULL END                                              AS bias,
            AVG(dos)::double precision                                      AS avg_dos
        FROM mv_inventory_forecast_monthly
        {where_sql}
        GROUP BY model_id
        ORDER BY model_id
    """

    ordered_params = [excess_dos_threshold] + params[:threshold_idx]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, ordered_params)
        rows = cur.fetchall()

    by_model: dict[str, dict[str, Any]] = {}
    model_list: list[str] = []
    for r in rows:
        mid = r[0]
        dfu_months = int(r[1])
        so_count = int(r[2])
        ex_count = int(r[3])
        model_list.append(mid)
        by_model[mid] = {
            "dfu_months": dfu_months,
            "stockout_count": so_count,
            "stockout_rate": round(so_count / dfu_months * 100, 2) if dfu_months > 0 else 0,
            "excess_count": ex_count,
            "excess_rate": round(ex_count / dfu_months * 100, 2) if dfu_months > 0 else 0,
            "service_level": round((1 - so_count / dfu_months) * 100, 2) if dfu_months > 0 else 100,
            "avg_dos": round(float(r[6]), 1) if r[6] is not None else None,
            "wape": round(float(r[4]), 2) if r[4] is not None else None,
            "bias": round(float(r[5]), 2) if r[5] is not None else None,
        }
    return {"models": model_list, "excess_dos_threshold": excess_dos_threshold, "by_model": by_model}


@router.get("/inventory-backtest/trend")
def inv_backtest_trend(
    response: FastAPIResponse,
    models: str = Query(default="", max_length=500),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    cluster_assignment: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    excess_dos_threshold: int = Query(default=90, ge=1, le=365),
):
    """Monthly inventory-outcome trend by model."""
    set_cache(response, max_age=120)

    parts, params = _inv_backtest_filters(
        models, month_from, month_to, item, location,
        cluster_assignment, abc_vol, region,
    )
    where_sql = f"WHERE {' AND '.join(parts)}" if parts else ""

    sql = f"""
        SELECT
            month_start,
            model_id,
            COUNT(*)::bigint                                                AS dfu_months,
            SUM(CASE WHEN eom_qty_on_hand <= 0 THEN 1 ELSE 0 END)::bigint  AS stockout_count,
            SUM(CASE WHEN dos IS NOT NULL AND dos > %s THEN 1 ELSE 0 END)::bigint AS excess_count,
            AVG(dos)::double precision                                      AS avg_dos,
            CASE WHEN ABS(SUM(actual_demand)) > 0
                 THEN (SUM(abs_error) / ABS(SUM(actual_demand)) * 100)::double precision
                 ELSE NULL END                                              AS wape
        FROM mv_inventory_forecast_monthly
        {where_sql}
        GROUP BY month_start, model_id
        ORDER BY month_start, model_id
    """

    ordered_params: list[Any] = [excess_dos_threshold] + params
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, ordered_params)
        rows = cur.fetchall()

    trend: list[dict[str, Any]] = []
    current_month: str | None = None
    current_entry: dict[str, Any] = {}
    for r in rows:
        month_str = str(r[0])
        if month_str != current_month:
            if current_month is not None:
                trend.append(current_entry)
            current_month = month_str
            current_entry = {"month": month_str, "by_model": {}}
        dfu_m = int(r[2])
        so_c = int(r[3])
        ex_c = int(r[4])
        current_entry["by_model"][r[1]] = {
            "stockout_rate": round(so_c / dfu_m * 100, 2) if dfu_m > 0 else 0,
            "excess_rate": round(ex_c / dfu_m * 100, 2) if dfu_m > 0 else 0,
            "avg_dos": round(float(r[5]), 1) if r[5] is not None else None,
            "wape": round(float(r[6]), 2) if r[6] is not None else None,
        }
    if current_month is not None:
        trend.append(current_entry)

    return {"trend": trend}


@router.get("/inventory-backtest/root-cause")
def inv_backtest_root_cause(
    response: FastAPIResponse,
    model_id: str = Query(min_length=1, max_length=120),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    cluster_assignment: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    excess_dos_threshold: int = Query(default=90, ge=1, le=365),
):
    """Stockout/excess root cause breakdown by forecast bias direction for a single model."""
    set_cache(response, max_age=120)

    parts, params = _inv_backtest_filters(
        "", month_from, month_to, item, location,
        cluster_assignment, abc_vol, region,
    )
    parts.append("model_id = %s")
    params.append(model_id.strip())
    where_sql = f"WHERE {' AND '.join(parts)}"

    sql = f"""
        SELECT
            SUM(CASE WHEN eom_qty_on_hand <= 0 THEN 1 ELSE 0 END)::bigint                                AS stockout_total,
            SUM(CASE WHEN eom_qty_on_hand <= 0 AND bias_direction = 'under' THEN 1 ELSE 0 END)::bigint   AS stockout_under_forecast,
            SUM(CASE WHEN eom_qty_on_hand <= 0 AND bias_direction = 'over' THEN 1 ELSE 0 END)::bigint    AS stockout_over_forecast,
            SUM(CASE WHEN eom_qty_on_hand <= 0 AND bias_direction = 'exact' THEN 1 ELSE 0 END)::bigint   AS stockout_exact,
            SUM(CASE WHEN dos IS NOT NULL AND dos > %s THEN 1 ELSE 0 END)::bigint                        AS excess_total,
            SUM(CASE WHEN dos IS NOT NULL AND dos > %s AND bias_direction = 'over' THEN 1 ELSE 0 END)::bigint  AS excess_over_forecast,
            SUM(CASE WHEN dos IS NOT NULL AND dos > %s AND bias_direction = 'under' THEN 1 ELSE 0 END)::bigint AS excess_under_forecast,
            SUM(CASE WHEN dos IS NOT NULL AND dos > %s AND bias_direction = 'exact' THEN 1 ELSE 0 END)::bigint AS excess_exact
        FROM mv_inventory_forecast_monthly
        {where_sql}
    """

    ordered_params: list[Any] = [excess_dos_threshold] * 4 + params
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, ordered_params)
        row = cur.fetchone()

    if row is None:
        return {
            "model_id": model_id.strip(),
            "stockout_total": 0, "stockout_under_forecast": 0,
            "stockout_over_forecast": 0, "stockout_exact": 0,
            "excess_total": 0, "excess_over_forecast": 0,
            "excess_under_forecast": 0, "excess_exact": 0,
        }

    return {
        "model_id": model_id.strip(),
        "stockout_total": int(row[0] or 0),
        "stockout_under_forecast": int(row[1] or 0),
        "stockout_over_forecast": int(row[2] or 0),
        "stockout_exact": int(row[3] or 0),
        "excess_total": int(row[4] or 0),
        "excess_over_forecast": int(row[5] or 0),
        "excess_under_forecast": int(row[6] or 0),
        "excess_exact": int(row[7] or 0),
    }


_INV_BACKTEST_DETAIL_SORT_COLS = {
    "item_no", "loc", "month_start", "model_id", "forecast",
    "actual_demand", "eom_qty_on_hand", "dos", "forecast_error", "abs_error",
}


@router.get("/inventory-backtest/detail")
def inv_backtest_detail(
    response: FastAPIResponse,
    models: str = Query(default="", max_length=500),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    cluster_assignment: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    event_type: str = Query(default="all", max_length=20),
    excess_dos_threshold: int = Query(default=90, ge=1, le=365),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="month_start", max_length=60),
    sort_dir: str = Query(default="desc", max_length=4),
):
    """Paginated DFU-level inventory events (stockout / excess) with forecast error detail."""
    set_cache(response, max_age=60)

    parts, params = _inv_backtest_filters(
        models, month_from, month_to, item, location,
        cluster_assignment, abc_vol, region,
    )
    if event_type == "stockout":
        parts.append("eom_qty_on_hand <= 0")
    elif event_type == "excess":
        parts.append("dos IS NOT NULL AND dos > %s")
        params.append(excess_dos_threshold)

    where_sql = f"WHERE {' AND '.join(parts)}" if parts else ""

    safe_sort = sort_by if sort_by in _INV_BACKTEST_DETAIL_SORT_COLS else "month_start"
    safe_dir = "ASC" if sort_dir.upper() == "ASC" else "DESC"

    count_sql = f"SELECT COUNT(*) FROM mv_inventory_forecast_monthly {where_sql}"
    data_sql = f"""
        SELECT
            item_no, loc, month_start, model_id,
            forecast, actual_demand, eom_qty_on_hand, dos,
            forecast_error, abs_error, bias_direction
        FROM mv_inventory_forecast_monthly
        {where_sql}
        ORDER BY {safe_sort} {safe_dir}
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(count_sql, params)
        total = int((cur.fetchone() or (0,))[0])
        cur.execute(data_sql, params + [limit, offset])
        rows = cur.fetchall()

    detail_rows = []
    for r in rows:
        eom = float(r[6]) if r[6] is not None else 0.0
        d = float(r[7]) if r[7] is not None else None
        et = "stockout" if eom <= 0 else ("excess" if d is not None and d > excess_dos_threshold else "normal")
        actual = float(r[5]) if r[5] is not None else 0.0
        fc_err = float(r[8]) if r[8] is not None else 0.0
        detail_rows.append({
            "item_no": r[0],
            "loc": r[1],
            "month": str(r[2]),
            "model_id": r[3],
            "forecast": round(float(r[4]), 2) if r[4] is not None else 0.0,
            "actual_demand": round(actual, 2),
            "eom_qty_on_hand": round(eom, 2),
            "dos": round(d, 1) if d is not None else None,
            "event_type": et,
            "forecast_error": round(fc_err, 2),
            "pct_error": round(fc_err / actual * 100, 1) if actual != 0 else None,
            "bias_direction": r[10] or "exact",
        })

    return {"total": total, "limit": limit, "offset": offset, "rows": detail_rows}
