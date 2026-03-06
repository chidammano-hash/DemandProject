"""Inventory Planning — IPfeature2: Lead Time Variability endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

router = APIRouter()


def _f(v: Any) -> float | None:
    return float(v) if v is not None else None


@router.get("/inv-planning/lead-time/summary")
def lt_summary(
    response: FastAPIResponse,
    abc_vol: str = Query(default="", max_length=10),
) -> dict:
    """Portfolio-level lead time variability summary.

    Returns by_class breakdown (stable/moderate/volatile) and top 10 most
    volatile item-locs by lt_cv.  Cache: 300s.
    """
    set_cache(response, max_age=300)

    where_parts: list[str] = ["p.lt_variability_class IS NOT NULL"]
    params: list[Any] = []

    if abc_vol.strip():
        where_parts.append("d.abc_vol = %s")
        params.append(abc_vol.strip().upper())
        from_clause = (
            "dim_item_lead_time_profile p "
            "LEFT JOIN dim_dfu d ON d.dmdunit = p.item_no AND d.loc = p.loc"
        )
    else:
        from_clause = "dim_item_lead_time_profile p"

    where_clause = "WHERE " + " AND ".join(where_parts)

    summary_sql = f"""
        SELECT
            COUNT(*)                                                          AS total_profiles,
            COUNT(*) FILTER (WHERE p.lt_variability_class = 'stable')        AS stable_count,
            COUNT(*) FILTER (WHERE p.lt_variability_class = 'moderate')      AS moderate_count,
            COUNT(*) FILTER (WHERE p.lt_variability_class = 'volatile')      AS volatile_count,
            AVG(p.lt_cv)                                                      AS avg_lt_cv,
            AVG(p.lt_mean_days)                                               AS avg_lt_mean_days,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY p.lt_cv)            AS lt_cv_p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY p.lt_cv)            AS lt_cv_p95
        FROM {from_clause}
        {where_clause}
    """

    top_sql = f"""
        SELECT
            p.item_no,
            p.loc,
            p.lt_mean_days,
            p.lt_std_days,
            p.lt_cv,
            p.lt_min_days,
            p.lt_max_days,
            p.observation_count,
            p.lt_variability_class
        FROM {from_clause}
        {where_clause}
        ORDER BY p.lt_cv DESC NULLS LAST
        LIMIT 10
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(summary_sql, params)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            summary = dict(zip(cols, row)) if row else {}

            cur.execute(top_sql, params)
            top_rows = cur.fetchall()
            top_cols = [d[0] for d in cur.description]
            top_volatile = [dict(zip(top_cols, r)) for r in top_rows]

    return {
        "total_profiles": int(summary.get("total_profiles") or 0),
        "by_class": {
            "stable":   int(summary.get("stable_count") or 0),
            "moderate": int(summary.get("moderate_count") or 0),
            "volatile": int(summary.get("volatile_count") or 0),
        },
        "avg_lt_cv":        _f(summary.get("avg_lt_cv")),
        "avg_lt_mean_days": _f(summary.get("avg_lt_mean_days")),
        "lt_cv_p50":        _f(summary.get("lt_cv_p50")),
        "lt_cv_p95":        _f(summary.get("lt_cv_p95")),
        "top_volatile": top_volatile,
    }


@router.get("/inv-planning/lead-time/profile")
def lt_profile(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    lt_variability_class: str = Query(default="", max_length=20),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="lt_cv", max_length=40),
    sort_dir: str = Query(default="desc", max_length=4),
) -> dict:
    """Paginated lead time profiles per item-location.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    allowed_sort = {"lt_cv", "lt_std_days", "lt_mean_days", "lt_max_days", "observation_count"}
    order_col = sort_by if sort_by in allowed_sort else "lt_cv"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list[Any] = []

    if item.strip():
        where_parts.append("item_no ILIKE %s")
        params.append(f"%{item.strip()}%")
    if location.strip():
        where_parts.append("loc ILIKE %s")
        params.append(f"%{location.strip()}%")
    if lt_variability_class.strip():
        where_parts.append("lt_variability_class = %s")
        params.append(lt_variability_class.strip().lower())

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM dim_item_lead_time_profile {where_clause}"
    data_sql = f"""
        SELECT
            item_no, loc,
            lt_mean_days, lt_std_days, lt_cv,
            lt_min_days, lt_max_days,
            lt_p25_days, lt_p50_days, lt_p75_days, lt_p95_days,
            observation_count, observation_months,
            lt_variability_class, computed_at
        FROM dim_item_lead_time_profile
        {where_clause}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()[0] or 0

            cur.execute(data_sql, [*params, limit, offset])
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

    return {
        "total": int(total),
        "rows": [dict(zip(cols, r)) for r in rows],
    }


@router.get("/inv-planning/lead-time/histogram")
def lt_histogram(
    response: FastAPIResponse,
    metric: str = Query(default="lt_cv", max_length=40),
    bins: int = Query(default=20, ge=5, le=50),
) -> dict:
    """Histogram data for a given LT variability metric.

    metric: lt_cv | lt_std_days | lt_mean_days | lt_max_days | lt_p95_days
    Cache: 300s.
    """
    set_cache(response, max_age=300)

    allowed_metrics = {"lt_cv", "lt_std_days", "lt_mean_days", "lt_max_days", "lt_p95_days"}
    col = metric if metric in allowed_metrics else "lt_cv"

    where_clause = f"WHERE {col} IS NOT NULL"

    bounds_sql = (
        f"SELECT MIN({col}) AS lo, MAX({col}) AS hi "
        f"FROM dim_item_lead_time_profile {where_clause}"
    )
    hist_sql = f"""
        SELECT
            width_bucket({col}, %s, %s + 0.000001, %s) AS bucket,
            COUNT(*) AS count
        FROM dim_item_lead_time_profile
        {where_clause}
        GROUP BY 1
        ORDER BY 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(bounds_sql)
            bounds_row = cur.fetchone()
            if not bounds_row or bounds_row[0] is None:
                return {"metric": col, "bins": []}
            lo, hi = float(bounds_row[0]), float(bounds_row[1])

            cur.execute(hist_sql, [lo, hi, bins])
            rows = cur.fetchall()

    bin_width = (hi - lo) / bins if bins > 0 else 0
    result_bins = [
        {
            "bucket": int(r[0]),
            "bin_start": round(lo + (int(r[0]) - 1) * bin_width, 6),
            "bin_end": round(lo + int(r[0]) * bin_width, 6),
            "count": int(r[1]),
        }
        for r in rows
    ]

    return {
        "metric": col,
        "bins": result_bins,
    }
