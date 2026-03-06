"""Inventory Planning — IPfeature1: Demand Variability endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

router = APIRouter()


def _f(v: Any) -> float | None:
    return float(v) if v is not None else None


@router.get("/inv-planning/variability/summary")
def variability_summary(
    response: FastAPIResponse,
    abc_vol: str = Query(default="", max_length=10),
    cluster_assignment: str = Query(default="", max_length=120),
) -> dict:
    """Portfolio-level demand variability summary.

    Returns by-class breakdown, CV percentile stats, and top 10 most volatile DFUs.
    Cache: 300s.
    """
    set_cache(response, max_age=300)

    where_parts: list[str] = ["variability_class IS NOT NULL"]
    params: list[Any] = []

    if abc_vol.strip():
        where_parts.append("abc_vol = %s")
        params.append(abc_vol.strip().upper())
    if cluster_assignment.strip():
        where_parts.append("cluster_assignment ILIKE %s")
        params.append(f"%{cluster_assignment.strip()}%")

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    summary_sql = f"""
        SELECT
            COUNT(*)                                                    AS total_dfus,
            COUNT(*) FILTER (WHERE variability_class = 'low')          AS low_count,
            COUNT(*) FILTER (WHERE variability_class = 'medium')       AS medium_count,
            COUNT(*) FILTER (WHERE variability_class = 'high')         AS high_count,
            COUNT(*) FILTER (WHERE variability_class = 'lumpy')        AS lumpy_count,
            AVG(demand_cv)                                             AS avg_cv,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY demand_cv)   AS cv_p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY demand_cv)   AS cv_p50,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY demand_cv)   AS cv_p75,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY demand_cv)   AS cv_p95,
            AVG(intermittency_ratio)                                   AS avg_intermittency_ratio
        FROM dim_dfu
        {where_clause}
    """

    top_sql = f"""
        SELECT
            dmdunit AS item_no,
            loc,
            abc_vol,
            cluster_assignment,
            demand_mean,
            demand_std,
            demand_cv,
            demand_mad,
            intermittency_ratio,
            variability_class
        FROM dim_dfu
        {where_clause}
        ORDER BY demand_cv DESC NULLS LAST
        LIMIT 20
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

    by_class = {
        "low":    int(summary.get("low_count") or 0),
        "medium": int(summary.get("medium_count") or 0),
        "high":   int(summary.get("high_count") or 0),
        "lumpy":  int(summary.get("lumpy_count") or 0),
    }

    return {
        "total_dfus": int(summary.get("total_dfus") or 0),
        "by_class": by_class,
        "cv_percentiles": {
            "p25": _f(summary.get("cv_p25")),
            "p50": _f(summary.get("cv_p50")),
            "p75": _f(summary.get("cv_p75")),
            "p95": _f(summary.get("cv_p95")),
        },
        "avg_cv": _f(summary.get("avg_cv")),
        "avg_intermittency_ratio": _f(summary.get("avg_intermittency_ratio")),
        "top_volatile": [
            {**{k: (_f(v) if isinstance(v, float.__class__) else v) for k, v in r.items()}}
            for r in top_volatile
        ],
    }


@router.get("/inv-planning/variability/detail")
def variability_detail(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=10),
    variability_class: str = Query(default="", max_length=20),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="demand_cv", max_length=40),
    sort_dir: str = Query(default="desc", max_length=4),
) -> dict:
    """Paginated DFU demand variability detail table.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    allowed_sort = {
        "demand_cv", "demand_std", "demand_mean", "demand_mad",
        "intermittency_ratio", "demand_p90", "variability_class",
    }
    order_col = sort_by if sort_by in allowed_sort else "demand_cv"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list[Any] = []

    if item.strip():
        where_parts.append("dmdunit ILIKE %s")
        params.append(f"%{item.strip()}%")
    if location.strip():
        where_parts.append("loc ILIKE %s")
        params.append(f"%{location.strip()}%")
    if abc_vol.strip():
        where_parts.append("abc_vol = %s")
        params.append(abc_vol.strip().upper())
    if variability_class.strip():
        where_parts.append("variability_class = %s")
        params.append(variability_class.strip().lower())

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM dim_dfu {where_clause}"
    data_sql = f"""
        SELECT
            dmdunit          AS item_no,
            loc,
            abc_vol,
            cluster_assignment,
            demand_mean,
            demand_std,
            demand_cv,
            demand_mad,
            demand_p50,
            demand_p90,
            demand_skewness,
            demand_kurtosis,
            zero_demand_months,
            total_demand_months,
            intermittency_ratio,
            variability_class,
            demand_profile_ts
        FROM dim_dfu
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


@router.get("/inv-planning/variability/histogram")
def variability_histogram(
    response: FastAPIResponse,
    metric: str = Query(default="demand_cv", max_length=40),
    bins: int = Query(default=20, ge=5, le=50),
    abc_vol: str = Query(default="", max_length=10),
) -> dict:
    """Histogram data for a given variability metric.

    metric: demand_cv | demand_std | demand_mean | intermittency_ratio
    Cache: 300s.
    """
    set_cache(response, max_age=300)

    allowed_metrics = {"demand_cv", "demand_std", "demand_mean", "intermittency_ratio", "demand_p90"}
    col = metric if metric in allowed_metrics else "demand_cv"

    where_parts = [f"{col} IS NOT NULL"]
    params: list[Any] = []
    if abc_vol.strip():
        where_parts.append("abc_vol = %s")
        params.append(abc_vol.strip().upper())
    where_clause = "WHERE " + " AND ".join(where_parts)

    # Two-step: fetch bounds, then compute histogram buckets
    bounds_sql = f"SELECT MIN({col}) AS lo, MAX({col}) AS hi FROM dim_dfu {where_clause}"
    hist_sql = f"""
        SELECT
            width_bucket({col}, %s, %s + 0.000001, %s) AS bucket,
            COUNT(*) AS count
        FROM dim_dfu
        {where_clause}
        GROUP BY 1
        ORDER BY 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(bounds_sql, params)
            bounds_row = cur.fetchone()
            if not bounds_row or bounds_row[0] is None:
                return {"metric": col, "bins": []}
            lo, hi = float(bounds_row[0]), float(bounds_row[1])

            cur.execute(hist_sql, [lo, hi, bins, *params])
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
