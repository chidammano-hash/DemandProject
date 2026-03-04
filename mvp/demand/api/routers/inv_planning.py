"""Inventory Planning endpoints — IPfeature1: Demand Variability.

Router mounted at /inv-planning in api/main.py.
Additional IPfeatures will add endpoints to this same router.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

router = APIRouter()


# ---------------------------------------------------------------------------
# IPfeature1: Demand Variability endpoints
# ---------------------------------------------------------------------------

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

    # Coerce decimals to float
    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

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


# ---------------------------------------------------------------------------
# IPfeature2: Lead Time Variability endpoints
# ---------------------------------------------------------------------------

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

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

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


# ---------------------------------------------------------------------------
# IPfeature4 — EOQ & Cycle Stock
# ---------------------------------------------------------------------------

@router.get("/eoq/summary")
async def eoq_summary(abc_vol: str | None = None):
    """Portfolio EOQ summary with by-ABC breakdown."""
    pool = _get_pool()
    wheres = []
    params: list = []
    if abc_vol:
        wheres.append("abc_vol = %s")
        params.append(abc_vol)
    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    summary_sql = f"""
        SELECT
            COUNT(*)                        AS total_dfus,
            AVG(effective_eoq)              AS avg_effective_eoq,
            SUM(eoq_cycle_stock)            AS total_cycle_stock,
            AVG(order_frequency)            AS avg_order_frequency,
            SUM(total_annual_cost)          AS total_annual_cost
        FROM fact_eoq_targets
        {where_clause}
    """
    abc_sql = f"""
        SELECT
            COALESCE(abc_vol, 'Unknown')    AS abc_vol,
            COUNT(*)                        AS count,
            AVG(effective_eoq)              AS avg_eoq,
            SUM(eoq_cycle_stock)            AS total_cycle_stock,
            SUM(total_annual_cost)          AS total_annual_cost,
            AVG(order_frequency)            AS avg_order_frequency
        FROM fact_eoq_targets
        {where_clause}
        GROUP BY abc_vol
        ORDER BY abc_vol
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(summary_sql, params)
            row = cur.fetchone()
            cur.execute(abc_sql, params)
            abc_rows = cur.fetchall()

    if not row or row[0] == 0:
        return {
            "total_dfus": 0,
            "avg_effective_eoq": None,
            "total_cycle_stock": None,
            "avg_order_frequency": None,
            "total_annual_cost": None,
            "by_abc": [],
        }

    by_abc = [
        {
            "abc_vol": r[0],
            "count": int(r[1]),
            "avg_eoq": float(r[2]) if r[2] is not None else None,
            "total_cycle_stock": float(r[3]) if r[3] is not None else None,
            "total_annual_cost": float(r[4]) if r[4] is not None else None,
            "avg_order_frequency": float(r[5]) if r[5] is not None else None,
        }
        for r in abc_rows
    ]

    return {
        "total_dfus": int(row[0]),
        "avg_effective_eoq": float(row[1]) if row[1] is not None else None,
        "total_cycle_stock": float(row[2]) if row[2] is not None else None,
        "avg_order_frequency": float(row[3]) if row[3] is not None else None,
        "total_annual_cost": float(row[4]) if row[4] is not None else None,
        "by_abc": by_abc,
    }


_EOQ_SORT_COLS = {
    "effective_eoq", "eoq", "eoq_cycle_stock", "order_frequency",
    "total_annual_cost", "annual_holding_cost", "annual_order_cost",
    "demand_mean_monthly", "annual_demand",
}


@router.get("/eoq/detail")
async def eoq_detail(
    item: str | None = None,
    loc: str | None = None,
    abc_vol: str | None = None,
    sort_by: str = "total_annual_cost",
    sort_dir: str = "desc",
    limit: int = 50,
    offset: int = 0,
):
    """Paginated EOQ detail per item-location."""
    pool = _get_pool()
    col = sort_by if sort_by in _EOQ_SORT_COLS else "total_annual_cost"
    direction = "DESC" if sort_dir.lower() != "asc" else "ASC"

    wheres = []
    params: list = []
    if item:
        wheres.append("item_no ILIKE %s")
        params.append(f"%{item}%")
    if loc:
        wheres.append("loc ILIKE %s")
        params.append(f"%{loc}%")
    if abc_vol:
        wheres.append("abc_vol = %s")
        params.append(abc_vol)
    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    count_sql = f"SELECT COUNT(*) FROM fact_eoq_targets {where_clause}"
    rows_sql = f"""
        SELECT
            item_no, loc, abc_vol,
            demand_mean_monthly, annual_demand,
            ordering_cost, holding_cost_pct, unit_cost, moq,
            eoq, effective_eoq, eoq_cycle_stock, order_frequency,
            annual_holding_cost, annual_order_cost, total_annual_cost,
            computed_at
        FROM fact_eoq_targets
        {where_clause}
        ORDER BY {col} {direction} NULLS LAST
        LIMIT %s OFFSET %s
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = (cur.fetchone() or (0,))[0]
            cur.execute(rows_sql, params + [limit, offset])
            rows = cur.fetchall()

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "rows": [
            {
                "item_no": r[0], "loc": r[1], "abc_vol": r[2],
                "demand_mean_monthly": float(r[3]) if r[3] is not None else None,
                "annual_demand": float(r[4]) if r[4] is not None else None,
                "ordering_cost": float(r[5]) if r[5] is not None else None,
                "holding_cost_pct": float(r[6]) if r[6] is not None else None,
                "unit_cost": float(r[7]) if r[7] is not None else None,
                "moq": float(r[8]) if r[8] is not None else None,
                "eoq": float(r[9]) if r[9] is not None else None,
                "effective_eoq": float(r[10]) if r[10] is not None else None,
                "eoq_cycle_stock": float(r[11]) if r[11] is not None else None,
                "order_frequency": float(r[12]) if r[12] is not None else None,
                "annual_holding_cost": float(r[13]) if r[13] is not None else None,
                "annual_order_cost": float(r[14]) if r[14] is not None else None,
                "total_annual_cost": float(r[15]) if r[15] is not None else None,
                "computed_at": r[16].isoformat() if r[16] else None,
            }
            for r in rows
        ],
    }


@router.get("/eoq/sensitivity")
async def eoq_sensitivity(item: str | None = None, loc: str | None = None):
    """EOQ sensitivity curve: how EOQ changes as ordering_cost varies."""
    import yaml
    import os
    from scripts.compute_eoq import sensitivity_curve

    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "eoq_config.yaml")
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    pool = _get_pool()
    if item and loc:
        sql = "SELECT demand_mean_monthly FROM fact_eoq_targets WHERE item_no = %s AND loc = %s LIMIT 1"
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [item, loc])
                row = cur.fetchone()
        avg_demand = float(row[0]) if row and row[0] else 100.0
    else:
        sql = "SELECT AVG(demand_mean_monthly) FROM fact_eoq_targets WHERE demand_mean_monthly > 0"
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
        avg_demand = float(row[0]) if row and row[0] else 100.0

    curve = sensitivity_curve(avg_demand, config)
    return {
        "item_no": item,
        "loc": loc,
        "avg_demand_monthly": avg_demand,
        "curve": curve,
    }
