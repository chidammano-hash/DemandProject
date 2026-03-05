"""Inventory Planning endpoints — IPfeature1: Demand Variability.

Router mounted at /inv-planning in api/main.py.
Additional IPfeatures will add endpoints to this same router.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from api.auth import require_api_key
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

@router.get("/inv-planning/eoq/summary")
async def eoq_summary(abc_vol: str | None = None):
    """Portfolio EOQ summary with by-ABC breakdown."""
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

    with get_conn() as conn:
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


@router.get("/inv-planning/eoq/detail")
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

    with get_conn() as conn:
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


@router.get("/inv-planning/eoq/sensitivity")
async def eoq_sensitivity(item: str | None = None, loc: str | None = None):
    """EOQ sensitivity curve: how EOQ changes as ordering_cost varies."""
    import yaml
    import os
    from scripts.compute_eoq import sensitivity_curve

    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "eoq_config.yaml")
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    if item and loc:
        sql = "SELECT demand_mean_monthly FROM fact_eoq_targets WHERE item_no = %s AND loc = %s LIMIT 1"
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [item, loc])
                row = cur.fetchone()
        avg_demand = float(row[0]) if row and row[0] else 100.0
    else:
        sql = "SELECT AVG(demand_mean_monthly) FROM fact_eoq_targets WHERE demand_mean_monthly > 0"
        with get_conn() as conn:
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


# ---------------------------------------------------------------------------
# IPfeature5 — Replenishment Policy Management
# ---------------------------------------------------------------------------

class PolicyCreateBody(BaseModel):
    policy_id: str
    policy_name: str
    policy_type: str
    segment: str | None = None
    review_cycle_days: int | None = None
    service_level: float | None = None
    use_eoq: bool = True
    use_safety_stock: bool = True
    notes: str | None = None


class PolicyUpdateBody(BaseModel):
    policy_name: str | None = None
    policy_type: str | None = None
    segment: str | None = None
    review_cycle_days: int | None = None
    service_level: float | None = None
    use_eoq: bool | None = None
    use_safety_stock: bool | None = None
    active: bool | None = None
    notes: str | None = None


class PolicyAssignBody(BaseModel):
    # Individual assignment
    item_no: str | None = None
    loc: str | None = None
    policy_id: str | None = None
    override_reason: str | None = None
    # Bulk by segment
    segment: str | None = None


_VALID_POLICY_TYPES = {"continuous_rop", "periodic_review", "min_max", "manual"}


@router.get("/inv-planning/policies")
def get_policies(response: FastAPIResponse) -> dict:
    """List all active policies with DFU assignment counts.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    sql = """
        SELECT
            p.policy_id,
            p.policy_name,
            p.policy_type,
            p.segment,
            p.review_cycle_days,
            p.service_level,
            p.use_eoq,
            p.use_safety_stock,
            p.active,
            COUNT(a.item_no) AS dfu_count
        FROM dim_replenishment_policy p
        LEFT JOIN fact_dfu_policy_assignment a ON a.policy_id = p.policy_id
        GROUP BY p.policy_sk, p.policy_id, p.policy_name, p.policy_type,
                 p.segment, p.review_cycle_days, p.service_level,
                 p.use_eoq, p.use_safety_stock, p.active
        ORDER BY p.policy_id
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()

    def _row(r: tuple) -> dict:
        d = dict(zip(cols, r))
        return {
            "policy_id":         d["policy_id"],
            "policy_name":       d["policy_name"],
            "policy_type":       d["policy_type"],
            "segment":           d["segment"],
            "review_cycle_days": d["review_cycle_days"],
            "service_level":     float(d["service_level"]) if d["service_level"] is not None else None,
            "use_eoq":           bool(d["use_eoq"]),
            "use_safety_stock":  bool(d["use_safety_stock"]),
            "active":            bool(d["active"]),
            "dfu_count":         int(d["dfu_count"]),
        }

    return {"policies": [_row(r) for r in rows]}


@router.post("/inv-planning/policies", status_code=201, dependencies=[Depends(require_api_key)])
def create_policy(body: PolicyCreateBody) -> dict:
    """Create a new replenishment policy. Auth required."""
    if body.policy_type not in _VALID_POLICY_TYPES:
        raise HTTPException(status_code=422, detail=f"policy_type must be one of {sorted(_VALID_POLICY_TYPES)}")

    sql = """
        INSERT INTO dim_replenishment_policy
            (policy_id, policy_name, policy_type, segment, review_cycle_days,
             service_level, use_eoq, use_safety_stock, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING policy_id, policy_name, policy_type, segment,
                  review_cycle_days, service_level, use_eoq, use_safety_stock, active, notes
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, (
                    body.policy_id, body.policy_name, body.policy_type,
                    body.segment, body.review_cycle_days, body.service_level,
                    body.use_eoq, body.use_safety_stock, body.notes,
                ))
                conn.commit()
                row = cur.fetchone()
                cols = [d[0] for d in cur.description]
            except Exception as exc:
                conn.rollback()
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    d = dict(zip(cols, row))
    return {
        "policy_id":         d["policy_id"],
        "policy_name":       d["policy_name"],
        "policy_type":       d["policy_type"],
        "segment":           d["segment"],
        "review_cycle_days": d["review_cycle_days"],
        "service_level":     float(d["service_level"]) if d["service_level"] is not None else None,
        "use_eoq":           bool(d["use_eoq"]),
        "use_safety_stock":  bool(d["use_safety_stock"]),
        "active":            bool(d["active"]),
        "notes":             d["notes"],
        "dfu_count":         0,
    }


@router.put("/inv-planning/policies/{policy_id}", dependencies=[Depends(require_api_key)])
def update_policy(policy_id: str, body: PolicyUpdateBody) -> dict:
    """Update an existing policy by policy_id. Auth required."""
    updates: list[str] = ["modified_ts = NOW()"]
    params: list[Any] = []

    if body.policy_name is not None:
        updates.append("policy_name = %s"); params.append(body.policy_name)
    if body.policy_type is not None:
        if body.policy_type not in _VALID_POLICY_TYPES:
            raise HTTPException(status_code=422, detail=f"policy_type must be one of {sorted(_VALID_POLICY_TYPES)}")
        updates.append("policy_type = %s"); params.append(body.policy_type)
    if body.segment is not None:
        updates.append("segment = %s"); params.append(body.segment)
    if body.review_cycle_days is not None:
        updates.append("review_cycle_days = %s"); params.append(body.review_cycle_days)
    if body.service_level is not None:
        updates.append("service_level = %s"); params.append(body.service_level)
    if body.use_eoq is not None:
        updates.append("use_eoq = %s"); params.append(body.use_eoq)
    if body.use_safety_stock is not None:
        updates.append("use_safety_stock = %s"); params.append(body.use_safety_stock)
    if body.active is not None:
        updates.append("active = %s"); params.append(body.active)
    if body.notes is not None:
        updates.append("notes = %s"); params.append(body.notes)

    params.append(policy_id)
    sql = f"""
        UPDATE dim_replenishment_policy
        SET {', '.join(updates)}
        WHERE policy_id = %s
        RETURNING policy_id, policy_name, policy_type, segment,
                  review_cycle_days, service_level, use_eoq, use_safety_stock, active, notes
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
            cols = [d[0] for d in cur.description]

    # Count DFUs assigned
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fact_dfu_policy_assignment WHERE policy_id = %s", [policy_id])
            dfu_count = cur.fetchone()[0] or 0

    d = dict(zip(cols, row))
    return {
        "policy_id":         d["policy_id"],
        "policy_name":       d["policy_name"],
        "policy_type":       d["policy_type"],
        "segment":           d["segment"],
        "review_cycle_days": d["review_cycle_days"],
        "service_level":     float(d["service_level"]) if d["service_level"] is not None else None,
        "use_eoq":           bool(d["use_eoq"]),
        "use_safety_stock":  bool(d["use_safety_stock"]),
        "active":            bool(d["active"]),
        "notes":             d["notes"],
        "dfu_count":         int(dfu_count),
    }


@router.get("/inv-planning/policy-assignments")
def get_policy_assignments(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    policy_id: str = Query(default="", max_length=80),
    assigned_by: str = Query(default="", max_length=20),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Paginated DFU policy assignments.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    where_parts: list[str] = []
    params: list[Any] = []

    if item.strip():
        where_parts.append("a.item_no ILIKE %s")
        params.append(f"%{item.strip()}%")
    if location.strip():
        where_parts.append("a.loc ILIKE %s")
        params.append(f"%{location.strip()}%")
    if policy_id.strip():
        where_parts.append("a.policy_id = %s")
        params.append(policy_id.strip())
    if assigned_by.strip():
        where_parts.append("a.assigned_by = %s")
        params.append(assigned_by.strip())

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM fact_dfu_policy_assignment a {where_clause}"
    data_sql = f"""
        SELECT
            a.item_no,
            a.loc,
            a.policy_id,
            p.policy_name,
            p.policy_type,
            a.override_reason,
            a.assigned_by,
            a.effective_date
        FROM fact_dfu_policy_assignment a
        JOIN dim_replenishment_policy p ON p.policy_id = a.policy_id
        {where_clause}
        ORDER BY a.item_no, a.loc
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
        "rows": [
            {
                "item_no":         r[0],
                "loc":             r[1],
                "policy_id":       r[2],
                "policy_name":     r[3],
                "policy_type":     r[4],
                "override_reason": r[5],
                "assigned_by":     r[6],
                "effective_date":  r[7].isoformat() if r[7] else None,
            }
            for r in rows
        ],
    }


@router.post("/inv-planning/policy-assignments/assign", dependencies=[Depends(require_api_key)])
def assign_policy(body: PolicyAssignBody) -> dict:
    """Assign a policy to one DFU (individual) or all DFUs in a segment (bulk).

    Individual: { item_no, loc, policy_id, override_reason }
    Bulk:       { segment, policy_id }
    Auth required.
    """
    from datetime import date

    effective_date = date.today()
    assigned_count = 0
    failed_count = 0
    already_assigned_count = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Verify policy exists
            if body.policy_id:
                cur.execute("SELECT 1 FROM dim_replenishment_policy WHERE policy_id = %s", [body.policy_id])
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail=f"Policy '{body.policy_id}' not found")

            if body.item_no and body.loc and body.policy_id:
                # Individual assignment
                upsert_sql = """
                    INSERT INTO fact_dfu_policy_assignment
                        (item_no, loc, policy_id, override_reason, assigned_by, effective_date)
                    VALUES (%s, %s, %s, %s, 'manual', %s)
                    ON CONFLICT (item_no, loc) DO UPDATE SET
                        policy_id      = EXCLUDED.policy_id,
                        override_reason= EXCLUDED.override_reason,
                        assigned_by    = 'manual',
                        effective_date = EXCLUDED.effective_date,
                        modified_ts    = NOW()
                """
                try:
                    cur.execute(upsert_sql, (
                        body.item_no, body.loc, body.policy_id,
                        body.override_reason, effective_date,
                    ))
                    assigned_count = 1
                except Exception:
                    failed_count = 1

            elif body.segment and body.policy_id:
                # Bulk assignment by segment — assign all DFUs with matching abc_vol
                # or variability_class matching the segment
                bulk_sql = """
                    INSERT INTO fact_dfu_policy_assignment
                        (item_no, loc, policy_id, assigned_by, effective_date)
                    SELECT
                        d.dmdunit,
                        d.loc,
                        %s,
                        'system',
                        %s
                    FROM dim_dfu d
                    WHERE d.abc_vol = %s OR d.variability_class = %s
                    ON CONFLICT (item_no, loc) DO UPDATE SET
                        policy_id      = EXCLUDED.policy_id,
                        assigned_by    = 'system',
                        effective_date = EXCLUDED.effective_date,
                        modified_ts    = NOW()
                    WHERE fact_dfu_policy_assignment.assigned_by = 'system'
                """
                try:
                    cur.execute(bulk_sql, (
                        body.policy_id, effective_date,
                        body.segment.upper(), body.segment.lower(),
                    ))
                    assigned_count = cur.rowcount
                except Exception:
                    failed_count = 1
            else:
                raise HTTPException(
                    status_code=422,
                    detail="Provide either (item_no + loc + policy_id) or (segment + policy_id)",
                )

        conn.commit()

    return {
        "assigned_count":         assigned_count,
        "failed_count":           failed_count,
        "already_assigned_count": already_assigned_count,
    }


@router.get("/inv-planning/policy-assignments/compliance")
def get_policy_compliance(response: FastAPIResponse) -> dict:
    """Portfolio-level policy compliance metrics.

    Returns: total_dfus, assigned_count, unassigned_count, assignment_pct,
    and per-policy breakdown with avg_dos where available.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Total DFUs
            cur.execute("SELECT COUNT(*) FROM dim_dfu")
            total_dfus = cur.fetchone()[0] or 0

            # Assigned DFUs
            cur.execute("SELECT COUNT(DISTINCT item_no || '|' || loc) FROM fact_dfu_policy_assignment")
            assigned_count = cur.fetchone()[0] or 0

            # Per-policy breakdown
            by_policy_sql = """
                SELECT
                    p.policy_id,
                    p.policy_name,
                    p.policy_type,
                    COUNT(a.item_no)       AS dfu_count,
                    AVG(inv.dos)           AS avg_dos
                FROM dim_replenishment_policy p
                LEFT JOIN fact_dfu_policy_assignment a ON a.policy_id = p.policy_id
                LEFT JOIN (
                    SELECT item_no, loc,
                           CASE WHEN AVG(daily_sales) > 0
                                THEN AVG(eom_on_hand) / AVG(daily_sales)
                                ELSE NULL
                           END AS dos
                    FROM agg_inventory_monthly
                    GROUP BY item_no, loc
                ) inv ON inv.item_no = a.item_no AND inv.loc = a.loc
                GROUP BY p.policy_id, p.policy_name, p.policy_type
                ORDER BY p.policy_id
            """
            try:
                cur.execute(by_policy_sql)
                policy_rows = cur.fetchall()
                policy_cols = [d[0] for d in cur.description]
            except Exception:
                # If agg_inventory_monthly doesn't exist, fall back to simpler query
                cur.execute("""
                    SELECT
                        p.policy_id,
                        p.policy_name,
                        p.policy_type,
                        COUNT(a.item_no) AS dfu_count,
                        NULL             AS avg_dos
                    FROM dim_replenishment_policy p
                    LEFT JOIN fact_dfu_policy_assignment a ON a.policy_id = p.policy_id
                    GROUP BY p.policy_id, p.policy_name, p.policy_type
                    ORDER BY p.policy_id
                """)
                policy_rows = cur.fetchall()
                policy_cols = [d[0] for d in cur.description]

    unassigned_count = int(total_dfus) - int(assigned_count)
    assignment_pct = (float(assigned_count) / float(total_dfus) * 100.0) if total_dfus > 0 else 0.0

    by_policy: dict[str, dict] = {}
    for row in policy_rows:
        d = dict(zip(policy_cols, row))
        by_policy[d["policy_id"]] = {
            "policy_name":    d["policy_name"],
            "policy_type":    d["policy_type"],
            "dfu_count":      int(d["dfu_count"]),
            "below_ss_pct":   None,   # IPfeature3 not yet implemented
            "avg_ss_coverage": None,  # IPfeature3 not yet implemented
            "avg_dos":        float(d["avg_dos"]) if d["avg_dos"] is not None else None,
        }

    return {
        "total_dfus":       int(total_dfus),
        "assigned_count":   int(assigned_count),
        "unassigned_count": int(unassigned_count),
        "assignment_pct":   round(assignment_pct, 2),
        "by_policy":        by_policy,
    }


# ---------------------------------------------------------------------------
# IPfeature6 — Inventory Health Score endpoints
# ---------------------------------------------------------------------------

@router.get("/inv-planning/health/summary")
def get_health_summary(
    abc_vol:            Optional[str] = None,
    cluster_assignment: Optional[str] = None,
    region:             Optional[str] = None,
    variability_class:  Optional[str] = None,
):
    """Aggregate health score summary with tier breakdown."""
    where_clauses: list[str] = []
    params: list = []

    if abc_vol:
        where_clauses.append(f"abc_vol = ${len(params)+1}")
        params.append(abc_vol)
    if cluster_assignment:
        where_clauses.append(f"cluster_assignment = ${len(params)+1}")
        params.append(cluster_assignment)
    if region:
        where_clauses.append(f"region = ${len(params)+1}")
        params.append(region)
    if variability_class:
        where_clauses.append(f"variability_class = ${len(params)+1}")
        params.append(variability_class)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    summary_sql = f"""
        SELECT
            COUNT(*)                                                    AS total_dfus,
            AVG(health_score)                                           AS avg_health_score,
            SUM(CASE WHEN health_tier = 'healthy'  THEN 1 ELSE 0 END)  AS healthy_count,
            SUM(CASE WHEN health_tier = 'monitor'  THEN 1 ELSE 0 END)  AS monitor_count,
            SUM(CASE WHEN health_tier = 'at_risk'  THEN 1 ELSE 0 END)  AS at_risk_count,
            SUM(CASE WHEN health_tier = 'critical' THEN 1 ELSE 0 END)  AS critical_count,
            AVG(score_ss_coverage)                                      AS avg_score_ss,
            AVG(score_dos_target)                                       AS avg_score_dos,
            AVG(score_stockout_risk)                                    AS avg_score_stockout,
            AVG(score_forecast_accuracy)                                AS avg_score_forecast
        FROM mv_inventory_health_score
        {where_sql}
    """

    histogram_sql = f"""
        SELECT
            CASE
                WHEN health_score < 20  THEN '0-19'
                WHEN health_score < 40  THEN '20-39'
                WHEN health_score < 60  THEN '40-59'
                WHEN health_score < 80  THEN '60-79'
                ELSE '80-100'
            END AS bucket,
            COUNT(*) AS count
        FROM mv_inventory_health_score
        {where_sql}
        GROUP BY 1
        ORDER BY 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(summary_sql, params)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            summary = dict(zip(cols, row)) if row else {}

            cur.execute(histogram_sql, params)
            hist_rows = cur.fetchall()

    total_dfus = int(summary.get("total_dfus") or 0)

    def _pct(n):
        v = summary.get(n)
        return int(v) if v is not None else 0

    by_tier = {
        "healthy":  _pct("healthy_count"),
        "monitor":  _pct("monitor_count"),
        "at_risk":  _pct("at_risk_count"),
        "critical": _pct("critical_count"),
    }

    def _fval(n):
        v = summary.get(n)
        return round(float(v), 2) if v is not None else None

    score_histogram = [{"bucket": r[0], "count": int(r[1])} for r in hist_rows]

    return {
        "total_dfus":      total_dfus,
        "by_tier":         by_tier,
        "avg_health_score": _fval("avg_health_score"),
        "component_avgs": {
            "ss_coverage":       _fval("avg_score_ss"),
            "dos_target":        _fval("avg_score_dos"),
            "stockout_risk":     _fval("avg_score_stockout"),
            "forecast_accuracy": _fval("avg_score_forecast"),
        },
        "score_histogram": score_histogram,
    }


@router.get("/inv-planning/health/detail")
def get_health_detail(
    item:               Optional[str] = None,
    location:           Optional[str] = None,
    health_tier:        Optional[str] = None,
    abc_vol:            Optional[str] = None,
    cluster_assignment: Optional[str] = None,
    variability_class:  Optional[str] = None,
    limit:  int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by:  str = Query("health_score"),
    sort_dir: str = Query("asc"),
):
    """Paginated health score detail rows with filtering and sorting."""
    allowed_sort = {
        "health_score", "health_tier", "item_no", "loc", "abc_vol",
        "variability_class", "current_dos", "recent_wape",
        "score_ss_coverage", "score_dos_target", "score_stockout_risk",
        "score_forecast_accuracy",
    }
    if sort_by not in allowed_sort:
        sort_by = "health_score"
    sort_dir = "DESC" if sort_dir.upper() == "DESC" else "ASC"

    where_clauses: list[str] = []
    params: list = []

    if item:
        where_clauses.append(f"item_no ILIKE ${len(params)+1}")
        params.append(f"%{item}%")
    if location:
        where_clauses.append(f"loc ILIKE ${len(params)+1}")
        params.append(f"%{location}%")
    if health_tier:
        where_clauses.append(f"health_tier = ${len(params)+1}")
        params.append(health_tier)
    if abc_vol:
        where_clauses.append(f"abc_vol = ${len(params)+1}")
        params.append(abc_vol)
    if cluster_assignment:
        where_clauses.append(f"cluster_assignment = ${len(params)+1}")
        params.append(cluster_assignment)
    if variability_class:
        where_clauses.append(f"variability_class = ${len(params)+1}")
        params.append(variability_class)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    count_sql = f"SELECT COUNT(*) FROM mv_inventory_health_score {where_sql}"
    rows_sql = f"""
        SELECT
            item_no, loc, abc_vol, variability_class, cluster_assignment,
            health_score, health_tier,
            score_ss_coverage, score_dos_target, score_stockout_risk, score_forecast_accuracy,
            ss_coverage, current_dos, target_dos_min, target_dos_max, is_below_ss,
            recent_wape, stockout_count_3m
        FROM mv_inventory_health_score
        {where_sql}
        ORDER BY {sort_by} {sort_dir}
        LIMIT ${len(params)+1} OFFSET ${len(params)+2}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()[0] or 0

            cur.execute(rows_sql, params + [limit, offset])
            raw_rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

    def _coerce(k, v):
        if v is None:
            return None
        if k in ("health_score", "score_ss_coverage", "score_dos_target",
                  "score_stockout_risk", "score_forecast_accuracy", "stockout_count_3m"):
            return int(v)
        if k in ("ss_coverage", "current_dos", "target_dos_min", "target_dos_max", "recent_wape"):
            return round(float(v), 4)
        return v

    rows = [
        {k: _coerce(k, v) for k, v in zip(cols, r)}
        for r in raw_rows
    ]

    return {"total": int(total), "rows": rows}


@router.get("/inv-planning/health/heatmap")
def get_health_heatmap(
    group_x: str = Query("abc_vol", description="Column for X-axis grouping"),
    group_y: str = Query("variability_class", description="Column for Y-axis grouping"),
):
    """Average health score heatmap grouped by two dimensions."""
    allowed_groups = {"abc_vol", "variability_class", "cluster_assignment", "region", "health_tier"}
    if group_x not in allowed_groups:
        group_x = "abc_vol"
    if group_y not in allowed_groups:
        group_y = "variability_class"

    sql = f"""
        SELECT
            COALESCE({group_x}::TEXT, 'Unknown') AS x_label,
            COALESCE({group_y}::TEXT, 'Unknown') AS y_label,
            AVG(health_score)                     AS avg_health_score,
            COUNT(*)                              AS count,
            SUM(CASE WHEN health_tier = 'critical' THEN 1 ELSE 0 END) AS critical_count
        FROM mv_inventory_health_score
        GROUP BY 1, 2
        ORDER BY 1, 2
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            raw_rows = cur.fetchall()

    cells = [
        {
            "x":                r[0],
            "y":                r[1],
            "avg_health_score": round(float(r[2]), 2) if r[2] is not None else None,
            "count":            int(r[3]),
            "critical_count":   int(r[4]),
        }
        for r in raw_rows
    ]

    x_labels = sorted(set(c["x"] for c in cells))
    y_labels = sorted(set(c["y"] for c in cells))

    return {"x_labels": x_labels, "y_labels": y_labels, "cells": cells}


# ---------------------------------------------------------------------------
# IPfeature7 — Exception Queue & Replenishment Recommendations
# ---------------------------------------------------------------------------

_VALID_EXCEPTION_TYPES = {
    "below_rop", "below_rop_critical", "below_ss", "stockout", "excess", "zero_velocity"
}
_VALID_SEVERITIES = {"critical", "high", "medium", "low"}
_VALID_STATUSES = {"open", "acknowledged", "ordered", "resolved"}
_EXCEPTION_SORT_COLS = {
    "severity", "exception_date", "recommended_order_by",
    "current_qty_on_hand", "item_no", "loc",
}
_SEVERITY_ORDER = "CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END"


class ExceptionAcknowledgeBody(BaseModel):
    acknowledged_by: str
    notes: Optional[str] = None


class ExceptionStatusBody(BaseModel):
    status: str
    notes: Optional[str] = None


@router.get("/inv-planning/exceptions")
def get_exceptions(
    response: FastAPIResponse,
    exception_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: str = "open",
    item: Optional[str] = None,
    location: Optional[str] = None,
    sort_by: str = "severity",
    sort_dir: str = "asc",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Paginated exception queue with filters.

    Default: open exceptions sorted by severity (critical first), then exception_date asc.
    Cache: 60s.
    """
    set_cache(response, max_age=60)

    wheres: list[str] = []
    params: list = []

    if status and status in _VALID_STATUSES:
        wheres.append("status = %s")
        params.append(status)
    if exception_type and exception_type in _VALID_EXCEPTION_TYPES:
        wheres.append("exception_type = %s")
        params.append(exception_type)
    if severity and severity in _VALID_SEVERITIES:
        wheres.append("severity = %s")
        params.append(severity)
    if item:
        wheres.append("item_no ILIKE %s")
        params.append(f"%{item}%")
    if location:
        wheres.append("loc ILIKE %s")
        params.append(f"%{location}%")

    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    col = sort_by if sort_by in _EXCEPTION_SORT_COLS else "severity"
    direction = "ASC" if sort_dir.lower() == "asc" else "DESC"

    if col == "severity":
        order_clause = f"{_SEVERITY_ORDER} {direction}, exception_date ASC"
    else:
        order_clause = f"{col} {direction} NULLS LAST"

    count_sql = f"SELECT COUNT(*) FROM fact_replenishment_exceptions {where_clause}"
    rows_sql = f"""
        SELECT
            exception_id, item_no, loc, exception_date, exception_type, severity,
            current_qty_on_hand, current_dos, ss_combined, reorder_point,
            recommended_order_qty, recommended_order_by, expected_receipt_date,
            estimated_order_value, policy_id, status, acknowledged_by, notes
        FROM fact_replenishment_exceptions
        {where_clause}
        ORDER BY {order_clause}
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = (cur.fetchone() or (0,))[0]
            cur.execute(rows_sql, params + [limit, offset])
            rows = cur.fetchall()

    def _f(v):
        return float(v) if v is not None else None

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "rows": [
            {
                "exception_id":           r[0],
                "item_no":                r[1],
                "loc":                    r[2],
                "exception_date":         r[3].isoformat() if r[3] else None,
                "exception_type":         r[4],
                "severity":               r[5],
                "current_qty_on_hand":    _f(r[6]),
                "current_dos":            _f(r[7]),
                "ss_combined":            _f(r[8]),
                "reorder_point":          _f(r[9]),
                "recommended_order_qty":  _f(r[10]),
                "recommended_order_by":   r[11].isoformat() if r[11] else None,
                "expected_receipt_date":  r[12].isoformat() if r[12] else None,
                "estimated_order_value":  _f(r[13]),
                "policy_id":              r[14],
                "status":                 r[15],
                "acknowledged_by":        r[16],
                "notes":                  r[17],
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/exceptions/summary")
def get_exception_summary(
    response: FastAPIResponse,
    status: str = "open",
) -> dict:
    """Aggregate exception counts by type and severity.

    Cache: 60s.
    """
    set_cache(response, max_age=60)

    where = "WHERE status = %s" if status in _VALID_STATUSES else ""
    params_status = [status] if where else []

    sql = f"""
        SELECT
            COUNT(*)                                                               AS open_count,
            SUM(CASE WHEN exception_type = 'below_rop'          THEN 1 ELSE 0 END) AS below_rop,
            SUM(CASE WHEN exception_type = 'below_rop_critical'  THEN 1 ELSE 0 END) AS below_rop_critical,
            SUM(CASE WHEN exception_type = 'below_ss'           THEN 1 ELSE 0 END) AS below_ss,
            SUM(CASE WHEN exception_type = 'stockout'           THEN 1 ELSE 0 END) AS stockout,
            SUM(CASE WHEN exception_type = 'excess'             THEN 1 ELSE 0 END) AS excess,
            SUM(CASE WHEN exception_type = 'zero_velocity'      THEN 1 ELSE 0 END) AS zero_velocity,
            SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical_count,
            SUM(CASE WHEN severity = 'high'     THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN severity = 'medium'   THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN severity = 'low'      THEN 1 ELSE 0 END) AS low_count,
            COALESCE(SUM(estimated_order_value), 0)                  AS total_recommended_order_value,
            COALESCE(
                MAX(EXTRACT(DAY FROM NOW() - exception_date::TIMESTAMPTZ))::INT,
                0
            ) AS oldest_open_days
        FROM fact_replenishment_exceptions
        {where}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params_status)
            row = cur.fetchone() or (0,) * 13

    return {
        "open_count": int(row[0] or 0),
        "by_type": {
            "below_rop":          int(row[1] or 0),
            "below_rop_critical": int(row[2] or 0),
            "below_ss":           int(row[3] or 0),
            "stockout":           int(row[4] or 0),
            "excess":             int(row[5] or 0),
            "zero_velocity":      int(row[6] or 0),
        },
        "by_severity": {
            "critical": int(row[7] or 0),
            "high":     int(row[8] or 0),
            "medium":   int(row[9] or 0),
            "low":      int(row[10] or 0),
        },
        "total_recommended_order_value": float(row[11] or 0),
        "oldest_open_days":              int(row[12] or 0),
    }


@router.put("/inv-planning/exceptions/{exception_id}/acknowledge")
def acknowledge_exception(
    exception_id: str,
    body: ExceptionAcknowledgeBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Acknowledge an open exception (sets status to 'acknowledged')."""
    import datetime as _dt

    sql = """
        UPDATE fact_replenishment_exceptions
        SET
            status           = 'acknowledged',
            acknowledged_by  = %s,
            acknowledged_ts  = %s,
            notes            = COALESCE(%s, notes),
            modified_ts      = %s
        WHERE exception_id = %s
          AND status       = 'open'
        RETURNING
            exception_id, item_no, loc, exception_type, severity,
            status, acknowledged_by, acknowledged_ts, notes
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [body.acknowledged_by, now, body.notes, now, exception_id])
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Exception not found or already acknowledged")

    return {
        "exception_id":    row[0],
        "item_no":         row[1],
        "loc":             row[2],
        "exception_type":  row[3],
        "severity":        row[4],
        "status":          row[5],
        "acknowledged_by": row[6],
        "acknowledged_ts": row[7].isoformat() if row[7] else None,
        "notes":           row[8],
    }


@router.put("/inv-planning/exceptions/{exception_id}/status")
def update_exception_status(
    exception_id: str,
    body: ExceptionStatusBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Advance exception status to 'ordered' or 'resolved'."""
    import datetime as _dt

    if body.status not in ("ordered", "resolved"):
        raise HTTPException(status_code=422, detail="status must be 'ordered' or 'resolved'")

    now = _dt.datetime.now(_dt.timezone.utc)
    ts_col = "ordered_ts" if body.status == "ordered" else "resolved_ts"

    sql = f"""
        UPDATE fact_replenishment_exceptions
        SET
            status      = %s,
            {ts_col}    = %s,
            notes       = COALESCE(%s, notes),
            modified_ts = %s
        WHERE exception_id = %s
        RETURNING
            exception_id, item_no, loc, exception_type, severity,
            status, acknowledged_by, {ts_col}, notes
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [body.status, now, body.notes, now, exception_id])
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Exception not found")

    return {
        "exception_id":   row[0],
        "item_no":        row[1],
        "loc":            row[2],
        "exception_type": row[3],
        "severity":       row[4],
        "status":         row[5],
        "acknowledged_by": row[6],
        ts_col:           row[7].isoformat() if row[7] else None,
        "notes":          row[8],
    }


@router.post("/inv-planning/exceptions/generate")
def generate_exceptions(
    _: None = Depends(require_api_key),
) -> dict:
    """Trigger exception detection and insert new exceptions into the queue."""
    from scripts.generate_replenishment_exceptions import run as _gen_run

    result = _gen_run(dry_run=False)
    return {
        "generated_count": result["generated_count"],
        "skipped_dedup":   result["skipped_dedup"],
        "by_type":         result["by_type"],
    }


# ---------------------------------------------------------------------------
# IPfeature11: ABC-XYZ Policy Matrix endpoints
# ---------------------------------------------------------------------------

@router.get("/inv-planning/abc-xyz/matrix")
def get_abc_xyz_matrix(
    response: FastAPIResponse,
) -> dict:
    """9-cell ABC-XYZ matrix with DFU counts and avg service level per cell."""
    set_cache(response, max_age=3600)
    sql = """
        SELECT
            abc_vol, xyz_class,
            COUNT(*)                     AS dfu_count,
            AVG(abc_xyz_service_level)   AS avg_service_level,
            AVG(abc_xyz_dos_min)         AS avg_dos_min,
            AVG(abc_xyz_dos_max)         AS avg_dos_max
        FROM dim_dfu
        WHERE abc_vol IS NOT NULL AND xyz_class IS NOT NULL
        GROUP BY abc_vol, xyz_class
        ORDER BY abc_vol, xyz_class
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [])
            rows = cur.fetchall()

    cells = [
        {
            "abc_vol":           r[0],
            "xyz_class":         r[1],
            "segment":           (r[0] or "") + (r[1] or ""),
            "dfu_count":         int(r[2]),
            "avg_service_level": float(r[3]) if r[3] is not None else None,
            "avg_dos_min":       float(r[4]) if r[4] is not None else None,
            "avg_dos_max":       float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]
    return {"cells": cells, "total_classified": sum(c["dfu_count"] for c in cells)}


@router.get("/inv-planning/abc-xyz/summary")
def get_abc_xyz_summary(
    response: FastAPIResponse,
) -> dict:
    """Portfolio-level ABC-XYZ classification summary."""
    set_cache(response, max_age=3600)
    sql = """
        SELECT
            COUNT(*)                                  AS total_dfus,
            COUNT(*) FILTER (WHERE xyz_class IS NOT NULL) AS classified_dfus,
            COUNT(*) FILTER (WHERE xyz_class = 'X')   AS x_count,
            COUNT(*) FILTER (WHERE xyz_class = 'Y')   AS y_count,
            COUNT(*) FILTER (WHERE xyz_class = 'Z')   AS z_count,
            AVG(demand_cv)                            AS avg_demand_cv,
            AVG(intermittency_ratio)                  AS avg_intermittency_ratio
        FROM dim_dfu
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [])
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]

    result = dict(zip(cols, row)) if row else {}
    return {k: (float(v) if isinstance(v, (int, float)) and v is not None else v)
            for k, v in result.items()}


@router.get("/inv-planning/abc-xyz/detail")
def get_abc_xyz_detail(
    response: FastAPIResponse,
    abc_vol: Optional[str] = Query(None, max_length=10),
    xyz_class: Optional[str] = Query(None, max_length=5),
    segment: Optional[str] = Query(None, max_length=10),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Paginated list of DFUs with their ABC-XYZ classification."""
    set_cache(response, max_age=600)

    where_clauses: list[str] = []
    params: list = []

    if abc_vol:
        params.append(abc_vol.upper())
        where_clauses.append(f"abc_vol = ${len(params)}")
    if xyz_class:
        params.append(xyz_class.upper())
        where_clauses.append(f"xyz_class = ${len(params)}")
    if segment:
        params.append(segment.upper())
        where_clauses.append(f"abc_xyz_segment = ${len(params)}")
    if item:
        params.append(f"%{item}%")
        where_clauses.append(f"dmdunit ILIKE ${len(params)}")
    if location:
        params.append(f"%{location}%")
        where_clauses.append(f"loc ILIKE ${len(params)}")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    count_sql = f"SELECT COUNT(*) FROM dim_dfu {where_sql}"
    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT dmdunit, dmdgroup, loc,
               abc_vol, xyz_class, abc_xyz_segment,
               demand_cv, intermittency_ratio,
               abc_xyz_dos_min, abc_xyz_dos_max, abc_xyz_service_level
        FROM dim_dfu
        {where_sql}
        ORDER BY abc_xyz_segment NULLS LAST, dmdunit
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
                "dmdunit":               r[0],
                "dmdgroup":              r[1],
                "loc":                   r[2],
                "abc_vol":               r[3],
                "xyz_class":             r[4],
                "abc_xyz_segment":       r[5],
                "demand_cv":             float(r[6]) if r[6] is not None else None,
                "intermittency_ratio":   float(r[7]) if r[7] is not None else None,
                "abc_xyz_dos_min":       float(r[8]) if r[8] is not None else None,
                "abc_xyz_dos_max":       float(r[9]) if r[9] is not None else None,
                "abc_xyz_service_level": float(r[10]) if r[10] is not None else None,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# IPfeature12: Supplier Performance Intelligence endpoints
# ---------------------------------------------------------------------------

@router.get("/inv-planning/supplier-performance/summary")
def get_supplier_performance_summary(
    response: FastAPIResponse,
) -> dict:
    """Portfolio-level supplier reliability summary."""
    set_cache(response, max_age=3600)
    sql = """
        SELECT
            COUNT(*)                             AS total_suppliers,
            AVG(supplier_reliability_score)      AS avg_reliability_score,
            AVG(avg_lt_mean_days)                AS avg_lead_time_days,
            AVG(avg_lt_cv)                       AS avg_lt_cv,
            AVG(pct_stable_lt)                   AS avg_pct_stable,
            AVG(pct_volatile_lt)                 AS avg_pct_volatile,
            SUM(total_ss_value)                  AS total_ss_value,
            COUNT(*) FILTER (WHERE supplier_reliability_score < 40) AS low_reliability_count
        FROM mv_supplier_performance
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [])
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]

    result = dict(zip(cols, row)) if row else {}
    return {k: (float(v) if v is not None and isinstance(v, (int, float)) else v)
            for k, v in result.items()}


@router.get("/inv-planning/supplier-performance/detail")
def get_supplier_performance_detail(
    response: FastAPIResponse,
    supplier: Optional[str] = Query(None, max_length=120),
    min_score: Optional[int] = Query(None, ge=0, le=100),
    max_score: Optional[int] = Query(None, ge=0, le=100),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("supplier_reliability_score", max_length=40),
    sort_dir: str = Query("asc", max_length=4),
) -> dict:
    """Paginated supplier performance detail."""
    set_cache(response, max_age=3600)

    allowed_sort = {"supplier_reliability_score", "avg_lt_mean_days", "avg_lt_cv", "sku_loc_count"}
    order_col = sort_by if sort_by in allowed_sort else "supplier_reliability_score"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_clauses: list[str] = []
    params: list = []

    if supplier:
        params.append(f"%{supplier}%")
        where_clauses.append(f"(supplier_no ILIKE ${len(params)} OR supplier_name ILIKE ${len(params)})")
    if min_score is not None:
        params.append(min_score)
        where_clauses.append(f"supplier_reliability_score >= ${len(params)}")
    if max_score is not None:
        params.append(max_score)
        where_clauses.append(f"supplier_reliability_score <= ${len(params)}")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    count_sql = f"SELECT COUNT(*) FROM mv_supplier_performance {where_sql}"
    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT supplier_no, supplier_name, sku_loc_count, distinct_items,
               avg_lt_mean_days, avg_lt_cv, avg_lt_std_days,
               pct_stable_lt, pct_volatile_lt,
               total_safety_stock_units, total_ss_value,
               supplier_reliability_score
        FROM mv_supplier_performance
        {where_sql}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

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
                "supplier_no":                r[0],
                "supplier_name":              r[1],
                "sku_loc_count":              int(r[2] or 0),
                "distinct_items":             int(r[3] or 0),
                "avg_lt_mean_days":           _f(r[4]),
                "avg_lt_cv":                  _f(r[5]),
                "avg_lt_std_days":            _f(r[6]),
                "pct_stable_lt":              _f(r[7]),
                "pct_volatile_lt":            _f(r[8]),
                "total_safety_stock_units":   _f(r[9]),
                "total_ss_value":             _f(r[10]),
                "supplier_reliability_score": int(r[11]) if r[11] is not None else None,
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/supplier-performance/items")
def get_supplier_items(
    response: FastAPIResponse,
    supplier_no: str = Query(..., max_length=120),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """Items supplied by a specific supplier with LT profile data."""
    set_cache(response, max_age=3600)
    sql = """
        SELECT ltp.item_no, ltp.loc,
               ltp.lt_mean_days, ltp.lt_std_days, ltp.lt_cv, ltp.lt_variability_class,
               ltp.observation_months,
               d.abc_vol, d.cluster_assignment
        FROM dim_item_lead_time_profile ltp
        INNER JOIN dim_item i ON ltp.item_no = i.item_no
        LEFT JOIN dim_dfu d ON ltp.item_no = d.dmdunit AND ltp.loc = d.loc
        WHERE i.supplier_no = %s
        ORDER BY ltp.lt_cv DESC NULLS LAST
        LIMIT %s OFFSET %s
    """
    count_sql = """
        SELECT COUNT(*) FROM dim_item_lead_time_profile ltp
        INNER JOIN dim_item i ON ltp.item_no = i.item_no
        WHERE i.supplier_no = %s
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, [supplier_no])
            total = cur.fetchone()[0] or 0
            cur.execute(sql, [supplier_no, limit, offset])
            rows = cur.fetchall()

    return {
        "supplier_no": supplier_no,
        "total": int(total),
        "rows": [
            {
                "item_no":              r[0],
                "loc":                  r[1],
                "lt_mean_days":         _f(r[2]),
                "lt_std_days":          _f(r[3]),
                "lt_cv":                _f(r[4]),
                "lt_variability_class": r[5],
                "observation_months":   int(r[6]) if r[6] else None,
                "abc_vol":              r[7],
                "cluster_assignment":   r[8],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# IPfeature14: Intra-Month Stockout Detection endpoints
# ---------------------------------------------------------------------------

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

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

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

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

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

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

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


# ---------------------------------------------------------------------------
# IPfeature9: Demand Sensing endpoints
# ---------------------------------------------------------------------------

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
        where_clauses.append("s.item_no ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_clauses.append("s.loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_clauses.append("d.abc_vol = %s")

    where_sql = "WHERE " + " AND ".join(where_clauses)
    count_sql = f"""
        SELECT COUNT(*) FROM fact_demand_signals s
        LEFT JOIN dim_dfu d ON s.item_no = d.dmdunit AND s.loc = d.loc
        {where_sql}
    """

    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT s.item_no, s.loc, s.signal_date, s.signal_type, s.alert_priority,
               s.mtd_actual, s.projected_monthly, s.forecast_monthly,
               s.demand_vs_forecast_pct, s.projected_stockout, s.projected_excess,
               s.current_on_hand, s.is_below_ss, s.days_remaining,
               d.abc_vol
        FROM fact_demand_signals s
        LEFT JOIN dim_dfu d ON s.item_no = d.dmdunit AND s.loc = d.loc
        {where_sql}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT %s OFFSET %s
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

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
                "item_no":                r[0],
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
        SELECT item_no, loc, signal_date, signal_type, alert_priority,
               mtd_actual, projected_monthly, forecast_monthly,
               demand_vs_forecast_pct, days_elapsed, days_remaining,
               current_on_hand, is_below_ss
        FROM fact_demand_signals
        WHERE item_no = %s AND loc = %s
        ORDER BY signal_date DESC
        LIMIT 1
    """

    daily_sql = """
        SELECT snapshot_date, mtd_sales,
               mtd_sales / NULLIF(EXTRACT(day FROM snapshot_date), 0) *
               EXTRACT(days IN month FROM snapshot_date) AS mtd_expected_pace
        FROM fact_inventory_snapshot
        WHERE item_no = %s AND loc = %s
          AND snapshot_date >= DATE_TRUNC('month', CURRENT_DATE)
        ORDER BY snapshot_date
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [item, location])
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="No signal found for item/location")

            cur.execute(daily_sql, [item, location])
            daily_rows = cur.fetchall()

    return {
        "item_no":                row[0],
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


# ---------------------------------------------------------------------------
# IPfeature10: Monte Carlo Simulation endpoints
# ---------------------------------------------------------------------------

@router.post("/inv-planning/simulation/run")
def run_simulation(
    item_no: str,
    loc: str,
    n_simulations: int = 10000,
    target_csl: Optional[float] = None,
    _: None = Depends(require_api_key),
) -> dict:
    """Trigger a Monte Carlo safety stock simulation for an item-location."""
    import uuid as _uuid
    sim_run_id = str(_uuid.uuid4())
    # Run synchronously for simplicity (large N is still fast enough ~30s for 10k)
    try:
        from scripts.run_ss_simulation import run as _sim_run
        result = _sim_run(
            item_no=item_no,
            loc=loc,
            n_simulations=n_simulations,
            target_csl=target_csl,
        )
        return {"sim_run_id": result["sim_run_id"], "status": "completed"}
    except Exception as exc:
        return {"sim_run_id": sim_run_id, "status": "failed", "error": str(exc)}


@router.get("/inv-planning/simulation/results")
def get_simulation_results(
    response: FastAPIResponse,
    item: str = Query(..., max_length=120),
    location: str = Query(..., max_length=120),
) -> dict:
    """Latest simulation results for an item-location."""
    set_cache(response, max_age=300)
    import json as _json

    sql = """
        SELECT sim_run_id, item_no, loc, simulation_date, n_simulations,
               demand_distribution, demand_mean, demand_std,
               lt_distribution, lt_mean_days, lt_std_days,
               results_by_ss_level,
               target_csl, recommended_ss, recommended_ss_days,
               analytical_ss, sim_vs_analytical_pct
        FROM fact_ss_simulation_results
        WHERE item_no = %s AND loc = %s
        ORDER BY simulation_date DESC, load_ts DESC
        LIMIT 1
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [item, location])
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="No simulation results found")

    curve = _json.loads(row[11]) if isinstance(row[11], str) else (row[11] or [])
    return {
        "sim_run_id":           row[0],
        "item_no":              row[1],
        "loc":                  row[2],
        "simulation_date":      str(row[3]),
        "n_simulations":        int(row[4]),
        "demand_distribution":  row[5],
        "demand_mean":          _f(row[6]),
        "demand_std":           _f(row[7]),
        "lt_distribution":      row[8],
        "lt_mean_days":         _f(row[9]),
        "lt_std_days":          _f(row[10]),
        "service_level_curve":  curve,
        "target_csl":           _f(row[12]),
        "recommended_ss":       _f(row[13]),
        "recommended_ss_days":  _f(row[14]),
        "analytical_ss":        _f(row[15]),
        "sim_vs_analytical_pct":_f(row[16]),
    }


@router.get("/inv-planning/simulation/compare")
def get_simulation_compare(
    response: FastAPIResponse,
    item: str = Query(..., max_length=120),
    location: str = Query(..., max_length=120),
) -> dict:
    """Compare simulated vs analytical SS for an item-location."""
    set_cache(response, max_age=300)
    import json as _json

    sql = """
        SELECT recommended_ss, analytical_ss, sim_vs_analytical_pct,
               results_by_ss_level, target_csl
        FROM fact_ss_simulation_results
        WHERE item_no = %s AND loc = %s
        ORDER BY simulation_date DESC
        LIMIT 1
    """
    eom_sql = """
        SELECT eom_qty_on_hand FROM agg_inventory_monthly
        WHERE item_no = %s AND loc = %s
        ORDER BY month_start DESC LIMIT 1
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [item, location])
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="No simulation results found")

            cur.execute(eom_sql, [item, location])
            eom_row = cur.fetchone()
            current_on_hand = _f(eom_row[0]) if eom_row else None

    curve = _json.loads(row[3]) if isinstance(row[3], str) else (row[3] or [])
    # Find current CSL from curve
    current_csl = None
    if current_on_hand is not None and curve:
        for pt in curve:
            if pt["ss_qty"] >= current_on_hand:
                current_csl = pt["csl"]
                break

    return {
        "item_no":              item,
        "loc":                  location,
        "analytical_ss":        _f(row[1]),
        "simulated_ss":         _f(row[0]),
        "difference_pct":       _f(row[2]),
        "service_level_curve":  curve,
        "current_on_hand":      current_on_hand,
        "current_csl":          current_csl,
    }


@router.get("/inv-planning/simulation/{sim_run_id}/status")
def get_simulation_status(
    sim_run_id: str,
) -> dict:
    """Get status of a simulation run."""
    sql = """
        SELECT item_no, loc, simulation_date, load_ts
        FROM fact_ss_simulation_results
        WHERE sim_run_id = %s
        LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [sim_run_id])
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Simulation run not found")

    return {
        "sim_run_id":    sim_run_id,
        "status":        "completed",
        "progress_pct":  100,
        "item_no":       row[0],
        "loc":           row[1],
        "started_at":    None,
        "completed_at":  str(row[3]) if row[3] else None,
        "error":         None,
    }


# ---------------------------------------------------------------------------
# IPfeature13: Investment Optimization endpoints
# ---------------------------------------------------------------------------

@router.get("/inv-planning/investment/efficient-frontier")
def get_efficient_frontier(
    response: FastAPIResponse,
    plan_id: Optional[str] = Query(None, max_length=100),
) -> dict:
    """Efficient frontier curve for investment optimization."""
    set_cache(response, max_age=300)

    params: list = []
    plan_filter = "WHERE plan_id = (SELECT MAX(plan_id) FROM fact_efficient_frontier)"
    if plan_id:
        params.append(plan_id)
        plan_filter = "WHERE plan_id = %s"

    summary_sql = """
        SELECT plan_id, COUNT(*) AS total_items,
               SUM(investment_increment) AS recommended_portfolio_investment,
               SUM(current_ss_value) AS current_portfolio_investment,
               AVG(current_csl) AS current_portfolio_csl,
               AVG(recommended_csl) AS recommended_portfolio_csl
        FROM fact_inventory_investment_plan
        WHERE plan_id = (SELECT MAX(plan_id) FROM fact_efficient_frontier)
        GROUP BY plan_id
    """

    frontier_sql = f"""
        SELECT budget_point, items_funded, achievable_csl, marginal_item
        FROM fact_efficient_frontier
        {plan_filter}
        ORDER BY budget_point
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(summary_sql, [])
            s = cur.fetchone()
            cur.execute(frontier_sql, params)
            rows = cur.fetchall()

    return {
        "plan_id":                         s[0] if s else None,
        "total_items":                     int(s[1]) if s else 0,
        "current_portfolio_investment":    _f(s[3]) if s else None,
        "recommended_portfolio_investment":_f(s[2]) if s else None,
        "current_portfolio_csl":           _f(s[4]) if s else None,
        "recommended_portfolio_csl":       _f(s[5]) if s else None,
        "curve": [
            {
                "budget":        _f(r[0]),
                "items_funded":  int(r[1]),
                "achievable_csl":_f(r[2]),
                "marginal_item": r[3],
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/investment/summary")
def get_investment_summary(
    response: FastAPIResponse,
) -> dict:
    """Investment optimization summary with top ROI items."""
    set_cache(response, max_age=300)

    sql = """
        SELECT
            COUNT(*)                                   AS total_items,
            SUM(current_ss_value)                      AS total_current_investment,
            SUM(recommended_ss_value)                  AS total_recommended_investment,
            SUM(investment_increment)                  AS total_investment_gap,
            AVG(current_csl)                           AS portfolio_csl_current,
            AVG(recommended_csl)                       AS portfolio_csl_recommended
        FROM fact_inventory_investment_plan
        WHERE plan_id = (SELECT MAX(plan_id) FROM fact_inventory_investment_plan)
    """
    top_sql = """
        SELECT item_no, loc, marginal_roi, investment_increment, csl_increment
        FROM fact_inventory_investment_plan
        WHERE plan_id = (SELECT MAX(plan_id) FROM fact_inventory_investment_plan)
        ORDER BY investment_rank
        LIMIT 10
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [])
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            cur.execute(top_sql, [])
            top_rows = cur.fetchall()

    result = dict(zip(cols, row)) if row else {}
    return {
        **{k: _f(v) for k, v in result.items()},
        "top_roi_items": [
            {
                "item_no":             r[0],
                "loc":                 r[1],
                "marginal_roi":        _f(r[2]),
                "investment_increment":_f(r[3]),
                "csl_increment":       _f(r[4]),
            }
            for r in top_rows
        ],
    }


@router.get("/inv-planning/investment/detail")
def get_investment_detail(
    response: FastAPIResponse,
    plan_id: Optional[str] = Query(None, max_length=100),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    abc_xyz_segment: Optional[str] = Query(None, max_length=10),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("marginal_roi", max_length=40),
    sort_dir: str = Query("desc", max_length=4),
) -> dict:
    """Paginated investment plan detail rows."""
    set_cache(response, max_age=120)

    allowed_sort = {"marginal_roi", "investment_increment", "csl_increment", "investment_rank"}
    order_col = sort_by if sort_by in allowed_sort else "investment_rank"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_clauses: list[str] = [
        "plan_id = COALESCE(%s, (SELECT MAX(plan_id) FROM fact_inventory_investment_plan))"
    ]
    params: list = [plan_id]

    if item:
        params.append(f"%{item}%")
        where_clauses.append("item_no ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_clauses.append("loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_clauses.append("abc_vol = %s")
    if abc_xyz_segment:
        params.append(abc_xyz_segment.upper())
        where_clauses.append("abc_xyz_segment = %s")

    where_sql = "WHERE " + " AND ".join(where_clauses)
    count_sql = f"SELECT COUNT(*) FROM fact_inventory_investment_plan {where_sql}"

    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT item_no, loc, abc_vol, abc_xyz_segment,
               current_ss_qty, current_ss_value, current_csl,
               recommended_ss_qty, recommended_ss_value, recommended_csl,
               ss_increment_qty, investment_increment, csl_increment, marginal_roi,
               investment_rank, cumulative_investment
        FROM fact_inventory_investment_plan
        {where_sql}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT %s OFFSET %s
    """

    def _f(v: Any) -> float | None:
        return float(v) if v is not None else None

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
                "abc_vol":              r[2],
                "abc_xyz_segment":      r[3],
                "current_ss_qty":       _f(r[4]),
                "current_ss_value":     _f(r[5]),
                "current_csl":          _f(r[6]),
                "recommended_ss_qty":   _f(r[7]),
                "recommended_ss_value": _f(r[8]),
                "recommended_csl":      _f(r[9]),
                "ss_increment_qty":     _f(r[10]),
                "investment_increment": _f(r[11]),
                "csl_increment":        _f(r[12]),
                "marginal_roi":         _f(r[13]),
                "investment_rank":      int(r[14]) if r[14] else None,
                "cumulative_investment":_f(r[15]),
            }
            for r in rows
        ],
    }


@router.post("/inv-planning/investment/plan")
def run_investment_plan(
    budget_constraint: Optional[float] = None,
    target_csl: Optional[float] = None,
    _: None = Depends(require_api_key),
) -> dict:
    """Trigger capital investment optimization plan computation."""
    from scripts.compute_investment_plan import run as _plan_run
    result = _plan_run(budget_constraint=budget_constraint, target_csl=target_csl)
    return {"plan_id": result["plan_id"], "status": "completed", "total_items": result["total_items"]}
