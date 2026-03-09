"""Inventory Planning — IPfeature12: Supplier Performance Intelligence endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, _s, get_conn, set_cache

router = APIRouter(tags=["inv-planning"])




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
