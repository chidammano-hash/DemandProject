"""Inventory Planning — IPfeature12: Supplier Performance Intelligence endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, add_cross_dim_filters, get_conn, set_cache

router = APIRouter(tags=["inv-planning"])




@router.get("/inv-planning/supplier-performance/summary")
def get_supplier_performance_summary(
    response: FastAPIResponse,
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
) -> dict:
    """Portfolio-level supplier reliability summary."""
    set_cache(response, max_age=3600)

    where_clauses: list[str] = []
    params: list = []

    add_cross_dim_filters(where_clauses, params, brand=brand, category=category, market=market,
                          item_col="t.supplier_no")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Gen-4 Roadmap 1.6 — switched to mv_supplier_po_performance (the legacy
    # mv_supplier_performance was retired in sql/143).
    sql = f"""
        SELECT
            COUNT(*)                             AS total_suppliers,
            AVG(reliability_score)               AS avg_reliability_score,
            AVG(avg_lead_time_days)              AS avg_lead_time_days,
            AVG(CASE WHEN avg_lead_time_days > 0
                     THEN stddev_lead_time_days / avg_lead_time_days END)
                                                 AS avg_lt_cv,
            AVG(otd_pct / 100.0)                 AS avg_otd,
            AVG(otif_pct / 100.0)                AS avg_otif,
            SUM(total_value)                     AS total_value,
            COUNT(*) FILTER (WHERE reliability_score < 40) AS low_reliability_count
        FROM mv_supplier_po_performance t
        {where_sql}
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
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
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("supplier_reliability_score", max_length=40),
    sort_dir: str = Query("asc", max_length=4),
) -> dict:
    """Paginated supplier performance detail."""
    set_cache(response, max_age=3600)

    # Gen-4 Roadmap 1.6 — view renamed from mv_supplier_performance.
    # Column aliases preserve the old JSON shape for clients that depend on it.
    allowed_sort_map = {
        "supplier_reliability_score": "reliability_score",
        "avg_lt_mean_days": "avg_lead_time_days",
        "avg_lt_cv": "avg_lt_cv",
        "sku_loc_count": "distinct_items",
        "otif_pct": "otif_pct",
        "otd_pct": "otd_pct",
    }
    order_col = allowed_sort_map.get(sort_by, "reliability_score")
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_clauses: list[str] = []
    params: list = []

    if supplier:
        params.append(f"%{supplier}%")
        where_clauses.append("(supplier_id ILIKE %s OR supplier_name ILIKE %s)")
        params.append(f"%{supplier}%")
    if min_score is not None:
        params.append(min_score)
        where_clauses.append("reliability_score >= %s")
    if max_score is not None:
        params.append(max_score)
        where_clauses.append("reliability_score <= %s")
    add_cross_dim_filters(where_clauses, params, brand=brand, category=category, market=market,
                          item_col="t.supplier_id")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    count_sql = f"SELECT COUNT(*) FROM mv_supplier_po_performance t {where_sql}"
    filter_params = list(params)
    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT supplier_id                 AS supplier_no,
               supplier_name,
               distinct_items              AS sku_loc_count,
               distinct_items,
               avg_lead_time_days          AS avg_lt_mean_days,
               CASE WHEN avg_lead_time_days > 0
                    THEN stddev_lead_time_days / avg_lead_time_days END
                                           AS avg_lt_cv,
               stddev_lead_time_days       AS avg_lt_std_days,
               otd_pct,
               otif_pct,
               in_full_pct,
               total_value                 AS total_ss_value,
               reliability_score           AS supplier_reliability_score
        FROM mv_supplier_po_performance t
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
                "supplier_no":                r[0],
                "supplier_name":              r[1],
                "sku_loc_count":              int(r[2] or 0),
                "distinct_items":             int(r[3] or 0),
                "avg_lt_mean_days":           _f(r[4]),
                "avg_lt_cv":                  _f(r[5]),
                "avg_lt_std_days":            _f(r[6]),
                "otd_pct":                    _f(r[7]),
                "otif_pct":                   _f(r[8]),
                "in_full_pct":                _f(r[9]),
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
        SELECT ltp.item_id, ltp.loc,
               ltp.lt_mean_days, ltp.lt_std_days, ltp.lt_cv, ltp.lt_variability_class,
               ltp.observation_months,
               d.abc_vol, d.cluster_assignment
        FROM dim_item_lead_time_profile ltp
        INNER JOIN dim_item i ON ltp.item_id = i.item_id
        LEFT JOIN dim_sku d ON ltp.item_id = d.item_id AND ltp.loc = d.loc
        WHERE i.supplier_no = %s
        ORDER BY ltp.lt_cv DESC NULLS LAST
        LIMIT %s OFFSET %s
    """
    count_sql = """
        SELECT COUNT(*) FROM dim_item_lead_time_profile ltp
        INNER JOIN dim_item i ON ltp.item_id = i.item_id
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
                "item_id":              r[0],
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
