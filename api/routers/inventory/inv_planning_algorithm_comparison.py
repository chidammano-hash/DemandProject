"""Inventory Planning — Multi-Algorithm Comparison endpoints.

Serves pre-computed SS, EOQ, and ROP results from
fact_inventory_algorithm_comparison for side-by-side analysis of how
different forecast algorithms affect inventory targets.
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from api.core import get_conn

router = APIRouter(tags=["inv-planning"])


@router.get("/inv-planning/algorithm-comparison/summary")
async def get_algorithm_comparison_summary():
    """Aggregate comparison: avg SS, EOQ, ROP by model_id."""
    sql = """
        SELECT model_id,
               COUNT(*)                             AS n_skus,
               AVG(ss_combined)::numeric(10,2)      AS avg_ss,
               AVG(eoq)::numeric(10,2)              AS avg_eoq,
               AVG(reorder_point)::numeric(10,2)    AS avg_rop,
               SUM(ss_combined)::numeric(12,2)      AS total_ss_units,
               SUM(cycle_stock)::numeric(12,2)      AS total_cycle_stock,
               AVG(forecast_avg_monthly)::numeric(10,2) AS avg_forecast
        FROM fact_inventory_algorithm_comparison
        GROUP BY model_id
        ORDER BY model_id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    if not rows:
        return {"models": []}

    return {
        "models": [
            {
                "model_id": r[0],
                "n_skus": int(r[1]),
                "avg_ss": float(r[2]) if r[2] is not None else None,
                "avg_eoq": float(r[3]) if r[3] is not None else None,
                "avg_rop": float(r[4]) if r[4] is not None else None,
                "total_ss_units": float(r[5]) if r[5] is not None else None,
                "total_cycle_stock": float(r[6]) if r[6] is not None else None,
                "avg_forecast": float(r[7]) if r[7] is not None else None,
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/algorithm-comparison/detail")
async def get_algorithm_comparison_detail(
    item_id: str | None = Query(None, description="Filter by item_id (exact match)"),
    loc: str | None = Query(None, description="Filter by location (exact match)"),
    model_id: str | None = Query(None, description="Filter by model_id"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Per-DFU comparison across models with optional filters."""
    wheres: list[str] = []
    params: list = []

    if item_id:
        wheres.append("item_id = %s")
        params.append(item_id)
    if loc:
        wheres.append("loc = %s")
        params.append(loc)
    if model_id:
        wheres.append("model_id = %s")
        params.append(model_id)

    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    count_sql = f"SELECT COUNT(*) FROM fact_inventory_algorithm_comparison {where_clause}"
    rows_sql = f"""
        SELECT model_id, item_id, loc,
               forecast_avg_monthly, forecast_std_monthly, forecast_cv,
               ss_combined, ss_demand_only,
               eoq, effective_eoq,
               reorder_point, cycle_stock,
               abc_vol, service_level,
               computed_at
        FROM fact_inventory_algorithm_comparison
        {where_clause}
        ORDER BY model_id, item_id, loc
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = (cur.fetchone() or (0,))[0]
            cur.execute(rows_sql, [*params, limit, offset])
            rows = cur.fetchall()

    def _f(val: object) -> float | None:
        return float(val) if val is not None else None

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "rows": [
            {
                "model_id": r[0],
                "item_id": r[1],
                "loc": r[2],
                "forecast_avg_monthly": _f(r[3]),
                "forecast_std_monthly": _f(r[4]),
                "forecast_cv": _f(r[5]),
                "ss_combined": _f(r[6]),
                "ss_demand_only": _f(r[7]),
                "eoq": _f(r[8]),
                "effective_eoq": _f(r[9]),
                "reorder_point": _f(r[10]),
                "cycle_stock": _f(r[11]),
                "abc_vol": r[12],
                "service_level": _f(r[13]),
                "computed_at": r[14].isoformat() if r[14] else None,
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/algorithm-comparison/models")
async def get_available_models():
    """List distinct model_ids present in the comparison table."""
    sql = """
        SELECT model_id, COUNT(*) AS n_skus
        FROM fact_inventory_algorithm_comparison
        GROUP BY model_id
        ORDER BY model_id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    return {
        "models": [
            {"model_id": r[0], "n_skus": int(r[1])}
            for r in rows
        ],
    }
