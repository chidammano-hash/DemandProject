"""Inventory Planning — IPfeature13: Investment Optimization endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response as FastAPIResponse

from api.auth import require_api_key
from api.core import _f, _s, get_conn, set_cache

router = APIRouter(tags=["inv-planning"])




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


@router.post("/inv-planning/investment/plan", status_code=201)
def run_investment_plan(
    budget_constraint: Optional[float] = None,
    target_csl: Optional[float] = None,
    _: None = Depends(require_api_key),
) -> dict:
    """Trigger capital investment optimization plan computation."""
    from scripts.compute_investment_plan import run as _plan_run
    result = _plan_run(budget_constraint=budget_constraint, target_csl=target_csl)
    return {"plan_id": result["plan_id"], "status": "completed", "total_items": result["total_items"]}
