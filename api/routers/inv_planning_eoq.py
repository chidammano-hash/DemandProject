"""Inventory Planning — IPfeature4: EOQ & Cycle Stock endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from api.core import add_cross_dim_filters, get_conn

router = APIRouter(tags=["inv-planning"])


_EOQ_SORT_COLS = {
    "effective_eoq", "eoq", "eoq_cycle_stock", "order_frequency",
    "total_annual_cost", "annual_holding_cost", "annual_order_cost",
    "demand_mean_monthly", "annual_demand",
}


@router.get("/inv-planning/eoq/summary")
async def eoq_summary(
    abc_vol: str | None = None,
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
):
    """Portfolio EOQ summary with by-ABC breakdown."""
    wheres = []
    params: list = []

    if abc_vol:
        wheres.append("abc_vol = %s")
        params.append(abc_vol)
    add_cross_dim_filters(wheres, params, brand=brand, category=category, market=market)
    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    summary_sql = f"""
        SELECT
            COUNT(*)                        AS total_dfus,
            AVG(effective_eoq)              AS avg_effective_eoq,
            SUM(eoq_cycle_stock)            AS total_cycle_stock,
            AVG(order_frequency)            AS avg_order_frequency,
            SUM(total_annual_cost)          AS total_annual_cost
        FROM fact_eoq_targets t
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
        FROM fact_eoq_targets t
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


@router.get("/inv-planning/eoq/detail")
async def eoq_detail(
    item: str | None = None,
    loc: str | None = None,
    abc_vol: str | None = None,
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
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
        wheres.append("item_id ILIKE %s")
        params.append(f"%{item}%")
    if loc:
        wheres.append("loc ILIKE %s")
        params.append(f"%{loc}%")
    if abc_vol:
        wheres.append("abc_vol = %s")
        params.append(abc_vol)
    add_cross_dim_filters(wheres, params, brand=brand, category=category, market=market)
    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    count_sql = f"SELECT COUNT(*) FROM fact_eoq_targets t {where_clause}"
    rows_sql = f"""
        SELECT
            item_id, loc, abc_vol,
            demand_mean_monthly, annual_demand,
            ordering_cost, holding_cost_pct, unit_cost, moq,
            eoq, effective_eoq, eoq_cycle_stock, order_frequency,
            annual_holding_cost, annual_order_cost, total_annual_cost,
            computed_at
        FROM fact_eoq_targets t
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
                "item_id": r[0], "loc": r[1], "abc_vol": r[2],
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
        sql = "SELECT demand_mean_monthly FROM fact_eoq_targets WHERE item_id = %s AND loc = %s LIMIT 1"
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
        "item_id": item,
        "loc": loc,
        "avg_demand_monthly": avg_demand,
        "curve": curve,
    }
