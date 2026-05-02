"""Demand History Workbench endpoints."""
from __future__ import annotations

import logging
from typing import Any

import psycopg
from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache
from common.planning_date import get_planning_date
from common.utils import load_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demand-history", tags=["demand-history"])

_CFG: dict[str, Any] | None = None


def _cfg() -> dict[str, Any]:
    """Load demand_history section from inventory_planning_config.yaml."""
    global _CFG
    if _CFG is None:
        full = load_config("inventory_planning_config.yaml")
        _CFG = full.get("demand_history", {})
    return _CFG


def _date_cutoff(months: int | None = None) -> str:
    """Return ISO date string for N months before the planning date."""
    cfg = _cfg()
    n = months if months is not None else cfg.get("default_months", 24)
    n = min(n, cfg.get("max_months", 60))
    pd = get_planning_date()
    cutoff = pd.replace(day=1) - relativedelta(months=n)
    return cutoff.isoformat()


def _f(v: Any) -> float | None:
    return float(v) if v is not None else None


def _s(v: Any) -> str | None:
    return str(v) if v is not None else None


# ---------------------------------------------------------------------------
# Feature 1: Demand Reference Panel
# ---------------------------------------------------------------------------


@router.get("/reference")
def demand_reference(
    response: FastAPIResponse,
    item_id: str = Query(..., max_length=120),
    loc: str = Query(..., max_length=120),
    months: int | None = Query(default=None, ge=1, le=60),
):
    """Quick-reference summary for an item+loc."""
    cfg = _cfg()
    set_cache(response, max_age=cfg.get("cache_ttl_seconds", 120))
    cutoff = _date_cutoff(months)
    top_n = cfg.get("pareto_top_n", 5)

    sql_history = """
        SELECT TO_CHAR(f.startdate, 'YYYY-MM') AS month,
               SUM(f.demand_qty) AS demand_qty,
               SUM(f.sales_qty)  AS sales_qty
        FROM fact_customer_demand_monthly f
        WHERE f.item_id = %s AND f.location_id = %s
          AND f.startdate >= %s::date
        GROUP BY f.startdate
        ORDER BY f.startdate
    """

    sql_top_customers = """
        SELECT f.customer_no,
               c.customer_name,
               SUM(f.demand_qty) AS total_demand
        FROM fact_customer_demand_monthly f
        LEFT JOIN dim_customer c ON c.customer_no = f.customer_no
        WHERE f.item_id = %s AND f.location_id = %s
          AND f.startdate >= %s::date
        GROUP BY f.customer_no, c.customer_name
        ORDER BY total_demand DESC
        LIMIT %s
    """

    sql_inventory = """
        SELECT eom_qty_on_hand, latest_lead_time_days
        FROM agg_inventory_monthly
        WHERE item_id = %s AND loc = %s
        ORDER BY month_start DESC
        LIMIT 1
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql_history, [item_id, loc, cutoff])
        history_rows = cur.fetchall()

        cur.execute(sql_top_customers, [item_id, loc, cutoff, top_n])
        top_cust_rows = cur.fetchall()

        try:
            cur.execute(sql_inventory, [item_id, loc])
            inv_row = cur.fetchone()
        except psycopg.Error:
            conn.rollback()
            inv_row = None

    history = [
        {"month": r[0], "demand_qty": _f(r[1]), "sales_qty": _f(r[2])}
        for r in history_rows
    ]

    total_demand = sum(h["demand_qty"] or 0 for h in history)
    top_customers = []
    for r in top_cust_rows:
        cust_demand = _f(r[2]) or 0
        pct = round(100.0 * cust_demand / total_demand, 2) if total_demand > 0 else 0
        top_customers.append({
            "customer_no": _s(r[0]),
            "customer_name": _s(r[1]),
            "total_demand": _f(r[2]),
            "pct_share": pct,
        })

    # MoM trend
    trend_pct = None
    if len(history) >= 2:
        prev = history[-2]["demand_qty"] or 0
        curr = history[-1]["demand_qty"] or 0
        if prev > 0:
            trend_pct = round(100.0 * (curr - prev) / prev, 2)

    inventory = None
    if inv_row:
        inventory = {
            "qty_on_hand": _f(inv_row[0]),
            "lead_time_days": _f(inv_row[1]),
        }

    return {
        "item_id": item_id,
        "loc": loc,
        "history": history,
        "top_customers": top_customers,
        "total_demand": round(total_demand, 2),
        "trend_mom_pct": trend_pct,
        "inventory": inventory,
    }


# ---------------------------------------------------------------------------
# Feature 2: Proportional Decomposition
# ---------------------------------------------------------------------------


@router.get("/decomposition")
def demand_decomposition(
    response: FastAPIResponse,
    item_id: str = Query(..., max_length=120),
    loc: str = Query(..., max_length=120),
    months: int | None = Query(default=None, ge=1, le=60),
):
    """Monthly demand decomposition by customer for an item+loc."""
    cfg = _cfg()
    set_cache(response, max_age=cfg.get("cache_ttl_seconds", 120))
    cutoff = _date_cutoff(months)

    sql_series = """
        SELECT TO_CHAR(f.startdate, 'YYYY-MM') AS month,
               f.customer_no,
               c.customer_name,
               SUM(f.demand_qty) AS demand_qty
        FROM fact_customer_demand_monthly f
        LEFT JOIN dim_customer c ON c.customer_no = f.customer_no
        WHERE f.item_id = %s AND f.location_id = %s
          AND f.startdate >= %s::date
        GROUP BY f.startdate, f.customer_no, c.customer_name
        ORDER BY f.startdate, demand_qty DESC
    """

    sql_pareto = """
        SELECT f.customer_no,
               c.customer_name,
               SUM(f.demand_qty) AS total_demand
        FROM fact_customer_demand_monthly f
        LEFT JOIN dim_customer c ON c.customer_no = f.customer_no
        WHERE f.item_id = %s AND f.location_id = %s
          AND f.startdate >= %s::date
        GROUP BY f.customer_no, c.customer_name
        ORDER BY total_demand DESC
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql_series, [item_id, loc, cutoff])
        series_rows = cur.fetchall()

        cur.execute(sql_pareto, [item_id, loc, cutoff])
        pareto_rows = cur.fetchall()

    month_totals: dict[str, float] = {}
    for r in series_rows:
        month_totals[r[0]] = month_totals.get(r[0], 0) + (_f(r[3]) or 0)

    series = []
    for r in series_rows:
        month = r[0]
        demand = _f(r[3]) or 0
        mt = month_totals.get(month, 0)
        pct = round(100.0 * demand / mt, 2) if mt > 0 else 0
        series.append({
            "month": month,
            "customer_no": _s(r[1]),
            "customer_name": _s(r[2]),
            "demand_qty": _f(r[3]),
            "pct_share": pct,
        })

    grand_total = sum(_f(r[2]) or 0 for r in pareto_rows)
    cumulative = 0.0
    pareto = []
    for r in pareto_rows:
        td = _f(r[2]) or 0
        cumulative += td
        pct = round(100.0 * td / grand_total, 2) if grand_total > 0 else 0
        cum_pct = round(100.0 * cumulative / grand_total, 2) if grand_total > 0 else 0
        pareto.append({
            "customer_no": _s(r[0]),
            "customer_name": _s(r[1]),
            "total_demand": _f(r[2]),
            "pct_share": pct,
            "cumulative_pct": cum_pct,
        })

    return {
        "item_id": item_id,
        "loc": loc,
        "series": series,
        "pareto": pareto,
    }


# ---------------------------------------------------------------------------
# Feature 3: Demand Comparison (hierarchical reconciliation)
# ---------------------------------------------------------------------------


@router.get("/comparison")
def demand_comparison(
    response: FastAPIResponse,
    item_id: str = Query(..., max_length=120),
    loc: str = Query(..., max_length=120),
    months: int | None = Query(default=None, ge=1, le=60),
):
    """Bottom-up vs top-down vs reconciled vs actual."""
    cfg = _cfg()
    set_cache(response, max_age=cfg.get("cache_ttl_seconds", 120))
    cutoff = _date_cutoff(months)
    hier_ids = cfg.get("hierarchical_model_ids", ["bolt_hierarchical"])
    td_ids = cfg.get("top_down_model_ids", ["chronos_bolt"])

    sql_actual = """
        SELECT TO_CHAR(f.startdate, 'YYYY-MM') AS month,
               SUM(f.demand_qty) AS actual_qty
        FROM fact_customer_demand_monthly f
        WHERE f.item_id = %s AND f.location_id = %s
          AND f.startdate >= %s::date
        GROUP BY f.startdate
        ORDER BY f.startdate
    """

    # fact_external_forecast_monthly is where backtest_load writes execution-lag
    # predictions. basefcst_pref is the point forecast; startdate is the period.
    sql_predictions = """
        SELECT TO_CHAR(b.startdate, 'YYYY-MM') AS month,
               b.model_id,
               SUM(b.basefcst_pref) AS pred_qty
        FROM fact_external_forecast_monthly b
        WHERE b.item_id = %s AND b.loc = %s
          AND b.startdate >= %s::date
          AND b.model_id = ANY(%s)
        GROUP BY b.startdate, b.model_id
        ORDER BY b.startdate
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql_actual, [item_id, loc, cutoff])
        actual_rows = cur.fetchall()

        all_model_ids = hier_ids + td_ids
        try:
            cur.execute(sql_predictions, [item_id, loc, cutoff, all_model_ids])
            pred_rows = cur.fetchall()
        except psycopg.Error:
            logger.debug("fact_external_forecast_monthly not yet populated for this DFU")
            pred_rows = []

    pred_map: dict[tuple[str, str], float] = {}
    for r in pred_rows:
        pred_map[(r[0], r[1])] = _f(r[2]) or 0

    comparison = []
    for r in actual_rows:
        month = r[0]
        actual = _f(r[1])

        bu_qty = None
        for mid in hier_ids:
            v = pred_map.get((month, mid))
            if v is not None:
                bu_qty = v
                break

        td_qty = None
        for mid in td_ids:
            v = pred_map.get((month, mid))
            if v is not None:
                td_qty = v
                break

        reconciled = None
        if bu_qty is not None and td_qty is not None:
            reconciled = round((bu_qty + td_qty) / 2, 2)

        comparison.append({
            "month": month,
            "actual_qty": actual,
            "bottom_up_qty": bu_qty,
            "top_down_qty": td_qty,
            "reconciled_qty": reconciled,
        })

    return {
        "item_id": item_id,
        "loc": loc,
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# Feature 4: Demand Workbench (hierarchical drill-down)
# ---------------------------------------------------------------------------

_GRAIN_COLUMNS = {
    "item": {
        "group": "f.item_id",
        "key_expr": "f.item_id",
        "label_join": "LEFT JOIN dim_item di ON di.item_id = f.item_id",
        "label_expr": "COALESCE(di.item_desc, f.item_id)",
    },
    "item_loc": {
        "group": "f.item_id, f.location_id",
        "key_expr": "f.item_id || '||' || f.location_id",
        "label_join": (
            "LEFT JOIN dim_item di ON di.item_id = f.item_id "
            "LEFT JOIN dim_location dl ON dl.location_id = f.location_id"
        ),
        "label_expr": (
            "COALESCE(di.item_desc, f.item_id) || ' @ ' || "
            "COALESCE(dl.site_desc, f.location_id)"
        ),
    },
    "item_loc_customer": {
        "group": "f.item_id, f.location_id, f.customer_no",
        "key_expr": "f.item_id || '||' || f.location_id || '||' || f.customer_no",
        "label_join": (
            "LEFT JOIN dim_item di ON di.item_id = f.item_id "
            "LEFT JOIN dim_location dl ON dl.location_id = f.location_id "
            "LEFT JOIN dim_customer dc ON dc.customer_no = f.customer_no"
        ),
        "label_expr": "COALESCE(dc.customer_name, f.customer_no)",
    },
}

_CHILDREN_MAP = {
    "item": "item_loc",
    "item_loc": "item_loc_customer",
}


@router.get("/workbench")
def demand_workbench(
    response: FastAPIResponse,
    grain: str = Query(default="item", pattern="^(item|item_loc|item_loc_customer)$"),
    item_id: str | None = Query(default=None, max_length=120),
    loc: str | None = Query(default=None, max_length=120),
    customer_no: str | None = Query(default=None, max_length=120),
    months: int | None = Query(default=None, ge=1, le=60),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Hierarchical drill-down data at item, item+loc, or item+loc+customer grain."""
    cfg = _cfg()
    set_cache(response, max_age=cfg.get("cache_ttl_seconds", 120))
    cutoff = _date_cutoff(months)

    g = _GRAIN_COLUMNS[grain]
    group_cols = g["group"]
    key_expr = g["key_expr"]
    label_join = g["label_join"]
    label_expr = g["label_expr"]

    where_parts = ["f.startdate >= %s::date"]
    params: list[Any] = [cutoff]

    if item_id:
        where_parts.append("f.item_id = %s")
        params.append(item_id)
    if loc:
        where_parts.append("f.location_id = %s")
        params.append(loc)
    if customer_no:
        where_parts.append("f.customer_no = %s")
        params.append(customer_no)

    where_sql = " AND ".join(where_parts)

    count_sql = f"""
        SELECT COUNT(DISTINCT ({key_expr}))
        FROM fact_customer_demand_monthly f
        WHERE {where_sql}
    """

    summary_sql = f"""
        SELECT sub.key, sub.label, sub.total_demand
        FROM (
            SELECT {key_expr} AS key,
                   {label_expr} AS label,
                   SUM(f.demand_qty) AS total_demand
            FROM fact_customer_demand_monthly f
            {label_join}
            WHERE {where_sql}
            GROUP BY {group_cols}, {label_expr}
            ORDER BY total_demand DESC
            LIMIT %s OFFSET %s
        ) sub
    """

    detail_sql = f"""
        SELECT {key_expr} AS key,
               TO_CHAR(f.startdate, 'YYYY-MM') AS month,
               SUM(f.demand_qty) AS demand_qty
        FROM fact_customer_demand_monthly f
        WHERE {where_sql}
          AND ({key_expr}) = ANY(%s)
        GROUP BY {key_expr}, f.startdate
        ORDER BY key, f.startdate
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(count_sql, params)
        total = int(cur.fetchone()[0] or 0)

        cur.execute(summary_sql, [*params, limit, offset])
        summary_rows = cur.fetchall()

        keys = [_s(r[0]) for r in summary_rows]

        if keys:
            cur.execute(detail_sql, [*params, keys])
            detail_rows = cur.fetchall()
        else:
            detail_rows = []

    months_map: dict[str, list[dict[str, Any]]] = {}
    for r in detail_rows:
        k = _s(r[0]) or ""
        if k not in months_map:
            months_map[k] = []
        months_map[k].append({"month": r[1], "demand_qty": _f(r[2])})

    series = []
    for r in summary_rows:
        k = _s(r[0]) or ""
        series.append({
            "key": k,
            "label": _s(r[1]),
            "total_demand": _f(r[2]),
            "months": months_map.get(k, []),
        })

    child_grain = _CHILDREN_MAP.get(grain)

    return {
        "grain": grain,
        "total": total,
        "limit": limit,
        "offset": offset,
        "series": series,
        "hierarchy_children": child_grain,
    }


# ---------------------------------------------------------------------------
# Feature 5: Cross-Reference Matrix
# ---------------------------------------------------------------------------

_DIM_CONFIG = {
    "item": {
        "col": "f.item_id",
        "label_join": "LEFT JOIN dim_item di ON di.item_id = f.item_id",
        "label_expr": "COALESCE(di.item_desc, f.item_id)",
    },
    "location": {
        "col": "f.location_id",
        "label_join": "LEFT JOIN dim_location dl ON dl.location_id = f.location_id",
        "label_expr": "COALESCE(dl.site_desc, f.location_id)",
    },
    "customer": {
        "col": "f.customer_no",
        "label_join": "LEFT JOIN dim_customer dc ON dc.customer_no = f.customer_no",
        "label_expr": "COALESCE(dc.customer_name, f.customer_no)",
    },
}


@router.get("/matrix")
def demand_matrix(
    response: FastAPIResponse,
    row_dim: str = Query(default="item", pattern="^(item|location|customer)$"),
    col_dim: str = Query(default="location", pattern="^(item|location|customer)$"),
    metric: str = Query(default="demand_qty", pattern="^(demand_qty|sales_qty|fill_rate)$"),
    months: int | None = Query(default=None, ge=1, le=60),
    limit: int = Query(default=50, ge=1, le=100),
):
    """Pivot grid -- items vs locations vs customers."""
    if row_dim == col_dim:
        raise HTTPException(status_code=422, detail="row_dim and col_dim must be different")

    cfg = _cfg()
    set_cache(response, max_age=cfg.get("cache_ttl_seconds", 120))
    cutoff = _date_cutoff(months)
    max_rows = min(limit, cfg.get("matrix_max_rows", 100))
    max_cols = cfg.get("matrix_max_cols", 50)

    rd = _DIM_CONFIG[row_dim]
    cd = _DIM_CONFIG[col_dim]

    if metric == "fill_rate":
        agg_expr = (
            "CASE WHEN SUM(f.demand_qty) > 0"
            " THEN ROUND(100.0 * SUM(f.sales_qty) / SUM(f.demand_qty), 2)"
            " ELSE NULL END"
        )
    else:
        agg_expr = f"SUM(f.{metric})"

    sql_rows = f"""
        SELECT {rd['col']} AS dim_val, {rd['label_expr']} AS label
        FROM fact_customer_demand_monthly f
        {rd['label_join']}
        WHERE f.startdate >= %s::date
        GROUP BY {rd['col']}, {rd['label_expr']}
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT %s
    """

    sql_cols = f"""
        SELECT {cd['col']} AS dim_val, {cd['label_expr']} AS label
        FROM fact_customer_demand_monthly f
        {cd['label_join']}
        WHERE f.startdate >= %s::date
        GROUP BY {cd['col']}, {cd['label_expr']}
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT %s
    """

    sql_cells = f"""
        SELECT {rd['col']} AS row_val,
               {cd['col']} AS col_val,
               {agg_expr} AS metric_val
        FROM fact_customer_demand_monthly f
        WHERE f.startdate >= %s::date
          AND {rd['col']} = ANY(%s)
          AND {cd['col']} = ANY(%s)
        GROUP BY {rd['col']}, {cd['col']}
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql_rows, [cutoff, max_rows])
        row_data = cur.fetchall()

        cur.execute(sql_cols, [cutoff, max_cols])
        col_data = cur.fetchall()

        row_keys = [_s(r[0]) for r in row_data]
        col_keys = [_s(r[0]) for r in col_data]

        if row_keys and col_keys:
            cur.execute(sql_cells, [cutoff, row_keys, col_keys])
            cell_rows = cur.fetchall()
        else:
            cell_rows = []

    row_labels = {_s(r[0]): _s(r[1]) for r in row_data}
    col_labels = {_s(r[0]): _s(r[1]) for r in col_data}

    col_idx = {k: i for i, k in enumerate(col_keys)}
    cells: list[list[float | None]] = [
        [None] * len(col_keys) for _ in range(len(row_keys))
    ]
    row_idx = {k: i for i, k in enumerate(row_keys)}

    for r in cell_rows:
        ri = row_idx.get(_s(r[0]))
        ci = col_idx.get(_s(r[1]))
        if ri is not None and ci is not None:
            cells[ri][ci] = _f(r[2])

    return {
        "row_dim": row_dim,
        "col_dim": col_dim,
        "metric": metric,
        "rows": row_keys,
        "cols": col_keys,
        "cells": cells,
        "row_labels": row_labels,
        "col_labels": col_labels,
    }


@router.get("/matrix/drill")
def demand_matrix_drill(
    response: FastAPIResponse,
    item_id: str = Query(..., max_length=120),
    loc: str = Query(..., max_length=120),
    months: int | None = Query(default=None, ge=1, le=60),
):
    """Drill into a single cell: monthly history for item+loc."""
    cfg = _cfg()
    set_cache(response, max_age=cfg.get("cache_ttl_seconds", 120))
    cutoff = _date_cutoff(months)

    sql = """
        SELECT TO_CHAR(f.startdate, 'YYYY-MM') AS month,
               SUM(f.demand_qty) AS demand_qty,
               SUM(f.sales_qty)  AS sales_qty
        FROM fact_customer_demand_monthly f
        WHERE f.item_id = %s AND f.location_id = %s
          AND f.startdate >= %s::date
        GROUP BY f.startdate
        ORDER BY f.startdate
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [item_id, loc, cutoff])
        rows = cur.fetchall()

    history = [
        {"month": r[0], "demand_qty": _f(r[1]), "sales_qty": _f(r[2])}
        for r in rows
    ]

    return {
        "item_id": item_id,
        "loc": loc,
        "history": history,
    }
