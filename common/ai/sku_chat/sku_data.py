"""Read-only per-SKU data access for the SKU Chatbot tools.

Pure functions that take a psycopg connection pool and return compact,
JSON-serialisable dicts. No writes. The pool is passed down from the router so
``common/`` never imports ``api/`` (the layering ``common/ai/ai_planner.py``
uses). All forecast/accuracy joins use the full SKU grain
``(item_id, customer_group, loc)`` per the CLAUDE.md fan-out rule.
"""
from __future__ import annotations

import logging
from typing import Any

from common.core.sql_helpers import row_to_dict_from_cursor

log = logging.getLogger(__name__)


def _rows(cur: Any) -> list[dict[str, Any]]:
    return [row_to_dict_from_cursor(cur, r) for r in cur.fetchall()]


def search_skus(pool: Any, query: str, limit: int = 10) -> dict[str, Any]:
    """Resolve a fuzzy item_id search to concrete SKU keys."""
    sql = (
        "SELECT item_id, customer_group, loc, abc_vol, ml_cluster, variability_class "
        "FROM dim_sku WHERE item_id ILIKE %s ORDER BY item_id, loc LIMIT %s"
    )
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, [f"%{query}%", int(limit)])
        rows = _rows(cur)
    return {"query": query, "count": len(rows), "results": rows}


def fetch_sku_profile(
    pool: Any, item_id: str, customer_group: str, loc: str
) -> dict[str, Any]:
    """Demand-behaviour profile + classifications + cluster for one SKU."""
    sql = (
        "SELECT item_id, customer_group, loc, abc_vol, "
        "demand_mean, demand_std, demand_cv, demand_p50, demand_p90, "
        "intermittency_ratio, variability_class, "
        "seasonality_profile, seasonality_strength, is_yearly_seasonal, "
        "peak_month, trough_month, "
        "xyz_class, abc_xyz_segment, abc_xyz_service_level, "
        "execution_lag, ml_cluster, cluster_assignment "
        "FROM dim_sku WHERE item_id = %s AND loc = %s"
    )
    params: list[Any] = [item_id, loc]
    if customer_group:
        sql += " AND customer_group = %s"
        params.append(customer_group)
    sql += " LIMIT 1"
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return {"found": False, "item_id": item_id, "loc": loc}
        data = row_to_dict_from_cursor(cur, row)
    data["found"] = True
    return data


def fetch_sku_sales_history(
    pool: Any, item_id: str, loc: str, months: int = 24
) -> dict[str, Any]:
    """Monthly demand history (item + loc grain) ascending by month."""
    sql = (
        "SELECT month_start, qty, qty_shipped, qty_ordered "
        "FROM agg_sales_monthly WHERE item_id = %s AND loc = %s "
        "ORDER BY month_start DESC LIMIT %s"
    )
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, [item_id, loc, int(months)])
        rows = _rows(cur)
    rows.reverse()
    return {"item_id": item_id, "loc": loc, "months": len(rows), "history": rows}


def fetch_sku_forecast(pool: Any, item_id: str, loc: str) -> dict[str, Any]:
    """Forward production forecast (latest plan version) with CI bands."""
    sql = (
        "SELECT forecast_month, forecast_qty, forecast_qty_lower, forecast_qty_upper, "
        "model_id, horizon_months, lag_source "
        "FROM fact_production_forecast WHERE item_id = %s AND loc = %s "
        "AND plan_version = (SELECT MAX(plan_version) FROM fact_production_forecast "
        "WHERE item_id = %s AND loc = %s) ORDER BY forecast_month"
    )
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, [item_id, loc, item_id, loc])
        rows = _rows(cur)
    return {"item_id": item_id, "loc": loc, "horizon": len(rows), "forecast": rows}


def fetch_sku_inventory(
    pool: Any, item_id: str, loc: str, months: int = 24
) -> dict[str, Any]:
    """Monthly inventory position: on-hand, on-order, sales, lead time."""
    sql = (
        "SELECT month_start, avg_qty_on_hand, eom_qty_on_hand, "
        "eom_qty_on_hand_on_order, monthly_sales, avg_daily_sls, "
        "latest_lead_time_days "
        "FROM agg_inventory_monthly WHERE item_id = %s AND loc = %s "
        "ORDER BY month_start DESC LIMIT %s"
    )
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, [item_id, loc, int(months)])
        rows = _rows(cur)
    rows.reverse()
    return {"item_id": item_id, "loc": loc, "months": len(rows), "inventory": rows}


def fetch_sku_accuracy(
    pool: Any, item_id: str, customer_group: str, loc: str
) -> dict[str, Any]:
    """Per-model, per-lag WAPE / bias / accuracy from the per-DFU accuracy MV.

    Sums the MV's own pre-aggregated sums per (model_id, lag) — valid
    aggregation, not a dim_sku fan-out — so an unknown customer_group can be
    rolled up across customer groups safely.
    """
    sql = (
        "SELECT model_id, lag, SUM(sum_forecast) AS sum_forecast, "
        "SUM(sum_actual) AS sum_actual, SUM(sum_abs_error) AS sum_abs_error, "
        "SUM(row_count) AS row_count "
        "FROM agg_accuracy_by_dfu WHERE item_id = %s AND loc = %s"
    )
    params: list[Any] = [item_id, loc]
    if customer_group:
        sql += " AND customer_group = %s"
        params.append(customer_group)
    sql += " GROUP BY model_id, lag ORDER BY lag, model_id"
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = _rows(cur)

    metrics: list[dict[str, Any]] = []
    for r in rows:
        sa = float(r.get("sum_actual") or 0.0)
        sf = float(r.get("sum_forecast") or 0.0)
        se = float(r.get("sum_abs_error") or 0.0)
        wape = (se / sa * 100.0) if sa else None
        bias = ((sf / sa) - 1.0) * 100.0 if sa else None
        acc = (100.0 - wape) if wape is not None else None
        metrics.append(
            {
                "model_id": r.get("model_id"),
                "lag": r.get("lag"),
                "months": r.get("row_count"),
                "wape_pct": round(wape, 2) if wape is not None else None,
                "bias_pct": round(bias, 2) if bias is not None else None,
                "accuracy_pct": round(acc, 2) if acc is not None else None,
            }
        )
    return {"item_id": item_id, "loc": loc, "metrics": metrics}


def fetch_sku_cluster_peers(
    pool: Any, item_id: str, customer_group: str, loc: str, limit: int = 10
) -> dict[str, Any]:
    """Other SKUs sharing this SKU's ``ml_cluster`` for comparison."""
    sub_where = "item_id = %s AND loc = %s"
    sub_params: list[Any] = [item_id, loc]
    if customer_group:
        sub_where += " AND customer_group = %s"
        sub_params.append(customer_group)
    sql = (
        "SELECT item_id, customer_group, loc, abc_vol, demand_mean, demand_cv, "
        "variability_class FROM dim_sku WHERE ml_cluster = "
        "(SELECT ml_cluster FROM dim_sku WHERE " + sub_where + " LIMIT 1) "
        "AND ml_cluster IS NOT NULL AND NOT (item_id = %s AND loc = %s) "
        "ORDER BY demand_mean DESC NULLS LAST LIMIT %s"
    )
    params = [*sub_params, item_id, loc, int(limit)]
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = _rows(cur)
    return {"item_id": item_id, "loc": loc, "peer_count": len(rows), "peers": rows}
