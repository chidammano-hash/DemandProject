"""Inventory planning endpoints (feature 34)."""
from __future__ import annotations

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache, qident

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inventory"])


@router.get("/inventory/position")
def inventory_position(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="snapshot_date", max_length=60),
    sort_dir: str = Query(default="desc", max_length=4),
):
    """Latest inventory snapshot per item-location with optional filters."""
    set_cache(response, max_age=120)

    allowed_sort = {"item_id", "loc", "snapshot_date", "lead_time_days",
                    "qty_on_hand", "qty_on_hand_on_order", "qty_on_order", "mtd_sales"}
    order_col = sort_by if sort_by in allowed_sort else "snapshot_date"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list[Any] = []
    has_filter = bool(item.strip() or location.strip())
    if item.strip():
        where_parts.append("item_id ILIKE %s")
        params.append(f"%{item.strip()}%")
    if location.strip():
        where_parts.append("loc ILIKE %s")
        params.append(f"%{location.strip()}%")

    if has_filter:
        inner_where = f"WHERE {' AND '.join(where_parts)}"
        count_sql = f"""
            SELECT count(*) FROM (
                SELECT DISTINCT ON (item_id, loc) 1
                FROM fact_inventory_snapshot
                {inner_where}
                ORDER BY item_id, loc, snapshot_date DESC
            ) _sub
        """
        data_sql = f"""
            SELECT * FROM (
                SELECT DISTINCT ON (item_id, loc)
                       item_id, loc, snapshot_date,
                       lead_time_days, qty_on_hand, qty_on_hand_on_order,
                       qty_on_order, mtd_sales
                FROM fact_inventory_snapshot
                {inner_where}
                ORDER BY item_id, loc, snapshot_date DESC
            ) latest
            ORDER BY {qident(order_col)} {order_dir}
            LIMIT %s OFFSET %s
        """
    else:
        count_sql = """
            SELECT count(*) FROM fact_inventory_snapshot
            WHERE snapshot_date = (SELECT max(snapshot_date) FROM fact_inventory_snapshot)
        """
        data_sql = f"""
            SELECT item_id, loc, snapshot_date,
                   lead_time_days, qty_on_hand, qty_on_hand_on_order,
                   qty_on_order, mtd_sales
            FROM fact_inventory_snapshot
            WHERE snapshot_date = (SELECT max(snapshot_date) FROM fact_inventory_snapshot)
            ORDER BY {qident(order_col)} {order_dir}
            LIMIT %s OFFSET %s
        """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(count_sql, params)
        total = int(cur.fetchone()[0] or 0)
        cur.execute(data_sql, [*params, limit, offset])
        rows = cur.fetchall()

    positions = [
        {
            "item_id": r[0],
            "loc": r[1],
            "snapshot_date": str(r[2]) if r[2] else None,
            "lead_time_days": float(r[3]) if r[3] is not None else None,
            "qty_on_hand": float(r[4]) if r[4] is not None else None,
            "qty_on_hand_on_order": float(r[5]) if r[5] is not None else None,
            "qty_on_order": float(r[6]) if r[6] is not None else None,
            "mtd_sales": float(r[7]) if r[7] is not None else None,
        }
        for r in rows
    ]
    return {"total": total, "limit": limit, "offset": offset, "positions": positions}


@router.get("/inventory/kpis")
def inventory_kpis(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    months: int = Query(default=3, ge=1, le=60),
):
    """Inventory KPIs: point-in-time totals from latest snapshot + supply chain metrics.

    DOS and Turns use the *current month* sales rate (most recent month in
    agg_inventory_monthly) rather than a trailing average, so declining-demand
    items are not understated. dos_delta and dos_prev_month expose the
    period-over-period direction.
    """
    set_cache(response, max_age=120)

    filter_parts: list[str] = []
    filter_params: list[Any] = []
    if item.strip():
        filter_parts.append("item_id ILIKE %s")
        filter_params.append(f"%{item.strip()}%")
    if location.strip():
        filter_parts.append("loc ILIKE %s")
        filter_params.append(f"%{location.strip()}%")
    filter_sql = f"AND {' AND '.join(filter_parts)}" if filter_parts else ""

    # ------------------------------------------------------------------
    # Query 1: point-in-time totals from the latest inventory snapshot
    # ------------------------------------------------------------------
    latest_sql = f"""
        SELECT
            COALESCE(SUM(qty_on_hand), 0)::double precision   AS total_on_hand,
            COALESCE(SUM(qty_on_order), 0)::double precision   AS total_on_order,
            COUNT(DISTINCT item_id)::bigint                    AS distinct_items,
            COUNT(DISTINCT loc)::bigint                        AS distinct_locations,
            MAX(snapshot_date)                                 AS last_snapshot_date
        FROM fact_inventory_snapshot
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date) FROM fact_inventory_snapshot
        )
        {filter_sql}
    """

    # ------------------------------------------------------------------
    # Query 2: current month (latest month_start in agg_inventory_monthly)
    # Used for DOS, WOC, Turns, Lead Time, LT Coverage
    # ------------------------------------------------------------------
    cur_parts: list[str] = [
        "month_start = (SELECT MAX(month_start) FROM agg_inventory_monthly)"
    ]
    cur_params: list[Any] = []
    if item.strip():
        cur_parts.append("item_id ILIKE %s")
        cur_params.append(f"%{item.strip()}%")
    if location.strip():
        cur_parts.append("loc ILIKE %s")
        cur_params.append(f"%{location.strip()}%")
    cur_where = f"WHERE {' AND '.join(cur_parts)}"

    cur_month_sql = f"""
        SELECT
            COALESCE(SUM(monthly_sales), 0)::double precision       AS monthly_sales,
            COALESCE(SUM(avg_qty_on_hand), 0)::double precision     AS avg_on_hand,
            CASE WHEN SUM(monthly_sales) > 0
                 THEN (SUM(latest_lead_time_days * monthly_sales)
                       / SUM(monthly_sales))::double precision
                 ELSE 0 END                                         AS weighted_lt
        FROM agg_inventory_monthly
        {cur_where}
    """

    # ------------------------------------------------------------------
    # Query 3: previous month (one step before the latest month_start)
    # Used for dos_delta and dos_prev_month
    # ------------------------------------------------------------------
    prev_parts: list[str] = [
        "month_start = ("
        "  SELECT MAX(month_start) FROM agg_inventory_monthly"
        "  WHERE month_start < (SELECT MAX(month_start) FROM agg_inventory_monthly)"
        ")"
    ]
    prev_params: list[Any] = []
    if item.strip():
        prev_parts.append("item_id ILIKE %s")
        prev_params.append(f"%{item.strip()}%")
    if location.strip():
        prev_parts.append("loc ILIKE %s")
        prev_params.append(f"%{location.strip()}%")
    prev_where = f"WHERE {' AND '.join(prev_parts)}"

    prev_month_sql = f"""
        SELECT
            COALESCE(SUM(monthly_sales), 0)::double precision       AS monthly_sales,
            COALESCE(SUM(avg_qty_on_hand), 0)::double precision     AS avg_on_hand
        FROM agg_inventory_monthly
        {prev_where}
    """

    cur_row = None
    prev_row = None
    with get_conn() as conn, conn.cursor() as cur:
        # Query 1 reads fact_inventory_snapshot (a base table) and always works.
        cur.execute(latest_sql, filter_params)
        latest_row = cur.fetchone()
        # Queries 2 & 3 read agg_inventory_monthly (an MV). If it has never been
        # refreshed, degrade the MV-derived KPIs (DOS, turns, lead-time coverage)
        # to neutral nulls inside a SAVEPOINT instead of 500-ing — the snapshot
        # totals from Query 1 still render (F1.4). Run `make refresh-mvs-tiered`
        # to populate the derived metrics.
        try:
            with conn.transaction():
                cur.execute(cur_month_sql, cur_params)
                cur_row = cur.fetchone()
                cur.execute(prev_month_sql, prev_params)
                prev_row = cur.fetchone()
        except (psycopg.errors.ObjectNotInPrerequisiteState, psycopg.errors.UndefinedTable) as exc:
            logger.warning("inventory/kpis: agg_inventory_monthly unavailable (%s)", exc)

    # --- point-in-time snapshot values ---
    total_on_hand = float(latest_row[0]) if latest_row else 0.0
    total_on_order = float(latest_row[1]) if latest_row else 0.0
    distinct_items = int(latest_row[2]) if latest_row else 0
    distinct_locations = int(latest_row[3]) if latest_row else 0
    last_snapshot_date = latest_row[4] if latest_row else None

    # --- current-month aggregates ---
    cur_monthly_sales = float(cur_row[0]) if cur_row and cur_row[0] else 0.0
    cur_avg_on_hand = float(cur_row[1]) if cur_row and cur_row[1] else 0.0
    w_lead_time = float(cur_row[2]) if cur_row and cur_row[2] else 0.0

    # --- previous-month aggregates ---
    prev_monthly_sales = float(prev_row[0]) if prev_row and prev_row[0] else 0.0
    prev_avg_on_hand = float(prev_row[1]) if prev_row and prev_row[1] else 0.0

    # ------------------------------------------------------------------
    # KPI computations
    # ------------------------------------------------------------------
    # Daily sales rate based on the single most-recent month (not a trailing avg)
    current_daily = cur_monthly_sales / 30.44 if cur_monthly_sales > 0 else 0.0

    # DOS: current on-hand ÷ current daily rate
    dos = round(total_on_hand / current_daily, 1) if current_daily > 0 else None
    woc = round(dos / 7, 1) if dos is not None else None

    # Previous month approximation: prev avg on-hand / prev daily rate
    prev_daily = prev_monthly_sales / 30.44 if prev_monthly_sales > 0 else 0.0
    prev_dos = (
        round(prev_avg_on_hand / prev_daily, 1)
        if prev_daily > 0 and prev_avg_on_hand > 0
        else None
    )
    dos_delta = (
        round(dos - prev_dos, 1)
        if dos is not None and prev_dos is not None
        else None
    )

    # Inventory Turns: annualise current month sales ÷ current avg on-hand
    # (current state, not diluted by older high-inventory periods)
    turns = (
        round((cur_monthly_sales * 12) / cur_avg_on_hand, 1)
        if cur_avg_on_hand > 0
        else None
    )

    # Lead-time coverage uses current daily rate
    lt_demand = w_lead_time * current_daily
    lt_coverage = (
        round((total_on_hand + total_on_order) / lt_demand, 2)
        if lt_demand > 0
        else None
    )

    return {
        "total_on_hand": total_on_hand,
        "total_on_order": total_on_order,
        "avg_lead_time_days": round(w_lead_time, 1) if w_lead_time else None,
        "dos": dos,
        "dos_prev_month": prev_dos,
        "dos_delta": dos_delta,
        "woc": woc,
        "inventory_turns": turns,
        "lt_coverage": lt_coverage,
        "distinct_items": distinct_items,
        "distinct_locations": distinct_locations,
        "months_covered": months,
        "last_snapshot_date": str(last_snapshot_date) if last_snapshot_date else None,
    }


def _set_inv_params(inv_params: dict, prow: tuple) -> None:
    """Populate the safety-stock / EOQ / policy detail dict from a query row."""
    inv_params.update({
        "safety_stock": round(float(prow[0]), 1) if prow[0] is not None else None,
        "reorder_point_units": round(float(prow[1]), 1) if prow[1] is not None else None,
        "service_level_target": float(prow[2]) if prow[2] is not None else None,
        "z_score": round(float(prow[3]), 3) if prow[3] is not None else None,
        "demand_cv": round(float(prow[4]), 4) if prow[4] is not None else None,
        "eoq": round(float(prow[5]), 1) if prow[5] is not None else None,
        "eoq_cycle_stock": round(float(prow[6]), 1) if prow[6] is not None else None,
        "order_frequency": round(float(prow[7]), 1) if prow[7] is not None else None,
        "order_policy": str(prow[8]) if prow[8] is not None else None,
        "policy_type": str(prow[9]) if prow[9] is not None else None,
    })


@router.get("/inventory/trend")
def inventory_trend(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    months: int = Query(default=12, ge=1, le=120),
):
    """Monthly inventory trend with DOS from rebuilt materialized view."""
    set_cache(response, max_age=120)

    where_parts: list[str] = ["month_start >= (CURRENT_DATE - (%s || ' months')::interval)"]
    params: list[Any] = [months]
    if item.strip():
        where_parts.append("item_id ILIKE %s")
        params.append(f"%{item.strip()}%")
    if location.strip():
        where_parts.append("loc ILIKE %s")
        params.append(f"%{location.strip()}%")
    where_sql = f"WHERE {' AND '.join(where_parts)}"

    sql = f"""
        SELECT
            month_start,
            COALESCE(SUM(eom_qty_on_hand), 0)::double precision           AS total_on_hand,
            COALESCE(SUM(eom_qty_on_hand_on_order - eom_qty_on_hand), 0)::double precision AS total_on_order,
            COALESCE(SUM(monthly_sales), 0)::double precision              AS monthly_sales,
            CASE WHEN SUM(monthly_sales) > 0
                 THEN (SUM(latest_lead_time_days * monthly_sales)
                       / SUM(monthly_sales))::double precision
                 ELSE 0 END                                                AS avg_lead_time,
            CASE WHEN SUM(avg_daily_sls) > 0
                 THEN (SUM(eom_qty_on_hand)
                       / NULLIF(SUM(avg_daily_sls), 0))::double precision
                 ELSE NULL END                                             AS dos
        FROM agg_inventory_monthly
        {where_sql}
        GROUP BY 1
        ORDER BY 1 ASC
    """

    # Optional: fetch safety stock, EOQ, and order policy when item+location filter is set
    inv_params: dict = {}
    if item.strip() and location.strip():
        params_sql = """
            SELECT
                s.ss_combined,
                s.reorder_point,
                s.service_level_target,
                s.z_score,
                s.demand_cv,
                e.effective_eoq,
                e.eoq_cycle_stock,
                e.order_frequency,
                p.policy_id,
                p.policy_type
            FROM fact_safety_stock_targets s
            LEFT JOIN fact_eoq_targets e
                   ON e.item_id = s.item_id AND e.loc = s.loc
            LEFT JOIN fact_dfu_policy_assignment a
                   ON a.item_id = s.item_id AND a.loc = s.loc
            LEFT JOIN dim_replenishment_policy p
                   ON p.policy_id = a.policy_id
            WHERE s.item_id ILIKE %s AND s.loc ILIKE %s
            ORDER BY s.computed_at DESC
            LIMIT 1
        """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            if item.strip() and location.strip():
                cur.execute(params_sql, [f"%{item.strip()}%", f"%{location.strip()}%"])
                prow = cur.fetchone()
                if prow:
                    _set_inv_params(inv_params, prow)
    except (psycopg.errors.ObjectNotInPrerequisiteState, psycopg.errors.UndefinedTable) as exc:
        # agg_inventory_monthly created but not yet refreshed (or missing).
        # Degrade to an empty trend + hint instead of 500 (F1.3).
        logger.warning("inventory/trend: MV unavailable (%s)", exc)
        return {
            "trend": [],
            "params": {},
            "warning": "Upstream materialized view not yet refreshed. Run `make refresh-mvs-tiered`.",
        }

    ss = inv_params.get("safety_stock")
    trend = [
        {
            "month": str(r[0]),
            "total_on_hand": round(float(r[1]), 2),
            "total_on_order": round(float(r[2]), 2),
            "monthly_sales": round(float(r[3]), 2),
            "avg_lead_time": round(float(r[4]), 2),
            "dos": round(float(r[5]), 1) if r[5] is not None else None,
            "safety_stock": ss,
            "cycle_stock": round(max(0.0, float(r[1]) - ss), 1) if ss is not None else None,
        }
        for r in rows
    ]
    return {"trend": trend, "params": inv_params}


@router.get("/inventory/item-detail")
def inventory_item_detail(
    response: FastAPIResponse,
    item: str = Query(min_length=1, max_length=120),
    location: str = Query(min_length=1, max_length=120),
    months: int = Query(default=14, ge=1, le=120),
):
    """Full snapshot history for a specific item-location pair."""
    set_cache(response, max_age=120)

    sql = """
        SELECT snapshot_date, lead_time_days, qty_on_hand,
               qty_on_hand_on_order, qty_on_order, mtd_sales
        FROM fact_inventory_snapshot
        WHERE item_id = %s
          AND loc = %s
          AND snapshot_date >= (CURRENT_DATE - (%s || ' months')::interval)
        ORDER BY snapshot_date ASC
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [item.strip(), location.strip(), months])
        rows = cur.fetchall()

    snapshots = [
        {
            "snapshot_date": str(r[0]) if r[0] else None,
            "lead_time_days": float(r[1]) if r[1] is not None else None,
            "qty_on_hand": float(r[2]) if r[2] is not None else None,
            "qty_on_hand_on_order": float(r[3]) if r[3] is not None else None,
            "qty_on_order": float(r[4]) if r[4] is not None else None,
            "mtd_sales": float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]
    return {"item": item.strip(), "location": location.strip(), "snapshots": snapshots}
