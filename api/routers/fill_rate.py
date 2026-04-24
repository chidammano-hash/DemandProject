"""Fill Rate & Demand Fulfillment Analytics — IPfeature8.

Router mounted at /fill-rate in api/main.py.
Queries mv_fill_rate_monthly materialized view for fill rate metrics.
"""
from __future__ import annotations

import logging
from typing import Optional

import psycopg

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, add_cross_dim_filters, get_conn, set_cache
from common.core.service_levels import load_sl_targets_by_abc

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fill-rate"])

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# GET /fill-rate/summary
# ---------------------------------------------------------------------------

@router.get("/fill-rate/summary")
def get_fill_rate_summary(
    response: FastAPIResponse,
    month_from: Optional[str] = Query(None),
    month_to: Optional[str] = Query(None),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    cluster_assignment: Optional[str] = Query(None, max_length=120),
    region: Optional[str] = Query(None, max_length=120),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
) -> dict:
    """Portfolio fill rate summary with by_abc breakdown, worst items, and trend.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    where_parts: list[str] = []
    params: list = []

    if month_from:
        params.append(month_from)
        where_parts.append("month_start >= %s")
    if month_to:
        params.append(month_to)
        where_parts.append("month_start <= %s")
    if item:
        params.append(f"%{item}%")
        where_parts.append("item_id ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_parts.append("loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_parts.append("abc_vol = %s")
    if cluster_assignment:
        params.append(cluster_assignment)
        where_parts.append("cluster_assignment = %s")
    if region:
        params.append(region)
        where_parts.append("region = %s")
    add_cross_dim_filters(where_parts, params, brand=brand, category=category, market=market)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    totals_sql = f"""
        SELECT
            CASE WHEN SUM(total_ordered) > 0
                 THEN SUM(total_shipped) / SUM(total_ordered)
                 ELSE NULL END                              AS portfolio_fill_rate,
            COALESCE(SUM(total_ordered), 0)                AS total_ordered,
            COALESCE(SUM(total_shipped), 0)                AS total_shipped,
            COALESCE(SUM(shortage_qty), 0)                 AS total_shortage_qty,
            COUNT(*) FILTER (WHERE had_partial_fulfillment) AS partial_fulfillment_events
        FROM mv_fill_rate_monthly t
        {where_sql}
    """

    abc_sql = f"""
        SELECT
            abc_vol,
            CASE WHEN SUM(total_ordered) > 0
                 THEN SUM(total_shipped) / SUM(total_ordered) ELSE NULL END AS avg_fill_rate,
            COALESCE(SUM(shortage_qty), 0)                                  AS total_shortage_qty,
            COUNT(*) FILTER (WHERE had_partial_fulfillment)                 AS events
        FROM mv_fill_rate_monthly t
        {where_sql}
        GROUP BY abc_vol
        ORDER BY abc_vol
    """

    # If there are existing filters, we need AND, else WHERE
    if where_parts:
        worst_sql = f"""
            SELECT item_id, loc, fill_rate, shortage_qty, abc_vol
            FROM mv_fill_rate_monthly t
            WHERE {' AND '.join(where_parts)} AND shortage_qty > 0
            ORDER BY shortage_qty DESC
            LIMIT 10
        """
    else:
        worst_sql = """
            SELECT item_id, loc, fill_rate, shortage_qty, abc_vol
            FROM mv_fill_rate_monthly t
            WHERE shortage_qty > 0
            ORDER BY shortage_qty DESC
            LIMIT 10
        """

    trend_sql = f"""
        SELECT
            month_start,
            CASE WHEN SUM(total_ordered) > 0
                 THEN SUM(total_shipped) / SUM(total_ordered) ELSE NULL END AS portfolio_fill_rate,
            COALESCE(SUM(shortage_qty), 0) AS total_shortage_qty
        FROM mv_fill_rate_monthly t
        {where_sql}
        GROUP BY month_start
        ORDER BY month_start
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(totals_sql, params)
            totals_row = cur.fetchone()
            totals_cols = [d[0] for d in cur.description]
            totals = dict(zip(totals_cols, totals_row)) if totals_row else {}

            cur.execute(abc_sql, params)
            abc_rows = cur.fetchall()

            cur.execute(worst_sql, params)
            worst_rows = cur.fetchall()

            cur.execute(trend_sql, params)
            trend_rows = cur.fetchall()

    by_abc: dict = {}
    for r in abc_rows:
        seg = r[0]
        by_abc[seg] = {
            "avg_fill_rate": _f(r[1]),
            "total_shortage_qty": _f(r[2]),
            "events": int(r[3] or 0),
        }

    worst_items = [
        {
            "item_id": r[0],
            "loc": r[1],
            "fill_rate": _f(r[2]),
            "shortage_qty": _f(r[3]),
            "abc_vol": r[4],
        }
        for r in worst_rows
    ]

    trend = [
        {
            "month_start": str(r[0]),
            "portfolio_fill_rate": _f(r[1]),
            "total_shortage_qty": _f(r[2]),
        }
        for r in trend_rows
    ]

    return {
        "portfolio_fill_rate": _f(totals.get("portfolio_fill_rate")),
        "total_ordered": _f(totals.get("total_ordered")) or 0.0,
        "total_shipped": _f(totals.get("total_shipped")) or 0.0,
        "total_shortage_qty": _f(totals.get("total_shortage_qty")) or 0.0,
        "partial_fulfillment_events": int(totals.get("partial_fulfillment_events") or 0),
        "by_abc": by_abc,
        "worst_items": worst_items,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# GET /fill-rate/trend
# ---------------------------------------------------------------------------

@router.get("/fill-rate/trend")
def get_fill_rate_trend(
    response: FastAPIResponse,
    month_from: Optional[str] = Query(None),
    month_to: Optional[str] = Query(None),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
) -> dict:
    """Monthly fill rate trend.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    where_parts: list[str] = []
    params: list = []

    if month_from:
        params.append(month_from)
        where_parts.append("month_start >= %s")
    if month_to:
        params.append(month_to)
        where_parts.append("month_start <= %s")
    if item:
        params.append(f"%{item}%")
        where_parts.append("item_id ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_parts.append("loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_parts.append("abc_vol = %s")
    add_cross_dim_filters(where_parts, params, brand=brand, category=category, market=market)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        SELECT
            month_start,
            CASE WHEN SUM(total_ordered) > 0
                 THEN SUM(total_shipped) / SUM(total_ordered) ELSE NULL END AS fill_rate,
            COALESCE(SUM(total_ordered), 0)  AS total_ordered,
            COALESCE(SUM(total_shipped), 0)  AS total_shipped,
            COALESCE(SUM(shortage_qty), 0)   AS shortage_qty
        FROM mv_fill_rate_monthly t
        {where_sql}
        GROUP BY month_start
        ORDER BY month_start
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return {
        "months": [
            {
                "month_start": str(r[0]),
                "fill_rate": _f(r[1]),
                "total_ordered": _f(r[2]),
                "total_shipped": _f(r[3]),
                "shortage_qty": _f(r[4]),
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# GET /fill-rate/detail
# ---------------------------------------------------------------------------

@router.get("/fill-rate/detail")
def get_fill_rate_detail(
    response: FastAPIResponse,
    month_from: Optional[str] = Query(None),
    month_to: Optional[str] = Query(None),
    item: Optional[str] = Query(None, max_length=120),
    location: Optional[str] = Query(None, max_length=120),
    abc_vol: Optional[str] = Query(None, max_length=10),
    had_partial_fulfillment: Optional[bool] = Query(None),
    brand: Optional[str] = Query(None, max_length=120),
    category: Optional[str] = Query(None, max_length=120),
    market: Optional[str] = Query(None, max_length=120),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("shortage_qty", max_length=40),
    sort_dir: str = Query("desc", max_length=4),
) -> dict:
    """Paginated fill rate detail rows.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    allowed_sort = {"fill_rate", "shortage_qty", "total_ordered"}
    order_col = sort_by if sort_by in allowed_sort else "shortage_qty"
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

    where_parts: list[str] = []
    params: list = []

    if month_from:
        params.append(month_from)
        where_parts.append("month_start >= %s")
    if month_to:
        params.append(month_to)
        where_parts.append("month_start <= %s")
    if item:
        params.append(f"%{item}%")
        where_parts.append("item_id ILIKE %s")
    if location:
        params.append(f"%{location}%")
        where_parts.append("loc ILIKE %s")
    if abc_vol:
        params.append(abc_vol.upper())
        where_parts.append("abc_vol = %s")
    if had_partial_fulfillment is not None:
        params.append(had_partial_fulfillment)
        where_parts.append("had_partial_fulfillment = %s")
    add_cross_dim_filters(where_parts, params, brand=brand, category=category, market=market)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM mv_fill_rate_monthly t {where_sql}"

    filter_params = list(params)
    params.append(limit)
    params.append(offset)
    data_sql = f"""
        SELECT
            item_id, loc, month_start,
            total_ordered, total_shipped,
            fill_rate, shortage_qty,
            had_partial_fulfillment,
            abc_vol, cluster_assignment, region
        FROM mv_fill_rate_monthly t
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
                "item_id": r[0],
                "loc": r[1],
                "month_start": str(r[2]),
                "total_ordered": _f(r[3]),
                "total_shipped": _f(r[4]),
                "fill_rate": _f(r[5]),
                "shortage_qty": _f(r[6]),
                "had_partial_fulfillment": r[7],
                "abc_vol": r[8],
                "cluster_assignment": r[9],
                "region": r[10],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# GET /fill-rate/gap-analysis
# ---------------------------------------------------------------------------

# Service level targets: authoritative source is `fact_service_level_targets`
# (resolved via `common.core.service_levels.load_sl_targets_by_abc`).
# YAML `shared_constants.service_levels_by_abc` serves as fallback.
# See docs/specs/04-inventory-planning/10-service-level-unification.md.


@router.get("/fill-rate/gap-analysis")
def get_fill_rate_gap_analysis(
    response: FastAPIResponse,
    month: str | None = Query(None, description="Month filter e.g. '2026-03'"),
    abc_vol: str | None = Query(None, max_length=10),
) -> dict:
    """Fill rate gap decomposition — breakdown of why fill rate missed target.

    Decomposes the gap between target and actual fill rate into approximate
    causal buckets:
      - SS shortfall: items where on_hand < ss_combined contributed to shortages
      - Demand spike: items where actual demand > forecast by >20%
      - Lead time delay: items where actual LT exceeded expected LT
      - Other / Data gap: remaining shortage not attributed to the above

    The decomposition is *approximate* — individual SKUs may appear in more
    than one bucket, and the impact percentages are heuristic allocations that
    sum roughly (not exactly) to the total gap.  This is intended for planner
    guidance, not accounting precision.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    # --- build WHERE clause --------------------------------------------------
    where_parts: list[str] = []
    params: list = []

    # Month filter: match on month_start (date column)
    if month:
        params.append(month + "-01")  # e.g. "2026-03" -> "2026-03-01"
        where_parts.append("f.month_start = %s::date")
    if abc_vol:
        params.append(abc_vol.upper())
        where_parts.append("f.abc_vol = %s")

    # When no month is given, default to the latest available month
    if not month:
        where_parts.append(
            "f.month_start = (SELECT MAX(month_start) FROM mv_fill_rate_monthly)"
        )

    where_sql = "WHERE " + " AND ".join(where_parts)

    # --- main decomposition query --------------------------------------------
    # LEFT JOIN safety stock for SS-shortfall attribution.
    # LEFT JOIN inventory-forecast bridge for demand-spike and forecast-error
    # attribution.  We pick model_id = 'external' (source-system forecast) to
    # compare against actuals.
    sql = f"""
        WITH fill_data AS (
            SELECT
                f.item_id,
                f.loc,
                f.fill_rate,
                f.shortage_qty,
                f.total_ordered,
                f.abc_vol,
                -- Safety stock shortfall flags
                ss.is_below_ss,
                ss.ss_gap,
                ss.current_qty_on_hand,
                ss.ss_combined,
                -- Forecast vs actual from inventory-forecast bridge
                ivf.forecast,
                ivf.actual_demand,
                ivf.forecast_error,
                ivf.abs_error,
                -- Lead time info
                ivf.latest_lead_time_days,
                ss.lead_time_mean_days AS expected_lt_days
            FROM mv_fill_rate_monthly f
            LEFT JOIN fact_safety_stock_targets ss
                ON f.item_id = ss.item_id AND f.loc = ss.loc
            LEFT JOIN mv_inventory_forecast_monthly ivf
                ON f.item_id = ivf.item_id
               AND f.loc = ivf.loc
               AND f.month_start = ivf.month_start
               AND ivf.model_id = 'external'
            {where_sql}
        )
        SELECT
            -- Portfolio totals
            COUNT(*)                                            AS total_skus,
            COALESCE(AVG(fill_rate), 0)::numeric(7,5)          AS avg_fill_rate,
            COUNT(*) FILTER (WHERE shortage_qty > 0)            AS shortage_sku_count,
            COALESCE(SUM(shortage_qty), 0)                      AS total_shortage_qty,
            COALESCE(SUM(total_ordered), 0)                     AS total_ordered,

            -- 1) SS Shortfall: items below safety stock that had shortages
            COUNT(*) FILTER (
                WHERE is_below_ss AND shortage_qty > 0
            )                                                   AS ss_shortfall_count,
            COALESCE(SUM(shortage_qty) FILTER (
                WHERE is_below_ss AND shortage_qty > 0
            ), 0)                                               AS ss_shortfall_qty,

            -- 2) Demand spike: actual demand > forecast by >20%
            COUNT(*) FILTER (
                WHERE actual_demand > forecast * 1.20
                  AND shortage_qty > 0
                  AND forecast IS NOT NULL
                  AND forecast > 0
            )                                                   AS demand_spike_count,
            COALESCE(SUM(shortage_qty) FILTER (
                WHERE actual_demand > forecast * 1.20
                  AND shortage_qty > 0
                  AND forecast IS NOT NULL
                  AND forecast > 0
            ), 0)                                               AS demand_spike_qty,

            -- 3) Lead time delay: actual LT > expected LT
            COUNT(*) FILTER (
                WHERE latest_lead_time_days > expected_lt_days
                  AND shortage_qty > 0
                  AND expected_lt_days IS NOT NULL
            )                                                   AS lt_delay_count,
            COALESCE(SUM(shortage_qty) FILTER (
                WHERE latest_lead_time_days > expected_lt_days
                  AND shortage_qty > 0
                  AND expected_lt_days IS NOT NULL
            ), 0)                                               AS lt_delay_qty,

            -- Month (for response envelope)
            MIN(fill_data.item_id)                              AS _dummy
        FROM fill_data
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            sl_targets = load_sl_targets_by_abc(cursor=cur)

    # --- handle empty / no-data case ----------------------------------------
    if not row or row[0] == 0:
        return {
            "target_fill_rate": sl_targets.get(
                (abc_vol or "").upper(), sl_targets["default"]
            ),
            "actual_fill_rate": None,
            "gap_pct": None,
            "decomposition": [],
            "month": month or None,
        }

    (
        _total_skus,
        avg_fill_rate,
        shortage_sku_count,
        total_shortage_qty,
        total_ordered,
        ss_shortfall_count,
        ss_shortfall_qty,
        demand_spike_count,
        demand_spike_qty,
        lt_delay_count,
        lt_delay_qty,
        _dummy,
    ) = row

    avg_fill_rate = float(avg_fill_rate)
    total_shortage_qty = float(total_shortage_qty)
    total_ordered = float(total_ordered)
    ss_shortfall_qty = float(ss_shortfall_qty)
    demand_spike_qty = float(demand_spike_qty)
    lt_delay_qty = float(lt_delay_qty)

    # --- compute target and gap ----------------------------------------------
    target = sl_targets.get((abc_vol or "").upper(), sl_targets["default"])
    gap_pct = round((avg_fill_rate - target) * 100, 2)

    # --- allocate shortage qty into causal buckets ---------------------------
    # Impact pct is each bucket's share of shortage scaled to the total gap.
    # Because SKUs can appear in multiple buckets, the raw sum may exceed the
    # total — we proportionally scale so the sum equals gap_pct.
    raw_buckets = [
        ("Safety Stock Shortfall", int(ss_shortfall_count), ss_shortfall_qty),
        ("Demand Spike (>20% above forecast)", int(demand_spike_count), demand_spike_qty),
        ("Lead Time Delay", int(lt_delay_count), lt_delay_qty),
    ]

    attributed_qty = sum(b[2] for b in raw_buckets)
    other_qty = max(total_shortage_qty - attributed_qty, 0)
    other_count = max(int(shortage_sku_count) - sum(b[1] for b in raw_buckets), 0)

    raw_buckets.append(("Other / Data Gap", other_count, other_qty))

    decomposition = []
    for cause, sku_count, qty in raw_buckets:
        if total_ordered > 0 and gap_pct != 0:
            # Scale: shortage qty as pct of total ordered, then align to gap sign
            impact = -(qty / total_ordered) * 100
        else:
            impact = 0.0
        decomposition.append({
            "cause": cause,
            "impact_pct": round(impact, 2),
            "sku_count": sku_count,
            "shortage_qty": round(qty, 1),
        })

    return {
        "target_fill_rate": target,
        "actual_fill_rate": round(avg_fill_rate, 5),
        "gap_pct": gap_pct,
        "decomposition": decomposition,
        "month": month or None,
    }


# ---------------------------------------------------------------------------
# GET /inv-planning/service-level/waterfall  (Issue #16 — Target vs Actual Bridge)
# ---------------------------------------------------------------------------

# Weighted portfolio target: weighted average of ABC class targets by SKU count.
# Targets are resolved via `common.core.service_levels.load_sl_targets_by_abc`
# (DB `fact_service_level_targets` with YAML fallback).


@router.get("/inv-planning/service-level/waterfall")
def get_service_level_waterfall_bridge(
    response: FastAPIResponse,
    month: str | None = Query(None, description="Month filter e.g. '2026-03'"),
) -> dict:
    """Service level target-to-actual bridge chart data.

    Shows how the overall service level target (e.g., 97%) breaks down into
    positive and negative contributions by ABC class and root cause.
    Returns waterfall steps: opening target -> per-class deltas -> closing actual.

    Cache: 300s.
    """
    set_cache(response, max_age=300)

    # --- build WHERE clause --------------------------------------------------
    where_parts: list[str] = []
    params: list = []

    if month:
        params.append(month + "-01")  # e.g. "2026-03" -> "2026-03-01"
        where_parts.append("f.month_start = %s::date")

    # When no month is given, default to the latest available month
    if not month:
        where_parts.append(
            "f.month_start = (SELECT MAX(month_start) FROM mv_fill_rate_monthly)"
        )

    where_sql = "WHERE " + " AND ".join(where_parts)

    # --- Per-ABC class contribution to portfolio fill rate -------------------
    sql = f"""
        SELECT
            ss.abc_vol,
            COUNT(*)                                        AS sku_count,
            CASE WHEN SUM(f.total_ordered) > 0
                 THEN (SUM(f.total_shipped) / SUM(f.total_ordered))::numeric(7,5)
                 ELSE NULL END                              AS avg_fill_rate,
            SUM(f.total_ordered)::numeric                   AS class_ordered
        FROM mv_fill_rate_monthly f
        JOIN fact_safety_stock_targets ss
            ON f.item_id = ss.item_id AND f.loc = ss.loc
        {where_sql}
        GROUP BY ss.abc_vol
        ORDER BY ss.abc_vol
    """

    # Portfolio total (weighted by ordered qty)
    total_sql = f"""
        SELECT
            CASE WHEN SUM(f.total_ordered) > 0
                 THEN (SUM(f.total_shipped) / SUM(f.total_ordered))::numeric(7,5)
                 ELSE NULL END                              AS portfolio_fill_rate,
            SUM(f.total_ordered)::numeric                   AS total_ordered
        FROM mv_fill_rate_monthly f
        JOIN fact_safety_stock_targets ss
            ON f.item_id = ss.item_id AND f.loc = ss.loc
        {where_sql}
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                abc_rows = cur.fetchall()

                cur.execute(total_sql, params)
                total_row = cur.fetchone()

                sl_targets = load_sl_targets_by_abc(cursor=cur)
    except psycopg.Error as e:
        logger.exception("service-level/waterfall: DB query failed: %s", e)
        return {
            "target": None,
            "actual": None,
            "steps": [],
            "month": month or None,
        }

    # --- Handle empty data ---------------------------------------------------
    if not abc_rows or not total_row or total_row[1] is None or float(total_row[1]) == 0:
        return {
            "target": None,
            "actual": None,
            "steps": [],
            "month": month or None,
        }

    portfolio_fill_rate = float(total_row[0]) if total_row[0] is not None else 0.0
    total_ordered = float(total_row[1])

    # --- Build per-class data ------------------------------------------------
    class_data: list[dict] = []
    weighted_target_sum = 0.0
    for row in abc_rows:
        abc_class = row[0] or "?"
        sku_count = int(row[1] or 0)
        avg_fr = float(row[2]) if row[2] is not None else 0.0
        class_ordered = float(row[3]) if row[3] is not None else 0.0
        target_sl = sl_targets.get(abc_class, sl_targets["default"])
        gap = round(avg_fr - target_sl, 5)
        weight = class_ordered / total_ordered if total_ordered > 0 else 0.0

        weighted_target_sum += target_sl * weight

        class_data.append({
            "abc_class": abc_class,
            "sku_count": sku_count,
            "avg_fill_rate": round(avg_fr, 5),
            "target_sl": target_sl,
            "gap": round(gap, 5),
            "weight": round(weight, 4),
            # Weighted contribution of this class's gap to the portfolio delta
            "weighted_gap": round(gap * weight, 5),
        })

    # Weighted portfolio target
    portfolio_target = round(weighted_target_sum, 5)

    # --- Assemble waterfall steps --------------------------------------------
    steps: list[dict] = []

    # Opening: target bar
    steps.append({
        "label": "Target",
        "value": portfolio_target,
        "type": "total",
    })

    # Per-class delta bars (sorted: positive first, then negative)
    sorted_classes = sorted(class_data, key=lambda c: c["weighted_gap"], reverse=True)
    for c in sorted_classes:
        wg = c["weighted_gap"]
        if abs(wg) < 0.00001:
            continue  # skip negligible deltas
        step_type = "positive" if wg > 0 else "negative"
        label_suffix = "over" if wg > 0 else "under"
        steps.append({
            "label": f"{c['abc_class']}-class {label_suffix}",
            "value": round(wg, 5),
            "type": step_type,
        })

    # Closing: actual bar
    steps.append({
        "label": "Actual",
        "value": round(portfolio_fill_rate, 5),
        "type": "total",
    })

    return {
        "target": portfolio_target,
        "actual": round(portfolio_fill_rate, 5),
        "steps": steps,
        "by_class": class_data,
        "month": month or None,
    }
