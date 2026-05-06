"""
Inventory Planning Insights — Expert-recommended aggregate views.

Provides cross-domain aggregation endpoints that combine data from multiple
inventory planning subsystems into unified, actionable views.

All endpoints use get_conn() directly (not Depends), matching the
inv_planning_*.py router pattern.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

import psycopg

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _s, get_conn, set_cache
from common.core.constants import ABC_CLASSES
from common.core.planning_date import get_planning_date

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inv-planning"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"critical": 1, "high": 2, "medium": 3, "low": 4, "urgent": 2}


def _safe_float(v) -> float | None:
    """Coerce to float, returning None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _tier_for_dos(dos: float | None) -> str:
    """Return a color tier label for days-of-supply value."""
    if dos is None:
        return "unknown"
    if dos < 7:
        return "critical"
    if dos < 15:
        return "low"
    if dos <= 30:
        return "adequate"
    if dos <= 60:
        return "healthy"
    return "excess"


def _trend_label(current: float | None, prior: float | None) -> str:
    """Return improving / stable / declining based on two values."""
    if current is None or prior is None:
        return "stable"
    if current > prior * 1.02:
        return "improving"
    if current < prior * 0.98:
        return "declining"
    return "stable"


# ───────────────────────────────────────────────────────────────────────────
# 1. GET /inv-planning/action-feed — Unified Action Inbox (Issue #13)
# ───────────────────────────────────────────────────────────────────────────

_URGENCY_THRESHOLDS = {"urgent": 0.75, "high": 0.5, "medium": 0.0}


def _urgency_label(score: float) -> str:
    """Map a 0-1 urgency score to a human-readable label."""
    if score >= _URGENCY_THRESHOLDS["urgent"]:
        return "URGENT"
    if score >= _URGENCY_THRESHOLDS["high"]:
        return "HIGH"
    return "MEDIUM"


def _severity_from_urgency(score: float) -> str:
    """Map urgency score to severity for consistent frontend badge styling."""
    if score >= _URGENCY_THRESHOLDS["urgent"]:
        return "critical"
    if score >= _URGENCY_THRESHOLDS["high"]:
        return "high"
    return "medium"


@router.get("/inv-planning/action-feed")
def get_action_feed(
    response: FastAPIResponse,
    limit: int = Query(20, ge=1, le=100),
    urgency: str = Query("all"),
) -> dict:
    """Unified action inbox: top priority items across exceptions, orders, and projections.

    Combines:
    1. Open exceptions (from fact_replenishment_exceptions) with financial impact
    2. Pending planned orders (from fact_planned_orders WHERE status='proposed')
    3. High-risk items (from mv_integrated_planning_targets WHERE stockout_risk_score >= 60)

    Returns items ranked by urgency + financial impact.
    """
    set_cache(response, max_age=60)

    # Urgency filter: map label to minimum urgency_score threshold
    urgency_min = 0.0
    if urgency == "urgent":
        urgency_min = _URGENCY_THRESHOLDS["urgent"]
    elif urgency == "high":
        urgency_min = _URGENCY_THRESHOLDS["high"]
    elif urgency == "medium":
        urgency_min = _URGENCY_THRESHOLDS["medium"]
    # "all" => 0.0

    actions: list[dict] = []

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # --- Source 1: Open exceptions by financial impact ---
                try:
                    cur.execute("""
                        SELECT 'exception' AS source,
                               item_id, loc, exception_type AS action_type,
                               CASE severity
                                   WHEN 'critical' THEN 0.95
                                   WHEN 'high' THEN 0.75
                                   WHEN 'medium' THEN 0.50
                                   ELSE 0.25
                               END AS urgency_score,
                               COALESCE(financial_impact_total, estimated_order_value, 0) AS financial_impact,
                               'Resolve ' || exception_type || ' exception' AS action_label,
                               created_at
                        FROM fact_replenishment_exceptions
                        WHERE status = 'open'
                        ORDER BY urgency_score DESC, financial_impact DESC NULLS LAST
                        LIMIT %s
                    """, [limit * 2])
                    for r in cur.fetchall():
                        actions.append({
                            "source": r[0],
                            "item_id": r[1],
                            "loc": r[2],
                            "action_type": r[3],
                            "urgency_score": float(r[4]),
                            "financial_impact": _safe_float(r[5]),
                            "action_label": r[6],
                            "created_at": str(r[7]) if r[7] else None,
                        })
                except psycopg.Error as exc:
                    logger.warning("action-inbox: exceptions table query failed: %s", exc)

                # --- Source 2: Pending planned orders ---
                try:
                    cur.execute("""
                        SELECT 'planned_order' AS source,
                               item_id, loc, 'approve_order' AS action_type,
                               CASE WHEN is_past_due THEN 0.9 ELSE 0.6 END AS urgency_score,
                               COALESCE(order_value, 0) AS financial_impact,
                               'Approve order: ' || recommended_qty || ' units' AS action_label,
                               created_at
                        FROM fact_planned_orders
                        WHERE status = 'proposed'
                        ORDER BY urgency_score DESC, order_value DESC NULLS LAST
                        LIMIT %s
                    """, [limit * 2])
                    for r in cur.fetchall():
                        actions.append({
                            "source": r[0],
                            "item_id": r[1],
                            "loc": r[2],
                            "action_type": r[3],
                            "urgency_score": float(r[4]),
                            "financial_impact": _safe_float(r[5]),
                            "action_label": r[6],
                            "created_at": str(r[7]) if r[7] else None,
                        })
                except psycopg.Error as exc:
                    logger.warning("action-inbox: planned_orders table query failed: %s", exc)

                # --- Source 3: High-risk items (stockout risk >= 60) ---
                # Exclude items that already have an open exception to avoid duplicates
                try:
                    cur.execute("""
                        SELECT 'stockout_risk' AS source,
                               t.item_id, t.loc, 'review_risk' AS action_type,
                               t.stockout_risk_score / 100.0 AS urgency_score,
                               t.monthly_total_holding_cost * 12 AS financial_impact,
                               'Review stockout risk (score: ' || t.stockout_risk_score || ')' AS action_label,
                               t.computed_at AS created_at
                        FROM mv_integrated_planning_targets t
                        WHERE t.stockout_risk_score >= 60
                          AND NOT EXISTS (
                              SELECT 1 FROM fact_replenishment_exceptions e
                              WHERE e.item_id = t.item_id
                                AND e.loc = t.loc
                                AND e.status = 'open'
                          )
                        ORDER BY t.stockout_risk_score DESC, financial_impact DESC NULLS LAST
                        LIMIT %s
                    """, [limit * 2])
                    for r in cur.fetchall():
                        actions.append({
                            "source": r[0],
                            "item_id": r[1],
                            "loc": r[2],
                            "action_type": r[3],
                            "urgency_score": float(r[4]),
                            "financial_impact": _safe_float(r[5]),
                            "action_label": r[6],
                            "created_at": str(r[7]) if r[7] else None,
                        })
                except psycopg.Error as exc:
                    logger.warning("action-inbox: integrated_targets MV query failed: %s", exc)

    except psycopg.Error as exc:
        logger.exception("action-inbox: DB connection failed: %s", exc)
        return {
            "actions": [],
            "summary": {
                "total_actions": 0,
                "urgent_count": 0,
                "high_count": 0,
                "total_financial_impact": 0.0,
            },
        }

    # Apply urgency filter
    if urgency_min > 0:
        actions = [a for a in actions if a["urgency_score"] >= urgency_min]

    # Sort: urgency desc, then financial impact desc
    actions.sort(key=lambda a: (-a["urgency_score"], -(a["financial_impact"] or 0)))
    actions = actions[:limit]

    # Enrich with derived labels + id for the frontend
    for i, a in enumerate(actions):
        a["id"] = f"{a['source']}:{a['item_id']}:{a['loc']}:{i}"
        a["urgency_label"] = _urgency_label(a["urgency_score"])
        a["severity"] = _severity_from_urgency(a["urgency_score"])
        a["title"] = a.pop("action_label")
        a["detail"] = f"{a['action_type'].replace('_', ' ').title()} — {a['item_id']} @ {a['loc']}"
        a["action_url"] = None

    # Build summary from the full (unsliced) set would be ideal, but for
    # simplicity we summarise what we return
    urgent_count = sum(1 for a in actions if a["urgency_score"] >= _URGENCY_THRESHOLDS["urgent"])
    high_count = sum(
        1 for a in actions
        if _URGENCY_THRESHOLDS["high"] <= a["urgency_score"] < _URGENCY_THRESHOLDS["urgent"]
    )
    total_impact = sum(a["financial_impact"] or 0 for a in actions)

    return {
        "actions": actions,
        "summary": {
            "total": len(actions),
            "critical": urgent_count,
            "high": high_count,
            "financial_at_risk": round(total_impact, 2) if total_impact else None,
        },
    }


# ───────────────────────────────────────────────────────────────────────────
# 2. GET /inv-planning/exceptions/{item_id}/{loc}/root-cause (Expert #2)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/exceptions/{item_id}/{loc}/root-cause")
def get_exception_root_cause(
    item_id: str,
    loc: str,
) -> dict:
    """Root cause analysis for a specific item+loc exception."""
    causes: list[dict] = []

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # --- Recent demand signals ---
                try:
                    cur.execute("""
                        SELECT signal_type, signal_strength, alert_priority,
                               projected_stockout, signal_date
                        FROM fact_demand_signals
                        WHERE item_id = %s AND loc = %s
                        ORDER BY signal_date DESC
                        LIMIT 5
                    """, [item_id, loc])
                    rows = cur.fetchall()
                    if rows:
                        latest = rows[0]
                        sev = "high" if latest[2] == "urgent" else ("medium" if latest[2] == "watch" else "low")
                        causes.append({
                            "factor": "demand_signal",
                            "description": f"Latest signal: {latest[0]}, strength {_safe_float(latest[1])}",
                            "severity": sev,
                            "data": {
                                "signal_type": latest[0],
                                "signal_strength": _safe_float(latest[1]),
                                "alert_priority": latest[2],
                                "projected_stockout": latest[3],
                                "signal_count": len(rows),
                            },
                        })
                except psycopg.Error as e:
                    logger.exception("DB error fetching demand signals for root cause: %s", e)

                # --- Forecast accuracy ---
                try:
                    cur.execute("""
                        SELECT
                            SUM(basefcst_pref) AS total_fcst,
                            SUM(tothist_dmd)   AS total_actual
                        FROM agg_forecast_monthly
                        WHERE item_id = %s AND loc = %s
                          AND model_id = 'external'
                          AND month_start >= (CURRENT_DATE - INTERVAL '6 months')
                    """, [item_id, loc])
                    row = cur.fetchone()
                    if row and row[0] is not None and row[1] is not None:
                        fcst, actual = float(row[0]), float(row[1])
                        if abs(actual) > 0:
                            wape = abs(fcst - actual) / abs(actual)
                            bias = (fcst / actual) - 1
                            sev = "high" if wape > 0.3 else ("medium" if wape > 0.15 else "low")
                            causes.append({
                                "factor": "forecast_accuracy",
                                "description": f"WAPE {wape:.1%}, Bias {bias:+.1%} over last 6 months",
                                "severity": sev,
                                "data": {"wape": round(wape, 4), "bias": round(bias, 4),
                                         "total_forecast": round(fcst, 2), "total_actual": round(actual, 2)},
                            })
                except psycopg.Error as e:
                    logger.exception("DB error fetching forecast accuracy for root cause: %s", e)

                # --- Supplier lead time reliability ---
                # Gen-4 Roadmap 1.6 — switched to mv_supplier_po_performance.
                try:
                    cur.execute("""
                        SELECT supplier_id AS supplier_no,
                               supplier_name,
                               avg_lead_time_days AS avg_lt_mean_days,
                               CASE WHEN avg_lead_time_days > 0
                                    THEN stddev_lead_time_days / avg_lead_time_days END AS avg_lt_cv,
                               CASE WHEN reliability_score >= 80 THEN 'stable'
                                    WHEN reliability_score >= 50 THEN 'moderate'
                                    ELSE 'volatile' END AS avg_lt_variability_class
                        FROM mv_supplier_po_performance sp
                        WHERE EXISTS (
                            SELECT 1 FROM dim_item i
                            WHERE i.item_id = %s
                              AND i.supplier_no = sp.supplier_id
                        )
                        LIMIT 1
                    """, [item_id])
                    row = cur.fetchone()
                    if row:
                        cv = _safe_float(row[3])
                        sev = "high" if (cv or 0) > 0.3 else ("medium" if (cv or 0) > 0.15 else "low")
                        causes.append({
                            "factor": "supplier_reliability",
                            "description": f"Supplier {row[1]}: avg LT {_safe_float(row[2])}d, CV {cv}",
                            "severity": sev,
                            "data": {
                                "supplier_no": _s(row[0]),
                                "supplier_name": _s(row[1]),
                                "avg_lt_mean_days": _safe_float(row[2]),
                                "lt_cv": cv,
                                "variability_class": _s(row[4]),
                            },
                        })
                except psycopg.Error as e:
                    logger.exception("DB error fetching supplier performance for root cause: %s", e)

                # --- Open exceptions (current state) ---
                try:
                    cur.execute("""
                        SELECT status, exception_type, severity,
                               current_qty_on_hand, ss_combined, reorder_point,
                               recommended_order_qty
                        FROM fact_replenishment_exceptions
                        WHERE item_id = %s AND loc = %s
                          AND status IN ('open', 'ordered')
                        ORDER BY exception_date DESC
                        LIMIT 3
                    """, [item_id, loc])
                    rows = cur.fetchall()
                    if rows:
                        causes.append({
                            "factor": "open_exceptions",
                            "description": f"{len(rows)} open/ordered exception(s), latest: {rows[0][1]}",
                            "severity": rows[0][2] if rows[0][2] in ("high", "medium", "low") else "high",
                            "data": {
                                "count": len(rows),
                                "types": list({r[1] for r in rows}),
                                "current_qty_on_hand": _safe_float(rows[0][3]),
                                "ss_target": _safe_float(rows[0][4]),
                                "reorder_point": _safe_float(rows[0][5]),
                            },
                        })
                except psycopg.Error as e:
                    logger.exception("DB error fetching exceptions for root cause: %s", e)

    except psycopg.Error as e:
        logger.exception("root-cause: DB connection failed: %s", e)

    return {"causes": causes}


# ───────────────────────────────────────────────────────────────────────────
# 3. GET /inv-planning/segment-dashboard (Expert #3)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/segment-dashboard")
def get_segment_dashboard(
    response: FastAPIResponse,
    segment: str = Query(..., max_length=10, description="ABC-XYZ segment, e.g. AX, CZ"),
) -> dict:
    """Aggregate KPIs and exceptions for an ABC-XYZ segment."""
    set_cache(response, max_age=120)
    segment = segment.strip().upper()

    result: dict = {
        "segment": segment,
        "dfu_count": 0,
        "kpis": {},
        "exceptions": [],
        "policy_distribution": {},
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # --- DFU count & basic stats ---
                try:
                    cur.execute("""
                        SELECT COUNT(*) AS dfu_count
                        FROM dim_sku
                        WHERE abc_xyz_segment = %s
                    """, [segment])
                    row = cur.fetchone()
                    result["dfu_count"] = int(row[0]) if row and row[0] else 0
                except psycopg.Error as e:
                    logger.exception("DB error fetching DFU segment count: %s", e)

                # --- Health score KPIs ---
                try:
                    cur.execute("""
                        SELECT
                            AVG(health_score)  AS avg_health_score,
                            AVG(score_dos_target) AS avg_dos_score
                        FROM mv_inventory_health_score h
                        JOIN dim_sku d ON h.item_id = d.item_id AND h.loc = d.loc
                        WHERE d.abc_xyz_segment = %s
                    """, [segment])
                    row = cur.fetchone()
                    if row:
                        result["kpis"]["avg_health_score"] = _safe_float(row[0])
                        result["kpis"]["avg_dos_score"] = _safe_float(row[1])
                except psycopg.Error as e:
                    logger.exception("DB error fetching health score for segment: %s", e)

                # --- Fill rate ---
                try:
                    cur.execute("""
                        SELECT AVG(fill_rate) AS avg_fill_rate
                        FROM mv_fill_rate_monthly f
                        JOIN dim_sku d ON f.item_id = d.item_id AND f.loc = d.loc
                        WHERE d.abc_xyz_segment = %s
                          AND f.month_start >= (CURRENT_DATE - INTERVAL '3 months')
                    """, [segment])
                    row = cur.fetchone()
                    if row:
                        result["kpis"]["avg_fill_rate"] = _safe_float(row[0])
                except psycopg.Error as e:
                    logger.exception("DB error fetching fill rate for segment: %s", e)

                # --- Below-SS count ---
                try:
                    cur.execute("""
                        SELECT COUNT(*) FROM fact_safety_stock_targets s
                        JOIN dim_sku d ON s.item_id = d.item_id AND s.loc = d.loc
                        WHERE d.abc_xyz_segment = %s
                          AND s.is_below_ss = TRUE
                          AND s.policy_version = 'v1'
                    """, [segment])
                    row = cur.fetchone()
                    result["kpis"]["below_ss_count"] = int(row[0]) if row and row[0] else 0
                except psycopg.Error as e:
                    logger.exception("DB error fetching SS below count for segment: %s", e)

                # --- Exception count ---
                try:
                    cur.execute("""
                        SELECT COUNT(*) FROM fact_replenishment_exceptions e
                        JOIN dim_sku d ON e.item_id = d.item_id AND e.loc = d.loc
                        WHERE d.abc_xyz_segment = %s AND e.status = 'open'
                    """, [segment])
                    row = cur.fetchone()
                    result["kpis"]["open_exception_count"] = int(row[0]) if row and row[0] else 0
                except psycopg.Error as e:
                    logger.exception("DB error fetching exception count for segment: %s", e)

                # --- Policy distribution ---
                try:
                    cur.execute("""
                        SELECT p.policy_id, COUNT(*) AS cnt
                        FROM fact_dfu_policy_assignment p
                        JOIN dim_sku d ON p.item_id = d.item_id AND p.loc = d.loc
                        WHERE d.abc_xyz_segment = %s
                          AND p.is_active = TRUE
                        GROUP BY p.policy_id
                        ORDER BY cnt DESC
                    """, [segment])
                    result["policy_distribution"] = {
                        r[0]: int(r[1]) for r in cur.fetchall()
                    }
                except psycopg.Error as e:
                    logger.exception("DB error fetching policy distribution for segment: %s", e)

                # --- Top exceptions ---
                try:
                    cur.execute("""
                        SELECT e.item_id, e.loc, e.exception_type, e.severity,
                               e.estimated_order_value
                        FROM fact_replenishment_exceptions e
                        JOIN dim_sku d ON e.item_id = d.item_id AND e.loc = d.loc
                        WHERE d.abc_xyz_segment = %s AND e.status = 'open'
                        ORDER BY CASE e.severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                                 WHEN 'medium' THEN 3 ELSE 4 END,
                                 e.estimated_order_value DESC NULLS LAST
                        LIMIT 10
                    """, [segment])
                    result["exceptions"] = [
                        {
                            "item_id": r[0], "loc": r[1],
                            "exception_type": r[2], "severity": r[3],
                            "estimated_order_value": _safe_float(r[4]),
                        }
                        for r in cur.fetchall()
                    ]
                except psycopg.Error as e:
                    logger.exception("DB error fetching top exceptions for segment: %s", e)

    except psycopg.Error as e:
        logger.exception("segment-dashboard: DB connection failed: %s", e)

    return result


# ───────────────────────────────────────────────────────────────────────────
# 4. GET /inv-planning/ss-cost-benefit (Expert #5)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/ss-cost-benefit")
def get_ss_cost_benefit(
    response: FastAPIResponse,
    item: Optional[str] = Query(None, max_length=120),
    loc: Optional[str] = Query(None, max_length=120),
    abc_class: Optional[str] = Query(None, max_length=10),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """Safety stock cost-benefit analysis per DFU."""
    set_cache(response, max_age=120)

    empty = {"items": [], "total": 0, "summary": {
        "total_holding_cost": 0.0, "total_stockout_risk": 0.0,
        "over_stocked_count": 0, "under_stocked_count": 0,
    }}

    where_parts: list[str] = ["s.policy_version = 'v1'"]
    params: list = []

    if item:
        where_parts.append("s.item_id ILIKE %s")
        params.append(f"%{item}%")
    if loc:
        where_parts.append("s.loc ILIKE %s")
        params.append(f"%{loc}%")
    if abc_class:
        where_parts.append("s.abc_vol = %s")
        params.append(abc_class.strip().upper())

    where_sql = " AND ".join(where_parts)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Count total
                cur.execute(f"""
                    SELECT COUNT(*)
                    FROM fact_safety_stock_targets s
                    WHERE {where_sql}
                      AND s.ss_combined IS NOT NULL
                      AND s.ss_combined > 0
                """, params)
                total = int(cur.fetchone()[0])
                if total == 0:
                    return empty

                # Fetch page — use demand_mean_monthly as proxy for unit_cost
                # since unit_cost is not on this table. We estimate holding with
                # a notional unit cost of 1.0 and scale by demand magnitude.
                cur.execute(f"""
                    SELECT s.item_id, s.loc, s.ss_combined,
                           s.demand_mean_monthly, s.service_level_target,
                           s.ss_coverage, s.is_below_ss, s.abc_vol,
                           s.current_qty_on_hand
                    FROM fact_safety_stock_targets s
                    WHERE {where_sql}
                      AND s.ss_combined IS NOT NULL
                      AND s.ss_combined > 0
                    ORDER BY s.ss_combined DESC
                    LIMIT %s OFFSET %s
                """, params + [limit, offset])

                rows = cur.fetchall()

                items = []
                total_holding = 0.0
                total_stockout = 0.0
                over_count = 0
                under_count = 0

                for r in rows:
                    ss_target = float(r[2]) if r[2] else 0.0
                    demand_mean = float(r[3]) if r[3] else 0.0
                    service_level = float(r[4]) if r[4] else 0.95
                    # Estimate unit cost as 1.0 (no unit_cost column); the
                    # relative comparison still works for over/under assessment.
                    unit_cost = 1.0
                    holding_cost = ss_target * unit_cost * 0.25 / 12.0
                    p_stockout = max(0.0, 1.0 - service_level)
                    # Assume 30% margin for lost-sale cost estimation
                    lost_margin = unit_cost * 0.3
                    stockout_cost = p_stockout * lost_margin * demand_mean

                    if holding_cost > 2 * stockout_cost and stockout_cost > 0:
                        assessment = "over-stocked"
                        over_count += 1
                    elif stockout_cost > 2 * holding_cost and holding_cost > 0:
                        assessment = "under-stocked"
                        under_count += 1
                    else:
                        assessment = "balanced"

                    total_holding += holding_cost
                    total_stockout += stockout_cost

                    items.append({
                        "item_id": r[0],
                        "loc": r[1],
                        "ss_target": round(ss_target, 2),
                        "demand_mean_monthly": round(demand_mean, 2),
                        "service_level": round(service_level, 4),
                        "holding_cost_monthly": round(holding_cost, 2),
                        "stockout_cost_monthly": round(stockout_cost, 2),
                        "assessment": assessment,
                        "is_below_ss": bool(r[6]),
                        "abc_vol": _s(r[7]),
                    })

                return {
                    "items": items,
                    "total": total,
                    "summary": {
                        "total_holding_cost": round(total_holding, 2),
                        "total_stockout_risk": round(total_stockout, 2),
                        "over_stocked_count": over_count,
                        "under_stocked_count": under_count,
                    },
                }

    except psycopg.Error as e:
        logger.exception("ss-cost-benefit: query failed: %s", e)
        return empty


# ───────────────────────────────────────────────────────────────────────────
# 5. GET /inv-planning/service-level-waterfall (Expert #6)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/service-level-waterfall")
def get_service_level_waterfall(
    response: FastAPIResponse,
) -> dict:
    """Service level waterfall showing each inventory lever's contribution."""
    set_cache(response, max_age=300)

    steps: list[dict] = []
    achieved_csl = 0.0

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # --- Base forecast accuracy ---
                base_accuracy = 0.0
                try:
                    cur.execute("""
                        SELECT
                            CASE WHEN ABS(SUM(tothist_dmd)) > 0
                                 THEN 100.0 - (100.0 * SUM(ABS(basefcst_pref - tothist_dmd))
                                               / ABS(SUM(tothist_dmd)))
                                 ELSE 0 END AS accuracy
                        FROM agg_forecast_monthly
                        WHERE model_id = 'external'
                          AND month_start >= (CURRENT_DATE - INTERVAL '6 months')
                    """)
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        base_accuracy = max(0.0, min(100.0, float(row[0])))
                except psycopg.Error as e:
                    logger.exception("DB error fetching forecast accuracy for waterfall: %s", e)

                # Express as a service level contribution (scaled to 0-100)
                # Forecast accuracy contributes ~60% of achievable CSL
                base_contribution = base_accuracy * 0.60
                steps.append({
                    "lever": "base_forecast_accuracy",
                    "label": "Forecast Accuracy",
                    "contribution_pct": round(base_contribution, 2),
                    "cumulative_pct": round(base_contribution, 2),
                })

                # --- Safety stock buffer ---
                ss_contribution = 0.0
                try:
                    cur.execute("""
                        SELECT AVG(ss_coverage) AS avg_coverage
                        FROM fact_safety_stock_targets
                        WHERE policy_version = 'v1'
                          AND ss_combined > 0
                    """)
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        avg_cov = float(row[0])
                        # SS coverage > 1.0 means well stocked; contributes up to ~25%
                        ss_contribution = min(25.0, avg_cov * 25.0)
                except psycopg.Error as e:
                    logger.exception("DB error fetching SS coverage for waterfall: %s", e)

                cumulative = base_contribution + ss_contribution
                steps.append({
                    "lever": "ss_buffer_contribution",
                    "label": "Safety Stock Buffer",
                    "contribution_pct": round(ss_contribution, 2),
                    "cumulative_pct": round(cumulative, 2),
                })

                # --- Lead time buffer ---
                # Gen-4 Roadmap 1.6 — switched to mv_supplier_po_performance.
                lt_contribution = 0.0
                try:
                    cur.execute("""
                        SELECT AVG(1.0 - LEAST(
                            CASE WHEN avg_lead_time_days > 0
                                 THEN stddev_lead_time_days / avg_lead_time_days END,
                            1.0)) AS reliability
                        FROM mv_supplier_po_performance
                    """)
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        # More reliable LT → more contribution, up to ~10%
                        lt_contribution = float(row[0]) * 10.0
                except psycopg.Error as e:
                    logger.exception("DB error fetching supplier LT for waterfall: %s", e)

                cumulative += lt_contribution
                steps.append({
                    "lever": "lt_buffer_contribution",
                    "label": "Lead Time Reliability",
                    "contribution_pct": round(lt_contribution, 2),
                    "cumulative_pct": round(cumulative, 2),
                })

                # --- Demand sensing adjustment ---
                sensing_contribution = 0.0
                try:
                    cur.execute("""
                        SELECT
                            COUNT(*) FILTER (WHERE signal_type = 'on_plan') AS on_plan,
                            COUNT(*) AS total
                        FROM fact_demand_signals
                        WHERE signal_date = (SELECT MAX(signal_date) FROM fact_demand_signals)
                    """)
                    row = cur.fetchone()
                    if row and row[1] and int(row[1]) > 0:
                        on_plan_pct = int(row[0]) / int(row[1])
                        # On-plan signals contribute up to ~5%
                        sensing_contribution = on_plan_pct * 5.0
                except psycopg.Error as e:
                    logger.exception("DB error fetching demand signals for waterfall: %s", e)

                cumulative += sensing_contribution
                steps.append({
                    "lever": "sensing_adjustment",
                    "label": "Demand Sensing",
                    "contribution_pct": round(sensing_contribution, 2),
                    "cumulative_pct": round(cumulative, 2),
                })

                achieved_csl = min(100.0, cumulative)

    except psycopg.Error as e:
        logger.exception("service-level-waterfall: DB connection failed: %s", e)

    return {"steps": steps, "achieved_csl": round(achieved_csl, 2)}


# ───────────────────────────────────────────────────────────────────────────
# 6. GET /inv-planning/network-heatmap (Expert #10)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/network-heatmap")
def get_network_heatmap(
    response: FastAPIResponse,
) -> dict:
    """Location x category DOS heatmap from latest inventory snapshot."""
    set_cache(response, max_age=300)

    empty: dict = {"locations": [], "categories": [], "cells": []}

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH latest AS (
                        SELECT MAX(snapshot_date) AS sd
                        FROM fact_inventory_snapshot
                    ),
                    snapshot AS (
                        SELECT
                            inv.item_id,
                            inv.loc,
                            inv.qty_on_hand,
                            inv.mtd_sales,
                            COALESCE(i.prod_class_desc, '(unknown)') AS category
                        FROM fact_inventory_snapshot inv
                        CROSS JOIN latest l
                        JOIN dim_item i ON inv.item_id = i.item_id
                        WHERE inv.snapshot_date = l.sd
                    )
                    SELECT
                        s.loc,
                        s.category,
                        CASE WHEN SUM(s.mtd_sales) > 0
                             THEN SUM(s.qty_on_hand) / (SUM(s.mtd_sales) / GREATEST(EXTRACT(day FROM CURRENT_DATE), 1))
                             ELSE NULL END AS avg_dos,
                        COUNT(DISTINCT s.item_id) AS item_count
                    FROM snapshot s
                    GROUP BY s.loc, s.category
                    ORDER BY s.loc, s.category
                """)
                rows = cur.fetchall()
                if not rows:
                    return empty

                locations_set: set[str] = set()
                categories_set: set[str] = set()
                cells = []
                for r in rows:
                    loc_val = str(r[0])
                    cat_val = str(r[1])
                    dos_val = _safe_float(r[2])
                    locations_set.add(loc_val)
                    categories_set.add(cat_val)
                    cells.append({
                        "location": loc_val,
                        "category": cat_val,
                        "avg_dos": round(dos_val, 1) if dos_val is not None else None,
                        "tier": _tier_for_dos(dos_val),
                        "item_count": int(r[3]),
                    })

                return {
                    "locations": sorted(locations_set),
                    "categories": sorted(categories_set),
                    "cells": cells,
                }

    except psycopg.Error as e:
        logger.exception("network-heatmap: query failed: %s", e)
        return empty


# ───────────────────────────────────────────────────────────────────────────
# 7. GET /inv-planning/planning-scorecard (Expert #20)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/planning-scorecard")
def get_planning_scorecard(
    response: FastAPIResponse,
) -> dict:
    """Trailing effectiveness scorecard with trend analysis."""
    set_cache(response, max_age=300)

    metrics: list[dict] = []

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # --- Forecast accuracy trend (last 6 months) ---
                try:
                    cur.execute("""
                        SELECT month_start,
                               CASE WHEN ABS(SUM(tothist_dmd)) > 0
                                    THEN 100.0 - 100.0 * SUM(ABS(basefcst_pref - tothist_dmd))
                                                       / ABS(SUM(tothist_dmd))
                                    ELSE NULL END AS accuracy
                        FROM agg_forecast_monthly
                        WHERE model_id = 'external'
                          AND month_start >= (CURRENT_DATE - INTERVAL '6 months')
                        GROUP BY month_start
                        ORDER BY month_start
                    """)
                    rows = cur.fetchall()
                    sparkline = [round(float(r[1]), 2) if r[1] is not None else 0.0 for r in rows]
                    current = sparkline[-1] if sparkline else None
                    prior = sparkline[-2] if len(sparkline) >= 2 else None
                    metrics.append({
                        "name": "forecast_accuracy",
                        "current": current,
                        "prior": prior,
                        "trend": _trend_label(current, prior),
                        "sparkline": sparkline,
                    })
                except psycopg.Error as e:
                    logger.exception("DB error fetching forecast accuracy trend: %s", e)

                # --- Fill rate trend ---
                try:
                    cur.execute("""
                        SELECT month_start,
                               AVG(fill_rate) AS avg_fill_rate
                        FROM mv_fill_rate_monthly
                        WHERE month_start >= (CURRENT_DATE - INTERVAL '6 months')
                        GROUP BY month_start
                        ORDER BY month_start
                    """)
                    rows = cur.fetchall()
                    sparkline = [round(float(r[1]), 4) if r[1] is not None else 0.0 for r in rows]
                    current = sparkline[-1] if sparkline else None
                    prior = sparkline[-2] if len(sparkline) >= 2 else None
                    metrics.append({
                        "name": "fill_rate",
                        "current": current,
                        "prior": prior,
                        "trend": _trend_label(current, prior),
                        "sparkline": sparkline,
                    })
                except psycopg.Error as e:
                    logger.exception("DB error fetching fill rate trend: %s", e)

                # --- Exception resolution avg days ---
                try:
                    cur.execute("""
                        SELECT
                            AVG(EXTRACT(epoch FROM (resolved_ts - load_ts)) / 86400.0)
                                FILTER (WHERE status = 'resolved'
                                        AND resolved_ts >= (CURRENT_DATE - INTERVAL '3 months'))
                                AS current_avg_days,
                            AVG(EXTRACT(epoch FROM (resolved_ts - load_ts)) / 86400.0)
                                FILTER (WHERE status = 'resolved'
                                        AND resolved_ts >= (CURRENT_DATE - INTERVAL '6 months')
                                        AND resolved_ts < (CURRENT_DATE - INTERVAL '3 months'))
                                AS prior_avg_days
                        FROM fact_replenishment_exceptions
                    """)
                    row = cur.fetchone()
                    cur_val = _safe_float(row[0]) if row else None
                    pri_val = _safe_float(row[1]) if row else None
                    # For resolution time, lower is better → invert trend
                    trend = "stable"
                    if cur_val is not None and pri_val is not None:
                        if cur_val < pri_val * 0.98:
                            trend = "improving"
                        elif cur_val > pri_val * 1.02:
                            trend = "declining"
                    metrics.append({
                        "name": "exception_resolution_avg_days",
                        "current": round(cur_val, 1) if cur_val is not None else None,
                        "prior": round(pri_val, 1) if pri_val is not None else None,
                        "trend": trend,
                        "sparkline": [],
                    })
                except psycopg.Error as e:
                    logger.exception("DB error fetching exception resolution: %s", e)

                # --- SS optimization (total SS change) ---
                try:
                    cur.execute("""
                        SELECT SUM(ss_combined) FROM fact_safety_stock_targets
                        WHERE policy_version = 'v1'
                    """)
                    row = cur.fetchone()
                    ss_total = _safe_float(row[0]) if row else None
                    metrics.append({
                        "name": "total_safety_stock_units",
                        "current": round(ss_total, 0) if ss_total is not None else None,
                        "prior": None,
                        "trend": "stable",
                        "sparkline": [],
                    })
                except psycopg.Error as e:
                    logger.exception("DB error fetching SS total: %s", e)

                # --- On-time PO % (based on exception ordered vs resolved) ---
                try:
                    cur.execute("""
                        SELECT
                            COUNT(*) FILTER (WHERE status = 'resolved') AS resolved,
                            COUNT(*) FILTER (WHERE status IN ('ordered', 'resolved')) AS total_ordered
                        FROM fact_replenishment_exceptions
                        WHERE load_ts >= (CURRENT_DATE - INTERVAL '3 months')
                    """)
                    row = cur.fetchone()
                    if row and row[1] and int(row[1]) > 0:
                        pct = round(100.0 * int(row[0]) / int(row[1]), 1)
                    else:
                        pct = None
                    metrics.append({
                        "name": "on_time_po_pct",
                        "current": pct,
                        "prior": None,
                        "trend": "stable",
                        "sparkline": [],
                    })
                except psycopg.Error as e:
                    logger.exception("DB error fetching PO on-time: %s", e)

    except psycopg.Error as e:
        logger.exception("planning-scorecard: DB connection failed: %s", e)

    return {"metrics": metrics}


# ───────────────────────────────────────────────────────────────────────────
# 8. GET /inv-planning/cash-flow-timeline (Expert #16)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/cash-flow-timeline")
def get_cash_flow_timeline(
    response: FastAPIResponse,
    months_ahead: int = Query(6, ge=1, le=24),
) -> dict:
    """Monthly cash outflow projection for the next N months."""
    set_cache(response, max_age=300)

    today = get_planning_date()
    # Build month buckets
    month_buckets: list[dict] = []
    for i in range(months_ahead):
        m = today.replace(day=1) + timedelta(days=32 * i)
        m = m.replace(day=1)  # normalize to first-of-month
        month_buckets.append({
            "month": m.strftime("%Y-%m"),
            "po_committed": 0.0,
            "planned_orders": 0.0,
            "ss_investment": 0.0,
            "total": 0.0,
        })

    month_keys = [b["month"] for b in month_buckets]
    month_index = {k: i for i, k in enumerate(month_keys)}

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # --- Committed POs (ordered exceptions with expected receipt) ---
                try:
                    cur.execute("""
                        SELECT TO_CHAR(expected_receipt_date, 'YYYY-MM') AS month,
                               COALESCE(SUM(estimated_order_value), 0) AS total_value
                        FROM fact_replenishment_exceptions
                        WHERE status = 'ordered'
                          AND expected_receipt_date >= %s
                          AND expected_receipt_date < %s
                        GROUP BY 1
                    """, [month_buckets[0]["month"] + "-01",
                          month_buckets[-1]["month"] + "-28"])
                    for r in cur.fetchall():
                        idx = month_index.get(r[0])
                        if idx is not None:
                            month_buckets[idx]["po_committed"] = round(float(r[1]), 2)
                except psycopg.Error as e:
                    logger.exception("DB error fetching PO committed for cash flow: %s", e)

                # --- Planned orders (open exceptions with recommended order) ---
                try:
                    cur.execute("""
                        SELECT TO_CHAR(recommended_order_by, 'YYYY-MM') AS month,
                               COALESCE(SUM(estimated_order_value), 0) AS total_value
                        FROM fact_replenishment_exceptions
                        WHERE status = 'open'
                          AND recommended_order_by >= %s
                          AND recommended_order_by < %s
                        GROUP BY 1
                    """, [month_buckets[0]["month"] + "-01",
                          month_buckets[-1]["month"] + "-28"])
                    for r in cur.fetchall():
                        idx = month_index.get(r[0])
                        if idx is not None:
                            month_buckets[idx]["planned_orders"] = round(float(r[1]), 2)
                except psycopg.Error as e:
                    logger.exception("DB error fetching planned orders for cash flow: %s", e)

                # --- SS investment required (gap * unit_cost proxy) ---
                try:
                    cur.execute("""
                        SELECT COALESCE(SUM(GREATEST(-ss_gap, 0)), 0) AS total_gap
                        FROM fact_safety_stock_targets
                        WHERE policy_version = 'v1'
                          AND ss_gap < 0
                    """)
                    row = cur.fetchone()
                    if row and row[0]:
                        # Spread SS investment across first 3 months
                        ss_total = float(row[0])
                        spread = min(3, len(month_buckets))
                        per_month = round(ss_total / spread, 2)
                        for i in range(spread):
                            month_buckets[i]["ss_investment"] = per_month
                except psycopg.Error as e:
                    logger.exception("DB error fetching SS investment for cash flow: %s", e)

    except psycopg.Error as e:
        logger.exception("cash-flow-timeline: DB connection failed: %s", e)

    # Compute totals
    for b in month_buckets:
        b["total"] = round(b["po_committed"] + b["planned_orders"] + b["ss_investment"], 2)

    return {"months": month_buckets}


# ───────────────────────────────────────────────────────────────────────────
# 9. GET /inv-planning/constrained-optimization (Expert #15)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/constrained-optimization")
def get_constrained_optimization(
    response: FastAPIResponse,
    budget: float = Query(..., gt=0, description="Max inventory investment budget ($)"),
) -> dict:
    """Greedy allocation of budget to maximize service level uplift."""
    set_cache(response, max_age=120)

    result: dict = {
        "budget": budget,
        "allocated": 0.0,
        "items_improved": 0,
        "avg_csl_before": 0.0,
        "avg_csl_after": 0.0,
        "allocations": [],
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Fetch items below SS with their gap and service level info
                cur.execute("""
                    SELECT item_id, loc, ss_combined, current_qty_on_hand,
                           service_level_target, demand_mean_monthly, ss_gap
                    FROM fact_safety_stock_targets
                    WHERE policy_version = 'v1'
                      AND is_below_ss = TRUE
                      AND ss_combined > 0
                      AND demand_mean_monthly > 0
                    ORDER BY ABS(COALESCE(ss_gap, 0)) DESC
                """)
                rows = cur.fetchall()

                if not rows:
                    return result

                # Build candidates: each needs investment = abs(gap) * unit_cost_proxy
                # Rank by service-level-uplift-per-dollar
                candidates = []
                for r in rows:
                    ss_target = float(r[2]) if r[2] else 0.0
                    current_oh = float(r[3]) if r[3] else 0.0
                    slt = float(r[4]) if r[4] else 0.95
                    gap = float(r[6]) if r[6] else 0.0
                    if gap >= 0:
                        continue  # not below SS
                    investment_needed = abs(gap)  # unit_cost proxy = 1.0
                    # CSL before: proportional to coverage
                    csl_before = min(slt, (current_oh / ss_target) * slt) if ss_target > 0 else 0.0
                    csl_after = slt  # if fully funded
                    uplift = csl_after - csl_before
                    uplift_per_dollar = uplift / investment_needed if investment_needed > 0 else 0.0

                    candidates.append({
                        "item_id": r[0],
                        "loc": r[1],
                        "current_ss": round(ss_target, 2),
                        "recommended_ss": round(ss_target, 2),
                        "investment": round(investment_needed, 2),
                        "csl_before": round(csl_before, 4),
                        "csl_after": round(csl_after, 4),
                        "uplift_per_dollar": uplift_per_dollar,
                    })

                # Sort by uplift_per_dollar descending (best bang for buck first)
                candidates.sort(key=lambda c: -c["uplift_per_dollar"])

                # Greedy allocation
                allocated = 0.0
                allocations = []
                sum_csl_before = 0.0
                sum_csl_after = 0.0

                for c in candidates:
                    if allocated + c["investment"] > budget:
                        continue
                    allocated += c["investment"]
                    sum_csl_before += c["csl_before"]
                    sum_csl_after += c["csl_after"]
                    alloc = {k: v for k, v in c.items() if k != "uplift_per_dollar"}
                    allocations.append(alloc)

                n = len(allocations)
                result["allocated"] = round(allocated, 2)
                result["items_improved"] = n
                result["avg_csl_before"] = round(sum_csl_before / n, 4) if n > 0 else 0.0
                result["avg_csl_after"] = round(sum_csl_after / n, 4) if n > 0 else 0.0
                result["allocations"] = allocations[:100]  # cap response size

    except psycopg.Error as e:
        logger.exception("constrained-optimization: query failed: %s", e)

    return result


# ───────────────────────────────────────────────────────────────────────────
# 10. GET /inv-planning/proactive-rebalancing (Expert #9)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/proactive-rebalancing")
def get_proactive_rebalancing(
    response: FastAPIResponse,
    dos_low_threshold: float = Query(10.0, ge=0),
    dos_high_threshold: float = Query(45.0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """Cross-location rebalancing opportunities for the same item."""
    set_cache(response, max_age=300)

    empty: dict = {"opportunities": [], "total_opportunities": 0}

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH latest AS (
                        SELECT MAX(snapshot_date) AS sd
                        FROM fact_inventory_snapshot
                    ),
                    item_loc_dos AS (
                        SELECT
                            inv.item_id,
                            inv.loc,
                            inv.qty_on_hand,
                            CASE WHEN inv.mtd_sales > 0
                                 THEN inv.qty_on_hand
                                      / (inv.mtd_sales / GREATEST(EXTRACT(day FROM inv.snapshot_date), 1))
                                 ELSE NULL END AS dos
                        FROM fact_inventory_snapshot inv
                        CROSS JOIN latest l
                        WHERE inv.snapshot_date = l.sd
                          AND inv.qty_on_hand IS NOT NULL
                    ),
                    deficit AS (
                        SELECT item_id, loc, qty_on_hand, dos
                        FROM item_loc_dos
                        WHERE dos IS NOT NULL AND dos < %s
                    ),
                    surplus AS (
                        SELECT item_id, loc, qty_on_hand, dos
                        FROM item_loc_dos
                        WHERE dos IS NOT NULL AND dos > %s
                    )
                    SELECT
                        d.item_id,
                        s.loc AS from_loc,
                        d.loc AS to_loc,
                        s.dos AS from_dos,
                        d.dos AS to_dos,
                        LEAST(
                            s.qty_on_hand - (s.qty_on_hand * %s / NULLIF(s.dos, 0)),
                            (d.qty_on_hand * %s / NULLIF(d.dos, 0)) - d.qty_on_hand
                        ) AS suggested_qty
                    FROM deficit d
                    JOIN surplus s ON d.item_id = s.item_id AND d.loc <> s.loc
                    WHERE LEAST(
                        s.qty_on_hand - (s.qty_on_hand * %s / NULLIF(s.dos, 0)),
                        (d.qty_on_hand * %s / NULLIF(d.dos, 0)) - d.qty_on_hand
                    ) > 0
                    ORDER BY (s.dos - d.dos) DESC
                    LIMIT %s
                """, [dos_low_threshold, dos_high_threshold,
                      dos_low_threshold, dos_high_threshold,
                      dos_low_threshold, dos_high_threshold,
                      limit])

                rows = cur.fetchall()
                opportunities = []
                for r in rows:
                    from_dos = _safe_float(r[3])
                    to_dos = _safe_float(r[4])
                    urgency = "critical" if (to_dos or 0) < 3 else (
                        "high" if (to_dos or 0) < 7 else "medium"
                    )
                    opportunities.append({
                        "item_id": r[0],
                        "from_loc": r[1],
                        "to_loc": r[2],
                        "from_dos": round(from_dos, 1) if from_dos is not None else None,
                        "to_dos": round(to_dos, 1) if to_dos is not None else None,
                        "suggested_qty": round(float(r[5]), 0) if r[5] is not None else None,
                        "transit_days": None,
                        "urgency": urgency,
                    })

                return {
                    "opportunities": opportunities,
                    "total_opportunities": len(opportunities),
                }

    except psycopg.Error as e:
        logger.exception("proactive-rebalancing: query failed: %s", e)
        return empty


# ───────────────────────────────────────────────────────────────────────────
# GET /inv-planning/daily-briefing — Daily Planner Summary Report (Issue #23)
# ───────────────────────────────────────────────────────────────────────────

@router.get("/inv-planning/daily-briefing")
def get_daily_briefing(
    response: FastAPIResponse,
) -> dict:
    """Auto-generated daily planner briefing.

    Synthesizes data from exceptions, projections, integrated targets,
    and planned orders into a prioritized summary with urgent actions,
    portfolio health, and recommended priorities.
    """
    set_cache(response, max_age=120)

    today = get_planning_date()
    urgent_items: list[dict] = []
    this_week_items: list[dict] = []
    portfolio_items: list[dict] = []
    action_items: list[dict] = []
    stats: dict = {
        "total_skus": 0,
        "below_ss_count": 0,
        "excess_count": 0,
        "total_excess_value": 0.0,
        "total_stockout_risk_value": 0.0,
        "avg_health_score": None,
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:

                # ── Section 1: Urgent items (act within 24 hours) ──

                # 1a. Critical/high open exceptions with stockout risk
                try:
                    cur.execute("""
                        SELECT COUNT(*) AS cnt,
                               COALESCE(SUM(
                                   COALESCE(financial_impact_total, estimated_order_value, 0)
                               ), 0) AS total_value
                        FROM fact_replenishment_exceptions
                        WHERE status = 'open'
                          AND severity IN ('critical', 'high')
                    """)
                    row = cur.fetchone()
                    if row and row[0] and int(row[0]) > 0:
                        cnt = int(row[0])
                        val = _safe_float(row[1]) or 0.0
                        urgent_items.append({
                            "text": f"{cnt} items at stockout risk requiring immediate action",
                            "value": f"${val / 1000:.1f}K at risk",
                            "severity": "critical",
                        })
                        stats["total_stockout_risk_value"] = round(val, 2)
                except psycopg.Error as exc:
                    logger.warning("daily-briefing: critical exceptions query failed: %s", exc)

                # 1b. Past-due planned orders
                try:
                    cur.execute("""
                        SELECT COUNT(*) AS cnt,
                               COALESCE(SUM(order_value), 0) AS total_value
                        FROM fact_planned_orders
                        WHERE status = 'proposed'
                          AND is_past_due = TRUE
                    """)
                    row = cur.fetchone()
                    if row and row[0] and int(row[0]) > 0:
                        cnt = int(row[0])
                        val = _safe_float(row[1]) or 0.0
                        urgent_items.append({
                            "text": f"{cnt} past-due planned orders need approval",
                            "value": f"${val / 1000:.1f}K",
                            "severity": "high",
                        })
                except psycopg.Error as exc:
                    logger.warning("daily-briefing: past-due orders query failed: %s", exc)

                # ── Section 2: This week review items ──

                # 2a. Items below safety stock
                try:
                    cur.execute("""
                        SELECT COUNT(*) AS cnt,
                               COALESCE(SUM(
                                   GREATEST(0, safety_stock_qty - current_qty_on_hand)
                                   * unit_cost
                               ), 0) AS recommended_value
                        FROM mv_integrated_planning_targets
                        WHERE is_below_ss = TRUE
                    """)
                    row = cur.fetchone()
                    if row and row[0] and int(row[0]) > 0:
                        cnt = int(row[0])
                        val = _safe_float(row[1]) or 0.0
                        stats["below_ss_count"] = cnt
                        label = f"${val / 1_000_000:.1f}M" if val >= 1_000_000 else f"${val / 1000:.0f}K"
                        this_week_items.append({
                            "text": f"{cnt} items below safety buffer",
                            "value": f"{label} recommended orders",
                        })
                except psycopg.Error as exc:
                    logger.warning("daily-briefing: below-SS query failed: %s", exc)

                # 2b. Items with high forecast miss (WAPE > 20% in recent period)
                try:
                    cur.execute("""
                        SELECT COUNT(*) AS cnt
                        FROM (
                            SELECT item_id, loc,
                                   SUM(ABS(basefcst_pref - tothist_dmd))
                                       / NULLIF(ABS(SUM(tothist_dmd)), 0) AS wape
                            FROM agg_forecast_monthly
                            WHERE model_id = 'external'
                              AND month_start >= (CURRENT_DATE - INTERVAL '3 months')
                            GROUP BY item_id, loc
                            HAVING SUM(ABS(basefcst_pref - tothist_dmd))
                                       / NULLIF(ABS(SUM(tothist_dmd)), 0) > 0.20
                        ) sub
                    """)
                    row = cur.fetchone()
                    if row and row[0] and int(row[0]) > 0:
                        cnt = int(row[0])
                        this_week_items.append({
                            "text": f"{cnt} items with forecast miss >20%",
                            "value": "Review demand patterns",
                        })
                except psycopg.Error as exc:
                    logger.warning("daily-briefing: forecast miss query failed: %s", exc)

                # ── Section 3: Portfolio health ──

                # 3a. Overall health score + trend
                try:
                    cur.execute("""
                        SELECT AVG(health_score)::numeric(5,1) AS avg_score,
                               COUNT(*) AS total_skus
                        FROM mv_inventory_health_score
                    """)
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        avg_score = float(row[0])
                        total_skus = int(row[1]) if row[1] else 0
                        stats["avg_health_score"] = round(avg_score, 1)
                        stats["total_skus"] = total_skus
                        # Compute healthy tier percentage
                        healthy_pct = round(avg_score) if avg_score else 0
                        portfolio_items.append({
                            "text": f"{healthy_pct}% portfolio health score",
                            "trend": "up" if avg_score >= 70 else "flat",
                            "delta": f"{total_skus:,} SKUs tracked",
                        })
                except psycopg.Error as exc:
                    logger.warning("daily-briefing: health score query failed: %s", exc)

                # 3b. Health by ABC class
                try:
                    cur.execute("""
                        SELECT d.abc_vol,
                               AVG(h.health_score)::numeric(5,1) AS avg_score
                        FROM mv_inventory_health_score h
                        JOIN dim_sku d ON h.item_id = d.item_id AND h.loc = d.loc
                        WHERE d.abc_vol = ANY(%s)
                        GROUP BY d.abc_vol
                        ORDER BY d.abc_vol
                    """, [list(ABC_CLASSES)])
                    for r in cur.fetchall():
                        abc_class = r[0]
                        score = float(r[1]) if r[1] is not None else 0
                        status = "good" if score >= 70 else ("attention" if score >= 50 else "critical")
                        portfolio_items.append({
                            "text": f"{abc_class}-class SKUs: Avg health {score:.0f}/100",
                            "status": status,
                        })
                except psycopg.Error as exc:
                    logger.warning("daily-briefing: ABC health query failed: %s", exc)

                # 3c. Excess inventory stats
                try:
                    cur.execute("""
                        SELECT COUNT(*) AS cnt,
                               COALESCE(SUM(excess_value_usd), 0) AS total_excess
                        FROM mv_integrated_planning_targets
                        WHERE excess_qty > 0
                    """)
                    row = cur.fetchone()
                    if row:
                        stats["excess_count"] = int(row[0]) if row[0] else 0
                        stats["total_excess_value"] = round(float(row[1]), 2) if row[1] else 0.0
                except psycopg.Error as exc:
                    logger.warning("daily-briefing: excess inventory query failed: %s", exc)

                # ── Section 4: Top 3 recommended actions ──
                # Re-use the action-feed logic but just pull top 3 by urgency
                try:
                    cur.execute("""
                        (
                            SELECT 'exception' AS source,
                                   item_id, loc,
                                   'Resolve ' || exception_type || ' for ' || item_id || ' @ ' || loc AS action_text,
                                   COALESCE(financial_impact_total, estimated_order_value, 0) AS impact,
                                   CASE severity
                                       WHEN 'critical' THEN 0.95
                                       WHEN 'high' THEN 0.75
                                       ELSE 0.50
                                   END AS urgency_score,
                                   CASE severity
                                       WHEN 'critical' THEN 'Today'
                                       WHEN 'high' THEN 'Today'
                                       ELSE 'This week'
                                   END AS deadline
                            FROM fact_replenishment_exceptions
                            WHERE status = 'open'
                            ORDER BY urgency_score DESC,
                                     COALESCE(financial_impact_total, estimated_order_value, 0) DESC
                            LIMIT 3
                        )
                        UNION ALL
                        (
                            SELECT 'planned_order' AS source,
                                   item_id, loc,
                                   'Approve order for ' || item_id || ' @ ' || loc AS action_text,
                                   COALESCE(order_value, 0) AS impact,
                                   CASE WHEN is_past_due THEN 0.9 ELSE 0.6 END AS urgency_score,
                                   CASE WHEN is_past_due THEN 'Today' ELSE 'This week' END AS deadline
                            FROM fact_planned_orders
                            WHERE status = 'proposed'
                            ORDER BY urgency_score DESC, order_value DESC NULLS LAST
                            LIMIT 3
                        )
                        ORDER BY urgency_score DESC, impact DESC
                        LIMIT 3
                    """)
                    for i, r in enumerate(cur.fetchall(), start=1):
                        impact_val = _safe_float(r[4]) or 0
                        impact_str = (
                            f"${impact_val / 1000:.0f}K"
                            if impact_val >= 1000
                            else f"${impact_val:.0f}"
                        )
                        action_items.append({
                            "priority": i,
                            "text": r[3],
                            "impact": impact_str,
                            "deadline": r[6],
                        })
                except psycopg.Error as exc:
                    logger.warning("daily-briefing: top actions query failed: %s", exc)

    except psycopg.Error as exc:
        logger.exception("daily-briefing: DB connection failed: %s", exc)

    return {
        "date": today.isoformat(),
        "urgent": {
            "label": "Act within 24 hours",
            "items": urgent_items,
        },
        "this_week": {
            "label": "Review this week",
            "items": this_week_items,
        },
        "portfolio": {
            "label": "Portfolio Health",
            "items": portfolio_items,
        },
        "actions": {
            "label": "Top 3 Recommended Actions",
            "items": action_items,
        },
        "stats": stats,
    }
