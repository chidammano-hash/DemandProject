"""Control Tower endpoints — IPfeature15: Unified Inventory Control Tower.

Router mounted at /control-tower in api/main.py.
Aggregates KPIs from all upstream IP features into a single command center view.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import psycopg
from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import _f, get_conn, set_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["control-tower"])




# ---------------------------------------------------------------------------
# GET /control-tower/kpis
# ---------------------------------------------------------------------------

@router.get("/control-tower/kpis")
def get_control_tower_kpis(
    response: FastAPIResponse,
) -> dict:
    """Unified control tower KPIs from all upstream IP features.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    sql = """
        SELECT
            computed_at,
            total_dfus, healthy_count, monitor_count, at_risk_count, critical_count,
            avg_health_score, avg_ss_coverage, below_ss_count, below_ss_pct, avg_portfolio_dos,
            open_exceptions_total, critical_exceptions, high_exceptions, recommended_order_value,
            portfolio_fill_rate_3m, total_shortage_qty_3m,
            urgent_demand_signals, projected_stockouts_today,
            items_with_stockout_this_month, extended_stockouts_this_month
        FROM mv_control_tower_kpis
        LIMIT 1
    """

    empty_payload = {
        "computed_at": None,
        "health": {k: 0 for k in [
            "total_dfus", "healthy_count", "monitor_count", "at_risk_count",
            "critical_count", "avg_health_score", "avg_ss_coverage",
            "below_ss_count", "below_ss_pct", "avg_portfolio_dos"
        ]},
        "exceptions": {k: 0 for k in [
            "open_exceptions_total", "critical_exceptions",
            "high_exceptions", "recommended_order_value"
        ]},
        "fill_rate": {"portfolio_fill_rate_3m": None, "total_shortage_qty_3m": 0},
        "demand_signals": {"urgent_demand_signals": 0, "projected_stockouts_today": 0},
        "intramonth": {"items_with_stockout_this_month": 0, "extended_stockouts_this_month": 0},
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [])
                row = cur.fetchone()
    except psycopg.errors.ObjectNotInPrerequisiteState as exc:
        logger.warning("control-tower/kpis: MV not populated (%s)", exc)
        return {**empty_payload, "warning": "mv_control_tower_kpis not yet refreshed. Run `make refresh-mvs-tiered`."}

    if not row:
        # View exists and is populated, just has no rows
        return empty_payload

    return {
        "computed_at": str(row[0]) if row[0] else None,
        "health": {
            "total_dfus":          int(row[1] or 0),
            "healthy_count":       int(row[2] or 0),
            "monitor_count":       int(row[3] or 0),
            "at_risk_count":       int(row[4] or 0),
            "critical_count":      int(row[5] or 0),
            "avg_health_score":    _f(row[6]),
            "avg_ss_coverage":     _f(row[7]),
            "below_ss_count":      int(row[8] or 0),
            "below_ss_pct":        _f(row[9]),
            "avg_portfolio_dos":   _f(row[10]),
        },
        "exceptions": {
            "open_exceptions_total":   int(row[11] or 0),
            "critical_exceptions":     int(row[12] or 0),
            "high_exceptions":         int(row[13] or 0),
            "recommended_order_value": _f(row[14]),
        },
        "fill_rate": {
            "portfolio_fill_rate_3m": _f(row[15]),
            "total_shortage_qty_3m":  _f(row[16]),
        },
        "demand_signals": {
            "urgent_demand_signals":    int(row[17] or 0),
            "projected_stockouts_today":int(row[18] or 0),
        },
        "intramonth": {
            "items_with_stockout_this_month": int(row[19] or 0),
            "extended_stockouts_this_month":  int(row[20] or 0),
        },
    }


# ---------------------------------------------------------------------------
# GET /control-tower/alerts
# ---------------------------------------------------------------------------

@router.get("/control-tower/alerts")
def get_control_tower_alerts(
    response: FastAPIResponse,
    limit: int = Query(20, ge=1, le=100),
    severity: Optional[str] = Query(None, max_length=20),
    item: Optional[str] = Query(None, max_length=100),
    location: Optional[str] = Query(None, max_length=100),
    brand: Optional[str] = Query(None, max_length=100),
    category: Optional[str] = Query(None, max_length=100),
    market: Optional[str] = Query(None, max_length=100),
) -> dict:
    """Merged alert list from exceptions + demand signals + health drops.

    Cache: 60s.
    """
    set_cache(response, max_age=60)

    sev_filter = ""
    sev_params: list = []
    if severity:
        sev_params.append(severity)
        sev_filter = "AND severity = %s"

    exc_sql = f"""
        SELECT
            'EXC-' || exception_id::TEXT AS alert_id,
            'exception'                  AS source,
            severity,
            item_id, loc,
            exception_type               AS alert_type,
            'Exception: ' || exception_type || ' — ' || COALESCE(notes, '')  AS description,
            'Review and action required' AS action,
            exception_date::TIMESTAMPTZ  AS alert_ts
        FROM fact_replenishment_exceptions
        WHERE status = 'open'
        {sev_filter}
        ORDER BY
            CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                          WHEN 'medium' THEN 3 ELSE 4 END,
            exception_date DESC
        LIMIT %s
    """

    ds_sev_filter = ""
    if severity:
        ds_sev_filter = ("AND CASE alert_priority WHEN 'urgent' THEN 'critical' "
                         "WHEN 'watch' THEN 'high' ELSE 'medium' END = %s")
    ds_params = sev_params + [limit]
    ds_sql = f"""
        SELECT
            'DS-' || item_id || '-' || loc AS alert_id,
            'demand_signal'                AS source,
            CASE alert_priority
                WHEN 'urgent' THEN 'critical'
                WHEN 'watch'  THEN 'high'
                ELSE 'medium' END          AS severity,
            item_id, loc,
            signal_type                    AS alert_type,
            'Demand ' || signal_type || ' vs forecast by ' ||
                ROUND(ABS(demand_vs_forecast_pct)::NUMERIC, 1) || '%%' AS description,
            CASE projected_stockout WHEN TRUE THEN 'Place emergency order'
                                    ELSE 'Monitor demand pace' END     AS action,
            load_ts                        AS alert_ts
        FROM fact_demand_signals
        WHERE signal_date = (SELECT MAX(signal_date) FROM fact_demand_signals)
          AND alert_priority IN ('urgent', 'watch')
          {ds_sev_filter}
        ORDER BY
            CASE alert_priority WHEN 'urgent' THEN 1 ELSE 2 END,
            load_ts DESC
        LIMIT %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(exc_sql, sev_params + [limit])
            exc_rows = cur.fetchall()
            cur.execute(ds_sql, ds_params)
            ds_rows = cur.fetchall()

    def _row_to_alert(r: tuple) -> dict:
        return {
            "alert_id":    r[0],
            "source":      r[1],
            "severity":    r[2],
            "item_id":     r[3],
            "loc":         r[4],
            "alert_type":  r[5],
            "description": r[6],
            "action":      r[7],
            "alert_ts":    str(r[8]) if r[8] else None,
            "abc_vol":     None,
        }

    alerts = [_row_to_alert(r) for r in exc_rows] + [_row_to_alert(r) for r in ds_rows]
    # Sort: critical first, then high, then by timestamp desc
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    alerts.sort(key=lambda a: (severity_order.get(a["severity"] or "", 4), -(hash(a["alert_ts"] or ""))))
    alerts = alerts[:limit]

    return {"total": len(alerts), "alerts": alerts}


# ---------------------------------------------------------------------------
# GET /control-tower/top-critical
# ---------------------------------------------------------------------------

@router.get("/control-tower/top-critical")
def get_top_critical_items(
    response: FastAPIResponse,
    limit: int = Query(10, ge=1, le=50),
    item: Optional[str] = Query(None, max_length=100),
    location: Optional[str] = Query(None, max_length=100),
    brand: Optional[str] = Query(None, max_length=100),
    category: Optional[str] = Query(None, max_length=100),
    market: Optional[str] = Query(None, max_length=100),
) -> dict:
    """Top critical items from health score view, enriched with exception and fill rate data.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    where_clauses = []
    params: list[Any] = []
    if item:
        params.append(item)
        where_clauses.append("h.item_id = %s")
    if location:
        params.append(location)
        where_clauses.append("h.loc = %s")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        WITH ranked_health AS (
            SELECT
                h.item_id, h.loc,
                d.abc_vol, d.abc_xyz_segment,
                h.health_score, h.health_tier,
                h.ss_coverage, h.is_below_ss,
                h.current_dos, h.dos_min_target, h.dos_max_target
            FROM mv_inventory_health_score h
            LEFT JOIN dim_sku d ON h.item_id = d.item_id AND h.loc = d.loc
            {where_sql}
            ORDER BY h.health_score ASC NULLS LAST
            LIMIT %s
        ),
        exc_agg AS (
            SELECT
                e.item_id, e.loc,
                COUNT(*) AS open_exception_count,
                MAX(CASE WHEN e.severity = 'critical'
                         THEN e.recommended_order_qty END) AS recommended_order_qty
            FROM fact_replenishment_exceptions e
            INNER JOIN ranked_health rh
                ON e.item_id = rh.item_id AND e.loc = rh.loc
            WHERE e.status = 'open'
            GROUP BY e.item_id, e.loc
        ),
        latest_fr AS (
            SELECT DISTINCT ON (fr.item_id, fr.loc)
                fr.item_id, fr.loc, fr.fill_rate
            FROM mv_fill_rate_monthly fr
            INNER JOIN ranked_health rh
                ON fr.item_id = rh.item_id AND fr.loc = rh.loc
            ORDER BY fr.item_id, fr.loc, fr.month_start DESC
        ),
        cur_stockout AS (
            SELECT ms.item_id, ms.loc, ms.stockout_days
            FROM mv_intramonth_stockout ms
            INNER JOIN ranked_health rh
                ON ms.item_id = rh.item_id AND ms.loc = rh.loc
            WHERE ms.month_start = DATE_TRUNC('month', CURRENT_DATE)::DATE
        )
        SELECT
            rh.item_id, rh.loc,
            rh.abc_vol, rh.abc_xyz_segment,
            rh.health_score, rh.health_tier,
            rh.ss_coverage, rh.is_below_ss,
            rh.current_dos, rh.dos_min_target, rh.dos_max_target,
            COALESCE(ea.open_exception_count, 0)  AS open_exception_count,
            ea.recommended_order_qty,
            lfr.fill_rate                          AS fill_rate_last_3m,
            cs.stockout_days                       AS stockout_days_this_month
        FROM ranked_health rh
        LEFT JOIN exc_agg ea      ON rh.item_id = ea.item_id AND rh.loc = ea.loc
        LEFT JOIN latest_fr lfr   ON rh.item_id = lfr.item_id AND rh.loc = lfr.loc
        LEFT JOIN cur_stockout cs ON rh.item_id = cs.item_id AND rh.loc = cs.loc
        ORDER BY rh.health_score ASC NULLS LAST
    """
    params.append(limit)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
    except psycopg.errors.ObjectNotInPrerequisiteState as exc:
        logger.warning("control-tower/drill-down: MV not populated (%s)", exc)
        return {"items": [], "warning": "Upstream MV not refreshed. Run `make refresh-mvs-tiered`."}

    return {
        "items": [
            {
                "item_id":               r[0],
                "loc":                   r[1],
                "abc_vol":               r[2],
                "abc_xyz_segment":       r[3],
                "health_score":          _f(r[4]),
                "health_tier":           r[5],
                "ss_coverage":           _f(r[6]),
                "is_below_ss":           bool(r[7]) if r[7] is not None else False,
                "current_dos":           _f(r[8]),
                "target_dos_min":        _f(r[9]),
                "target_dos_max":        _f(r[10]),
                "open_exception_count":  int(r[11] or 0),
                "recommended_order_qty": _f(r[12]),
                "fill_rate_last_3m":     _f(r[13]),
                "stockout_days_this_month": int(r[14]) if r[14] is not None else 0,
            }
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# GET /control-tower/kpis-financial — Gen-4 Roadmap 1.7
# ---------------------------------------------------------------------------


@router.get("/control-tower/kpis-financial")
def get_control_tower_kpis_financial(
    response: FastAPIResponse,
) -> dict:
    """$-denominated Control Tower KPIs.

    Joins unit cost (``fact_eoq_targets.unit_cost``) against inventory / below-SS
    / exception unit counts so the Command Center can render dollar-weighted
    KPIs next to the existing unit-count KPIs.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    # Aggregate in-stock value, below-SS exposure, excess exposure, open
    # exception dollar impact. All numbers are denominated in USD (or the unit
    # currency of fact_eoq_targets.unit_cost, whichever the caller configured).
    sql = """
        WITH inv_value AS (
            SELECT
                SUM(h.eom_qty_on_hand * COALESCE(eoq.unit_cost, 0)) AS inventory_value,
                SUM(CASE WHEN h.is_below_ss
                         THEN GREATEST(0, (h.dos_min_target - h.current_dos))
                              * h.avg_daily_sls
                              * COALESCE(eoq.unit_cost, 0)
                         ELSE 0 END)                                AS below_ss_value_gap,
                SUM(CASE WHEN h.current_dos > h.dos_max_target
                         THEN (h.current_dos - h.dos_max_target)
                              * h.avg_daily_sls
                              * COALESCE(eoq.unit_cost, 0)
                         ELSE 0 END)                                AS excess_value
            FROM mv_inventory_health_score h
            LEFT JOIN fact_eoq_targets eoq
                ON eoq.item_id = h.item_id AND eoq.loc = h.loc
        ),
        exc_value AS (
            SELECT
                COALESCE(SUM(financial_impact_total), 0)            AS open_exception_value,
                COALESCE(SUM(CASE WHEN severity = 'critical'
                                   THEN financial_impact_total ELSE 0 END), 0)
                                                                    AS critical_exception_value,
                COALESCE(SUM(loss_of_sales_7d), 0)                  AS loss_of_sales_7d_value,
                COALESCE(SUM(monthly_holding_cost), 0)              AS monthly_holding_cost
            FROM fact_replenishment_exceptions
            WHERE status = 'open'
        ),
        fr_value AS (
            -- 3-month weighted fill rate revenue-style value: ordered_value * (1 - fill_rate)
            SELECT
                COALESCE(SUM(
                    GREATEST(0, fr.total_ordered - fr.total_shipped)
                    * COALESCE(eoq.unit_cost, 0)
                ), 0)                                               AS shortage_value_3m
            FROM mv_fill_rate_monthly fr
            LEFT JOIN fact_eoq_targets eoq
                ON eoq.item_id = fr.item_id AND eoq.loc = fr.loc
            WHERE fr.month_start >= (
                SELECT MAX(month_start) FROM mv_fill_rate_monthly
            ) - INTERVAL '2 months'
        )
        SELECT
            i.inventory_value,
            i.below_ss_value_gap,
            i.excess_value,
            e.open_exception_value,
            e.critical_exception_value,
            e.loss_of_sales_7d_value,
            e.monthly_holding_cost,
            f.shortage_value_3m
        FROM inv_value i, exc_value e, fr_value f
    """

    empty_fin = {
        "currency": "USD",
        "inventory_value": 0.0,
        "below_ss_value_gap": 0.0,
        "excess_value": 0.0,
        "open_exception_value": 0.0,
        "critical_exception_value": 0.0,
        "loss_of_sales_7d_value": 0.0,
        "monthly_holding_cost": 0.0,
        "shortage_value_3m": 0.0,
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [])
                row = cur.fetchone()
    except psycopg.errors.ObjectNotInPrerequisiteState as exc:
        logger.warning("control-tower/kpis-financial: MV not populated (%s)", exc)
        return {**empty_fin, "warning": "Upstream MV not refreshed. Run `make refresh-mvs-tiered`."}

    if not row:
        return empty_fin

    return {
        "currency": "USD",
        "inventory_value":          _f(row[0]) or 0.0,
        "below_ss_value_gap":       _f(row[1]) or 0.0,
        "excess_value":             _f(row[2]) or 0.0,
        "open_exception_value":     _f(row[3]) or 0.0,
        "critical_exception_value": _f(row[4]) or 0.0,
        "loss_of_sales_7d_value":   _f(row[5]) or 0.0,
        "monthly_holding_cost":     _f(row[6]) or 0.0,
        "shortage_value_3m":        _f(row[7]) or 0.0,
    }


# ---------------------------------------------------------------------------
# GET /control-tower/trend
# ---------------------------------------------------------------------------

@router.get("/control-tower/trend")
def get_control_tower_trend(
    response: FastAPIResponse,
    months: int = Query(6, ge=1, le=24),
) -> dict:
    """6-month portfolio trend: health, fill rate, stockout rate, below-SS%.

    Cache: 3600s.
    """
    set_cache(response, max_age=3600)

    sql = """
        SELECT
            fr.month_start,
            AVG(h.health_score)        AS avg_health_score,
            SUM(fr.total_shipped)::NUMERIC / NULLIF(SUM(fr.total_ordered), 0)
                                       AS fill_rate,
            AVG(ms.stockout_day_rate)  AS stockout_day_rate,
            NULL::NUMERIC              AS below_ss_pct,
            AVG(ms.avg_qty_on_hand)    AS avg_dos
        FROM mv_fill_rate_monthly fr
        LEFT JOIN mv_inventory_health_score h
            ON fr.item_id = h.item_id AND fr.loc = h.loc
        LEFT JOIN mv_intramonth_stockout ms
            ON fr.item_id = ms.item_id AND fr.loc = ms.loc AND fr.month_start = ms.month_start
        WHERE fr.month_start >= (SELECT MAX(month_start) FROM mv_fill_rate_monthly)
                                - ((%s - 1) || ' months')::INTERVAL
        GROUP BY fr.month_start
        ORDER BY fr.month_start
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [months])
                rows = cur.fetchall()
    except psycopg.errors.ObjectNotInPrerequisiteState as exc:
        # An upstream MV (mv_fill_rate_monthly / mv_inventory_health_score /
        # mv_intramonth_stockout) has been created but not yet refreshed.
        # Return an empty trend with a hint so the UI can render the empty
        # state instead of a 500. Run `make refresh-mvs-tiered` to populate.
        logger.warning("control-tower/trend: MV not populated (%s)", exc)
        return {
            "trend": [],
            "warning": "Upstream materialized view not yet refreshed. Run `make refresh-mvs-tiered`.",
        }

    return {
        "trend": [
            {
                "month_start":       str(r[0]),
                "avg_health_score":  _f(r[1]),
                "fill_rate":         _f(r[2]),
                "stockout_day_rate": _f(r[3]),
                "below_ss_pct":      _f(r[4]),
                "avg_dos":           _f(r[5]),
            }
            for r in rows
        ]
    }
