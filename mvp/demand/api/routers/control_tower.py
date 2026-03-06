"""Control Tower endpoints — IPfeature15: Unified Inventory Control Tower.

Router mounted at /control-tower in api/main.py.
Aggregates KPIs from all upstream IP features into a single command center view.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

router = APIRouter(tags=["control-tower"])


def _f(v: Any) -> float | None:
    return float(v) if v is not None else None


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

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [])
            row = cur.fetchone()
            if not row:
                # Return zeros if view not yet populated
                return {
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
) -> dict:
    """Merged alert list from exceptions + demand signals + health drops.

    Cache: 60s.
    """
    set_cache(response, max_age=60)

    sev_filter = ""
    sev_params: list = []
    if severity:
        sev_params.append(severity)
        sev_filter = f"AND severity = ${len(sev_params)}"

    exc_sql = f"""
        SELECT
            'EXC-' || exception_id::TEXT AS alert_id,
            'exception'                  AS source,
            severity,
            item_no, loc,
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
        LIMIT ${ len(sev_params) + 1}
    """

    ds_params = sev_params + [limit]
    ds_sql = f"""
        SELECT
            'DS-' || item_no || '-' || loc AS alert_id,
            'demand_signal'                AS source,
            CASE alert_priority
                WHEN 'urgent' THEN 'critical'
                WHEN 'watch'  THEN 'high'
                ELSE 'medium' END          AS severity,
            item_no, loc,
            signal_type                    AS alert_type,
            'Demand ' || signal_type || ' vs forecast by ' ||
                ROUND(ABS(demand_vs_forecast_pct)::NUMERIC, 1) || '%' AS description,
            CASE projected_stockout WHEN TRUE THEN 'Place emergency order'
                                    ELSE 'Monitor demand pace' END     AS action,
            load_ts                        AS alert_ts
        FROM fact_demand_signals
        WHERE signal_date = (SELECT MAX(signal_date) FROM fact_demand_signals)
          AND alert_priority IN ('urgent', 'watch')
          { ("AND CASE alert_priority WHEN 'urgent' THEN 'critical' WHEN 'watch' THEN 'high' ELSE 'medium' END = " + f"${len(sev_params)}") if severity else "" }
        ORDER BY
            CASE alert_priority WHEN 'urgent' THEN 1 ELSE 2 END,
            load_ts DESC
        LIMIT ${len(ds_params)}
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
            "item_no":     r[3],
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
) -> dict:
    """Top critical items from health score view, enriched with exception and fill rate data.

    Cache: 120s.
    """
    set_cache(response, max_age=120)

    sql = """
        SELECT
            h.item_no, h.loc,
            d.abc_vol, d.abc_xyz_segment,
            h.health_score, h.health_tier,
            h.ss_coverage, h.is_below_ss,
            h.current_dos, h.dos_min_target, h.dos_max_target,
            -- Exceptions
            (SELECT COUNT(*) FROM fact_replenishment_exceptions e
             WHERE e.item_no = h.item_no AND e.loc = h.loc AND e.status = 'open') AS open_exception_count,
            (SELECT MAX(e.recommended_order_qty) FROM fact_replenishment_exceptions e
             WHERE e.item_no = h.item_no AND e.loc = h.loc AND e.status = 'open'
               AND e.severity = 'critical') AS recommended_order_qty,
            -- Fill rate (latest available month)
            (SELECT fr.fill_rate FROM mv_fill_rate_monthly fr
             WHERE fr.item_no = h.item_no AND fr.loc = h.loc
             ORDER BY fr.month_start DESC LIMIT 1) AS fill_rate_last_3m,
            -- Intra-month stockout days this month
            (SELECT ms.stockout_days FROM mv_intramonth_stockout ms
             WHERE ms.item_no = h.item_no AND ms.loc = h.loc
               AND ms.month_start = DATE_TRUNC('month', CURRENT_DATE)::DATE
             LIMIT 1) AS stockout_days_this_month
        FROM mv_inventory_health_score h
        LEFT JOIN dim_dfu d ON h.item_no = d.dmdunit AND h.loc = d.loc
        ORDER BY h.health_score ASC NULLS LAST
        LIMIT %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [limit])
            rows = cur.fetchall()

    return {
        "items": [
            {
                "item_no":               r[0],
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
            ON fr.item_no = h.item_no AND fr.loc = h.loc
        LEFT JOIN mv_intramonth_stockout ms
            ON fr.item_no = ms.item_no AND fr.loc = ms.loc AND fr.month_start = ms.month_start
        WHERE fr.month_start >= (SELECT MAX(month_start) FROM mv_fill_rate_monthly)
                                - ((%s - 1) || ' months')::INTERVAL
        GROUP BY fr.month_start
        ORDER BY fr.month_start
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [months])
            rows = cur.fetchall()

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
