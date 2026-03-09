"""Dashboard KPI, alerts, top movers, heatmap endpoints (feature 36)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any
import logging

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from datetime import date

from api.core import get_conn, set_cache
from common.planning_date import get_planning_date

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])


# ---------------------------------------------------------------------------
# GET /dashboard/planning-date
# ---------------------------------------------------------------------------

@router.get("/dashboard/planning-date")
async def get_planning_date_info():
    """Return the current planning date and whether it is frozen (dev mode)."""
    planning = get_planning_date()
    system = date.today()
    return {
        "planning_date": planning.isoformat(),
        "system_date": system.isoformat(),
        "is_frozen": planning != system,
        "days_behind": (system - planning).days,
    }


# ---------------------------------------------------------------------------
# Shared filter builder for dashboard queries
# ---------------------------------------------------------------------------

_MAX_FILTER_VALUES = 50


def _split(raw: str) -> list[str]:
    """Split a comma-separated filter string, capped at _MAX_FILTER_VALUES items."""
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return vals[:_MAX_FILTER_VALUES]


def _dashboard_filter_clause(
    brand: str, category: str, market: str, channel: str,
    item: str = "", location: str = "",
) -> tuple[str, list[Any]]:
    """Build WHERE fragment from global filter params for dashboard queries.

    Note: channel filter is skipped for forecast-based queries because
    fact_external_forecast_monthly has no customer dimension join key.
    Item/location filter on f.dmdunit/f.loc directly (no extra JOINs needed).
    """
    clauses: list[str] = []
    params: list[Any] = []
    if brand.strip():
        brands = _split(brand)
        if brands:
            clauses.append("i.brand_name = ANY(%s)")
            params.append(brands)
    if category.strip():
        cats = _split(category)
        if cats:
            clauses.append("i.class = ANY(%s)")
            params.append(cats)
    if market.strip():
        markets = _split(market)
        if markets:
            clauses.append("lo.state_id = ANY(%s)")
            params.append(markets)
    if item.strip():
        items = _split(item)
        if items:
            clauses.append("f.dmdunit = ANY(%s)")
            params.append(items)
    if location.strip():
        locs = _split(location)
        if locs:
            clauses.append("f.loc = ANY(%s)")
            params.append(locs)
    frag = " AND ".join(clauses) if clauses else ""
    return frag, params


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/dashboard/kpis")
def dashboard_kpis(
    response: FastAPIResponse,
    window: int = Query(default=3, ge=1, le=24),
    brand: str = Query(default=""),
    category: str = Query(default=""),
    market: str = Query(default=""),
    channel: str = Query(default=""),
    item: str = Query(default=""),
    location: str = Query(default=""),
):
    """Aggregated KPI metrics for the overview dashboard."""
    set_cache(response, max_age=120)

    filter_frag, filter_params = _dashboard_filter_clause(brand, category, market, channel, item, location)

    join_clause = ""
    if brand.strip() or category.strip():
        join_clause += " JOIN dim_item i ON f.dmdunit = i.item_no"
    if market.strip():
        join_clause += " JOIN dim_location lo ON f.loc = lo.location_id"

    where_base = "f.startdate >= (CURRENT_DATE - (%s || ' months')::interval) AND f.lag = 0"
    where_prior = (
        "f.startdate >= (CURRENT_DATE - (%s || ' months')::interval) "
        "AND f.startdate < (CURRENT_DATE - (%s || ' months')::interval) AND f.lag = 0"
    )

    where_extra = f" AND {filter_frag}" if filter_frag else ""

    sql_current = f"""
        SELECT
            CASE WHEN ABS(SUM(f.tothist_dmd)) > 0
                 THEN 100.0 - 100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / ABS(SUM(f.tothist_dmd))
                 ELSE NULL END AS accuracy_pct,
            CASE WHEN ABS(SUM(f.tothist_dmd)) > 0
                 THEN 100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / ABS(SUM(f.tothist_dmd))
                 ELSE NULL END AS wape_pct,
            CASE WHEN ABS(SUM(f.tothist_dmd)) > 0
                 THEN 100.0 * (SUM(f.basefcst_pref) / ABS(SUM(f.tothist_dmd)) - 1)
                 ELSE NULL END AS bias_pct,
            SUM(f.basefcst_pref) AS total_forecast,
            SUM(f.tothist_dmd) AS total_actual
        FROM fact_external_forecast_monthly f
        {join_clause}
        WHERE {where_base}{where_extra}
    """

    sql_prior = f"""
        SELECT
            CASE WHEN ABS(SUM(f.tothist_dmd)) > 0
                 THEN 100.0 - 100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / ABS(SUM(f.tothist_dmd))
                 ELSE NULL END AS accuracy_pct,
            CASE WHEN ABS(SUM(f.tothist_dmd)) > 0
                 THEN 100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / ABS(SUM(f.tothist_dmd))
                 ELSE NULL END AS wape_pct,
            CASE WHEN ABS(SUM(f.tothist_dmd)) > 0
                 THEN 100.0 * (SUM(f.basefcst_pref) / ABS(SUM(f.tothist_dmd)) - 1)
                 ELSE NULL END AS bias_pct
        FROM fact_external_forecast_monthly f
        {join_clause}
        WHERE {where_prior}{where_extra}
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql_current, [window] + filter_params)
        row = cur.fetchone()
        accuracy = float(row[0]) if row and row[0] is not None else None
        wape = float(row[1]) if row and row[1] is not None else None
        bias = float(row[2]) if row and row[2] is not None else None
        total_fcst = float(row[3]) if row and row[3] is not None else None
        total_act = float(row[4]) if row and row[4] is not None else None

        cur.execute(sql_prior, [window * 2, window] + filter_params)
        prow = cur.fetchone()
        prior_acc = float(prow[0]) if prow and prow[0] is not None else None
        prior_wape = float(prow[1]) if prow and prow[1] is not None else None
        prior_bias = float(prow[2]) if prow and prow[2] is not None else None

    deltas = {
        "accuracy_pct": round(accuracy - prior_acc, 2) if accuracy is not None and prior_acc is not None else None,
        "wape_pct": round(wape - prior_wape, 2) if wape is not None and prior_wape is not None else None,
        "bias_pct": round(bias - prior_bias, 2) if bias is not None and prior_bias is not None else None,
    }

    return {
        "accuracy_pct": round(accuracy, 2) if accuracy is not None else None,
        "wape_pct": round(wape, 2) if wape is not None else None,
        "bias_pct": round(bias, 2) if bias is not None else None,
        "total_forecast": round(total_fcst, 0) if total_fcst is not None else None,
        "total_actual": round(total_act, 0) if total_act is not None else None,
        "weeks_of_supply": None,
        "window_months": window,
        "deltas": deltas,
    }


@router.get("/dashboard/alerts")
def dashboard_alerts(
    response: FastAPIResponse,
    limit: int = Query(default=10, ge=1, le=50),
    brand: str = Query(default=""),
    category: str = Query(default=""),
    market: str = Query(default=""),
    channel: str = Query(default=""),
    item: str = Query(default=""),
    location: str = Query(default=""),
):
    """Active alerts based on threshold-breaching metrics."""
    set_cache(response, max_age=120)
    alerts: list[dict[str, Any]] = []

    # Build item/location WHERE fragments for alert sub-queries
    _alert_where_parts: list[str] = []
    _alert_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _alert_where_parts.append("dmdunit = ANY(%s)")
            _alert_params.append(items)
    if location.strip():
        locs = [lc.strip() for lc in location.split(",") if lc.strip()]
        if locs:
            _alert_where_parts.append("loc = ANY(%s)")
            _alert_params.append(locs)
    _alert_extra = (" AND " + " AND ".join(_alert_where_parts)) if _alert_where_parts else ""

    # Low accuracy DFUs (< 70%)
    try:
        sql = f"""
            SELECT COUNT(DISTINCT forecast_ck)
            FROM fact_external_forecast_monthly
            WHERE lag = 0
              AND tothist_dmd IS NOT NULL AND tothist_dmd != 0
              AND startdate >= (CURRENT_DATE - INTERVAL '3 months')
              {_alert_extra}
            GROUP BY forecast_ck
            HAVING (100.0 - 100.0 * SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0)) < 70
        """
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, _alert_params)
            low_acc_count = len(cur.fetchall())
        if low_acc_count > 0:
            alerts.append({
                "id": "low-acc-001",
                "type": "low_accuracy",
                "severity": "high" if low_acc_count > 20 else "medium",
                "title": "Low Accuracy DFUs",
                "detail": f"{low_acc_count} DFUs below 70% accuracy",
                "count": low_acc_count,
            })
    except Exception:
        logger.exception("Failed to compute low_accuracy alert")

    # Bias drift (categories with |bias| > 20%)
    _bias_extra_parts: list[str] = []
    _bias_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _bias_extra_parts.append("f.dmdunit = ANY(%s)")
            _bias_params.append(items)
    if location.strip():
        locs = [lc.strip() for lc in location.split(",") if lc.strip()]
        if locs:
            _bias_extra_parts.append("f.loc = ANY(%s)")
            _bias_params.append(locs)
    _bias_extra = (" AND " + " AND ".join(_bias_extra_parts)) if _bias_extra_parts else ""
    try:
        sql = f"""
            SELECT i.class, 100.0 * (SUM(f.basefcst_pref) / NULLIF(ABS(SUM(f.tothist_dmd)), 0) - 1) AS bias
            FROM fact_external_forecast_monthly f
            JOIN dim_item i ON f.dmdunit = i.item_no
            WHERE f.lag = 0
              AND f.tothist_dmd IS NOT NULL AND f.tothist_dmd != 0
              AND f.startdate >= (CURRENT_DATE - INTERVAL '3 months')
              {_bias_extra}
            GROUP BY i.class
            HAVING ABS(100.0 * (SUM(f.basefcst_pref) / NULLIF(ABS(SUM(f.tothist_dmd)), 0) - 1)) > 20
        """
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, _bias_params)
            bias_cats = cur.fetchall()
        if bias_cats:
            alerts.append({
                "id": "bias-drift-001",
                "type": "bias_drift",
                "severity": "medium",
                "title": "Bias Drift",
                "detail": f"{len(bias_cats)} categories with >20% bias",
                "count": len(bias_cats),
            })
    except Exception:
        logger.exception("Failed to compute bias_drift alert")

    # Demand spike (items with >30% period-over-period change)
    _spike_extra_parts: list[str] = []
    _spike_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _spike_extra_parts.append("dmdunit = ANY(%s)")
            _spike_params.append(items)
    if location.strip():
        locs = [lc.strip() for lc in location.split(",") if lc.strip()]
        if locs:
            _spike_extra_parts.append("loc = ANY(%s)")
            _spike_params.append(locs)
    _spike_where = ("WHERE " + " AND ".join(_spike_extra_parts)) if _spike_extra_parts else ""
    try:
        sql = f"""
            SELECT COUNT(*) FROM (
                SELECT dmdunit,
                    SUM(CASE WHEN startdate >= (CURRENT_DATE - INTERVAL '1 month') THEN qty END) AS curr,
                    SUM(CASE WHEN startdate >= (CURRENT_DATE - INTERVAL '2 months')
                              AND startdate < (CURRENT_DATE - INTERVAL '1 month') THEN qty END) AS prev
                FROM fact_sales_monthly
                {_spike_where}
                GROUP BY dmdunit
                HAVING SUM(CASE WHEN startdate >= (CURRENT_DATE - INTERVAL '2 months')
                                  AND startdate < (CURRENT_DATE - INTERVAL '1 month') THEN qty END) > 0
                   AND ABS(
                       (COALESCE(SUM(CASE WHEN startdate >= (CURRENT_DATE - INTERVAL '1 month') THEN qty END), 0)
                        - SUM(CASE WHEN startdate >= (CURRENT_DATE - INTERVAL '2 months')
                                    AND startdate < (CURRENT_DATE - INTERVAL '1 month') THEN qty END))
                       / SUM(CASE WHEN startdate >= (CURRENT_DATE - INTERVAL '2 months')
                                    AND startdate < (CURRENT_DATE - INTERVAL '1 month') THEN qty END)::float
                   ) > 0.3
            ) sub
        """
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, _spike_params)
            spike_row = cur.fetchone()
            spike_count = int(spike_row[0]) if spike_row else 0
        if spike_count > 0:
            alerts.append({
                "id": "demand-spike-001",
                "type": "demand_spike",
                "severity": "low" if spike_count < 10 else "medium",
                "title": "Demand Spikes",
                "detail": f"{spike_count} items with >30% change",
                "count": spike_count,
            })
    except Exception:
        logger.exception("Failed to compute demand_spike alert")

    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    alerts.sort(key=lambda a: sev_order.get(a["severity"], 9))

    return {"alerts": alerts[:limit]}


@router.get("/dashboard/top-movers")
def dashboard_top_movers(
    response: FastAPIResponse,
    limit: int = Query(default=5, ge=1, le=20),
    direction: str = Query(default="both"),
    brand: str = Query(default=""),
    category: str = Query(default=""),
    market: str = Query(default=""),
    channel: str = Query(default=""),
    item: str = Query(default=""),
    location: str = Query(default=""),
):
    """Items with the largest period-over-period volume change."""
    set_cache(response, max_age=120)

    _tm_where_parts: list[str] = []
    _tm_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _tm_where_parts.append("s.dmdunit = ANY(%s)")
            _tm_params.append(items)
    if location.strip():
        locs = [lc.strip() for lc in location.split(",") if lc.strip()]
        if locs:
            _tm_where_parts.append("s.loc = ANY(%s)")
            _tm_params.append(locs)
    _tm_where = ("WHERE " + " AND ".join(_tm_where_parts)) if _tm_where_parts else ""

    sql = f"""
        SELECT
            s.dmdunit,
            i.item_desc,
            COALESCE(SUM(CASE WHEN s.startdate >= (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0) AS curr,
            COALESCE(SUM(CASE WHEN s.startdate >= (CURRENT_DATE - INTERVAL '2 months')
                              AND s.startdate < (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0) AS prev
        FROM fact_sales_monthly s
        JOIN dim_item i ON s.dmdunit = i.item_no
        {_tm_where}
        GROUP BY s.dmdunit, i.item_desc
        HAVING COALESCE(SUM(CASE WHEN s.startdate >= (CURRENT_DATE - INTERVAL '2 months')
                                  AND s.startdate < (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0) > 0
        ORDER BY ABS(
            COALESCE(SUM(CASE WHEN s.startdate >= (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0)
            - COALESCE(SUM(CASE WHEN s.startdate >= (CURRENT_DATE - INTERVAL '2 months')
                                AND s.startdate < (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0)
        ) DESC
        LIMIT %s
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, _tm_params + [limit * 2])
        rows = cur.fetchall()

    movers = []
    for r in rows:
        curr_qty = float(r[2]) if r[2] is not None else 0
        prev_qty = float(r[3]) if r[3] is not None else 0
        delta = curr_qty - prev_qty
        pct_change = (delta / prev_qty * 100) if prev_qty != 0 else 0
        d = "up" if delta >= 0 else "down"
        if direction != "both" and d != direction:
            continue
        movers.append({
            "item_description": r[1] or r[0],
            "delta": round(delta, 0),
            "pct_change": round(pct_change, 1),
            "direction": d,
        })

    return {"movers": movers[:limit]}


@router.get("/dashboard/heatmap")
def dashboard_heatmap(
    response: FastAPIResponse,
    grain: str = Query(default="category"),
    periods: int = Query(default=4, ge=1, le=12),
    brand: str = Query(default=""),
    category: str = Query(default=""),
    market: str = Query(default=""),
    channel: str = Query(default=""),
    item: str = Query(default=""),
    location: str = Query(default=""),
):
    """Performance matrix by category/brand and time period."""
    set_cache(response, max_age=300)

    if grain == "brand":
        group_col = "i.brand_name"
    elif grain == "location":
        group_col = "f.loc"
    else:
        group_col = "i.class"

    _hm_extra_parts: list[str] = []
    _hm_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _hm_extra_parts.append("f.dmdunit = ANY(%s)")
            _hm_params.append(items)
    if location.strip():
        locs = [lc.strip() for lc in location.split(",") if lc.strip()]
        if locs:
            _hm_extra_parts.append("f.loc = ANY(%s)")
            _hm_params.append(locs)
    _hm_extra = (" AND " + " AND ".join(_hm_extra_parts)) if _hm_extra_parts else ""

    sql = f"""
        SELECT
            {group_col} AS label,
            TO_CHAR(f.startdate, 'Mon YY') AS period,
            CASE WHEN ABS(SUM(f.tothist_dmd)) > 0
                 THEN 100.0 - 100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / ABS(SUM(f.tothist_dmd))
                 ELSE NULL END AS accuracy_pct
        FROM fact_external_forecast_monthly f
        JOIN dim_item i ON f.dmdunit = i.item_no
        WHERE f.lag = 0
          AND f.tothist_dmd IS NOT NULL AND f.tothist_dmd != 0
          AND f.startdate >= (CURRENT_DATE - (%s || ' months')::interval)
          {_hm_extra}
        GROUP BY {group_col}, f.startdate, TO_CHAR(f.startdate, 'Mon YY')
        ORDER BY {group_col}, f.startdate
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [periods] + _hm_params)
        rows = cur.fetchall()

    label_map: dict[str, dict[str, float | None]] = OrderedDict()
    period_set: list[str] = []

    for r in rows:
        label = str(r[0]) if r[0] else "Unknown"
        period = str(r[1])
        acc = round(float(r[2]), 1) if r[2] is not None else None
        if label not in label_map:
            label_map[label] = {}
        label_map[label][period] = acc
        if period not in period_set:
            period_set.append(period)

    result_rows = []
    for label, period_vals in label_map.items():
        values = [period_vals.get(p, 0) or 0 for p in period_set]
        result_rows.append({"label": label, "values": values})

    return {"rows": result_rows, "period_labels": period_set, "metric": "accuracy_pct"}
