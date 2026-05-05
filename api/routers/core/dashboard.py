"""Dashboard KPI, alerts, top movers, heatmap endpoints (feature 36)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any
import logging
import math
import threading

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from datetime import date

import pgeocode
import psycopg

from api.core import get_conn, set_cache
from common.core.planning_date import get_planning_date

# Lazy-initialized US zip code geocoder (pgeocode downloads ~2MB file on first use)
_nomi: pgeocode.Nominatim | None = None
_nomi_lock = threading.Lock()


def _get_nomi() -> pgeocode.Nominatim:
    global _nomi
    if _nomi is None:
        with _nomi_lock:
            if _nomi is None:
                _nomi = pgeocode.Nominatim("US")
    return _nomi

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
    item: str = "", location: str = "", cluster_assignment: str = "",
) -> tuple[str, list[Any]]:
    """Build WHERE fragment from global filter params for dashboard queries.

    Note: channel filter is skipped for forecast-based queries because
    fact_external_forecast_monthly has no customer dimension join key.
    Item/location filter on f.item_id/f.loc directly (no extra JOINs needed).
    Cluster filter uses EXISTS subquery against dim_sku.
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
            clauses.append("f.item_id = ANY(%s)")
            params.append(items)
    if location.strip():
        locs = _split(location)
        if locs:
            clauses.append("f.loc = ANY(%s)")
            params.append(locs)
    if cluster_assignment.strip():
        clusters = _split(cluster_assignment)
        if clusters:
            clauses.append(
                "EXISTS (SELECT 1 FROM dim_sku _d WHERE _d.item_id = f.item_id AND _d.loc = f.loc AND _d.cluster_assignment = ANY(%s))"
            )
            params.append(clusters)
    frag = " AND ".join(clauses) if clauses else ""
    return frag, params


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/dashboard/kpis")
def dashboard_kpis(
    response: FastAPIResponse,
    window: int = Query(default=3, ge=1, le=24),
    model: str = Query(default="external"),
    brand: str = Query(default=""),
    category: str = Query(default=""),
    market: str = Query(default=""),
    channel: str = Query(default=""),
    item: str = Query(default=""),
    location: str = Query(default=""),
    cluster_assignment: str = Query(default=""),
    time_grain: str = Query(default="month"),
):
    """Aggregated KPI metrics for the overview dashboard."""
    set_cache(response, max_age=120)
    pd = get_planning_date().isoformat()

    filter_frag, filter_params = _dashboard_filter_clause(brand, category, market, channel, item, location, cluster_assignment)

    join_clause = ""
    if brand.strip() or category.strip():
        join_clause += " JOIN dim_item i ON f.item_id = i.item_id"
    if market.strip():
        join_clause += " JOIN dim_location lo ON f.loc = lo.location_id"

    where_base = (
        f"f.model_id = %s"
        f" AND f.tothist_dmd IS NOT NULL AND f.tothist_dmd != 0"
        f" AND f.basefcst_pref IS NOT NULL AND f.basefcst_pref != 0"
        f" AND f.startdate >= ('{pd}'::date - (%s || ' months')::interval)"
    )
    where_prior = (
        f"f.model_id = %s"
        f" AND f.tothist_dmd IS NOT NULL AND f.tothist_dmd != 0"
        f" AND f.basefcst_pref IS NOT NULL AND f.basefcst_pref != 0"
        f" AND f.startdate >= ('{pd}'::date - (%s || ' months')::interval)"
        f" AND f.startdate < ('{pd}'::date - (%s || ' months')::interval)"
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
        cur.execute(sql_current, [model, window] + filter_params)
        row = cur.fetchone()
        accuracy = float(row[0]) if row and row[0] is not None else None
        wape = float(row[1]) if row and row[1] is not None else None
        bias = float(row[2]) if row and row[2] is not None else None
        total_fcst = float(row[3]) if row and row[3] is not None else None
        total_act = float(row[4]) if row and row[4] is not None else None

        cur.execute(sql_prior, [model, window * 2, window] + filter_params)
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
    cluster_assignment: str = Query(default=""),
    time_grain: str = Query(default="month"),
):
    """Active alerts based on threshold-breaching metrics."""
    set_cache(response, max_age=120)
    pd = get_planning_date().isoformat()
    alerts: list[dict[str, Any]] = []

    # Build item/location/cluster WHERE fragments for alert sub-queries
    _alert_where_parts: list[str] = []
    _alert_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _alert_where_parts.append("item_id = ANY(%s)")
            _alert_params.append(items)
    if location.strip():
        locs = [lc.strip() for lc in location.split(",") if lc.strip()]
        if locs:
            _alert_where_parts.append("loc = ANY(%s)")
            _alert_params.append(locs)
    if cluster_assignment.strip():
        clusters = [c.strip() for c in cluster_assignment.split(",") if c.strip()]
        if clusters:
            _alert_where_parts.append(
                "EXISTS (SELECT 1 FROM dim_sku _d WHERE _d.item_id = item_id AND _d.loc = loc AND _d.cluster_assignment = ANY(%s))"
            )
            _alert_params.append(clusters)
    _alert_extra = (" AND " + " AND ".join(_alert_where_parts)) if _alert_where_parts else ""

    # Low accuracy DFUs (< 70%)
    try:
        sql = f"""
            SELECT COUNT(DISTINCT forecast_ck)
            FROM fact_external_forecast_monthly
            WHERE model_id = 'external'
              AND tothist_dmd IS NOT NULL AND tothist_dmd != 0
              AND basefcst_pref IS NOT NULL
              AND startdate >= ('{pd}'::date - INTERVAL '3 months')
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
    except psycopg.Error:
        logger.exception("Failed to compute low_accuracy alert")

    # Bias drift (categories with |bias| > 20%)
    _bias_extra_parts: list[str] = []
    _bias_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _bias_extra_parts.append("f.item_id = ANY(%s)")
            _bias_params.append(items)
    if location.strip():
        locs = [lc.strip() for lc in location.split(",") if lc.strip()]
        if locs:
            _bias_extra_parts.append("f.loc = ANY(%s)")
            _bias_params.append(locs)
    if cluster_assignment.strip():
        clusters = [c.strip() for c in cluster_assignment.split(",") if c.strip()]
        if clusters:
            _bias_extra_parts.append(
                "EXISTS (SELECT 1 FROM dim_sku _d WHERE _d.item_id = f.item_id AND _d.loc = f.loc AND _d.cluster_assignment = ANY(%s))"
            )
            _bias_params.append(clusters)
    _bias_extra = (" AND " + " AND ".join(_bias_extra_parts)) if _bias_extra_parts else ""
    try:
        sql = f"""
            SELECT i.class, 100.0 * (SUM(f.basefcst_pref) / NULLIF(ABS(SUM(f.tothist_dmd)), 0) - 1) AS bias
            FROM fact_external_forecast_monthly f
            JOIN dim_item i ON f.item_id = i.item_id
            WHERE f.model_id = 'external'
              AND f.tothist_dmd IS NOT NULL AND f.tothist_dmd != 0
              AND f.basefcst_pref IS NOT NULL
              AND f.startdate >= ('{pd}'::date - INTERVAL '3 months')
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
    except psycopg.Error:
        logger.exception("Failed to compute bias_drift alert")

    # Demand spike (items with >30% period-over-period change)
    _spike_extra_parts: list[str] = []
    _spike_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _spike_extra_parts.append("item_id = ANY(%s)")
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
                SELECT item_id,
                    SUM(CASE WHEN startdate >= ('{pd}'::date - INTERVAL '1 month') THEN qty END) AS curr,
                    SUM(CASE WHEN startdate >= ('{pd}'::date - INTERVAL '2 months')
                              AND startdate < ('{pd}'::date - INTERVAL '1 month') THEN qty END) AS prev
                FROM fact_sales_monthly
                {_spike_where}
                GROUP BY item_id
                HAVING SUM(CASE WHEN startdate >= ('{pd}'::date - INTERVAL '2 months')
                                  AND startdate < ('{pd}'::date - INTERVAL '1 month') THEN qty END) > 0
                   AND ABS(
                       (COALESCE(SUM(CASE WHEN startdate >= ('{pd}'::date - INTERVAL '1 month') THEN qty END), 0)
                        - SUM(CASE WHEN startdate >= ('{pd}'::date - INTERVAL '2 months')
                                    AND startdate < ('{pd}'::date - INTERVAL '1 month') THEN qty END))
                       / SUM(CASE WHEN startdate >= ('{pd}'::date - INTERVAL '2 months')
                                    AND startdate < ('{pd}'::date - INTERVAL '1 month') THEN qty END)::float
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
    except psycopg.Error:
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
    cluster_assignment: str = Query(default=""),
):
    """Items with the largest period-over-period volume change."""
    set_cache(response, max_age=120)
    pd = get_planning_date().isoformat()

    _tm_where_parts: list[str] = []
    _tm_params: list[Any] = []
    if item.strip():
        items = [it.strip() for it in item.split(",") if it.strip()]
        if items:
            _tm_where_parts.append("s.item_id = ANY(%s)")
            _tm_params.append(items)
    if location.strip():
        locs = [lc.strip() for lc in location.split(",") if lc.strip()]
        if locs:
            _tm_where_parts.append("s.loc = ANY(%s)")
            _tm_params.append(locs)
    if cluster_assignment.strip():
        clusters = [c.strip() for c in cluster_assignment.split(",") if c.strip()]
        if clusters:
            _tm_where_parts.append(
                "EXISTS (SELECT 1 FROM dim_sku _d WHERE _d.item_id = s.item_id AND _d.loc = s.loc AND _d.cluster_assignment = ANY(%s))"
            )
            _tm_params.append(clusters)
    _tm_where = ("WHERE " + " AND ".join(_tm_where_parts)) if _tm_where_parts else ""

    sql = f"""
        SELECT
            s.item_id,
            i.item_desc,
            COALESCE(SUM(CASE WHEN s.startdate >= ('{pd}'::date - INTERVAL '1 month') THEN s.qty END), 0) AS curr,
            COALESCE(SUM(CASE WHEN s.startdate >= ('{pd}'::date - INTERVAL '2 months')
                              AND s.startdate < ('{pd}'::date - INTERVAL '1 month') THEN s.qty END), 0) AS prev
        FROM fact_sales_monthly s
        JOIN dim_item i ON s.item_id = i.item_id
        {_tm_where}
        GROUP BY s.item_id, i.item_desc
        HAVING COALESCE(SUM(CASE WHEN s.startdate >= ('{pd}'::date - INTERVAL '2 months')
                                  AND s.startdate < ('{pd}'::date - INTERVAL '1 month') THEN s.qty END), 0) > 0
        ORDER BY ABS(
            COALESCE(SUM(CASE WHEN s.startdate >= ('{pd}'::date - INTERVAL '1 month') THEN s.qty END), 0)
            - COALESCE(SUM(CASE WHEN s.startdate >= ('{pd}'::date - INTERVAL '2 months')
                                AND s.startdate < ('{pd}'::date - INTERVAL '1 month') THEN s.qty END), 0)
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
    col_grain: str = Query(default="date"),
    periods: int = Query(default=4, ge=1, le=12),
    brand: str = Query(default=""),
    category: str = Query(default=""),
    market: str = Query(default=""),
    channel: str = Query(default=""),
    item: str = Query(default=""),
    location: str = Query(default=""),
    cluster_assignment: str = Query(default=""),
    model: str = Query(default="external"),
):
    """Performance matrix — both row and column axes are selectable.

    ``grain`` controls rows, ``col_grain`` controls columns.
    Either axis (but not both) may be ``date``.
    ``model`` selects which forecast model to compute accuracy for (default: external).
    """
    set_cache(response, max_age=300)
    pd = get_planning_date().isoformat()

    _GRAIN_MAP = {
        "category": "i.category",
        "brand": "i.brand_name",
        "location": "f.loc",
        "class": "i.\"class\"",
        "sub_class": "i.sub_class",
        "date": "TO_CHAR(f.startdate, 'Mon YY')",
    }
    _ORDER_MAP = {
        "category": "i.category",
        "brand": "i.brand_name",
        "location": "f.loc",
        "class": "i.\"class\"",
        "sub_class": "i.sub_class",
        "date": "f.startdate",
    }

    row_grain = grain
    if row_grain == col_grain:
        col_grain = "date" if row_grain != "date" else "category"

    row_col = _GRAIN_MAP.get(row_grain, "i.category")
    col_col = _GRAIN_MAP.get(col_grain, "TO_CHAR(f.startdate, 'Mon YY')")
    row_order = _ORDER_MAP.get(row_grain, "i.category")
    col_order = _ORDER_MAP.get(col_grain, "f.startdate")

    filter_frag, filter_params = _dashboard_filter_clause(
        brand=brand, category=category, market=market, channel=channel,
        item=item, location=location, cluster_assignment=cluster_assignment,
    )
    filter_where = f" AND {filter_frag}" if filter_frag else ""
    join_clause = "JOIN dim_item i ON f.item_id = i.item_id"
    if market.strip() or row_grain == "location" or col_grain == "location":
        join_clause += " LEFT JOIN dim_location lo ON f.loc = lo.location_id"

    # Use UNION of main forecast table + backtest archive so that models
    # stored only in the archive (not yet loaded into main table) also work.
    # Filter basefcst_pref != 0 to exclude rows where no forecast was issued.
    sql = f"""
        WITH src AS (
            SELECT item_id, customer_group, loc, model_id, lag, startdate,
                   basefcst_pref, tothist_dmd
            FROM fact_external_forecast_monthly
            WHERE model_id = %s
              AND tothist_dmd IS NOT NULL AND tothist_dmd != 0
              AND basefcst_pref IS NOT NULL AND basefcst_pref != 0
              AND startdate >= ('{pd}'::date - (%s || ' months')::interval)
            UNION ALL
            SELECT item_id, customer_group, loc, model_id, lag, startdate,
                   basefcst_pref, tothist_dmd
            FROM backtest_lag_archive
            WHERE model_id = %s
              AND tothist_dmd IS NOT NULL AND tothist_dmd != 0
              AND basefcst_pref IS NOT NULL AND basefcst_pref != 0
              AND startdate >= ('{pd}'::date - (%s || ' months')::interval)
              AND NOT EXISTS (
                  SELECT 1 FROM fact_external_forecast_monthly m
                  WHERE m.model_id = %s
                    AND m.item_id = backtest_lag_archive.item_id
                    AND m.loc = backtest_lag_archive.loc
                    AND m.startdate = backtest_lag_archive.startdate
                    AND m.lag = backtest_lag_archive.lag
              )
        )
        SELECT
            {row_col} AS row_label,
            {col_col} AS col_label,
            CASE WHEN ABS(SUM(f.tothist_dmd)) > 0
                 THEN 100.0 - 100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / ABS(SUM(f.tothist_dmd))
                 ELSE NULL END AS accuracy_pct,
            COUNT(DISTINCT (f.item_id, f.loc)) AS dfu_count
        FROM src f
        {join_clause}
        WHERE 1=1
          {filter_where}
        GROUP BY {row_col}, {col_col}, {row_order}, {col_order}
        ORDER BY {row_order}, {col_order}
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [model, periods, model, periods, model] + filter_params)
        rows = cur.fetchall()

    label_map: dict[str, dict[str, tuple[float, int]]] = OrderedDict()
    col_labels: list[str] = []

    for r in rows:
        rl = str(r[0]) if r[0] else "Unknown"
        cl = str(r[1]) if r[1] else "Unknown"
        acc = round(float(r[2]), 1) if r[2] is not None else 0.0
        count = int(r[3]) if r[3] is not None else 0
        if rl not in label_map:
            label_map[rl] = {}
        label_map[rl][cl] = (acc, count)
        if cl not in col_labels:
            col_labels.append(cl)

    # Prune columns with zero DFUs across all rows
    col_labels = [
        cl for cl in col_labels
        if any(label_map[rl].get(cl, (0, 0))[1] > 0 for rl in label_map)
    ]
    # Prune rows with zero DFUs across all remaining columns
    pruned_rows = [
        rl for rl in label_map
        if any(label_map[rl].get(cl, (0, 0))[1] > 0 for cl in col_labels)
    ]

    result_rows = []
    for rl in pruned_rows:
        col_vals = label_map[rl]
        values = []
        counts = []
        for cl in col_labels:
            cell = col_vals.get(cl)
            values.append(cell[0] if cell else 0.0)
            counts.append(cell[1] if cell else 0)
        result_rows.append({"label": rl, "values": values, "counts": counts})

    return {"rows": result_rows, "period_labels": col_labels, "metric": "accuracy_pct"}


@router.get("/dashboard/trend")
def dashboard_trend(
    response: FastAPIResponse,
    window: int = Query(default=12, ge=1, le=36),
    brand: str = Query(default=""),
    category: str = Query(default=""),
    market: str = Query(default=""),
    channel: str = Query(default=""),
    item: str = Query(default=""),
    location: str = Query(default=""),
    cluster_assignment: str = Query(default=""),
    model: str = Query(default="external"),
):
    """Monthly aggregate forecast vs actual totals for trend chart."""
    set_cache(response, max_age=120)
    pd = get_planning_date().isoformat()

    filter_frag, filter_params = _dashboard_filter_clause(brand, category, market, channel, item, location, cluster_assignment)

    join_clause = ""
    if brand.strip() or category.strip():
        join_clause += " JOIN dim_item i ON f.item_id = i.item_id"
    if market.strip():
        join_clause += " JOIN dim_location lo ON f.loc = lo.location_id"

    where_extra = f" AND {filter_frag}" if filter_frag else ""

    sql = f"""
        WITH src AS (
            SELECT item_id, loc, model_id, startdate, basefcst_pref, tothist_dmd
            FROM fact_external_forecast_monthly
            WHERE model_id = %s
              AND tothist_dmd IS NOT NULL
              AND startdate >= ('{pd}'::date - (%s || ' months')::interval)
            UNION ALL
            SELECT item_id, loc, model_id, startdate, basefcst_pref, tothist_dmd
            FROM backtest_lag_archive
            WHERE model_id = %s
              AND tothist_dmd IS NOT NULL
              AND startdate >= ('{pd}'::date - (%s || ' months')::interval)
              AND NOT EXISTS (
                  SELECT 1 FROM fact_external_forecast_monthly m
                  WHERE m.model_id = %s
                    AND m.item_id = backtest_lag_archive.item_id
                    AND m.loc = backtest_lag_archive.loc
                    AND m.startdate = backtest_lag_archive.startdate
                    AND m.lag = backtest_lag_archive.lag
              )
        )
        SELECT
            TO_CHAR(f.startdate, 'YYYY-MM') AS month,
            SUM(f.basefcst_pref) AS forecast,
            SUM(f.tothist_dmd) AS actual
        FROM src f
        {join_clause}
        WHERE 1=1
          {where_extra}
        GROUP BY f.startdate, TO_CHAR(f.startdate, 'YYYY-MM')
        ORDER BY f.startdate
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [model, window, model, window, model] + filter_params)
        rows = cur.fetchall()

    months = []
    for r in rows:
        months.append({
            "month": r[0],
            "forecast": round(float(r[1]), 0) if r[1] is not None else 0,
            "actual": round(float(r[2]), 0) if r[2] is not None else 0,
        })

    return {"months": months}


# ---------------------------------------------------------------------------
# GET /dashboard/customer-map  —  aggregate customer locations by state
# ---------------------------------------------------------------------------

# US state centroids (approximate) for map plotting
_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783), "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337), "ID": (44.240459, -114.478773),
    "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526), "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067), "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106), "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192), "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082), "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896), "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419), "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915), "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780), "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828), "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686), "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494), "WV": (38.491226, -80.954456),
    "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490),
    "DC": (38.897438, -77.026817),
}


@router.get("/dashboard/customer-map")
def dashboard_customer_map(
    response: FastAPIResponse,
    group_by: str = Query(default="state", pattern="^(state|zip|city)$"),
):
    """Aggregate customer locations for map display.

    Returns a list of locations with customer counts and coordinates.
    ``group_by`` controls the grouping granularity (state/zip/city).
    - state: coordinates from state centroids.
    - zip: coordinates from pgeocode (offline USPS centroid data).
    - city: coordinates averaged from zip centroids within each city+state.
    """
    set_cache(response, max_age=600)

    _MARKER_LIMIT = 1000  # cap rows to avoid huge payloads / slow geocoding

    if group_by == "state":
        sql = """
            SELECT state, COUNT(*) AS customer_count
            FROM dim_customer
            WHERE state IS NOT NULL AND TRIM(state) != ''
            GROUP BY state
            ORDER BY COUNT(*) DESC
        """
    elif group_by == "zip":
        sql = f"""
            SELECT zip, state, COUNT(*) AS customer_count
            FROM dim_customer
            WHERE zip IS NOT NULL AND TRIM(zip) != ''
            GROUP BY zip, state
            ORDER BY COUNT(*) DESC
            LIMIT {_MARKER_LIMIT}
        """
    else:
        # city: get count + a representative zip for geocoding
        sql = f"""
            SELECT city, state, COUNT(*) AS customer_count,
                   MODE() WITHIN GROUP (ORDER BY zip) AS common_zip
            FROM dim_customer
            WHERE city IS NOT NULL AND TRIM(city) != ''
            GROUP BY city, state
            ORDER BY COUNT(*) DESC
            LIMIT {_MARKER_LIMIT}
        """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    locations: list[dict[str, Any]] = []

    if group_by == "state":
        for r in rows:
            label = str(r[0]).strip()
            count = int(r[1])
            entry: dict[str, Any] = {"label": label, "customer_count": count, "state": label}
            coords = _STATE_CENTROIDS.get(label.upper())
            if coords:
                entry["lat"] = coords[0]
                entry["lon"] = coords[1]
            locations.append(entry)
    elif group_by == "zip":
        # Batch-resolve all zip codes via pgeocode
        nomi = _get_nomi()
        zip_codes = [str(r[0]).strip() for r in rows]
        # pgeocode accepts list of postal codes for batch lookup
        geo_df = nomi.query_postal_code(zip_codes)
        for i, r in enumerate(rows):
            label = str(r[0]).strip()
            state = str(r[1]).strip() if r[1] else ""
            count = int(r[2])
            entry = {"label": label, "customer_count": count}
            if state:
                entry["state"] = state
            lat = geo_df.iloc[i]["latitude"]
            lon = geo_df.iloc[i]["longitude"]
            if not (math.isnan(lat) or math.isnan(lon)):
                entry["lat"] = round(float(lat), 4)
                entry["lon"] = round(float(lon), 4)
            else:
                # Fallback to state centroid
                coords = _STATE_CENTROIDS.get(state.upper()) if state else None
                if coords:
                    entry["lat"] = coords[0]
                    entry["lon"] = coords[1]
            locations.append(entry)
    else:
        # city: resolve via the representative zip per city+state
        nomi = _get_nomi()
        rep_zips = [str(r[3]).strip() if r[3] else "" for r in rows]
        geo_df = nomi.query_postal_code(rep_zips)
        for i, r in enumerate(rows):
            label = str(r[0]).strip()
            state = str(r[1]).strip() if r[1] else ""
            count = int(r[2])
            entry = {"label": label, "customer_count": count}
            if state:
                entry["state"] = state
            lat = geo_df.iloc[i]["latitude"]
            lon = geo_df.iloc[i]["longitude"]
            if not (math.isnan(lat) or math.isnan(lon)):
                entry["lat"] = round(float(lat), 4)
                entry["lon"] = round(float(lon), 4)
            else:
                coords = _STATE_CENTROIDS.get(state.upper()) if state else None
                if coords:
                    entry["lat"] = coords[0]
                    entry["lon"] = coords[1]
            locations.append(entry)

    total = sum(loc["customer_count"] for loc in locations)
    return {"locations": locations, "group_by": group_by, "total": total}
