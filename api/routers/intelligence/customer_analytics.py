"""Customer Analytics endpoints — demand-aware geographic, segment, and channel analytics.

Joins fact_customer_demand_monthly with dim_customer / dim_item to provide
rich demand visualizations: map, treemap, heatmap, channel mix, segment trends,
ranking, and OOS impact.
"""
from __future__ import annotations

import logging
import math
import threading
from datetime import date
from typing import Any

import pgeocode
from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache
from common.planning_date import get_planning_date
from common.services.cache import cached_sync

# All customer-analytics aggregates hit fact_customer_demand_monthly with
# the same join pattern. They are heavy (single-digit to ~16 second queries
# on a year of data) but the inputs are stable per-filter, so a 5-minute
# server-side cache turns repeat hits — which dominate dashboard usage —
# into millisecond responses. Invalidate via the "customer_analytics" group
# after a customer demand reload.
_CA_CACHE = cached_sync(ttl=300, group="customer_analytics")

logger = logging.getLogger(__name__)

router = APIRouter(tags=["customer-analytics"])

# ---------------------------------------------------------------------------
# Geocoding helpers (shared with dashboard.py pattern)
# ---------------------------------------------------------------------------
_nomi: pgeocode.Nominatim | None = None
_nomi_lock = threading.Lock()


def _get_nomi() -> pgeocode.Nominatim:
    global _nomi
    if _nomi is None:
        with _nomi_lock:
            if _nomi is None:
                _nomi = pgeocode.Nominatim("US")
    return _nomi


# State centroids for fallback geocoding
_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (32.81, -86.79), "AK": (63.59, -154.49), "AZ": (34.05, -111.09),
    "AR": (34.80, -92.20), "CA": (36.78, -119.42), "CO": (39.06, -105.31),
    "CT": (41.60, -72.76), "DE": (38.99, -75.51), "FL": (27.99, -81.76),
    "GA": (33.25, -83.44), "HI": (19.74, -155.84), "ID": (44.07, -114.74),
    "IL": (40.00, -89.00), "IN": (39.85, -86.27), "IA": (42.01, -93.21),
    "KS": (38.50, -98.00), "KY": (37.67, -84.67), "LA": (31.17, -91.87),
    "ME": (44.69, -69.38), "MD": (39.06, -76.80), "MA": (42.41, -71.38),
    "MI": (44.18, -84.51), "MN": (46.39, -94.64), "MS": (32.75, -89.66),
    "MO": (38.46, -92.29), "MT": (46.68, -110.04), "NE": (41.13, -98.27),
    "NV": (38.31, -117.06), "NH": (43.45, -71.56), "NJ": (40.30, -74.52),
    "NM": (34.84, -106.25), "NY": (42.17, -74.95), "NC": (35.63, -79.81),
    "ND": (47.53, -99.78), "OH": (40.39, -82.76), "OK": (35.57, -96.93),
    "OR": (43.80, -120.55), "PA": (41.20, -77.19), "RI": (41.58, -71.48),
    "SC": (33.86, -80.95), "SD": (43.97, -99.90), "TN": (35.75, -86.69),
    "TX": (31.05, -97.56), "UT": (39.32, -111.09), "VT": (44.07, -72.67),
    "VA": (37.77, -78.17), "WA": (47.40, -121.49), "WV": (38.47, -80.95),
    "WI": (44.27, -89.62), "WY": (43.08, -107.29), "DC": (38.91, -77.01),
}

# Marker count cap for the choropleth/bubble map. State-level needs all 50,
# but city/zip can return thousands — and the frontend grid-clusters anything
# above ~100 anyway, so shipping more is wasted bandwidth + render cost.
_MARKER_LIMIT_STATE = 60
_MARKER_LIMIT_CITY_ZIP = 150
_MARKER_LIMIT = _MARKER_LIMIT_STATE  # back-compat alias


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_date_range() -> tuple[str, str]:
    """Return (date_from, date_to) for the last 12 months."""
    pd = get_planning_date()
    dt_to = pd.replace(day=1)
    dt_from = dt_to - relativedelta(months=12)
    return dt_from.isoformat(), dt_to.isoformat()


def _build_where(
    params: list[Any],
    item_id: str | None,
    date_from: str | None,
    date_to: str | None,
    channel: str | None,
    store_type: str | None,
    state: str | None = None,
) -> str:
    """Build WHERE clause fragments and append params. Returns SQL fragment.

    The startdate range is mandatory: without it Postgres can't prune the
    monthly partitions of fact_customer_demand_monthly and degrades to a full
    scan across millions of rows. We always emit `f.startdate >= %s AND
    f.startdate < %s` and fall back to the last 12 months if the caller
    passes empty values. The empty-string check guards against callers that
    pass query-string params through unsanitized.
    """
    df, dt = _default_date_range()
    actual_from = (date_from or "").strip() or df
    actual_to = (date_to or "").strip() or dt
    if not actual_from or not actual_to:  # belt-and-braces — _default_date_range never returns empty
        raise ValueError("startdate range is mandatory for fact_customer_demand_monthly queries")
    clauses = ["f.startdate >= %s", "f.startdate < %s"]
    params.extend([actual_from, actual_to])
    if item_id:
        clauses.append("f.item_id = %s")
        params.append(item_id)
    if channel:
        clauses.append("c.rpt_channel_desc = %s")
        params.append(channel)
    if store_type:
        clauses.append("c.store_type_desc = %s")
        params.append(store_type)
    if state:
        clauses.append("UPPER(c.state) = %s")
        params.append(state.strip().upper())
    return " AND ".join(clauses)


def _build_where_mv(
    params: list[Any],
    date_from: str | None,
    date_to: str | None,
    channel: str | None,
    store_type: str | None,
) -> str:
    """Variant of _build_where for queries that hit mv_customer_activity_monthly.

    The MV has the dim_customer columns inlined under the same `f` alias used
    by the original queries, so callers can swap source tables with minimal
    changes. No item_id filter — the MV is item-aggregated by design (use the
    raw fact table when an item filter is required).
    """
    df, dt = _default_date_range()
    actual_from = (date_from or "").strip() or df
    actual_to = (date_to or "").strip() or dt
    if not actual_from or not actual_to:
        raise ValueError("startdate range is mandatory for mv_customer_activity_monthly queries")
    clauses = ["f.startdate >= %s", "f.startdate < %s"]
    params.extend([actual_from, actual_to])
    if channel:
        clauses.append("f.rpt_channel_desc = %s")
        params.append(channel)
    if store_type:
        clauses.append("f.store_type_desc = %s")
        params.append(store_type)
    return " AND ".join(clauses)


def _customer_activity_source(item_id: str | None) -> tuple[str, bool]:
    """Pick fact vs MV. Returns (FROM clause fragment, uses_mv).

    The MV pre-joins fact_customer_demand_monthly with dim_customer and
    aggregates to (customer_no, site, startdate) granularity — ~10x smaller
    than the raw fact join. We can only use it when item_id is NOT filtered.
    """
    if item_id:
        return (
            "fact_customer_demand_monthly f "
            "JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site"
        ), False
    return "mv_customer_activity_monthly f", True


def _geocode_zip(zip_code: str) -> tuple[float | None, float | None]:
    """Resolve a single zip code to (lat, lon) via pgeocode."""
    nomi = _get_nomi()
    result = nomi.query_postal_code(zip_code)
    lat = result.get("latitude") if hasattr(result, "get") else getattr(result, "latitude", None)
    lon = result.get("longitude") if hasattr(result, "get") else getattr(result, "longitude", None)
    if lat is not None and lon is not None and not (math.isnan(lat) or math.isnan(lon)):
        return round(float(lat), 4), round(float(lon), 4)
    return None, None


def _add_state_coords(entries: list[dict[str, Any]]) -> None:
    """Add lat/lon from state centroids to entries that have a 'state' key."""
    for e in entries:
        state = (e.get("state") or "").strip().upper()
        coords = _STATE_CENTROIDS.get(state)
        if coords:
            e["lat"] = coords[0]
            e["lon"] = coords[1]


# ---------------------------------------------------------------------------
# 1. Enhanced Demand Map
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/map")
@_CA_CACHE
def customer_analytics_map(
    response: FastAPIResponse,
    metric: str = Query(default="demand_qty", pattern="^(customer_count|demand_qty|sales_qty|oos_qty|fill_rate)$"),
    group_by: str = Query(default="state", pattern="^(state|city|zip)$"),
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
    state: str | None = Query(default=None),
):
    """Demand-aware customer map with metric selection."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, channel, store_type, state=state)

    geo_col = {"state": "c.state", "city": "c.city", "zip": "c.zip"}[group_by]

    sql = f"""
        SELECT {geo_col} AS geo_label,
               {'c.state AS state_col,' if group_by != 'state' else ''}
               COUNT(DISTINCT c.customer_no) AS customer_count,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty,
               COALESCE(SUM(f.oos_qty), 0) AS oos_qty
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
          AND {geo_col} IS NOT NULL AND TRIM({geo_col}) != ''
        GROUP BY {geo_col} {',' + 'c.state' if group_by != 'state' else ''}
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT %s
    """
    params.append(_MARKER_LIMIT_STATE if group_by == "state" else _MARKER_LIMIT_CITY_ZIP)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    locations: list[dict[str, Any]] = []
    for r in rows:
        if group_by == "state":
            geo_label, cust_count, demand, sales, oos = r
            state_val = str(geo_label).strip()
        else:
            geo_label, state_val, cust_count, demand, sales, oos = r
            state_val = str(state_val).strip() if state_val else ""

        d = float(demand or 0)
        s = float(sales or 0)
        o = float(oos or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0

        entry: dict[str, Any] = {
            "label": str(geo_label).strip(),
            "state": state_val if group_by == "state" else state_val,
            "customer_count": int(cust_count),
            "demand_qty": round(d, 1),
            "sales_qty": round(s, 1),
            "oos_qty": round(o, 1),
            "fill_rate": fr,
        }
        locations.append(entry)

    # Add coordinates
    if group_by == "state":
        _add_state_coords(locations)
    else:
        nomi = _get_nomi()
        if group_by == "zip":
            zips = [loc["label"] for loc in locations]
        else:
            # city — use state centroid as fallback
            zips = []
        if zips:
            geo_df = nomi.query_postal_code(zips)
            for i, loc in enumerate(locations):
                lat_val = geo_df.iloc[i]["latitude"]
                lon_val = geo_df.iloc[i]["longitude"]
                if not (math.isnan(lat_val) or math.isnan(lon_val)):
                    loc["lat"] = round(float(lat_val), 4)
                    loc["lon"] = round(float(lon_val), 4)
                else:
                    coords = _STATE_CENTROIDS.get((loc.get("state") or "").upper())
                    if coords:
                        loc["lat"] = coords[0]
                        loc["lon"] = coords[1]
        else:
            _add_state_coords(locations)

    total_demand = sum(loc["demand_qty"] for loc in locations)
    total_customers = sum(loc["customer_count"] for loc in locations)
    return {
        "locations": locations,
        "group_by": group_by,
        "metric": metric,
        "total_demand": round(total_demand, 1),
        "total_customers": total_customers,
    }


# ---------------------------------------------------------------------------
# 2. Customer Concentration Treemap
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/treemap")
@_CA_CACHE
def customer_analytics_treemap(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
):
    """Hierarchical treemap: State > Channel > Customer."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, channel, store_type)

    sql = f"""
        SELECT c.state,
               COALESCE(c.rpt_channel_desc, 'Unknown') AS channel,
               c.customer_name,
               c.customer_no,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
          AND c.state IS NOT NULL AND TRIM(c.state) != ''
        GROUP BY c.state, c.rpt_channel_desc, c.customer_name, c.customer_no
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT 500
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Build hierarchy: state -> channel -> customer
    states: dict[str, dict[str, Any]] = {}
    for state, ch, cust_name, cust_no, demand, sales in rows:
        state = str(state).strip()
        ch = str(ch).strip()
        d = float(demand or 0)
        s = float(sales or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0

        if state not in states:
            states[state] = {"name": state, "value": 0, "children": {}}
        states[state]["value"] += d

        channels = states[state]["children"]
        if ch not in channels:
            channels[ch] = {"name": ch, "value": 0, "children": []}
        channels[ch]["value"] += d
        channels[ch]["children"].append({
            "name": cust_name or cust_no,
            "value": round(d, 1),
            "fill_rate": fr,
        })

    # Convert to list format
    tree: list[dict[str, Any]] = []
    for s_data in sorted(states.values(), key=lambda x: x["value"], reverse=True):
        children = []
        for ch_data in sorted(s_data["children"].values(), key=lambda x: x["value"], reverse=True):
            ch_data["value"] = round(ch_data["value"], 1)
            # Keep top 10 customers per channel
            ch_data["children"] = sorted(ch_data["children"], key=lambda x: x["value"], reverse=True)[:10]
            children.append(ch_data)
        tree.append({
            "name": s_data["name"],
            "value": round(s_data["value"], 1),
            "children": children,
        })

    return {"tree": tree[:30]}  # Top 30 states


# ---------------------------------------------------------------------------
# 3. Item x State Heatmap
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/heatmap")
@_CA_CACHE
def customer_analytics_heatmap(
    response: FastAPIResponse,
    metric: str = Query(default="demand_qty", pattern="^(demand_qty|customer_count|fill_rate)$"),
    top_n: int = Query(default=25, ge=5, le=100),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
):
    """Item x State heatmap matrix."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, None, date_from, date_to, channel, store_type)

    # Push the (top_n items x 30 states) reduction into Postgres so we don't
    # ship every (item, state) aggregate row to Python just to discard most.
    sql = f"""
        WITH agg AS (
            SELECT f.item_id,
                   COALESCE(i.item_desc, f.item_id) AS item_desc,
                   c.state,
                   COUNT(DISTINCT c.customer_no) AS customer_count,
                   COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
                   COALESCE(SUM(f.sales_qty), 0) AS sales_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            LEFT JOIN dim_item i ON i.item_id = f.item_id
            WHERE {where}
              AND c.state IS NOT NULL AND TRIM(c.state) != ''
            GROUP BY f.item_id, i.item_desc, c.state
        ),
        top_items AS (
            SELECT item_id
            FROM agg
            GROUP BY item_id
            ORDER BY SUM(demand_qty) DESC
            LIMIT %s
        ),
        top_states AS (
            SELECT state
            FROM agg
            GROUP BY state
            ORDER BY SUM(demand_qty) DESC
            LIMIT 30
        )
        SELECT a.item_id, a.item_desc, a.state, a.customer_count, a.demand_qty, a.sales_qty
        FROM agg a
        JOIN top_items ti ON ti.item_id = a.item_id
        JOIN top_states ts ON ts.state = a.state
        ORDER BY a.demand_qty DESC
    """
    params.append(top_n)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # SQL already filtered to top_n items × top 30 states. Just compute axis
    # ordering from the returned rows (no second top-N pass).
    item_totals: dict[str, float] = {}
    item_descs: dict[str, str] = {}
    state_totals: dict[str, float] = {}
    cells: list[dict[str, Any]] = []
    for item_id, item_desc, state, cc, demand, sales in rows:
        state = str(state).strip()
        d = float(demand or 0)
        s = float(sales or 0)
        item_totals[item_id] = item_totals.get(item_id, 0) + d
        item_descs[item_id] = item_desc or item_id
        state_totals[state] = state_totals.get(state, 0) + d
        cells.append({
            "item_id": item_id,
            "state": state,
            "demand_qty": round(d, 1),
            "customer_count": int(cc),
            "fill_rate": round(s / d * 100, 1) if d > 0 else 100.0,
        })

    items_sorted = sorted(item_totals, key=item_totals.get, reverse=True)  # type: ignore[arg-type]
    states_sorted = sorted(state_totals, key=state_totals.get, reverse=True)  # type: ignore[arg-type]

    return {
        "items": [{"item_id": it, "item_desc": item_descs[it]} for it in items_sorted],
        "states": states_sorted,
        "cells": cells,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# 4. Channel Mix Sunburst
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/channel-mix")
@_CA_CACHE
def customer_analytics_channel_mix(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    state: str | None = Query(default=None),
):
    """Channel/store-type/sub-channel sunburst hierarchy."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, None, None)
    if state:
        where += " AND c.state = %s"
        params.append(state)

    # Cap groups: the rendered sunburst only keeps the top channels with
    # 12 store-types x 15 sub-channels each. A 1000-row LIMIT is far above
    # that and prevents a runaway when a tenant has thousands of sub-channels.
    sql = f"""
        SELECT COALESCE(c.rpt_channel_desc, 'Unknown') AS channel,
               COALESCE(c.store_type_desc, 'Unknown') AS store_type,
               COALESCE(c.rpt_sub_channel_desc, 'Unknown') AS sub_channel,
               COUNT(DISTINCT c.customer_no) AS customer_count,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
        GROUP BY c.rpt_channel_desc, c.store_type_desc, c.rpt_sub_channel_desc
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT 1000
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Build hierarchy: channel -> store_type -> sub_channel
    # Clean null/unknown labels and compute grand total for threshold
    grand_total = sum(float(r[4] or 0) for r in rows)
    min_threshold = grand_total * 0.01  # 1% minimum to show individually

    channels: dict[str, dict[str, Any]] = {}
    for ch, st, sub, cc, demand, sales in rows:
        ch = str(ch).strip() if ch else ""
        st = str(st).strip() if st else ""
        sub = str(sub).strip() if sub else ""
        # Clean up null-ish labels
        if not ch or ch.lower() in ("unknown", "null", "none"):
            ch = "Unclassified"
        if not st or st.lower() in ("unknown", "null", "none"):
            st = "Other"
        if not sub or sub.lower() in ("unknown", "null", "none"):
            sub = "Other"
        d = float(demand or 0)
        s = float(sales or 0)

        if ch not in channels:
            channels[ch] = {"name": ch, "value": 0, "customer_count": 0, "children": {}}
        channels[ch]["value"] += d
        channels[ch]["customer_count"] += int(cc)

        store_types = channels[ch]["children"]
        if st not in store_types:
            store_types[st] = {"name": st, "value": 0, "customer_count": 0, "children": {}}
        store_types[st]["value"] += d
        store_types[st]["customer_count"] += int(cc)

        sub_channels = store_types[st]["children"]
        if sub not in sub_channels:
            sub_channels[sub] = {"name": sub, "value": 0, "customer_count": 0}
        sub_channels[sub]["value"] += d
        sub_channels[sub]["customer_count"] += int(cc)

    # Build final tree, rolling up tiny segments into "Other"
    tree: list[dict[str, Any]] = []
    for ch_data in sorted(channels.values(), key=lambda x: x["value"], reverse=True):
        ch_children: list[dict[str, Any]] = []
        other_st = {"name": "Other", "value": 0.0, "customer_count": 0, "children": []}

        for st_data in sorted(ch_data["children"].values(), key=lambda x: x["value"], reverse=True):
            if st_data["value"] < min_threshold:
                other_st["value"] += st_data["value"]
                other_st["customer_count"] += st_data["customer_count"]
                continue

            # Roll up tiny sub-channels
            st_children: list[dict[str, Any]] = []
            other_sub = {"name": "Other", "value": 0.0, "customer_count": 0}
            sub_threshold = st_data["value"] * 0.03  # 3% of parent

            for sub_data in sorted(st_data["children"].values(), key=lambda x: x["value"], reverse=True):
                if sub_data["value"] < sub_threshold:
                    other_sub["value"] += sub_data["value"]
                    other_sub["customer_count"] += sub_data["customer_count"]
                else:
                    st_children.append({
                        "name": sub_data["name"],
                        "value": round(sub_data["value"], 1),
                        "customer_count": sub_data["customer_count"],
                    })

            if other_sub["value"] > 0:
                st_children.append({
                    "name": other_sub["name"],
                    "value": round(other_sub["value"], 1),
                    "customer_count": other_sub["customer_count"],
                })

            ch_children.append({
                "name": st_data["name"],
                "value": round(st_data["value"], 1),
                "customer_count": st_data["customer_count"],
                "children": st_children[:15],  # cap at 15 sub-channels per store type
            })

        if other_st["value"] > 0:
            ch_children.append({
                "name": other_st["name"],
                "value": round(other_st["value"], 1),
                "customer_count": other_st["customer_count"],
                "children": [],
            })

        tree.append({
            "name": ch_data["name"],
            "value": round(ch_data["value"], 1),
            "customer_count": ch_data["customer_count"],
            "children": ch_children[:12],  # cap at 12 store types per channel
        })

    return {"tree": tree, "grand_total": round(grand_total, 1)}


# ---------------------------------------------------------------------------
# 5. Segment Trend Sparklines
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/segment-trends")
@_CA_CACHE
def customer_analytics_segment_trends(
    response: FastAPIResponse,
    segment_by: str = Query(default="rpt_channel_desc", pattern="^(rpt_channel_desc|store_type_desc|chain_type_desc|state)$"),
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Monthly demand trends grouped by segment dimension."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, None, None)

    seg_col = f"c.{segment_by}" if segment_by != "state" else "c.state"

    # Pre-rank segments so we only return monthly rows for the top 30 we
    # actually render (was returning every segment x month combination).
    sql = f"""
        WITH agg AS (
            SELECT {seg_col} AS segment,
                   f.startdate,
                   COUNT(DISTINCT c.customer_no) AS customer_count,
                   COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
                   COALESCE(SUM(f.sales_qty), 0) AS sales_qty,
                   COALESCE(SUM(f.oos_qty), 0) AS oos_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where}
              AND {seg_col} IS NOT NULL AND TRIM({seg_col}) != ''
            GROUP BY {seg_col}, f.startdate
        ),
        top_segments AS (
            SELECT segment
            FROM agg
            GROUP BY segment
            ORDER BY SUM(demand_qty) DESC
            LIMIT 30
        )
        SELECT a.segment, a.startdate, a.customer_count, a.demand_qty, a.sales_qty, a.oos_qty
        FROM agg a
        JOIN top_segments ts ON ts.segment = a.segment
        ORDER BY a.segment, a.startdate
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Group by segment
    segments: dict[str, dict[str, Any]] = {}
    for seg, sd, cc, demand, sales, oos in rows:
        seg = str(seg).strip()
        if seg not in segments:
            segments[seg] = {"segment": seg, "total_demand": 0, "total_customers": 0, "trend": []}
        d = float(demand or 0)
        s = float(sales or 0)
        o = float(oos or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0
        segments[seg]["total_demand"] += d
        segments[seg]["total_customers"] = max(segments[seg]["total_customers"], int(cc))
        segments[seg]["trend"].append({
            "month": sd.isoformat() if hasattr(sd, "isoformat") else str(sd),
            "demand_qty": round(d, 1),
            "sales_qty": round(s, 1),
            "fill_rate": fr,
        })

    result = sorted(segments.values(), key=lambda x: x["total_demand"], reverse=True)
    for seg in result:
        seg["total_demand"] = round(seg["total_demand"], 1)
        d_total = seg["total_demand"]
        # Calculate fill rate for the segment
        s_total = sum(t["sales_qty"] for t in seg["trend"])
        seg["fill_rate"] = round(s_total / d_total * 100, 1) if d_total > 0 else 100.0
        # MoM change (last 2 months)
        trend = seg["trend"]
        if len(trend) >= 2:
            prev = trend[-2]["demand_qty"]
            curr = trend[-1]["demand_qty"]
            seg["mom_change"] = round((curr - prev) / prev * 100, 1) if prev > 0 else 0.0
        else:
            seg["mom_change"] = 0.0

    return {"segments": result[:30], "segment_by": segment_by}


# ---------------------------------------------------------------------------
# 6. Customer Ranking
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/ranking")
@_CA_CACHE
def customer_analytics_ranking(
    response: FastAPIResponse,
    sort: str = Query(default="demand_desc", pattern="^(demand_desc|fill_rate_asc)$"),
    top_n: int = Query(default=20, ge=5, le=50),
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
    min_demand: float = Query(default=0, ge=0),
):
    """Top/bottom customer ranking by demand or fill rate."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, channel, store_type)

    order = "SUM(f.demand_qty) DESC" if sort == "demand_desc" else "CASE WHEN SUM(f.demand_qty) > 0 THEN SUM(f.sales_qty)::float / SUM(f.demand_qty) ELSE 1 END ASC"

    sql = f"""
        SELECT c.customer_no,
               c.customer_name,
               c.state,
               COALESCE(c.rpt_channel_desc, 'Unknown') AS channel,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty,
               COALESCE(SUM(f.oos_qty), 0) AS oos_qty
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
        GROUP BY c.customer_no, c.customer_name, c.state, c.rpt_channel_desc
        HAVING SUM(f.demand_qty) >= %s
        ORDER BY {order}
        LIMIT %s
    """
    params.extend([min_demand, top_n])

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    customers: list[dict[str, Any]] = []
    for cno, cname, state, ch, demand, sales, oos in rows:
        d = float(demand or 0)
        s = float(sales or 0)
        o = float(oos or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0
        customers.append({
            "customer_no": cno,
            "customer_name": cname or cno,
            "state": state or "",
            "channel": ch,
            "demand_qty": round(d, 1),
            "sales_qty": round(s, 1),
            "oos_qty": round(o, 1),
            "fill_rate": fr,
        })

    return {"customers": customers, "sort": sort, "top_n": top_n}


# ---------------------------------------------------------------------------
# 7. OOS Impact Bubble Chart
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/oos-impact")
@_CA_CACHE
def customer_analytics_oos_impact(
    response: FastAPIResponse,
    grain: str = Query(default="customer", pattern="^(customer|state)$"),
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
):
    """Bubble chart data: demand vs fill rate, bubble size = OOS qty."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, channel, None)

    if grain == "customer":
        group = "c.customer_no, c.customer_name, c.state, c.rpt_channel_desc"
        select_extra = "c.customer_no, c.customer_name, c.state, COALESCE(c.rpt_channel_desc, 'Unknown') AS channel"
    else:
        group = "c.state"
        select_extra = "c.state, c.state AS label, c.state AS state_col, 'All' AS channel"

    sql = f"""
        SELECT {select_extra},
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty,
               COALESCE(SUM(f.oos_qty), 0) AS oos_qty
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
          AND c.state IS NOT NULL AND TRIM(c.state) != ''
        GROUP BY {group}
        HAVING SUM(f.demand_qty) > 0
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT 200
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    bubbles: list[dict[str, Any]] = []
    for r in rows:
        if grain == "customer":
            cno, cname, state, ch, demand, sales, oos = r
            label = cname or cno
        else:
            state, label, _sc, ch, demand, sales, oos = r
            cno = None
        d = float(demand or 0)
        s = float(sales or 0)
        o = float(oos or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0
        entry: dict[str, Any] = {
            "label": str(label).strip() if label else (state or ""),
            "state": str(state).strip() if state else "",
            "channel": str(ch).strip() if ch else "Unknown",
            "demand_qty": round(d, 1),
            "sales_qty": round(s, 1),
            "oos_qty": round(o, 1),
            "fill_rate": fr,
        }
        if cno:
            entry["customer_no"] = cno
        bubbles.append(entry)

    return {"bubbles": bubbles, "grain": grain}


# ---------------------------------------------------------------------------
# 8. Item Search (typeahead for filter picker)
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/items")
def customer_analytics_items(
    response: FastAPIResponse,
    search: str = Query(default="", min_length=0),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Typeahead item search for the customer analytics filter bar.

    Filters dim_item to only items that have at least one row in
    fact_customer_demand_monthly for the selected date range. Without this
    scoping, planners would pick items that look valid in dim_item but have
    zero demand history (we observed this with item 100012 — exists in
    dim_item, zero rows in fact, panels rendered as visually-blank).

    Date range defaults to the same trailing 12 months used by the rest of
    the CA endpoints so the picker matches the default filter window.
    """
    set_cache(response, max_age=600)
    search = search.strip()
    df, dt = _default_date_range()
    actual_from = (date_from or "").strip() or df
    actual_to = (date_to or "").strip() or dt

    base_sql = """
        SELECT DISTINCT i.item_id, i.item_desc
        FROM dim_item i
        WHERE EXISTS (
            SELECT 1 FROM fact_customer_demand_monthly f
            WHERE f.item_id = i.item_id
              AND f.startdate >= %s AND f.startdate < %s
        )
    """
    query_params: list[Any] = [actual_from, actual_to]

    if search:
        base_sql += " AND (i.item_id ILIKE %s OR i.item_desc ILIKE %s)"
        pattern = f"%{search}%"
        query_params.extend([pattern, pattern])

    base_sql += " ORDER BY i.item_id LIMIT 50"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(base_sql, query_params)
        rows = cur.fetchall()

    items = [{"item_id": r[0], "item_desc": r[1] or r[0]} for r in rows]
    return {"items": items}


# ---------------------------------------------------------------------------
# 9. KPI Summary
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/kpis")
@_CA_CACHE
def customer_analytics_kpis(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
    state: str | None = Query(default=None),
):
    """KPI values aggregated over the selected date range, with MoM deltas
    computed from the latest two months in the range."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, channel, store_type, state=state)

    # Values use full-range totals so they line up with the map / treemap /
    # other panels on this dashboard. Deltas are still month-over-month
    # (latest available month vs prior month) — that's the only meaningful
    # comparison for monthly-grain data and matches how planners read the
    # arrows ("did things move this month?").
    sql = f"""
        WITH base AS (
            SELECT f.startdate,
                   f.customer_no,
                   f.demand_qty,
                   f.sales_qty,
                   f.oos_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where}
        ),
        bounds AS (
            SELECT MAX(startdate) AS cur_month,
                   MAX(startdate) - INTERVAL '1 month' AS prev_month
            FROM base
        ),
        totals AS (
            SELECT COALESCE(SUM(b.demand_qty), 0) AS demand,
                   COALESCE(SUM(b.sales_qty), 0) AS sales,
                   COALESCE(SUM(b.oos_qty), 0) AS oos,
                   COUNT(DISTINCT b.customer_no) AS active_cust
            FROM base b
        ),
        cur AS (
            SELECT COALESCE(SUM(b.demand_qty), 0) AS demand,
                   COALESCE(SUM(b.sales_qty), 0) AS sales,
                   COALESCE(SUM(b.oos_qty), 0) AS oos,
                   COUNT(DISTINCT b.customer_no) AS active_cust
            FROM base b, bounds
            WHERE b.startdate = bounds.cur_month
        ),
        prev AS (
            SELECT COALESCE(SUM(b.demand_qty), 0) AS demand,
                   COALESCE(SUM(b.sales_qty), 0) AS sales,
                   COALESCE(SUM(b.oos_qty), 0) AS oos,
                   COUNT(DISTINCT b.customer_no) AS active_cust
            FROM base b, bounds
            WHERE b.startdate = bounds.prev_month
        ),
        top10 AS (
            SELECT COALESCE(SUM(sub.demand), 0) AS top10_demand
            FROM (
                SELECT b.customer_no, SUM(b.demand_qty) AS demand
                FROM base b
                GROUP BY b.customer_no
                ORDER BY SUM(b.demand_qty) DESC
                LIMIT 10
            ) sub
        )
        SELECT totals.demand, totals.sales, totals.oos, totals.active_cust,
               cur.demand, cur.sales, cur.oos, cur.active_cust,
               prev.demand, prev.sales, prev.oos, prev.active_cust,
               top10.top10_demand
        FROM totals, cur, prev, top10
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    if not row:
        return {"kpis": []}

    t_demand = float(row[0] or 0)
    t_sales = float(row[1] or 0)
    t_oos = float(row[2] or 0)
    t_cust = int(row[3] or 0)
    c_demand = float(row[4] or 0)
    c_sales = float(row[5] or 0)
    c_oos = float(row[6] or 0)
    c_cust = int(row[7] or 0)
    p_demand = float(row[8] or 0)
    p_sales = float(row[9] or 0)
    p_oos = float(row[10] or 0)
    p_cust = int(row[11] or 0)
    top10_d = float(row[12] or 0)

    def _delta(cur_val: float, prev_val: float) -> float:
        if prev_val == 0:
            return 0.0
        return round((cur_val - prev_val) / prev_val * 100, 1)

    t_fr = round(t_sales / t_demand * 100, 1) if t_demand > 0 else 100.0
    c_fr = round(c_sales / c_demand * 100, 1) if c_demand > 0 else 100.0
    p_fr = round(p_sales / p_demand * 100, 1) if p_demand > 0 else 100.0
    conc = round(top10_d / t_demand * 100, 1) if t_demand > 0 else 0.0
    odr = round(t_sales / t_demand, 3) if t_demand > 0 else 0.0

    kpis = [
        {"key": "total_demand", "value": round(t_demand, 1), "delta": _delta(c_demand, p_demand)},
        {"key": "fill_rate", "value": t_fr, "delta": round(c_fr - p_fr, 1)},
        {"key": "oos_volume", "value": round(t_oos, 1), "delta": _delta(c_oos, p_oos)},
        {"key": "active_customers", "value": t_cust, "delta": _delta(float(c_cust), float(p_cust))},
        {"key": "concentration_top10", "value": conc, "delta": 0.0},
        {"key": "order_demand_ratio", "value": odr, "delta": 0.0},
    ]
    return {"kpis": kpis}


# ---------------------------------------------------------------------------
# 10. Customer Lifecycle
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/lifecycle")
@_CA_CACHE
def customer_analytics_lifecycle(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Cohort retention + waterfall (new vs churned)."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    # Use the pre-aggregated MV when no item filter is requested (the common
    # case) — it avoids the fact x dim_customer JOIN + DISTINCT for every
    # request. With an item filter we have to hit the raw fact table since
    # the MV is item-aggregated.
    source_from, uses_mv = _customer_activity_source(item_id)
    if uses_mv:
        where = _build_where_mv(params, date_from, date_to, None, None)
    else:
        where = _build_where(params, item_id, date_from, date_to, None, None)

    # --- cohort retention ---
    # Cap the result set: the UI only renders ~24 cohorts x ~24 months_since
    # cells. Without a cap, datasets with many cohort months produce huge
    # payloads that dominate both DB time and JSON serialization.
    cohort_sql = f"""
        WITH base AS (
            SELECT DISTINCT f.customer_no, f.startdate
            FROM {source_from}
            WHERE {where}
        ),
        first_order AS (
            SELECT customer_no, MIN(startdate) AS cohort_month
            FROM base
            GROUP BY customer_no
        ),
        cohort_activity AS (
            SELECT fo.cohort_month,
                   EXTRACT(YEAR FROM age(b.startdate, fo.cohort_month)) * 12
                     + EXTRACT(MONTH FROM age(b.startdate, fo.cohort_month)) AS months_since,
                   COUNT(DISTINCT b.customer_no) AS active_customers
            FROM base b
            JOIN first_order fo ON fo.customer_no = b.customer_no
            GROUP BY fo.cohort_month, months_since
        ),
        cohort_size AS (
            SELECT cohort_month, COUNT(*) AS size
            FROM first_order
            GROUP BY cohort_month
        )
        SELECT ca.cohort_month, ca.months_since::int, ca.active_customers, cs.size
        FROM cohort_activity ca
        JOIN cohort_size cs ON cs.cohort_month = ca.cohort_month
        WHERE ca.months_since <= 24
        ORDER BY ca.cohort_month, ca.months_since
        LIMIT 1000
    """

    # --- waterfall (new / churned) ---
    params2: list[Any] = []
    if uses_mv:
        where2 = _build_where_mv(params2, date_from, date_to, None, None)
    else:
        where2 = _build_where(params2, item_id, date_from, date_to, None, None)
    # Window-function rewrite of the churn waterfall. The previous version
    # had two `months × base` range joins (`b.startdate >= m.month - 6mo`)
    # which materialize an N×M intermediate set. Here we instead annotate each
    # base row with `last_order_per_customer` via a window function, then
    # detect churn at the customer level (last activity 3-6 mo before MAX),
    # and finally aggregate per month. One pass over `base`, no Cartesian.
    waterfall_sql = f"""
        WITH base AS (
            SELECT DISTINCT f.customer_no, f.startdate
            FROM {source_from}
            WHERE {where2}
        ),
        months AS (
            SELECT DISTINCT startdate AS month FROM base
        ),
        first_order AS (
            SELECT customer_no, MIN(startdate) AS first_month, MAX(startdate) AS last_month
            FROM base GROUP BY customer_no
        ),
        new_per_month AS (
            SELECT first_month AS month, COUNT(*) AS new_customers
            FROM first_order GROUP BY first_month
        ),
        -- For each month m, churn = customers whose last activity falls in
        -- [m - 6mo, m - 3mo) — i.e. they were active in the older window
        -- but NOT in the recent window. Computed without re-joining base.
        churned AS (
            SELECT m.month,
                   COUNT(*) AS churned_customers
            FROM months m
            JOIN first_order fo
              ON fo.last_month >= m.month - INTERVAL '6 months'
             AND fo.last_month <  m.month - INTERVAL '3 months'
            GROUP BY m.month
        )
        SELECT m.month,
               COALESCE(n.new_customers, 0) AS new_customers,
               COALESCE(ch.churned_customers, 0) AS churned_customers
        FROM months m
        LEFT JOIN new_per_month n ON n.month = m.month
        LEFT JOIN churned ch ON ch.month = m.month
        ORDER BY m.month
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(cohort_sql, params)
        cohort_rows = cur.fetchall()
        cur.execute(waterfall_sql, params2)
        waterfall_rows = cur.fetchall()

    # Build cohort response
    cohort_map: dict[str, dict[str, Any]] = {}
    for cm, ms, active, size in cohort_rows:
        key = cm.isoformat() if hasattr(cm, "isoformat") else str(cm)
        if key not in cohort_map:
            cohort_map[key] = {"cohort_month": key, "months_since": [], "retention_pct": []}
        cohort_map[key]["months_since"].append(int(ms))
        pct = round(int(active) / int(size) * 100, 1) if int(size) > 0 else 0.0
        cohort_map[key]["retention_pct"].append(pct)

    cohorts = sorted(cohort_map.values(), key=lambda x: x["cohort_month"])

    waterfall = []
    for month, new_c, churned_c in waterfall_rows:
        m_str = month.isoformat() if hasattr(month, "isoformat") else str(month)
        n = int(new_c)
        ch = int(churned_c)
        waterfall.append({
            "month": m_str,
            "new_customers": n,
            "churned_customers": ch,
            "net_change": n - ch,
        })

    return {"cohorts": cohorts, "waterfall": waterfall}


# ---------------------------------------------------------------------------
# 11. Demand at Risk
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/demand-at-risk")
@_CA_CACHE
def customer_analytics_demand_at_risk(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Waterfall breakdown of demand risk categories."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, None, None)

    sql = f"""
        WITH base AS (
            SELECT f.customer_no,
                   f.item_id,
                   f.location_id,
                   f.startdate,
                   f.demand_qty,
                   f.sales_qty,
                   f.oos_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where}
        ),
        totals AS (
            SELECT COALESCE(SUM(demand_qty), 0) AS total_demand
            FROM base
        ),
        -- concentration risk: customers whose share > 40%
        cust_share AS (
            SELECT customer_no, SUM(demand_qty) AS cust_demand
            FROM base GROUP BY customer_no
        ),
        conc_risk AS (
            SELECT COALESCE(SUM(cs.cust_demand), 0) AS concentration_risk
            FROM cust_share cs, totals t
            WHERE t.total_demand > 0
              AND cs.cust_demand / t.total_demand > 0.4
        ),
        -- oos loss: demand * oos_rate at item-loc level
        oos_agg AS (
            SELECT item_id, location_id,
                   SUM(demand_qty) AS d, SUM(oos_qty) AS o
            FROM base GROUP BY item_id, location_id
            HAVING SUM(demand_qty) > 0
        ),
        oos_loss AS (
            SELECT COALESCE(SUM(d * (o / NULLIF(d, 0))), 0) AS oos_loss
            FROM oos_agg
            WHERE o > 0
        ),
        -- churn risk: customers in [-6,-3] but not [-3,0]
        bounds AS (
            SELECT MAX(startdate) AS max_dt FROM base
        ),
        recent AS (
            SELECT DISTINCT customer_no FROM base, bounds
            WHERE startdate > bounds.max_dt - INTERVAL '3 months'
        ),
        older AS (
            SELECT DISTINCT customer_no FROM base, bounds
            WHERE startdate <= bounds.max_dt - INTERVAL '3 months'
              AND startdate > bounds.max_dt - INTERVAL '6 months'
        ),
        churned_custs AS (
            SELECT o.customer_no
            FROM older o
            LEFT JOIN recent r ON r.customer_no = o.customer_no
            WHERE r.customer_no IS NULL
        ),
        churn_risk AS (
            SELECT COALESCE(SUM(b.demand_qty), 0) AS churn_risk
            FROM base b
            JOIN churned_custs cc ON cc.customer_no = b.customer_no
        )
        SELECT t.total_demand,
               cr.concentration_risk,
               ol.oos_loss,
               chr.churn_risk
        FROM totals t, conc_risk cr, oos_loss ol, churn_risk chr
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()

    if not row:
        return {"waterfall": []}

    total = float(row[0] or 0)
    conc = float(row[1] or 0)
    oos = float(row[2] or 0)
    churn = float(row[3] or 0)
    secure = max(total - conc - oos - churn, 0)

    waterfall = [
        {"category": "total_demand", "value": round(total, 1)},
        {"category": "concentration_risk", "value": round(conc, 1)},
        {"category": "oos_loss", "value": round(oos, 1)},
        {"category": "churn_risk", "value": round(churn, 1)},
        {"category": "secure_demand", "value": round(secure, 1)},
    ]
    return {"waterfall": waterfall}


# ---------------------------------------------------------------------------
# 12. Customer-Item Affinity
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/affinity")
@_CA_CACHE
def customer_analytics_affinity(
    response: FastAPIResponse,
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    top_n: int = Query(default=20, ge=5, le=50),
):
    """Heatmap of top N customers x top N items."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, None, date_from, date_to, None, None)

    sql = f"""
        WITH base AS (
            SELECT f.customer_no,
                   c.customer_name,
                   f.item_id,
                   COALESCE(i.item_desc, f.item_id) AS item_desc,
                   SUM(f.demand_qty) AS demand_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            LEFT JOIN dim_item i ON i.item_id = f.item_id
            WHERE {where}
            GROUP BY f.customer_no, c.customer_name, f.item_id, i.item_desc
        ),
        top_customers AS (
            SELECT customer_no, customer_name
            FROM base
            GROUP BY customer_no, customer_name
            ORDER BY SUM(demand_qty) DESC
            LIMIT %s
        ),
        top_items AS (
            SELECT item_id, item_desc
            FROM base
            GROUP BY item_id, item_desc
            ORDER BY SUM(demand_qty) DESC
            LIMIT %s
        )
        SELECT b.customer_no, tc.customer_name,
               b.item_id, ti.item_desc,
               b.demand_qty
        FROM base b
        JOIN top_customers tc ON tc.customer_no = b.customer_no
        JOIN top_items ti ON ti.item_id = b.item_id
        ORDER BY b.demand_qty DESC
    """
    params.extend([top_n, top_n])

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    cust_set: dict[str, str] = {}
    item_set: dict[str, str] = {}
    cells: list[dict[str, Any]] = []
    for cno, cname, iid, idesc, dq in rows:
        cust_set[cno] = cname or cno
        item_set[iid] = idesc or iid
        cells.append({
            "customer_no": cno,
            "item_id": iid,
            "demand_qty": round(float(dq or 0), 1),
        })

    return {
        "customers": [{"customer_no": k, "customer_name": v} for k, v in cust_set.items()],
        "items": [{"item_id": k, "item_desc": v} for k, v in item_set.items()],
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# 13. Order Patterns
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/order-patterns")
@_CA_CACHE
def customer_analytics_order_patterns(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Frequency histogram + regularity scatter for ordering cadence."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, None, None)

    # Two-stage shape: rank customers by total demand FIRST, keep only the
    # top ~1000, then run the expensive STDDEV/window work on that bounded
    # set. The previous version computed LAG, AVG, STDDEV across all ~33K
    # customers and only sliced the top 200 at the very end — wasted work
    # since the response only renders 200 rows anyway. Top 1000 (5x the
    # display cap) gives some headroom for ties at the boundary.
    sql = f"""
        WITH base AS (
            SELECT f.customer_no,
                   c.customer_name,
                   f.startdate,
                   SUM(f.demand_qty) AS demand_qty
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where}
            GROUP BY f.customer_no, c.customer_name, f.startdate
        ),
        cust_total AS (
            SELECT customer_no,
                   MAX(customer_name) AS customer_name,
                   SUM(demand_qty) AS total_demand
            FROM base
            GROUP BY customer_no
            ORDER BY SUM(demand_qty) DESC
            LIMIT 1000
        ),
        base_topn AS (
            SELECT b.customer_no, b.startdate
            FROM base b
            JOIN cust_total ct ON ct.customer_no = b.customer_no
        ),
        intervals AS (
            SELECT customer_no,
                   startdate,
                   EXTRACT(YEAR FROM age(startdate, LAG(startdate) OVER (PARTITION BY customer_no ORDER BY startdate))) * 12
                     + EXTRACT(MONTH FROM age(startdate, LAG(startdate) OVER (PARTITION BY customer_no ORDER BY startdate))) AS gap_months
            FROM base_topn
        ),
        cust_stats AS (
            SELECT customer_no,
                   AVG(gap_months) AS avg_interval,
                   CASE WHEN AVG(gap_months) > 0
                        THEN STDDEV(gap_months) / AVG(gap_months)
                        ELSE 0 END AS interval_cv,
                   COUNT(*) AS order_count
            FROM intervals
            WHERE gap_months IS NOT NULL
            GROUP BY customer_no
        )
        SELECT ct.customer_no, ct.customer_name,
               cs.avg_interval, cs.interval_cv, cs.order_count,
               ct.total_demand
        FROM cust_total ct
        LEFT JOIN cust_stats cs ON cs.customer_no = ct.customer_no
        ORDER BY ct.total_demand DESC
        LIMIT 200
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    buckets = {"monthly": 0, "bimonthly": 0, "quarterly": 0, "sporadic": 0}
    scatter: list[dict[str, Any]] = []

    for cno, cname, avg_int, cv, _oc, total_d in rows:
        avg_i = float(avg_int or 0)
        cv_val = float(cv or 0)
        td_val = float(total_d or 0)
        scatter.append({
            "customer_no": cno,
            "customer_name": cname or cno,
            "avg_interval_months": round(avg_i, 2),
            "interval_cv": round(cv_val, 2),
            "total_demand": round(td_val, 1),
        })
        if avg_i <= 1.5:
            buckets["monthly"] += 1
        elif avg_i <= 2.5:
            buckets["bimonthly"] += 1
        elif avg_i <= 4.0:
            buckets["quarterly"] += 1
        else:
            buckets["sporadic"] += 1

    total_custs = max(sum(buckets.values()), 1)
    histogram = [
        {"bucket": k, "count": v, "pct": round(v / total_custs * 100, 1)}
        for k, v in buckets.items()
    ]

    return {"frequency_histogram": histogram, "regularity_scatter": scatter}


# ---------------------------------------------------------------------------
# 14. Demand Flow Sankey
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/demand-flow")
@_CA_CACHE
def customer_analytics_demand_flow(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Sankey nodes + links: warehouse -> state -> channel."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, None, None)

    sql = f"""
        SELECT f.location_id,
               COALESCE(c.state, 'Unknown') AS state,
               COALESCE(c.rpt_channel_desc, 'Unknown') AS channel,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
        GROUP BY f.location_id, c.state, c.rpt_channel_desc
        HAVING SUM(f.demand_qty) > 0
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT 500
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    node_set: set[str] = set()
    links_wh_state: dict[tuple[str, str], float] = {}
    links_state_ch: dict[tuple[str, str], float] = {}

    for loc, state, ch, dq in rows:
        wh_name = f"WH_{loc}" if loc else "WH_Unknown"
        st_name = str(state).strip() or "Unknown"
        ch_name = str(ch).strip() or "Unknown"
        val = float(dq or 0)

        node_set.update([wh_name, st_name, ch_name])

        key_ws = (wh_name, st_name)
        links_wh_state[key_ws] = links_wh_state.get(key_ws, 0) + val

        key_sc = (st_name, ch_name)
        links_state_ch[key_sc] = links_state_ch.get(key_sc, 0) + val

    nodes = [{"name": n} for n in sorted(node_set)]
    links: list[dict[str, Any]] = []
    for (src, tgt), val in sorted(links_wh_state.items(), key=lambda x: -x[1]):
        links.append({"source": src, "target": tgt, "value": round(val, 1)})
    for (src, tgt), val in sorted(links_state_ch.items(), key=lambda x: -x[1]):
        links.append({"source": src, "target": tgt, "value": round(val, 1)})

    return {"nodes": nodes, "links": links}


# ---------------------------------------------------------------------------
# 15. Filter Options
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/filter-options")
def customer_analytics_filter_options(
    response: FastAPIResponse,
):
    """Distinct dropdown values for channel, store type, state."""
    # Filter enums change at most when dim_customer is reloaded (~daily).
    # Long browser cache (1h) + 6h stale-while-revalidate eliminates the
    # repeat fetch on every tab switch / tab reopen.
    set_cache(response, max_age=3600, stale_while_revalidate=21600)

    sql = """
        SELECT
            ARRAY_AGG(DISTINCT c.rpt_channel_desc ORDER BY c.rpt_channel_desc)
                FILTER (WHERE c.rpt_channel_desc IS NOT NULL AND TRIM(c.rpt_channel_desc) != ''),
            ARRAY_AGG(DISTINCT c.store_type_desc ORDER BY c.store_type_desc)
                FILTER (WHERE c.store_type_desc IS NOT NULL AND TRIM(c.store_type_desc) != ''),
            ARRAY_AGG(DISTINCT c.state ORDER BY c.state)
                FILTER (WHERE c.state IS NOT NULL AND TRIM(c.state) != '')
        FROM dim_customer c
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()

    channels = row[0] if row and row[0] else []
    store_types = row[1] if row and row[1] else []
    states = row[2] if row and row[2] else []

    return {"channels": channels, "store_types": store_types, "states": states}


# ---------------------------------------------------------------------------
# 16. Alerts
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/alerts")
@_CA_CACHE
def customer_analytics_alerts(
    response: FastAPIResponse,
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Evaluate threshold rules and return active alerts."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, None, date_from, date_to, None, None)

    # 1) fill rate < 85% per item-loc
    fr_sql = f"""
        SELECT f.item_id, f.location_id,
               CASE WHEN SUM(f.demand_qty) > 0
                    THEN SUM(f.sales_qty)::float / SUM(f.demand_qty) * 100
                    ELSE 100 END AS fill_rate
        FROM fact_customer_demand_monthly f
        JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
        WHERE {where}
        GROUP BY f.item_id, f.location_id
        HAVING SUM(f.demand_qty) > 0
           AND SUM(f.sales_qty)::float / SUM(f.demand_qty) * 100 < 85
        ORDER BY SUM(f.sales_qty)::float / SUM(f.demand_qty) ASC
        LIMIT 50
    """

    # 2) HHI > 0.6 per item-loc
    params2: list[Any] = []
    where2 = _build_where(params2, None, date_from, date_to, None, None)
    hhi_sql = f"""
        WITH item_loc AS (
            SELECT f.item_id, f.location_id, f.customer_no,
                   SUM(f.demand_qty) AS cust_demand
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where2}
            GROUP BY f.item_id, f.location_id, f.customer_no
        ),
        il_total AS (
            SELECT item_id, location_id, SUM(cust_demand) AS total_demand
            FROM item_loc GROUP BY item_id, location_id
            HAVING SUM(cust_demand) > 0
        ),
        hhi AS (
            SELECT il.item_id, il.location_id,
                   SUM(POWER(il.cust_demand / t.total_demand, 2)) AS hhi
            FROM item_loc il
            JOIN il_total t ON t.item_id = il.item_id AND t.location_id = il.location_id
            GROUP BY il.item_id, il.location_id
            HAVING SUM(POWER(il.cust_demand / t.total_demand, 2)) > 0.6
        )
        SELECT item_id, location_id, ROUND(hhi::numeric, 3)
        FROM hhi
        ORDER BY hhi DESC
        LIMIT 50
    """

    # 3) churn rate > 10% MoM + 4) demand surge > 30% MoM
    params3: list[Any] = []
    where3 = _build_where(params3, None, date_from, date_to, None, None)
    mom_sql = f"""
        WITH base AS (
            SELECT f.startdate,
                   COUNT(DISTINCT f.customer_no) AS active_cust,
                   SUM(f.demand_qty) AS demand
            FROM fact_customer_demand_monthly f
            JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
            WHERE {where3}
            GROUP BY f.startdate
            ORDER BY f.startdate
        ),
        lagged AS (
            SELECT startdate, active_cust, demand,
                   LAG(active_cust) OVER (ORDER BY startdate) AS prev_cust,
                   LAG(demand) OVER (ORDER BY startdate) AS prev_demand
            FROM base
        )
        SELECT startdate, active_cust, prev_cust, demand, prev_demand
        FROM lagged
        WHERE prev_cust IS NOT NULL
        ORDER BY startdate DESC
        LIMIT 1
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(fr_sql, params)
        fr_rows = cur.fetchall()
        cur.execute(hhi_sql, params2)
        hhi_rows = cur.fetchall()
        cur.execute(mom_sql, params3)
        mom_row = cur.fetchone()

    alerts: list[dict[str, Any]] = []

    for item_id_val, loc, fill_rate_val in fr_rows:
        alerts.append({
            "alert_type": "low_fill_rate",
            "severity": "red" if float(fill_rate_val) < 70 else "amber",
            "message": f"Fill rate {round(float(fill_rate_val), 1)}% for {item_id_val} at {loc}",
            "item_id": item_id_val,
            "loc": loc,
            "value": round(float(fill_rate_val), 1),
            "threshold": 85,
        })

    for item_id_val, loc, hhi_val in hhi_rows:
        alerts.append({
            "alert_type": "high_concentration",
            "severity": "red" if float(hhi_val) > 0.8 else "amber",
            "message": f"HHI {hhi_val} for {item_id_val} at {loc}",
            "item_id": item_id_val,
            "loc": loc,
            "value": float(hhi_val),
            "threshold": 0.6,
        })

    if mom_row:
        _sd, cur_cust, prev_cust, cur_demand, prev_demand = mom_row
        cur_c = int(cur_cust or 0)
        prev_c = int(prev_cust or 0)
        if prev_c > 0:
            churn_rate = round((prev_c - cur_c) / prev_c * 100, 1)
            if churn_rate > 10:
                alerts.append({
                    "alert_type": "high_churn",
                    "severity": "red" if churn_rate > 20 else "amber",
                    "message": f"Customer churn rate {churn_rate}% MoM",
                    "item_id": None,
                    "loc": None,
                    "value": churn_rate,
                    "threshold": 10,
                })
        cur_d = float(cur_demand or 0)
        prev_d = float(prev_demand or 0)
        if prev_d > 0:
            surge = round((cur_d - prev_d) / prev_d * 100, 1)
            if surge > 30:
                alerts.append({
                    "alert_type": "demand_surge",
                    "severity": "amber",
                    "message": f"New demand surge {surge}% MoM",
                    "item_id": None,
                    "loc": None,
                    "value": surge,
                    "threshold": 30,
                })

    return {"alerts": alerts}
