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

_MARKER_LIMIT = 500


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
) -> str:
    """Build WHERE clause fragments and append params. Returns SQL fragment."""
    df, dt = _default_date_range()
    clauses = ["f.startdate >= %s", "f.startdate < %s"]
    params.extend([date_from or df, date_to or dt])
    if item_id:
        clauses.append("f.item_id = %s")
        params.append(item_id)
    if channel:
        clauses.append("c.rpt_channel_desc = %s")
        params.append(channel)
    if store_type:
        clauses.append("c.store_type_desc = %s")
        params.append(store_type)
    return " AND ".join(clauses)


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
def customer_analytics_map(
    response: FastAPIResponse,
    metric: str = Query(default="demand_qty", pattern="^(customer_count|demand_qty|sales_qty|oos_qty|fill_rate)$"),
    group_by: str = Query(default="state", pattern="^(state|city|zip)$"),
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
):
    """Demand-aware customer map with metric selection."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where(params, item_id, date_from, date_to, channel, store_type)

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
    params.append(_MARKER_LIMIT)

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

    sql = f"""
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
        ORDER BY SUM(f.demand_qty) DESC
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Determine top N items by total demand
    item_totals: dict[str, float] = {}
    item_descs: dict[str, str] = {}
    for item_id, item_desc, _state, _cc, demand, _sales in rows:
        item_totals[item_id] = item_totals.get(item_id, 0) + float(demand or 0)
        item_descs[item_id] = item_desc or item_id

    top_items = sorted(item_totals, key=item_totals.get, reverse=True)[:top_n]  # type: ignore[arg-type]
    top_item_set = set(top_items)

    # Determine states (sorted by total demand)
    state_totals: dict[str, float] = {}
    for _item_id, _desc, state, _cc, demand, _sales in rows:
        state = str(state).strip()
        state_totals[state] = state_totals.get(state, 0) + float(demand or 0)
    top_states = sorted(state_totals, key=state_totals.get, reverse=True)[:30]  # type: ignore[arg-type]
    top_state_set = set(top_states)

    # Build matrix
    cells: list[dict[str, Any]] = []
    for item_id, _desc, state, cc, demand, sales in rows:
        state = str(state).strip()
        if item_id not in top_item_set or state not in top_state_set:
            continue
        d = float(demand or 0)
        s = float(sales or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0
        cells.append({
            "item_id": item_id,
            "state": state,
            "demand_qty": round(d, 1),
            "customer_count": int(cc),
            "fill_rate": fr,
        })

    return {
        "items": [{"item_id": it, "item_desc": item_descs[it]} for it in top_items],
        "states": top_states,
        "cells": cells,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# 4. Channel Mix Sunburst
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/channel-mix")
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

    sql = f"""
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
        ORDER BY {seg_col}, f.startdate
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
):
    """Typeahead item search for the customer analytics filter bar."""
    set_cache(response, max_age=600)
    search = search.strip()

    if search:
        sql = """
            SELECT DISTINCT i.item_id, i.item_desc
            FROM dim_item i
            WHERE i.item_id ILIKE %s OR i.item_desc ILIKE %s
            ORDER BY i.item_id
            LIMIT 50
        """
        pattern = f"%{search}%"
        query_params = [pattern, pattern]
    else:
        sql = """
            SELECT DISTINCT i.item_id, i.item_desc
            FROM dim_item i
            ORDER BY i.item_id
            LIMIT 50
        """
        query_params = []

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, query_params)
        rows = cur.fetchall()

    items = [{"item_id": r[0], "item_desc": r[1] or r[0]} for r in rows]
    return {"items": items}
