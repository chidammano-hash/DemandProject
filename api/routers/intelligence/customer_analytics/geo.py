"""Geographic customer-analytics endpoints — map, treemap, heatmap, demand flow.

Item 19 pilot: handlers are ``async def`` and use ``get_async_conn``.
"""
from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_async_conn, get_async_read_only_conn, set_cache

from ._helpers import (
    _CA_CACHE,
    _MARKER_LIMIT_CITY_ZIP,
    _MARKER_LIMIT_STATE,
    _STATE_CENTROIDS,
    _add_state_coords,
    _build_where,
    _build_where_item_state,
    _build_where_mv,
    _customer_activity_source,
    _get_nomi,
)

router = APIRouter(tags=["customer-analytics"])


# ---------------------------------------------------------------------------
# 1. Enhanced Demand Map
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/map")
@_CA_CACHE
async def customer_analytics_map(
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
    # Route through MV when no item filter — the MV inlines state, city,
    # and zip from dim_customer.
    source_from, uses_mv = _customer_activity_source(item_id)
    dim_alias = "f" if uses_mv else "c"
    if uses_mv:
        where = _build_where_mv(params, date_from, date_to, channel, store_type, state=state)
    else:
        where = _build_where(params, item_id, date_from, date_to, channel, store_type, state=state)

    geo_col = {
        "state": f"{dim_alias}.state",
        "city": f"{dim_alias}.city",
        "zip": f"{dim_alias}.zip",
    }[group_by]

    sql = f"""
        SELECT {geo_col} AS geo_label,
               {f'{dim_alias}.state AS state_col,' if group_by != 'state' else ''}
               COUNT(DISTINCT {dim_alias}.customer_no) AS customer_count,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty,
               COALESCE(SUM(f.oos_qty), 0) AS oos_qty
        FROM {source_from}
        WHERE {where}
          AND {geo_col} IS NOT NULL AND TRIM({geo_col}) != ''
        GROUP BY {geo_col} {',' + f'{dim_alias}.state' if group_by != 'state' else ''}
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT %s
    """
    params.append(_MARKER_LIMIT_STATE if group_by == "state" else _MARKER_LIMIT_CITY_ZIP)

    # Read-only geo aggregate — replica-safe (Item 24).
    async with get_async_read_only_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

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
async def customer_analytics_treemap(
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
    # Route through MV when no item filter — the MV inlines state,
    # rpt_channel_desc, customer_name.
    source_from, uses_mv = _customer_activity_source(item_id)
    dim_alias = "f" if uses_mv else "c"
    if uses_mv:
        where = _build_where_mv(params, date_from, date_to, channel, store_type)
    else:
        where = _build_where(params, item_id, date_from, date_to, channel, store_type)

    sql = f"""
        SELECT {dim_alias}.state,
               COALESCE({dim_alias}.rpt_channel_desc, 'Unknown') AS channel,
               {dim_alias}.customer_name,
               {dim_alias}.customer_no,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty,
               COALESCE(SUM(f.sales_qty), 0) AS sales_qty
        FROM {source_from}
        WHERE {where}
          AND {dim_alias}.state IS NOT NULL AND TRIM({dim_alias}.state) != ''
        GROUP BY {dim_alias}.state, {dim_alias}.rpt_channel_desc, {dim_alias}.customer_name, {dim_alias}.customer_no
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT 500
    """

    # Read-only treemap aggregate — replica-safe (Item 24).
    async with get_async_read_only_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

    # Build hierarchy: state -> channel -> customer.
    # Track demand/sales sums at every level so we can derive a weighted
    # fill_rate at parents — the frontend treemap's visualMap colors by
    # `dimension: "fill_rate"` and ECharts hides any node that lacks the
    # dimension (with leafDepth=2 that means channel-level rectangles
    # disappear entirely).
    states: dict[str, dict[str, Any]] = {}
    for state, ch, cust_name, cust_no, demand, sales in rows:
        state = str(state).strip()
        ch = str(ch).strip()
        d = float(demand or 0)
        s = float(sales or 0)
        fr = round(s / d * 100, 1) if d > 0 else 100.0

        if state not in states:
            states[state] = {"name": state, "value": 0, "_d": 0.0, "_s": 0.0, "children": {}}
        states[state]["value"] += d
        states[state]["_d"] += d
        states[state]["_s"] += s

        channels = states[state]["children"]
        if ch not in channels:
            channels[ch] = {"name": ch, "value": 0, "_d": 0.0, "_s": 0.0, "children": []}
        channels[ch]["value"] += d
        channels[ch]["_d"] += d
        channels[ch]["_s"] += s
        channels[ch]["children"].append({
            "name": cust_name or cust_no,
            "value": round(d, 1),
            "fill_rate": fr,
        })

    # Convert to list format and propagate weighted fill_rate up the tree.
    tree: list[dict[str, Any]] = []
    for s_data in sorted(states.values(), key=lambda x: x["value"], reverse=True):
        children = []
        for ch_data in sorted(s_data["children"].values(), key=lambda x: x["value"], reverse=True):
            ch_d, ch_s = ch_data.pop("_d"), ch_data.pop("_s")
            ch_data["value"] = round(ch_data["value"], 1)
            ch_data["fill_rate"] = round(ch_s / ch_d * 100, 1) if ch_d > 0 else 100.0
            # Keep top 10 customers per channel
            ch_data["children"] = sorted(ch_data["children"], key=lambda x: x["value"], reverse=True)[:10]
            children.append(ch_data)
        s_d, s_s = s_data.pop("_d"), s_data.pop("_s")
        tree.append({
            "name": s_data["name"],
            "value": round(s_data["value"], 1),
            "fill_rate": round(s_s / s_d * 100, 1) if s_d > 0 else 100.0,
            "children": children,
        })

    return {"tree": tree[:30]}  # Top 30 states


# ---------------------------------------------------------------------------
# 3. Item x State Heatmap
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/heatmap")
@_CA_CACHE
async def customer_analytics_heatmap(
    response: FastAPIResponse,
    metric: str = Query(default="demand_qty", pattern="^(demand_qty|customer_count|fill_rate)$"),
    top_n: int = Query(default=25, ge=5, le=100),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    store_type: str | None = Query(default=None),
):
    """Item x State heatmap matrix.

    F5.1: sources the (item_id, item_desc, state) aggregate from the
    pre-joined/pre-aggregated ``mv_ca_item_state`` instead of the raw
    fact_customer_demand_monthly JOIN dim_customer JOIN dim_item — the same
    fast-path pattern the other CA panels use. Cold load dropped from ~9.4 s
    to well under 1.5 s. The MV's ``item_desc`` is pre-resolved, so the
    endpoint never touches the ~500k-row dim_item. ``customer_count`` is the
    sum of per-(channel, store_type, month) distinct customer counts — a
    close upper-bound used only as a secondary cell metric (the headline
    metric is demand_qty).
    """
    set_cache(response, max_age=300)
    params: list[Any] = []
    where = _build_where_item_state(params, date_from, date_to, channel, store_type)

    # Push the (top_n items x 30 states) reduction into Postgres so we don't
    # ship every (item, state) aggregate row to Python just to discard most.
    sql = f"""
        WITH agg AS (
            SELECT m.item_id,
                   m.item_desc,
                   m.state,
                   COALESCE(SUM(m.customer_count), 0) AS customer_count,
                   COALESCE(SUM(m.demand_qty), 0) AS demand_qty,
                   COALESCE(SUM(m.sales_qty), 0) AS sales_qty
            FROM mv_ca_item_state m
            WHERE {where}
            GROUP BY m.item_id, m.item_desc, m.state
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

    # Read-only heatmap aggregate — replica-safe (Item 24).
    async with get_async_read_only_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

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
# 14. Demand Flow Sankey
# ---------------------------------------------------------------------------

@router.get("/customer-analytics/demand-flow")
@_CA_CACHE
async def customer_analytics_demand_flow(
    response: FastAPIResponse,
    item_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    """Sankey nodes + links: warehouse -> state -> channel."""
    set_cache(response, max_age=300)
    params: list[Any] = []
    # Route through MV when no item filter — the MV grain now includes
    # location_id and inlines state + rpt_channel_desc, which is exactly the
    # 3-tuple this endpoint groups on.
    source_from, uses_mv = _customer_activity_source(item_id)
    dim_alias = "f" if uses_mv else "c"
    if uses_mv:
        where = _build_where_mv(params, date_from, date_to, None, None)
    else:
        where = _build_where(params, item_id, date_from, date_to, None, None)

    sql = f"""
        SELECT f.location_id,
               COALESCE({dim_alias}.state, 'Unknown') AS state,
               COALESCE({dim_alias}.rpt_channel_desc, 'Unknown') AS channel,
               COALESCE(SUM(f.demand_qty), 0) AS demand_qty
        FROM {source_from}
        WHERE {where}
        GROUP BY f.location_id, {dim_alias}.state, {dim_alias}.rpt_channel_desc
        HAVING SUM(f.demand_qty) > 0
        ORDER BY SUM(f.demand_qty) DESC
        LIMIT 500
    """

    async with get_async_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

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

