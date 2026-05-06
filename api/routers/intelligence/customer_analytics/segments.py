"""Customer-analytics segmentation endpoints — channel mix, segment trends, filter options."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _CA_CACHE, _build_where

router = APIRouter(tags=["customer-analytics"])


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
