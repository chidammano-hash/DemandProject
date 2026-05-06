"""Shared helpers for customer-analytics endpoints.

Geocoding, default date range, WHERE-clause builders, and the per-group
server-side cache decorator all live here so each sub-router stays small
and free of duplicated boilerplate.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any

import pgeocode
from dateutil.relativedelta import relativedelta

from common.core.planning_date import get_planning_date
from common.services.cache import cached_async

# All customer-analytics aggregates hit fact_customer_demand_monthly with
# the same join pattern. They are heavy (single-digit to ~16 second queries
# on a year of data) but the inputs are stable per-filter, so a 5-minute
# server-side cache turns repeat hits — which dominate dashboard usage —
# into millisecond responses. Invalidate via the "customer_analytics" group
# after a customer demand reload.
#
# Item 19 pilot: handlers are ``async def`` so this uses ``cached_async``
# (the async sibling of ``cached_sync``) which awaits the wrapped coroutine.
_CA_CACHE = cached_async(ttl=300, group="customer_analytics")

logger = logging.getLogger(__name__)

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
    state: str | None = None,
) -> str:
    """Variant of _build_where for queries that hit mv_customer_activity_monthly.

    The MV has the dim_customer columns inlined under the same `f` alias used
    by the original queries (channel, sub_channel, store_type, state, city,
    zip, customer_name, chain_type, location_id), so callers can swap source
    tables with minimal changes. No item_id filter — the MV is item-aggregated
    by design (use the raw fact table when an item filter is required).
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
    if state:
        clauses.append("UPPER(f.state) = %s")
        params.append(state.strip().upper())
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
