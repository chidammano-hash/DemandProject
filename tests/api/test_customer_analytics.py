"""Tests for customer analytics endpoints — /customer-analytics/*.

Item 19 pilot: handlers were converted to ``async def`` and use
``api.core._get_async_pool``. Tests therefore use :func:`make_async_pool`
(awaitable cursor methods) and patch the async pool getter.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_async_pool as _make_pool
from common.services.cache import reset_cache


# Per-test cache reset — endpoints share an in-memory cache singleton, so
# without this each test sees the prior test's response under a matching key.
@pytest.fixture(autouse=True)
def _isolate_cache():
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_planning_date(d=date(2026, 3, 1)):
    return patch("api.routers.intelligence.customer_analytics.get_planning_date", return_value=d)


def _patch_nomi():
    """Mock pgeocode so we don't need the data file."""
    return patch("api.routers.intelligence.customer_analytics._get_nomi", return_value=MagicMock())


# ---------------------------------------------------------------------------
# /customer-analytics/map
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_returns_locations():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("CA", 50, 10000.0, 9000.0, 1000.0),
        ("TX", 30, 8000.0, 7500.0, 500.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date(), _patch_nomi():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/map?metric=demand_qty&group_by=state")
    assert resp.status_code == 200
    data = resp.json()
    assert "locations" in data
    assert len(data["locations"]) == 2
    assert data["locations"][0]["label"] == "CA"
    assert data["locations"][0]["demand_qty"] == 10000.0
    assert data["locations"][0]["fill_rate"] == 90.0
    assert data["total_customers"] == 80
    assert data["total_demand"] == 18000.0


@pytest.mark.asyncio
async def test_map_with_item_filter():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("NY", 10, 5000.0, 4500.0, 500.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date(), _patch_nomi():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/map?item_id=ITEM001")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["locations"]) == 1


@pytest.mark.asyncio
async def test_map_empty_result():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = []
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date(), _patch_nomi():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/map")
    assert resp.status_code == 200
    data = resp.json()
    assert data["locations"] == []
    assert data["total_demand"] == 0


# ---------------------------------------------------------------------------
# /customer-analytics/treemap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_treemap_returns_hierarchy():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("CA", "On Premise", "Acme Corp", "C001", 5000.0, 4500.0),
        ("CA", "Off Premise", "Beta LLC", "C002", 3000.0, 2800.0),
        ("TX", "On Premise", "Gamma Inc", "C003", 2000.0, 1900.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/treemap")
    assert resp.status_code == 200
    data = resp.json()
    assert "tree" in data
    assert len(data["tree"]) >= 1
    # CA should be first (highest demand)
    assert data["tree"][0]["name"] == "CA"
    assert len(data["tree"][0]["children"]) == 2


# ---------------------------------------------------------------------------
# /customer-analytics/heatmap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heatmap_returns_matrix():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("ITEM001", "Widget A", "CA", 20, 5000.0, 4800.0),
        ("ITEM001", "Widget A", "TX", 15, 3000.0, 2900.0),
        ("ITEM002", "Gadget B", "CA", 10, 2000.0, 1900.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/heatmap?top_n=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "states" in data
    assert "cells" in data
    assert len(data["items"]) == 2
    assert "CA" in data["states"]


@pytest.mark.asyncio
async def test_heatmap_routes_through_item_state_mv():
    """F5.1 — the heatmap must source from mv_ca_item_state (the fast path),
    not the raw fact_customer_demand_monthly JOIN dim_item that took ~9.4 s
    cold. It must NOT scan dim_item, and channel/store_type filters must be
    applied against the MV columns."""
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("ITEM001", "Widget A", "CA", 20, 5000.0, 4800.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/customer-analytics/heatmap?top_n=10&channel=Grocery&store_type=CHAIN"
            )
    assert resp.status_code == 200
    executed_sql = cursor.execute.call_args[0][0]
    assert "mv_ca_item_state" in executed_sql
    assert "fact_customer_demand_monthly" not in executed_sql
    assert "dim_item" not in executed_sql
    # channel + store_type predicates applied against the MV columns
    assert "rpt_channel_desc" in executed_sql
    assert "store_type_desc" in executed_sql
    bound = cursor.execute.call_args[0][1]
    assert "Grocery" in bound
    assert "CHAIN" in bound


# ---------------------------------------------------------------------------
# /customer-analytics/channel-mix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_channel_mix_returns_tree():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("On Premise", "Restaurant", "Fine Dining", 30, 8000.0, 7500.0),
        ("On Premise", "Bar", "Sports Bar", 20, 4000.0, 3800.0),
        ("Off Premise", "Grocery", "Chain", 50, 12000.0, 11500.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/channel-mix")
    assert resp.status_code == 200
    data = resp.json()
    assert "tree" in data
    assert len(data["tree"]) >= 1


# ---------------------------------------------------------------------------
# /customer-analytics/segment-trends
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_segment_trends_returns_sparklines():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("On Premise", date(2026, 1, 1), 100, 5000.0, 4500.0, 500.0),
        ("On Premise", date(2026, 2, 1), 105, 5200.0, 4700.0, 500.0),
        ("Off Premise", date(2026, 1, 1), 200, 10000.0, 9500.0, 500.0),
        ("Off Premise", date(2026, 2, 1), 210, 10500.0, 10000.0, 500.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/segment-trends?segment_by=rpt_channel_desc")
    assert resp.status_code == 200
    data = resp.json()
    assert "segments" in data
    assert len(data["segments"]) == 2
    # Off Premise has more demand, should be first
    assert data["segments"][0]["segment"] == "Off Premise"
    assert len(data["segments"][0]["trend"]) == 2
    assert "fill_rate" in data["segments"][0]
    assert "mom_change" in data["segments"][0]


# ---------------------------------------------------------------------------
# /customer-analytics/ranking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ranking_demand_desc():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("C001", "Acme Corp", "CA", "On Premise", 10000.0, 9000.0, 1000.0),
        ("C002", "Beta LLC", "TX", "Off Premise", 8000.0, 7500.0, 500.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/ranking?sort=demand_desc&top_n=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "customers" in data
    assert len(data["customers"]) == 2
    assert data["customers"][0]["customer_name"] == "Acme Corp"
    assert data["customers"][0]["fill_rate"] == 90.0


@pytest.mark.asyncio
async def test_ranking_fill_rate_asc():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("C003", "Low Service", "FL", "On Premise", 5000.0, 3500.0, 1500.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/ranking?sort=fill_rate_asc")
    assert resp.status_code == 200
    data = resp.json()
    assert data["customers"][0]["fill_rate"] == 70.0


# ---------------------------------------------------------------------------
# /customer-analytics/oos-impact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oos_impact_customer_grain():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("C001", "Acme Corp", "CA", "On Premise", 10000.0, 9000.0, 1000.0),
        ("C002", "Beta LLC", "TX", "Off Premise", 5000.0, 2500.0, 2500.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/oos-impact?grain=customer")
    assert resp.status_code == 200
    data = resp.json()
    assert "bubbles" in data
    assert len(data["bubbles"]) == 2
    assert data["bubbles"][0]["fill_rate"] == 90.0
    assert data["grain"] == "customer"


@pytest.mark.asyncio
async def test_oos_impact_state_grain():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("CA", "CA", "CA", "All", 15000.0, 13000.0, 2000.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/oos-impact?grain=state")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["bubbles"]) == 1
    assert data["grain"] == "state"


# ---------------------------------------------------------------------------
# /customer-analytics/items
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_items_search():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("ITEM001", "Widget A"),
        ("ITEM002", "Widget B"),
    ]
    with patch("api.core._get_async_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/items?search=widget")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert len(data["items"]) == 2
    assert data["items"][0]["item_id"] == "ITEM001"


@pytest.mark.asyncio
async def test_items_empty_search():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [("ITEM001", "Widget A")]
    with patch("api.core._get_async_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/items")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_invalid_metric():
    pool, _, cursor = _make_pool()
    with patch("api.core._get_async_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/map?metric=invalid")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_map_invalid_group_by():
    pool, _, cursor = _make_pool()
    with patch("api.core._get_async_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/map?group_by=invalid")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ranking_invalid_sort():
    pool, _, cursor = _make_pool()
    with patch("api.core._get_async_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/ranking?sort=badval")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /customer-analytics/kpis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kpis_returns_six_metrics():
    pool, _, cursor = _make_pool()
    # Row: totals(demand,sales,oos,cust), cur(demand,sales,oos,cust),
    # prev(demand,sales,oos,cust), top10_demand. Values are full-range; delta is MoM.
    cursor.fetchone.return_value = (
        10000.0, 9000.0, 1000.0, 50,   # totals (full range)
        10000.0, 9000.0, 1000.0, 50,   # cur month
        8000.0, 7000.0, 1000.0, 45,    # prev month
        4000.0,                        # top10 demand (full range)
    )
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/kpis")
    assert resp.status_code == 200
    data = resp.json()
    assert "kpis" in data
    assert len(data["kpis"]) == 6
    keys = [k["key"] for k in data["kpis"]]
    assert "total_demand" in keys
    assert "fill_rate" in keys
    assert "oos_volume" in keys
    assert "active_customers" in keys
    assert "concentration_top10" in keys
    assert "order_demand_ratio" in keys
    # total_demand value is full-range total = 10000
    td = next(k for k in data["kpis"] if k["key"] == "total_demand")
    assert td["value"] == 10000.0
    # delta is MoM (cur 10000 vs prev 8000) = 25.0
    assert td["delta"] == 25.0


@pytest.mark.asyncio
async def test_kpis_concentration_and_ratio_have_null_delta():
    """U3.4 — concentration_top10 and order_demand_ratio have NO month-over-month
    computed (no prior-period anchor). They must report delta=null so the UI can
    show "no prior period" instead of a fabricated "0.0% MoM"."""
    pool, _, cursor = _make_pool()
    cursor.fetchone.return_value = (
        10000.0, 9000.0, 1000.0, 50,
        10000.0, 9000.0, 1000.0, 50,
        8000.0, 7000.0, 1000.0, 45,
        4000.0,
    )
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/kpis")
    assert resp.status_code == 200
    kpis = {k["key"]: k for k in resp.json()["kpis"]}
    assert kpis["concentration_top10"]["delta"] is None
    assert kpis["order_demand_ratio"]["delta"] is None
    # A genuinely-computed metric still carries a numeric delta.
    assert kpis["total_demand"]["delta"] == 25.0


@pytest.mark.asyncio
async def test_kpis_empty():
    pool, _, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/kpis")
    assert resp.status_code == 200
    assert resp.json()["kpis"] == []


# ---------------------------------------------------------------------------
# /customer-analytics/lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifecycle_returns_cohorts_and_waterfall():
    pool, _, cursor = _make_pool()
    # First query = cohort rows: (cohort_month, months_since, active, size)
    cohort_rows = [
        (date(2025, 6, 1), 0, 10, 10),
        (date(2025, 6, 1), 1, 8, 10),
        (date(2025, 7, 1), 0, 5, 5),
    ]
    # Second query = waterfall rows: (month, new_customers, churned_customers)
    waterfall_rows = [
        (date(2025, 6, 1), 10, 0),
        (date(2025, 7, 1), 5, 2),
    ]
    cursor.fetchall.side_effect = [cohort_rows, waterfall_rows]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/lifecycle")
    assert resp.status_code == 200
    data = resp.json()
    assert "cohorts" in data
    assert "waterfall" in data
    assert len(data["cohorts"]) == 2
    assert data["cohorts"][0]["cohort_month"] == "2025-06-01"
    assert data["cohorts"][0]["retention_pct"] == [100.0, 80.0]
    assert len(data["waterfall"]) == 2
    assert data["waterfall"][1]["net_change"] == 3  # 5 - 2


# ---------------------------------------------------------------------------
# /customer-analytics/demand-at-risk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_demand_at_risk_waterfall():
    pool, _, cursor = _make_pool()
    # Row: total_demand, concentration_risk, oos_loss, churn_risk
    cursor.fetchone.return_value = (100000.0, 20000.0, 5000.0, 3000.0)
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/demand-at-risk")
    assert resp.status_code == 200
    data = resp.json()
    assert "waterfall" in data
    assert len(data["waterfall"]) == 5
    cats = [w["category"] for w in data["waterfall"]]
    assert cats == ["total_demand", "concentration_risk", "oos_loss", "churn_risk", "secure_demand"]
    assert data["waterfall"][0]["value"] == 100000.0
    # secure = 100000 - 20000 - 5000 - 3000 = 72000
    assert data["waterfall"][4]["value"] == 72000.0


@pytest.mark.asyncio
async def test_demand_at_risk_empty():
    pool, _, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/demand-at-risk")
    assert resp.status_code == 200
    assert resp.json()["waterfall"] == []


# ---------------------------------------------------------------------------
# /customer-analytics/affinity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_affinity_returns_matrix():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("C001", "Acme Corp", "ITEM001", "Widget A", 5000.0),
        ("C001", "Acme Corp", "ITEM002", "Gadget B", 3000.0),
        ("C002", "Beta LLC", "ITEM001", "Widget A", 2000.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/affinity?top_n=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "customers" in data
    assert "items" in data
    assert "cells" in data
    assert len(data["customers"]) == 2
    assert len(data["items"]) == 2
    assert len(data["cells"]) == 3
    assert data["cells"][0]["demand_qty"] == 5000.0


# ---------------------------------------------------------------------------
# /customer-analytics/order-patterns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_order_patterns_returns_histogram_and_scatter():
    pool, _, cursor = _make_pool()
    # Rows: customer_no, customer_name, avg_interval, cv, order_count, total_demand
    cursor.fetchall.return_value = [
        ("C001", "Acme Corp", 1.0, 0.2, 12, 50000.0),
        ("C002", "Beta LLC", 3.0, 0.5, 4, 20000.0),
        ("C003", "Gamma Inc", 6.0, 1.2, 2, 5000.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/order-patterns")
    assert resp.status_code == 200
    data = resp.json()
    assert "frequency_histogram" in data
    assert "regularity_scatter" in data
    hist = {h["bucket"]: h["count"] for h in data["frequency_histogram"]}
    assert hist["monthly"] == 1
    assert hist["quarterly"] == 1
    assert hist["sporadic"] == 1
    assert len(data["regularity_scatter"]) == 3


# ---------------------------------------------------------------------------
# /customer-analytics/demand-flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_demand_flow_returns_sankey():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("LOC01", "CA", "On Premise", 5000.0),
        ("LOC01", "TX", "Off Premise", 3000.0),
        ("LOC02", "CA", "On Premise", 2000.0),
    ]
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/demand-flow")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "links" in data
    # Should have warehouse nodes, state nodes, and channel nodes
    node_names = {n["name"] for n in data["nodes"]}
    assert "WH_LOC01" in node_names
    assert "CA" in node_names
    assert "On Premise" in node_names
    # Links: 3 warehouse->state + 2 unique state->channel
    assert len(data["links"]) >= 4


# ---------------------------------------------------------------------------
# /customer-analytics/filter-options
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filter_options_returns_lists():
    # Endpoint now reads from mv_customer_filter_options (sql/173): one row
    # per category, value column is a TEXT[] of distinct values.
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("channels", ["Off Premise", "On Premise"]),
        ("store_types", ["Bar", "Grocery", "Restaurant"]),
        ("states", ["CA", "FL", "TX"]),
    ]
    with patch("api.core._get_async_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/filter-options")
    assert resp.status_code == 200
    data = resp.json()
    assert data["channels"] == ["Off Premise", "On Premise"]
    assert data["store_types"] == ["Bar", "Grocery", "Restaurant"]
    assert data["states"] == ["CA", "FL", "TX"]


@pytest.mark.asyncio
async def test_filter_options_empty():
    # MV returns three rows even when dim_customer is empty (NULLs coalesced
    # to empty arrays at MV-build time).
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("channels", []),
        ("store_types", []),
        ("states", []),
    ]
    with patch("api.core._get_async_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/filter-options")
    assert resp.status_code == 200
    data = resp.json()
    assert data["channels"] == []
    assert data["store_types"] == []
    assert data["states"] == []


# ---------------------------------------------------------------------------
# /customer-analytics/alerts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alerts_returns_low_fill_rate():
    pool, _, cursor = _make_pool()
    # fill rate query returns rows; hhi returns empty; mom returns None
    fr_rows = [("ITEM001", "LOC01", 65.0), ("ITEM002", "LOC02", 80.0)]
    hhi_rows = []
    cursor.fetchall.side_effect = [fr_rows, hhi_rows]
    cursor.fetchone.return_value = None  # mom query
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data
    assert len(data["alerts"]) == 2
    # 65% < 70 => red, 80% < 85 => amber
    severities = {a["value"]: a["severity"] for a in data["alerts"]}
    assert severities[65.0] == "red"
    assert severities[80.0] == "amber"


@pytest.mark.asyncio
async def test_alerts_hhi_and_churn():
    pool, _, cursor = _make_pool()
    fr_rows = []
    hhi_rows = [("ITEM005", "LOC03", 0.85)]
    cursor.fetchall.side_effect = [fr_rows, hhi_rows]
    # mom: startdate, cur_cust, prev_cust, cur_demand, prev_demand
    # churn = (100-70)/100 = 30% > 10% => alert
    cursor.fetchone.return_value = (date(2026, 2, 1), 70, 100, 50000.0, 35000.0)
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/alerts")
    assert resp.status_code == 200
    data = resp.json()
    types = [a["alert_type"] for a in data["alerts"]]
    assert "high_concentration" in types
    assert "high_churn" in types
    # demand surge: (50000-35000)/35000 = 42.9% > 30%
    assert "demand_surge" in types


@pytest.mark.asyncio
async def test_alerts_no_alerts():
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [[], []]  # no fill-rate or hhi alerts
    cursor.fetchone.return_value = None
    with patch("api.core._get_async_pool", return_value=pool), \
         _patch_planning_date():
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/alerts")
    assert resp.status_code == 200
    assert resp.json()["alerts"] == []
