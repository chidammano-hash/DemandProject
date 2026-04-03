"""Tests for customer analytics endpoints — /customer-analytics/*."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_planning_date(d=date(2026, 3, 1)):
    return patch("api.routers.customer_analytics.get_planning_date", return_value=d)


def _patch_nomi():
    """Mock pgeocode so we don't need the data file."""
    return patch("api.routers.customer_analytics._get_nomi", return_value=MagicMock())


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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool), \
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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
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
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/map?metric=invalid")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_map_invalid_group_by():
    pool, _, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/map?group_by=invalid")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ranking_invalid_sort():
    pool, _, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/customer-analytics/ranking?sort=badval")
    assert resp.status_code == 422
