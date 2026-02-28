"""Tests for inventory API endpoints."""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


@pytest.fixture
def mock_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = mock_conn

    return pool, mock_conn, mock_cursor


# ---------------------------------------------------------------------------
# /inventory/position
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inventory_position_returns_200(mock_pool):
    """GET /inventory/position returns 200 with positions."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (10,)
    cursor.fetchall.return_value = [
        ("item1", "loc1", "2025-06-01", 30.0, 100.0, 150.0, 50.0, 25.0),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/position")
            assert resp.status_code == 200
            data = resp.json()
            assert "positions" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert len(data["positions"]) == 1
            pos = data["positions"][0]
            assert pos["item_no"] == "item1"
            assert pos["loc"] == "loc1"
            assert pos["qty_on_hand"] == 100.0


@pytest.mark.asyncio
async def test_inventory_position_with_filters(mock_pool):
    """GET /inventory/position with item and location filters."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (5,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/position?item=12345&location=LOC1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 5


@pytest.mark.asyncio
async def test_inventory_position_with_sort(mock_pool):
    """GET /inventory/position respects sort_by and sort_dir."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/position?sort_by=qty_on_hand&sort_dir=asc")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_inventory_position_invalid_sort_falls_back(mock_pool):
    """Invalid sort_by column should fall back to snapshot_date."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/position?sort_by=nonexistent")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_inventory_position_pagination(mock_pool):
    """GET /inventory/position respects limit and offset params."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (100,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/position?limit=10&offset=20")
            assert resp.status_code == 200
            data = resp.json()
            assert data["limit"] == 10
            assert data["offset"] == 20
            assert data["total"] == 100


@pytest.mark.asyncio
async def test_inventory_position_limit_validation(mock_pool):
    """limit > 1000 should be rejected with 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/position?limit=2000")
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /inventory/kpis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inventory_kpis_returns_200(mock_pool):
    """GET /inventory/kpis returns 200 with supply chain KPIs."""
    pool, _, cursor = mock_pool
    # Query 1: latest snapshot (on_hand, on_order, items, locations)
    # Query 2: agg trailing months (sum_avg_oh, sum_monthly, w_lead_time, count_months)
    cursor.fetchone.side_effect = [
        (50000.0, 15000.0, 500, 50),
        (40000.0, 120000.0, 30.5, 3),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/kpis")
            assert resp.status_code == 200
            data = resp.json()
            assert "total_on_hand" in data
            assert "total_on_order" in data
            assert "avg_lead_time_days" in data
            assert "dos" in data
            assert "woc" in data
            assert "inventory_turns" in data
            assert "lt_coverage" in data
            assert "distinct_items" in data
            assert "distinct_locations" in data
            assert "months_covered" in data
            assert data["total_on_hand"] == 50000.0
            assert data["distinct_items"] == 500
            # Removed fields should NOT be present
            assert "total_inventory_value" not in data
            assert "snapshot_count" not in data


@pytest.mark.asyncio
async def test_inventory_kpis_with_filters(mock_pool):
    """GET /inventory/kpis with item, location, and months filters."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (200.0, 100.0, 5, 2),
        (180.0, 600.0, 15.0, 6),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/kpis?item=12345&location=LOC1&months=6")
            assert resp.status_code == 200
            data = resp.json()
            assert data["months_covered"] == 6


@pytest.mark.asyncio
async def test_inventory_kpis_no_data(mock_pool):
    """When fetchone returns None, endpoint should return zeros/nulls."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [None, None]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/kpis")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_on_hand"] == 0.0
            assert data["distinct_items"] == 0
            assert data["dos"] is None
            assert data["woc"] is None
            assert data["months_covered"] == 3


@pytest.mark.asyncio
async def test_inventory_kpis_dos_calculation(mock_pool):
    """DOS = total_on_hand / portfolio_daily_demand."""
    pool, _, cursor = mock_pool
    # on_hand = 1000, sum_monthly=608.8, count_months=1
    # portfolio_daily = 608.8 / 1 / 30.44 = 20.0 => DOS = 1000 / 20 = 50
    cursor.fetchone.side_effect = [
        (1000.0, 200.0, 10, 5),
        (900.0, 608.8, 14.0, 1),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/kpis")
            assert resp.status_code == 200
            data = resp.json()
            assert data["dos"] == 50.0
            assert data["woc"] is not None


@pytest.mark.asyncio
async def test_inventory_kpis_zero_demand_null_dos(mock_pool):
    """When daily sales is zero, DOS/WOC should be null."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (500.0, 100.0, 5, 2),
        (400.0, 0.0, 15.0, 3),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/kpis")
            data = resp.json()
            assert data["dos"] is None
            assert data["woc"] is None


# ---------------------------------------------------------------------------
# /inventory/trend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inventory_trend_returns_200(mock_pool):
    """GET /inventory/trend returns 200 with monthly trend data."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("2025-06-01", 100000.0, 50000.0, 250000.0, 30.0, 45.2),
        ("2025-07-01", 110000.0, 55000.0, 260000.0, 31.0, 42.8),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/trend?months=6")
            assert resp.status_code == 200
            data = resp.json()
            assert "trend" in data
            assert len(data["trend"]) == 2
            point = data["trend"][0]
            assert "month" in point
            assert "total_on_hand" in point
            assert "total_on_order" in point
            assert "monthly_sales" in point
            assert "avg_lead_time" in point
            assert "dos" in point
            # Old field names should NOT be present
            assert "avg_on_hand" not in point
            assert "total_mtd_sales" not in point


@pytest.mark.asyncio
async def test_inventory_trend_empty(mock_pool):
    """GET /inventory/trend returns empty trend list when no data."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/trend")
            assert resp.status_code == 200
            data = resp.json()
            assert data["trend"] == []


@pytest.mark.asyncio
async def test_inventory_trend_with_filters(mock_pool):
    """GET /inventory/trend with item and location filters."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/trend?item=12345&location=LOC1&months=12")
            assert resp.status_code == 200
            data = resp.json()
            assert "trend" in data


# ---------------------------------------------------------------------------
# /inventory/item-detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inventory_item_detail_returns_200(mock_pool):
    """GET /inventory/item-detail returns 200 with snapshot history."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("2025-06-15", 30.0, 100.0, 150.0, 50.0, 25.0),
        ("2025-07-15", 28.0, 90.0, 140.0, 45.0, 30.0),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/item-detail?item=12345&location=LOC1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["item"] == "12345"
            assert data["location"] == "LOC1"
            assert "snapshots" in data
            assert len(data["snapshots"]) == 2
            snap = data["snapshots"][0]
            assert "snapshot_date" in snap
            assert "lead_time_days" in snap
            assert "qty_on_hand" in snap
            assert "qty_on_hand_on_order" in snap
            assert "qty_on_order" in snap
            assert "mtd_sales" in snap


@pytest.mark.asyncio
async def test_inventory_item_detail_with_months(mock_pool):
    """GET /inventory/item-detail respects months parameter."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/item-detail?item=12345&location=LOC1&months=24")
            assert resp.status_code == 200
            data = resp.json()
            assert data["item"] == "12345"
            assert data["location"] == "LOC1"
            assert data["snapshots"] == []


@pytest.mark.asyncio
async def test_inventory_item_detail_missing_params(mock_pool):
    """GET /inventory/item-detail without both params returns 422."""
    pool, _, cursor = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/item-detail")
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_inventory_item_detail_missing_item(mock_pool):
    """GET /inventory/item-detail without item param returns 422."""
    pool, _, cursor = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/item-detail?location=LOC1")
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_inventory_item_detail_missing_location(mock_pool):
    """GET /inventory/item-detail without location param returns 422."""
    pool, _, cursor = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/item-detail?item=12345")
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_inventory_item_detail_null_values(mock_pool):
    """Null DB values should map to None in the response."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        (date(2025, 3, 1), None, None, None, None, None),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/item-detail?item=100320&location=1401-BULK")
            assert resp.status_code == 200
            data = resp.json()
            snap = data["snapshots"][0]
            assert snap["snapshot_date"] == "2025-03-01"
            assert snap["lead_time_days"] is None
            assert snap["qty_on_hand"] is None
            assert snap["qty_on_order"] is None
            assert snap["mtd_sales"] is None


# ---------------------------------------------------------------------------
# /inventory/trend — DOS-specific tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inventory_trend_includes_dos(mock_pool):
    """Trend response should include dos field per month."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("2025-06-01", 100000.0, 50000.0, 250000.0, 30.0, 45.2),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/trend")
            data = resp.json()
            assert data["trend"][0]["dos"] == 45.2


@pytest.mark.asyncio
async def test_inventory_trend_null_dos(mock_pool):
    """When DOS is NULL in DB, it should be null in JSON."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("2025-06-01", 100000.0, 50000.0, 250000.0, 30.0, None),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory/trend")
            data = resp.json()
            assert data["trend"][0]["dos"] is None
