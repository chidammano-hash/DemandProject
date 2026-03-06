"""API tests for IPfeature8 Fill Rate endpoints."""
import pytest
import httpx
from unittest.mock import MagicMock, patch
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]


@pytest.mark.asyncio
async def test_fill_rate_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(0.95, 10000.0, 9500.0, 500.0, 3),
    )
    cursor.description = [
        ("portfolio_fill_rate",), ("total_ordered",), ("total_shipped",),
        ("total_shortage_qty",), ("partial_fulfillment_events",),
    ]
    cursor.fetchall.side_effect = [[], [], []]  # abc, worst, trend queries

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "portfolio_fill_rate" in data
    assert "by_abc" in data
    assert "worst_items" in data
    assert "trend" in data


@pytest.mark.asyncio
async def test_fill_rate_trend_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("2025-01-01", 0.93, 5000.0, 4650.0, 350.0),
    ])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert "months" in data


@pytest.mark.asyncio
async def test_fill_rate_detail_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(25,),
        fetchall_return=[
            ("ITEM1", "LOC1", "2025-01-01", 1000.0, 920.0, 0.92, 80.0, False, "A", None, "East"),
        ],
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/detail")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data


@pytest.mark.asyncio
async def test_fill_rate_detail_filter_abc():
    pool, conn, cursor = _make_pool(
        fetchone_return=(5,),
        fetchall_return=[("ITEM1", "LOC1", "2025-01-01", 1000.0, 950.0, 0.95, 50.0, True, "A", None, None)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/detail?abc_vol=A")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_fill_rate_detail_pagination():
    pool, conn, cursor = _make_pool(
        fetchone_return=(100,),
        fetchall_return=[("ITEM1", "LOC1", "2025-01-01", 100.0, 90.0, 0.9, 10.0, False, "B", None, None)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fill-rate/detail?limit=10&offset=20")
    assert resp.status_code == 200
