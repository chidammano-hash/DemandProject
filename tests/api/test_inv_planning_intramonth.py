"""API tests for IPfeature14 Intra-Month Stockout endpoints."""
import pytest
from unittest.mock import MagicMock, patch
from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_intramonth_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(500, 45, 12, 0.08, 120, 3500.0, 25.0),
    )
    cursor.description = [
        ("total_records",), ("items_with_stockout",), ("items_with_extended_stockout",),
        ("avg_stockout_day_rate",), ("total_stockout_days",), ("total_est_lost_sales",),
        ("avg_qty_on_hand",),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/intramonth-stockouts/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_records" in data


@pytest.mark.asyncio
async def test_intramonth_summary_with_filters():
    pool, conn, cursor = _make_pool(fetchone_return=(100, 10, 2, 0.05, 30, 500.0, 40.0))
    cursor.description = [
        ("total_records",), ("items_with_stockout",), ("items_with_extended_stockout",),
        ("avg_stockout_day_rate",), ("total_stockout_days",), ("total_est_lost_sales",),
        ("avg_qty_on_hand",),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/intramonth-stockouts/summary?abc_vol=A")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_intramonth_detail_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(50,),
        fetchall_return=[
            ("ITEM1", "LOC1", "2025-01-01", 22, 5, 0.23, 0.0, 100.0, 45.0, 250.0, True, False, "A", "AX", "cluster1"),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/intramonth-stockouts/detail")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data


@pytest.mark.asyncio
async def test_intramonth_detail_filter_had_stockout():
    pool, conn, cursor = _make_pool(
        fetchone_return=(10,),
        fetchall_return=[("ITEM2", "LOC2", "2025-02-01", 28, 8, 0.29, 0.0, 50.0, 20.0, 180.0, True, True, "B", "BZ", None)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/intramonth-stockouts/detail?had_stockout=true")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_intramonth_daily_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("2025-01-15", 50.0, 200.0, 15.0),
        ("2025-01-16", 40.0, 220.0, 20.0),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/intramonth-stockouts/daily?item=ITEM1&location=LOC1")
    assert resp.status_code == 200
    data = resp.json()
    assert "daily" in data
    assert "item_id" in data


@pytest.mark.asyncio
async def test_intramonth_daily_missing_required_params():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/intramonth-stockouts/daily")
    assert resp.status_code == 422
