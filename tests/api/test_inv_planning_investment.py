"""API tests for IPfeature13 investment optimization endpoints."""
import pytest
import datetime
from unittest.mock import MagicMock, patch
from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_efficient_frontier_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (10000.0, 15, 0.90, "ITEM1"),
        (20000.0, 35, 0.93, "ITEM2"),
        (30000.0, 55, 0.96, "ITEM3"),
    ])
    # summary fetchone: plan_id, total_items, rec_investment, cur_investment, cur_csl, rec_csl
    cursor.fetchone.return_value = ("plan-1", 55, 30000.0, 20000.0, 0.85, 0.94)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/investment/efficient-frontier")
    assert resp.status_code == 200
    data = resp.json()
    assert "curve" in data or "frontier" in data


@pytest.mark.asyncio
async def test_efficient_frontier_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/investment/efficient-frontier")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("curve", data.get("frontier")) == []


@pytest.mark.asyncio
async def test_investment_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(50, 100000.0, 120000.0, 20000.0, 0.85, 0.92)
    )
    # description needed for dict(zip(cols, row))
    cursor.description = [
        ("total_items",), ("total_current_investment",),
        ("total_recommended_investment",), ("total_investment_gap",),
        ("portfolio_csl_current",), ("portfolio_csl_recommended",),
    ]
    cursor.fetchall.return_value = []  # top_roi_items
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/investment/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_items" in data or "top_roi_items" in data


@pytest.mark.asyncio
async def test_investment_detail_200():
    # 16 columns: item_no, loc, abc_vol, abc_xyz_segment, current_ss_qty, current_ss_value,
    # current_csl, recommended_ss_qty, recommended_ss_value, recommended_csl,
    # ss_increment_qty, investment_increment, csl_increment, marginal_roi,
    # investment_rank, cumulative_investment
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("ITEM1", "LOC1", "A", "AX",
         250.0, 5000.0, 0.92,
         300.0, 6000.0, 0.96,
         50.0, 1000.0, 0.04, 0.000040,
         1, 1000.0),
        ("ITEM2", "LOC2", "B", "BY",
         180.0, 3000.0, 0.80,
         220.0, 3700.0, 0.88,
         40.0, 700.0, 0.08, 0.000114,
         2, 1700.0),
    ])
    cursor.fetchone.return_value = (2,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/investment/detail")
    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data or "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_investment_detail_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    cursor.fetchone.return_value = (0,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/investment/detail")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
