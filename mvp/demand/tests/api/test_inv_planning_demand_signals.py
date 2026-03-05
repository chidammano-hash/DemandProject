"""API tests for IPfeature9 Demand Signals endpoints."""
import pytest
import datetime
from unittest.mock import MagicMock, patch


def _make_pool(fetchall_return=None, fetchone_return=None):
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return or (0,)
    cursor.description = [("col",)]
    cursor.rowcount = 1
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn, cursor


@pytest.mark.asyncio
async def test_demand_signals_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            datetime.date(2026, 3, 1), 500, 120, 80, 300, 15, 40, 20
        ),
    )
    cursor.description = [
        ("signal_date",), ("total_items_with_signals",), ("above_plan",),
        ("below_plan",), ("on_plan",), ("urgent_alerts",), ("watch_alerts",),
        ("projected_stockouts",),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/demand-signals/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_items_with_signals" in data
    assert "urgent_alerts" in data
    assert "projected_stockouts" in data


@pytest.mark.asyncio
async def test_demand_signals_list_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(200,),
        fetchall_return=[
            ("ITEM1", "LOC1", datetime.date(2026, 3, 1), "above_plan", "watch",
             150.0, 310.0, 280.0, 10.7, False, False, 80.0, False, 14, "A"),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/demand-signals")
    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_demand_signals_filter_urgent():
    pool, conn, cursor = _make_pool(
        fetchone_return=(5,),
        fetchall_return=[
            ("ITEM2", "LOC2", datetime.date(2026, 3, 1), "above_plan", "urgent",
             200.0, 450.0, 300.0, 50.0, True, False, 20.0, True, 10, "A"),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/demand-signals?alert_priority=urgent")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_demand_signals_item_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        "ITEM1", "LOC1", datetime.date(2026, 3, 1), "above_plan", "watch",
        150.0, 310.0, 280.0, 10.7, 15, 14, 80.0, False,
    )
    cursor.fetchall.return_value = [
        (datetime.date(2026, 3, 1), 150.0, 140.0),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/demand-signals/item?item=ITEM1&location=LOC1")
    assert resp.status_code == 200
    data = resp.json()
    assert "daily_series" in data
    assert "signal_type" in data


@pytest.mark.asyncio
async def test_demand_signals_item_missing_params():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/demand-signals/item")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_demand_signals_item_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/demand-signals/item?item=UNKNOWN&location=LOC1")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_demand_signals_empty_returns_200():
    pool, conn, cursor = _make_pool(fetchone_return=(0,), fetchall_return=[])
    cursor.fetchone.return_value = (None, 0, 0, 0, 0, 0, 0, 0)
    cursor.description = [
        ("signal_date",), ("total_items_with_signals",), ("above_plan",),
        ("below_plan",), ("on_plan",), ("urgent_alerts",), ("watch_alerts",),
        ("projected_stockouts",),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/demand-signals/summary")
    assert resp.status_code == 200
