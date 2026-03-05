"""API tests for IPfeature15 Control Tower endpoints."""
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
async def test_control_tower_kpis_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            datetime.datetime(2026, 3, 1, 10, 0, 0),  # computed_at
            1000, 600, 250, 100, 50,  # total_dfus, healthy, monitor, at_risk, critical
            72.5, 0.85, 120, 0.12, 21.0,  # health scores/coverage
            45, 12, 18, 250000.0,  # exceptions
            0.94, 5000.0,  # fill rate
            8, 15,  # demand signals
            30, 5,  # intramonth
        )
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/kpis")
    assert resp.status_code == 200
    data = resp.json()
    assert "health" in data
    assert "exceptions" in data
    assert "fill_rate" in data
    assert "demand_signals" in data
    assert "intramonth" in data


@pytest.mark.asyncio
async def test_control_tower_kpis_avg_health_range():
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            datetime.datetime(2026, 3, 1),
            500, 300, 150, 40, 10,
            75.0, 0.80, 60, 0.12, 18.0,
            20, 5, 8, 100000.0,
            0.92, 2000.0, 3, 7, 15, 2,
        )
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/kpis")
    assert resp.status_code == 200
    health = resp.json()["health"]
    score = health.get("avg_health_score")
    if score is not None:
        assert 0 <= score <= 100


@pytest.mark.asyncio
async def test_control_tower_kpis_empty_view():
    """If view has no rows, should return zeros not 500."""
    pool, conn, cursor = _make_pool(fetchone_return=None)
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/kpis")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_control_tower_alerts_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("EXC-1", "exception", "critical", "ITEM1", "LOC1", "below_rop",
         "Below ROP by 200 units", "Order 400 units", datetime.datetime(2026, 3, 1)),
        ("DS-ITEM2-LOC2", "demand_signal", "high", "ITEM2", "LOC2", "above_plan",
         "Demand above plan by 15%", "Monitor demand pace", datetime.datetime(2026, 3, 1)),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_control_tower_alerts_sorted_by_severity():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("EXC-1", "exception", "critical", "ITEM1", "LOC1", "below_rop", "Desc1", "Act1", datetime.datetime(2026, 3, 1)),
    ])
    cursor.fetchall.side_effect = [
        [("EXC-1", "exception", "critical", "ITEM1", "LOC1", "below_rop", "Desc1", "Act1", datetime.datetime(2026, 3, 1))],
        [("DS-1", "demand_signal", "high", "ITEM2", "LOC2", "above_plan", "Desc2", "Act2", datetime.datetime(2026, 3, 1))],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/alerts?limit=20")
    assert resp.status_code == 200
    alerts = resp.json()["alerts"]
    # Critical should come before high
    if len(alerts) >= 2:
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for i in range(len(alerts) - 1):
            assert sev_order.get(alerts[i]["severity"], 99) <= sev_order.get(alerts[i+1]["severity"], 99)


@pytest.mark.asyncio
async def test_control_tower_top_critical_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("ITEM1", "LOC1", "A", "AX", 18.0, "critical", 0.12, True, 2.0, 14.0, 28.0, 3, 480.0, 0.72, 12),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/top-critical")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data


@pytest.mark.asyncio
async def test_control_tower_top_critical_worst_first():
    """Items should be ordered by health_score ascending (worst first)."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("ITEM1", "LOC1", "A", "AX", 10.0, "critical", 0.05, True, 1.0, 14.0, 28.0, 5, 500.0, 0.60, 15),
        ("ITEM2", "LOC2", "B", "BX", 35.0, "at_risk", 0.40, True, 8.0, 14.0, 28.0, 1, 100.0, 0.88, 3),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/top-critical?limit=10")
    assert resp.status_code == 200
    items = resp.json()["items"]
    scores = [i["health_score"] for i in items if i["health_score"] is not None]
    assert scores == sorted(scores)


@pytest.mark.asyncio
async def test_control_tower_trend_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("2025-10-01", 70.0, 0.92, 0.08, None, 18.0),
        ("2025-11-01", 72.0, 0.93, 0.07, None, 19.0),
        ("2025-12-01", 75.0, 0.94, 0.06, None, 20.0),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/trend?months=6")
    assert resp.status_code == 200
    data = resp.json()
    assert "trend" in data
    assert len(data["trend"]) <= 6


@pytest.mark.asyncio
async def test_control_tower_trend_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/control-tower/trend")
    assert resp.status_code == 200
    assert resp.json()["trend"] == []
