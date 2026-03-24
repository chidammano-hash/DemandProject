"""API tests for F3.4 Demand Sensing / Blended Forecast endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_WEEKLY_ROW = (
    datetime.date(2025, 3, 3),
    0.45,
    320.0,
    280.0,
    302.0,
    1.08,
    False,
    "v2025-03",
)

_SENSING_ACTIVE_ROW = (
    "ITEM001",
    "LOC001",
    datetime.date(2025, 3, 3),
    0.72,
    450.0,
    380.0,
    420.0,
    1.18,
)


@pytest.mark.asyncio
async def test_get_blended_forecast_200():
    pool, conn, cursor = _make_pool(fetchall_return=[_WEEKLY_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/blended",
                params={"item_id": "ITEM001", "loc": "LOC001"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "ITEM001"
    assert data["loc"] == "LOC001"
    assert "weekly_forecast" in data
    assert len(data["weekly_forecast"]) == 1
    week = data["weekly_forecast"][0]
    assert week["week_start"] == "2025-03-03"
    assert week["alpha_weight"] == pytest.approx(0.45)
    assert week["blended_qty"] == pytest.approx(302.0)
    assert "monthly_total_blended" in data


@pytest.mark.asyncio
async def test_get_blended_forecast_missing_params():
    pool, conn, cursor = _make_pool(fetchall_return=[])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/blended")

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_blended_forecast_empty_weeks():
    pool, conn, cursor = _make_pool(fetchall_return=[])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/blended",
                params={"item_id": "ITEM_NODATA", "loc": "LOC_NODATA"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["weekly_forecast"] == []
    assert data["monthly_total_blended"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_get_blended_forecast_with_plan_version():
    pool, conn, cursor = _make_pool(fetchall_return=[_WEEKLY_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/blended",
                params={"item_id": "ITEM001", "loc": "LOC001", "plan_version": "v2025-03", "weeks": 4},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_version"] == "v2025-03"
    assert data["weeks"] == 4


@pytest.mark.asyncio
async def test_get_blended_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(1500, 320, 1.12, 45),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/blended/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 1500
    assert data["sensing_active_count"] == 320
    assert data["avg_spike_ratio"] == pytest.approx(1.12)
    assert data["outlier_capped_count"] == 45


@pytest.mark.asyncio
async def test_get_blended_summary_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # router checks `if not row`

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/blended/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 0


@pytest.mark.asyncio
async def test_get_sensing_active_200():
    pool, conn, cursor = _make_pool(fetchone_return=(8,))
    cursor.fetchall.return_value = [_SENSING_ACTIVE_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/sensing-active")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 8
    assert "active_overrides" in data
    assert len(data["active_overrides"]) == 1
    item = data["active_overrides"][0]
    assert item["item_id"] == "ITEM001"
    assert item["alpha_weight"] == pytest.approx(0.72)
    assert item["week_start"] == "2025-03-03"


@pytest.mark.asyncio
async def test_get_sensing_active_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/sensing-active")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["active_overrides"] == []
