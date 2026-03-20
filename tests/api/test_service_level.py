"""API tests for F3.2 Service Level Actuals vs. Targets endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_SL_DETAIL_ROW = (
    "ITEM001",
    "LOC001",
    datetime.date(2025, 3, 1),
    "A",
    0.94,
    0.95,
    -0.01,
    "below",
    2,
    0,
    "stockout",
    False,
)

_CHRONIC_MISS_ROW = (
    "ITEM002",
    "LOC002",
    datetime.date(2025, 3, 1),
    "B",
    0.80,
    0.92,
    -0.12,
    4,
    "forecast_error",
)


@pytest.mark.asyncio
async def test_get_sl_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            500,
            0.93,
            0.95,
            -0.02,
            45,
            12,
            datetime.date(2025, 3, 1),
        ),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analytics/service-level/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 500
    assert data["avg_fill_rate"] == pytest.approx(0.93)
    assert data["avg_target"] == pytest.approx(0.95)
    assert data["miss_count"] == 45
    assert data["flagged_count"] == 12
    assert data["latest_month"] == "2025-03-01"


@pytest.mark.asyncio
async def test_get_sl_summary_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # router checks `if not row`

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analytics/service-level/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 0
    assert data["avg_fill_rate"] is None


@pytest.mark.asyncio
async def test_get_sl_summary_with_period_filter():
    pool, conn, cursor = _make_pool(
        fetchone_return=(100, 0.91, 0.95, -0.04, 20, 5, datetime.date(2025, 2, 1)),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analytics/service-level/summary", params={"period": "2025-02-01"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["period"] == "2025-02-01"
    assert data["total_dfus"] == 100


@pytest.mark.asyncio
async def test_get_sl_detail_200():
    pool, conn, cursor = _make_pool(fetchone_return=(3,))
    cursor.fetchall.return_value = [_SL_DETAIL_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analytics/service-level/detail")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert "performance" in data
    assert len(data["performance"]) == 1
    item = data["performance"][0]
    assert item["item_no"] == "ITEM001"
    assert item["abc_class"] == "A"
    assert item["actual_fill_rate"] == pytest.approx(0.94)
    assert item["perf_month"] == "2025-03-01"


@pytest.mark.asyncio
async def test_get_sl_detail_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analytics/service-level/detail")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["performance"] == []


@pytest.mark.asyncio
async def test_get_sl_detail_with_filters():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    cursor.fetchall.return_value = [_SL_DETAIL_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/analytics/service-level/detail",
                params={"item_no": "ITEM001", "loc": "LOC001", "abc_class": "A"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_get_chronic_misses_200():
    pool, conn, cursor = _make_pool(fetchone_return=(2,))
    cursor.fetchall.return_value = [_CHRONIC_MISS_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analytics/service-level/chronic-misses")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert "chronic_misses" in data
    assert len(data["chronic_misses"]) == 1
    item = data["chronic_misses"][0]
    assert item["item_no"] == "ITEM002"
    assert item["miss_streak_months"] == 4
    assert item["primary_miss_reason"] == "forecast_error"
    assert item["perf_month"] == "2025-03-01"


@pytest.mark.asyncio
async def test_get_chronic_misses_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analytics/service-level/chronic-misses", params={"min_streak": 6})

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["chronic_misses"] == []
    assert data["min_streak"] == 6


@pytest.mark.asyncio
async def test_get_chronic_misses_with_abc_filter():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    cursor.fetchall.return_value = [_CHRONIC_MISS_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/analytics/service-level/chronic-misses",
                params={"abc_class": "B", "min_streak": 3},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_upsert_sl_target_200():
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/analytics/service-level/targets",
                json={"abc_class": "A", "target_fill_rate": 0.97},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["abc_class"] == "A"
    assert data["target_fill_rate"] == pytest.approx(0.97)
