"""API tests for F3.3 Supplier Lead Time Learning endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_PROFILE_ROW = (
    "SUP001",
    "electronics",
    "LOC001",
    14.5,
    2.3,
    14.0,
    18.0,
    0.88,
    120,
    False,
    datetime.datetime(2025, 3, 10, 6, 0, 0),
)

_ALERT_ROW = (
    1,
    "SUP001",
    "mean_lt_increase",
    12.0,
    18.0,
    1.5,
    2.8,
    45,
    "open",
    datetime.datetime(2025, 3, 12, 9, 0, 0),
)


@pytest.mark.asyncio
async def test_get_supplier_lead_times_200():
    pool, conn, cursor = _make_pool(fetchone_return=(5,))
    cursor.fetchall.return_value = [_PROFILE_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/supplier-lead-times")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert "profiles" in data
    assert len(data["profiles"]) == 1
    item = data["profiles"][0]
    assert item["supplier_id"] == "SUP001"
    assert item["mean_lt_days"] == pytest.approx(14.5)
    assert item["on_time_delivery_rate"] == pytest.approx(0.88)
    assert item["updated_at"] is not None


@pytest.mark.asyncio
async def test_get_supplier_lead_times_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/supplier-lead-times")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["profiles"] == []


@pytest.mark.asyncio
async def test_get_supplier_lead_times_with_filter():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    cursor.fetchall.return_value = [_PROFILE_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/supply/supplier-lead-times",
                params={"supplier_id": "SUP001", "item_category": "electronics"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_get_supplier_lt_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(25, 0.84, 15.2, 4, 3),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/supplier-lead-times/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["supplier_count"] == 25
    assert data["avg_otdr"] == pytest.approx(0.84)
    assert data["avg_mean_lt_days"] == pytest.approx(15.2)
    assert data["poor_suppliers"] == 4
    assert data["flagged_suppliers"] == 3


@pytest.mark.asyncio
async def test_get_supplier_lt_summary_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # router checks `if not row`

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/supplier-lead-times/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["supplier_count"] == 0


@pytest.mark.asyncio
async def test_get_lead_time_alerts_200():
    pool, conn, cursor = _make_pool(fetchone_return=(2,))
    cursor.fetchall.return_value = [_ALERT_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/lead-time-alerts")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert "alerts" in data
    assert len(data["alerts"]) == 1
    alert = data["alerts"][0]
    assert alert["id"] == 1
    assert alert["supplier_id"] == "SUP001"
    assert alert["trigger_type"] == "mean_lt_increase"
    assert alert["review_status"] == "open"
    assert alert["triggered_at"] is not None


@pytest.mark.asyncio
async def test_get_lead_time_alerts_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/lead-time-alerts", params={"review_status": "acknowledged"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["alerts"] == []
    assert data["review_status"] == "acknowledged"


@pytest.mark.asyncio
async def test_acknowledge_lt_trigger_200():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/lead-time-review/1/acknowledge",
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 1
    assert data["review_status"] == "acknowledged"


@pytest.mark.asyncio
async def test_acknowledge_lt_trigger_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # RETURNING returns None when no row matched

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/lead-time-review/9999/acknowledge",
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 404
