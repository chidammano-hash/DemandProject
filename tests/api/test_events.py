"""API tests for F4.3 Promotion & Event Planning endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_EVENT_ROW = (
    1,
    "promotion",
    "Spring Sale",
    datetime.date(2025, 4, 1),
    datetime.date(2025, 4, 15),
    0.20,
    2,
    0.0,
    8,
    "approved",
    None,
    datetime.datetime(2025, 3, 1, 10, 0, 0),
)

_EVENT_DETAIL_ROW = (
    1,
    "promotion",
    "Spring Sale",
    datetime.date(2025, 4, 1),
    datetime.date(2025, 4, 15),
    0.20,
    2,
    0.0,
    0,
    8,
    "approved",
    None,
    "[]",
    "[]",
    "[]",
    datetime.datetime(2025, 3, 1, 10, 0, 0),
)

_IMPACT_ROW = (
    "ITEM001",
    "LOC001",
    datetime.date(2025, 4, 1),
    1,
    1000.0,
    200.0,
    -50.0,
    1150.0,
    "uplift",
    datetime.date(2025, 3, 15),
)


@pytest.mark.asyncio
async def test_get_event_calendar_200():
    pool, conn, cursor = _make_pool(fetchone_return=(2,))
    cursor.fetchall.return_value = [_EVENT_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/events/calendar")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert "events" in data
    assert len(data["events"]) == 1
    event = data["events"][0]
    assert event["event_id"] == 1
    assert event["event_type"] == "promotion"
    assert event["event_name"] == "Spring Sale"
    assert event["status"] == "approved"
    assert event["event_start"] == "2025-04-01"
    assert event["event_end"] == "2025-04-15"


@pytest.mark.asyncio
async def test_get_event_calendar_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/events/calendar")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["events"] == []


@pytest.mark.asyncio
async def test_get_event_calendar_with_filters():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    cursor.fetchall.return_value = [_EVENT_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/events/calendar",
                params={"event_type": "promotion", "status": "approved"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_create_event_201():
    pool, conn, cursor = _make_pool(fetchone_return=(42,))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/events/calendar",
                json={
                    "event_type": "promotion",
                    "event_name": "Summer Sale",
                    "event_start": "2025-06-01",
                    "event_end": "2025-06-15",
                    "uplift_pct": 0.15,
                    "ramp_weeks": 1,
                    "status": "draft",
                },
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["event_id"] == 42
    assert data["status"] == "draft"


@pytest.mark.asyncio
async def test_get_event_detail_200():
    pool, conn, cursor = _make_pool(fetchone_return=_EVENT_DETAIL_ROW)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/events/calendar/1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["event_id"] == 1
    assert data["event_type"] == "promotion"
    assert data["event_name"] == "Spring Sale"
    assert data["status"] == "approved"
    assert data["event_start"] == "2025-04-01"
    assert data["created_at"] is not None


@pytest.mark.asyncio
async def test_get_event_detail_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # router checks `if not row: raise 404`

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/events/calendar/9999")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_event_impact_preview_200():
    pool, conn, cursor = _make_pool(fetchall_return=[_IMPACT_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/events/impact-preview", params={"event_id": 1})

    assert resp.status_code == 200
    data = resp.json()
    assert "adjustments" in data
    assert len(data["adjustments"]) == 1
    adj = data["adjustments"][0]
    assert adj["item_id"] == "ITEM001"
    assert adj["base_forecast_qty"] == pytest.approx(1000.0)
    assert adj["event_adjustment_qty"] == pytest.approx(200.0)
    assert adj["adjusted_forecast_qty"] == pytest.approx(1150.0)
    assert adj["plan_month"] == "2025-04-01"


@pytest.mark.asyncio
async def test_get_event_impact_preview_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/events/impact-preview", params={"event_id": 99})

    assert resp.status_code == 200
    data = resp.json()
    assert data["adjustments"] == []


@pytest.mark.asyncio
async def test_approve_event_200():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/events/calendar/1/approve",
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["event_id"] == 1
    assert data["status"] == "approved"


@pytest.mark.asyncio
async def test_approve_event_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # RETURNING returns None when event not found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/events/calendar/9999/approve",
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 404
