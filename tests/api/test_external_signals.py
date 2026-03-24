"""Tests for External Demand Signals endpoints (Spec 08-06)."""
import datetime
import pytest
from unittest.mock import patch

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


def _viewer_user():
    from common.auth import CurrentUser
    return CurrentUser(user_id="viewer-1", email="viewer@test.com", role="viewer")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_external_signals():
    """GET /demand-signals/external returns signals list."""
    now = datetime.datetime(2025, 3, 1, 12, 0, 0)
    rows = [
        (1, "NOAA Weather", datetime.date(2025, 3, 1), "100320", "1401-BULK",
         "weather", 0.85, 0.9, now),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-signals/external")
    assert resp.status_code == 200
    data = resp.json()
    assert "signals" in data
    assert len(data["signals"]) == 1
    sig = data["signals"][0]
    assert sig["signal_id"] == 1
    assert sig["source_name"] == "NOAA Weather"
    assert sig["item_id"] == "100320"
    assert sig["signal_type"] == "weather"
    assert sig["confidence"] == 0.9


@pytest.mark.asyncio
async def test_list_external_signals_empty():
    """GET /demand-signals/external returns empty list when no data."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-signals/external")
    assert resp.status_code == 200
    assert resp.json()["signals"] == []


@pytest.mark.asyncio
async def test_list_signal_sources():
    """GET /demand-signals/external/sources returns sources."""
    now = datetime.datetime(2025, 3, 1, 12, 0, 0)
    rows = [
        (1, "NOAA Weather", "weather", True, 24, now),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-signals/external/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sources"]) == 1
    src = data["sources"][0]
    assert src["source_id"] == 1
    assert src["name"] == "NOAA Weather"
    assert src["source_type"] == "weather"
    assert src["enabled"] is True
    assert src["refresh_interval_hours"] == 24


@pytest.mark.asyncio
async def test_refresh_source_requires_manager():
    """POST /demand-signals/external/sources/{id}/refresh returns 403 for viewer role."""
    pool, conn, cursor = _make_pool(fetchone_return=(1, "NOAA", "weather"))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from common.auth import get_current_user
        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/demand-signals/external/sources/1/refresh")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_refresh_source_success_as_admin():
    """POST /demand-signals/external/sources/{id}/refresh succeeds for admin."""
    pool, conn, cursor = _make_pool(fetchone_return=(1, "NOAA Weather", "weather"))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/demand-signals/external/sources/1/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_id"] == 1
    assert data["status"] == "refresh_queued"


@pytest.mark.asyncio
async def test_demand_decomposition():
    """GET /demand-signals/external/decomposition returns decomposition data."""
    rows = [
        (datetime.date(2025, 1, 1), 100.0, 5.0, 10.0, 2.0, 1.5, 0.5),
        (datetime.date(2025, 2, 1), 105.0, 5.5, 11.0, 3.0, 2.0, 0.3),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/demand-signals/external/decomposition",
                params={"item_id": "100320", "loc": "1401-BULK"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "100320"
    assert data["loc"] == "1401-BULK"
    assert len(data["decomposition"]) == 2
    d = data["decomposition"][0]
    assert d["base_demand"] == 100.0
    assert d["trend"] == 5.0
    assert d["seasonal"] == 10.0


@pytest.mark.asyncio
async def test_demand_decomposition_not_found():
    """GET /demand-signals/external/decomposition returns 404 when no data."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/demand-signals/external/decomposition",
                params={"item_id": "NONE", "loc": "NONE"},
            )
    assert resp.status_code == 404
