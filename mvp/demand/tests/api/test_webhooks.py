"""Tests for Webhook management endpoints (Spec 08-10)."""
import datetime
import pytest
from unittest.mock import patch

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


def _viewer_user():
    """Return a CurrentUser with viewer role (insufficient for manager endpoints)."""
    from common.auth import CurrentUser
    return CurrentUser(user_id="viewer-1", email="viewer@test.com", role="viewer")


# ---------------------------------------------------------------------------
# Tests -- role enforcement (all webhook endpoints require manager)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_register_webhook_requires_manager():
    """POST /webhooks/register returns 403 for viewer role."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from common.auth import get_current_user
        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/webhooks/register",
                    json={"url": "https://example.com/hook", "event_types": ["alert"]},
                )
        finally:
            app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_webhooks_requires_manager():
    """GET /webhooks returns 403 for viewer role."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from common.auth import get_current_user
        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/webhooks")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_webhook_requires_manager():
    """DELETE /webhooks/{id} returns 403 for viewer role."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from common.auth import get_current_user
        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/webhooks/1")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_webhook_deliveries_requires_manager():
    """GET /webhooks/deliveries returns 403 for viewer role."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from common.auth import get_current_user
        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/webhooks/deliveries")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Tests -- happy paths (admin in dev mode)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_register_webhook_success():
    """POST /webhooks/register succeeds for admin."""
    pool, conn, cursor = _make_pool(fetchone_return=(42,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/webhooks/register",
                json={"url": "https://example.com/hook", "event_types": ["alert"]},
            )
    assert resp.status_code == 201
    data = resp.json()
    assert data["webhook_id"] == 42
    assert "secret" in data
    assert len(data["secret"]) == 64  # hex of 32 bytes


@pytest.mark.asyncio
async def test_list_webhooks_success():
    """GET /webhooks returns webhook list for admin."""
    now = datetime.datetime(2025, 3, 1, 12, 0, 0)
    rows = [
        (1, "https://example.com/hook", '["alert"]', True, now),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/webhooks")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["webhooks"]) == 1
    wh = data["webhooks"][0]
    assert wh["webhook_id"] == 1
    assert wh["url"] == "https://example.com/hook"
    assert wh["is_active"] is True


@pytest.mark.asyncio
async def test_delete_webhook_success():
    """DELETE /webhooks/{id} deactivates webhook for admin."""
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/webhooks/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["webhook_id"] == 1
    assert data["deactivated"] is True


@pytest.mark.asyncio
async def test_delete_webhook_not_found():
    """DELETE /webhooks/{id} returns 404 for nonexistent webhook."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/webhooks/999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_webhook_deliveries_success():
    """GET /webhooks/deliveries returns delivery history for admin."""
    now = datetime.datetime(2025, 3, 1, 12, 0, 0)
    rows = [
        (1, 1, "https://example.com/hook", "alert", 200, 1, "delivered", now, now),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/webhooks/deliveries")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["deliveries"]) == 1
    d = data["deliveries"][0]
    assert d["delivery_id"] == 1
    assert d["webhook_id"] == 1
    assert d["status_code"] == 200
    assert d["status"] == "delivered"


@pytest.mark.asyncio
async def test_webhook_deliveries_empty():
    """GET /webhooks/deliveries returns empty list."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/webhooks/deliveries")
    assert resp.status_code == 200
    assert resp.json()["deliveries"] == []
