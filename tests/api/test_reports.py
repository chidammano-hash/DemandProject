"""Tests for Reports & Distribution endpoints (Spec 08-08)."""
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
# Tests — /reports/templates
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_templates():
    """GET /reports/templates returns template list."""
    now = datetime.datetime(2025, 3, 1, 12, 0, 0)
    rows = [
        (1, "Accuracy Summary", "accuracy", True, now),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/reports/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["templates"]) == 1
    t = data["templates"][0]
    assert t["template_id"] == 1
    assert t["name"] == "Accuracy Summary"
    assert t["report_type"] == "accuracy"
    assert t["is_system"] is True


@pytest.mark.asyncio
async def test_list_templates_empty():
    """GET /reports/templates returns empty list when no templates."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/reports/templates")
    assert resp.status_code == 200
    assert resp.json()["templates"] == []


# ---------------------------------------------------------------------------
# Tests — /reports/generate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_report_requires_planner():
    """POST /reports/generate returns 403 for viewer role."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from common.auth import get_current_user
        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/reports/generate", params={"template_id": 1})
        finally:
            app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_generate_report_success():
    """POST /reports/generate succeeds for admin (dev mode)."""
    pool, conn, cursor = _make_pool(
        fetchone_return=(1, "Accuracy Summary", "accuracy", "{}"),
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/reports/generate", params={"template_id": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert data["template_id"] == 1
    assert data["format"] == "pdf"


@pytest.mark.asyncio
async def test_generate_report_template_not_found():
    """POST /reports/generate returns 404 when template missing."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/reports/generate", params={"template_id": 999})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests — /reports/schedules
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_schedules():
    """GET /reports/schedules returns schedule list."""
    now = datetime.datetime(2025, 3, 1, 12, 0, 0)
    rows = [
        (1, "Accuracy Summary", "accuracy", '["admin@test.com"]',
         "0 8 * * 1", "pdf", True, now, now),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/reports/schedules")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["schedules"]) == 1
    s = data["schedules"][0]
    assert s["schedule_id"] == 1
    assert s["template_name"] == "Accuracy Summary"
    assert s["cron"] == "0 8 * * 1"
    assert s["format"] == "pdf"
    assert s["enabled"] is True


@pytest.mark.asyncio
async def test_create_schedule_requires_manager():
    """POST /reports/schedules returns 403 for viewer role."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from common.auth import get_current_user
        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/reports/schedules",
                    json={"template_id": 1, "recipients": ["a@b.com"], "cron": "0 8 * * 1", "format": "pdf"},
                )
        finally:
            app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_schedule_requires_manager():
    """DELETE /reports/schedules/{id} returns 403 for viewer role."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from common.auth import get_current_user
        app.dependency_overrides[get_current_user] = _viewer_user
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/reports/schedules/1")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Tests — /reports/deliveries
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_deliveries():
    """GET /reports/deliveries returns delivery history."""
    now = datetime.datetime(2025, 3, 1, 12, 0, 0)
    rows = [
        (1, "Accuracy Summary", "delivered", "/reports/1.pdf", None, now, now),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/reports/deliveries")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["deliveries"]) == 1
    d = data["deliveries"][0]
    assert d["delivery_id"] == 1
    assert d["template_name"] == "Accuracy Summary"
    assert d["status"] == "delivered"
    assert d["file_path"] == "/reports/1.pdf"
    assert d["error"] is None


@pytest.mark.asyncio
async def test_list_deliveries_empty():
    """GET /reports/deliveries returns empty list."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/reports/deliveries")
    assert resp.status_code == 200
    assert resp.json()["deliveries"] == []
