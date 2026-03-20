"""API tests for notification endpoints (Spec 08-04).

Tests all 3 notification REST endpoints using httpx AsyncClient with
ASGITransport -- no running server needed.
"""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


_NOW = datetime.datetime(2026, 3, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# GET /notifications/history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_history_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "job_complete", "info", "admin@test.com", "Job Done",
         "delivered", None, _NOW, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/notifications/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "notifications" in data
    assert len(data["notifications"]) == 1
    n = data["notifications"][0]
    assert n["notification_id"] == 1
    assert n["event_type"] == "job_complete"
    assert n["severity"] == "info"
    assert n["recipient"] == "admin@test.com"
    assert n["subject"] == "Job Done"
    assert n["status"] == "delivered"
    assert n["error"] is None
    assert n["created_at"] is not None
    assert n["delivered_at"] is not None


@pytest.mark.asyncio
async def test_notification_history_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/notifications/history")
    assert resp.status_code == 200
    assert resp.json()["notifications"] == []


@pytest.mark.asyncio
async def test_notification_history_with_filters():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/notifications/history?event_type=job_complete&status=delivered&limit=10"
            )
    assert resp.status_code == 200
    assert resp.json()["notifications"] == []
    # Verify filters were passed as params
    call_args = cursor.execute.call_args
    assert "job_complete" in call_args[0][1]
    assert "delivered" in call_args[0][1]


@pytest.mark.asyncio
async def test_notification_history_with_error():
    """Notification with delivery error should include error field."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        (2, "alert", "critical", "user@test.com", "Stockout Alert",
         "failed", "SMTP timeout", _NOW, None),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/notifications/history")
    assert resp.status_code == 200
    n = resp.json()["notifications"][0]
    assert n["status"] == "failed"
    assert n["error"] == "SMTP timeout"
    assert n["delivered_at"] is None


# ---------------------------------------------------------------------------
# GET /notifications/channels
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_channels_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "slack", True, _NOW),
        (2, "email", True, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/notifications/channels")
    assert resp.status_code == 200
    channels = resp.json()["channels"]
    assert len(channels) == 2
    assert channels[0]["channel_type"] == "slack"
    assert channels[0]["enabled"] is True
    assert channels[1]["channel_type"] == "email"


@pytest.mark.asyncio
async def test_notification_channels_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/notifications/channels")
    assert resp.status_code == 200
    assert resp.json()["channels"] == []


# ---------------------------------------------------------------------------
# POST /notifications/test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_test_200():
    """POST /notifications/test sends a test notification (manager+ role)."""
    pool, conn, cursor = _make_pool()
    mock_engine = MagicMock()
    mock_engine.send.return_value = {"status": "delivered", "channel": "email"}
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {
             "common.notification_engine": MagicMock(
                 NotificationEngine=lambda: mock_engine
             )
         }):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/notifications/test", json={
                "channel": "email",
                "recipient": "test@test.com",
                "message": "hello",
            })
    assert resp.status_code == 200
    assert "results" in resp.json()
    mock_engine.send.assert_called_once()


@pytest.mark.asyncio
async def test_notification_test_default_body():
    """POST /notifications/test with empty body uses default message."""
    pool, conn, cursor = _make_pool()
    mock_engine = MagicMock()
    mock_engine.send.return_value = {"status": "delivered"}
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {
             "common.notification_engine": MagicMock(
                 NotificationEngine=lambda: mock_engine
             )
         }):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/notifications/test", json={})
    assert resp.status_code == 200
