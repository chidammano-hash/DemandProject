"""Tests for user management endpoints (Spec 08-02)."""
import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool
import uuid
import datetime


@pytest.mark.asyncio
async def test_list_users():
    """GET /users returns user list."""
    uid = uuid.uuid4()
    now = datetime.datetime.now()
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(3,)]  # count
    cursor.fetchall.return_value = [
        (uid, "admin@test.com", "Admin", "admin", True, now, now),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/users")
    assert resp.status_code == 200
    data = resp.json()
    assert "users" in data


@pytest.mark.asyncio
async def test_create_user():
    """POST /users creates a new user."""
    uid = uuid.uuid4()
    pool, conn, cursor = _make_pool(fetchone_return=(uid,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/users", json={
                "email": "new@test.com",
                "display_name": "New User",
                "role": "planner",
                "password": "securepass123",
            })
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "new@test.com"


@pytest.mark.asyncio
async def test_create_user_short_password():
    """POST /users rejects short passwords."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/users", json={
                "email": "short@test.com",
                "password": "short",
                "role": "viewer",
            })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_update_user():
    """PUT /users/{id} updates user."""
    uid = str(uuid.uuid4())
    pool, conn, cursor = _make_pool(fetchone_return=(uid,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(f"/users/{uid}", json={"role": "manager"})
    assert resp.status_code == 200
    assert resp.json()["updated"] is True


@pytest.mark.asyncio
async def test_audit_log():
    """GET /users/audit-log returns entries."""
    import datetime
    now = datetime.datetime.now()
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(5,)]
    cursor.fetchall.return_value = [
        (1, uuid.uuid4(), "admin@test.com", "Admin", "login", "session", "", None, None, now),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/users/audit-log")
    assert resp.status_code == 200
    assert "entries" in resp.json()
