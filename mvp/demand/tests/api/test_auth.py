"""Tests for auth endpoints (Spec 08-02)."""
import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_login_success():
    """POST /auth/login with valid credentials returns tokens."""
    from common.auth import hash_password
    pw_hash = hash_password("testpass123")
    import uuid
    uid = str(uuid.uuid4())

    pool, conn, cursor = _make_pool(
        fetchone_return=(uid, "test@example.com", "Test User", "planner", pw_hash, True),
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/auth/login", json={"email": "test@example.com", "password": "testpass123"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == "test@example.com"


@pytest.mark.asyncio
async def test_login_invalid_password():
    """POST /auth/login with wrong password returns 401."""
    from common.auth import hash_password
    pw_hash = hash_password("correctpass")
    import uuid
    uid = str(uuid.uuid4())

    pool, conn, cursor = _make_pool(
        fetchone_return=(uid, "test@example.com", "Test", "viewer", pw_hash, True),
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/auth/login", json={"email": "test@example.com", "password": "wrongpass"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_user_not_found():
    """POST /auth/login with nonexistent user returns 401."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/auth/login", json={"email": "none@example.com", "password": "test"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_me_anonymous():
    """GET /auth/me returns anonymous user in dev mode."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/auth/me")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "anonymous"


@pytest.mark.asyncio
async def test_logout():
    """POST /auth/logout returns success."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/auth/logout")
    assert resp.status_code == 200
