"""Tests for auth endpoints (Spec 08-02)."""
import pytest
import uuid


@pytest.mark.asyncio
async def test_login_success(mock_pool, async_client):
    """POST /auth/login with valid credentials returns tokens."""
    from common.auth import hash_password
    _, _, cursor = mock_pool
    pw_hash = hash_password("testpass123")
    uid = str(uuid.uuid4())
    cursor.fetchone.return_value = (uid, "test@example.com", "Test User", "planner", pw_hash, True)
    resp = await async_client.post("/auth/login", json={"email": "test@example.com", "password": "testpass123"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == "test@example.com"


@pytest.mark.asyncio
async def test_login_invalid_password(mock_pool, async_client):
    """POST /auth/login with wrong password returns 401."""
    from common.auth import hash_password
    _, _, cursor = mock_pool
    pw_hash = hash_password("correctpass")
    uid = str(uuid.uuid4())
    cursor.fetchone.return_value = (uid, "test@example.com", "Test", "viewer", pw_hash, True)
    resp = await async_client.post("/auth/login", json={"email": "test@example.com", "password": "wrongpass"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_user_not_found(mock_pool, async_client):
    """POST /auth/login with nonexistent user returns 401."""
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = None
    resp = await async_client.post("/auth/login", json={"email": "none@example.com", "password": "test"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_me_anonymous(async_client):
    """GET /auth/me returns anonymous user in dev mode."""
    resp = await async_client.get("/auth/me")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "anonymous"


@pytest.mark.asyncio
async def test_logout(async_client):
    """POST /auth/logout returns success."""
    resp = await async_client.post("/auth/logout")
    assert resp.status_code == 200
