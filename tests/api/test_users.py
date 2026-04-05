"""Tests for user management endpoints (Spec 08-02)."""
import pytest
import uuid
import datetime


@pytest.mark.asyncio
async def test_list_users(mock_pool, async_client):
    """GET /users returns user list."""
    _, _, cursor = mock_pool
    uid = uuid.uuid4()
    now = datetime.datetime.now()
    cursor.fetchone.side_effect = [(3,)]  # count
    cursor.fetchall.return_value = [
        (uid, "admin@test.com", "Admin", "admin", True, now, now),
    ]
    resp = await async_client.get("/users")
    assert resp.status_code == 200
    data = resp.json()
    assert "users" in data


@pytest.mark.asyncio
async def test_create_user(mock_pool, async_client):
    """POST /users creates a new user."""
    _, _, cursor = mock_pool
    uid = uuid.uuid4()
    cursor.fetchone.return_value = (uid,)
    resp = await async_client.post("/users", json={
        "email": "new@test.com",
        "display_name": "New User",
        "role": "planner",
        "password": "securepass123",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "new@test.com"


@pytest.mark.asyncio
async def test_create_user_short_password(async_client):
    """POST /users rejects short passwords."""
    resp = await async_client.post("/users", json={
        "email": "short@test.com",
        "password": "short",
        "role": "viewer",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_update_user(mock_pool, async_client):
    """PUT /users/{id} updates user."""
    _, _, cursor = mock_pool
    uid = str(uuid.uuid4())
    cursor.fetchone.return_value = (uid,)
    resp = await async_client.put(f"/users/{uid}", json={"role": "manager"})
    assert resp.status_code == 200
    assert resp.json()["updated"] is True


@pytest.mark.asyncio
async def test_audit_log(mock_pool, async_client):
    """GET /users/audit-log returns entries."""
    _, _, cursor = mock_pool
    now = datetime.datetime.now()
    cursor.fetchone.side_effect = [(5,)]
    cursor.fetchall.return_value = [
        (1, uuid.uuid4(), "admin@test.com", "Admin", "login", "session", "", None, None, now),
    ]
    resp = await async_client.get("/users/audit-log")
    assert resp.status_code == 200
    assert "entries" in resp.json()
