"""API tests for collaboration & annotations endpoints (Spec 08-05).

Tests all 8 collaboration REST endpoints using httpx AsyncClient with
ASGITransport -- no running server needed.
"""
from __future__ import annotations

import datetime
import json
import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


_NOW = datetime.datetime(2026, 3, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# POST /collaboration/annotations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_annotation_201():
    pool, conn, cursor = _make_pool(fetchone_return=(42,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/collaboration/annotations", json={
                "resource_type": "insight",
                "resource_id": "123",
                "body": "Good insight, I agree with the recommendation.",
            })
    assert resp.status_code == 201
    assert resp.json()["annotation_id"] == 42


@pytest.mark.asyncio
async def test_create_annotation_with_mentions():
    """Annotations with @mentions should trigger notifications (best-effort)."""
    pool, conn, cursor = _make_pool(fetchone_return=(43,))
    mock_engine = MagicMock()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {
             "common.notification_engine": MagicMock(
                 NotificationEngine=lambda: mock_engine
             )
         }):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/collaboration/annotations", json={
                "resource_type": "insight",
                "resource_id": "123",
                "body": "Hey @alice check this out",
                "mentions": ["alice@test.com"],
            })
    assert resp.status_code == 201
    assert resp.json()["annotation_id"] == 43


@pytest.mark.asyncio
async def test_create_annotation_with_parent():
    """Reply annotations should include parent_id."""
    pool, conn, cursor = _make_pool(fetchone_return=(44,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/collaboration/annotations", json={
                "resource_type": "insight",
                "resource_id": "123",
                "body": "I agree with the above.",
                "parent_id": 42,
            })
    assert resp.status_code == 201
    # Verify parent_id was passed to SQL
    call_args = cursor.execute.call_args
    assert 42 in call_args[0][1]


# ---------------------------------------------------------------------------
# GET /collaboration/annotations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_annotations_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, None, "anon@test.com", "Anonymous", None, "Test comment",
         "[]", False, _NOW, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/collaboration/annotations?resource_type=insight&resource_id=42"
            )
    assert resp.status_code == 200
    annotations = resp.json()["annotations"]
    assert len(annotations) == 1
    a = annotations[0]
    assert a["annotation_id"] == 1
    assert a["body"] == "Test comment"
    assert a["is_resolved"] is False


@pytest.mark.asyncio
async def test_list_annotations_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/collaboration/annotations?resource_type=insight&resource_id=999"
            )
    assert resp.status_code == 200
    assert resp.json()["annotations"] == []


@pytest.mark.asyncio
async def test_list_annotations_missing_params():
    """resource_type and resource_id are required query params."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/collaboration/annotations")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# PUT /collaboration/annotations/{annotation_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_annotation_200():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/collaboration/annotations/1?body=Updated+comment"
            )
    assert resp.status_code == 200
    assert resp.json()["updated"] is True


@pytest.mark.asyncio
async def test_update_annotation_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/collaboration/annotations/999?body=Updated+comment"
            )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /collaboration/annotations/{annotation_id}/resolve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_annotation_200():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/collaboration/annotations/1/resolve")
    assert resp.status_code == 200
    assert resp.json()["resolved"] is True


@pytest.mark.asyncio
async def test_resolve_annotation_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/collaboration/annotations/999/resolve")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /collaboration/annotations/{annotation_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_annotation_200():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/collaboration/annotations/1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


@pytest.mark.asyncio
async def test_delete_annotation_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/collaboration/annotations/999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /collaboration/mentions/me
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_my_mentions_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "insight", "42", "Check this out", False, _NOW, "Admin User"),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/collaboration/mentions/me")
    assert resp.status_code == 200
    mentions = resp.json()["mentions"]
    assert len(mentions) == 1
    assert mentions[0]["annotation_id"] == 1
    assert mentions[0]["resource_type"] == "insight"
    assert mentions[0]["body"] == "Check this out"
    assert mentions[0]["author"] == "Admin User"


@pytest.mark.asyncio
async def test_my_mentions_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/collaboration/mentions/me")
    assert resp.status_code == 200
    assert resp.json()["mentions"] == []


# ---------------------------------------------------------------------------
# POST /collaboration/shared-views
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_shared_view_201():
    pool, conn, cursor = _make_pool(fetchone_return=(10,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/collaboration/shared-views", json={
                "title": "My Dashboard View",
                "tab": "dashboard",
                "filters": {"brand": "BrandX"},
                "is_public": True,
            })
    assert resp.status_code == 201
    assert resp.json()["view_id"] == 10


@pytest.mark.asyncio
async def test_create_shared_view_minimal():
    """Minimal shared view with only required fields."""
    pool, conn, cursor = _make_pool(fetchone_return=(11,))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/collaboration/shared-views", json={
                "title": "Quick View",
                "tab": "accuracy",
            })
    assert resp.status_code == 201
    assert resp.json()["view_id"] == 11


# ---------------------------------------------------------------------------
# GET /collaboration/shared-views
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_shared_views_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, None, "My View", "dashboard", {"brand": "X"}, {}, True, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/collaboration/shared-views")
    assert resp.status_code == 200
    views = resp.json()["views"]
    assert len(views) == 1
    assert views[0]["title"] == "My View"
    assert views[0]["is_public"] is True


@pytest.mark.asyncio
async def test_list_shared_views_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/collaboration/shared-views")
    assert resp.status_code == 200
    assert resp.json()["views"] == []


# ---------------------------------------------------------------------------
# GET /collaboration/shared-views/{view_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_shared_view_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(1, None, "My View", "dashboard", {}, {}, True, _NOW)
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/collaboration/shared-views/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["view_id"] == 1
    assert data["title"] == "My View"
    assert data["is_public"] is True


@pytest.mark.asyncio
async def test_get_shared_view_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/collaboration/shared-views/999")
    assert resp.status_code == 404
