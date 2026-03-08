"""API tests for F3.5 Multi-Echelon Planning endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_NETWORK_ROW = (
    "DC001",
    "STORE001",
    1,
    "direct",
    3,
    True,
)

_TARGET_ROW = (
    "ITEM001",
    "DC001",
    1,
    85.0,
    120.0,
    10.5,
    0.95,
    datetime.datetime(2025, 3, 10, 8, 0, 0),
)

_ROP_ROW = (
    "ITEM001",
    "DC001",
    1,
    210.0,
    85.0,
    125.0,
    False,
    datetime.datetime(2025, 3, 10, 8, 0, 0),
)


@pytest.mark.asyncio
async def test_get_echelon_network_200():
    pool, conn, cursor = _make_pool(fetchall_return=[_NETWORK_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/echelon/network")

    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert len(data["nodes"]) == 1
    node = data["nodes"][0]
    assert node["parent_loc"] == "DC001"
    assert node["child_loc"] == "STORE001"
    assert node["echelon_level"] == 1
    assert node["link_type"] == "direct"
    assert node["replenishment_lead_time_days"] == 3
    assert node["is_active"] is True


@pytest.mark.asyncio
async def test_get_echelon_network_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/echelon/network")

    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []


@pytest.mark.asyncio
async def test_get_echelon_targets_200():
    pool, conn, cursor = _make_pool(fetchone_return=(2,))
    cursor.fetchall.return_value = [_TARGET_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/echelon/targets")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert "targets" in data
    assert len(data["targets"]) == 1
    target = data["targets"][0]
    assert target["item_no"] == "ITEM001"
    assert target["loc"] == "DC001"
    assert target["echelon_level"] == 1
    assert target["echelon_ss_qty"] == pytest.approx(85.0)
    assert target["computed_at"] is not None


@pytest.mark.asyncio
async def test_get_echelon_targets_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/echelon/targets")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["targets"] == []


@pytest.mark.asyncio
async def test_get_echelon_targets_with_filters():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    cursor.fetchall.return_value = [_TARGET_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/supply/echelon/targets",
                params={"item_no": "ITEM001", "loc": "DC001", "echelon_level": 1},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_get_echelon_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(500, 18.5, 2450.0, 3, datetime.datetime(2025, 3, 10, 8, 0, 0)),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/echelon/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_sku_locs"] == 500
    assert data["avg_pooling_benefit_pct"] == pytest.approx(18.5)
    assert data["total_units_saved"] == pytest.approx(2450.0)
    assert data["echelon_depth"] == 3
    assert data["last_computed_at"] is not None


@pytest.mark.asyncio
async def test_get_echelon_summary_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(None, None, None, None, None))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/echelon/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_sku_locs"] == 0
    assert data["avg_pooling_benefit_pct"] is None


@pytest.mark.asyncio
async def test_get_echelon_reorder_points_200():
    pool, conn, cursor = _make_pool(fetchone_return=(3,))
    cursor.fetchall.return_value = [_ROP_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/echelon/reorder-points")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert "reorder_points" in data
    assert len(data["reorder_points"]) == 1
    rp = data["reorder_points"][0]
    assert rp["item_no"] == "ITEM001"
    assert rp["reorder_point_qty"] == pytest.approx(210.0)
    assert rp["cascade_risk_flag"] is False
    assert rp["computed_at"] is not None


@pytest.mark.asyncio
async def test_get_echelon_reorder_points_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/echelon/reorder-points")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["reorder_points"] == []
