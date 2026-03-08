"""API tests for F4.2 Sales & Operations Planning (S&OP) endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_CYCLE_ROW = (
    1,
    datetime.date(2025, 3, 1),
    "demand_review",
    "alice@co.com",
    None,
    None,
    datetime.datetime(2025, 3, 1, 9, 0, 0),
    datetime.datetime(2025, 3, 5, 14, 0, 0),
)

_APPROVED_PLAN_ROW = (
    1,
    "ITEM001",
    "LOC001",
    datetime.date(2025, 4, 1),
    1500.0,
    1400.0,
    100.0,
    "statistical",
    True,
)


@pytest.mark.asyncio
async def test_list_sop_cycles_200():
    pool, conn, cursor = _make_pool(fetchone_return=(2,))
    cursor.fetchall.return_value = [_CYCLE_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sop/cycles")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert "cycles" in data
    assert len(data["cycles"]) == 1
    cycle = data["cycles"][0]
    assert cycle["cycle_id"] == 1
    assert cycle["current_stage"] == "demand_review"
    assert cycle["cycle_month"] == "2025-03-01"
    assert cycle["created_at"] is not None


@pytest.mark.asyncio
async def test_list_sop_cycles_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sop/cycles")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["cycles"] == []


@pytest.mark.asyncio
async def test_get_sop_cycle_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _CYCLE_ROW
    cursor.fetchall.side_effect = [
        [("food", 5000.0, 5200.0, 5100.0, "approved")],
        [("capacity", "SUP001", 200.0, "2025-04", "open")],
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sop/cycles/1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["cycle_id"] == 1
    assert data["current_stage"] == "demand_review"
    assert "demand_review" in data
    assert "supply_constraints" in data
    assert len(data["demand_review"]) == 1
    assert data["demand_review"][0]["item_category"] == "food"


@pytest.mark.asyncio
async def test_get_sop_cycle_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sop/cycles/9999")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_advance_sop_cycle_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("demand_review",)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sop/cycles/1/advance",
                json={"facilitated_by": "alice@co.com", "notes": "Demand review complete"},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["cycle_id"] == 1
    assert data["previous_status"] == "demand_review"
    assert data["new_status"] == "supply_review"


@pytest.mark.asyncio
async def test_advance_sop_cycle_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sop/cycles/9999/advance",
                json={"facilitated_by": "alice@co.com"},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_advance_sop_cycle_already_closed():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("closed",)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sop/cycles/1/advance",
                json={"facilitated_by": "alice@co.com"},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_approved_plan_200():
    pool, conn, cursor = _make_pool(fetchone_return=(5,))
    cursor.fetchall.return_value = [_APPROVED_PLAN_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sop/approved-plan")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert "approved_plan" in data
    assert len(data["approved_plan"]) == 1
    row = data["approved_plan"][0]
    assert row["item_no"] == "ITEM001"
    assert row["approved_qty"] == pytest.approx(1500.0)
    assert row["plan_month"] == "2025-04-01"
    assert row["locked"] is True


@pytest.mark.asyncio
async def test_get_approved_plan_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sop/approved-plan", params={"cycle_id": 1})

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["approved_plan"] == []


@pytest.mark.asyncio
async def test_approve_sop_cycle_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("executive_sop",)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sop/cycles/1/approve",
                json={"approved_by": "cfo@co.com", "plan_version": "v2025-03"},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["cycle_id"] == 1
    assert data["status"] == "approved"
    assert data["approved_by"] == "cfo@co.com"
    assert data["plan_version"] == "v2025-03"
