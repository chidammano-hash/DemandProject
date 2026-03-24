"""API tests for F4.1 Financial Inventory Plan endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_INVENTORY_PLAN_SUMMARY_ROW = (
    1200,
    4500000.0,
    800000.0,
    112500.0,
    250000.0,
    3,
    datetime.date(2025, 3, 1),
)

_BUDGET_ROW = (
    1,
    "global",
    "all",
    datetime.date(2025, 1, 1),
    datetime.date(2025, 12, 31),
    1000000.0,
    750000.0,
    250000.0,
    0.75,
)

_EXCESS_ROW = (
    "ITEM001",
    "LOC001",
    datetime.date(2025, 3, 1),
    500.0,
    25000.0,
    120000.0,
    50000.0,
    True,
)


@pytest.mark.asyncio
async def test_get_inventory_plan_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _INVENTORY_PLAN_SUMMARY_ROW
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/finance/inventory-plan")

    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "by_category" in data
    assert data["summary"]["sku_loc_count"] == 1200
    assert data["summary"]["total_projected_value"] == pytest.approx(4500000.0)
    assert data["summary"]["budget_breach_count"] == 3
    assert data["summary"]["latest_plan_month"] == "2025-03-01"


@pytest.mark.asyncio
async def test_get_inventory_plan_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/finance/inventory-plan")

    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] == {}
    assert data["by_category"] == []


@pytest.mark.asyncio
async def test_get_inventory_plan_with_category_breakdown():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _INVENTORY_PLAN_SUMMARY_ROW
    cursor.fetchall.return_value = [
        ("electronics", datetime.date(2025, 3, 1), 2000000.0, 400000.0, 100000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/finance/inventory-plan", params={"plan_version": "v2025-03"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_version"] == "v2025-03"
    assert len(data["by_category"]) == 1
    cat = data["by_category"][0]
    assert cat["item_category"] == "electronics"
    assert cat["plan_month"] == "2025-03-01"


@pytest.mark.asyncio
async def test_get_budget_status_200():
    pool, conn, cursor = _make_pool(fetchall_return=[_BUDGET_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/finance/budget-status")

    assert resp.status_code == 200
    data = resp.json()
    assert "budgets" in data
    assert len(data["budgets"]) == 1
    b = data["budgets"][0]
    assert b["budget_id"] == 1
    assert b["scope_type"] == "global"
    assert b["budget_cap"] == pytest.approx(1000000.0)
    assert b["budget_start"] == "2025-01-01"
    assert b["budget_end"] == "2025-12-31"


@pytest.mark.asyncio
async def test_get_budget_status_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/finance/budget-status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["budgets"] == []


@pytest.mark.asyncio
async def test_get_excess_value_200():
    pool, conn, cursor = _make_pool(fetchall_return=[_EXCESS_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/finance/excess-value")

    assert resp.status_code == 200
    data = resp.json()
    assert "excess_items" in data
    assert len(data["excess_items"]) == 1
    item = data["excess_items"][0]
    assert item["item_id"] == "ITEM001"
    assert item["excess_qty"] == pytest.approx(500.0)
    assert item["excess_value"] == pytest.approx(25000.0)
    assert item["plan_month"] == "2025-03-01"


@pytest.mark.asyncio
async def test_get_excess_value_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/finance/excess-value", params={"min_excess_value": 100000.0})

    assert resp.status_code == 200
    data = resp.json()
    assert data["excess_items"] == []


@pytest.mark.asyncio
async def test_create_budget_201():
    pool, conn, cursor = _make_pool(fetchone_return=(42,))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/finance/budget",
                json={
                    "scope_type": "global",
                    "scope_value": "all",
                    "period_type": "monthly",
                    "budget_start": "2025-01-01",
                    "budget_end": "2025-12-31",
                    "budget_cap": 2000000.0,
                    "carrying_cost_pct": 0.25,
                },
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["budget_id"] == 42
    assert data["status"] == "created"


@pytest.mark.asyncio
async def test_update_budget_200():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/finance/budget/1",
                json={"budget_cap": 1500000.0},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["budget_id"] == 1
    assert data["budget_cap"] == pytest.approx(1500000.0)
    assert data["status"] == "updated"


@pytest.mark.asyncio
async def test_update_budget_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # RETURNING returns None when budget not found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/finance/budget/9999",
                json={"budget_cap": 500000.0},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 404
