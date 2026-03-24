"""API tests for F4.4 Supply Chain Disruption Scenario Planning endpoints."""
import pytest
import datetime
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool

pytest_plugins = ["anyio"]

_SCENARIO_ROW = (
    1,
    "Port Strike Impact",
    "logistics_disruption",
    "West Coast port strike scenario",
    3,
    "draft",
    "api",
    datetime.datetime(2025, 3, 1, 9, 0, 0),
    None,
)

_SCENARIO_DETAIL_ROW = (
    1,
    "Port Strike Impact",
    "logistics_disruption",
    "West Coast port strike scenario",
    "{}",
    "[]",
    "[]",
    "[]",
    3,
    "running",
    "api",
    datetime.datetime(2025, 3, 1, 9, 0, 0),
    datetime.datetime(2025, 3, 2, 10, 0, 0),
)

_RESULT_ROW = (
    "ITEM001",
    "LOC001",
    datetime.date(2025, 4, 1),
    1000.0,
    600.0,
    -400.0,
    -40.0,
    15.0,
    200.0,
    "expedite_air_freight",
)


@pytest.mark.asyncio
async def test_list_supply_scenarios_200():
    pool, conn, cursor = _make_pool(fetchone_return=(3,))
    cursor.fetchall.return_value = [_SCENARIO_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/scenarios/supply")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert "scenarios" in data
    assert len(data["scenarios"]) == 1
    sc = data["scenarios"][0]
    assert sc["scenario_id"] == 1
    assert sc["scenario_name"] == "Port Strike Impact"
    assert sc["scenario_type"] == "logistics_disruption"
    assert sc["status"] == "draft"
    assert sc["horizon_months"] == 3
    assert sc["created_at"] is not None


@pytest.mark.asyncio
async def test_list_supply_scenarios_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/scenarios/supply")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["scenarios"] == []


@pytest.mark.asyncio
async def test_list_supply_scenarios_with_filters():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))
    cursor.fetchall.return_value = [_SCENARIO_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/scenarios/supply",
                params={"scenario_type": "logistics_disruption", "status": "draft"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_create_supply_scenario_201():
    pool, conn, cursor = _make_pool(fetchone_return=(7,))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/scenarios/supply",
                json={
                    "scenario_name": "Demand Spike Q2",
                    "scenario_type": "demand_shock",
                    "description": "20% demand surge in Q2",
                    "shock_parameters": {"uplift_pct": 20},
                    "horizon_months": 3,
                },
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["scenario_id"] == 7
    assert data["status"] == "draft"


@pytest.mark.asyncio
async def test_get_supply_scenario_200():
    pool, conn, cursor = _make_pool(fetchone_return=_SCENARIO_DETAIL_ROW)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/scenarios/supply/1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_id"] == 1
    assert data["scenario_name"] == "Port Strike Impact"
    assert data["scenario_type"] == "logistics_disruption"
    assert data["status"] == "running"
    assert data["created_at"] is not None
    assert data["last_run_at"] is not None


@pytest.mark.asyncio
async def test_get_supply_scenario_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # router checks `if not row: raise 404`

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/scenarios/supply/9999")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_run_supply_scenario_200():
    pool, conn, cursor = _make_pool(fetchone_return=(1,))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/scenarios/supply/1/run",
                json={"run_by": "planner@co.com", "baseline_plan_version": "latest"},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_id"] == 1
    assert data["status"] == "running"
    assert "message" in data


@pytest.mark.asyncio
async def test_run_supply_scenario_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # SELECT returns None when scenario not found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/scenarios/supply/9999/run",
                json={"run_by": "planner@co.com"},
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_scenario_results_200():
    pool, conn, cursor = _make_pool(fetchone_return=(5,))
    cursor.fetchall.return_value = [_RESULT_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/scenarios/supply/1/results")

    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_id"] == 1
    assert data["total"] == 5
    assert "results" in data
    assert len(data["results"]) == 1
    res = data["results"][0]
    assert res["item_id"] == "ITEM001"
    assert res["baseline_qty"] == pytest.approx(1000.0)
    assert res["scenario_qty"] == pytest.approx(600.0)
    assert res["impact_qty"] == pytest.approx(-400.0)
    assert res["impact_pct"] == pytest.approx(-40.0)
    assert res["plan_month"] == "2025-04-01"
    assert res["mitigation_option"] == "expedite_air_freight"


@pytest.mark.asyncio
async def test_get_scenario_results_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,))
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/scenarios/supply/1/results")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["results"] == []
