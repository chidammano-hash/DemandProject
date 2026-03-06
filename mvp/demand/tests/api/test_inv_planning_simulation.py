"""API tests for IPfeature10 simulation endpoints."""
import pytest
import datetime
from unittest.mock import MagicMock, patch
from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_simulation_results_200():
    # 17 columns: sim_run_id, item_no, loc, simulation_date, n_simulations,
    # demand_distribution, demand_mean, demand_std, lt_distribution,
    # lt_mean_days, lt_std_days, results_by_ss_level, target_csl,
    # recommended_ss, recommended_ss_days, analytical_ss, sim_vs_analytical_pct
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            "sim-uuid-1", "ITEM1", "LOC1", datetime.date(2026, 3, 1),
            10000, "empirical", 250.0, 45.0, "empirical", 14.0, 2.0,
            None,   # results_by_ss_level JSONB
            0.95, 120.0, 5.0, 110.0, 9.1,
        )
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/simulation/results?item=ITEM1&location=LOC1")
    assert resp.status_code == 200
    data = resp.json()
    assert "item_no" in data
    assert "sim_run_id" in data


@pytest.mark.asyncio
async def test_simulation_results_404():
    pool, conn, cursor = _make_pool(fetchone_return=None)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/simulation/results?item=MISSING&location=LOC1")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_simulation_compare_200():
    # compare: 5 cols — recommended_ss, analytical_ss, sim_vs_analytical_pct,
    #                    results_by_ss_level, target_csl
    # two fetchone calls: first the sim result, then EOM inventory
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (120.0, 110.0, 9.1, None, 0.95),   # sim result row
        (80.0,),                             # eom_qty_on_hand
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/simulation/compare?item=ITEM1&location=LOC1")
    assert resp.status_code == 200
    data = resp.json()
    assert "analytical_ss" in data
    assert "simulated_ss" in data


@pytest.mark.asyncio
async def test_simulation_compare_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/simulation/compare?item=MISSING&location=L1")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_simulation_status_found():
    pool, conn, cursor = _make_pool(
        fetchone_return=("ITEM1", "LOC1", datetime.date(2026, 3, 1), datetime.datetime(2026, 3, 1))
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/simulation/sim-uuid-1/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_simulation_status_not_found():
    pool, conn, cursor = _make_pool(fetchone_return=None)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/simulation/missing-id/status")
    assert resp.status_code == 404
