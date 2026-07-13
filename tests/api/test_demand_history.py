"""Tests for demand history workbench endpoints -- /demand-history/*."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_CFG = {
    "default_months": 24,
    "max_months": 60,
    "pareto_top_n": 5,
    "matrix_max_rows": 100,
    "matrix_max_cols": 50,
    "workbench_default_limit": 50,
    "cache_ttl_seconds": 120,
    "hierarchical_model_ids": ["nbeats"],
    "top_down_model_ids": ["chronos2_enriched"],
}


def _patch_planning_date(d=date(2026, 3, 1)):
    return patch(
        "api.routers.inventory.demand_history.get_planning_date",
        return_value=d,
    )


def _patch_config():
    return patch(
        "api.routers.inventory.demand_history.load_config",
        return_value={"demand_history": _MOCK_CFG},
    )


def _reset_cfg():
    """Reset the module-level config cache."""
    import api.routers.inventory.demand_history as mod
    mod._CFG = None


# Feature 1: /demand-history/reference


@pytest.mark.asyncio
async def test_reference_returns_summary():
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [
        [("2026-01", 100.0, 90.0), ("2026-02", 120.0, 110.0)],
        [("C001", "Acme Corp", 150.0), ("C002", "Beta LLC", 70.0)],
    ]
    cursor.fetchone.return_value = (500.0, 7.0)
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/reference?item_id=ITEM1&loc=LOC1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "ITEM1"
    assert len(data["history"]) == 2
    assert data["total_demand"] == 220.0
    assert data["trend_mom_pct"] == 20.0
    assert data["inventory"]["qty_on_hand"] == 500.0


@pytest.mark.asyncio
async def test_reference_no_data():
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [[], []]
    cursor.fetchone.return_value = None
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/reference?item_id=X&loc=Y")
    assert resp.status_code == 200
    data = resp.json()
    assert data["history"] == []
    assert data["total_demand"] == 0
    assert data["inventory"] is None


@pytest.mark.asyncio
async def test_reference_missing_params():
    pool, _, cursor = _make_pool()
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/reference")
    assert resp.status_code == 422


# Feature 2: /demand-history/decomposition


@pytest.mark.asyncio
async def test_decomposition_returns_series_and_pareto():
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [
        # series
        [("2026-01", "C001", "Acme", 60.0), ("2026-01", "C002", "Beta", 40.0)],
        # pareto
        [("C001", "Acme", 60.0), ("C002", "Beta", 40.0)],
    ]
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/decomposition?item_id=I1&loc=L1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["series"]) == 2
    assert data["series"][0]["pct_share"] == 60.0
    assert len(data["pareto"]) == 2
    assert data["pareto"][0]["cumulative_pct"] == 60.0
    assert data["pareto"][1]["cumulative_pct"] == 100.0


@pytest.mark.asyncio
async def test_decomposition_empty():
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [[], []]
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/decomposition?item_id=X&loc=Y")
    assert resp.status_code == 200
    data = resp.json()
    assert data["series"] == []
    assert data["pareto"] == []


# Feature 3: /demand-history/comparison


@pytest.mark.asyncio
async def test_comparison_actuals_only():
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [
        [("2026-01", 100.0), ("2026-02", 120.0)],
        [],  # no predictions
    ]
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/comparison?item_id=I1&loc=L1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["comparison"]) == 2
    assert data["comparison"][0]["actual_qty"] == 100.0
    assert data["comparison"][0]["bottom_up_qty"] is None
    assert data["comparison"][0]["top_down_qty"] is None
    assert data["comparison"][0]["reconciled_qty"] is None


@pytest.mark.asyncio
async def test_comparison_with_predictions():
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [
        [("2026-01", 100.0)],
        [("2026-01", "nbeats", 95.0), ("2026-01", "chronos2_enriched", 105.0)],
    ]
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/comparison?item_id=I1&loc=L1")
    assert resp.status_code == 200
    data = resp.json()
    c = data["comparison"][0]
    assert c["actual_qty"] == 100.0
    assert c["bottom_up_qty"] == 95.0
    assert c["top_down_qty"] == 105.0
    assert c["reconciled_qty"] == 100.0


# Feature 4: /demand-history/workbench


@pytest.mark.asyncio
async def test_workbench_item_grain():
    pool, _, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.side_effect = [
        [("I1", "Widget A", 500.0), ("I2", "Widget B", 300.0)],
        [("I1", "2026-01", 250.0), ("I1", "2026-02", 250.0), ("I2", "2026-01", 300.0)],
    ]
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/workbench?grain=item")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grain"] == "item"
    assert data["total"] == 2
    assert len(data["series"]) == 2
    assert data["series"][0]["key"] == "I1"
    assert data["series"][0]["total_demand"] == 500.0
    assert len(data["series"][0]["months"]) == 2
    assert data["hierarchy_children"] == "item_loc"

    executed = [call.args for call in cursor.execute.call_args_list]
    assert all("fact_sales_monthly" in sql for sql, _params in executed)
    assert all("f.startdate <= %s::date" in sql for sql, _params in executed)
    assert executed[0][1][:2] == ["2024-03-01", "2026-02-01"]


@pytest.mark.asyncio
async def test_workbench_customer_grain_keeps_customer_demand_source():
    pool, _, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(date(2026, 7, 11)),
        _patch_config(),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/demand-history/workbench?grain=item_loc_customer&item_id=I1&loc=L1"
            )

    assert resp.status_code == 200
    executed = [call.args for call in cursor.execute.call_args_list]
    assert all("fact_customer_demand_monthly" in sql for sql, _params in executed)
    assert "dc.site = f.site" in executed[1][0]
    assert executed[0][1][:2] == ["2024-07-01", "2026-06-01"]


@pytest.mark.asyncio
async def test_workbench_empty():
    pool, _, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/workbench")
    assert resp.status_code == 200
    data = resp.json()
    assert data["series"] == []
    assert data["total"] == 0


# Feature 5: /demand-history/matrix


@pytest.mark.asyncio
async def test_matrix_returns_grid():
    pool, _, cursor = _make_pool()
    cursor.fetchall.side_effect = [
        [("I1", "Widget A"), ("I2", "Widget B")],
        [("L1", "NYC"), ("L2", "LA")],
        [("I1", "L1", 100.0), ("I1", "L2", 50.0), ("I2", "L1", 80.0)],
    ]
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/matrix?row_dim=item&col_dim=location")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == ["I1", "I2"]
    assert data["cols"] == ["L1", "L2"]
    assert data["cells"][0][0] == 100.0
    assert data["cells"][0][1] == 50.0
    assert data["cells"][1][0] == 80.0
    assert data["cells"][1][1] is None
    assert data["row_labels"]["I1"] == "Widget A"


@pytest.mark.asyncio
async def test_matrix_same_dim_rejected():
    pool, _, cursor = _make_pool()
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/matrix?row_dim=item&col_dim=item")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_matrix_drill_returns_history():
    pool, _, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("2026-01", 100.0, 90.0),
        ("2026-02", 110.0, 100.0),
    ]
    _reset_cfg()
    with (
        patch("api.core._get_pool", return_value=pool),
        _patch_planning_date(),
        _patch_config(),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/demand-history/matrix/drill?item_id=I1&loc=L1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["history"]) == 2
    assert data["history"][0]["demand_qty"] == 100.0
