"""Tests for Forecast Value Added (FVA) endpoints (Spec 08-07)."""
import datetime
import pytest
from unittest.mock import patch

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Tests — /fva/waterfall
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fva_waterfall():
    """GET /fva/waterfall returns staged FVA ladder plus benchmark."""
    rows = [
        ("seasonal_naive", 60.2, 1000),
        ("external", 72.5, 1000),
        ("champion", 78.3, 1000),
        ("ceiling", 85.1, 1000),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall")
    assert resp.status_code == 200
    data = resp.json()
    assert data["months"] == 12
    wf = data["waterfall"]
    stages = wf["stages"]
    assert [stage["stage_id"] for stage in stages] == [
        "seasonal_naive",
        "external",
        "champion",
        "ai_adjusted",
        "planner_adjusted",
    ]
    assert stages[0]["label"] == "Naive Seasonal"
    assert stages[0]["accuracy_pct"] == 60.2
    assert stages[0]["delta_vs_prev"] is None
    assert stages[1]["accuracy_pct"] == 72.5
    assert stages[1]["delta_vs_prev"] == 12.3
    assert stages[2]["accuracy_pct"] == 78.3
    assert stages[2]["delta_vs_prev"] == 5.8
    assert stages[3]["state"] == "planned"
    assert stages[3]["accuracy_pct"] is None
    assert stages[4]["state"] == "planned"
    assert stages[4]["accuracy_pct"] is None
    assert wf["benchmark"]["stage_id"] == "ceiling"
    assert wf["benchmark"]["accuracy_pct"] == 85.1
    assert wf["external"]["model_id"] == "external"
    assert wf["external"]["accuracy_pct"] == 72.5
    assert wf["champion"]["accuracy_pct"] == 78.3
    assert wf["ceiling"]["accuracy_pct"] == 85.1
    assert len(wf["models"]) == 4
    executed_sql = cursor.execute.call_args_list[0].args[0]
    assert "current_date - (%s * interval '1 month')" in executed_sql


@pytest.mark.asyncio
async def test_fva_waterfall_empty():
    """GET /fva/waterfall returns placeholder stages when no data is available."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall")
    assert resp.status_code == 200
    data = resp.json()
    stages = data["waterfall"]["stages"]
    assert [stage["state"] for stage in stages[:3]] == ["missing", "missing", "missing"]
    assert [stage["state"] for stage in stages[3:]] == ["planned", "planned"]
    assert data["waterfall"]["benchmark"]["state"] == "missing"
    assert data["waterfall"]["external"] is None
    assert data["waterfall"]["champion"] is None
    assert data["waterfall"]["models"] == []


@pytest.mark.asyncio
async def test_fva_waterfall_custom_months():
    """GET /fva/waterfall accepts months param."""
    rows = [("external", 70.0, 500)]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall", params={"months": 6})
    assert resp.status_code == 200
    assert resp.json()["months"] == 6
    executed_sql = cursor.execute.call_args_list[0].args[0]
    assert "current_date - (%s * interval '1 month')" in executed_sql


# ---------------------------------------------------------------------------
# Tests — /fva/interventions
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fva_interventions():
    """GET /fva/interventions returns intervention list with total count."""
    now = datetime.datetime(2025, 3, 1, 12, 0, 0)
    # The endpoint does fetchone (count) then fetchall (rows) on same cursor
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5,)
    cursor.fetchall.return_value = [
        (1, None, "policy_change", "sku", "100320-1401", None, None,
         5000.0, None, None, None, "pending", now),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/interventions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["interventions"]) == 1
    inv = data["interventions"][0]
    assert inv["intervention_id"] == 1
    assert inv["intervention_type"] == "policy_change"
    assert inv["resource_type"] == "sku"
    assert inv["financial_impact_estimate"] == 5000.0
    assert inv["status"] == "pending"


@pytest.mark.asyncio
async def test_fva_interventions_empty():
    """GET /fva/interventions returns empty when no interventions."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/interventions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["interventions"] == []


# ---------------------------------------------------------------------------
# Tests — /fva/roi-summary
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fva_roi_summary():
    """GET /fva/roi-summary returns aggregate ROI metrics."""
    pool, conn, cursor = _make_pool(fetchone_return=(10, 4, 6, 50000.0, 20000.0))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/roi-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["months"] == 12
    assert data["total_interventions"] == 10
    assert data["measured"] == 4
    assert data["pending"] == 6
    assert data["total_estimated_impact"] == 50000.0
    assert data["total_actual_impact"] == 20000.0
    executed_sql = cursor.execute.call_args_list[0].args[0]
    assert "current_date - (%s * interval '1 month')" in executed_sql


@pytest.mark.asyncio
async def test_fva_roi_summary_zeros():
    """GET /fva/roi-summary handles zero counts."""
    pool, conn, cursor = _make_pool(fetchone_return=(0, 0, 0, 0, 0))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/roi-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_interventions"] == 0
    assert data["total_estimated_impact"] == 0
    assert data["total_actual_impact"] == 0
