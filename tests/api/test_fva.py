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
    """GET /fva/waterfall returns model accuracy waterfall."""
    rows = [
        ("ceiling", 85.1, 1000),
        ("champion", 78.3, 1000),
        ("external", 72.5, 1000),
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
    assert wf["external"]["model_id"] == "external"
    assert wf["external"]["accuracy_pct"] == 72.5
    assert wf["champion"]["accuracy_pct"] == 78.3
    assert wf["ceiling"]["accuracy_pct"] == 85.1
    assert len(wf["models"]) == 3


@pytest.mark.asyncio
async def test_fva_waterfall_empty():
    """GET /fva/waterfall returns empty waterfall when no data."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall")
    assert resp.status_code == 200
    data = resp.json()
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
