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
        ("external", 72.5, 1000),
        ("champion", 78.3, 1000),
        ("ceiling", 85.1, 1000),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    cursor.fetchone.return_value = (60.2, 1000)  # seasonal naive row
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
    assert len(wf["models"]) == 4  # external, champion, ceiling, seasonal_naive
    executed_sql = cursor.execute.call_args_list[0].args[0]
    assert "%s::date - (%s * interval '1 month')" in executed_sql
    assert "current_date" not in executed_sql


@pytest.mark.asyncio
async def test_fva_waterfall_windows_on_planning_date_not_wallclock():
    """F9.1: the waterfall horizon must anchor to the planning date, NOT the
    DB wall-clock ``current_date``. The demo forecast horizon ends ~2026-02
    while the system clock is months ahead, so a ``current_date``-anchored
    3-month window matches zero rows and blanks the ladder. The bound planning
    date keeps the window aligned with the data.
    """
    rows = [("external", 71.4, 500)]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    cursor.fetchone.return_value = (64.2, 500)
    planning = datetime.date(2026, 4, 2)
    with patch("api.core._get_pool", return_value=pool), patch(
        "api.routers.forecasting.fva.get_planning_date", return_value=planning
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall", params={"months": 3})
    assert resp.status_code == 200
    executed_sql = cursor.execute.call_args_list[0].args[0]
    # No wall-clock anchor.
    assert "current_date" not in executed_sql
    # The planning date is bound as a parameter, not interpolated.
    params = cursor.execute.call_args_list[0].args[1]
    assert planning in params
    assert 3 in params


@pytest.mark.asyncio
async def test_fva_waterfall_empty():
    """GET /fva/waterfall returns placeholder stages when no data is available."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    cursor.fetchone.return_value = (None, 0)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall")
    assert resp.status_code == 200
    data = resp.json()
    stages = data["waterfall"]["stages"]
    # F2.2: champion (index 2) degrades to "planned" (reserved), not "missing".
    assert [stage["state"] for stage in stages[:2]] == ["missing", "missing"]
    assert [stage["state"] for stage in stages[2:]] == ["planned", "planned", "planned"]
    assert data["waterfall"]["benchmark"]["state"] == "missing"
    assert data["waterfall"]["external"] is None
    assert data["waterfall"]["champion"] is None
    assert data["waterfall"]["models"] == []


@pytest.mark.asyncio
async def test_fva_waterfall_champion_missing_renders_as_reserved_not_broken():
    """F2.2: when no champion accuracy is measurable in the external forecast,
    the champion rung must read as a reserved/planned stage (consistent with the
    AI Adjusted / Planner Adjusted stages) rather than ``state="missing"`` which
    the UI renders as a broken-looking "No data".

    The champion forecast lives in fact_production_forecast and has no measurable
    overlap with actuals in the window, so the external-forecast query (which only
    ever contains model_id='external') yields no champion row. The ladder must then
    present champion, ai_adjusted and planner_adjusted as three consistent reserved
    stages, not one that reads as a failure.
    """
    rows = [("external", 71.2, 1000)]  # no champion row from the external table
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    cursor.fetchone.return_value = (65.3, 1000)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall")
    assert resp.status_code == 200
    stages = resp.json()["waterfall"]["stages"]
    champion = next(s for s in stages if s["stage_id"] == "champion")
    # Reserved treatment — identical to AI/Planner, NOT the broken "missing".
    assert champion["state"] == "planned"
    assert champion["accuracy_pct"] is None
    # All three forward stages now read consistently as reserved.
    assert [s["state"] for s in stages if s["stage_id"] in
            ("champion", "ai_adjusted", "planner_adjusted")] == ["planned", "planned", "planned"]


@pytest.mark.asyncio
async def test_fva_waterfall_champion_actual_when_measured():
    """F2.2 guard: when the external forecast DOES carry a champion row (measured),
    the champion rung must still surface as ``actual`` with its accuracy — the
    reserved fallback only applies when champion is genuinely unmeasured.
    """
    rows = [("external", 72.5, 1000), ("champion", 78.3, 1000)]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    cursor.fetchone.return_value = (60.2, 1000)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall")
    assert resp.status_code == 200
    stages = resp.json()["waterfall"]["stages"]
    champion = next(s for s in stages if s["stage_id"] == "champion")
    assert champion["state"] == "actual"
    assert champion["accuracy_pct"] == 78.3


@pytest.mark.asyncio
async def test_fva_waterfall_custom_months():
    """GET /fva/waterfall accepts months param."""
    rows = [("external", 70.0, 500)]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    cursor.fetchone.return_value = (None, 0)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall", params={"months": 6})
    assert resp.status_code == 200
    assert resp.json()["months"] == 6
    executed_sql = cursor.execute.call_args_list[0].args[0]
    assert "%s::date - (%s * interval '1 month')" in executed_sql
    assert "current_date" not in executed_sql


@pytest.mark.asyncio
async def test_fva_waterfall_ai_adjusted_reserved():
    """The ai_adjusted and planner_adjusted stages are always reserved ("planned").

    The AI FVA backtest was removed; the forward-only AI Champion forecast has
    no historical actual overlap, so it cannot feed this accuracy waterfall.
    """
    rows = [("external", 72.5, 1000), ("champion", 78.3, 1000)]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    cursor.fetchone.return_value = (60.2, 1000)  # naive; champion falls back to rollup
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall")
    assert resp.status_code == 200
    stages = resp.json()["waterfall"]["stages"]
    ai_stage = next(s for s in stages if s["stage_id"] == "ai_adjusted")
    assert ai_stage["state"] == "planned"
    assert ai_stage["accuracy_pct"] is None
    assert "ai_fva_run_id" not in ai_stage
    # No SQL in the endpoint may reference the removed backtest tables.
    all_sql = " ".join(str(c.args[0]) for c in cursor.execute.call_args_list)
    assert "ai_fva" not in all_sql
    assert "mv_ai_fva" not in all_sql


@pytest.mark.asyncio
async def test_fva_waterfall_champion_from_backtest_experiment():
    """Champion accuracy is sourced from the promoted champion-selection
    experiment's backtest, month-windowed via ``champion_experiment_month``
    (champ, ceiling, n). It must surface champion as ``actual`` AND populate the
    ceiling benchmark, with delta_vs_prev computed against external.
    """
    # external from the rollup; no champion/ceiling there (production reality).
    rows = [("external", 71.5, 111146)]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    # fetchone order after the AI-backtest removal: naive, then champion experiment.
    cursor.fetchone.side_effect = [
        (64.7, 111146),         # naive
        (71.62, 75.63, 20517),  # champion_experiment_month aggregate: champ, ceiling, n
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/fva/waterfall")
    assert resp.status_code == 200
    wf = resp.json()["waterfall"]
    champion = next(s for s in wf["stages"] if s["stage_id"] == "champion")
    assert champion["state"] == "actual"
    assert champion["accuracy_pct"] == 71.62
    assert champion["n_rows"] == 20517
    assert champion["delta_vs_prev"] == 0.1  # 71.62 - 71.5, rounded to 1dp
    # Ceiling benchmark populated from the same row.
    assert wf["benchmark"]["state"] == "actual"
    assert wf["benchmark"]["accuracy_pct"] == 75.63
    # Champion is the 3rd execute (rollup, naive, champion) — month-windowed source.
    champ_sql = cursor.execute.call_args_list[2].args[0]
    assert "champion_experiment_month" in champ_sql
    assert "is_promoted = TRUE" in champ_sql
    assert "month_start" in champ_sql


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
    assert "%s::date - (%s * interval '1 month')" in executed_sql
    assert "current_date" not in executed_sql


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
