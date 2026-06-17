"""Tests for the interactive AI Champion adjuster endpoints (/ai-champion/*)."""
import datetime
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from common.ai.champion_adjust_service import NoChampionForecast, UnknownProvider
from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_forecast_returns_saved_rows_for_dfu():
    """GET /ai-champion/forecast returns the saved adjustment for a DFU."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("100", "L1", datetime.date(2026, 5, 1), 1, 100.0, 110.0, "SCALE_UP", 10.0, 0.8, "uptrend"),
        ("100", "L1", datetime.date(2026, 6, 1), 2, 95.0, 95.0, "SCALE_UP", 0.0, 0.8, "uptrend"),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-champion/forecast", params={"item_id": "100", "loc": "L1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["rows"][0]["champion_qty"] == 100.0
    assert data["rows"][0]["ai_qty"] == 110.0
    assert data["rows"][0]["recommendation_code"] == "SCALE_UP"


@pytest.mark.asyncio
async def test_adjust_returns_preview():
    """POST /ai-champion/adjust returns the LLM preview (no DB write)."""
    preview = {
        "item_id": "100", "loc": "L1", "plan_version": "2026-04",
        "provider": "ollama", "model": "llama3.1:8b", "prompt_version": "v1.1.0",
        "recommendation_code": "SCALE_UP", "rec_pct_change": 15.0, "proposed_qty": None,
        "apply_horizon_months": 3, "confidence": 0.82, "rationale": "uptrend",
        "evidence_keys": ["trend_break"], "months": [],
    }

    class _Preview:
        def to_dict(self):
            return preview

    with patch("api.routers.forecasting.ai_champion.adjust_dfu", return_value=_Preview()) as m:
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/ai-champion/adjust", json={"item_id": "100", "loc": "L1", "provider": "ollama"})
    assert resp.status_code == 200
    assert resp.json()["recommendation_code"] == "SCALE_UP"
    m.assert_called_once_with("100", "L1", provider="ollama")


@pytest.mark.asyncio
async def test_adjust_404_when_no_champion():
    """POST /ai-champion/adjust maps NoChampionForecast to 404."""
    with patch("api.routers.forecasting.ai_champion.adjust_dfu", side_effect=NoChampionForecast("none")):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/ai-champion/adjust", json={"item_id": "X", "loc": "Y"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_adjust_400_on_unknown_provider():
    """POST /ai-champion/adjust maps UnknownProvider to 400."""
    with patch("api.routers.forecasting.ai_champion.adjust_dfu", side_effect=UnknownProvider("nope")):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/ai-champion/adjust", json={"item_id": "X", "loc": "Y", "provider": "nope"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_save_persists_adjustment():
    """POST /ai-champion/save persists and returns the run summary."""
    result = {"item_id": "100", "loc": "L1", "plan_version": "2026-04",
              "run_id": "r1", "recommendation_code": "SCALE_UP", "saved_months": 3}
    with patch("api.routers.forecasting.ai_champion.save_adjustment", return_value=result) as m:
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/ai-champion/save", json={
                "item_id": "100", "loc": "L1", "provider": "ollama",
                "recommendation": {
                    "recommendation_code": "SCALE_UP", "pct_change": 15.0,
                    "apply_horizon_months": 3, "confidence": 0.82,
                    "rationale": "uptrend", "evidence_keys": ["trend_break"],
                },
            })
    assert resp.status_code == 200
    assert resp.json()["saved_months"] == 3
    m.assert_called_once()
