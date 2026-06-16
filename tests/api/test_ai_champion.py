"""Tests for the AI Champion forward adjuster endpoints (/ai-champion/*)."""
import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_latest_returns_run_and_rollup():
    """GET /ai-champion/latest returns the latest run + recommendation rollup."""
    started = datetime.datetime(2026, 4, 2, 12, 0, 0, tzinfo=datetime.UTC)
    completed = datetime.datetime(2026, 4, 2, 12, 5, 0, tzinfo=datetime.UTC)
    run_row = ("run-1", "2026-04", "ollama", "llama3.1:8b", "succeeded",
               7612, 1804, 0.0, started, completed)
    pool, _conn, cursor = _make_pool(fetchone_return=run_row)
    cursor.fetchall.return_value = [("KEEP", 5808), ("SCALE_DOWN", 1200), ("SCALE_UP", 604)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-champion/latest")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run"]["provider"] == "ollama"
    assert data["run"]["ai_model"] == "llama3.1:8b"
    assert data["run"]["n_adjusted"] == 1804
    assert data["run"]["status"] == "succeeded"
    assert {r["recommendation_code"] for r in data["by_recommendation"]} == {"KEEP", "SCALE_DOWN", "SCALE_UP"}


@pytest.mark.asyncio
async def test_latest_empty_when_no_runs():
    """GET /ai-champion/latest returns run=None when no run exists."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # no run rows
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-champion/latest")
    assert resp.status_code == 200
    assert resp.json() == {"run": None, "by_recommendation": []}


@pytest.mark.asyncio
async def test_forecast_returns_rows():
    """GET /ai-champion/forecast returns champion-vs-ai rows for the latest run."""
    pool, _conn, cursor = _make_pool(fetchone_return=(2,))
    cursor.fetchall.return_value = [
        ("100", "L1", datetime.date(2026, 4, 1), 1, 100.0, 110.0, "SCALE_UP", 10.0, 0.8, "uptrend"),
        ("100", "L1", datetime.date(2026, 5, 1), 2, 95.0, 95.0, "SCALE_UP", 10.0, 0.8, "uptrend"),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/ai-champion/forecast?item_id=100&adjusted_only=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["rows"][0]["champion_qty"] == 100.0
    assert data["rows"][0]["ai_qty"] == 110.0
    assert data["rows"][0]["recommendation_code"] == "SCALE_UP"


@pytest.mark.asyncio
async def test_generate_submits_job():
    """POST /ai-champion/generate submits a generate_ai_champion job."""
    pool, _conn, _cursor = _make_pool()
    mock_mgr = MagicMock()
    mock_mgr.submit_job.return_value = "job-ac-1"
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", return_value=mock_mgr),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/ai-champion/generate", json={"provider": "anthropic", "limit_dfus": 50})
    assert resp.status_code == 202
    assert resp.json() == {"job_id": "job-ac-1", "status": "queued"}
    mock_mgr.submit_job.assert_called_once_with(
        "generate_ai_champion",
        {"provider": "anthropic", "limit_dfus": 50},
        label="Generate AI Champion Forecast",
        triggered_by="api",
    )
