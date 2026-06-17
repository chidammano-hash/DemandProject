"""Tests for backtest management API — /backtest-management/* endpoints.

Tests summary listing, model runs, current metadata, run submission,
and load submission.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROUTER_MOD = "api.routers.forecasting.backtest_management"

_NOW = datetime(2026, 4, 6, 10, 0, 0, tzinfo=timezone.utc)
_NOW_ISO = _NOW.isoformat()


def _run_row(
    run_id: int = 1,
    model_id: str = "lgbm_cluster",
    job_id: str | None = "job-bt-100",
    status: str = "completed",
    accuracy_pct: float | None = 72.5,
    wape: float | None = 0.275,
    bias: float | None = -0.02,
    n_predictions: int = 50000,
    n_dfus: int = 5000,
    n_timeframes: int = 5,
    metadata: dict | None = None,
    is_loaded_to_db: bool = True,
    loaded_at=None,
    load_job_id: str | None = None,
    started_at=None,
    completed_at=None,
    created_at=None,
) -> tuple:
    """Build a mock backtest_run row tuple (17 columns)."""
    return (
        run_id,
        model_id,
        job_id,
        status,
        accuracy_pct,
        wape,
        bias,
        n_predictions,
        n_dfus,
        n_timeframes,
        metadata or {},
        is_loaded_to_db,
        loaded_at or _NOW,
        load_job_id,
        started_at or _NOW,
        completed_at or _NOW,
        created_at or _NOW,
    )


def _mock_roster():
    """Return a minimal algorithm roster dict for testing."""
    return {
        "lgbm_cluster": {"type": "tree", "enabled": True},
        "catboost_cluster": {"type": "tree", "enabled": True},
        "chronos_bolt": {"type": "foundation", "enabled": True},
    }


# ---------------------------------------------------------------------------
# 1. GET /backtest-management/summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_summary_returns_all_models():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        _run_row(run_id=1, model_id="lgbm_cluster", accuracy_pct=72.5),
        _run_row(run_id=2, model_id="catboost_cluster", accuracy_pct=70.0),
    ]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch(f"{_ROUTER_MOD}._read_metadata_from_disk", return_value=None),
        patch(f"{_ROUTER_MOD}._BACKTEST_DIR", new=MagicMock()),
    ):
        # Make has_predictions_csv return False via the Path mock
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/summary")

    assert resp.status_code == 200
    data = resp.json()
    # Should have entries for all 3 roster models
    assert len(data) == 3
    assert "lgbm_cluster" in data
    assert "catboost_cluster" in data
    assert "chronos_bolt" in data
    # lgbm_cluster has a latest_run
    assert data["lgbm_cluster"]["latest_run"] is not None
    assert data["lgbm_cluster"]["latest_run"]["accuracy_pct"] == 72.5
    # chronos_bolt has no run
    assert data["chronos_bolt"]["latest_run"] is None


# ---------------------------------------------------------------------------
# 2. GET /backtest-management/{model_id}/runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_runs_returns_list():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        _run_row(run_id=10, model_id="lgbm_cluster", status="completed"),
        _run_row(run_id=9, model_id="lgbm_cluster", status="running", accuracy_pct=None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/lgbm_cluster/runs")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["id"] == 10
    assert data[0]["status"] == "completed"
    assert data[1]["accuracy_pct"] is None


@pytest.mark.asyncio
async def test_get_model_runs_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/lgbm_cluster/runs")

    assert resp.status_code == 200
    data = resp.json()
    assert data == []


# ---------------------------------------------------------------------------
# 3. GET /backtest-management/{model_id}/current
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_current_metadata_success():
    meta = {
        "model_id": "lgbm_cluster",
        "accuracy_pct": 73.1,
        "n_predictions": 50000,
        "completed_at": "2026-04-06T08:00:00Z",
    }

    with patch(f"{_ROUTER_MOD}._read_metadata_from_disk", return_value=meta):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/lgbm_cluster/current")

    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "lgbm_cluster"
    assert data["accuracy_pct"] == 73.1


@pytest.mark.asyncio
async def test_get_current_metadata_not_found():
    with patch(f"{_ROUTER_MOD}._read_metadata_from_disk", return_value=None):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/unknown_model/current")

    assert resp.status_code == 404
    assert "No backtest metadata" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 4. POST /backtest-management/{model_id}/run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_run_success():
    pool, conn, cursor = _make_pool()
    # fetchone order: (1) duplicate-check -> None (no dup), (2) INSERT RETURNING id -> 42
    cursor.fetchone.side_effect = [None, (42,)]

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-bt-999"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run")

    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] == 42
    assert data["job_id"] == "job-bt-999"
    assert data["model_id"] == "lgbm_cluster"
    assert data["status"] == "queued"
    # Sequential (default): no per-family group override.
    assert mock_jm.return_value.submit_job.call_args.kwargs["group_override"] is None


@pytest.mark.asyncio
async def test_submit_run_rejects_duplicate():
    """A second run of the same family while one is running/queued -> 409."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [("job-bt-existing",)]  # duplicate-check finds one

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run")

    assert resp.status_code == 409
    assert "already running or queued" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_submit_run_parallel_uses_per_family_group():
    """parallel=true -> submit_job gets the per-job-type group so families run concurrently."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [None, (43,)]

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-bt-1000"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run?parallel=true")

    assert resp.status_code == 201
    assert mock_jm.return_value.submit_job.call_args.kwargs["group_override"] == "backtest_lgbm"


@pytest.mark.asyncio
async def test_submit_run_invalid_model():
    pool, conn, cursor = _make_pool()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/nonexistent_model/run")

    assert resp.status_code == 404
    assert "Unknown model_id" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 5. POST /backtest-management/{model_id}/load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_load_success():
    pool, conn, cursor = _make_pool()

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-load-555"

    mock_pred_path = MagicMock()
    mock_pred_path.exists.return_value = True
    mock_pred_path.relative_to.return_value = "data/backtest/lgbm_cluster/backtest_predictions.csv"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}._BACKTEST_DIR") as mock_dir,
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        # Set up the path resolution: _BACKTEST_DIR / dir_name / "backtest_predictions.csv"
        mock_dir.__truediv__ = MagicMock(return_value=MagicMock())
        mock_dir.__truediv__.return_value.__truediv__ = MagicMock(return_value=mock_pred_path)
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/load")

    assert resp.status_code == 201
    data = resp.json()
    assert data["job_id"] == "job-load-555"
    assert data["model_id"] == "lgbm_cluster"


@pytest.mark.asyncio
async def test_submit_load_no_predictions():
    pool, conn, cursor = _make_pool()

    mock_pred_path = MagicMock()
    mock_pred_path.exists.return_value = False
    mock_pred_path.relative_to.return_value = "data/backtest/lgbm_cluster/backtest_predictions.csv"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}._BACKTEST_DIR") as mock_dir,
    ):
        mock_dir.__truediv__ = MagicMock(return_value=MagicMock())
        mock_dir.__truediv__.return_value.__truediv__ = MagicMock(return_value=mock_pred_path)
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/load")

    assert resp.status_code == 404
    assert "No predictions CSV" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 6. POST /backtest-management/{model_id}/promote — Gen-4 CC#6 gating
# ---------------------------------------------------------------------------


def _gate_cfg_enabled(min_wape_impr: float = 1.0, min_cov: float = 0.8) -> dict:
    return {
        "champion": {
            "promote_gate": {
                "enabled": True,
                "min_wape_improvement_pct": min_wape_impr,
                "min_coverage_frac": min_cov,
                "bypass_token": None,
            }
        }
    }


@pytest.mark.asyncio
async def test_promote_gate_rejects_when_wape_regresses():
    pool, conn, cursor = _make_pool()
    # Sequence of cursor calls inside promote_model when the gate triggers:
    #  1) SELECT active champion      -> row
    #  2) SELECT champion wape        -> row
    #  3) SELECT candidate wape       -> row (worse)
    #  4) append_decision: SELECT latest ledger hash (fetchone)
    #  5) append_decision: INSERT ... RETURNING id (fetchone)
    # After raising we expect the HTTP response to be 409.
    cursor.fetchone.side_effect = [
        ("catboost_cluster", 1000),      # active champion row (model, dfu_count)
        (0.25,),                          # champion wape
        (0.30,),                          # candidate wape (worse)
        (None,),                          # ledger previous hash (none)
        (1,),                             # ledger insert RETURNING id
    ]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.load_forecast_pipeline_config", return_value=_gate_cfg_enabled()),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/promote")

    assert resp.status_code == 409
    assert "wape_improvement_too_small" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_promote_gate_first_promotion_allowed_and_logs_to_ledger():
    pool, conn, cursor = _make_pool()
    # No active champion -> gate allows as first_promotion; ledger logs success.
    # Then the rest of promote_model proceeds: COUNT staging, DELETE production,
    # INSERT from staging, COUNT DFUs, INSERT into model_promotion_log.
    cursor.fetchone.side_effect = [
        None,           # SELECT active champion -> no row -> first_promotion
        (None,),        # ledger prev hash (genesis)
        (1,),           # ledger insert returning id
        (500,),         # staging count
        (250,),         # DFU count
        (99,),          # lineage emit returning id (Gen-4 G)
    ]
    cursor.rowcount = 500

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.load_forecast_pipeline_config", return_value=_gate_cfg_enabled()),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/promote")

    assert resp.status_code == 201
    data = resp.json()
    assert data["model_id"] == "lgbm_cluster"
    assert data["promotion_type"] == "single"


@pytest.mark.asyncio
async def test_promote_gate_disabled_skips_checks():
    pool, conn, cursor = _make_pool()
    # Gate disabled -> short-circuits to "applied" ledger log, then:
    #   staging COUNT -> DFU COUNT -> ledger prev_hash -> ledger insert id.
    cursor.fetchone.side_effect = [
        (None,),        # ledger prev_hash (genesis fallback)
        (1,),           # ledger insert returning id (for applied log)
        (500,),         # staging count
        (250,),         # DFU count
        (99,),          # lineage emit returning id (Gen-4 G)
    ]
    cursor.rowcount = 500

    cfg = {"champion": {"promote_gate": {"enabled": False}}}
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.load_forecast_pipeline_config", return_value=cfg),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/promote")

    assert resp.status_code == 201
