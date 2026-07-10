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
_PROMOTION_ROUTER_MOD = "api.routers.forecasting.forecast_promotion"


@pytest.fixture(autouse=True)
def _no_real_mv_refresh():
    """Successful promotes refresh fact_production_forecast dependents through
    the central MV service, which opens its own DB connection — stub it so
    tests never touch a live database."""
    with patch("common.core.mv_refresh.refresh_for_tables") as refresh:
        yield refresh


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
        "chronos2_enriched": {"type": "foundation", "enabled": True},
        "mstl": {"type": "statistical", "enabled": True},
        "nbeats": {"type": "deep_learning", "enabled": True},
        "nhits": {"type": "deep_learning", "enabled": True},
    }


# ---------------------------------------------------------------------------
# 1. GET /backtest-management/summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_summary_returns_all_models():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        _run_row(run_id=1, model_id="lgbm_cluster", accuracy_pct=72.5),
            _run_row(run_id=2, model_id="mstl", accuracy_pct=70.0),
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
    # Should have entries for the complete lite roster.
    assert len(data) == len(_mock_roster())
    assert "lgbm_cluster" in data
    assert "mstl" in data
    assert data["chronos2_enriched"]["has_job_type"] is True
    # lgbm_cluster has a latest_run
    assert data["lgbm_cluster"]["latest_run"] is not None
    assert data["lgbm_cluster"]["latest_run"]["accuracy_pct"] == 72.5


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
    kwargs = mock_jm.return_value.submit_job.call_args.kwargs
    assert kwargs["params"] == {"backtest_run_id": 42, "model_id": "lgbm_cluster"}
    assert kwargs["group_override"] is None


@pytest.mark.asyncio
async def test_submit_run_already_running_is_informational():
    """Re-running a model with a run already in flight is a no-op, not an error.

    The endpoint returns 200 with status="already_running" and the existing job,
    and does NOT submit a duplicate job — concurrency never blocks the user.
    """
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [("job-bt-existing",)]  # in-flight check finds a run

    mock_jm = MagicMock()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "already_running"
    assert data["job_id"] == "job-bt-existing"
    assert data["run_id"] is None
    # No duplicate job submitted.
    mock_jm.return_value.submit_job.assert_not_called()


@pytest.mark.asyncio
async def test_submit_run_releases_row_when_submit_fails():
    """If submit_job fails, the queued tracking row is marked failed so the model
    is not permanently locked out of future runs (the in-flight check keys on
    status IN ('queued','running'))."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [None, (44,)]  # in-flight None, INSERT RETURNING 44

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.side_effect = ValueError("unknown job type")

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/lgbm_cluster/run")

    assert resp.status_code == 400
    # The orphaned 'queued' row was released (marked failed) in the finally block.
    executed = " ".join(str(c.args[0]) for c in cursor.execute.call_args_list if c.args)
    assert "status = 'failed'" in executed


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
# 7. POST /backtest-management/{model_id}/generate — horizon + CI threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_threads_horizon_and_confidence_intervals():
    """horizon + confidence_intervals query params reach the job params.

    Regression: these were previously dropped for single-model generation, so
    the Forecast panel's horizon input and CI toggle silently had no effect.
    """
    pool, conn, cursor = _make_pool()
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-gen-1"
    source_run_id = "00000000-0000-0000-0000-000000000091"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
        patch(f"{_PROMOTION_ROUTER_MOD}.uuid4", return_value=source_run_id),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/backtest-management/lgbm_cluster/generate?horizon=9&confidence_intervals=true"
            )

    assert resp.status_code == 201
    assert resp.json()["job_id"] == "job-gen-1"
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {
        "model_id": "lgbm_cluster",
        "run_id": source_run_id,
        "generation_purpose": "release_candidate",
        "horizon": 9,
        "confidence_intervals": True,
    }


@pytest.mark.asyncio
async def test_generate_omits_unset_params_for_config_default():
    """Without query params, only model_id is passed (script/config defaults apply)."""
    pool, conn, cursor = _make_pool()
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-gen-2"
    source_run_id = "00000000-0000-0000-0000-000000000092"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
        patch(f"{_PROMOTION_ROUTER_MOD}.uuid4", return_value=source_run_id),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/catboost_cluster/generate")

    assert resp.status_code == 201
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {
        "model_id": "catboost_cluster",
        "run_id": source_run_id,
        "generation_purpose": "release_candidate",
    }


@pytest.mark.asyncio
async def test_generate_threads_confidence_intervals_false():
    """confidence_intervals=false threads an explicit False (force CI off)."""
    pool, conn, cursor = _make_pool()
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-gen-3"
    source_run_id = "00000000-0000-0000-0000-000000000093"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
        patch(f"{_PROMOTION_ROUTER_MOD}.uuid4", return_value=source_run_id),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/backtest-management/lgbm_cluster/generate?confidence_intervals=false"
            )

    assert resp.status_code == 201
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {
        "model_id": "lgbm_cluster",
        "run_id": source_run_id,
        "generation_purpose": "release_candidate",
        "confidence_intervals": False,
    }


@pytest.mark.asyncio
async def test_generate_champion_omits_model_override_and_returns_source_run():
    pool, _, _ = _make_pool()
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-champion"
    source_run_id = "00000000-0000-0000-0000-000000000094"
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
        patch(f"{_PROMOTION_ROUTER_MOD}.uuid4", return_value=source_run_id),
    ):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/backtest-management/champion/generate?horizon=12")

    assert response.status_code == 201
    assert response.json()["source_run_id"] == source_run_id
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {
        "run_id": source_run_id,
        "generation_purpose": "release_candidate",
        "horizon": 12,
    }
