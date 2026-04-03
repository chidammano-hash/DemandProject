"""Tests for the champion experiments API — /champion-experiments/* endpoints.

Tests CRUD lifecycle, comparison, promotion (config + results), templates,
promoted experiment, audit trail, cancel, delete, and logs.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Row builder matching the router's _SELECT_COLS (28 columns)
# ---------------------------------------------------------------------------

def _experiment_row(
    experiment_id: int = 1,
    label: str = "Test Champion",
    notes: str | None = None,
    template_id: str | None = "expanding_conservative",
    status: str = "completed",
    created_at: str = "2026-03-25T10:00:00+00:00",
    started_at: str = "2026-03-25T10:00:05+00:00",
    completed_at: str = "2026-03-25T10:05:00+00:00",
    runtime_seconds: float = 295.0,
    job_id: str | None = "job-champ-123",
    strategy: str = "expanding",
    strategy_params: str | None = '{"min_prior_months": 3}',
    meta_learner_params: str | None = None,
    models: str = '["lgbm_cluster", "catboost_cluster", "xgboost_cluster"]',
    metric: str = "accuracy_pct",
    lag_mode: str = "execution",
    min_sku_rows: int = 3,
    champion_accuracy: float | None = 71.50,
    ceiling_accuracy: float | None = 78.20,
    gap_bps: float | None = 670.0,
    n_champions: int | None = 5000,
    n_dfu_months: int | None = 60000,
    model_distribution: str | None = '{"lgbm_cluster": 45.2, "catboost_cluster": 30.1, "xgboost_cluster": 24.7}',
    is_promoted: bool = False,
    promoted_at: str | None = None,
    is_results_promoted: bool = False,
    results_promoted_at: str | None = None,
    results_promote_job_id: str | None = None,
) -> tuple:
    """Build a mock champion_experiment row (28 columns)."""
    return (
        experiment_id, label, notes, template_id, status,
        created_at, started_at, completed_at, runtime_seconds, job_id,
        strategy, strategy_params, meta_learner_params, models, metric, lag_mode, min_sku_rows,
        champion_accuracy, ceiling_accuracy, gap_bps, n_champions, n_dfu_months, model_distribution,
        is_promoted, promoted_at, is_results_promoted, results_promoted_at, results_promote_job_id,
    )


def _lag_row(exec_lag=0, champion_accuracy=71.5, ceiling_accuracy=78.2,
             gap_bps=670, n_dfu_months=12000, model_distribution='{}'):
    return (exec_lag, champion_accuracy, ceiling_accuracy, gap_bps,
            n_dfu_months, model_distribution)


def _month_row(month_start="2025-01-01", champion_accuracy=71.5, ceiling_accuracy=78.2,
               gap_bps=670, n_champions=1000, model_distribution='{}'):
    return (month_start, champion_accuracy, ceiling_accuracy, gap_bps,
            n_champions, model_distribution)


# ---------------------------------------------------------------------------
# 1. List experiments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_experiments():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = [
        _experiment_row(experiment_id=1, label="Exp A"),
        _experiment_row(experiment_id=2, label="Exp B", status="running"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments?offset=0&limit=50")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["experiments"]) == 2
    assert data["experiments"][0]["label"] == "Exp A"


@pytest.mark.asyncio
async def test_list_experiments_with_status_filter():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        _experiment_row(experiment_id=1, status="completed"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments?status=completed")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["experiments"]) == 1


# ---------------------------------------------------------------------------
# 2. Get experiment detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_detail():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _experiment_row(experiment_id=5, label="Detail Test")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/5")

    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_id"] == 5
    assert data["label"] == "Detail Test"
    assert data["strategy"] == "expanding"


@pytest.mark.asyncio
async def test_get_experiment_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/999")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 3. Create experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (42,)  # RETURNING experiment_id

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-999"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/champion-experiments",
                json={
                    "label": "New Exp",
                    "strategy": "rolling",
                    "strategy_params": {"window_months": 6},
                    "models": ["lgbm_cluster", "catboost_cluster", "xgboost_cluster"],
                },
            )

    assert resp.status_code == 202
    data = resp.json()
    assert data["experiment_id"] == 42
    assert data["job_id"] == "job-999"
    assert data["strategy"] == "rolling"


@pytest.mark.asyncio
async def test_create_experiment_invalid_strategy():
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/champion-experiments",
                json={
                    "label": "Bad Strategy",
                    "strategy": "invalid_strategy",
                    "models": ["lgbm_cluster", "catboost_cluster"],
                },
            )

    assert resp.status_code == 400
    assert "Invalid strategy" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_experiment_insufficient_models():
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/champion-experiments",
                json={
                    "label": "One Model Only",
                    "strategy": "expanding",
                    "models": ["lgbm_cluster"],
                },
            )

    assert resp.status_code == 400
    assert "2 models" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 4. Get lags
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_lags():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)  # exists check
    cursor.fetchall.return_value = [
        _lag_row(exec_lag=0, champion_accuracy=72.0),
        _lag_row(exec_lag=1, champion_accuracy=70.5),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/1/lags")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["lags"]) == 2
    assert data["lags"][0]["exec_lag"] == 0


# ---------------------------------------------------------------------------
# 5. Get months
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_months():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        _month_row(month_start="2025-01-01"),
        _month_row(month_start="2025-02-01"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/1/months")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["months"]) == 2


# ---------------------------------------------------------------------------
# 6. Get logs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_logs():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("job-123", "running")

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.services.job_state.get_job_log",
            return_value="Log line 1\nLog line 2",
        ),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/1/logs?offset=0")

    assert resp.status_code == 200
    data = resp.json()
    assert "Log line" in data["log"]
    assert data["has_more"] is True


@pytest.mark.asyncio
async def test_get_experiment_logs_no_job():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (None, "queued")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/1/logs")

    assert resp.status_code == 200
    data = resp.json()
    assert data["log"] == ""


# ---------------------------------------------------------------------------
# 7. Compare experiments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_experiments_cached():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        '{"verdict": "b_better"}',  # overall
        '[]',  # per_lag
        '[]',  # per_month
        '[]',  # model_dist
        '[]',  # config_diffs
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/compare?a_id=1&b_id=2")

    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "cache"


@pytest.mark.asyncio
async def test_compare_same_id():
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/compare?a_id=1&b_id=1")

    assert resp.status_code == 400
    assert "itself" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_compare_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # cache miss
    cursor.fetchall.return_value = [
        _experiment_row(experiment_id=1),
    ]  # only 1 found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/compare?a_id=1&b_id=999")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 8. Promote experiment (Stage 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        _experiment_row(experiment_id=1, status="completed"),  # fetch
        None,  # previous promoted
    ]

    yaml_content = "competition:\n  strategy: expanding\n  models:\n    - lgbm_cluster\n"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open(read_data=yaml_content)),
        patch("pathlib.Path.exists", return_value=True),
        patch("shutil.copy2"),
        patch("common.utils.reset_config"),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-experiments/1/promote")

    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is True
    assert data["experiment_id"] == 1


@pytest.mark.asyncio
async def test_promote_experiment_not_completed():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _experiment_row(experiment_id=1, status="running")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-experiments/1/promote")

    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_promote_experiment_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-experiments/999/promote")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 9. Promote results (Stage 2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_results():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("completed", True, False)  # status, promoted, results_promoted

    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-load-999"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-experiments/1/promote-results")

    assert resp.status_code == 201
    data = resp.json()
    assert data["job_id"] == "job-load-999"


@pytest.mark.asyncio
async def test_promote_results_not_promoted():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("completed", False, False)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-experiments/1/promote-results")

    assert resp.status_code == 400
    assert "promoted" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_promote_results_already_loaded():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("completed", True, True)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-experiments/1/promote-results")

    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 10. Promote results status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_results_status():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (True, "2026-03-25T12:00:00+00:00", "job-load-999")

    mock_db_get = MagicMock(return_value={"status": "completed", "progress_pct": 100, "progress_msg": "done", "error": None})

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager._db_get", mock_db_get),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/1/promote-results/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["is_results_promoted"] is True


# ---------------------------------------------------------------------------
# 11. Cancel experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("running", "job-123")

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager"),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-experiments/1/cancel")

    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] is True


@pytest.mark.asyncio
async def test_cancel_completed_fails():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("completed", "job-123")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-experiments/1/cancel")

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 12. Delete experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("completed", False)  # status, is_promoted

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/champion-experiments/1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True


@pytest.mark.asyncio
async def test_delete_running_fails():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("running", False)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/champion-experiments/1")

    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_delete_promoted_fails():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("completed", True)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/champion-experiments/1")

    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 13. Templates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_templates():
    pool, conn, cursor = _make_pool()
    yaml_content = """
templates:
  - id: expanding_conservative
    label: "Expanding (Conservative)"
    strategy: expanding
    strategy_params:
      min_prior_months: 5
"""
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open(read_data=yaml_content)),
        patch("pathlib.Path.exists", return_value=True),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/templates")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["templates"]) == 1
    assert data["templates"][0]["id"] == "expanding_conservative"


@pytest.mark.asyncio
async def test_get_templates_missing_file():
    pool, conn, cursor = _make_pool()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", side_effect=FileNotFoundError("not found")),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/templates")

    assert resp.status_code == 200
    data = resp.json()
    assert data["templates"] == []


# ---------------------------------------------------------------------------
# 14. Promoted experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_promoted():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _experiment_row(
        experiment_id=3, is_promoted=True, promoted_at="2026-03-25T12:00:00+00:00",
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/promoted")

    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"]["experiment_id"] == 3
    assert data["promoted"]["is_promoted"] is True


@pytest.mark.asyncio
async def test_get_promoted_none():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/promoted")

    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is None


# ---------------------------------------------------------------------------
# 15. Promotions audit trail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_promotions():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (1, 3, "2026-03-25T12:00:00+00:00", "manual", None, "expanding", 71.5, "{}"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/promotions")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["promotions"]) == 1
    assert data["promotions"][0]["strategy"] == "expanding"


# ---------------------------------------------------------------------------
# 16. Execution lag filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_experiments_with_exec_lag():
    """exec_lag param overrides portfolio KPIs with lag-specific values."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)  # count
    cursor.fetchall.side_effect = [
        [_experiment_row(experiment_id=1, champion_accuracy=71.5)],
        # lag data for experiment 1 at exec_lag=2
        [(1, 68.0, 75.0, 700.0, 10000, '{"lgbm_cluster": 55}')],
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments?exec_lag=2")

    assert resp.status_code == 200
    data = resp.json()
    exp = data["experiments"][0]
    assert exp["champion_accuracy"] == 68.0
    assert exp["ceiling_accuracy"] == 75.0
    assert exp["gap_bps"] == 700.0
    assert exp["exec_lag_filter"] == 2


@pytest.mark.asyncio
async def test_list_experiments_exec_lag_no_data():
    """exec_lag with no lag rows keeps portfolio values, adds exec_lag_filter."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.side_effect = [
        [_experiment_row(experiment_id=1, champion_accuracy=71.5)],
        [],  # no lag data
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments?exec_lag=3")

    assert resp.status_code == 200
    data = resp.json()
    exp = data["experiments"][0]
    assert exp["champion_accuracy"] == 71.5  # unchanged
    assert exp["exec_lag_filter"] == 3


@pytest.mark.asyncio
async def test_get_experiment_detail_with_exec_lag():
    """exec_lag on detail endpoint overrides KPIs."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        _experiment_row(experiment_id=5, champion_accuracy=71.5),
        (69.0, 76.0, 700.0, 11000, '{"catboost_cluster": 60}'),  # lag row
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/5?exec_lag=1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["champion_accuracy"] == 69.0
    assert data["ceiling_accuracy"] == 76.0
    assert data["exec_lag_filter"] == 1


@pytest.mark.asyncio
async def test_compare_with_exec_lag():
    """exec_lag on compare endpoint overrides overall comparison KPIs."""
    pool, conn, cursor = _make_pool()
    # cache miss, then both experiments, then lag override, then per-lag, per-month, model_dist
    cursor.fetchone.return_value = None  # cache miss
    cursor.fetchall.side_effect = [
        # both experiments
        [
            _experiment_row(experiment_id=1, champion_accuracy=71.5),
            _experiment_row(experiment_id=2, champion_accuracy=73.0),
        ],
        # lag override for exec_lag=0
        [
            (1, 70.0, 76.0, 600.0, 10000, '{}'),
            (2, 72.0, 77.0, 500.0, 10000, '{}'),
        ],
        # per-lag comparison
        [],
        # per-month comparison
        [],
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments/compare?a_id=1&b_id=2&exec_lag=0")

    assert resp.status_code == 200
    data = resp.json()
    assert data["exec_lag_filter"] == 0
    overall = data["overall_comparison"]
    assert overall["experiment_a"]["champion_accuracy"] == 70.0
    assert overall["experiment_b"]["champion_accuracy"] == 72.0


@pytest.mark.asyncio
async def test_exec_lag_out_of_range():
    """exec_lag > 4 or < 0 should return 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-experiments?exec_lag=5")

    assert resp.status_code == 422
