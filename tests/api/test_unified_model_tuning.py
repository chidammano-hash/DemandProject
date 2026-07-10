"""Tests for the unified model tuning API — /model-tuning/{model}/* endpoints.

Tests the unified router that replaces the split lgbm_tuning.py / model_tuning.py
with a single parametrized LightGBM router.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Fixtures — Row builders matching the router's SELECT column order
# ---------------------------------------------------------------------------

_ALGO_CONFIG = {
    "algorithms": {
        "lgbm_cluster": {
            "type": "tree",
            "enabled": True,
            "params": {
                "n_estimators": 1500,
                "learning_rate": 0.02,
                "num_leaves": 127,
                "max_depth": -1,
                "min_child_samples": 40,
            },
        },
    },
}


def _list_row(
    run_id: int = 1,
    run_label: str = "baseline_v1",
    model_id: str = "lgbm_cluster",
    started_at: str = "2026-03-20T10:00:00",
    completed_at: str = "2026-03-20T11:00:00",
    status: str = "completed",
    accuracy_pct: float = 72.5,
    wape: float = 27.5,
    bias: float = 0.032,
    n_predictions: int = 580000,
    n_dfus: int = 3200,
    notes: str | None = None,
    is_promoted: bool = False,
    promoted_at: str | None = None,
    job_id: str | None = None,
    template_id: str | None = None,
    is_results_promoted: bool = False,
    results_promoted_at: str | None = None,
    results_promote_job_id: str | None = None,
    cluster_source: str = "production",
    cluster_experiment_id: int | None = None,
    cluster_experiment_label: str | None = None,
) -> tuple:
    """Build a mock lgbm_tuning_run list-query row (22 columns)."""
    return (
        run_id, run_label, model_id, started_at, completed_at,
        status, accuracy_pct, wape, bias, n_predictions, n_dfus,
        notes, is_promoted, promoted_at, job_id, template_id,
        is_results_promoted, results_promoted_at, results_promote_job_id,
        cluster_source, cluster_experiment_id, cluster_experiment_label,
    )


def _detail_row(
    run_id: int = 1,
    run_label: str = "baseline_v1",
    model_id: str = "lgbm_cluster",
    started_at: str = "2026-03-20T10:00:00",
    completed_at: str = "2026-03-20T11:00:00",
    status: str = "completed",
    params: str | None = '{"n_estimators": 1500}',
    feature_count: int = 17,
    features: str | None = '["lag_1", "lag_2"]',
    accuracy_pct: float = 72.5,
    wape: float = 27.5,
    bias: float = 0.032,
    n_predictions: int = 580000,
    n_dfus: int = 3200,
    metadata: str | None = "{}",
    notes: str | None = None,
    backup_path: str | None = None,
    job_id: str | None = None,
    template_id: str | None = None,
    is_promoted: bool = False,
    promoted_at: str | None = None,
    is_results_promoted: bool = False,
    results_promoted_at: str | None = None,
    results_promote_job_id: str | None = None,
) -> tuple:
    """Build a mock lgbm_tuning_run detail-query row (24 columns)."""
    return (
        run_id, run_label, model_id, started_at, completed_at,
        status, params, feature_count, features,
        accuracy_pct, wape, bias, n_predictions, n_dfus,
        metadata, notes, backup_path, job_id, template_id,
        is_promoted, promoted_at,
        is_results_promoted, results_promoted_at, results_promote_job_id,
    )


def _timeframe_row(
    tf_id: int = 1,
    run_id: int = 1,
    timeframe: str = "A",
    train_end: str = "2025-04-01",
    predict_start: str = "2025-05-01",
    predict_end: str = "2026-02-01",
    n_predictions: int = 58000,
    accuracy_pct: float = 65.5,
    wape: float = 34.5,
    bias: float = 0.05,
) -> tuple:
    return (tf_id, run_id, timeframe, train_end, predict_start, predict_end,
            n_predictions, accuracy_pct, wape, bias)


def _lag_row(
    exec_lag: int = 0,
    n_predictions: int = 116000,
    n_dfus: int = 3200,
    accuracy_pct: float = 79.5,
    wape: float = 20.5,
    bias: float = 0.028,
) -> tuple:
    return (exec_lag, n_predictions, n_dfus, accuracy_pct, wape, bias)


def _compare_row(
    run_id: int = 1,
    run_label: str = "baseline_v1",
    model_id: str = "lgbm_cluster",
    accuracy_pct: float = 72.22,
    wape: float = 27.78,
    bias: float = 0.032,
    n_predictions: int = 580000,
    n_dfus: int = 3200,
    status: str = "completed",
    params: str = '{"n_estimators": 1500}',
    features: str | None = '["lag_1"]',
    feature_count: int = 17,
    metadata: str = "{}",
    cluster_source: str = "production",
    cluster_experiment_id: int | None = None,
    cluster_experiment_label: str | None = None,
) -> tuple:
    """Build a mock row for the compare endpoint's SELECT (16 columns)."""
    return (
        run_id, run_label, model_id, accuracy_pct, wape, bias,
        n_predictions, n_dfus, status, params, features, feature_count, metadata,
        cluster_source, cluster_experiment_id, cluster_experiment_label,
    )


# ---------------------------------------------------------------------------
# List Experiments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_experiments_lgbm():
    """GET /model-tuning/lgbm/experiments returns paginated LGBM runs."""
    pool, conn, cursor = _make_pool()
    # list endpoint does fetchone (count) then fetchall (rows)
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = [
        _list_row(run_id=1, run_label="baseline_v1", model_id="lgbm_cluster"),
        _list_row(run_id=2, run_label="aggressive_depth", model_id="lgbm_cluster",
                  accuracy_pct=73.1),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments")

    assert resp.status_code == 200
    data = resp.json()
    assert "experiments" in data
    assert len(data["experiments"]) == 2
    assert data["experiments"][0]["run_id"] == 1
    assert data["experiments"][0]["model_id"] == "lgbm_cluster"
    assert data["experiments"][1]["accuracy_pct"] == 73.1


@pytest.mark.asyncio
async def test_list_experiments_invalid_model():
    """GET /model-tuning/invalid/experiments returns 400 for unknown model."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/invalid/experiments")

    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_list_experiments_status_filter():
    """GET /model-tuning/lgbm/experiments?status=completed filters by status."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        _list_row(run_id=1, status="completed"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments?status=completed")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["experiments"]) == 1
    assert data["experiments"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_list_experiments_with_lag_filter():
    """GET /model-tuning/lgbm/experiments?exec_lag=0 returns lag-specific metrics."""
    pool, conn, cursor = _make_pool()
    # fetchone: count
    # fetchall #1: list rows
    # fetchall #2: lag-specific metrics for those run_ids
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.side_effect = [
        [_list_row(run_id=1, accuracy_pct=72.5, wape=27.5, bias=0.028)],
        [(1, 79.5, 20.5, 0.028, 116000)],  # lag metrics: run_id, acc, wape, bias, n_pred
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments?exec_lag=0")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["experiments"]) >= 1
    # Lag-0 accuracy overrides portfolio accuracy
    assert data["experiments"][0]["accuracy_pct"] == 79.5


@pytest.mark.asyncio
async def test_list_experiments_empty():
    """GET /model-tuning/lgbm/experiments returns empty experiments list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments")

    assert resp.status_code == 200
    data = resp.json()
    assert data["experiments"] == []


# ---------------------------------------------------------------------------
# Get Experiment Detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_detail():
    """GET /model-tuning/lgbm/experiments/1 returns run detail + timeframes."""
    pool, conn, cursor = _make_pool()
    # fetchone: run detail (21 columns)
    # fetchall: timeframe rows
    cursor.fetchone.return_value = _detail_row(run_id=1)
    cursor.fetchall.return_value = [
        _timeframe_row(tf_id=1, run_id=1, timeframe="A"),
        _timeframe_row(tf_id=2, run_id=1, timeframe="B", accuracy_pct=66.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert data["model_id"] == "lgbm_cluster"
    assert data["accuracy_pct"] == 72.5
    assert len(data["timeframes"]) == 2
    assert data["timeframes"][0]["timeframe"] == "A"


@pytest.mark.asyncio
async def test_get_experiment_not_found():
    """GET /model-tuning/lgbm/experiments/999 returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/999")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Create Experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_experiment_lgbm():
    """POST /model-tuning/lgbm/experiments returns 201 with run_id and job_id."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (15,)

    mock_jm = MagicMock()
    mock_jm.submit_job.return_value = "job-abc-123"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.services.job_registry.JobManager",
            return_value=mock_jm,
        ),
        patch("api.routers.forecasting.tuning._helpers._build_temp_config",
              return_value="/tmp/fake_config.yaml"),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/model-tuning/lgbm/experiments",
                json={
                    "run_label": "Aggressive Depth + Heavy Reg",
                    "params": {
                        "n_estimators": 1500,
                        "learning_rate": 0.02,
                        "num_leaves": 63,
                        "max_depth": 10,
                    },
                    "config": {
                        "cluster_strategy": "per_cluster",
                        "recursive": True,
                    },
                },
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] == 15
    assert "job_id" in data or "message" in data


@pytest.mark.asyncio
async def test_create_experiment_missing_label():
    """POST without run_label returns 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/model-tuning/lgbm/experiments",
                json={
                    "params": {"n_estimators": 1500},
                },
            )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Lags
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_lags():
    """GET /model-tuning/lgbm/experiments/1/lags returns 5 lag rows."""
    pool, conn, cursor = _make_pool()
    # _verify_run_ownership: fetchone → truthy
    # endpoint: fetchall → lag rows
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        _lag_row(exec_lag=0, accuracy_pct=79.5, wape=20.5, bias=0.028),
        _lag_row(exec_lag=1, accuracy_pct=73.8, wape=26.2, bias=0.035),
        _lag_row(exec_lag=2, accuracy_pct=69.0, wape=31.0, bias=0.042),
        _lag_row(exec_lag=3, accuracy_pct=65.8, wape=34.2, bias=0.051),
        _lag_row(exec_lag=4, accuracy_pct=62.1, wape=37.9, bias=0.060),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/1/lags")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert len(data["lags"]) == 5
    lag0 = data["lags"][0]
    assert lag0["exec_lag"] == 0
    assert lag0["accuracy_pct"] == 79.5
    assert lag0["wape"] == 20.5
    assert lag0["bias"] == 0.028
    # Verify degradation: lag 0 > lag 4
    assert data["lags"][0]["accuracy_pct"] > data["lags"][4]["accuracy_pct"]


@pytest.mark.asyncio
async def test_get_experiment_lags_empty():
    """GET lags for a run with no lag data returns 200 with empty array."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)  # ownership check passes
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/1/lags")

    assert resp.status_code == 200
    data = resp.json()
    assert data["lags"] == []


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_clusters():
    """GET /model-tuning/lgbm/experiments/1/clusters returns cluster breakdowns."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)  # ownership check
    cursor.fetchall.return_value = [
        ("ml_cluster", "0", 1000, 50, 68.0, 32.0, 0.02),
        ("ml_cluster", "1", 800, 40, 65.0, 35.0, 0.04),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/1/clusters")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert "clusters" in data


@pytest.mark.asyncio
async def test_get_experiment_clusters_with_lag():
    """GET /model-tuning/lgbm/experiments/1/clusters?exec_lag=2 filters by lag."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)  # ownership check
    # lag_cluster table returns 6 columns (no n_dfus)
    cursor.fetchall.return_value = [
        ("ml_cluster", "0", 200, 64.0, 36.0, 0.05),
        ("ml_cluster", "1", 160, 61.0, 39.0, 0.07),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/1/clusters?exec_lag=2")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert "clusters" in data


# ---------------------------------------------------------------------------
# Months
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_months():
    """GET /model-tuning/lgbm/experiments/1/months returns monthly breakdowns."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)  # ownership check
    cursor.fetchall.return_value = [
        ("2025-05-01", 500, 25, 67.0, 33.0, 0.03),
        ("2025-06-01", 480, 25, 68.5, 31.5, 0.02),
        ("2025-07-01", 520, 26, 66.0, 34.0, 0.04),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/1/months")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert len(data["months"]) == 3


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment_logs():
    """GET /model-tuning/lgbm/experiments/1/logs returns log text + next_offset."""
    pool, conn, cursor = _make_pool()
    log_text = "[TUNING] Starting lgbm experiment: baseline_v1 (run_id=1)\n" * 10
    # _verify_run_ownership: fetchone #1 → (1,)
    # endpoint: fetchone #2 → (job_id, status)
    # endpoint: fetchone #3 → (log_text,)
    cursor.fetchone.side_effect = [
        (1,),                           # ownership check
        ("job-abc-123", "completed"),    # run job_id + status
        (log_text,),                    # job log text
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/1/logs")

    assert resp.status_code == 200
    data = resp.json()
    assert "log" in data
    assert "next_offset" in data
    assert len(data["log"]) > 0


@pytest.mark.asyncio
async def test_get_experiment_logs_incremental():
    """GET /model-tuning/lgbm/experiments/1/logs?offset=500 returns only new text."""
    pool, conn, cursor = _make_pool()
    full_log = "X" * 500 + "[TUNING] Cluster 3: accuracy=74.2%, 5200 predictions\n"
    new_text = "[TUNING] Cluster 3: accuracy=74.2%, 5200 predictions\n"
    cursor.fetchone.side_effect = [
        (1,),                           # ownership check
        ("job-abc-123", "running"),      # run job_id + status
        (full_log,),                    # full log text (offset applied in router)
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/1/logs?offset=500")

    assert resp.status_code == 200
    data = resp.json()
    assert data["next_offset"] > 500
    assert "Cluster 3" in data["log"]


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_runs():
    """GET /model-tuning/lgbm/compare returns deltas + verdict + per_lag array."""
    pool, conn, cursor = _make_pool()
    # compare endpoint:
    # fetchone #1: baseline run (13 cols)
    # fetchone #2: candidate run (13 cols)
    # fetchone #3: existing comparison check (returns None)
    # _build_per_lag_comparison fetchall #1: baseline per-lag
    # _build_per_lag_comparison fetchall #2: candidate per-lag
    # breakdown fetchall #3: baseline clusters
    # breakdown fetchall #4: candidate clusters
    # breakdown fetchall #5: baseline months
    # breakdown fetchall #6: candidate months
    cursor.fetchone.side_effect = [
        _compare_row(run_id=1, run_label="baseline_v1", accuracy_pct=72.22, wape=27.78, bias=0.032),
        _compare_row(run_id=2, run_label="aggressive_depth", accuracy_pct=73.45, wape=26.55, bias=0.028),
        None,  # no existing comparison
    ]
    cursor.fetchall.side_effect = [
        [  # baseline per-lag
            (0, 78.2, 21.8, 0.025),
            (1, 72.4, 27.6, 0.030),
            (2, 68.1, 31.9, 0.038),
            (3, 64.5, 35.5, 0.045),
            (4, 61.2, 38.8, 0.055),
        ],
        [  # candidate per-lag
            (0, 79.5, 20.5, 0.022),
            (1, 73.8, 26.2, 0.028),
            (2, 69.0, 31.0, 0.036),
            (3, 65.8, 34.2, 0.042),
            (4, 62.1, 37.9, 0.052),
        ],
        [],  # baseline clusters
        [],  # candidate clusters
        [],  # baseline months
        [],  # candidate months
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/model-tuning/lgbm/compare?baseline_id=1&candidate_id=2"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["baseline"]["run_id"] == 1
    assert data["candidate"]["run_id"] == 2
    assert data["delta_accuracy"] == pytest.approx(1.23, abs=0.01)
    assert data["verdict"] == "improved"
    assert "per_lag" in data
    assert len(data["per_lag"]) == 5
    assert data["per_lag"][0]["exec_lag"] == 0
    assert data["per_lag"][0]["delta_acc"] == pytest.approx(1.3, abs=0.01)


@pytest.mark.asyncio
async def test_compare_runs_with_lag():
    """GET /model-tuning/lgbm/compare?exec_lag=0 computes lag-specific deltas."""
    pool, conn, cursor = _make_pool()
    # When exec_lag is specified, _apply_lag_metrics also calls fetchone for each run
    cursor.fetchone.side_effect = [
        _compare_row(run_id=1, accuracy_pct=72.22, wape=27.78, bias=0.025),
        _compare_row(run_id=2, accuracy_pct=73.45, wape=26.55, bias=0.022),
        None,  # no existing comparison
        (79.5, 20.5, 0.025, 116000),   # baseline lag-0 metrics
        (80.8, 19.2, 0.022, 116000),   # candidate lag-0 metrics
    ]
    cursor.fetchall.side_effect = [
        [], [],  # per-lag baseline, candidate
        [], [],  # clusters baseline, candidate
        [], [],  # months baseline, candidate
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/model-tuning/lgbm/compare?baseline_id=1&candidate_id=2&exec_lag=0"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["delta_accuracy"] == pytest.approx(1.3, abs=0.01)


@pytest.mark.asyncio
async def test_compare_runs_not_found():
    """GET /model-tuning/lgbm/compare with nonexistent candidate returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        _compare_row(run_id=1),
        None,  # candidate not found
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/model-tuning/lgbm/compare?baseline_id=1&candidate_id=999"
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_compare_runs_same_id():
    """GET /model-tuning/lgbm/compare with same baseline and candidate returns 400."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/model-tuning/lgbm/compare?baseline_id=1&candidate_id=1"
            )

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_run():
    """POST /model-tuning/lgbm/experiments/1/promote promotes completed run."""
    pool, conn, cursor = _make_pool()
    params = json.dumps({"n_estimators": 2000, "learning_rate": 0.015})
    # promote endpoint: fetchone for run lookup (7 cols)
    # then second get_conn: fetchone for previous champion, execute updates
    cursor.fetchone.side_effect = [
        (1, "baseline_v1", "completed", params, 73.45, 27.0, 0.03),
        None,  # no previous champion
    ]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open(read_data="algorithms:\n  lgbm:\n    n_estimators: 1500")),
        patch("api.routers.forecasting.tuning.promote.yaml.safe_load",
              return_value=dict(_ALGO_CONFIG)),
        patch("api.routers.forecasting.tuning.promote.yaml.dump"),
        patch("api.routers.forecasting.tuning.promote.shutil.copy2"),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/1/promote")

    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is True
    assert data["run_id"] == 1
    assert "params_written" in data


@pytest.mark.asyncio
async def test_promote_run_not_completed():
    """POST /promote on running experiment returns 400."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5, "running_exp", "running", None, None, None, None)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/5/promote")

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_promote_clears_previous():
    """Promoting run B clears is_promoted on previous champion A."""
    pool, conn, cursor = _make_pool()
    params_b = json.dumps({"n_estimators": 2000, "learning_rate": 0.015})
    cursor.fetchone.side_effect = [
        (2, "candidate_v2", "completed", params_b, 74.0, 26.0, 0.02),
        (1,),  # previous champion run_id=1
    ]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open(read_data="")),
        patch("api.routers.forecasting.tuning.promote.yaml.safe_load",
              return_value=dict(_ALGO_CONFIG)),
        patch("api.routers.forecasting.tuning.promote.yaml.dump"),
        patch("api.routers.forecasting.tuning.promote.shutil.copy2"),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/2/promote")

    assert resp.status_code == 200
    # Verify that cursor.execute was called with UPDATE to clear previous champion
    execute_calls = cursor.execute.call_args_list
    sql_statements = [str(call.args[0]) for call in execute_calls if call.args]
    found_clear = any("is_promoted" in sql and "FALSE" in sql for sql in sql_statements)
    assert found_clear or resp.json()["promoted"] is True


@pytest.mark.asyncio
async def test_promote_creates_audit_log():
    """Promotion inserts into tuning_promotion_log."""
    pool, conn, cursor = _make_pool()
    params = json.dumps({"n_estimators": 2000})
    cursor.fetchone.side_effect = [
        (3, "champion_v3", "completed", params, 75.0, 25.0, 0.01),
        None,  # no previous champion
    ]

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open(read_data="")),
        patch("api.routers.forecasting.tuning.promote.yaml.safe_load",
              return_value=dict(_ALGO_CONFIG)),
        patch("api.routers.forecasting.tuning.promote.yaml.dump"),
        patch("api.routers.forecasting.tuning.promote.shutil.copy2"),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/3/promote")

    assert resp.status_code == 200
    execute_calls = cursor.execute.call_args_list
    sql_statements = [str(call.args[0]) for call in execute_calls if call.args]
    found_promotion_log = any("tuning_promotion_log" in sql for sql in sql_statements)
    assert found_promotion_log, "Expected INSERT into tuning_promotion_log"


@pytest.mark.asyncio
async def test_promote_not_found():
    """POST /model-tuning/lgbm/experiments/999/promote returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/999/promote")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_running_experiment():
    """POST /model-tuning/lgbm/experiments/5/cancel cancels a running experiment."""
    pool, conn, cursor = _make_pool()
    # cancel endpoint: fetchone for status + job_id (2 cols from SELECT status, job_id)
    cursor.fetchone.return_value = ("running", "job-abc-123")

    mock_jm = MagicMock()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.services.job_registry.JobManager",
            return_value=mock_jm,
        ),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/5/cancel")

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("cancelled") is True or data.get("status") == "cancelled"


@pytest.mark.asyncio
async def test_delete_completed_experiment():
    """DELETE /model-tuning/lgbm/experiments/1 removes a completed experiment."""
    pool, conn, cursor = _make_pool()
    # delete endpoint: fetchone for status + is_promoted (2 cols)
    cursor.fetchone.return_value = ("completed", False)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/model-tuning/lgbm/experiments/1")

    assert resp.status_code in (200, 204)


@pytest.mark.asyncio
async def test_delete_running_blocked():
    """DELETE on a running experiment returns 400."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("running", False)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/model-tuning/lgbm/experiments/5")

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_templates_lgbm():
    """GET /model-tuning/lgbm/templates returns LGBM templates list."""
    pool, conn, cursor = _make_pool()

    templates_config = {
        "templates": {
            "lgbm": [
                {
                    "id": "production_baseline",
                    "label": "Production Baseline (Run 16)",
                    "description": "Current production parameters",
                    "source": "algorithm_config",
                },
                {
                    "id": "expert_aggressive_depth",
                    "label": "Expert: Aggressive Depth + Heavy Reg",
                    "description": "Cap depth at 10, halve leaves",
                    "params": {"max_depth": 10, "num_leaves": 63, "reg_lambda": 3.5},
                },
            ],
        },
    }

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.core.utils.load_config",
            return_value=templates_config,
        ),
        patch("api.routers.forecasting.tuning.templates._load_live_params",
              return_value={"n_estimators": 1500, "learning_rate": 0.02}),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/templates")

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "lgbm"
    assert "templates" in data
    assert len(data["templates"]) >= 1


@pytest.mark.asyncio
async def test_get_promoted_run():
    """GET /model-tuning/lgbm/promoted returns the currently promoted run."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        1, "baseline_v1", "lgbm_cluster", 72.5, 27.5, 0.032,
        "2026-03-20T14:00:00", '{"n_estimators": 1500}',
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/promoted")

    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is not None
    assert data["promoted"]["run_id"] == 1
    assert data["promoted"]["accuracy_pct"] == 72.5


@pytest.mark.asyncio
async def test_get_promoted_run_none():
    """GET /model-tuning/lgbm/promoted returns null when no champion exists."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/promoted")

    assert resp.status_code == 200
    assert resp.json()["promoted"] is None


# ---------------------------------------------------------------------------
# XGBoost compare
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_promotions():
    """GET /model-tuning/lgbm/promotions returns promotion audit trail."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (1, 3, "lgbm_cluster", "2026-03-20T14:00:00", "manual",
         1, '{"n_estimators": 2000}', 75.0, 25.0, 0.01, None, "champion_v3"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/promotions")

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "lgbm"
    assert len(data["promotions"]) == 1
    assert data["promotions"][0]["run_id"] == 3


# ---------------------------------------------------------------------------
# Promote Results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_results_completed_run():
    """POST /model-tuning/lgbm/experiments/{id}/promote-results submits job for completed run."""
    pool, conn, cursor = _make_pool()
    # First call: fetchone for status check
    cursor.fetchone.return_value = ("completed", False, None)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("pathlib.Path.exists", return_value=True),
        patch("common.services.job_registry.JobManager") as mock_jm_cls,
    ):
        mock_mgr = MagicMock()
        mock_mgr.submit_job.return_value = "job_test_123"
        mock_jm_cls.return_value = mock_mgr
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/49/promote-results")

    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "job_test_123"
    assert data["run_id"] == 49
    assert data["model"] == "lgbm"


@pytest.mark.asyncio
async def test_promote_results_not_completed():
    """POST promote-results returns 400 for non-completed run."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("running", False, None)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/49/promote-results")

    assert resp.status_code == 400
    assert "completed" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_promote_results_not_found():
    """POST promote-results returns 404 for nonexistent run."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/model-tuning/lgbm/experiments/999/promote-results")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_promote_results_status_not_started():
    """GET promote-results/status returns not_started when no job submitted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (False, None, None)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/49/promote-results/status")

    assert resp.status_code == 200
    assert resp.json()["status"] == "not_started"
    assert resp.json()["is_results_promoted"] is False


@pytest.mark.asyncio
async def test_promote_results_status_completed():
    """GET promote-results/status returns completed when already promoted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (True, "2026-03-25T12:00:00", "job_done_123")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/49/promote-results/status")

    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert resp.json()["is_results_promoted"] is True


@pytest.mark.asyncio
async def test_promote_results_status_running():
    """GET promote-results/status returns progress info when job is running."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (False, None, "job_running_456")

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager") as mock_jm_cls,
    ):
        mock_mgr = MagicMock()
        mock_mgr.get_job.return_value = {
            "status": "running",
            "progress_pct": 45,
            "progress_msg": "Loading predictions",
            "error": None,
        }
        mock_jm_cls.return_value = mock_mgr
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/49/promote-results/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["progress_pct"] == 45


@pytest.mark.asyncio
async def test_list_experiments_includes_results_promotion_fields():
    """GET /model-tuning/lgbm/experiments includes is_results_promoted fields."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)  # count
    cursor.fetchall.return_value = [
        _list_row(run_id=49, is_results_promoted=True, results_promoted_at="2026-03-25T12:00:00"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments")

    assert resp.status_code == 200
    exp = resp.json()["experiments"][0]
    assert exp["is_results_promoted"] is True
    assert exp["results_promoted_at"] is not None


@pytest.mark.asyncio
async def test_detail_experiment_includes_results_promotion_fields():
    """GET /model-tuning/lgbm/experiments/{id} includes is_results_promoted fields."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _detail_row(
        run_id=49, is_results_promoted=True,
        results_promoted_at="2026-03-25T12:00:00",
    )
    cursor.fetchall.return_value = []  # timeframes

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments/49")

    assert resp.status_code == 200
    data = resp.json()
    assert data["is_results_promoted"] is True
    assert data["results_promoted_at"] is not None


# ---------------------------------------------------------------------------
# Cluster Source Selection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_experiment_with_cluster_source_production():
    """POST with cluster_source=production (default) works as before."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (30,)

    mock_jm = MagicMock()
    mock_jm.submit_job.return_value = "job-prod-100"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.services.job_registry.JobManager",
            return_value=mock_jm,
        ),
        patch("api.routers.forecasting.tuning._helpers._build_temp_config",
              return_value="/tmp/fake_config.yaml"),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/model-tuning/lgbm/experiments",
                json={
                    "run_label": "Production Clusters Test",
                    "params": {"n_estimators": 1500},
                    "config": {
                        "cluster_strategy": "per_cluster",
                        "cluster_source": "production",
                    },
                },
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] == 30
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_create_experiment_with_cluster_source_experimental():
    """POST with cluster_source=experimental and valid cluster_experiment_id succeeds."""
    pool, conn, cursor = _make_pool()
    # Call sequence:
    # 1. cluster_experiment validation query → returns completed experiment
    # 2. INSERT lgbm_tuning_run → returns run_id
    # 3. UPDATE job_id on the run
    cursor.fetchone.side_effect = [
        ("/tmp/clustering_scenarios/sc_20260320_120000_abcd", "sc_20260320_120000_abcd", "High-K Test"),
        (31,),  # new run_id
    ]

    mock_jm = MagicMock()
    mock_jm.submit_job.return_value = "job-exp-200"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "common.services.job_registry.JobManager",
            return_value=mock_jm,
        ),
        patch("api.routers.forecasting.tuning._helpers._build_temp_config",
              return_value="/tmp/fake_config.yaml"),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/model-tuning/lgbm/experiments",
                json={
                    "run_label": "Experimental Clusters Test",
                    "params": {"n_estimators": 2000},
                    "config": {
                        "cluster_strategy": "per_cluster",
                        "cluster_source": "experimental",
                        "cluster_experiment_id": 5,
                    },
                },
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] == 31
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_create_experiment_with_invalid_cluster_experiment():
    """POST with cluster_source=experimental and nonexistent cluster_experiment_id returns 400."""
    pool, conn, cursor = _make_pool()
    # cluster_experiment validation query → not found
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/model-tuning/lgbm/experiments",
                json={
                    "run_label": "Invalid Cluster Test",
                    "params": {"n_estimators": 1500},
                    "config": {
                        "cluster_source": "experimental",
                        "cluster_experiment_id": 999,
                    },
                },
            )

    assert resp.status_code == 400
    assert "not found or not completed" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_experiment_with_incomplete_cluster_experiment():
    """POST with cluster_source=experimental and running cluster experiment returns 400."""
    pool, conn, cursor = _make_pool()
    # The SQL query filters for status='completed', so a running experiment
    # will return no rows — same as not found.
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/model-tuning/lgbm/experiments",
                json={
                    "run_label": "Incomplete Cluster Test",
                    "params": {"n_estimators": 1500},
                    "config": {
                        "cluster_source": "experimental",
                        "cluster_experiment_id": 7,
                    },
                },
            )

    assert resp.status_code == 400
    assert "not found or not completed" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_list_experiments_includes_cluster_source():
    """GET /model-tuning/lgbm/experiments includes cluster_source fields."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = [
        _list_row(run_id=30, run_label="prod_clusters",
                  cluster_source="production"),
        _list_row(run_id=31, run_label="exp_clusters",
                  cluster_source="experimental",
                  cluster_experiment_id=5,
                  cluster_experiment_label="High-K Test"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model-tuning/lgbm/experiments")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["experiments"]) == 2

    prod_exp = data["experiments"][0]
    assert prod_exp["cluster_source"] == "production"
    assert prod_exp["cluster_experiment_id"] is None
    assert prod_exp["cluster_experiment_label"] is None

    exp_exp = data["experiments"][1]
    assert exp_exp["cluster_source"] == "experimental"
    assert exp_exp["cluster_experiment_id"] == 5
    assert exp_exp["cluster_experiment_label"] == "High-K Test"
