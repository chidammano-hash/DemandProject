"""Tests for LGBM tuning API endpoints."""

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
import httpx
from httpx import ASGITransport


@pytest.fixture
def mock_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = mock_conn
    return pool, mock_conn, mock_cursor


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_runs_returns_200(mock_pool):
    """GET /lgbm-tuning/runs returns 200 with runs array."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        (1, "baseline", "lgbm_cluster", "2026-03-22T10:00:00", "2026-03-22T11:00:00",
         "completed", 69.34, 30.66, -0.0132, 2725140, 50602, None, True, "2026-03-23T08:00:00"),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert len(data["runs"]) == 1
    run = data["runs"][0]
    assert run["run_id"] == 1
    assert run["run_label"] == "baseline"
    assert run["model_id"] == "lgbm_cluster"
    assert run["status"] == "completed"
    assert run["accuracy_pct"] == 69.34
    assert run["wape"] == 30.66
    assert run["bias"] == -0.0132
    assert run["n_predictions"] == 2725140
    assert run["n_dfus"] == 50602
    assert run["notes"] is None
    assert run["is_promoted"] is True
    assert run["promoted_at"] is not None


@pytest.mark.asyncio
async def test_list_runs_empty(mock_pool):
    """GET /lgbm-tuning/runs returns empty array when no runs exist."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"] == []


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/runs/{run_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_run_returns_200(mock_pool):
    """GET /lgbm-tuning/runs/1 returns run detail with timeframes."""
    pool, _, cursor = mock_pool
    # fetchone: run row (17 columns)
    cursor.fetchone.return_value = (
        1, "baseline", "lgbm_cluster",
        "2026-03-22T10:00:00", "2026-03-22T11:00:00",
        "completed", '{}', 37, '[]',
        69.34, 30.66, -0.0132, 2725140, 50602,
        '{}', None, None,
    )
    # fetchall: timeframe rows (10 columns each)
    cursor.fetchall.return_value = [
        (1, 1, "A", "2025-04-01", "2025-05-01", "2026-02-01", 505000, 70.5, 29.5, -0.01),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/runs/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert data["run_label"] == "baseline"
    assert data["model_id"] == "lgbm_cluster"
    assert data["status"] == "completed"
    assert data["accuracy_pct"] == 69.34
    assert data["wape"] == 30.66
    assert data["bias"] == -0.0132
    assert data["n_predictions"] == 2725140
    assert data["n_dfus"] == 50602
    assert data["feature_count"] == 37
    assert "timeframes" in data
    assert len(data["timeframes"]) == 1
    tf = data["timeframes"][0]
    assert tf["timeframe"] == "A"
    assert tf["accuracy_pct"] == 70.5
    assert tf["wape"] == 29.5
    assert tf["bias"] == -0.01
    assert tf["n_predictions"] == 505000


@pytest.mark.asyncio
async def test_get_run_not_found(mock_pool):
    """GET /lgbm-tuning/runs/999 returns 404 when run does not exist."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/runs/999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_run_returns_201(mock_pool):
    """POST /lgbm-tuning/runs returns 201 with new run_id."""
    pool, mock_conn, cursor = mock_pool
    cursor.fetchone.return_value = (1,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/runs",
                json={
                    "run_label": "v2_ts_features",
                    "model_id": "lgbm_cluster",
                    "params": {"learning_rate": 0.05},
                    "features": ["lag_1", "lag_2", "rolling_mean_3"],
                    "notes": "Added time-series features",
                },
            )
    assert resp.status_code == 201
    data = resp.json()
    assert data["run_id"] == 1
    mock_conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# PUT /lgbm-tuning/runs/{run_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_run_returns_200(mock_pool):
    """PUT /lgbm-tuning/runs/1 returns 200 with updated flag."""
    pool, mock_conn, cursor = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/lgbm-tuning/runs/1",
                json={
                    "status": "completed",
                    "accuracy_pct": 71.2,
                    "wape": 28.8,
                    "bias": -0.005,
                    "n_predictions": 2800000,
                    "n_dfus": 51000,
                },
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["updated"] is True
    assert data["run_id"] == 1
    mock_conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/compare
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_runs_returns_200(mock_pool):
    """GET /lgbm-tuning/compare returns delta metrics between two runs."""
    pool, _, cursor = mock_pool
    # Sequential fetchone calls:
    #   1) baseline run (13 cols: run_id..status + params + features + feature_count + metadata)
    #   2) candidate run (13 cols)
    #   3) existing comparison check (None = no existing)
    cursor.fetchone.side_effect = [
        (1, "baseline", "lgbm_cluster", 69.34, 30.66, -0.0132, 2725140, 50602, "completed",
         '{"learning_rate": 0.02, "num_leaves": 63}',
         '["lag_1", "lag_2", "rolling_mean_3m"]', 3,
         '{"cluster_strategy": "per_cluster", "recursive": true}'),
        (2, "v2_ts_features", "lgbm_cluster", 71.50, 28.50, -0.005, 2800000, 51000, "completed",
         '{"learning_rate": 0.05, "num_leaves": 63}',
         '["lag_1", "lag_2", "rolling_mean_3m", "cv_demand"]', 4,
         '{"cluster_strategy": "per_cluster", "recursive": true}'),
        None,
    ]
    # fetchall is called 4 times: baseline clusters, candidate clusters, baseline months, candidate months
    cursor.fetchall.side_effect = [
        [("ml_cluster", "0", 1000, 50, 70.0, 30.0, -0.01)],
        [("ml_cluster", "0", 1000, 50, 72.0, 28.0, -0.005)],
        [("2025-05-01", 500, 25, 71.0, 29.0, -0.01)],
        [("2025-05-01", 500, 25, 73.0, 27.0, -0.005)],
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/compare?baseline_id=1&candidate_id=2")
    assert resp.status_code == 200
    data = resp.json()
    assert "baseline" in data
    assert "candidate" in data
    assert data["baseline"]["run_id"] == 1
    assert data["candidate"]["run_id"] == 2
    assert data["delta_accuracy"] == 2.16
    assert data["delta_wape"] == -2.16
    assert data["verdict"] == "improved"
    assert data["existing_comparison_id"] is None
    # Parameter diffs — learning_rate differs
    assert "param_diffs" in data
    assert len(data["param_diffs"]) == 1
    assert data["param_diffs"][0]["param"] == "learning_rate"
    assert data["param_diffs"][0]["baseline"] == 0.02
    assert data["param_diffs"][0]["candidate"] == 0.05
    # Parameter common — num_leaves is same in both
    assert "param_common" in data
    assert len(data["param_common"]) == 1
    assert data["param_common"][0]["param"] == "num_leaves"
    assert data["param_common"][0]["value"] == 63
    # Feature diffs — candidate added cv_demand
    assert "feature_diffs" in data
    assert data["feature_diffs"]["baseline_count"] == 3
    assert data["feature_diffs"]["candidate_count"] == 4
    assert data["feature_diffs"]["added"] == ["cv_demand"]
    assert data["feature_diffs"]["removed"] == []
    assert data["feature_diffs"]["common_count"] == 3
    # Config diffs — both same config, so no diffs
    assert "config_diffs" in data
    assert len(data["config_diffs"]) == 0
    assert "config_common" in data
    assert len(data["config_common"]) == 2  # cluster_strategy + recursive
    # Cluster and month data
    assert "per_cluster" in data
    assert "per_month" in data
    assert len(data["per_cluster"]["ml_cluster"]) == 1
    assert data["per_cluster"]["ml_cluster"][0]["cluster"] == "0"
    assert data["per_cluster"]["ml_cluster"][0]["delta_accuracy"] == 2.0
    assert len(data["per_month"]) == 1
    assert data["per_month"][0]["delta_accuracy"] == 2.0
    # Breakdown availability flags
    assert data["baseline_has_breakdowns"] is True
    assert data["candidate_has_breakdowns"] is True


@pytest.mark.asyncio
async def test_get_run_clusters_returns_200(mock_pool):
    """GET /lgbm-tuning/runs/1/clusters returns per-cluster accuracy."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("ml_cluster", "0", 1000, 50, 70.0, 30.0, -0.01),
        ("ml_cluster", "1", 800, 40, 68.0, 32.0, -0.02),
        ("business_cluster", "seasonal", 500, 25, 72.0, 28.0, 0.01),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/runs/1/clusters")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert len(data["clusters"]["ml_cluster"]) == 2
    assert len(data["clusters"]["business_cluster"]) == 1
    assert data["clusters"]["ml_cluster"][0]["cluster_value"] == "0"
    assert data["clusters"]["ml_cluster"][0]["accuracy_pct"] == 70.0


@pytest.mark.asyncio
async def test_get_run_months_returns_200(mock_pool):
    """GET /lgbm-tuning/runs/1/months returns per-month accuracy."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("2025-05-01", 500, 25, 71.0, 29.0, -0.01),
        ("2025-06-01", 480, 25, 69.0, 31.0, -0.02),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/runs/1/months")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 1
    assert len(data["months"]) == 2
    assert data["months"][0]["month_start"] == "2025-05-01"
    assert data["months"][0]["accuracy_pct"] == 71.0
    assert data["months"][1]["month_start"] == "2025-06-01"


@pytest.mark.asyncio
async def test_compare_runs_missing_candidate_breakdowns(mock_pool):
    """GET /lgbm-tuning/compare flags missing candidate breakdown data."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (1, "baseline", "lgbm_cluster", 69.34, 30.66, -0.0132, 2725140, 50602, "completed",
         '{"learning_rate": 0.02}', '["lag_1"]', 1, '{}'),
        (5, "v10_lr_schedule", "lgbm_cluster", 71.0, 29.0, -0.005, 2800000, 51000, "completed",
         '{"learning_rate": 0.05}', '["lag_1"]', 1, '{}'),
        None,
    ]
    # baseline has cluster+month data; candidate has none (empty lists)
    cursor.fetchall.side_effect = [
        [("ml_cluster", "0", 1000, 50, 70.0, 30.0, -0.01)],  # baseline clusters
        [],  # candidate clusters (empty)
        [("2025-05-01", 500, 25, 71.0, 29.0, -0.01)],  # baseline months
        [],  # candidate months (empty)
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/compare?baseline_id=1&candidate_id=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["baseline_has_breakdowns"] is True
    assert data["candidate_has_breakdowns"] is False
    # per_cluster and per_month still have rows from baseline (candidate cols are null)
    assert len(data["per_cluster"]["ml_cluster"]) == 1
    assert data["per_cluster"]["ml_cluster"][0]["baseline_accuracy"] == 70.0
    assert data["per_cluster"]["ml_cluster"][0]["candidate_accuracy"] is None
    assert len(data["per_month"]) == 1
    assert data["per_month"][0]["baseline_accuracy"] == 71.0
    assert data["per_month"][0]["candidate_accuracy"] is None


@pytest.mark.asyncio
async def test_compare_runs_baseline_not_found(mock_pool):
    """GET /lgbm-tuning/compare returns 404 when baseline run does not exist."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/compare?baseline_id=999&candidate_id=2")
    assert resp.status_code == 404
    assert "baseline" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/comparisons
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_comparisons_returns_200(mock_pool):
    """GET /lgbm-tuning/comparisons returns array of saved comparisons."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        (1, 1, 2, "2026-03-22T12:00:00", 2.16, -2.16, 0.0082, "improved", "baseline", "v2_ts_features"),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/comparisons")
    assert resp.status_code == 200
    data = resp.json()
    assert "comparisons" in data
    assert len(data["comparisons"]) == 1
    comp = data["comparisons"][0]
    assert comp["id"] == 1
    assert comp["baseline_run_id"] == 1
    assert comp["candidate_run_id"] == 2
    assert comp["delta_accuracy"] == 2.16
    assert comp["delta_wape"] == -2.16
    assert comp["verdict"] == "improved"
    assert comp["baseline_label"] == "baseline"
    assert comp["candidate_label"] == "v2_ts_features"


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/runs/{run_id}/promote
# ---------------------------------------------------------------------------

_FAKE_CONFIG = {
    "algorithms": {
        "lgbm": {
            "n_estimators": 1500,
            "learning_rate": 0.02,
            "num_leaves": 63,
        },
    },
}


@pytest.fixture
def _mock_yaml(monkeypatch):
    """Patch yaml load/dump and open for the promote endpoint."""
    import yaml as _yaml

    written = {}

    def _fake_dump(data, stream, **_kw):
        written["cfg"] = data

    monkeypatch.setattr(_yaml, "safe_load", lambda _f: dict(_FAKE_CONFIG))
    monkeypatch.setattr(_yaml, "dump", _fake_dump)
    return written


@pytest.mark.asyncio
async def test_promote_run_returns_200(mock_pool, _mock_yaml):
    """POST /lgbm-tuning/runs/{id}/promote promotes a completed run."""
    pool, _, cursor = mock_pool
    params = json.dumps({"num_leaves": 127, "min_child_samples": 40, "learning_rate": 0.02})
    # fetchone for the run lookup
    cursor.fetchone.return_value = (17, "wider_leaves_128", "completed", params, 72.40, None)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open()),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/lgbm-tuning/runs/17/promote")
    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is True
    assert data["run_id"] == 17
    assert data["run_label"] == "wider_leaves_128"
    assert data["accuracy_pct"] == 72.40
    assert "num_leaves" in data["params_written"]
    assert "min_child_samples" in data["params_written"]


@pytest.mark.asyncio
async def test_promote_run_not_found(mock_pool):
    """POST /lgbm-tuning/runs/{id}/promote returns 404 for unknown run."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/lgbm-tuning/runs/999/promote")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_promote_run_not_completed(mock_pool):
    """POST /lgbm-tuning/runs/{id}/promote rejects non-completed runs."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (11, "some_run", "running", None, None, None)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/lgbm-tuning/runs/11/promote")
    assert resp.status_code == 400
    assert "completed" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_promote_run_no_params(mock_pool):
    """POST /lgbm-tuning/runs/{id}/promote rejects runs with no params."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (11, "some_run", "completed", None, 70.0, None)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/lgbm-tuning/runs/11/promote")
    assert resp.status_code == 400
    assert "no params" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/promoted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_promoted_returns_run(mock_pool):
    """GET /lgbm-tuning/promoted returns the promoted run."""
    pool, _, cursor = mock_pool
    params = {"num_leaves": 127, "learning_rate": 0.02}
    cursor.fetchone.return_value = (17, "wider_leaves_128", "lgbm_cluster", 72.40, 27.60, -0.005, "2026-03-23T08:00:00", params)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/promoted")
    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is not None
    assert data["promoted"]["run_id"] == 17
    assert data["promoted"]["accuracy_pct"] == 72.40
    assert data["promoted"]["params"]["num_leaves"] == 127


@pytest.mark.asyncio
async def test_get_promoted_returns_null(mock_pool):
    """GET /lgbm-tuning/promoted returns null when no run is promoted."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/promoted")
    assert resp.status_code == 200
    assert resp.json()["promoted"] is None
