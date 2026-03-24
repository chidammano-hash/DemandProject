"""Tests for CatBoost/XGBoost model tuning API endpoints."""

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
# CatBoost: GET /catboost-tuning/runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catboost_list_runs_returns_200(mock_pool):
    """GET /catboost-tuning/runs returns 200 with runs array."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        (18, "cb_baseline", "catboost_cluster", "2026-03-20T10:00:00", "2026-03-20T11:00:00",
         "completed", 66.82, 33.18, 0.0412, 580000, 3200, None, False, None),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert len(data["runs"]) == 1
    run = data["runs"][0]
    assert run["run_id"] == 18
    assert run["run_label"] == "cb_baseline"
    assert run["model_id"] == "catboost_cluster"
    assert run["accuracy_pct"] == 66.82


@pytest.mark.asyncio
async def test_catboost_list_runs_empty(mock_pool):
    """GET /catboost-tuning/runs returns empty array when no runs exist."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/runs")
    assert resp.status_code == 200
    assert resp.json()["runs"] == []


# ---------------------------------------------------------------------------
# XGBoost: GET /xgboost-tuning/runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_xgboost_list_runs_returns_200(mock_pool):
    """GET /xgboost-tuning/runs returns 200 with runs array."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        (23, "xgb_baseline", "xgboost_cluster", "2026-03-21T10:00:00", "2026-03-21T11:00:00",
         "completed", 65.47, 34.53, 0.0523, 580000, 3200, None, False, None),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/xgboost-tuning/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 1
    assert data["runs"][0]["model_id"] == "xgboost_cluster"
    assert data["runs"][0]["accuracy_pct"] == 65.47


# ---------------------------------------------------------------------------
# CatBoost: GET /catboost-tuning/runs/{run_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catboost_get_run_returns_200(mock_pool):
    """GET /catboost-tuning/runs/18 returns run detail with timeframes."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (
        18, "cb_baseline", "catboost_cluster",
        "2026-03-20T10:00:00", "2026-03-20T11:00:00",
        "completed", '{"iterations": 500}', 17, '["ml_cluster", "lag_1"]',
        66.82, 33.18, 0.0412, 580000, 3200,
        '{}', None, None,
    )
    cursor.fetchall.return_value = [
        (1, 18, "A", "2025-04-01", "2025-05-01", "2026-02-01", 58000, 65.5, 34.5, 0.05),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/runs/18")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 18
    assert data["model_id"] == "catboost_cluster"
    assert data["accuracy_pct"] == 66.82
    assert len(data["timeframes"]) == 1
    assert data["timeframes"][0]["timeframe"] == "A"


@pytest.mark.asyncio
async def test_catboost_get_run_not_found(mock_pool):
    """GET /catboost-tuning/runs/999 returns 404."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/runs/999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# CatBoost: POST /catboost-tuning/runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catboost_create_run_returns_201(mock_pool):
    """POST /catboost-tuning/runs returns 201 with new run_id."""
    pool, mock_conn, cursor = mock_pool
    cursor.fetchone.return_value = (18,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/catboost-tuning/runs",
                json={
                    "run_label": "cb_v2",
                    "params": {"iterations": 800, "learning_rate": 0.03},
                    "features": ["lag_1", "lag_2"],
                },
            )
    assert resp.status_code == 201
    assert resp.json()["run_id"] == 18
    mock_conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# XGBoost: POST /xgboost-tuning/runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_xgboost_create_run_returns_201(mock_pool):
    """POST /xgboost-tuning/runs returns 201 with new run_id."""
    pool, mock_conn, cursor = mock_pool
    cursor.fetchone.return_value = (23,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/xgboost-tuning/runs",
                json={
                    "run_label": "xgb_v2",
                    "params": {"n_estimators": 800, "learning_rate": 0.03},
                },
            )
    assert resp.status_code == 201
    assert resp.json()["run_id"] == 23
    mock_conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# CatBoost: PUT /catboost-tuning/runs/{run_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catboost_update_run_returns_200(mock_pool):
    """PUT /catboost-tuning/runs/18 returns 200."""
    pool, mock_conn, cursor = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/catboost-tuning/runs/18",
                json={"status": "completed", "accuracy_pct": 68.5},
            )
    assert resp.status_code == 200
    assert resp.json()["updated"] is True


# ---------------------------------------------------------------------------
# CatBoost: GET /catboost-tuning/compare
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catboost_compare_runs_returns_200(mock_pool):
    """GET /catboost-tuning/compare returns delta metrics."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (18, "cb_baseline", "catboost_cluster", 66.82, 33.18, 0.0412, 580000, 3200, "completed",
         '{"iterations": 500, "learning_rate": 0.05}', '["lag_1"]', 1, '{}'),
        (19, "cb_deeper_trees", "catboost_cluster", 67.95, 32.05, 0.0318, 580000, 3200, "completed",
         '{"iterations": 800, "learning_rate": 0.03}', '["lag_1"]', 1, '{}'),
        None,
    ]
    cursor.fetchall.side_effect = [[], [], [], []]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/compare?baseline_id=18&candidate_id=19")
    assert resp.status_code == 200
    data = resp.json()
    assert data["baseline"]["run_id"] == 18
    assert data["candidate"]["run_id"] == 19
    assert data["delta_accuracy"] == 1.13
    assert data["verdict"] == "improved"
    assert len(data["param_diffs"]) == 2  # iterations + learning_rate


# ---------------------------------------------------------------------------
# XGBoost: GET /xgboost-tuning/compare
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_xgboost_compare_runs_returns_200(mock_pool):
    """GET /xgboost-tuning/compare returns delta metrics."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (23, "xgb_baseline", "xgboost_cluster", 65.47, 34.53, 0.0523, 580000, 3200, "completed",
         '{"n_estimators": 500}', '["lag_1"]', 1, '{}'),
        (24, "xgb_more_trees", "xgboost_cluster", 66.89, 33.11, 0.0421, 580000, 3200, "completed",
         '{"n_estimators": 800}', '["lag_1"]', 1, '{}'),
        None,
    ]
    cursor.fetchall.side_effect = [[], [], [], []]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/xgboost-tuning/compare?baseline_id=23&candidate_id=24")
    assert resp.status_code == 200
    data = resp.json()
    assert data["delta_accuracy"] == 1.42
    assert data["verdict"] == "improved"


# ---------------------------------------------------------------------------
# CatBoost: GET clusters/months
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catboost_get_clusters_returns_200(mock_pool):
    """GET /catboost-tuning/runs/18/clusters returns per-cluster accuracy."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("ml_cluster", "0", 1000, 50, 68.0, 32.0, 0.02),
        ("ml_cluster", "1", 800, 40, 65.0, 35.0, 0.04),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/runs/18/clusters")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 18
    assert len(data["clusters"]["ml_cluster"]) == 2


@pytest.mark.asyncio
async def test_catboost_get_months_returns_200(mock_pool):
    """GET /catboost-tuning/runs/18/months returns per-month accuracy."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("2025-05-01", 500, 25, 67.0, 33.0, 0.03),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/runs/18/months")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 18
    assert len(data["months"]) == 1


# ---------------------------------------------------------------------------
# CatBoost: promote
# ---------------------------------------------------------------------------

_FAKE_CONFIG = {
    "algorithms": {
        "catboost": {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
        },
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
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
async def test_catboost_promote_run_returns_200(mock_pool, _mock_yaml):
    """POST /catboost-tuning/runs/22/promote promotes a completed CatBoost run."""
    pool, _, cursor = mock_pool
    params = json.dumps({"iterations": 1500, "learning_rate": 0.018, "depth": 7})
    cursor.fetchone.return_value = (22, "cb_best_combo_v1", "completed", params, 70.12, None)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open()),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/catboost-tuning/runs/22/promote")
    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is True
    assert data["run_id"] == 22
    assert "iterations" in data["params_written"]
    assert "depth" in data["params_written"]


@pytest.mark.asyncio
async def test_xgboost_promote_run_returns_200(mock_pool, _mock_yaml):
    """POST /xgboost-tuning/runs/27/promote promotes a completed XGBoost run."""
    pool, _, cursor = mock_pool
    params = json.dumps({"n_estimators": 1500, "learning_rate": 0.018, "max_depth": 6})
    cursor.fetchone.return_value = (27, "xgb_best_combo_v1", "completed", params, 69.28, None)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open()),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/xgboost-tuning/runs/27/promote")
    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is True
    assert data["run_id"] == 27
    assert "n_estimators" in data["params_written"]
    assert "max_depth" in data["params_written"]


@pytest.mark.asyncio
async def test_catboost_promote_not_found(mock_pool):
    """POST /catboost-tuning/runs/999/promote returns 404."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/catboost-tuning/runs/999/promote")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_xgboost_promote_not_completed(mock_pool):
    """POST /xgboost-tuning/runs/23/promote rejects non-completed runs."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (23, "xgb_baseline", "running", None, None, None)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/xgboost-tuning/runs/23/promote")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_catboost_promote_includes_phase3_params(mock_pool, _mock_yaml):
    """Phase 3 params (bootstrap_type, max_leaves, etc.) are written during promote."""
    pool, _, cursor = mock_pool
    params = json.dumps({
        "iterations": 3000, "learning_rate": 0.008, "depth": 10,
        "l2_leaf_reg": 7.5, "max_leaves": 127, "bootstrap_type": "MVS",
        "model_size_reg": 0.08, "boost_from_average": True,
    })
    cursor.fetchone.return_value = (37, "cb_champion_v2", "completed", params, 73.12, None)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open()),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/catboost-tuning/runs/37/promote")
    assert resp.status_code == 200
    data = resp.json()
    written = data["params_written"]
    assert "max_leaves" in written
    assert "bootstrap_type" in written
    assert "model_size_reg" in written
    assert "boost_from_average" in written


@pytest.mark.asyncio
async def test_xgboost_promote_includes_phase3_params(mock_pool, _mock_yaml):
    """Phase 3 params (booster, rate_drop, max_leaves, etc.) are written during promote."""
    pool, _, cursor = mock_pool
    params = json.dumps({
        "n_estimators": 2800, "learning_rate": 0.009, "max_depth": 10,
        "max_leaves": 127, "booster": "dart", "rate_drop": 0.05, "skip_drop": 0.60,
    })
    cursor.fetchone.return_value = (47, "xgb_champion_v2", "completed", params, 72.31, None)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("builtins.open", mock_open()),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/xgboost-tuning/runs/47/promote")
    assert resp.status_code == 200
    data = resp.json()
    written = data["params_written"]
    assert "max_leaves" in written
    assert "booster" in written
    assert "rate_drop" in written
    assert "skip_drop" in written


# ---------------------------------------------------------------------------
# CatBoost/XGBoost: GET promoted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catboost_get_promoted_returns_run(mock_pool):
    """GET /catboost-tuning/promoted returns the promoted run."""
    pool, _, cursor = mock_pool
    params = {"iterations": 1500, "learning_rate": 0.018}
    cursor.fetchone.return_value = (
        22, "cb_best_combo_v1", "catboost_cluster", 70.12, 29.88, 0.0142,
        "2026-03-23T08:00:00", params,
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/promoted")
    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is not None
    assert data["promoted"]["run_id"] == 22
    assert data["promoted"]["accuracy_pct"] == 70.12


@pytest.mark.asyncio
async def test_xgboost_get_promoted_returns_null(mock_pool):
    """GET /xgboost-tuning/promoted returns null when no run is promoted."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/xgboost-tuning/promoted")
    assert resp.status_code == 200
    assert resp.json()["promoted"] is None


# ---------------------------------------------------------------------------
# CatBoost/XGBoost: GET comparisons
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catboost_list_comparisons_returns_200(mock_pool):
    """GET /catboost-tuning/comparisons returns saved comparisons."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        (1, 18, 19, "2026-03-22T12:00:00", 1.13, -1.13, -0.0094, "improved",
         "cb_baseline", "cb_deeper_trees"),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/catboost-tuning/comparisons")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["comparisons"]) == 1
    assert data["comparisons"][0]["verdict"] == "improved"
