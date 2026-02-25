"""Tests for clustering scenario API endpoints (Feature 29)."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


@pytest.fixture
def mock_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = mock_conn

    return pool, mock_conn, mock_cursor


@pytest.mark.asyncio
async def test_clustering_defaults(mock_pool):
    """Verify /clustering/defaults returns structured params."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/clustering/defaults")
            assert response.status_code == 200
            data = response.json()
            assert "feature_params" in data
            assert "model_params" in data
            assert "label_params" in data
            assert "time_window_months" in data["feature_params"]
            assert "k_range" in data["model_params"]
            assert "volume_high" in data["label_params"]


@pytest.mark.asyncio
async def test_clustering_defaults_has_valid_ranges(mock_pool):
    """Verify default values are within expected ranges."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/clustering/defaults")
            data = response.json()
            mp = data["model_params"]
            lp = data["label_params"]
            assert len(mp["k_range"]) == 2
            assert mp["k_range"][0] < mp["k_range"][1]
            assert 0 < lp["volume_low"] < lp["volume_high"] < 1
            assert 0 < lp["cv_steady"] < lp["cv_volatile"]


@pytest.mark.asyncio
async def test_clustering_scenario_post(mock_pool):
    """Verify POST /clustering/scenario accepts params and runs scenario."""
    pool, _, _ = mock_pool
    mock_result = {
        "scenario_id": "sc_test_123",
        "status": "completed",
        "runtime_seconds": 1.0,
        "params": {},
        "result": {
            "optimal_k": 5,
            "silhouette_score": 0.4,
            "inertia": 1000,
            "n_clusters": 5,
            "total_dfus": 100,
            "cluster_sizes": {"0": 20, "1": 30, "2": 25, "3": 15, "4": 10},
            "k_selection_results": {
                "k_values": [3, 4, 5],
                "inertias": [2000, 1500, 1000],
                "silhouette_scores": [0.3, 0.35, 0.4],
                "gap_stats": None,
            },
            "profiles": [],
            "feature_importance": [],
        },
    }
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.run_clustering_scenario.run_scenario", return_value=mock_result):
        from api.main import app
        # Reset the running state
        import api.routers.clusters as clusters_module
        clusters_module._scenario_running = False
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/clustering/scenario",
                json={
                    "model_params": {"k_range": [3, 5], "skip_gap": True},
                    "label_params": {"volume_high": 0.8},
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["result"]["optimal_k"] == 5


@pytest.mark.asyncio
async def test_clustering_scenario_get_not_found(mock_pool):
    """Verify GET /clustering/scenario/<id> returns 404 for unknown ID."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.run_clustering_scenario.get_scenario_result", return_value=None):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/clustering/scenario/nonexistent_id")
            assert response.status_code == 404


@pytest.mark.asyncio
async def test_clustering_scenario_get_found(mock_pool):
    """Verify GET /clustering/scenario/<id> returns saved result."""
    pool, _, _ = mock_pool
    saved = {"scenario_id": "sc_test", "status": "completed", "result": {"optimal_k": 5}}
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.run_clustering_scenario.get_scenario_result", return_value=saved):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/clustering/scenario/sc_test")
            assert response.status_code == 200
            data = response.json()
            assert data["scenario_id"] == "sc_test"


@pytest.mark.asyncio
async def test_promote_scenario_not_found(mock_pool):
    """Verify promote returns 404 for unknown scenario."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.run_clustering_scenario.promote_scenario", side_effect=FileNotFoundError("not found")):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/clustering/scenario/nonexistent/promote")
            assert response.status_code == 404


@pytest.mark.asyncio
async def test_promote_scenario_success(mock_pool):
    """Verify promote returns success with dfus_updated count."""
    pool, _, _ = mock_pool
    result = {"status": "promoted", "scenario_id": "sc_test", "dfus_updated": 100, "cluster_distribution": {"a": 50, "b": 50}}
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.run_clustering_scenario.promote_scenario", return_value=result):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/clustering/scenario/sc_test/promote")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "promoted"
            assert data["dfus_updated"] == 100
