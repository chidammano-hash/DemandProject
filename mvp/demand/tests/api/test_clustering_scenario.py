"""Tests for clustering scenario API endpoints (Features 29, 38)."""

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
    """Verify POST /clustering/scenario returns 202 with scenario_id (async)."""
    pool, _, _ = mock_pool
    mock_mgr = MagicMock()
    mock_mgr.submit_job.return_value = "job_test_abc"
    mock_mgr.get_status.return_value = {"status": "running"}
    mock_mgr.start_job_in_background = MagicMock()
    with patch("api.core._get_pool", return_value=pool), \
         patch("common.job_registry.JobManager", return_value=mock_mgr), \
         patch("scripts.run_clustering_scenario.run_scenario"), \
         patch("scripts.run_clustering_scenario.generate_scenario_id", return_value="sc_test_123"):
        from api.main import app
        import api.routers.clusters as clusters_module
        clusters_module._scenario_running = False
        clusters_module._running_scenario_id = None
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/clustering/scenario",
                json={
                    "model_params": {"k_range": [3, 5], "skip_gap": True},
                    "label_params": {"volume_high": 0.8},
                },
            )
            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "running"
            assert data["scenario_id"] == "sc_test_123"
            assert "job_id" in data


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


# ---------------------------------------------------------------------------
# Feature 38: Estimate + Status endpoints
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_estimate_returns_200(mock_pool):
    """Verify /clustering/scenario/estimate returns estimate fields."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (1200,)  # DFU count
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/clustering/scenario/estimate",
                params={"k_min": 3, "k_max": 10, "skip_gap": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "estimated_seconds" in data
            assert data["dfu_count"] == 1200
            assert data["k_range"] == 8
            assert data["skip_gap"] is True
            assert data["estimated_seconds"] > 0


@pytest.mark.asyncio
async def test_estimate_with_gap(mock_pool):
    """Verify skip_gap=false increases estimate."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (100000,)  # Large enough for gap multiplier to be visible
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp_skip = await client.get(
                "/clustering/scenario/estimate",
                params={"k_min": 3, "k_max": 10, "skip_gap": "true"},
            )
            resp_gap = await client.get(
                "/clustering/scenario/estimate",
                params={"k_min": 3, "k_max": 10, "skip_gap": "false"},
            )
            assert resp_gap.json()["estimated_seconds"] > resp_skip.json()["estimated_seconds"]


@pytest.mark.asyncio
async def test_estimate_zero_dfus(mock_pool):
    """Verify estimate with zero DFUs returns overhead only."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/clustering/scenario/estimate",
                params={"k_min": 3, "k_max": 10, "skip_gap": "true"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["dfu_count"] == 0
            assert data["estimated_seconds"] >= 0


@pytest.mark.asyncio
async def test_scenario_status_running(mock_pool):
    """Verify status returns running with elapsed time."""
    pool, _, _ = mock_pool
    import api.routers.clusters as clusters_module
    import time as _time
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        # Simulate a running scenario
        clusters_module._scenario_running = True
        clusters_module._running_scenario_id = "sc_running"
        clusters_module._scenario_start_time = _time.time() - 10  # started 10s ago
        try:
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/clustering/scenario/sc_running/status")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "running"
                assert data["elapsed_seconds"] >= 9
        finally:
            clusters_module._scenario_running = False
            clusters_module._running_scenario_id = None
            clusters_module._scenario_start_time = None


@pytest.mark.asyncio
async def test_scenario_status_completed(mock_pool):
    """Verify status returns completed result."""
    pool, _, _ = mock_pool
    completed = {
        "scenario_id": "sc_done",
        "status": "completed",
        "runtime_seconds": 42.5,
        "result": {"optimal_k": 5},
    }
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.run_clustering_scenario.get_scenario_result", return_value=completed):
        from api.main import app
        import api.routers.clusters as clusters_module
        clusters_module._scenario_running = False
        clusters_module._running_scenario_id = None
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/clustering/scenario/sc_done/status")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["runtime_seconds"] == 42.5


@pytest.mark.asyncio
async def test_scenario_status_not_found(mock_pool):
    """Verify status returns 404 for unknown scenario."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.run_clustering_scenario.get_scenario_result", return_value=None):
        from api.main import app
        import api.routers.clusters as clusters_module
        clusters_module._scenario_running = False
        clusters_module._running_scenario_id = None
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/clustering/scenario/unknown_id/status")
            assert response.status_code == 404


@pytest.mark.asyncio
async def test_scenario_queued_when_busy(mock_pool):
    """Verify POST returns 202 with status=queued when group is busy (no 409)."""
    pool, _, _ = mock_pool
    mock_mgr = MagicMock()
    mock_mgr.submit_job.return_value = "job_queued_xyz"
    mock_mgr.get_status.return_value = {"status": "queued"}
    import api.routers.clusters as clusters_module
    with patch("api.core._get_pool", return_value=pool), \
         patch("common.job_registry.JobManager", return_value=mock_mgr), \
         patch("scripts.run_clustering_scenario.generate_scenario_id", return_value="sc_queued_123"):
        from api.main import app
        clusters_module._scenario_running = False
        clusters_module._running_scenario_id = None
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/clustering/scenario",
                json={},
            )
            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "queued"
            assert data["job_id"] == "job_queued_xyz"
