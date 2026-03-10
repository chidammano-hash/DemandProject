"""Tests for job scheduler API endpoints (Feature 39)."""

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


@pytest.fixture
def mock_manager():
    """Mock the JobManager singleton used by the jobs router."""
    mgr = MagicMock()
    mgr.get_types.return_value = [
        {
            "type_id": "cluster_scenario",
            "label": "Clustering What-If",
            "description": "Run a trial clustering pipeline",
            "group": "clustering",
            "params_schema": {},
        },
        {
            "type_id": "backtest_lgbm",
            "label": "LGBM Backtest",
            "description": "Run LightGBM backtest",
            "group": "backtest",
            "params_schema": {"cluster_strategy": "global"},
        },
    ]
    return mgr


@pytest.mark.asyncio
async def test_list_job_types(mock_pool, mock_manager):
    """GET /jobs/types returns available job types."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/jobs/types")
            assert response.status_code == 200
            data = response.json()
            assert "types" in data
            assert len(data["types"]) == 2
            assert data["types"][0]["type_id"] == "cluster_scenario"


@pytest.mark.asyncio
async def test_submit_job_returns_202(mock_pool, mock_manager):
    """POST /jobs returns 202 with job_id."""
    pool, _, _ = mock_pool
    mock_manager.submit_job.return_value = "job_20260227_120000_abc12345"
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/jobs",
                json={"job_type": "cluster_scenario", "params": {}, "label": "Test Job"},
            )
            assert response.status_code == 202
            data = response.json()
            assert data["job_id"] == "job_20260227_120000_abc12345"
            assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_submit_job_invalid_type(mock_pool, mock_manager):
    """POST /jobs with invalid type returns 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        # Remove the fake type from registry so validation fails
        with patch("common.job_registry.JOB_TYPE_REGISTRY", {}):
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/jobs",
                    json={"job_type": "nonexistent_type", "params": {}},
                )
                assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_job_queued_when_busy(mock_pool, mock_manager):
    """POST /jobs returns 202 with status=queued when group is busy (no 409)."""
    pool, _, _ = mock_pool
    # submit_job no longer raises RuntimeError — it queues instead
    mock_manager.submit_job.return_value = "job_20260227_120000_queued1"
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/jobs",
                json={"job_type": "cluster_scenario", "params": {}, "label": "Queued Job"},
            )
            assert response.status_code == 202
            data = response.json()
            assert data["job_id"] == "job_20260227_120000_queued1"
            assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_list_jobs(mock_pool, mock_manager):
    """GET /jobs returns paginated job list."""
    pool, _, _ = mock_pool
    mock_manager.list_jobs.return_value = (
        [{"job_id": "j1", "status": "completed", "job_type": "backtest_lgbm"}],
        1,
    )
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/jobs", params={"status": "completed", "limit": 10})
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["jobs"]) == 1


@pytest.mark.asyncio
async def test_get_job_detail_not_found(mock_pool, mock_manager):
    """GET /jobs/{id} returns 404 for unknown job."""
    pool, _, _ = mock_pool
    mock_manager.get_status.return_value = None
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/jobs/nonexistent")
            assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_job_detail_found(mock_pool, mock_manager):
    """GET /jobs/{id} returns full job record."""
    pool, _, _ = mock_pool
    mock_manager.get_status.return_value = {
        "job_id": "j1",
        "job_type": "cluster_scenario",
        "job_label": "Test",
        "status": "completed",
        "params": {},
        "result": {"optimal_k": 5},
        "error": None,
        "submitted_at": "2026-02-27T00:00:00",
        "started_at": "2026-02-27T00:00:01",
        "completed_at": "2026-02-27T00:00:10",
        "progress_pct": 100,
        "progress_msg": "Done",
    }
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/jobs/j1")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "j1"
            assert data["status"] == "completed"
            assert data["result"]["optimal_k"] == 5


@pytest.mark.asyncio
async def test_cancel_job(mock_pool, mock_manager):
    """POST /jobs/{id}/cancel marks job as cancelled."""
    pool, _, _ = mock_pool
    mock_manager.cancel_job.return_value = True
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/jobs/j1/cancel")
            assert response.status_code == 200
            assert response.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_job_not_found(mock_pool, mock_manager):
    """POST /jobs/{id}/cancel returns 404 for non-cancellable job."""
    pool, _, _ = mock_pool
    mock_manager.cancel_job.return_value = False
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/jobs/nonexistent/cancel")
            assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_job(mock_pool, mock_manager):
    """DELETE /jobs/{id} removes completed job."""
    pool, _, _ = mock_pool
    mock_manager.delete_job.return_value = True
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/jobs/j1")
            assert response.status_code == 200
            assert response.json()["deleted"] is True


@pytest.mark.asyncio
async def test_active_jobs(mock_pool, mock_manager):
    """GET /jobs/active returns running/queued jobs only."""
    pool, _, _ = mock_pool
    mock_manager.get_active_jobs.return_value = [
        {"job_id": "j_running", "status": "running", "job_type": "backtest_lgbm"},
    ]
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/jobs/active")
            assert response.status_code == 200
            data = response.json()
            assert len(data["jobs"]) == 1
            assert data["jobs"][0]["status"] == "running"


@pytest.mark.asyncio
async def test_job_stats(mock_pool, mock_manager):
    """GET /jobs/stats returns aggregate statistics."""
    pool, _, _ = mock_pool
    mock_manager.get_stats.return_value = {
        "total": 47, "active": 2, "completed": 42, "failed": 3, "cancelled": 0,
        "avg_duration_seconds": 245.6,
        "last_24h": {"submitted": 5, "completed": 4, "failed": 1},
    }
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/jobs/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 47
            assert data["active"] == 2
            assert data["avg_duration_seconds"] == 245.6
            assert data["last_24h"]["submitted"] == 5


@pytest.mark.asyncio
async def test_schedule_recurring_job(mock_pool, mock_manager):
    """POST /jobs/schedule creates a recurring schedule."""
    pool, _, _ = mock_pool
    mock_manager.schedule_recurring.return_value = "sched_abc123"
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager), \
         patch("common.job_registry.JOB_TYPE_REGISTRY", {"backtest_lgbm": MagicMock()}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/jobs/schedule",
                json={"job_type": "backtest_lgbm", "cron": "0 2 * * *", "label": "Nightly LGBM"},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["schedule_id"] == "sched_abc123"
            assert data["status"] == "active"


@pytest.mark.asyncio
async def test_schedule_missing_trigger(mock_pool, mock_manager):
    """POST /jobs/schedule returns 422 when no cron or interval."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager), \
         patch("common.job_registry.JOB_TYPE_REGISTRY", {"backtest_lgbm": MagicMock()}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/jobs/schedule",
                json={"job_type": "backtest_lgbm"},
            )
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_schedules(mock_pool, mock_manager):
    """GET /jobs/schedules returns active schedules."""
    pool, _, _ = mock_pool
    mock_manager.list_schedules.return_value = [
        {"schedule_id": "s1", "job_type": "backtest_lgbm", "cron_expr": "0 2 * * *"},
    ]
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/jobs/schedules")
            assert response.status_code == 200
            data = response.json()
            assert len(data["schedules"]) == 1


@pytest.mark.asyncio
async def test_submit_pipeline(mock_pool, mock_manager):
    """POST /jobs/pipeline submits a chained job pipeline."""
    pool, _, _ = mock_pool
    mock_manager.submit_pipeline.return_value = "pipe_xyz789"
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager), \
         patch("common.job_registry.JOB_TYPE_REGISTRY", {
             "cluster_pipeline": MagicMock(),
             "backtest_lgbm": MagicMock(),
         }):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/jobs/pipeline",
                json={
                    "steps": [
                        {"job_type": "cluster_pipeline", "label": "Re-cluster"},
                        {"job_type": "backtest_lgbm", "label": "Re-test"},
                    ],
                    "label": "Full Refresh",
                },
            )
            assert response.status_code == 202
            data = response.json()
            assert data["pipeline_id"] == "pipe_xyz789"
            assert data["steps"] == 2


@pytest.mark.asyncio
async def test_new_job_types_registered(mock_pool, mock_manager):
    """All 22 job types (10 original + 12 new) are in JOB_TYPE_REGISTRY."""
    from common.job_registry import JOB_TYPE_REGISTRY
    type_ids = set(JOB_TYPE_REGISTRY.keys())
    # New inventory group types
    for expected in [
        "compute_safety_stock", "compute_eoq", "assign_policies",
        "generate_exceptions", "classify_abc_xyz", "compute_variability",
        "compute_demand_signals", "compute_investment",
        "refresh_health_scores", "refresh_intramonth", "run_ss_simulation",
    ]:
        assert expected in type_ids, f"Missing job type: {expected}"
    # New ai group type
    assert "generate_storyboard" in type_ids
    # Verify total count >= 22
    assert len(type_ids) >= 22


@pytest.mark.asyncio
async def test_pipeline_with_inventory_steps(mock_pool, mock_manager):
    """POST /jobs/pipeline accepts inventory group job types."""
    pool, _, _ = mock_pool
    mock_manager.submit_pipeline.return_value = "pipe_inv_001"
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.jobs._get_manager", return_value=mock_manager), \
         patch("common.job_registry.JOB_TYPE_REGISTRY", {
             "compute_safety_stock": MagicMock(),
             "compute_eoq": MagicMock(),
             "refresh_health_scores": MagicMock(),
         }):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/jobs/pipeline",
                json={
                    "steps": [
                        {"job_type": "compute_safety_stock", "label": "SS"},
                        {"job_type": "compute_eoq", "label": "EOQ"},
                        {"job_type": "refresh_health_scores", "label": "Health"},
                    ],
                    "label": "Inventory Refresh",
                },
            )
    assert response.status_code == 202
    data = response.json()
    assert data["pipeline_id"] == "pipe_inv_001"
    assert data["steps"] == 3
