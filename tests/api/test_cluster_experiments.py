"""Tests for the cluster experiments API — /cluster-experiments/* endpoints.

Tests CRUD lifecycle, comparison, promotion, completed filter, used-by,
and delete protection for the Cluster Experimentation Studio.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Row builders matching the router's SELECT column order (22 columns)
# ---------------------------------------------------------------------------

def _experiment_row(
    experiment_id: int = 1,
    scenario_id: str = "sc_20260320_100000_a1b2",
    label: str = "Test Experiment",
    notes: str | None = None,
    template_id: str | None = "production_baseline",
    status: str = "completed",
    created_at: str = "2026-03-20T10:00:00+00:00",
    started_at: str = "2026-03-20T10:00:05+00:00",
    completed_at: str = "2026-03-20T10:05:00+00:00",
    runtime_seconds: float = 295.0,
    job_id: str | None = "job-123",
    feature_params: str | None = '{"time_window_months": 24, "min_months_history": 1}',
    model_params: str | None = '{"k_range": [3, 12], "min_cluster_size_pct": 2.0}',
    label_params: str | None = '{"volume_high": 0.75, "volume_low": 0.25}',
    optimal_k: int | None = 8,
    silhouette_score: float | None = 0.342,
    inertia: float | None = 150000.0,
    total_dfus: int | None = 12000,
    n_clusters: int | None = 8,
    cluster_sizes: str | None = '{"0": 3000, "1": 4000, "2": 5000}',
    profiles: str | None = '[{"label": "high_volume_steady", "count": 3000}]',
    k_selection_results: str | None = '{"k_values": [3,4,5], "inertias": [1000,800,600], "silhouette_scores": [0.3,0.35,0.32]}',
    is_promoted: bool = False,
    promoted_at: str | None = None,
    artifacts_path: str | None = "/tmp/clustering_scenarios/sc_test",
) -> tuple:
    """Build a mock cluster_experiment row (25 columns)."""
    return (
        experiment_id, scenario_id, label, notes, template_id,
        status, created_at, started_at, completed_at, runtime_seconds,
        job_id, feature_params, model_params, label_params,
        optimal_k, silhouette_score, inertia, total_dfus, n_clusters,
        cluster_sizes, profiles, k_selection_results,
        is_promoted, promoted_at, artifacts_path,
    )


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
            resp = await client.get("/cluster-experiments?offset=0&limit=50")

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
            resp = await client.get("/cluster-experiments?status=completed")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["experiments"][0]["status"] == "completed"


# ---------------------------------------------------------------------------
# 2. Get single experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _experiment_row(
        experiment_id=5, label="Detail Test", optimal_k=10,
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/5")

    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_id"] == 5
    assert data["label"] == "Detail Test"
    assert data["optimal_k"] == 10


@pytest.mark.asyncio
async def test_get_experiment_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/999")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 3. Create experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (42,)  # RETURNING experiment_id

    mock_gen = MagicMock(return_value="sc_20260325_120000_f1e2")
    mock_mgr = MagicMock()
    mock_mgr.submit_job.return_value = "job-abc"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.cluster_experiments.generate_scenario_id",
            mock_gen,
            create=True,
        ),
        patch("scripts.run_clustering_scenario.generate_scenario_id", mock_gen),
        patch("common.services.job_registry.JobManager", return_value=mock_mgr),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/cluster-experiments", json={
                "label": "High-K Test",
                "notes": "Testing K=15",
                "template": "high_k_granular",
                "feature_params": {"time_window_months": 36, "min_months_history": 12},
                "model_params": {"k_range": [10, 20], "min_cluster_size_pct": 1.5},
                "label_params": {"volume_high": 0.8, "volume_low": 0.2},
            })

    assert resp.status_code == 202
    data = resp.json()
    assert data["experiment_id"] == 42
    assert data["status"] == "queued"
    assert data["scenario_id"] == "sc_20260325_120000_f1e2"


@pytest.mark.asyncio
async def test_create_experiment_minimal():
    """Create experiment with only required label field."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)

    mock_gen = MagicMock(return_value="sc_20260325_120000_abcd")
    mock_mgr = MagicMock()
    mock_mgr.submit_job.return_value = "job-min"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("scripts.run_clustering_scenario.generate_scenario_id", mock_gen),
        patch("common.services.job_registry.JobManager", return_value=mock_mgr),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/cluster-experiments", json={
                "label": "Simple Test",
            })

    assert resp.status_code == 202
    data = resp.json()
    assert data["experiment_id"] == 1
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_create_experiment_missing_label():
    """Empty label should fail validation."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/cluster-experiments", json={
                "label": "",
            })

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. Update experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3,)  # RETURNING experiment_id

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch("/cluster-experiments/3", json={
                "label": "Updated Label",
                "notes": "Updated notes",
            })

    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_id"] == 3
    assert data["updated"] is True


@pytest.mark.asyncio
async def test_update_experiment_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch("/cluster-experiments/999", json={
                "label": "New Label",
            })

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_experiment_no_fields():
    """Sending no updatable fields should return 400."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch("/cluster-experiments/1", json={})

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 5. Delete experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("completed",)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/cluster-experiments/1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True


@pytest.mark.asyncio
async def test_delete_experiment_running_blocked():
    """Delete should return 409 for running experiments."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("running",)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/cluster-experiments/1")

    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_delete_experiment_queued_blocked():
    """Delete should return 409 for queued experiments."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("queued",)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/cluster-experiments/1")

    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_delete_experiment_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/cluster-experiments/999")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_experiment_failed_allowed():
    """Delete should succeed for failed experiments."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("failed",)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/cluster-experiments/2")

    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# 6. Promote experiment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_experiment():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _experiment_row(
        experiment_id=1, status="completed", scenario_id="sc_20260320_100000_a1b2",
    )

    mock_promote = MagicMock(return_value={"dfus_updated": 12000, "status": "promoted"})

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("scripts.run_clustering_scenario.promote_scenario", mock_promote),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/cluster-experiments/1/promote")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "promoted"
    assert data["dfus_updated"] == 12000
    mock_promote.assert_called_once_with("sc_20260320_100000_a1b2")


@pytest.mark.asyncio
async def test_promote_experiment_not_completed():
    """Promote should return 409 if experiment is not completed."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _experiment_row(
        experiment_id=1, status="running",
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/cluster-experiments/1/promote")

    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_promote_experiment_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/cluster-experiments/999/promote")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_promote_experiment_artifacts_missing():
    """Promote should return 404 if scenario artifacts are missing."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _experiment_row(
        experiment_id=1, status="completed", scenario_id="sc_20260320_100000_a1b2",
    )

    mock_promote = MagicMock(side_effect=FileNotFoundError("Not found"))

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("scripts.run_clustering_scenario.promote_scenario", mock_promote),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/cluster-experiments/1/promote")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 7. Compare experiments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_experiments_cached():
    """Compare should return cached result when available."""
    pool, conn, cursor = _make_pool()

    row_a = _experiment_row(experiment_id=1, label="Exp A", optimal_k=8, silhouette_score=0.35)
    row_b = _experiment_row(experiment_id=2, label="Exp B", optimal_k=10, silhouette_score=0.38)

    # First call: fetch both experiments; second call: check cache (hit)
    cursor.fetchall.return_value = [row_a, row_b]
    quality = {"silhouette_delta": 0.03, "k_delta": 2, "verdict": "b_better"}
    profile = {"clusters_only_in_a": [], "clusters_only_in_b": [], "common_clusters": []}
    migration = {"high_volume": {"high_volume": 3000}}
    cursor.fetchone.return_value = (
        json.dumps(migration), json.dumps(quality), json.dumps(profile),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/compare?a_id=1&b_id=2")

    assert resp.status_code == 200
    data = resp.json()
    assert "experiment_a" in data
    assert "experiment_b" in data
    assert data["quality_comparison"]["verdict"] == "b_better"


@pytest.mark.asyncio
async def test_compare_experiments_same_id():
    """Compare should return 400 when comparing same experiment."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/compare?a_id=1&b_id=1")

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_compare_experiments_not_found():
    """Compare should return 404 when an experiment is missing."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [_experiment_row(experiment_id=1)]  # Only one found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/compare?a_id=1&b_id=99")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 8. Templates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_templates():
    pool, conn, cursor = _make_pool()

    templates_yaml = {
        "templates": [
            {"id": "production_baseline", "label": "Production Baseline"},
            {"id": "high_k_granular", "label": "High-K Granular"},
        ]
    }

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="")),
        patch("yaml.safe_load", return_value=templates_yaml),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/templates")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["templates"]) == 2
    assert data["templates"][0]["id"] == "production_baseline"


@pytest.mark.asyncio
async def test_get_templates_file_missing():
    pool, conn, cursor = _make_pool()

    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            "api.routers.forecasting.cluster_experiments._TEMPLATES_PATH",
            MagicMock(exists=MagicMock(return_value=False)),
        ),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/templates")

    assert resp.status_code == 200
    assert resp.json()["templates"] == []


# ---------------------------------------------------------------------------
# 9. Completed experiments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_completed_experiments():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        _experiment_row(experiment_id=1, status="completed"),
        _experiment_row(experiment_id=3, status="completed", label="Another"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/completed")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["experiments"]) == 2
    for exp in data["experiments"]:
        assert exp["status"] == "completed"


# ---------------------------------------------------------------------------
# 10. Used-by
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_used_by():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (10, "LGBM Run A", "lgbm_cluster", "completed", 72.5, "2026-03-20T10:00:00"),
        (11, "CatBoost Run B", "catboost_cluster", "running", None, "2026-03-21T10:00:00"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/1/used-by")

    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_id"] == 1
    assert len(data["runs"]) == 2
    assert data["runs"][0]["run_id"] == 10
    assert data["runs"][0]["accuracy_pct"] == 72.5
    assert data["runs"][1]["run_label"] == "CatBoost Run B"


@pytest.mark.asyncio
async def test_used_by_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/cluster-experiments/1/used-by")

    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"] == []


# ---------------------------------------------------------------------------
# Comparison computation helpers (unit tests)
# ---------------------------------------------------------------------------

class TestQualityComparison:
    """Test _compute_quality_comparison logic."""

    def test_b_better(self):
        from api.routers.forecasting.cluster_experiments import _compute_quality_comparison
        exp_a = {"silhouette_score": 0.30, "inertia": 200000.0, "optimal_k": 8}
        exp_b = {"silhouette_score": 0.35, "inertia": 180000.0, "optimal_k": 10}
        result = _compute_quality_comparison(exp_a, exp_b)
        assert result["verdict"] == "b_better"
        assert result["silhouette_delta"] == 0.05
        assert result["k_delta"] == 2
        assert result["inertia_delta"] == -20000.0

    def test_a_better(self):
        from api.routers.forecasting.cluster_experiments import _compute_quality_comparison
        exp_a = {"silhouette_score": 0.40, "inertia": 100000.0, "optimal_k": 6}
        exp_b = {"silhouette_score": 0.35, "inertia": 120000.0, "optimal_k": 8}
        result = _compute_quality_comparison(exp_a, exp_b)
        assert result["verdict"] == "a_better"

    def test_mixed(self):
        from api.routers.forecasting.cluster_experiments import _compute_quality_comparison
        exp_a = {"silhouette_score": 0.35, "inertia": 150000.0, "optimal_k": 8}
        exp_b = {"silhouette_score": 0.355, "inertia": 145000.0, "optimal_k": 9}
        result = _compute_quality_comparison(exp_a, exp_b)
        assert result["verdict"] == "mixed"

    def test_null_values(self):
        from api.routers.forecasting.cluster_experiments import _compute_quality_comparison
        exp_a = {"silhouette_score": None, "inertia": None, "optimal_k": None}
        exp_b = {"silhouette_score": 0.35, "inertia": 150000.0, "optimal_k": 8}
        result = _compute_quality_comparison(exp_a, exp_b)
        assert result["silhouette_delta"] is None
        assert result["inertia_delta"] is None
        assert result["k_delta"] is None


class TestProfileComparison:
    """Test _compute_profile_comparison logic."""

    def test_common_and_unique(self):
        from api.routers.forecasting.cluster_experiments import _compute_profile_comparison
        exp_a = {"profiles": [
            {"label": "high_volume_steady", "count": 3000},
            {"label": "low_volume_volatile", "count": 1000},
        ]}
        exp_b = {"profiles": [
            {"label": "high_volume_steady", "count": 2800},
            {"label": "medium_seasonal", "count": 2000},
        ]}
        result = _compute_profile_comparison(exp_a, exp_b)
        assert "low_volume_volatile" in result["clusters_only_in_a"]
        assert "medium_seasonal" in result["clusters_only_in_b"]
        assert len(result["common_clusters"]) == 1
        assert result["common_clusters"][0]["label"] == "high_volume_steady"
        assert result["common_clusters"][0]["count_a"] == 3000
        assert result["common_clusters"][0]["count_b"] == 2800

    def test_empty_profiles(self):
        from api.routers.forecasting.cluster_experiments import _compute_profile_comparison
        exp_a = {"profiles": None}
        exp_b = {"profiles": []}
        result = _compute_profile_comparison(exp_a, exp_b)
        assert result["clusters_only_in_a"] == []
        assert result["clusters_only_in_b"] == []
        assert result["common_clusters"] == []
