"""Tests for sampled backtest endpoints — strata, preview, and run."""

# ruff: noqa: RUF059

from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Shared mock strata
# ---------------------------------------------------------------------------


def _mock_strata():
    """Return sample strata dict as returned by compute_cluster_strata."""
    return {
        0: {
            "cluster_label": "0",
            "n_dfus": 20000,
            "mean_demand": 250.5,
            "cv": 0.85,
            "zero_pct": 0.12,
            "sku_cks": [f"DFU_{i}" for i in range(20000)],
        },
        1: {
            "cluster_label": "1",
            "n_dfus": 15000,
            "mean_demand": 50.3,
            "cv": 1.45,
            "zero_pct": 0.35,
            "sku_cks": [f"DFU_{i}" for i in range(20000, 35000)],
        },
        2: {
            "cluster_label": "2",
            "n_dfus": 15000,
            "mean_demand": 120.0,
            "cv": 0.95,
            "zero_pct": 0.20,
            "sku_cks": [f"DFU_{i}" for i in range(35000, 50000)],
        },
    }


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/sampled/strata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_strata_returns_200():
    """GET /lgbm-tuning/sampled/strata returns cluster strata with stats."""
    pool, conn, cursor = _make_pool()
    strata = _mock_strata()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value=strata),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/sampled/strata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_clusters"] == 3
    assert data["total_dfus"] == 50000
    assert len(data["strata"]) == 3
    s0 = data["strata"][0]
    assert s0["cluster_id"] == 0
    assert s0["cluster_label"] == "0"
    assert s0["n_dfus"] == 20000
    assert s0["mean_demand"] == 250.5
    assert s0["cv"] == 0.85
    assert s0["zero_pct"] == 0.12


@pytest.mark.asyncio
async def test_get_strata_empty():
    """GET /lgbm-tuning/sampled/strata returns empty strata when no clusters."""
    pool, conn, cursor = _make_pool()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value={}),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/lgbm-tuning/sampled/strata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_clusters"] == 0
    assert data["total_dfus"] == 0
    assert data["strata"] == []


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/sampled/preview
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_sample_proportional():
    """POST /lgbm-tuning/sampled/preview returns allocation with proportional method."""
    pool, conn, cursor = _make_pool()
    strata = _mock_strata()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value=strata),
        patch(
            "common.ml.backtest_sampler._load_sampling_config",
            return_value={"min_per_cluster": 10, "seed": 42},
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/preview",
                json={"target_n": 5000, "method": "proportional"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_n"] == 5000
    assert data["method"] == "proportional"
    assert data["actual_n"] > 0
    assert data["estimated_deviation_pct"] > 0
    assert len(data["allocation"]) == 3
    for alloc in data["allocation"]:
        assert "cluster_id" in alloc
        assert "n_dfus_total" in alloc
        assert "n_sampled" in alloc
        assert "pct_of_cluster" in alloc


@pytest.mark.asyncio
async def test_preview_sample_equal():
    """POST /lgbm-tuning/sampled/preview works with equal method."""
    pool, conn, cursor = _make_pool()
    strata = _mock_strata()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value=strata),
        patch(
            "common.ml.backtest_sampler._load_sampling_config",
            return_value={"min_per_cluster": 10, "seed": 42},
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/preview",
                json={"target_n": 3000, "method": "equal"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "equal"
    # Equal method: each cluster gets same allocation
    allocations = [a["n_sampled"] for a in data["allocation"]]
    assert allocations[0] == allocations[1] == allocations[2]


@pytest.mark.asyncio
async def test_preview_sample_empty_strata():
    """POST /lgbm-tuning/sampled/preview handles empty strata."""
    pool, conn, cursor = _make_pool()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value={}),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/preview",
                json={"target_n": 5000, "method": "proportional"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["actual_n"] == 0
    assert data["allocation"] == []


@pytest.mark.asyncio
async def test_preview_sample_invalid_method():
    """POST /lgbm-tuning/sampled/preview rejects invalid method."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/preview",
                json={"target_n": 5000, "method": "invalid_method"},
            )
    # Pydantic validation should reject it
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/sampled/run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_sampled_run_returns_200():
    """POST /lgbm-tuning/sampled/run submits a sampled_backtest job."""
    pool, conn, cursor = _make_pool()
    strata = _mock_strata()
    sampled_skus = [f"DFU_{i}" for i in range(5000)]

    # fetchone for INSERT RETURNING run_id
    cursor.fetchone.return_value = (42,)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.stratified_sample", return_value=sampled_skus),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value=strata),
        patch("common.ml.backtest_sampler.estimate_accuracy_deviation", return_value=1.5),
        patch("common.services.job_registry.JobManager") as manager_cls,
    ):
        manager_cls.return_value.submit_job.return_value = "job_sampled_1"
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/run",
                json={"target_n": 5000, "method": "proportional"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 42
    assert data["job_id"] == "job_sampled_1"
    assert data["sample_size"] == 5000
    assert data["total_dfus"] == 50000
    assert data["method"] == "proportional"
    assert data["estimated_deviation_pct"] == 1.5
    # Verify the job was submitted through the JobManager (no bare Popen)
    submit = manager_cls.return_value.submit_job
    submit.assert_called_once()
    assert submit.call_args.args[0] == "sampled_backtest"
    assert submit.call_args.args[1]["run_id"] == 42
    assert submit.call_args.args[1]["sku_file"]
    assert any(
        "UPDATE lgbm_tuning_run SET job_id" in call.args[0]
        for call in cursor.execute.call_args_list
    )


@pytest.mark.asyncio
async def test_trigger_sampled_run_empty_sample():
    """POST /lgbm-tuning/sampled/run returns 400 when no DFUs available."""
    pool, conn, cursor = _make_pool()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.stratified_sample", return_value=[]),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value={}),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/run",
                json={"target_n": 5000},
            )
    assert resp.status_code == 400
    assert "no dfus" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_trigger_sampled_run_with_param_overrides():
    """POST /lgbm-tuning/sampled/run accepts param_overrides."""
    pool, conn, cursor = _make_pool()
    strata = _mock_strata()
    sampled_skus = [f"DFU_{i}" for i in range(3000)]
    cursor.fetchone.return_value = (99,)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.stratified_sample", return_value=sampled_skus),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value=strata),
        patch("common.ml.backtest_sampler.estimate_accuracy_deviation", return_value=2.0),
        patch("common.services.job_registry.JobManager") as manager_cls,
    ):
        manager_cls.return_value.submit_job.return_value = "job_sampled_2"
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/run",
                json={
                    "target_n": 3000,
                    "method": "sqrt",
                    "param_overrides": {"learning_rate": 0.05, "num_leaves": 127},
                },
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == 99
    assert data["sample_size"] == 3000
    # Overrides travel to the job params
    params = manager_cls.return_value.submit_job.call_args.args[1]
    assert params["param_overrides"] == {"learning_rate": 0.05, "num_leaves": 127}


@pytest.mark.asyncio
async def test_trigger_sampled_run_submit_failure():
    """POST /lgbm-tuning/sampled/run handles job-submission failure."""
    pool, conn, cursor = _make_pool()
    strata = _mock_strata()
    sampled_skus = [f"DFU_{i}" for i in range(2000)]
    cursor.fetchone.return_value = (50,)

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.sampled_backtest.get_conn", return_value=conn),
        patch("common.ml.backtest_sampler.stratified_sample", return_value=sampled_skus),
        patch("common.ml.backtest_sampler.compute_cluster_strata", return_value=strata),
        patch("common.ml.backtest_sampler.estimate_accuracy_deviation", return_value=2.5),
        patch("common.services.job_registry.JobManager") as manager_cls,
    ):
        manager_cls.return_value.submit_job.side_effect = RuntimeError("scheduler down")
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/run",
                json={"target_n": 2000},
            )
    assert resp.status_code == 500
    assert "failed to submit" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_preview_sample_target_validation():
    """POST /lgbm-tuning/sampled/preview validates target_n bounds."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # target_n < 100 should fail
            resp = await client.post(
                "/lgbm-tuning/sampled/preview",
                json={"target_n": 50, "method": "proportional"},
            )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_preview_sample_target_too_high():
    """POST /lgbm-tuning/sampled/preview rejects target_n > 20000."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/lgbm-tuning/sampled/preview",
                json={"target_n": 25000, "method": "proportional"},
            )
    assert resp.status_code == 422
