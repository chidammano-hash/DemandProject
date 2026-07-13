"""Tests for backtest management API — /backtest-management/* endpoints.

Tests summary listing, model runs, current metadata, run submission,
and load submission.
"""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch
from uuid import UUID

import httpx
import pytest
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


@pytest.fixture(autouse=True)
def _stable_neural_training_cohort():
    """Keep API readiness tests on one deterministic current neural cohort."""
    identity = SimpleNamespace(checksum="f" * 64, dfu_count=12_476)
    with (
        patch(
            f"{_ROUTER_MOD}.resolve_forecast_sales_table",
            return_value="fact_sales_monthly_original",
        ),
        patch(
            f"{_ROUTER_MOD}.load_neural_training_cohort_identity",
            return_value=identity,
        ),
    ):
        yield identity


@pytest.fixture(autouse=True)
def _stable_governed_champion_readiness():
    """Keep snapshot tests focused unless they explicitly exercise champion lineage."""
    with patch(
        f"{_ROUTER_MOD}.validate_active_champion_readiness",
        return_value={"experiment_id": 77},
    ):
        yield


_NOW = datetime(2026, 4, 6, 10, 0, 0, tzinfo=UTC)
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
    pool, _conn, cursor = _make_pool()
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


@pytest.mark.asyncio
async def test_training_status_reads_validated_neural_artifact_metadata(tmp_path):
    pool, conn, _cursor = _make_pool()
    sales_lineage = SimpleNamespace(batch_id=91, source_hash="c" * 64)
    artifact = SimpleNamespace(
        artifact_id="a" * 64,
        metadata={
            "trained_at": "2026-07-12T08:00:00+00:00",
            "training_dfu_count": 12_476,
            "history_end": "2026-06-01",
        },
    )
    with (
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "nhits": {
                    "type": "deep_learning",
                    "forecast": True,
                    "params": {"h": 6, "min_history": 12},
                }
            },
        ),
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": False},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=sales_lineage,
        ) as load_sales,
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(f"{_ROUTER_MOD}.read_active_neural_artifact_ref", return_value=artifact) as read,
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    assert resp.json()["nhits"] == {
        "model_id": "nhits",
        "type": "deep_learning",
        "trained": True,
        "ready": True,
        "trained_at": "2026-07-12T08:00:00+00:00",
        "training_mode": "production",
        "n_dfus": 12_476,
        "planning_date": "2026-06-01",
        "artifact_id": "a" * 64,
    }
    read.assert_called_once_with(
        model_id="nhits",
        base_dir=tmp_path,
        expected_params={"h": 6, "min_history": 12},
        expected_source_sales_batch_id=91,
        expected_data_checksum="c" * 64,
        expected_history_end=date(2026, 6, 1),
        expected_training_cohort_checksum="f" * 64,
        expected_training_dfu_count=12_476,
        generator_contract_version=ANY,
    )
    load_sales.assert_called_once_with(conn)


@pytest.mark.asyncio
async def test_training_status_marks_invalid_neural_artifact_not_ready(tmp_path):
    pool, _conn, _cursor = _make_pool()
    with (
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "nbeats": {
                    "type": "deep_learning",
                    "forecast": True,
                    "params": {"h": 6, "min_history": 12},
                }
            },
        ),
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": False},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=SimpleNamespace(batch_id=91, source_hash="c" * 64),
        ),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.read_active_neural_artifact_ref",
            side_effect=RuntimeError("checksum mismatch"),
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    assert resp.json()["nbeats"] == {
        "model_id": "nbeats",
        "type": "deep_learning",
        "trained": False,
        "ready": False,
        "trained_at": None,
        "training_mode": None,
        "n_dfus": None,
        "planning_date": None,
        "stale_reason": "The active nbeats production artifact is invalid. Run Forecast Publish to rebuild current production artifacts.",
    }


@pytest.mark.asyncio
async def test_training_status_reads_validated_lightgbm_bundle_metadata(tmp_path):
    pool, conn, _cursor = _make_pool()
    sales_lineage = SimpleNamespace(batch_id=91, source_hash="c" * 64)
    cluster_population = SimpleNamespace(
        experiment_id=44,
        assignment_count=12_476,
        assignment_checksum="d" * 64,
        cluster_labels=frozenset({"steady", "intermittent"}),
    )
    artifact = SimpleNamespace(
        artifact_set_id="b" * 32,
        metadata={
            "trained_at": "2026-07-12T08:00:00+00:00",
            "lineage": {"history_end": "2026-06-01"},
            "training_metadata": {"n_dfus": 12_476},
        },
    )
    expected_config = {"algorithm": "lgbm", "clustering": {"enabled": True}}
    with (
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "lgbm_cluster": {
                    "type": "tree",
                    "forecast": True,
                    "params": {"n_estimators": 2000},
                }
            },
        ),
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": True},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=sales_lineage,
        ) as load_sales,
        patch(
            f"{_ROUTER_MOD}.load_promoted_cluster_population",
            return_value=cluster_population,
        ) as load_clusters,
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.build_production_tree_model_config_payload",
            return_value=expected_config,
        ) as build_config,
        patch(
            f"{_ROUTER_MOD}.read_active_tree_artifact_ref",
            return_value=artifact,
        ) as read,
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    assert resp.json()["lgbm_cluster"] == {
        "model_id": "lgbm_cluster",
        "type": "tree",
        "trained": True,
        "ready": True,
        "trained_at": "2026-07-12T08:00:00+00:00",
        "training_mode": "production",
        "n_dfus": 12_476,
        "planning_date": "2026-06-01",
        "artifact_id": "b" * 32,
    }
    build_config.assert_called_once()
    read.assert_called_once_with(
        model_id="lgbm_cluster",
        base_dir=tmp_path,
        expected_spec=ANY,
    )
    expected_spec = read.call_args.kwargs["expected_spec"]
    assert expected_spec.model_config == expected_config
    assert expected_spec.cluster_strategy == "per_cluster"
    assert expected_spec.cluster_labels == ("intermittent", "steady")
    assert expected_spec.lineage.source_sales_batch_id == 91
    assert expected_spec.lineage.data_checksum == "c" * 64
    assert expected_spec.lineage.history_end == date(2026, 6, 1)
    assert expected_spec.lineage.cluster_experiment_id == 44
    assert expected_spec.lineage.cluster_assignment_count == 12_476
    assert expected_spec.lineage.cluster_assignment_checksum == "d" * 64
    load_sales.assert_called_once_with(conn)
    load_clusters.assert_called_once_with(conn)


@pytest.mark.asyncio
async def test_training_status_rejects_invalid_lightgbm_bundle_even_with_loose_pickle(
    tmp_path,
):
    pool, _conn, _cursor = _make_pool()
    legacy_dir = tmp_path / "lgbm_cluster"
    legacy_dir.mkdir()
    (legacy_dir / "cluster_0.pkl").write_bytes(b"legacy")
    with (
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "lgbm_cluster": {
                    "type": "tree",
                    "forecast": True,
                    "params": {"n_estimators": 2000},
                }
            },
        ),
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": False},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=SimpleNamespace(batch_id=91, source_hash="c" * 64),
        ),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.build_production_tree_model_config_payload",
            return_value={"algorithm": "lgbm"},
        ),
        patch(
            f"{_ROUTER_MOD}.read_active_tree_artifact_ref",
            side_effect=RuntimeError("checksum mismatch"),
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    assert resp.json()["lgbm_cluster"] == {
        "model_id": "lgbm_cluster",
        "type": "tree",
        "trained": False,
        "ready": False,
        "trained_at": None,
        "training_mode": None,
        "n_dfus": None,
        "planning_date": None,
        "stale_reason": "The active lgbm_cluster production artifact is invalid. Run Forecast Publish to rebuild current production artifacts.",
    }


@pytest.mark.asyncio
async def test_training_status_marks_stale_neural_artifact_with_retrain_reason(tmp_path):
    from common.ml.neural_artifacts import NeuralArtifactLineageMismatchError

    pool, _conn, _cursor = _make_pool()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "nbeats": {
                    "type": "deep_learning",
                    "forecast": True,
                    "params": {"h": 6, "min_history": 12},
                }
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": False},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=SimpleNamespace(batch_id=92, source_hash="e" * 64),
        ),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.read_active_neural_artifact_ref",
            side_effect=NeuralArtifactLineageMismatchError(
                "active artifact does not match current lineage"
            ),
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    status = resp.json()["nbeats"]
    assert status["trained"] is False
    assert status["ready"] is False
    assert "retrain nbeats" in status["stale_reason"].lower()


@pytest.mark.asyncio
async def test_training_status_marks_stale_lightgbm_bundle_with_retrain_reason(tmp_path):
    from common.ml.tree_artifacts import TreeArtifactLineageMismatchError

    pool, _conn, _cursor = _make_pool()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "lgbm_cluster": {
                    "type": "tree",
                    "forecast": True,
                    "params": {"n_estimators": 2000},
                }
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": False},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=SimpleNamespace(batch_id=92, source_hash="e" * 64),
        ),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.build_production_tree_model_config_payload",
            return_value={"algorithm": "lgbm"},
        ),
        patch(
            f"{_ROUTER_MOD}.read_active_tree_artifact_ref",
            side_effect=TreeArtifactLineageMismatchError(
                "active artifact does not match current lineage"
            ),
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    status = resp.json()["lgbm_cluster"]
    assert status["trained"] is False
    assert status["ready"] is False
    assert "retrain lgbm_cluster" in status["stale_reason"].lower()


@pytest.mark.asyncio
async def test_training_status_preflights_direct_models_against_current_lineage(tmp_path):
    lineage = SimpleNamespace(
        history_end=date(2026, 6, 1),
        sales=SimpleNamespace(batch_id=92, source_hash="e" * 64),
    )
    with (
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "mstl": {"type": "statistical", "forecast": True},
                "chronos2_enriched": {"type": "foundation", "forecast": True},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": True},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(f"{_ROUTER_MOD}._load_current_training_lineage", return_value=lineage),
        patch(
            f"{_ROUTER_MOD}.build_backtest_config_snapshot",
            side_effect=lambda _config, model_id: SimpleNamespace(
                checksum=f"{1 if model_id == 'mstl' else 2}" * 64
            ),
        ),
        patch(
            f"{_ROUTER_MOD}.direct_model_runtime_contract",
            side_effect=lambda model_id: {"adapter": f"{model_id}-runtime"},
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    assert resp.json()["mstl"] == {
        "model_id": "mstl",
        "type": "statistical",
        "trained": False,
        "ready": True,
        "trained_at": None,
        "training_mode": "direct_inference",
        "n_dfus": None,
        "planning_date": "2026-06-01",
        "source_sales_batch_id": 92,
        "config_checksum": "1" * 64,
        "runtime_contract": {"adapter": "mstl-runtime"},
    }
    assert resp.json()["chronos2_enriched"] == {
        "model_id": "chronos2_enriched",
        "type": "foundation",
        "trained": False,
        "ready": True,
        "trained_at": None,
        "training_mode": "direct_inference",
        "n_dfus": None,
        "planning_date": "2026-06-01",
        "source_sales_batch_id": 92,
        "config_checksum": "2" * 64,
        "runtime_contract": {"adapter": "chronos2_enriched-runtime"},
    }


@pytest.mark.asyncio
async def test_training_status_fails_closed_when_direct_runtime_is_missing(tmp_path):
    lineage = SimpleNamespace(
        history_end=date(2026, 6, 1),
        sales=SimpleNamespace(batch_id=92, source_hash="e" * 64),
    )
    with (
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={"mstl": {"type": "statistical", "forecast": True}},
        ),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": True},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(f"{_ROUTER_MOD}._load_current_training_lineage", return_value=lineage),
        patch(
            f"{_ROUTER_MOD}.build_backtest_config_snapshot",
            return_value=SimpleNamespace(checksum="1" * 64),
        ),
        patch(
            f"{_ROUTER_MOD}.direct_model_runtime_contract",
            side_effect=RuntimeError("statsforecast missing"),
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    status = resp.json()["mstl"]
    assert status["ready"] is False
    assert "forecast publish" in status["stale_reason"].lower()
    assert "statsforecast missing" not in status["stale_reason"].lower()


@pytest.mark.asyncio
async def test_training_status_marks_missing_artifact_actionable(tmp_path):
    pool, _conn, _cursor = _make_pool()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "nhits": {
                    "type": "deep_learning",
                    "forecast": True,
                    "params": {"h": 6, "min_history": 12},
                }
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": False},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=SimpleNamespace(batch_id=92, source_hash="e" * 64),
        ),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.read_active_neural_artifact_ref",
            side_effect=FileNotFoundError,
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    status = resp.json()["nhits"]
    assert status["trained"] is False
    assert status["ready"] is False
    assert "run forecast publish" in status["stale_reason"].lower()


@pytest.mark.asyncio
async def test_neural_readiness_does_not_depend_on_promoted_clustering(tmp_path):
    pool, _conn, _cursor = _make_pool()
    artifact = SimpleNamespace(
        artifact_id="a" * 64,
        metadata={
            "trained_at": "2026-07-12T08:00:00+00:00",
            "training_dfu_count": 12_476,
            "history_end": "2026-06-01",
        },
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "nhits": {
                    "type": "deep_learning",
                    "forecast": True,
                    "params": {"h": 6, "min_history": 12},
                }
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": True},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=SimpleNamespace(batch_id=92, source_hash="e" * 64),
        ),
        patch(
            f"{_ROUTER_MOD}.load_promoted_cluster_population",
            side_effect=RuntimeError("cluster assignments unavailable"),
        ),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(f"{_ROUTER_MOD}.read_active_neural_artifact_ref", return_value=artifact),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    assert resp.json()["nhits"]["ready"] is True


@pytest.mark.asyncio
async def test_snapshot_roster_readiness_validates_exact_champion_and_top_three():
    run_ids = [
        UUID("00000000-0000-0000-0000-000000000001"),
        UUID("00000000-0000-0000-0000-000000000002"),
        UUID("00000000-0000-0000-0000-000000000003"),
    ]
    pool, _conn, cursor = _make_pool(
        fetchall_return=[
            ("champion", "champion", None, None, None),
            ("lgbm_cluster", "contender", 1, 101, run_ids[0]),
            ("nhits", "contender", 2, 102, run_ids[1]),
            ("mstl", "contender", 3, 104, run_ids[2]),
        ]
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.validate_ready_snapshot_contender",
            return_value=SimpleNamespace(row_count=100),
        ) as validate,
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/snapshot-roster-readiness")

    assert resp.status_code == 200
    assert resp.json() == {
        "planning_month": "2026-07-01",
        "ready": True,
        "champion_ready": True,
        "roster_model_count": 4,
        "ready_contender_count": 3,
        "required_contender_count": 3,
        "contenders": [
            {"model_id": "lgbm_cluster", "rank": 1, "ready": True, "stale_reason": None},
            {"model_id": "nhits", "rank": 2, "ready": True, "stale_reason": None},
            {"model_id": "mstl", "rank": 3, "ready": True, "stale_reason": None},
        ],
        "stale_reason": None,
        "action_pipeline": None,
    }
    cursor.execute.assert_called()
    assert validate.call_count == 3


@pytest.mark.asyncio
async def test_snapshot_roster_readiness_returns_recovery_action_when_incomplete():
    pool, _conn, _cursor = _make_pool(
        fetchall_return=[("champion", "champion", None, None, None)]
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(f"{_ROUTER_MOD}.validate_ready_snapshot_contender") as validate,
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/snapshot-roster-readiness")

    assert resp.status_code == 200
    status = resp.json()
    assert status["ready"] is False
    assert status["champion_ready"] is True
    assert status["ready_contender_count"] == 0
    assert status["action_pipeline"] == "forecast-publish"
    assert "forecast publish" in status["stale_reason"].lower()
    validate.assert_not_called()


@pytest.mark.asyncio
async def test_snapshot_readiness_routes_missing_governed_champion_to_model_refresh():
    pool, _conn, _cursor = _make_pool(
        fetchall_return=[("champion", "champion", None, None, None)]
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.validate_active_champion_readiness",
            side_effect=RuntimeError("routing artifact missing"),
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/snapshot-roster-readiness")

    assert resp.status_code == 200
    status = resp.json()
    assert status["ready"] is False
    assert status["champion_ready"] is False
    assert status["action_pipeline"] == "model-refresh"
    assert "named model refresh" in status["stale_reason"].lower()
    assert "routing artifact missing" not in status["stale_reason"].lower()


@pytest.mark.asyncio
async def test_snapshot_roster_readiness_reports_stale_contender_without_500():
    from common.services.forecast_snapshot_validation import SnapshotContenderStaleError

    run_ids = [
        UUID("00000000-0000-0000-0000-000000000011"),
        UUID("00000000-0000-0000-0000-000000000012"),
        UUID("00000000-0000-0000-0000-000000000013"),
    ]
    pool, _conn, _cursor = _make_pool(
        fetchall_return=[
            ("champion", "champion", None, None, None),
            ("lgbm_cluster", "contender", 1, 101, run_ids[0]),
            ("nhits", "contender", 2, 102, run_ids[1]),
            ("mstl", "contender", 3, 104, run_ids[2]),
        ]
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
        patch(
            f"{_ROUTER_MOD}.validate_ready_snapshot_contender",
            side_effect=[
                SimpleNamespace(row_count=100),
                SnapshotContenderStaleError("N-HiTS artifact is stale"),
                SimpleNamespace(row_count=100),
            ],
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/snapshot-roster-readiness")

    assert resp.status_code == 200
    status = resp.json()
    assert status["ready"] is False
    assert status["ready_contender_count"] == 2
    assert status["contenders"][1]["stale_reason"] == "N-HiTS artifact is stale"


def test_training_status_lineage_loads_sales_and_clusters_from_one_connection():
    from api.routers.forecasting.backtest_management import _load_current_training_lineage

    pool, conn, cursor = _make_pool()
    sales_lineage = SimpleNamespace(batch_id=91, source_hash="c" * 64)
    cluster_population = SimpleNamespace(
        experiment_id=44,
        assignment_count=12_476,
        assignment_checksum="d" * 64,
        cluster_labels=frozenset({"steady", "intermittent"}),
    )
    with (
        patch("api.core._get_pool", return_value=pool),
        patch(
            f"{_ROUTER_MOD}.load_completed_sales_lineage",
            return_value=sales_lineage,
        ) as load_sales,
        patch(
            f"{_ROUTER_MOD}.load_promoted_cluster_population",
            return_value=cluster_population,
        ) as load_clusters,
        patch(
            f"{_ROUTER_MOD}.load_neural_training_cohort_identity",
            return_value=SimpleNamespace(checksum="f" * 64, dfu_count=12_476),
        ) as load_neural_cohort,
        patch(f"{_ROUTER_MOD}.get_planning_date", return_value=date(2026, 7, 12)),
    ):
        lineage = _load_current_training_lineage(
            {"clustering": {"enabled": True}},
            neural_min_history_values=(12, 12),
        )

    assert lineage.sales is sales_lineage
    assert lineage.history_end == date(2026, 6, 1)
    assert lineage.clusters is cluster_population
    pool.connection.assert_called_once_with()
    cursor.execute.assert_called_once_with(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    )
    load_sales.assert_called_once_with(conn)
    load_clusters.assert_called_once_with(conn)
    load_neural_cohort.assert_called_once_with(
        conn,
        sales_table="fact_sales_monthly_original",
        history_end=date(2026, 6, 1),
        min_history=12,
    )
    assert lineage.neural_cohorts[12].dfu_count == 12_476


def test_direct_readiness_rejects_sales_missing_latest_closed_month():
    from api.routers.forecasting._training_readiness import load_latest_closed_sales_month

    _pool, conn, cursor = _make_pool(fetchone_return=(date(2026, 5, 1),))

    with pytest.raises(RuntimeError, match="missing the latest closed month"):
        load_latest_closed_sales_month(
            conn,
            sales_table="fact_sales_monthly_original",
            expected_history_end=date(2026, 6, 1),
        )

    assert cursor.execute.call_args.args[1] == (date(2026, 6, 1),)


@pytest.mark.asyncio
async def test_training_status_returns_opaque_500_when_configuration_cannot_load():
    with (
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={"nhits": {"type": "deep_learning", "forecast": True}},
        ),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            side_effect=ValueError("sensitive configuration detail"),
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 500
    assert resp.json() == {"detail": "production training readiness check failed"}


@pytest.mark.asyncio
async def test_training_status_fails_closed_when_current_lineage_is_not_ready(tmp_path):
    with (
        patch(
            f"{_ROUTER_MOD}.get_algorithm_roster",
            return_value={
                "nhits": {
                    "type": "deep_learning",
                    "forecast": True,
                    "params": {"h": 6, "min_history": 12},
                }
            },
        ),
        patch(
            f"{_ROUTER_MOD}.load_forecast_pipeline_config",
            return_value={
                "clustering": {"enabled": False},
                "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
            },
        ),
        patch(
            f"{_ROUTER_MOD}._load_current_training_lineage",
            side_effect=RuntimeError("no canonical sales load"),
        ),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/backtest-management/training-status")

    assert resp.status_code == 200
    status = resp.json()["nhits"]
    assert status["trained"] is False
    assert "canonical sales load" in status["stale_reason"].lower()
    assert "no canonical sales load" not in status["stale_reason"].lower()


# ---------------------------------------------------------------------------
# 2. GET /backtest-management/{model_id}/runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_model_runs_returns_list():
    pool, _conn, cursor = _make_pool()
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
    pool, _conn, cursor = _make_pool()
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
    pool, _conn, cursor = _make_pool()
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
    pool, _conn, cursor = _make_pool()
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
    pool, _conn, cursor = _make_pool()
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
    pool, _conn, cursor = _make_pool()
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
    pool, _conn, _cursor = _make_pool()

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
    pool, _conn, _cursor = _make_pool()

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
    pool, _conn, _cursor = _make_pool()

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
# 6. POST /backtest-management/{model_id}/train
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["lgbm_cluster", "nhits", "nbeats"])
async def test_submit_training_accepts_every_persisted_model(model_id: str):
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-train-1"

    with (
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(f"/backtest-management/{model_id}/train")

    assert resp.status_code == 201
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {"model_id": model_id, "all_models": False}


@pytest.mark.asyncio
async def test_submit_training_all_uses_one_bulk_job():
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-train-all"

    with patch("common.services.job_registry.JobManager", mock_jm):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/all/train")

    assert resp.status_code == 201
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {"model_id": "", "all_models": True}
    assert kwargs["label"] == "Train Production: Required Models"


@pytest.mark.asyncio
async def test_submit_training_rejects_direct_inference_model():
    mock_jm = MagicMock()

    with (
        patch(f"{_ROUTER_MOD}.get_algorithm_roster", return_value=_mock_roster()),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/mstl/train")

    assert resp.status_code == 400
    assert "LightGBM, N-HiTS, and N-BEATS" in resp.json()["detail"]
    mock_jm.assert_not_called()


# ---------------------------------------------------------------------------
# 7. POST /backtest-management/{model_id}/generate — horizon + CI threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_threads_horizon_and_confidence_intervals():
    """horizon + confidence_intervals query params reach the job params.

    Regression: these were previously dropped for single-model generation, so
    the Forecast panel's horizon input and CI toggle silently had no effect.
    """
    pool, _conn, _cursor = _make_pool()
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
    """Without query params, only the retained model and run metadata are passed."""
    pool, _conn, _cursor = _make_pool()
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
            resp = await ac.post("/backtest-management/mstl/generate")

    assert resp.status_code == 201
    _, kwargs = mock_jm.return_value.submit_job.call_args
    assert kwargs["params"] == {
        "model_id": "mstl",
        "run_id": source_run_id,
        "generation_purpose": "release_candidate",
    }


@pytest.mark.asyncio
async def test_generate_rejects_retired_model_before_job_submission():
    pool, _, _ = _make_pool()
    mock_jm = MagicMock()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/backtest-management/catboost_cluster/generate")

    assert resp.status_code == 404
    mock_jm.assert_not_called()


@pytest.mark.asyncio
async def test_generate_threads_confidence_intervals_false():
    """confidence_intervals=false threads an explicit False (force CI off)."""
    pool, _conn, _cursor = _make_pool()
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
