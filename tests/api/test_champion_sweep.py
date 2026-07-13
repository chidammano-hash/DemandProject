"""Tests for the champion sweep API — /champion-sweeps/* endpoints.

Covers create (candidate-count preview + cap), list, detail, leaderboard,
segments, cancel, retired promote-winner containment, and delete.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool

_MANUAL_PROMOTION_RETIRED_DETAIL = {
    "code": "manual_champion_promotion_retired",
    "message": (
        "Manual two-stage champion promotion is retired. Select a completed experiment "
        "and run POST /champion-experiments/{experiment_id}/assign to re-evaluate and "
        "atomically assign a governed champion."
    ),
}

# Column order must match champion_sweeps._SWEEP_COLS (22 columns).
_SWEEP_COLS = [
    "sweep_id", "label", "notes", "mode", "segment_axis", "objective", "grid_spec",
    "parallel", "baseline_experiment_id", "status", "candidate_count", "completed_count",
    "job_id", "created_at", "started_at", "completed_at", "runtime_seconds",
    "best_global_experiment_id", "composite_experiment_id", "recommended_experiment_id",
    "recommended_score", "recommended_gate_eligible",
]
_SWEEP_DESC = [(c,) for c in _SWEEP_COLS]


def _sweep_row(
    sweep_id=1, label="June tournament", status="completed", mode="both",
    candidate_count=6, completed_count=6, recommended_experiment_id=7,
    recommended_score=86.2, recommended_gate_eligible=True,
):
    return (
        sweep_id, label, None, mode, "demand_class", "robust", "{}",
        False, None, status, candidate_count, completed_count,
        "job-1", "2026-06-18T10:00:00+00:00", None, None, 120.0,
        3, 7, recommended_experiment_id, recommended_score, recommended_gate_eligible,
    )


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_sweep_returns_candidate_count():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (42,)  # RETURNING sweep_id
    mock_jm = MagicMock()
    mock_jm.return_value.submit_job.return_value = "job-sweep-1"

    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/champion-sweeps",
                json={
                    "label": "June tournament",
                    "mode": "both",
                    "grid_spec": {
                        "templates": ["rolling_6m", "ensemble_top3_inverse", "per_segment"],
                        "models_variants": [
                            ["lgbm_cluster", "chronos2_enriched"],
                            ["lgbm_cluster", "mstl", "chronos2_enriched"],
                        ],
                    },
                },
            )

    assert resp.status_code == 202
    data = resp.json()
    assert data["sweep_id"] == 42
    assert data["job_id"] == "job-sweep-1"
    # 3 templates × 2 variants = 6 candidates (segmentation is NOT an axis)
    assert data["candidate_count"] == 6


@pytest.mark.asyncio
async def test_create_sweep_rejects_zero_candidates():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/champion-sweeps",
                json={"label": "empty", "grid_spec": {"templates": []}},
            )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_sweep_rejects_over_cap():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 30 templates × 1 variant = 30 > default cap of 24
            resp = await client.post(
                "/champion-sweeps",
                json={"label": "huge", "grid_spec": {"templates": [f"t{i}" for i in range(30)]}},
            )
    assert resp.status_code == 422
    assert "cap" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_create_sweep_invalid_mode():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/champion-sweeps",
                json={"label": "x", "mode": "nonsense", "grid_spec": {"templates": ["rolling_6m"]}},
            )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_sweep_rejects_retired_model_variant_before_insert():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/champion-sweeps",
                json={
                    "label": "retired",
                    "grid_spec": {
                        "templates": ["rolling_6m"],
                        "models_variants": [["lgbm_cluster", "catboost_cluster"]],
                    },
                },
            )

    assert resp.status_code == 422
    assert "catboost_cluster" in resp.json()["detail"]
    assert "Valid competing models" in resp.json()["detail"]
    cursor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_create_sweep_rejects_retired_model_in_explicit_config():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/champion-sweeps",
                json={
                    "label": "retired explicit",
                    "grid_spec": {
                        "configs": [
                            {"strategy": "expanding", "models": ["mstl", "prophet"]}
                        ]
                    },
                },
            )

    assert resp.status_code == 422
    assert "prophet" in resp.json()["detail"]
    cursor.execute.assert_not_called()


# ---------------------------------------------------------------------------
# List / detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_sweeps():
    pool, conn, cursor = _make_pool(description=_SWEEP_DESC)
    cursor.fetchall.return_value = [_sweep_row(sweep_id=1), _sweep_row(sweep_id=2, status="running")]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-sweeps")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sweeps"]) == 2
    assert data["sweeps"][0]["sweep_id"] == 1


@pytest.mark.asyncio
async def test_get_sweep_detail():
    pool, conn, cursor = _make_pool(description=_SWEEP_DESC)
    cursor.fetchone.return_value = _sweep_row(sweep_id=5, label="Detail")
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-sweeps/5")
    assert resp.status_code == 200
    assert resp.json()["sweep_id"] == 5


@pytest.mark.asyncio
async def test_get_sweep_not_found():
    pool, conn, cursor = _make_pool(description=_SWEEP_DESC)
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-sweeps/999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Leaderboard / segments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_leaderboard():
    pool, conn, cursor = _make_pool(description=[
        ("experiment_id",), ("global_rank",), ("global_score",), ("gate_eligible",),
        ("is_composite",), ("skipped_duplicate",), ("label",), ("strategy",),
        ("strategy_params",), ("models",), ("metric",), ("champion_accuracy",),
        ("ceiling_accuracy",), ("gap_bps",), ("status",),
    ])
    # First fetchone() = existence check; then fetchall() = members.
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (7, 1, 86.2, True, False, False, "[sweep] rolling", "rolling",
         '{"window_months": 6}', '["lgbm_cluster","chronos2_enriched"]', "wape", 88.0, 90.0, 200.0, "completed"),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-sweeps/1/leaderboard")
    assert resp.status_code == 200
    members = resp.json()["members"]
    assert members[0]["experiment_id"] == 7
    assert members[0]["strategy_params"] == {"window_months": 6}


@pytest.mark.asyncio
async def test_segments_groups_by_segment():
    pool, conn, cursor = _make_pool(description=[
        ("segment",), ("experiment_id",), ("n_dfus",), ("accuracy",),
        ("score",), ("segment_rank",), ("label",), ("strategy",),
    ])
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        ("smooth", 7, 120, 90.0, 90.0, 1, "ensemble", "ensemble"),
        ("smooth", 8, 120, 85.0, 85.0, 2, "rolling", "rolling"),
        ("intermittent", 9, 40, 70.0, 70.0, 1, "rolling", "rolling"),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/champion-sweeps/1/segments")
    assert resp.status_code == 200
    segs = {s["segment"]: s for s in resp.json()["segments"]}
    assert segs["smooth"]["winner"]["experiment_id"] == 7
    assert len(segs["smooth"]["candidates"]) == 2
    assert segs["intermittent"]["winner"]["strategy"] == "rolling"


# ---------------------------------------------------------------------------
# Cancel / promote-winner / delete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_running_sweep():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("running", "job-1")
    mock_jm = MagicMock()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("common.services.job_registry.JobManager", mock_jm),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-sweeps/1/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_promote_winner_is_gone_without_db_or_stage_one_delegation():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("completed", 7, True)
    fake_promote = MagicMock()
    with (
        patch("api.core._get_pool", return_value=pool),
        patch("api.routers.forecasting.champion_experiments.promote_experiment", fake_promote),
    ):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/champion-sweeps/1/promote-winner")
    assert resp.status_code == 410
    assert resp.json()["detail"] == _MANUAL_PROMOTION_RETIRED_DETAIL
    pool.connection.assert_not_called()
    conn.commit.assert_not_called()
    fake_promote.assert_not_called()


@pytest.mark.asyncio
async def test_delete_running_sweep_refused():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("running",)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/champion-sweeps/1")
    assert resp.status_code == 409
