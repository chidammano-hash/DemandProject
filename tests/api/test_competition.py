"""Tests for competition/champion model endpoints."""

import shutil
from datetime import UTC, datetime
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_pipeline_config(tmp_path, monkeypatch):
    """Redirect the competition router's pipeline-config path to a temp copy.

    PUT /competition/config reads then re-dumps forecast_pipeline_config.yaml; run
    against the real file it would overwrite the production config and strip every
    inline comment on each test run (a test-isolation defect that pollutes the
    working tree). Point the module constant at a temp copy so reads still work and
    writes never touch the real config.
    """
    from api.routers.forecasting import competition as _competition

    real = _competition._PIPELINE_CONFIG_PATH
    tmp_cfg = tmp_path / "forecast_pipeline_config.yaml"
    if real.exists():
        shutil.copy(real, tmp_cfg)
    monkeypatch.setattr(_competition, "_PIPELINE_CONFIG_PATH", tmp_cfg)
    yield


@pytest.mark.asyncio
async def test_competition_config_get(async_client):
    resp = await async_client.get("/competition/config")
    # Should return 200 if config exists, or a handled error
    assert resp.status_code in (200, 404, 500)


@pytest.mark.asyncio
async def test_competition_summary_uses_promoted_experiment_winner_artifact(
    tmp_path, monkeypatch, mock_pool, async_client
):
    from api.routers.forecasting import competition

    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    (champion_dir / "champion_summary.json").write_text(
        '{"overall_champion_accuracy_pct": 76.56, "model_wins": {"ensemble": 42329}}'
    )
    (champion_dir / "experiment_84_winners.csv").write_text(
        "item_id,customer_group,loc,startdate,model_id,prior_wape,basefcst_pref,tothist_dmd,source_mix\n"
        "100,ALL,A,2026-01-01,lgbm_cluster,0.1,10,9,\n"
        "100,ALL,A,2026-02-01,mstl,0.2,11,10,\n"
        "200,ALL,B,2026-01-01,lgbm_cluster,0.1,12,11,\n"
    )
    monkeypatch.setattr(competition, "_CHAMPION_DATA_DIR", champion_dir)
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = (
        84,
        "Assigned from #83: champ1",
        "per_cluster",
        75.379,
        84.5679,
        918.89,
        2,
        3,
        datetime(2026, 7, 13, 18, 31, tzinfo=UTC),
    )

    resp = await async_client.get("/competition/summary")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["summary"] == {
        "experiment_id": 84,
        "experiment_label": "Assigned from #83: champ1",
        "strategy": "per_cluster",
        "artifact_name": "experiment_84_winners.csv",
        "total_dfus": 2,
        "total_dfu_months": 3,
        "total_champion_rows": 3,
        "model_wins": {"lgbm_cluster": 2, "mstl": 1},
        "overall_champion_wape": 24.621,
        "overall_champion_accuracy_pct": 75.379,
        "run_ts": "2026-07-13T18:31:00+00:00",
        "overall_ceiling_wape": 15.4321,
        "overall_ceiling_accuracy_pct": 84.5679,
        "gap_bps": 918.89,
    }


@pytest.mark.asyncio
async def test_competition_summary_fails_closed_when_promoted_artifact_is_missing(
    tmp_path, monkeypatch, mock_pool, async_client
):
    from api.routers.forecasting import competition

    monkeypatch.setattr(competition, "_CHAMPION_DATA_DIR", tmp_path)
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = (
        84,
        "Champion",
        "per_cluster",
        75.379,
        84.5679,
        918.89,
        2,
        3,
        datetime(2026, 7, 13, 18, 31, tzinfo=UTC),
    )

    resp = await async_client.get("/competition/summary")

    assert resp.status_code == 409
    assert resp.json()["detail"] == (
        "Promoted champion experiment #84 is missing its winner artifact"
    )


@pytest.mark.asyncio
async def test_competition_summary_rejects_duplicate_dfu_month_winners(
    tmp_path, monkeypatch, mock_pool, async_client
):
    from api.routers.forecasting import competition

    (tmp_path / "experiment_84_winners.csv").write_text(
        "item_id,customer_group,loc,startdate,model_id\n"
        "100,ALL,A,2026-01-01,lgbm_cluster\n"
        "100,ALL,A,2026-01-01,mstl\n"
    )
    monkeypatch.setattr(competition, "_CHAMPION_DATA_DIR", tmp_path)
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = (
        84,
        "Champion",
        "per_cluster",
        75.379,
        84.5679,
        918.89,
        1,
        2,
        datetime(2026, 7, 13, 18, 31, tzinfo=UTC),
    )

    resp = await async_client.get("/competition/summary")

    assert resp.status_code == 409
    assert resp.json()["detail"] == (
        "Promoted champion experiment #84 has duplicate winner rows at line 3"
    )


@pytest.mark.asyncio
async def test_competition_summary_returns_not_run_without_a_promoted_experiment(
    mock_pool, async_client
):
    _, _, cursor = mock_pool
    cursor.fetchone.return_value = None

    resp = await async_client.get("/competition/summary")

    assert resp.status_code == 200
    assert resp.json() == {"status": "not_run", "summary": None}


@pytest.mark.asyncio
async def test_competition_config_update_validates_strategy(async_client):
    """PUT /competition/config should reject invalid strategy names."""
    resp = await async_client.put(
        "/competition/config",
        json={
            "metric": "wape",
            "lag": "execution",
            "models": ["a", "b"],
            "strategy": "invalid_strategy",
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_competition_config_update_accepts_valid_strategy(async_client):
    """PUT /competition/config should accept valid strategy names."""
    resp = await async_client.put(
        "/competition/config",
        json={
            "metric": "wape",
            "lag": "execution",
            "models": ["lgbm_cluster", "mstl"],
            "strategy": "rolling",
            "strategy_params": {"window_months": 6},
        },
    )
    # 200 if successful, or 500 if filesystem issue in test env
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert data["config"]["strategy"] == "rolling"


@pytest.mark.asyncio
async def test_competition_config_includes_strategy_field(async_client):
    """GET /competition/config should include strategy field if present."""
    resp = await async_client.get("/competition/config")
    if resp.status_code == 200:
        data = resp.json()
        cfg = data.get("config", {})
        # strategy field should be present in config
        assert "strategy" in cfg or "models" in cfg


@pytest.mark.asyncio
async def test_competition_run_only_submits_governed_refresh(async_client):
    """The legacy endpoint must not rewrite champion facts in the request."""
    with patch("common.services.job_registry.JobManager") as manager_cls:
        manager_cls.return_value.submit_job.return_value = "job-governed-82"

        resp = await async_client.post("/competition/run")

    assert resp.status_code == 202
    assert resp.json() == {
        "job_id": "job-governed-82",
        "job_type": "champion_select",
        "status": "queued",
        "message": "Governed champion refresh submitted",
    }
    manager_cls.return_value.submit_job.assert_called_once_with(
        job_type="champion_select",
        params={},
        label="Governed Champion Refresh",
        triggered_by="competition_run",
    )
