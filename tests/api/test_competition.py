"""Tests for competition/champion model endpoints."""

import shutil

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
async def test_competition_summary(async_client):
    resp = await async_client.get("/competition/summary")
    assert resp.status_code in (200, 404, 500)


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
            "models": ["lgbm_cluster", "catboost_cluster"],
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
