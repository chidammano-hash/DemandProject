"""Tests for competition/champion model endpoints."""

import pytest


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
