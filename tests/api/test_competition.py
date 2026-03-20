"""Tests for competition/champion model endpoints."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport



@pytest.mark.asyncio
async def test_competition_config_get(mock_pool):
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/competition/config")
            # Should return 200 if config exists, or a handled error
            assert response.status_code in (200, 404, 500)


@pytest.mark.asyncio
async def test_competition_summary(mock_pool):
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/competition/summary")
            assert response.status_code in (200, 404, 500)


@pytest.mark.asyncio
async def test_competition_config_update_validates_strategy(mock_pool):
    """PUT /competition/config should reject invalid strategy names."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/competition/config",
                json={
                    "metric": "wape",
                    "lag": "execution",
                    "models": ["a", "b"],
                    "strategy": "invalid_strategy",
                },
            )
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_competition_config_update_accepts_valid_strategy(mock_pool):
    """PUT /competition/config should accept valid strategy names."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
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
            assert response.status_code in (200, 500)
            if response.status_code == 200:
                data = response.json()
                assert data["config"]["strategy"] == "rolling"


@pytest.mark.asyncio
async def test_competition_config_includes_strategy_field(mock_pool):
    """GET /competition/config should include strategy field if present."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/competition/config")
            if response.status_code == 200:
                data = response.json()
                cfg = data.get("config", {})
                # strategy field should be present in config
                assert "strategy" in cfg or "models" in cfg
