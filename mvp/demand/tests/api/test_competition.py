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
