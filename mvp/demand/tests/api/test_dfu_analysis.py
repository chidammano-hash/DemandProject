"""Tests for DFU analysis endpoint."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport



@pytest.mark.asyncio
async def test_dfu_analysis_missing_params(mock_pool):
    """Should return 422 when item/location missing."""
    pool, _, _ = mock_pool
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/dfu/analysis")
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_dfu_analysis_with_params(mock_pool):
    """Should return 200 with required params (even if no data)."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.main._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/dfu/analysis?item=100320&location=1401-BULK&mode=item_location")
            assert response.status_code == 200
            data = response.json()
            assert "mode" in data
            assert "item" in data
            assert "location" in data
