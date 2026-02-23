"""Tests for DFU analysis endpoint."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


@pytest.fixture
def mock_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = mock_conn

    return pool, mock_conn, mock_cursor


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
