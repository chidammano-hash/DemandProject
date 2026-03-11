"""Tests for forecast accuracy endpoints."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


@pytest.fixture
def mock_pool():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.description = [("model_id",)]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = mock_conn

    return pool, mock_conn, mock_cursor


@pytest.mark.asyncio
async def test_forecast_models_endpoint(mock_pool):
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("external",), ("lgbm_global",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/forecast/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data


@pytest.mark.asyncio
async def test_accuracy_slice_endpoint(mock_pool):
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/accuracy/slice?group_by=cluster_assignment")
            assert response.status_code == 200
            data = response.json()
            assert "group_by" in data
            assert "rows" in data


@pytest.mark.asyncio
async def test_accuracy_slice_with_brand_filter(mock_pool):
    """brand param triggers raw-table path when no common_dfus."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice?group_by=cluster_assignment&brand=BrandA"
            )
            assert response.status_code == 200
            data = response.json()
            assert "rows" in data
            assert data.get("source") == "fact_external_forecast_monthly"


@pytest.mark.asyncio
async def test_accuracy_slice_with_category_and_market_filter(mock_pool):
    """category and market params trigger raw-table path."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/accuracy/slice?group_by=model_id&category=CAT1,CAT2&market=NY"
            )
            assert response.status_code == 200
            data = response.json()
            assert "rows" in data


@pytest.mark.asyncio
async def test_lag_curve_endpoint(mock_pool):
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/accuracy/lag-curve")
            assert response.status_code == 200
            data = response.json()
            assert "by_lag" in data


@pytest.mark.asyncio
async def test_lag_curve_with_brand_filter(mock_pool):
    """brand param triggers raw backtest_lag_archive path."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/accuracy/lag-curve?brand=BrandA,BrandB")
            assert response.status_code == 200
            data = response.json()
            assert "by_lag" in data
            assert data.get("source") == "backtest_lag_archive"
