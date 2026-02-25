"""Tests for seasonality columns exposed via generic domain API."""

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
    mock_cursor.description = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = mock_conn

    return pool, mock_conn, mock_cursor


@pytest.mark.asyncio
async def test_dfu_meta_includes_seasonality_columns(mock_pool):
    """Verify /domains/dfu/meta returns the new seasonality columns."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/dfu/meta")
            assert response.status_code == 200
            data = response.json()
            columns = data["columns"]
            for col in [
                "seasonality_profile",
                "seasonality_strength",
                "is_yearly_seasonal",
                "peak_month",
                "trough_month",
                "peak_trough_ratio",
            ]:
                assert col in columns, f"Column {col} missing from DFU meta"


@pytest.mark.asyncio
async def test_dfu_meta_seasonality_in_numeric_fields(mock_pool):
    """Verify seasonality numeric fields appear in numeric_fields."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/dfu/meta")
            assert response.status_code == 200
            data = response.json()
            numeric = data["numeric_fields"]
            for field in ["seasonality_strength", "peak_trough_ratio", "peak_month", "trough_month"]:
                assert field in numeric, f"Field {field} missing from numeric_fields"


@pytest.mark.asyncio
async def test_dfu_suggest_seasonality_profile(mock_pool):
    """Verify typeahead on seasonality_profile field works."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("high",), ("medium",), ("low",), ("none",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/dfu/suggest?field=seasonality_profile")
            assert response.status_code == 200
            data = response.json()
            assert "values" in data


@pytest.mark.asyncio
async def test_dfu_page_with_seasonality_filter(mock_pool):
    """Verify filtering by seasonality_profile via exact match works."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Exact match filter (=prefix convention)
            response = await client.get(
                '/domains/dfu/page?filters={"seasonality_profile":"=high"}'
            )
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_dfu_page_sort_by_seasonality_strength(mock_pool):
    """Verify sorting by seasonality_strength works."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/domains/dfu/page?sort_by=seasonality_strength&sort_dir=desc"
            )
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_seasonality_profiles_endpoint(mock_pool):
    """Verify /domains/dfu/seasonality-profiles returns profile list."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("high_seasonal", 1200),
        ("non_seasonal", 800),
        ("(unknown)", 500),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/dfu/seasonality-profiles")
            assert response.status_code == 200
            data = response.json()
            assert "profiles" in data
            assert len(data["profiles"]) == 3
            assert data["profiles"][0]["profile"] == "high_seasonal"
            assert data["profiles"][0]["count"] == 1200


@pytest.mark.asyncio
async def test_seasonality_profiles_empty(mock_pool):
    """Verify endpoint works with no profiles."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/dfu/seasonality-profiles")
            assert response.status_code == 200
            data = response.json()
            assert data["profiles"] == []
