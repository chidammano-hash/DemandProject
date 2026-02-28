"""Tests for domain endpoints — /domains, /domains/{domain}/meta, /domains/{domain}/page."""

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
async def test_list_domains(mock_pool):
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains")
            assert response.status_code == 200
            data = response.json()
            assert "domains" in data
            domains = data["domains"]
            assert "item" in domains
            assert "location" in domains
            assert "sales" in domains
            assert "forecast" in domains
            assert len(domains) == 8
            assert "inventory" in domains


@pytest.mark.asyncio
async def test_domain_meta_item(mock_pool):
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/item/meta")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "item"
            assert "columns" in data
            assert "item_no" in data["columns"]
            assert "numeric_fields" in data
            assert "date_fields" in data


@pytest.mark.asyncio
async def test_domain_meta_invalid_domain(mock_pool):
    """Invalid domain returns 404."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/nonexistent/meta")
            assert response.status_code == 404


@pytest.mark.asyncio
async def test_domain_meta_all_domains(mock_pool):
    """Every valid domain should return meta successfully."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for domain in ["item", "location", "customer", "time", "dfu", "sales", "forecast"]:
                response = await client.get(f"/domains/{domain}/meta")
                assert response.status_code == 200, f"Failed for domain: {domain}"
                data = response.json()
                assert data["name"] == domain


@pytest.mark.asyncio
async def test_domain_page_returns_structure(mock_pool):
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/item/page?limit=50&offset=0")
            assert response.status_code == 200
            data = response.json()
            assert "total" in data
            assert "limit" in data
            assert "offset" in data


@pytest.mark.asyncio
async def test_domain_suggest_missing_field(mock_pool):
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/item/suggest")
            assert response.status_code == 422  # missing required query param
