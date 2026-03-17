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


# ---------------------------------------------------------------------------
# Health and root endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_endpoint(mock_pool):
    """GET / returns status ok and links."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ui_url" in data


@pytest.mark.asyncio
async def test_health_endpoint(mock_pool):
    """GET /health returns domain list."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "domains" in data
    assert "item" in data["domains"]


# ---------------------------------------------------------------------------
# Domain suggest with valid field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_domain_suggest_valid_field(mock_pool):
    """Suggest with valid field returns values."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("Apple",), ("Apricot",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/item/suggest?field=item_no&q=Ap")
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "item"
    assert "values" in data
    assert len(data["values"]) == 2


@pytest.mark.asyncio
async def test_domain_suggest_invalid_field(mock_pool):
    """Suggest with invalid field returns 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/item/suggest?field=nonexistent_column")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Sample pair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_pair_returns_values(mock_pool):
    """Sample pair for forecast domain returns item + location."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = ("100320", "1401-BULK")
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/forecast/sample-pair")
    assert response.status_code == 200
    data = response.json()
    assert data["item"] == "100320"
    assert data["location"] == "1401-BULK"


@pytest.mark.asyncio
async def test_sample_pair_no_data(mock_pool):
    """Sample pair with no rows returns null values."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/forecast/sample-pair")
    assert response.status_code == 200
    data = response.json()
    assert data["item"] is None
    assert data["location"] is None


@pytest.mark.asyncio
async def test_sample_pair_domain_without_item_loc(mock_pool):
    """Sample pair for time domain (no item/location) returns 404."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/time/sample-pair")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# DFU count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dfu_count_no_filters(mock_pool):
    """DFU count with no filters returns total count."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (5000,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/dfu/count")
    assert response.status_code == 200
    assert response.json()["count"] == 5000


@pytest.mark.asyncio
async def test_dfu_count_with_filters(mock_pool):
    """DFU count with brand+location filters."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (42,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/domains/dfu/count?brand=BrandA&location=LOC1,LOC2"
            )
    assert response.status_code == 200
    assert response.json()["count"] == 42


# ---------------------------------------------------------------------------
# Distinct values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_distinct_values_valid(mock_pool):
    """Distinct values for allowed column returns values list."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("BrandA",), ("BrandB",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/domains/item/distinct?column=brand_name"
            )
    assert response.status_code == 200
    data = response.json()
    assert data["column"] == "brand_name"
    assert "BrandA" in data["values"]
    assert data["total"] == 2


@pytest.mark.asyncio
async def test_distinct_values_invalid_column(mock_pool):
    """Distinct values for disallowed column returns 400."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/domains/item/distinct?column=nonexistent"
            )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_distinct_with_cascade_filters(mock_pool):
    """Distinct with cascading filter params narrows results via dim_dfu."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("BrandX",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/domains/item/distinct?column=brand_name&location=LOC1"
            )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1


# ---------------------------------------------------------------------------
# Backward-compatible alias endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_items_endpoint(mock_pool):
    """GET /items returns item domain data (list format)."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/items")
    assert response.status_code == 200
    data = response.json()
    # list_domain returns whatever format the function produces
    assert isinstance(data, (list, dict))


@pytest.mark.asyncio
async def test_legacy_forecasts_page_endpoint(mock_pool):
    """GET /forecasts/page returns paginated forecast data."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecasts/page?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data


# ---------------------------------------------------------------------------
# Analytics endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analytics_time_domain_disabled(mock_pool):
    """Analytics is disabled for the time domain."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/time/analytics")
    assert response.status_code == 404
    assert "disabled" in response.json()["detail"]


@pytest.mark.asyncio
async def test_analytics_item_domain(mock_pool):
    """Analytics for item domain returns summary and config."""
    pool, _, cursor = mock_pool
    cursor.fetchone.side_effect = [
        (100, 5000.0, 50.0, 1.0, 200.0),   # summary
        (None, None),                         # min/max date
    ]
    cursor.fetchall.return_value = [("CategoryA", 3000.0)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/domains/item/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "item"
    assert "summary" in data
    assert "config" in data
    assert "available" in data
