"""Tests for GET /domains/{domain}/distinct endpoint."""

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


# ---------------------------------------------------------------------------
# Returns distinct values for allowed column
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distinct_returns_values_for_allowed_column(mock_pool):
    """GET /domains/item/distinct?column=brand_name returns distinct brand values."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("BrandA",), ("BrandB",), ("BrandC",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name")
            assert resp.status_code == 200
            data = resp.json()
            assert "column" in data
            assert "values" in data
            assert "total" in data
            assert data["column"] == "brand_name"
            assert data["values"] == ["BrandA", "BrandB", "BrandC"]
            assert data["total"] == 3


# ---------------------------------------------------------------------------
# Returns error 400 for disallowed column
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distinct_rejects_disallowed_column(mock_pool):
    """GET /domains/item/distinct?column=item_status returns 400 — not in allowed list."""
    pool, _, cursor = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=item_status")
            assert resp.status_code == 400
            data = resp.json()
            assert "detail" in data


@pytest.mark.asyncio
async def test_distinct_rejects_column_on_domain_without_allowed(mock_pool):
    """GET /domains/sales/distinct?column=qty returns 400 — sales has no allowed columns."""
    pool, _, cursor = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/sales/distinct?column=qty")
            assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Respects limit parameter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distinct_respects_limit_parameter(mock_pool):
    """GET /domains/item/distinct?column=brand_name&limit=2 caps the result count."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("BrandA",), ("BrandB",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name&limit=2")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 2
            assert len(data["values"]) == 2


@pytest.mark.asyncio
async def test_distinct_default_limit(mock_pool):
    """Without explicit limit, default is 100 — endpoint should still work."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("BrandA",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data["values"], list)


# ---------------------------------------------------------------------------
# Filters by search prefix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distinct_filters_by_search_prefix(mock_pool):
    """GET /domains/item/distinct?column=brand_name&search=Br returns only matching brands."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("BrandA",), ("BrandB",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name&search=Br")
            assert resp.status_code == 200
            data = resp.json()
            assert data["column"] == "brand_name"
            assert data["values"] == ["BrandA", "BrandB"]


@pytest.mark.asyncio
async def test_distinct_empty_search_returns_all(mock_pool):
    """GET /domains/item/distinct?column=brand_name&search= is same as no search."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("X",), ("Y",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name&search=")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 2


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distinct_response_structure(mock_pool):
    """Response body must contain exactly {column, values, total}."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("state1",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/location/distinct?column=state_id")
            assert resp.status_code == 200
            data = resp.json()
            assert set(data.keys()) == {"column", "values", "total"}
            assert isinstance(data["column"], str)
            assert isinstance(data["values"], list)
            assert isinstance(data["total"], int)


@pytest.mark.asyncio
async def test_distinct_empty_result(mock_pool):
    """No matching values returns empty list with total 0."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name&search=zzz_nonexistent")
            assert resp.status_code == 200
            data = resp.json()
            assert data["values"] == []
            assert data["total"] == 0


@pytest.mark.asyncio
async def test_distinct_missing_column_param(mock_pool):
    """Missing required column param returns 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct")
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_distinct_other_allowed_domains(mock_pool):
    """Verify distinct works for other allowed domain/column combos."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("group1",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # customer domain, rpt_channel_desc column
            resp = await client.get("/domains/customer/distinct?column=rpt_channel_desc")
            assert resp.status_code == 200
            data = resp.json()
            assert data["column"] == "rpt_channel_desc"
            assert data["values"] == ["group1"]

            # dfu domain, cluster_assignment column
            cursor.fetchall.return_value = [("high_volume_steady",)]
            resp = await client.get("/domains/dfu/distinct?column=cluster_assignment")
            assert resp.status_code == 200
            data = resp.json()
            assert data["column"] == "cluster_assignment"


@pytest.mark.asyncio
async def test_distinct_values_are_strings(mock_pool):
    """Even numeric DB values should be cast to strings in the response."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [(42,), (99,)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name")
            assert resp.status_code == 200
            data = resp.json()
            # Endpoint casts with str()
            assert data["values"] == ["42", "99"]


# ---------------------------------------------------------------------------
# item_no and location_id are now allowed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distinct_allows_item_no(mock_pool):
    """GET /domains/item/distinct?column=item_no returns 200 — now in allowed list."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("100320",), ("100321",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=item_no&search=100")
            assert resp.status_code == 200
            data = resp.json()
            assert data["column"] == "item_no"
            assert data["values"] == ["100320", "100321"]


@pytest.mark.asyncio
async def test_distinct_allows_location_id(mock_pool):
    """GET /domains/location/distinct?column=location_id returns 200."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("1401-BULK",), ("1402-BULK",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/location/distinct?column=location_id&search=14")
            assert resp.status_code == 200
            data = resp.json()
            assert data["column"] == "location_id"
            assert data["values"] == ["1401-BULK", "1402-BULK"]


# ---------------------------------------------------------------------------
# Cascading filter tests — narrowing dropdown options by other active filters
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_distinct_cascade_brand_filtered_by_location(mock_pool):
    """Brand dropdown narrows when location filter is active."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("BrandX",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name&location=1401-BULK")
            assert resp.status_code == 200
            data = resp.json()
            assert data["values"] == ["BrandX"]
            # Verify the SQL used dim_dfu-based cascading (not plain dim_item table)
            executed_sql = cursor.execute.call_args[0][0]
            assert "dim_dfu" in executed_sql
            assert "dim_item" in executed_sql


@pytest.mark.asyncio
async def test_distinct_cascade_location_filtered_by_brand(mock_pool):
    """Location dropdown narrows when brand filter is active."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("LOC1",), ("LOC2",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/location/distinct?column=location_id&brand=Nike")
            assert resp.status_code == 200
            data = resp.json()
            assert data["values"] == ["LOC1", "LOC2"]
            executed_sql = cursor.execute.call_args[0][0]
            assert "dim_dfu" in executed_sql


@pytest.mark.asyncio
async def test_distinct_cascade_item_filtered_by_brand_and_location(mock_pool):
    """Item dropdown narrows when both brand and location are active."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("100320",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=item_no&brand=Nike&location=1401-BULK&search=100")
            assert resp.status_code == 200
            data = resp.json()
            assert data["values"] == ["100320"]
            executed_sql = cursor.execute.call_args[0][0]
            assert "dim_dfu" in executed_sql
            assert "ILIKE" in executed_sql


@pytest.mark.asyncio
async def test_distinct_cascade_market_filtered_by_brand(mock_pool):
    """Market (state_id) dropdown narrows when brand filter is active."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("CA",), ("TX",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/location/distinct?column=state_id&brand=Nike")
            assert resp.status_code == 200
            data = resp.json()
            assert data["values"] == ["CA", "TX"]
            executed_sql = cursor.execute.call_args[0][0]
            assert "dim_dfu" in executed_sql
            assert "dim_location" in executed_sql


@pytest.mark.asyncio
async def test_distinct_no_cascade_without_filter_params(mock_pool):
    """Without cascade params, uses original simple query path."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("A",), ("B",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name")
            assert resp.status_code == 200
            executed_sql = cursor.execute.call_args[0][0]
            assert "dim_dfu" not in executed_sql
            assert "dim_item" in executed_sql


@pytest.mark.asyncio
async def test_distinct_cascade_multiple_values(mock_pool):
    """Cascade params with comma-separated multiple values work correctly."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [("ItemA",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/domains/item/distinct?column=brand_name&location=1401-BULK,1402-BULK")
            assert resp.status_code == 200
            executed_sql = cursor.execute.call_args[0][0]
            assert "dim_dfu" in executed_sql
            # Verify multi-value was passed as array
            executed_params = cursor.execute.call_args[0][1]
            assert ["1401-BULK", "1402-BULK"] in executed_params
