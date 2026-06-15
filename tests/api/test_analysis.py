"""API tests for /sku/analysis endpoint.

Feature 17 — DFU Analysis: sales vs multi-model forecast overlay.
Router: api/routers/analysis.py, path: GET /sku/analysis.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool


# ---------------------------------------------------------------------------
# Mock pool factory
# The analysis router uses:
#   with get_conn() as conn, conn.cursor() as cur:
#       cur.execute(...); cur.fetchall()  -- repeated 4 times
# get_conn() calls _get_pool().connection() so patching _get_pool works.
# Tests use the shared make_pool factory (tests/api/conftest.py); per-call
# fetchall values are wired via fetchall_returns (cursor.fetchall.side_effect).
# ---------------------------------------------------------------------------


def _empty_4():
    """Return a fresh list of 4 empty lists for the 4 fetchall() calls in analysis:
    1. sales_measures_sql
    2. forecast_sql
    3. actual_sql (dedup)
    4. dfu_sql (only when item or location provided)
    Must be called fresh per test — side_effect consumes the list.
    """
    return [[], [], [], []]


# ---------------------------------------------------------------------------
# Basic 200 tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sku_analysis_item_location_200():
    """GET /sku/analysis with item+location returns 200."""
    pool, conn, cursor = make_pool(fetchall_returns=_empty_4())
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "ABC123", "location": "W001", "mode": "item_location"},
            )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_sku_analysis_response_structure():
    """Response contains expected top-level keys."""
    pool, conn, cursor = make_pool(fetchall_returns=_empty_4())
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "ITEM1", "location": "LOC1", "mode": "item_location"},
            )
    data = resp.json()
    for key in ("mode", "item", "location", "points", "models", "series", "model_monthly", "dfu_attributes"):
        assert key in data, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_sku_analysis_returns_item_desc():
    """U3.5 — the response carries the item's human-readable description so the
    Item Analysis breadcrumb can show 'Item 185690 — DAMMANN JARDIN BLEU TEA'
    instead of a bare numeric code. dim_item.item_desc is fetched per item."""
    pool, conn, cursor = make_pool(fetchall_returns=_empty_4())
    cursor.fetchone.return_value = ("DAMMANN JARDIN BLEU TEA(96CT)",)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "185690", "mode": "item_at_all_locations"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert "item_desc" in data
    assert data["item_desc"] == "DAMMANN JARDIN BLEU TEA(96CT)"


@pytest.mark.asyncio
async def test_sku_analysis_mode_reflects_in_response():
    """Response mode field matches the requested mode."""
    pool, conn, cursor = make_pool(fetchall_returns=_empty_4())
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "X", "location": "Y", "mode": "item_location"},
            )
    assert resp.json()["mode"] == "item_location"


# ---------------------------------------------------------------------------
# Mode-specific tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sku_analysis_mode_all_items_at_location():
    """mode=all_items_at_location with location param returns 200."""
    pool, conn, cursor = make_pool(fetchall_returns=[[], [], [], []])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"location": "W001", "mode": "all_items_at_location"},
            )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "all_items_at_location"


@pytest.mark.asyncio
async def test_sku_analysis_mode_item_at_all_locations():
    """mode=item_at_all_locations with item param returns 200."""
    pool, conn, cursor = make_pool(fetchall_returns=[[], [], [], []])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "ABC123", "mode": "item_at_all_locations"},
            )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "item_at_all_locations"


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sku_analysis_invalid_mode_422():
    """Invalid mode value returns 422."""
    pool, conn, cursor = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "X", "location": "Y", "mode": "invalid_mode"},
            )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_sku_analysis_item_location_mode_missing_item_422():
    """item_location mode with missing item → 422."""
    pool, conn, cursor = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"location": "W001", "mode": "item_location"},
            )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_sku_analysis_item_location_mode_missing_location_422():
    """item_location mode with missing location → 422."""
    pool, conn, cursor = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "ITEM1", "mode": "item_location"},
            )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_sku_analysis_all_items_at_location_missing_location_422():
    """all_items_at_location mode without location → 422."""
    pool, conn, cursor = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"mode": "all_items_at_location"},
            )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_sku_analysis_item_at_all_locations_missing_item_422():
    """item_at_all_locations mode without item → 422."""
    pool, conn, cursor = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"mode": "item_at_all_locations"},
            )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Empty data tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sku_analysis_empty_db_returns_empty_series():
    """When DB returns no rows, series and dfu_attributes are empty lists."""
    pool, conn, cursor = make_pool(fetchall_returns=_empty_4())
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "NOEXIST", "location": "NOLOC", "mode": "item_location"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["series"] == []
    assert data["dfu_attributes"] == []
    assert data["models"] == []


@pytest.mark.asyncio
async def test_sku_analysis_with_data_populates_series():
    """When sales data is returned, series contains entries with month and qty fields."""
    import datetime
    sales_rows = [
        (datetime.date(2024, 1, 1), 500.0, 550.0, 520.0),
        (datetime.date(2024, 2, 1), 480.0, 500.0, 490.0),
    ]
    forecast_rows = [
        (datetime.date(2024, 1, 1), "external", 510.0, 500.0),
        (datetime.date(2024, 2, 1), "external", 490.0, 480.0),
    ]
    actual_rows = [
        (datetime.date(2024, 1, 1), 500.0),
        (datetime.date(2024, 2, 1), 480.0),
    ]
    dfu_rows = []

    pool, conn, cursor = make_pool(
        fetchall_returns=[sales_rows, forecast_rows, actual_rows, dfu_rows]
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "ITEM1", "location": "LOC1", "mode": "item_location"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["series"]) == 2
    assert "month" in data["series"][0]
    assert "qty_shipped" in data["series"][0]
    assert "models" in data
    assert "external" in data["models"]


# ---------------------------------------------------------------------------
# Seasonality profile filter (Feature 32)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sku_analysis_seasonality_profile_filter():
    """seasonality_profile param is accepted and does not raise errors."""
    pool, conn, cursor = make_pool(fetchall_returns=[[], [], [], []])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={
                    "location": "LOC1",
                    "mode": "all_items_at_location",
                    "seasonality_profile": "peak_summer",
                },
            )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Points parameter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sku_analysis_custom_points_param():
    """Custom points parameter is reflected in response."""
    pool, conn, cursor = make_pool(fetchall_returns=_empty_4())
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "X", "location": "Y", "mode": "item_location", "points": 12},
            )
    assert resp.status_code == 200
    assert resp.json()["points"] == 12


@pytest.mark.asyncio
async def test_sku_analysis_points_too_low_422():
    """points < 3 should return 422 (ge=3 constraint)."""
    pool, conn, cursor = make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "X", "location": "Y", "mode": "item_location", "points": 1},
            )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# model_monthly structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sku_analysis_model_monthly_structure():
    """model_monthly is a dict keyed by model_id, each value is a list of dicts."""
    import datetime
    forecast_rows = [
        (datetime.date(2024, 3, 1), "lgbm_cluster", 300.0, 290.0),
    ]
    pool, conn, cursor = make_pool(
        fetchall_returns=[[], forecast_rows, [(datetime.date(2024, 3, 1), 290.0)], []]
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/sku/analysis",
                params={"item": "X", "location": "Y", "mode": "item_location"},
            )
    assert resp.status_code == 200
    data = resp.json()
    mm = data["model_monthly"]
    assert isinstance(mm, dict)
    assert "lgbm_cluster" in mm
    assert isinstance(mm["lgbm_cluster"], list)
    entry = mm["lgbm_cluster"][0]
    for key in ("month", "forecast", "actual"):
        assert key in entry
