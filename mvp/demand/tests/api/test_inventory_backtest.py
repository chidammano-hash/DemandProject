"""Tests for inventory-backtest API endpoints."""

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
# /inventory-backtest/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_returns_200(mock_pool):
    """GET /inventory-backtest/summary returns 200 with per-model metrics."""
    pool, _, cursor = mock_pool
    # model_id, dfu_months, stockout_count, excess_count, wape, bias, avg_dos
    cursor.fetchall.return_value = [
        ("external", 5000, 150, 400, 28.5, 3.2, 42.0),
        ("lgbm_cluster", 5000, 100, 350, 22.1, -1.5, 38.5),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert "models" in data
            assert "by_model" in data
            assert "excess_dos_threshold" in data
            assert len(data["models"]) == 2
            ext = data["by_model"]["external"]
            assert "dfu_months" in ext
            assert "stockout_count" in ext
            assert "stockout_rate" in ext
            assert "excess_count" in ext
            assert "excess_rate" in ext
            assert "cycle_service_level" in ext
            assert "avg_dos" in ext
            assert "wape" in ext
            assert "bias" in ext
            assert ext["dfu_months"] == 5000
            assert ext["stockout_count"] == 150


@pytest.mark.asyncio
async def test_summary_with_filters(mock_pool):
    """GET /inventory-backtest/summary filters pass through correctly."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("external", 200, 10, 20, 30.0, 2.0, 40.0),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inventory-backtest/summary"
                "?models=external,lgbm_cluster"
                "&month_from=2025-01-01"
                "&cluster_assignment=high_volume_steady"
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["models"]) == 1


@pytest.mark.asyncio
async def test_summary_empty(mock_pool):
    """Empty data returns empty by_model dict."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["models"] == []
            assert data["by_model"] == {}


@pytest.mark.asyncio
async def test_summary_custom_threshold(mock_pool):
    """Custom excess_dos_threshold is accepted and returned."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = [
        ("external", 1000, 50, 600, 25.0, 1.0, 30.0),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/summary?excess_dos_threshold=60")
            assert resp.status_code == 200
            data = resp.json()
            assert data["excess_dos_threshold"] == 60


# ---------------------------------------------------------------------------
# /inventory-backtest/trend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trend_returns_200(mock_pool):
    """GET /inventory-backtest/trend returns 200 with nested monthly data."""
    pool, _, cursor = mock_pool
    # month_start, model_id, dfu_months, stockout_count, excess_count, avg_dos, wape
    cursor.fetchall.return_value = [
        ("2025-03-01", "external", 500, 15, 40, 42.0, 29.0),
        ("2025-03-01", "lgbm_cluster", 500, 10, 35, 38.0, 22.0),
        ("2025-04-01", "external", 500, 18, 42, 43.0, 28.5),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/trend")
            assert resp.status_code == 200
            data = resp.json()
            assert "trend" in data
            assert len(data["trend"]) == 2
            first = data["trend"][0]
            assert first["month"] == "2025-03-01"
            assert "by_model" in first
            assert "external" in first["by_model"]
            ext = first["by_model"]["external"]
            assert "stockout_rate" in ext
            assert "excess_rate" in ext
            assert "avg_dos" in ext
            assert "wape" in ext


@pytest.mark.asyncio
async def test_trend_empty(mock_pool):
    """Empty trend data returns empty list."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/trend")
            assert resp.status_code == 200
            data = resp.json()
            assert data["trend"] == []


# ---------------------------------------------------------------------------
# /inventory-backtest/root-cause
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_root_cause_returns_200(mock_pool):
    """GET /inventory-backtest/root-cause returns breakdown by bias direction."""
    pool, _, cursor = mock_pool
    # stockout_total, stockout_under, stockout_over, stockout_exact,
    # excess_total, excess_over, excess_under, excess_exact
    cursor.fetchone.return_value = (450, 320, 80, 50, 1200, 950, 150, 100)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/root-cause?model_id=lgbm_cluster")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_id"] == "lgbm_cluster"
            assert data["stockout_total"] == 450
            assert data["stockout_under_forecast"] == 320
            assert data["stockout_over_forecast"] == 80
            assert data["stockout_exact"] == 50
            assert data["excess_total"] == 1200
            assert data["excess_over_forecast"] == 950
            assert data["excess_under_forecast"] == 150
            assert data["excess_exact"] == 100


@pytest.mark.asyncio
async def test_root_cause_missing_model(mock_pool):
    """Omitting model_id should return 422."""
    pool, _, _ = mock_pool
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/root-cause")
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /inventory-backtest/detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detail_returns_200(mock_pool):
    """GET /inventory-backtest/detail returns paginated event rows."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (1,)
    # item_no, loc, month_start, model_id, forecast, actual_demand,
    # eom_qty_on_hand, dos, forecast_error, abs_error, bias_direction,
    # seasonality_profile, zero_velocity_flag
    cursor.fetchall.return_value = [
        ("100320", "1401-BULK", "2025-06-01", "lgbm_cluster", 120.5, 150.0, 0, None, -29.5, 29.5, "under", "seasonal_high", False),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/detail")
            assert resp.status_code == 200
            data = resp.json()
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert "rows" in data
            assert len(data["rows"]) == 1
            row = data["rows"][0]
            assert row["item_no"] == "100320"
            assert row["loc"] == "1401-BULK"
            assert row["model_id"] == "lgbm_cluster"
            assert row["event_type"] == "stockout"
            assert row["forecast_error"] == -29.5
            assert row["bias_direction"] == "under"
            assert row["seasonality_profile"] == "seasonal_high"
            assert row["zero_velocity_flag"] is False


@pytest.mark.asyncio
async def test_detail_event_filter(mock_pool):
    """event_type=stockout filters to stockout events only."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/detail?event_type=stockout")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_detail_pagination(mock_pool):
    """limit and offset are respected in the response."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (500,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/detail?limit=10&offset=20")
            assert resp.status_code == 200
            data = resp.json()
            assert data["limit"] == 10
            assert data["offset"] == 20
            assert data["total"] == 500


@pytest.mark.asyncio
async def test_detail_sort(mock_pool):
    """sort_by and sort_dir are accepted; invalid column falls back."""
    pool, _, cursor = mock_pool
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Valid sort column
            resp = await client.get("/inventory-backtest/detail?sort_by=forecast&sort_dir=asc")
            assert resp.status_code == 200
            # Invalid sort column falls back gracefully
            resp2 = await client.get("/inventory-backtest/detail?sort_by=nonexistent")
            assert resp2.status_code == 200


@pytest.mark.asyncio
async def test_seasonality_profile_filter(mock_pool):
    """seasonality_profile filter is accepted on all endpoints."""
    pool, _, cursor = mock_pool
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (0,)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inventory-backtest/summary?seasonality_profile=seasonal_high")
            assert resp.status_code == 200
            resp2 = await client.get("/inventory-backtest/trend?seasonality_profile=flat")
            assert resp2.status_code == 200
            resp3 = await client.get("/inventory-backtest/detail?seasonality_profile=seasonal_high")
            assert resp3.status_code == 200
