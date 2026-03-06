"""API tests for /inv-planning/variability/* endpoints — IPfeature1."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# /inv-planning/variability/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_variability_summary_200():
    """GET /inv-planning/variability/summary returns 200."""
    summary_row = (10, 4, 3, 2, 1, 0.42, 0.20, 0.40, 0.65, 0.95, 0.12)
    top_rows = [
        ("ITEM001", "LOC1", "A", "cluster_a", 100.0, 60.0, 0.6, 45.0, 0.10, "high"),
    ]

    pool, conn, cursor = _make_pool()
    call_count = 0

    def side_effect_fetchall():
        nonlocal call_count
        call_count += 1
        return top_rows if call_count >= 2 else []

    def side_effect_fetchone():
        return summary_row

    cursor.fetchall.side_effect = side_effect_fetchall
    cursor.fetchone.side_effect = side_effect_fetchone
    cursor.description = [
        ("total_dfus",), ("low_count",), ("medium_count",), ("high_count",),
        ("lumpy_count",), ("avg_cv",), ("cv_p25",), ("cv_p50",), ("cv_p75",),
        ("cv_p95",), ("avg_intermittency_ratio",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "by_class" in data
    assert "cv_percentiles" in data
    assert "top_volatile" in data


@pytest.mark.asyncio
async def test_variability_summary_by_class_keys():
    """Summary response has low/medium/high/lumpy keys."""
    pool, conn, cursor = _make_pool()
    summary_row = (50, 20, 15, 10, 5, 0.45, 0.18, 0.38, 0.62, 0.90, 0.08)
    cursor.fetchone.return_value = summary_row
    cursor.description = [
        ("total_dfus",), ("low_count",), ("medium_count",), ("high_count",),
        ("lumpy_count",), ("avg_cv",), ("cv_p25",), ("cv_p50",), ("cv_p75",),
        ("cv_p95",), ("avg_intermittency_ratio",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/summary")

    assert resp.status_code == 200
    data = resp.json()
    by_class = data["by_class"]
    assert set(by_class.keys()) == {"low", "medium", "high", "lumpy"}


@pytest.mark.asyncio
async def test_variability_summary_abc_vol_filter():
    """Summary accepts abc_vol query param without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5, 3, 2, 0, 0, 0.25, 0.10, 0.20, 0.35, 0.50, 0.05)
    cursor.description = [
        ("total_dfus",), ("low_count",), ("medium_count",), ("high_count",),
        ("lumpy_count",), ("avg_cv",), ("cv_p25",), ("cv_p50",), ("cv_p75",),
        ("cv_p95",), ("avg_intermittency_ratio",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/summary?abc_vol=A")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_variability_summary_empty_db_no_500():
    """Empty DB returns zeros, not 500."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 0


# ---------------------------------------------------------------------------
# /inv-planning/variability/detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_variability_detail_200():
    """GET /inv-planning/variability/detail returns 200 with rows and total."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3,)
    detail_rows = [
        ("ITEM001", "LOC1", "A", "cluster_a",
         120.0, 72.0, 0.60, 55.0, 110.0, 200.0,
         0.5, 1.2, 2, 24, 0.083, "high", "2024-01-01T00:00:00Z"),
    ]
    cursor.fetchall.return_value = detail_rows
    cursor.description = [
        ("item_no",), ("loc",), ("abc_vol",), ("cluster_assignment",),
        ("demand_mean",), ("demand_std",), ("demand_cv",), ("demand_mad",),
        ("demand_p50",), ("demand_p90",), ("demand_skewness",), ("demand_kurtosis",),
        ("zero_demand_months",), ("total_demand_months",), ("intermittency_ratio",),
        ("variability_class",), ("demand_profile_ts",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/detail")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_variability_detail_pagination():
    """Pagination params accepted; limit respected."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100,)
    cursor.fetchall.return_value = []
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/detail?limit=10&offset=20")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_variability_detail_filter_by_class():
    """variability_class filter accepted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = []
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/detail?variability_class=lumpy")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_variability_detail_invalid_sort_falls_back():
    """Invalid sort_by falls back to demand_cv without 422."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/detail?sort_by=invalid_col")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /inv-planning/variability/histogram
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_variability_histogram_200():
    """GET /inv-planning/variability/histogram returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0.05, 1.80)
    cursor.fetchall.return_value = [(1, 10), (2, 8), (3, 5)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/histogram")

    assert resp.status_code == 200
    data = resp.json()
    assert "metric" in data
    assert "bins" in data
    assert isinstance(data["bins"], list)


@pytest.mark.asyncio
async def test_variability_histogram_invalid_metric_falls_back():
    """Invalid metric falls back to demand_cv without 422."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0.0, 1.0)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/variability/histogram?metric=invalid_col")

    assert resp.status_code == 200
    data = resp.json()
    assert data["metric"] == "demand_cv"
