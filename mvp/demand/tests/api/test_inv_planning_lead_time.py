"""API tests for /inv-planning/lead-time/* endpoints — IPfeature2."""

import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport


def _make_pool(fetchall_return=None, fetchone_return=None):
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return or (0,)
    cursor.description = []

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn, cursor


# ---------------------------------------------------------------------------
# /inv-planning/lead-time/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lt_summary_200():
    """GET /inv-planning/lead-time/summary returns 200."""
    pool, conn, cursor = _make_pool()
    summary_row = (50, 30, 15, 5, 0.18, 28.5, 0.14, 0.55)
    top_rows = [
        ("ITEM001", "LOC1", 35.0, 18.0, 0.51, 14.0, 60.0, 8, "volatile"),
    ]
    call_count = 0

    def side_fetchone():
        return summary_row

    def side_fetchall():
        nonlocal call_count
        call_count += 1
        return top_rows if call_count >= 2 else []

    cursor.fetchone.side_effect = side_fetchone
    cursor.fetchall.side_effect = side_fetchall
    cursor.description = [
        ("total_profiles",), ("stable_count",), ("moderate_count",), ("volatile_count",),
        ("avg_lt_cv",), ("avg_lt_mean_days",), ("lt_cv_p50",), ("lt_cv_p95",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "by_class" in data
    assert "top_volatile" in data
    assert "avg_lt_cv" in data


@pytest.mark.asyncio
async def test_lt_summary_by_class_keys():
    """Summary response has stable/moderate/volatile keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (50, 30, 15, 5, 0.18, 28.5, 0.14, 0.55)
    cursor.description = [
        ("total_profiles",), ("stable_count",), ("moderate_count",), ("volatile_count",),
        ("avg_lt_cv",), ("avg_lt_mean_days",), ("lt_cv_p50",), ("lt_cv_p95",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/summary")

    assert resp.status_code == 200
    by_class = resp.json()["by_class"]
    assert set(by_class.keys()) == {"stable", "moderate", "volatile"}


@pytest.mark.asyncio
async def test_lt_summary_empty_db_no_500():
    """Empty DB returns zeros, not 500."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/summary")

    assert resp.status_code == 200
    assert resp.json()["total_profiles"] == 0


@pytest.mark.asyncio
async def test_lt_summary_abc_vol_filter():
    """abc_vol query param accepted without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5, 4, 1, 0, 0.10, 25.0, 0.09, 0.20)
    cursor.description = [
        ("total_profiles",), ("stable_count",), ("moderate_count",), ("volatile_count",),
        ("avg_lt_cv",), ("avg_lt_mean_days",), ("lt_cv_p50",), ("lt_cv_p95",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/summary?abc_vol=A")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /inv-planning/lead-time/profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lt_profile_200():
    """GET /inv-planning/lead-time/profile returns 200 with rows and total."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3,)
    detail_rows = [
        ("ITEM001", "LOC1", 28.5, 8.2, 0.29, 14.0, 45.0,
         21.0, 28.0, 35.0, 42.0, 7, 6, "moderate", "2024-01-01T00:00:00Z"),
    ]
    cursor.fetchall.return_value = detail_rows
    cursor.description = [
        ("item_no",), ("loc",),
        ("lt_mean_days",), ("lt_std_days",), ("lt_cv",),
        ("lt_min_days",), ("lt_max_days",),
        ("lt_p25_days",), ("lt_p50_days",), ("lt_p75_days",), ("lt_p95_days",),
        ("observation_count",), ("observation_months",),
        ("lt_variability_class",), ("computed_at",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/profile")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_lt_profile_pagination():
    """Pagination params accepted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100,)
    cursor.fetchall.return_value = []
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/profile?limit=10&offset=20")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_lt_profile_filter_by_class():
    """lt_variability_class filter accepted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = []
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/profile?lt_variability_class=volatile")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_lt_profile_invalid_sort_falls_back():
    """Invalid sort_by falls back to lt_cv without 422."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    cursor.description = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/profile?sort_by=injection_attempt")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /inv-planning/lead-time/histogram
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lt_histogram_200():
    """GET /inv-planning/lead-time/histogram returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0.05, 0.80)
    cursor.fetchall.return_value = [(1, 12), (2, 8), (3, 5)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/histogram")

    assert resp.status_code == 200
    data = resp.json()
    assert "metric" in data
    assert "bins" in data
    assert isinstance(data["bins"], list)


@pytest.mark.asyncio
async def test_lt_histogram_invalid_metric_falls_back():
    """Invalid metric falls back to lt_cv without 422."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0.0, 1.0)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/lead-time/histogram?metric=bad_col")

    assert resp.status_code == 200
    assert resp.json()["metric"] == "lt_cv"
