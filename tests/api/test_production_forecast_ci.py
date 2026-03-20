"""API tests for CI band fields in /forecast/production/* endpoints."""

import datetime
import pytest
import httpx
from httpx import ASGITransport
from unittest.mock import patch

from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Helper: build a standard forecast row tuple (10 columns matching SELECT)
# forecast_month, forecast_qty, forecast_qty_lower, forecast_qty_upper,
# model_id, cluster_id, horizon_months, is_recursive, lag_source, generated_at
# ---------------------------------------------------------------------------

def _forecast_row(lower=120.0, upper=180.0):
    return (
        datetime.date(2026, 3, 1),   # forecast_month
        150.0,                        # forecast_qty
        lower,                        # forecast_qty_lower
        upper,                        # forecast_qty_upper
        "lgbm_cluster",              # model_id
        "cluster_3",                 # cluster_id
        1,                            # horizon_months
        True,                         # is_recursive
        "actual",                     # lag_source
        datetime.datetime(2026, 2, 1),# generated_at
    )


# ---------------------------------------------------------------------------
# 1. GET /forecast/production returns non-null lower/upper when CI is populated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_production_forecast_returns_ci_bands():
    """GET /forecast/production returns forecast_qty_lower/upper when CI is present."""
    pool, conn, cursor = _make_pool()
    # plan_version is provided → skip fetchone; only fetchall for rows
    cursor.fetchall.return_value = [
        _forecast_row(lower=120.0, upper=180.0),
        _forecast_row(lower=128.0, upper=192.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/production?item_no=100320&loc=1401-BULK&plan_version=2026-02"
            )

    assert response.status_code == 200
    data = response.json()
    assert "forecasts" in data
    assert len(data["forecasts"]) == 2
    pt = data["forecasts"][0]
    assert "forecast_qty_lower" in pt
    assert "forecast_qty_upper" in pt
    assert pt["forecast_qty_lower"] == 120.0
    assert pt["forecast_qty_upper"] == 180.0


# ---------------------------------------------------------------------------
# 2. GET /forecast/production returns null lower/upper when CI is absent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_production_forecast_null_ci_bands():
    """Forecast rows with NULL CI bands yield None in response."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [_forecast_row(lower=None, upper=None)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/production?item_no=100320&loc=1401-BULK&plan_version=2026-02"
            )

    assert response.status_code == 200
    data = response.json()
    pt = data["forecasts"][0]
    assert pt["forecast_qty_lower"] is None
    assert pt["forecast_qty_upper"] is None


# ---------------------------------------------------------------------------
# 3. GET /forecast/production/summary includes ci_coverage_pct and avg_ci_width
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_ci_coverage_fields_present():
    """GET /forecast/production/summary response includes ci_coverage_pct and avg_ci_width."""
    pool, conn, cursor = _make_pool()
    # plan_version provided → no version-resolution fetchone
    # Call order: fetchone (summary), fetchall (abc_rows), fetchone (ci_row)
    cursor.fetchone.side_effect = [
        (18000, 172276000.0, datetime.datetime(2026, 2, 1)),  # summary row
        (155000, 172276, 45.5),                                # ci_row
    ]
    cursor.fetchall.return_value = [
        ("A", 5000, 50000000.0),
        ("B", 10000, 80000000.0),
        ("C", 20000, 42500000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/production/summary?plan_version=2026-02")

    assert response.status_code == 200
    data = response.json()
    assert "ci_coverage_pct" in data
    assert "avg_ci_width" in data
    # 155000 / 172276 * 100 ≈ 89.97 → rounded to 1 decimal
    assert isinstance(data["ci_coverage_pct"], float)
    assert data["avg_ci_width"] == pytest.approx(45.5)


# ---------------------------------------------------------------------------
# 4. ci_coverage_pct is computed correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_ci_coverage_pct_computed():
    """ci_coverage_pct = round(ci_count / total_count * 100, 1)."""
    pool, conn, cursor = _make_pool()
    # 75 out of 100 rows have CI bands → 75.0%
    cursor.fetchone.side_effect = [
        (10, 10000.0, datetime.datetime(2026, 2, 1)),  # summary row
        (75, 100, 60.0),                                # ci_row: ci_count=75, total=100
    ]
    cursor.fetchall.return_value = [("A", 10, 10000.0)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/production/summary?plan_version=2026-02")

    assert response.status_code == 200
    data = response.json()
    assert data["ci_coverage_pct"] == 75.0
    assert data["avg_ci_width"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# 5. avg_ci_width is None when no CI rows exist; ci_coverage_pct is 0.0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_ci_coverage_zero_when_no_ci():
    """When no rows have CI bands, ci_coverage_pct=0.0 and avg_ci_width=None."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (5, 5000.0, datetime.datetime(2026, 2, 1)),  # summary row
        (0, 50, None),                                # ci_row: no CI coverage, avg is NULL
    ]
    cursor.fetchall.return_value = [("B", 5, 5000.0)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/forecast/production/summary?plan_version=2026-02")

    assert response.status_code == 200
    data = response.json()
    assert data["ci_coverage_pct"] == 0.0
    assert data["avg_ci_width"] is None


# ---------------------------------------------------------------------------
# Branch coverage: explicit plan_version param (line 68) + category filter
# (lines 180-186) + summary explicit version (line 172)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_production_forecast_explicit_plan_version():
    """plan_version param provided → version lookup skipped, row returned (line 68)."""
    pool, conn, cursor = _make_pool()
    # Only one fetchall call — no fetchone needed for version lookup
    cursor.fetchall.return_value = [_forecast_row()]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/production",
                params={
                    "item_no": "100320",
                    "loc": "1401-BULK",
                    "plan_version": "2026-01",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["plan_version"] == "2026-01"
    assert len(data["forecasts"]) == 1


@pytest.mark.asyncio
async def test_summary_explicit_plan_version_skips_lookup():
    """Summary with explicit plan_version skips version fetchone (line 172).

    Call order (explicit plan_version):
      fetchone(1): summary row (COUNT, SUM, MIN)
      fetchall(1): ABC rows
      fetchone(2): CI stats row
    """
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (1000, 50000.0, datetime.datetime(2026, 2, 1)),   # summary row
        (800, 1000, 45.0),                                 # CI stats row
    ]
    cursor.fetchall.return_value = [
        ("A", 400, 20000.0),
        ("B", 600, 30000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/production/summary?plan_version=2026-01"
            )

    assert response.status_code == 200
    data = response.json()
    assert data["plan_version"] == "2026-01"
    assert data["total_dfu_count"] == 1000


@pytest.mark.asyncio
async def test_summary_category_filter_applied():
    """category param appended to WHERE clause in summary (lines 184-186).

    Call order (no explicit plan_version, category filter):
      fetchone(1): version lookup
      fetchone(2): summary row
      fetchall(1): ABC rows
      fetchone(3): CI stats row
    """
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        ("2026-02",),                                          # version lookup
        (500, 25000.0, datetime.datetime(2026, 2, 1)),         # summary row
        (400, 500, 55.0),                                      # CI stats row
    ]
    cursor.fetchall.return_value = [("A", 200, 12000.0)]       # ABC rows

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/forecast/production/summary?category=BEVERAGES"
            )

    assert response.status_code == 200
    data = response.json()
    assert data["total_dfu_count"] == 500
