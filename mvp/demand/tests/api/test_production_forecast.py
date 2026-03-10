"""API tests for /forecast/production/* endpoints — F1.1."""

import datetime
import pytest
from unittest.mock import patch, MagicMock
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# /forecast/production/versions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_versions_200():
    """GET /forecast/production/versions returns list of versions."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("2026-03", 1000, 12000, datetime.datetime(2026, 3, 1, 6, 0, 0)),
        ("2026-02", 980,  11760, datetime.datetime(2026, 2, 1, 6, 0, 0)),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production/versions")

    assert resp.status_code == 200
    data = resp.json()
    assert "versions" in data
    assert len(data["versions"]) == 2
    v = data["versions"][0]
    assert "plan_version" in v
    assert "dfu_count" in v
    assert "total_rows" in v
    assert "generated_at" in v


@pytest.mark.asyncio
async def test_versions_empty():
    """Empty table returns empty versions list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production/versions")

    assert resp.status_code == 200
    assert resp.json()["versions"] == []


# ---------------------------------------------------------------------------
# /forecast/production/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_no_data():
    """When table is empty, summary returns zeros without 500."""
    pool, conn, cursor = _make_pool()
    # fetchone returns None → no plan_version → early return
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfu_count"] == 0
    assert data["total_forecast_qty"] == 0.0
    assert data["by_abc_class"] == []


@pytest.mark.asyncio
async def test_summary_with_version_param():
    """When plan_version is provided, skips resolution and returns summary."""
    pool, conn, cursor = _make_pool()
    # plan_version provided → no version-resolution fetchone.
    # Call order: fetchone (summary), fetchall (abc_rows), fetchone (ci_row).
    cursor.fetchone.side_effect = [
        (1000, 55000.0, datetime.datetime(2026, 3, 1, 6, 0, 0)),  # summary row
        (800, 1000, 35.0),                                          # ci_row
    ]
    cursor.fetchall.return_value = [
        ("A", 200, 20000.0),
        ("B", 500, 25000.0),
        ("C", 300, 10000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production/summary?plan_version=2026-03&horizon_months=3")

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_version"] == "2026-03"
    assert data["total_dfu_count"] == 1000
    assert len(data["by_abc_class"]) == 3
    assert "ci_coverage_pct" in data
    assert "avg_ci_width" in data


# ---------------------------------------------------------------------------
# /forecast/production (DFU series)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_404_no_forecast():
    """DFU with no forecast rows returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # no plan_version found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/production?item_no=ITEM001&loc=LOC1")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dfu_200_with_version():
    """DFU forecast with plan_version returns forecast series."""
    pool, conn, cursor = _make_pool()
    # plan_version provided → skip fetchone, only fetchall for rows
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 4, 1),   # forecast_month
            150.0,                         # forecast_qty
            120.0,                         # forecast_qty_lower
            180.0,                         # forecast_qty_upper
            "lgbm_cluster",               # model_id
            2,                             # cluster_id
            1,                             # horizon_months
            True,                          # is_recursive
            "actual",                      # lag_source
            datetime.datetime(2026, 3, 1), # generated_at
        ),
        (
            datetime.date(2026, 5, 1),
            140.0, 110.0, 170.0,
            "lgbm_cluster", 2, 2, True, "predicted",
            datetime.datetime(2026, 3, 1),
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_no=ITEM001&loc=LOC1&plan_version=2026-03"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["item_no"] == "ITEM001"
    assert data["loc"] == "LOC1"
    assert data["plan_version"] == "2026-03"
    assert data["model_id"] == "lgbm_cluster"
    assert len(data["forecasts"]) == 2
    f = data["forecasts"][0]
    assert "forecast_month" in f
    assert "forecast_qty" in f
    assert "lag_source" in f
    assert f["lag_source"] == "actual"


@pytest.mark.asyncio
async def test_dfu_404_version_exists_but_no_rows():
    """Plan version exists but no rows for this DFU → 404."""
    pool, conn, cursor = _make_pool()
    # fetchall returns empty (no rows for this item/loc in the version)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_no=UNKNOWN&loc=LOC1&plan_version=2026-03"
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dfu_horizon_capped():
    """horizon param is capped at 18 without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 4, 1), 100.0, None, None,
            "lgbm_cluster", 1, 1, False, "actual",
            datetime.datetime(2026, 3, 1),
        )
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_no=ITEM001&loc=LOC1&plan_version=2026-02&horizon=99"
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_dfu_18_month_horizon():
    """18-month planning horizon returns all 18 forecast rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 2 + i if i < 11 else i - 10, 1),
            float(100 + i), None, None,
            "lgbm_cluster", 2, i + 1, i > 0, "actual" if i == 0 else "predicted",
            datetime.datetime(2026, 2, 24),
        )
        for i in range(18)
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/production?item_no=ITEM001&loc=LOC1&plan_version=2026-02&horizon=18"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["forecasts"]) == 18
    assert data["forecasts"][0]["lag_source"] == "actual"
    assert data["forecasts"][1]["lag_source"] == "predicted"
