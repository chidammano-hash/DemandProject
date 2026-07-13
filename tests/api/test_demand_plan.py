"""API tests for F2.2 — /forecast/demand-plan/* endpoints."""

import datetime
import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


def _plan_row(plan_month=None, quantile=0.50, qty=450.0, horizon=1):
    if plan_month is None:
        plan_month = datetime.date(2026, 4, 1)
    return (
        plan_month,       # plan_month
        quantile,         # quantile
        qty,              # forecast_qty
        320.0,            # lower_bound
        580.0,            # upper_bound
        101.6,            # sigma_forecast
        80.0,             # sigma_demand
        129.3,            # sigma_combined
        horizon,          # horizon_months
    )


def _plan_row_with_version(plan_month=None, version="v1", quantile=0.50, qty=450.0):
    if plan_month is None:
        plan_month = datetime.date(2026, 4, 1)
    return (
        plan_month,
        version,
        quantile,
        qty,
    )


# ---------------------------------------------------------------------------
# GET /forecast/demand-plan
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_demand_plan_returns_pivoted_months():
    """Returns rows pivoted by plan_month with p10/p50/p90."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        ("2026-04-01_production",),   # resolve latest version
        (datetime.datetime(2026, 4, 1, 6, 0),),  # version metadata
    ]
    # Three quantile rows for one month
    cursor.fetchall.return_value = [
        _plan_row(quantile=0.10, qty=320.0),
        _plan_row(quantile=0.50, qty=450.0),
        _plan_row(quantile=0.90, qty=580.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/demand-plan",
                params={"item_id": "100320", "loc": "1401-BULK"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data
    # With three quantile rows, should be 1 pivoted month
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert row["p10"] == pytest.approx(320.0)
    assert row["p50"] == pytest.approx(450.0)
    assert row["p90"] == pytest.approx(580.0)


@pytest.mark.asyncio
async def test_get_demand_plan_404_when_no_rows():
    """Returns 404 when no demand plan data exists for a DFU."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        None,   # no active version
        None,   # no fallback version
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/demand-plan",
                params={"item_id": "UNKNOWN", "loc": "X"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /forecast/demand-plan/versions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_demand_plan_versions_list():
    """Returns list of plan versions with status."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (
            "2026-04-01_production",
            datetime.date(2026, 4, 1),
            "production",
            "lgbm_cluster",
            12,
            4823,
            "active",
            datetime.datetime(2026, 4, 1, 6, 0),
        ),
        (
            "2026-03-01_production",
            datetime.date(2026, 3, 1),
            "production",
            "lgbm_cluster",
            12,
            4790,
            "archived",
            datetime.datetime(2026, 3, 1, 6, 0),
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/demand-plan/versions")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["versions"]) == 2
    v = data["versions"][0]
    assert v["plan_version"] == "2026-04-01_production"
    assert v["status"] == "active"


# ---------------------------------------------------------------------------
# GET /forecast/demand-plan/comparison
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_demand_plan_comparison_delta():
    """delta_p50 = v1_p50 - v2_p50, delta_pct computed correctly."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (datetime.date(2026, 4, 1), "v1", 0.50, 450.0),
        (datetime.date(2026, 4, 1), "v2", 0.50, 410.0),
        (datetime.date(2026, 4, 1), "v1", 0.10, 320.0),
        (datetime.date(2026, 4, 1), "v2", 0.10, 290.0),
        (datetime.date(2026, 4, 1), "v1", 0.90, 580.0),
        (datetime.date(2026, 4, 1), "v2", 0.90, 540.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/demand-plan/comparison",
                params={
                    "v1": "v1", "v2": "v2",
                    "item_id": "100320", "loc": "1401-BULK",
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["v1"] == "v1"
    assert data["v2"] == "v2"
    assert len(data["months"]) == 1
    m = data["months"][0]
    assert m["delta_p50"] == pytest.approx(40.0)
    assert m["delta_pct"] == pytest.approx(9.76, abs=0.1)


# ---------------------------------------------------------------------------
# GET /forecast/demand-plan/weekly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_demand_plan_weekly_8_weeks():
    """Returns up to 8 weeks of weekly disaggregated forecast."""
    pool, conn, cursor = _make_pool()
    today = datetime.date.today()

    def make_weekly_row(offset_days: int, quantile: float, qty: float):
        plan_week = today + datetime.timedelta(days=offset_days)
        return (
            plan_week,
            plan_week.isocalendar()[1],
            plan_week.isocalendar()[0],
            datetime.date(2026, 4, 1),
            quantile,
            qty,
            0.233,
        )

    cursor.fetchone.return_value = ("2026-04-01_production",)  # resolve version
    # 3 quantile rows per week × 8 weeks = 24
    rows = []
    for w in range(8):
        for q, qty in [(0.10, 300.0), (0.50, 450.0), (0.90, 600.0)]:
            rows.append(make_weekly_row(w * 7, q, qty))
    cursor.fetchall.return_value = rows

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/demand-plan/weekly",
                params={"item_id": "100320", "loc": "1401-BULK", "weeks_ahead": 8},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["weeks"]) <= 8
    w0 = data["weeks"][0]
    assert "p50_weekly" in w0
    assert "p10_weekly" in w0
    assert "p90_weekly" in w0


@pytest.mark.asyncio
async def test_get_demand_plan_weekly_404_no_data():
    """Returns 404 when no weekly data found for a DFU."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/demand-plan/weekly",
                params={"item_id": "UNKNOWN", "loc": "X"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Branch coverage: quantile filter (lines 505-506) + no-rows 404 (line 536)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_demand_plan_quantile_filter():
    """?quantile param appended to query (lines 505-506)."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        ("v1",),                    # version lookup
        None,                       # plan_versions table lookup
    ]
    cursor.fetchall.return_value = [_plan_row(quantile=0.50, qty=450.0)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/demand-plan",
                params={"item_id": "100320", "loc": "1401-BULK", "quantile": 0.50},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["rows"]) == 1


@pytest.mark.asyncio
async def test_get_demand_plan_no_rows_raises_404():
    """Empty rows after version found → 404 (line 536)."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        ("v1",),    # version lookup OK
        None,       # plan_versions table lookup
    ]
    cursor.fetchall.return_value = []   # no data rows

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/demand-plan",
                params={"item_id": "100320", "loc": "1401-BULK", "plan_version": "v1"},
            )

    assert resp.status_code == 404
