"""API tests for /inv-planning/replenishment/* endpoints.

Forward-Looking Replenishment Plan — 4 GET endpoints:
  /inv-planning/replenishment/summary
  /inv-planning/replenishment/detail
  /inv-planning/replenishment/comparison
  /inv-planning/replenishment/dfu

All endpoints use get_conn() directly (not Depends(_get_pool)), consistent
with the pattern established across all inv_planning_*.py routers.
"""
from __future__ import annotations

import datetime

import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# GET /inv-planning/replenishment/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_default():
    """GET /inv-planning/replenishment/summary returns 200 with expected top-level keys."""
    pool, conn, cursor = _make_pool()
    # fetchone calls: (1) version lookup, (2) summary row
    # fetchall calls: (1) policy_type breakdown
    cursor.fetchone.side_effect = [
        ("2026-02",),          # plan_version lookup
        (172276, 12450, 145.5, 320.8, 12.4, datetime.datetime(2026, 3, 1, 12, 0)),  # summary row (6 cols incl. last_computed_at)
    ]
    cursor.fetchall.return_value = [
        ("continuous_rop", 80000, 150.0, 300.0, 24000000.0),
        ("periodic_review", 60000, 130.0, 350.0, 21000000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "plan_version" in data
    assert "total_dfus" in data
    assert "below_ss_count" in data
    assert "below_ss_pct" in data
    assert "avg_ss" in data
    assert "avg_eoq" in data
    assert "avg_ss_delta_pct" in data
    assert "by_policy_type" in data
    assert isinstance(data["by_policy_type"], list)


@pytest.mark.asyncio
async def test_summary_with_policy_filter():
    """GET /inv-planning/replenishment/summary?policy_type=continuous_rop returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        ("2026-02",),
        (80000, 5000, 150.0, 300.0, 8.0, datetime.datetime(2026, 3, 1, 12, 0)),
    ]
    cursor.fetchall.return_value = [
        ("continuous_rop", 80000, 150.0, 300.0, 24000000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/summary?policy_type=continuous_rop"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 80000


@pytest.mark.asyncio
async def test_summary_with_abc_filter():
    """GET /inv-planning/replenishment/summary?abc_vol=A returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        ("2026-02",),
        (5000, 800, 500.0, 1200.0, 18.0, datetime.datetime(2026, 3, 1, 12, 0)),
    ]
    cursor.fetchall.return_value = [
        ("continuous_rop", 4000, 520.0, 1250.0, 5000000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/summary?abc_vol=A")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 5000


@pytest.mark.asyncio
async def test_summary_no_data():
    """When no plan_version exists, summary returns empty graceful response."""
    pool, conn, cursor = _make_pool()
    # Version lookup returns None → no plan data
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 0
    assert data["by_policy_type"] == []
    assert data["plan_version"] is None


@pytest.mark.asyncio
async def test_summary_by_policy_type_keys():
    """by_policy_type entries have the expected keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        ("2026-02",),
        (10000, 500, 200.0, 400.0, 5.0, datetime.datetime(2026, 3, 1, 12, 0)),
    ]
    cursor.fetchall.return_value = [
        ("continuous_rop", 6000, 210.0, 420.0, 2520000.0),
        ("periodic_review", 4000, 185.0, 370.0, 1480000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["by_policy_type"]) == 2
    entry = data["by_policy_type"][0]
    for key in ("policy_type", "dfu_count", "avg_ss", "avg_eoq", "total_order_qty"):
        assert key in entry, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# GET /inv-planning/replenishment/detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detail_default():
    """GET /inv-planning/replenishment/detail returns 200 with pagination fields + rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (172276,)  # COUNT(*)
    cursor.fetchall.return_value = [
        (
            "100320",                          # item_id
            "1401-BULK",                       # loc
            datetime.date(2026, 3, 1),         # plan_month
            "A",                               # abc_vol
            "continuous_rop",                  # policy_type
            5234.0,                            # forecast_qty
            450.2,                             # ss_combined
            380.5,                             # historical_ss
            69.7,                              # ss_delta
            18.3,                              # ss_delta_pct
            1200.0,                            # eoq
            600.0,                             # cycle_stock
            890.0,                             # reorder_point
            1200.0,                            # order_qty
            None,                              # order_up_to_level
            False,                             # is_below_ss
        )
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/detail")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert "rows" in data
    assert data["total"] == 172276
    assert data["limit"] == 50
    assert data["offset"] == 0
    assert len(data["rows"]) == 1


@pytest.mark.asyncio
async def test_detail_row_keys():
    """Detail rows contain all expected keys from the router SELECT list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (
            "200001", "LOC99",
            datetime.date(2026, 4, 1), "B", "periodic_review",
            800.0, 100.0, 80.0, 20.0, 25.0,
            300.0, 150.0, 200.0, 300.0, None, True,
        )
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/detail")

    assert resp.status_code == 200
    row = resp.json()["rows"][0]
    for key in (
        "item_id", "loc", "plan_month", "abc_vol", "policy_type",
        "forecast_qty", "ss_combined", "historical_ss",
        "ss_delta", "ss_delta_pct",
        "eoq", "cycle_stock", "reorder_point", "order_qty",
        "order_up_to_level", "is_below_ss",
    ):
        assert key in row, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_detail_with_filters():
    """item, location, policy_type, abc_vol, is_below_ss params accepted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/detail"
                "?item=100320&location=1401&policy_type=continuous_rop"
                "&abc_vol=A&is_below_ss=true"
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_detail_empty():
    """No matching rows returns total=0 and rows=[]."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/detail")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["rows"] == []


# ---------------------------------------------------------------------------
# GET /inv-planning/replenishment/comparison
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_comparison_default():
    """GET /inv-planning/replenishment/comparison returns 200 with by_abc list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("2026-02",)   # version lookup
    cursor.fetchall.return_value = [
        # abc_vol, dfu_count, avg_forecast_ss, avg_historical_ss, avg_ss_delta,
        # avg_ss_delta_pct, count_increased, count_decreased, count_unchanged
        ("A", 5000, 500.0, 420.0, 80.0, 19.0, 4200, 800, 0),
        ("B", 20000, 200.0, 180.0, 20.0, 11.1, 16000, 4000, 0),
        ("C", 80000, 80.0, 75.0, 5.0, 6.7, 60000, 20000, 0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/comparison")

    assert resp.status_code == 200
    data = resp.json()
    assert "plan_version" in data
    assert "by_abc" in data
    assert "total_increased" in data
    assert "total_decreased" in data
    assert isinstance(data["by_abc"], list)
    assert len(data["by_abc"]) == 3
    # Check aggregated totals
    assert data["total_increased"] == 4200 + 16000 + 60000
    assert data["total_decreased"] == 800 + 4000 + 20000


@pytest.mark.asyncio
async def test_comparison_by_abc_entry_keys():
    """by_abc entries contain all expected keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("2026-02",)
    cursor.fetchall.return_value = [
        ("A", 5000, 500.0, 420.0, 80.0, 19.0, 4200, 800, 0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/comparison")

    assert resp.status_code == 200
    entry = resp.json()["by_abc"][0]
    for key in (
        "abc_vol", "dfu_count", "avg_forecast_ss", "avg_historical_ss",
        "avg_ss_delta", "avg_ss_delta_pct", "count_increased",
        "count_decreased", "count_unchanged",
    ):
        assert key in entry, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_comparison_empty():
    """When no comparison data exists, returns empty by_abc list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None   # version lookup returns nothing
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/comparison")

    assert resp.status_code == 200
    data = resp.json()
    assert data["by_abc"] == []
    assert data["total_increased"] == 0
    assert data["total_decreased"] == 0


# ---------------------------------------------------------------------------
# GET /inv-planning/replenishment/dfu
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dfu_series_found():
    """GET /inv-planning/replenishment/dfu with valid item_id+loc returns series."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("2026-02",)   # version lookup
    cursor.fetchall.return_value = [
        # plan_month, horizon_months,
        # forecast_qty, forecast_qty_lower, forecast_qty_upper,
        # ss_combined, historical_ss, ss_delta,
        # eoq, cycle_stock,
        # reorder_point, order_qty, order_up_to_level,
        # avg_daily_demand, is_below_ss, sigma_method
        (
            datetime.date(2026, 3, 1), 1,
            5234.0, 4100.0, 6368.0,
            450.2, 380.5, 69.7,
            1200.0, 600.0,
            890.0, 1200.0, None,
            171.9, False, "ci_spread",
        ),
        (
            datetime.date(2026, 4, 1), 2,
            4800.0, 3900.0, 5700.0,
            450.2, 380.5, 69.7,
            1200.0, 600.0,
            890.0, 1200.0, None,
            160.0, False, "ci_spread",
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/dfu?item_id=100320&loc=1401-BULK"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "100320"
    assert data["loc"] == "1401-BULK"
    assert "plan_version" in data
    assert "series" in data
    assert isinstance(data["series"], list)
    assert len(data["series"]) == 2


@pytest.mark.asyncio
async def test_dfu_series_entry_keys():
    """Series entries contain all expected keys from the router SELECT list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("2026-02",)
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 3, 1), 1,
            5234.0, 4100.0, 6368.0,
            450.2, 380.5, 69.7,
            1200.0, 600.0,
            890.0, 1200.0, None,
            171.9, False, "ci_spread",
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/dfu?item_id=100320&loc=1401-BULK"
            )

    assert resp.status_code == 200
    entry = resp.json()["series"][0]
    for key in (
        "plan_month", "horizon_months",
        "forecast_qty", "forecast_qty_lower", "forecast_qty_upper",
        "ss_combined", "historical_ss", "ss_delta",
        "eoq", "cycle_stock",
        "reorder_point", "order_qty", "order_up_to_level",
        "avg_daily_demand", "is_below_ss", "sigma_method",
    ):
        assert key in entry, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_dfu_series_not_found():
    """GET /inv-planning/replenishment/dfu with unknown item/loc returns 404."""
    pool, conn, cursor = _make_pool()
    # Version lookup returns a value but series query returns no rows
    cursor.fetchone.return_value = ("2026-02",)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/dfu?item_id=UNKNOWN&loc=UNKNOWN_LOC"
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dfu_series_no_version_found_returns_404():
    """When version lookup returns None and no plan_version param provided → 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None    # no plan found for this DFU
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/dfu?item_id=RARE&loc=NOLOC"
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dfu_series_requires_item_id():
    """GET /inv-planning/replenishment/dfu without item_id → 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/dfu?loc=1401-BULK"
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_dfu_series_requires_loc():
    """GET /inv-planning/replenishment/dfu without loc → 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/dfu?item_id=100320"
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_dfu_series_explicit_plan_version():
    """Explicit plan_version param bypasses version lookup query."""
    pool, conn, cursor = _make_pool()
    # No fetchone call for version lookup when plan_version is explicit
    cursor.fetchone.return_value = ("2026-01",)   # should NOT be called
    cursor.fetchall.return_value = [
        (
            datetime.date(2026, 1, 1), 1,
            3000.0, 2500.0, 3500.0,
            300.0, 280.0, 20.0,
            800.0, 400.0,
            580.0, 800.0, None,
            100.0, False, "z_score",
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/dfu"
                "?item_id=100320&loc=1401-BULK&plan_version=2026-01"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_version"] == "2026-01"
    assert len(data["series"]) == 1


# ---------------------------------------------------------------------------
# Branch coverage: missing paths (line 49 — explicit plan_version in summary,
# line 114 — null summary_row, lines 175-176/193-194 — detail filters,
# lines 282/295-299 — comparison explicit version + abc/policy filters)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_explicit_plan_version_skips_lookup():
    """Providing plan_version explicitly skips the version-lookup fetchone call (line 49)."""
    pool, conn, cursor = _make_pool()
    # Only ONE fetchone (summary row) — version lookup is skipped
    cursor.fetchone.return_value = (50000, 2000, 210.0, 350.0, 6.5, datetime.datetime(2026, 1, 15, 10, 0))
    cursor.fetchall.return_value = [
        ("continuous_rop", 30000, 220.0, 360.0, 10800000.0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/summary?plan_version=2026-01"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_version"] == "2026-01"
    assert data["total_dfus"] == 50000


@pytest.mark.asyncio
async def test_summary_null_summary_row_defaults_to_zeros():
    """When summary fetchone returns None, summary_row defaults to zeros (line 114)."""
    pool, conn, cursor = _make_pool()
    # version lookup OK, but no summary rows for that version
    cursor.fetchone.side_effect = [
        ("2026-02",),   # version lookup
        None,           # summary query returns no row
    ]
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/replenishment/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 0
    assert data["below_ss_count"] == 0
    assert data["below_ss_pct"] == 0.0
    assert data["avg_ss"] is None


@pytest.mark.asyncio
async def test_detail_plan_version_filter_applied():
    """plan_version param is added to WHERE clause (lines 175-176)."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (
            "100320", "1401-BULK",
            datetime.date(2026, 1, 1), "A", "continuous_rop",
            5000.0, 440.0, 380.0, 60.0, 15.8,
            1200.0, 600.0, 870.0, 1200.0, None, False,
        )
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/detail?plan_version=2026-01"
            )

    assert resp.status_code == 200
    assert len(resp.json()["rows"]) == 1


@pytest.mark.asyncio
async def test_detail_plan_month_filter_applied():
    """plan_month param is added to WHERE clause (lines 193-194)."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (
            "100320", "1401-BULK",
            datetime.date(2026, 3, 1), "A", "continuous_rop",
            5000.0, 440.0, 380.0, 60.0, 15.8,
            1200.0, 600.0, 870.0, 1200.0, None, False,
        )
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/detail?plan_month=2026-03-01"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"][0]["plan_month"] == "2026-03-01"


@pytest.mark.asyncio
async def test_comparison_explicit_plan_version():
    """Providing plan_version skips version-lookup fetchone in comparison (line 282)."""
    pool, conn, cursor = _make_pool()
    # No fetchone needed — plan_version is explicit
    cursor.fetchall.return_value = [
        ("A", 5000, 500.0, 420.0, 80.0, 19.0, 4200, 800, 0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/comparison?plan_version=2026-01"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_version"] == "2026-01"
    assert len(data["by_abc"]) == 1


@pytest.mark.asyncio
async def test_comparison_abc_vol_filter():
    """abc_vol filter appended to comparison SQL (lines 295-296)."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("2026-02",)
    cursor.fetchall.return_value = [
        ("A", 5000, 500.0, 420.0, 80.0, 19.0, 4200, 800, 0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/comparison?abc_vol=A"
            )

    assert resp.status_code == 200
    assert resp.json()["by_abc"][0]["abc_vol"] == "A"


@pytest.mark.asyncio
async def test_comparison_policy_type_filter():
    """policy_type filter appended to comparison SQL (lines 297-299)."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("2026-02",)
    cursor.fetchall.return_value = [
        ("B", 20000, 200.0, 185.0, 15.0, 8.1, 16000, 4000, 0),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/replenishment/comparison?policy_type=periodic_review"
            )

    assert resp.status_code == 200
    assert len(resp.json()["by_abc"]) == 1
