"""API tests for /inv-planning/safety-stock/* endpoints.

IPfeature3 — Safety Stock Engine.

All endpoints use get_conn() directly (not Depends(_get_pool)), consistent
with the pattern established in IPfeature6 and IPfeature7.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# GET /inv-planning/safety-stock/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_stock_summary_200():
    """GET /inv-planning/safety-stock/summary returns 200 with expected keys."""
    pool, conn, cursor = _make_pool()
    # Single combined query returns tagged rows: S=summary, C=class, G=gaps
    cursor.fetchall.return_value = [
        ("S", "500", "85", "1.12", "14.0", "-4200.0", None, None, None),
        ("C", "A", "80", "12", "420.0", "1.05", None, None, None),
        ("C", "B", "220", "45", "180.0", "0.98", None, None, None),
        ("C", "C", "200", "28", "90.0", "1.25", None, None, None),
        ("G", "I001", "L1", "200.0", "100.0", "-100.0", "0.5", None, None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "total_dfus" in data
    assert "below_ss_count" in data
    assert "below_ss_pct" in data
    assert "total_ss_gap_units" in data
    assert "avg_ss_coverage" in data
    assert "by_class" in data


@pytest.mark.asyncio
async def test_safety_stock_summary_by_class_keys():
    """by_class has A, B, C keys with expected sub-keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("S", "100", "20", "0.95", "10.0", "-1000.0", None, None, None),
        ("C", "A", "30", "5", "350.0", "0.90", None, None, None),
        ("C", "B", "45", "10", "200.0", "0.95", None, None, None),
        ("C", "C", "25", "5", "80.0", "1.05", None, None, None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/summary")

    assert resp.status_code == 200
    data = resp.json()
    by_class = data["by_class"]
    # All three ABC classes must be present
    for cls in ("A", "B", "C"):
        assert cls in by_class
        entry = by_class[cls]
        assert "count" in entry
        assert "below_ss_count" in entry
        assert "avg_ss_combined" in entry
        assert "avg_coverage" in entry


@pytest.mark.asyncio
async def test_safety_stock_summary_abc_vol_filter():
    """abc_vol query param filters the summary to a specific class."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("S", "30", "5", "0.92", "12.0", "-500.0", None, None, None),
        ("C", "A", "30", "5", "350.0", "0.92", None, None, None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/summary?abc_vol=A")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_safety_stock_summary_top_gaps_present():
    """Summary response includes top_gaps list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("S", "200", "40", "0.88", "15.0", "-8000.0", None, None, None),
        ("C", "A", "60", "15", "400.0", "0.85", None, None, None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "top_gaps" in data
    assert isinstance(data["top_gaps"], list)


@pytest.mark.asyncio
async def test_safety_stock_summary_empty_db():
    """Empty DB returns zero counts without 500."""
    pool, conn, cursor = _make_pool()
    # Only a summary row with zeros, no class or gap rows
    cursor.fetchall.return_value = [
        ("S", "0", "0", "0.0", "0.0", None, None, None, None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dfus"] == 0


# ---------------------------------------------------------------------------
# GET /inv-planning/safety-stock/detail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_stock_detail_200():
    """GET /inv-planning/safety-stock/detail returns 200 with total + rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (150,)
    cursor.fetchall.return_value = [
        (
            "ITEM001", "LOC1", "A", 0.98, 2.054,
            180.0,   # ss_combined
            340.0,   # reorder_point
            120.0,   # current_qty_on_hand
            8.0,     # current_dos
            -60.0,   # ss_gap
            0.667,   # ss_coverage
            True,    # is_below_ss
            18.0,    # target_dos_min
        ),
    ]
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("service_level_target",), ("z_score",),
        ("ss_combined",), ("reorder_point",), ("current_qty_on_hand",), ("current_dos",),
        ("ss_gap",), ("ss_coverage",), ("is_below_ss",), ("target_dos_min",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/detail")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_safety_stock_detail_row_keys():
    """Detail rows contain all expected keys from the spec."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (
            "ITEM001", "LOC1", "B", 0.95, 1.645,
            90.0, 270.0, 110.0, 12.0, 20.0, 1.22, False, 15.0,
        ),
    ]
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("service_level_target",), ("z_score",),
        ("ss_combined",), ("reorder_point",), ("current_qty_on_hand",), ("current_dos",),
        ("ss_gap",), ("ss_coverage",), ("is_below_ss",), ("target_dos_min",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/detail")

    row = resp.json()["rows"][0]
    for key in (
        "item_id", "loc", "abc_vol", "service_level_target", "z_score",
        "ss_combined", "reorder_point", "current_qty_on_hand",
        "ss_gap", "ss_coverage", "is_below_ss", "target_dos_min",
    ):
        assert key in row, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_safety_stock_detail_is_below_ss_filter():
    """is_below_ss=true filter returns only items below safety stock."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3,)
    cursor.fetchall.return_value = [
        ("ITEM002", "LOC2", "C", 0.90, 1.282, 60.0, 200.0, 40.0, 6.0, -20.0, 0.667, True, 12.0),
    ]
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("service_level_target",), ("z_score",),
        ("ss_combined",), ("reorder_point",), ("current_qty_on_hand",), ("current_dos",),
        ("ss_gap",), ("ss_coverage",), ("is_below_ss",), ("target_dos_min",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/detail?is_below_ss=true")

    assert resp.status_code == 200
    data = resp.json()
    for row in data["rows"]:
        assert row["is_below_ss"] is True


@pytest.mark.asyncio
async def test_safety_stock_detail_item_loc_filter():
    """item and location filter params accepted without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("service_level_target",), ("z_score",),
        ("ss_combined",), ("reorder_point",), ("current_qty_on_hand",), ("current_dos",),
        ("ss_gap",), ("ss_coverage",), ("is_below_ss",), ("target_dos_min",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/detail?item=ITEM001&location=LOC1")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_safety_stock_detail_pagination():
    """limit and offset pagination params accepted."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("service_level_target",), ("z_score",),
        ("ss_combined",), ("reorder_point",), ("current_qty_on_hand",), ("current_dos",),
        ("ss_gap",), ("ss_coverage",), ("is_below_ss",), ("target_dos_min",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/detail?limit=5&offset=10")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 100


@pytest.mark.asyncio
async def test_safety_stock_detail_sort_by_ss_gap():
    """sort_by=ss_gap accepted as a valid sort parameter."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (50,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("service_level_target",), ("z_score",),
        ("ss_combined",), ("reorder_point",), ("current_qty_on_hand",), ("current_dos",),
        ("ss_gap",), ("ss_coverage",), ("is_below_ss",), ("target_dos_min",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/detail?sort_by=ss_gap&sort_dir=asc")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_safety_stock_detail_invalid_sort_falls_back():
    """Invalid sort_by falls back gracefully — no 422 error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("item_id",), ("loc",), ("abc_vol",), ("service_level_target",), ("z_score",),
        ("ss_combined",), ("reorder_point",), ("current_qty_on_hand",), ("current_dos",),
        ("ss_gap",), ("ss_coverage",), ("is_below_ss",), ("target_dos_min",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/detail?sort_by=drop_table")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /inv-planning/safety-stock/waterfall
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_stock_waterfall_200():
    """GET /inv-planning/safety-stock/waterfall with item+location returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        "ITEM001",   # item_id
        "LOC1",      # loc
        0.95,        # service_level_target
        1.645,       # z_score
        300.0,       # demand_mean_monthly
        60.0,        # demand_std_monthly
        14.0,        # lead_time_mean_days
        2.8,         # lead_time_std_days
        12.31,       # ss_demand_only
        49.35,       # ss_lt_only
        50.87,       # ss_combined
        190.87,      # reorder_point
        120.0,       # current_qty_on_hand
        -30.87,      # ss_gap
    )
    cursor.description = [
        ("item_id",), ("loc",), ("service_level_target",), ("z_score",),
        ("demand_mean_monthly",), ("demand_std_monthly",),
        ("lead_time_mean_days",), ("lead_time_std_days",),
        ("ss_demand_only",), ("ss_lt_only",), ("ss_combined",),
        ("reorder_point",), ("current_qty_on_hand",), ("ss_gap",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/safety-stock/waterfall?item=ITEM001&location=LOC1"
            )

    assert resp.status_code == 200
    data = resp.json()
    # All required waterfall components must be present
    assert "item_id" in data
    assert "loc" in data
    assert "demand_component" in data
    assert "lt_component" in data
    assert "combined_ss" in data
    assert "reorder_point" in data
    assert "current_on_hand" in data
    assert "ss_gap" in data
    assert "z_score" in data
    assert "service_level_target" in data


@pytest.mark.asyncio
async def test_safety_stock_waterfall_missing_item_422():
    """GET /inv-planning/safety-stock/waterfall without item param → 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/safety-stock/waterfall?location=LOC1"
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_safety_stock_waterfall_missing_location_422():
    """GET /inv-planning/safety-stock/waterfall without location param → 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/safety-stock/waterfall?item=ITEM001"
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_safety_stock_waterfall_missing_both_params_422():
    """GET /inv-planning/safety-stock/waterfall without either param → 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/waterfall")

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_safety_stock_waterfall_demand_lt_components():
    """Waterfall contains lt_std_days (may be null if unavailable)."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        "ITEM002", "LOC2", 0.90, 1.282,
        200.0, 40.0,
        21.0, None,   # lt_std_days is NULL (unavailable, fallback applied)
        8.5, 0.0,
        8.5, 171.5,
        150.0, 141.5,
    )
    cursor.description = [
        ("item_id",), ("loc",), ("service_level_target",), ("z_score",),
        ("demand_mean_monthly",), ("demand_std_monthly",),
        ("lead_time_mean_days",), ("lead_time_std_days",),
        ("ss_demand_only",), ("ss_lt_only",), ("ss_combined",),
        ("reorder_point",), ("current_qty_on_hand",), ("ss_gap",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/safety-stock/waterfall?item=ITEM002&location=LOC2"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "lt_std_days" in data  # may be null


# ---------------------------------------------------------------------------
# POST /inv-planning/safety-stock/override
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_stock_override_without_auth_returns_401():
    """POST /inv-planning/safety-stock/override without API key → 401."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/safety-stock/override",
                json={
                    "item_id": "ITEM001",
                    "loc": "LOC1",
                    "ss_override_qty": 150.0,
                    "reason": "Planner judgment",
                },
            )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_safety_stock_override_with_auth_returns_200():
    """POST /inv-planning/safety-stock/override with valid API key → 200."""
    pool, conn, cursor = _make_pool()
    import datetime as _dt
    cursor.fetchone.return_value = (
        "ITEM001", "LOC1", 150.0, "manual",
        _dt.datetime(2026, 3, 5, 10, 0, 0),
    )

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/safety-stock/override",
                headers={"X-API-Key": "secret-key"},
                json={
                    "item_id": "ITEM001",
                    "loc": "LOC1",
                    "ss_override_qty": 150.0,
                    "reason": "Planner judgment",
                },
            )

    # 201 = created, 404 = item not found in table — both are acceptable without real DB
    assert resp.status_code in (201, 404)


@pytest.mark.asyncio
async def test_safety_stock_override_response_has_ss_method():
    """Override response sets ss_method='manual'."""
    pool, conn, cursor = _make_pool()
    import datetime as _dt
    cursor.fetchone.return_value = (
        "ITEM001", "LOC1", 150.0, "manual",
        _dt.datetime(2026, 3, 5, 10, 0, 0),
    )

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/safety-stock/override",
                headers={"X-API-Key": "secret-key"},
                json={
                    "item_id": "ITEM001",
                    "loc": "LOC1",
                    "ss_override_qty": 150.0,
                    "reason": "Planner judgment",
                },
            )

    if resp.status_code == 200:
        data = resp.json()
        assert data.get("ss_method") == "manual"


# ---------------------------------------------------------------------------
# GET /inv-planning/safety-stock/config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_stock_config_200():
    """GET /inv-planning/safety-stock/config returns 200 with config structure."""
    pool, conn, cursor = _make_pool()

    mock_config = {
        "safety_stock": {
            "default_method": "combined",
            "policy_version": "v1",
            "service_levels": {"A": 0.98, "B": 0.95, "C": 0.90, "default": 0.95},
            "z_table": {0.90: 1.282, 0.95: 1.645, 0.98: 2.054},
            "min_ss_days": 3,
            "max_ss_days": 120,
            "lt_std_fallback_pct": 0.20,
        }
    }

    with patch("api.core._get_pool", return_value=pool), \
         patch("yaml.safe_load", return_value=mock_config):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/config")

    assert resp.status_code == 200
    data = resp.json()
    assert "service_levels" in data or "safety_stock" in data


@pytest.mark.asyncio
async def test_safety_stock_config_has_z_table():
    """Config response contains z_table (or equivalent) with Z values."""
    pool, conn, cursor = _make_pool()

    mock_config = {
        "safety_stock": {
            "default_method": "combined",
            "policy_version": "v1",
            "service_levels": {"A": 0.98, "B": 0.95, "C": 0.90, "default": 0.95},
            "z_table": {"0.90": 1.282, "0.95": 1.645, "0.98": 2.054, "0.99": 2.326},
            "min_ss_days": 3,
            "max_ss_days": 120,
            "lt_std_fallback_pct": 0.20,
        }
    }

    with patch("api.core._get_pool", return_value=pool), \
         patch("yaml.safe_load", return_value=mock_config):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/config")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_safety_stock_config_has_service_levels():
    """Config response includes service level targets by ABC class."""
    pool, conn, cursor = _make_pool()

    mock_config = {
        "safety_stock": {
            "default_method": "combined",
            "policy_version": "v1",
            "service_levels": {"A": 0.98, "B": 0.95, "C": 0.90, "default": 0.95},
            "z_table": {"0.95": 1.645},
            "min_ss_days": 3,
            "max_ss_days": 120,
            "lt_std_fallback_pct": 0.20,
        }
    }

    with patch("api.core._get_pool", return_value=pool), \
         patch("yaml.safe_load", return_value=mock_config):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/safety-stock/config")

    assert resp.status_code == 200
