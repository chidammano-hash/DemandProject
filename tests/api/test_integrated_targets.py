"""API tests for /inv-planning/integrated-targets endpoints.

Integrated Planning Targets — unified SS + EOQ + ROP with cost metrics.
All endpoints use get_conn() directly (not Depends(_get_pool)).
"""
from __future__ import annotations

import datetime as _dt
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool

# ---------------------------------------------------------------------------
# GET /inv-planning/integrated-targets/summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integrated_targets_summary_200():
    """GET /inv-planning/integrated-targets/summary returns 200 with expected keys."""
    pool, _conn, cursor = _make_pool()
    # summary_sql fetchone, then abc_sql fetchall
    cursor.fetchone.return_value = (
        500,     # total_skus
        85,      # below_ss_count
        120.5,   # avg_safety_stock
        350.0,   # avg_eoq
        295.3,   # avg_target_inventory
        42000.0, # total_monthly_holding_cost
        8500.0,  # total_monthly_ordering_cost
        50500.0, # total_monthly_cost
        38.5,    # avg_risk_score
        120,     # high_risk_count (score >= 60)
        45,      # critical_risk_count (score >= 80)
        125000.0, # total_excess_value_usd
        2604.17,  # total_excess_holding_cost_monthly
        210,      # excess_sku_count
        32.0,     # avg_excess_risk_score
    )
    cursor.fetchall.return_value = [
        ("A", 80, 420.0, 900.0, 870.0, 15000.0, 18000.0, 12),
        ("B", 220, 180.0, 500.0, 410.0, 12000.0, 14500.0, 45),
        ("C", 200, 90.0, 200.0, 145.0, 8000.0, 9500.0, 28),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/integrated-targets/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_skus"] == 500
    assert data["below_ss_count"] == 85
    assert "below_ss_pct" in data
    assert "avg_safety_stock" in data
    assert "avg_eoq" in data
    assert "avg_target_inventory" in data
    assert "total_monthly_holding_cost" in data
    assert "total_monthly_ordering_cost" in data
    assert "total_monthly_cost" in data
    assert "avg_risk_score" in data
    assert data["high_risk_count"] == 120
    assert data["critical_risk_count"] == 45
    assert "total_excess_value_usd" in data
    assert "total_excess_holding_cost_monthly" in data
    assert data["excess_sku_count"] == 210
    assert "avg_excess_risk_score" in data
    assert "by_abc" in data
    assert len(data["by_abc"]) == 3


@pytest.mark.asyncio
async def test_integrated_targets_summary_empty():
    """Empty DB returns zero counts without 500."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0, 0, None, None, None, None, None, None, None, 0, 0, None, None, 0, None)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/integrated-targets/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_skus"] == 0
    assert data["by_abc"] == []
    assert data["avg_risk_score"] is None
    assert data["high_risk_count"] == 0
    assert data["critical_risk_count"] == 0
    assert data["total_excess_value_usd"] is None
    assert data["total_excess_holding_cost_monthly"] is None
    assert data["excess_sku_count"] == 0
    assert data["avg_excess_risk_score"] is None


@pytest.mark.asyncio
async def test_integrated_targets_summary_abc_filter():
    """abc_vol query param accepted without error."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (80, 12, 420.0, 900.0, 870.0, 15000.0, 3000.0, 18000.0, 42.0, 15, 5, 25000.0, 520.83, 30, 35.0)
    cursor.fetchall.return_value = [
        ("A", 80, 420.0, 900.0, 870.0, 15000.0, 18000.0, 12),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/integrated-targets/summary?abc_vol=A")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_integrated_targets_summary_by_abc_keys():
    """by_abc entries contain all expected keys."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100, 10, 100.0, 200.0, 150.0, 5000.0, 1000.0, 6000.0, 35.0, 20, 8, 18000.0, 375.0, 40, 28.0)
    cursor.fetchall.return_value = [
        ("A", 50, 200.0, 400.0, 350.0, 3000.0, 3500.0, 5),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/integrated-targets/summary")

    data = resp.json()
    entry = data["by_abc"][0]
    for key in (
        "abc_vol", "count", "avg_safety_stock", "avg_eoq",
        "avg_target_inventory", "total_holding_cost", "total_cost",
        "below_ss_count",
    ):
        assert key in entry, f"Missing key in by_abc entry: {key}"


# ---------------------------------------------------------------------------
# GET /inv-planning/integrated-targets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integrated_targets_detail_200():
    """GET /inv-planning/integrated-targets returns 200 with total + rows."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (
            "ITEM001", "LOC1", "A", "AX",            # 0-3
            180.0, 340.0, 0.98,                        # 4-6: ss, rop, slt
            300.0, 60.0, 0.2,                          # 7-9: demand stats
            200.0, 200.0, 100.0, 6.0, 15.5,           # 10-14: eoq fields
            280.0, 180.0, 380.0,                       # 15-17: target inventory
            120.0, 8.0, -60.0, True,                   # 18-21: current pos
            5.63, 3.23, 8.85, 2.50, 11.35,            # 22-26: cost metrics
            14.0, 2.8,                                 # 27-28: lead time
            "v1", "champion", "lgbm_cluster",          # 29-31: metadata
            _dt.datetime(2026, 4, 1, 12, 0, 0),       # 32: computed_at
            82,                                        # 33: stockout_risk_score
            0.0, 0.0, 0.0,                            # 34-36: excess_qty, value, holding
            0.0, 5,                                    # 37-38: excess_months_supply, excess_risk_score
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/integrated-targets")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert data["total"] == 1
    assert len(data["rows"]) == 1
    assert data["rows"][0]["stockout_risk_score"] == 82
    assert data["rows"][0]["excess_risk_score"] == 5


@pytest.mark.asyncio
async def test_integrated_targets_detail_row_keys():
    """Detail rows contain all expected keys from the spec."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (
            "ITEM001", "LOC1", "B", "BZ",
            90.0, 270.0, 0.95,
            200.0, 40.0, 0.2,
            150.0, 150.0, 75.0, 8.0, 12.0,
            165.0, 90.0, 240.0,
            110.0, 12.0, 20.0, False,
            2.81, 1.56, 4.38, 1.67, 6.04,
            21.0, 3.5,
            "v1", "champion", "mstl",
            _dt.datetime(2026, 4, 1, 12, 0, 0),
            37,                                        # stockout_risk_score
            50.0, 600.0, 12.5,                         # excess_qty, value, holding
            2.5, 45,                                   # excess_months_supply, excess_risk_score
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/integrated-targets")

    row = resp.json()["rows"][0]
    expected_keys = [
        "item_id", "loc", "abc_vol", "abc_xyz_segment",
        "safety_stock_qty", "reorder_point", "service_level_target",
        "demand_mean_monthly", "demand_std_monthly", "demand_cv",
        "eoq_qty", "effective_eoq", "cycle_stock", "orders_per_year", "unit_cost",
        "target_avg_inventory", "target_min_inventory", "target_max_inventory",
        "current_qty_on_hand", "current_dos", "ss_gap", "is_below_ss",
        "monthly_ss_holding_cost", "monthly_cycle_holding_cost",
        "monthly_total_holding_cost", "monthly_ordering_cost", "monthly_total_cost",
        "lead_time_mean_days", "lead_time_std_days",
        "policy_version", "forecast_source", "forecast_model_id", "computed_at",
        "stockout_risk_score",
        "excess_qty", "excess_value_usd", "excess_holding_cost_monthly",
        "excess_months_supply", "excess_risk_score",
    ]
    for key in expected_keys:
        assert key in row, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_integrated_targets_detail_below_ss_filter():
    """below_ss_only=true filter accepted without error."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/integrated-targets?below_ss_only=true")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_integrated_targets_detail_item_loc_filter():
    """item_id and loc filter params accepted without error."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/integrated-targets?item_id=ITEM001&loc=LOC1"
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_integrated_targets_detail_pagination():
    """limit and offset pagination params accepted."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/integrated-targets?limit=10&offset=20")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 100
    assert data["limit"] == 10
    assert data["offset"] == 20


@pytest.mark.asyncio
async def test_integrated_targets_detail_sort_by():
    """sort_by parameter accepted for valid columns."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/integrated-targets?sort_by=safety_stock_qty&sort_dir=asc"
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_integrated_targets_detail_sort_by_risk_score():
    """sort_by=stockout_risk_score is the default and works correctly."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/integrated-targets?sort_by=stockout_risk_score"
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_integrated_targets_detail_invalid_sort_falls_back():
    """Invalid sort_by falls back gracefully -- no 422 error."""
    pool, _conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/inv-planning/integrated-targets?sort_by=drop_table"
            )

    assert resp.status_code == 200
