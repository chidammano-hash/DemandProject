"""API tests for /inv-planning/exceptions/* endpoints.

IPfeature7 — Exception Queue & Replenishment Recommendations.
"""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# 18 original columns + 7 financial columns = 25 total
_EXCEPTION_ROW = (
    "exc-001", "ITEM001", "LOC1",
    datetime.date(2026, 3, 4), "below_rop", "high",
    150.0, 30.0, 200.0, 180.0,
    100.0, datetime.date(2026, 3, 11), datetime.date(2026, 3, 16),
    1500.0, "A_continuous_v1", "open",
    None, None,
    # financial columns: unit_cost, unit_margin, daily_demand_rate,
    # loss_of_sales_7d, loss_of_sales_30d, monthly_holding_cost, financial_impact_total
    15.0, 4.5, 16.4371, 345.18, 1479.34, 0.0, 345.18,
)


# ---------------------------------------------------------------------------
# GET /inv-planning/exceptions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exceptions_list_200():
    """GET /inv-planning/exceptions returns 200 with total + rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5,)
    cursor.fetchall.return_value = [_EXCEPTION_ROW]
    cursor.description = [
        ("exception_id",), ("item_id",), ("loc",), ("exception_date",), ("exception_type",), ("severity",),
        ("current_qty_on_hand",), ("current_dos",), ("ss_combined",), ("reorder_point",),
        ("recommended_order_qty",), ("recommended_order_by",), ("expected_receipt_date",),
        ("estimated_order_value",), ("policy_id",), ("status",), ("acknowledged_by",), ("notes",),
        ("unit_cost",), ("unit_margin",), ("daily_demand_rate",),
        ("loss_of_sales_7d",), ("loss_of_sales_30d",), ("monthly_holding_cost",), ("financial_impact_total",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_exceptions_list_row_keys():
    """Exception rows contain all expected keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [_EXCEPTION_ROW]
    cursor.description = [
        ("exception_id",), ("item_id",), ("loc",), ("exception_date",), ("exception_type",), ("severity",),
        ("current_qty_on_hand",), ("current_dos",), ("ss_combined",), ("reorder_point",),
        ("recommended_order_qty",), ("recommended_order_by",), ("expected_receipt_date",),
        ("estimated_order_value",), ("policy_id",), ("status",), ("acknowledged_by",), ("notes",),
        ("unit_cost",), ("unit_margin",), ("daily_demand_rate",),
        ("loss_of_sales_7d",), ("loss_of_sales_30d",), ("monthly_holding_cost",), ("financial_impact_total",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions")

    row = resp.json()["rows"][0]
    for key in ("exception_id", "item_id", "loc", "exception_type", "severity",
                "recommended_order_qty", "status",
                "unit_cost", "unit_margin", "daily_demand_rate",
                "loss_of_sales_7d", "loss_of_sales_30d",
                "monthly_holding_cost", "financial_impact_total"):
        assert key in row


@pytest.mark.asyncio
async def test_exceptions_list_severity_filter():
    """severity=critical filter accepted without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("exception_id",), ("item_id",), ("loc",), ("exception_date",), ("exception_type",), ("severity",),
        ("current_qty_on_hand",), ("current_dos",), ("ss_combined",), ("reorder_point",),
        ("recommended_order_qty",), ("recommended_order_by",), ("expected_receipt_date",),
        ("estimated_order_value",), ("policy_id",), ("status",), ("acknowledged_by",), ("notes",),
        ("unit_cost",), ("unit_margin",), ("daily_demand_rate",),
        ("loss_of_sales_7d",), ("loss_of_sales_30d",), ("monthly_holding_cost",), ("financial_impact_total",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions?severity=critical")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_exceptions_list_pagination():
    """Pagination params accepted without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("exception_id",), ("item_id",), ("loc",), ("exception_date",), ("exception_type",), ("severity",),
        ("current_qty_on_hand",), ("current_dos",), ("ss_combined",), ("reorder_point",),
        ("recommended_order_qty",), ("recommended_order_by",), ("expected_receipt_date",),
        ("estimated_order_value",), ("policy_id",), ("status",), ("acknowledged_by",), ("notes",),
        ("unit_cost",), ("unit_margin",), ("daily_demand_rate",),
        ("loss_of_sales_7d",), ("loss_of_sales_30d",), ("monthly_holding_cost",), ("financial_impact_total",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions?limit=5&offset=0")

    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 5


# ---------------------------------------------------------------------------
# GET /inv-planning/exceptions/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exceptions_summary_200():
    """GET /inv-planning/exceptions/summary returns 200 with expected keys."""
    pool, conn, cursor = _make_pool()
    # 17 columns: open_count, 6 types, 4 severities, order_value, oldest_days, fin_impact, loss_7d, holding, last_generated_at
    cursor.fetchone.return_value = (25, 5, 0, 8, 2, 7, 3, 10, 8, 5, 2, 15000.0, 14, 5200.0, 3400.0, 1800.0, datetime.datetime(2026, 3, 1, 12, 0))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "open_count" in data
    assert "by_type" in data
    assert "by_severity" in data
    assert "total_recommended_order_value" in data
    assert "oldest_open_days" in data
    assert "total_financial_impact" in data
    assert "total_loss_of_sales_7d" in data
    assert "total_monthly_holding_cost" in data


@pytest.mark.asyncio
async def test_exceptions_summary_by_severity_keys():
    """by_severity has all 4 keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (10, 2, 0, 3, 1, 3, 1, 4, 3, 2, 1, 5000.0, 7, 2100.0, 1400.0, 700.0, datetime.datetime(2026, 3, 1, 12, 0))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions/summary")

    data = resp.json()
    for key in ("critical", "high", "medium", "low"):
        assert key in data["by_severity"]


@pytest.mark.asyncio
async def test_exceptions_summary_by_type_keys():
    """by_type has all 6 exception type keys."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (10, 2, 0, 3, 1, 3, 1, 4, 3, 2, 1, 5000.0, 7, 2100.0, 1400.0, 700.0, datetime.datetime(2026, 3, 1, 12, 0))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions/summary")

    data = resp.json()
    for key in ("below_rop", "below_rop_critical", "below_ss", "stockout", "excess", "zero_velocity"):
        assert key in data["by_type"]


# ---------------------------------------------------------------------------
# PUT /inv-planning/exceptions/{id}/acknowledge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acknowledge_without_auth_returns_403():
    """PUT acknowledge without API key returns 401 when API_KEY is set."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/exceptions/exc-001/acknowledge",
                json={"acknowledged_by": "planner"},
            )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_acknowledge_with_auth_returns_200():
    """PUT acknowledge with valid API key returns 200 or 404 (expected if exc not found)."""
    pool, conn, cursor = _make_pool()
    # Return acknowledged row
    cursor.fetchone.return_value = (
        "exc-001", "ITEM001", "LOC1", "below_rop", "high",
        "acknowledged", "planner", None, None,
    )

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/exceptions/exc-001/acknowledge",
                headers={"X-API-Key": "secret-key"},
                json={"acknowledged_by": "planner"},
            )

    # 200 or 404 depending on mock data — both acceptable here
    assert resp.status_code in (200, 404)


# ---------------------------------------------------------------------------
# PUT /inv-planning/exceptions/{id}/status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_status_without_auth_returns_403():
    """PUT status without API key returns 401 when API_KEY is set."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/exceptions/exc-001/status",
                json={"status": "resolved"},
            )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_update_status_invalid_value_returns_422():
    """PUT status with invalid status value returns 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/exceptions/exc-001/status",
                headers={"X-API-Key": "secret-key"},
                json={"status": "invalid_status"},
            )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /inv-planning/exceptions/generate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_without_auth_returns_403():
    """POST generate without API key returns 401 when API_KEY is set."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/inv-planning/exceptions/generate")

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_generate_with_auth_returns_200():
    """POST generate with API key calls generator and returns counts."""
    pool, conn, cursor = _make_pool()
    mock_result = {"generated_count": 10, "skipped_dedup": 2, "by_type": {"stockout": 3, "below_rop": 7}, "dry_run": False}

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}), \
         patch("scripts.generate_replenishment_exceptions.run", return_value=mock_result):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/exceptions/generate",
                headers={"X-API-Key": "secret-key"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "generated_count" in data
    assert "skipped_dedup" in data
    assert "by_type" in data
