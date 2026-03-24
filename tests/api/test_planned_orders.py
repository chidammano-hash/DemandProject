"""API tests for /supply/planned-orders/* endpoints — F2.1."""

import datetime
import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


def _planned_order_row(order_id=1001):
    today = datetime.date.today()
    return (
        order_id,                               # id
        "100320",                               # item_id
        "1401-BULK",                            # loc
        "VENDOR-0042",                          # supplier_id
        "Acme Supply Co.",                      # supplier_name
        233.40,                                 # net_requirement_qty
        300.0,                                  # recommended_qty
        100.0,                                  # moq
        12.50,                                  # unit_cost
        3750.0,                                 # order_value
        "USD",                                  # currency
        today + datetime.timedelta(days=4),     # trigger_date
        "projected_below_ss",                   # trigger_reason
        today + datetime.timedelta(days=4),     # order_by_date
        today + datetime.timedelta(days=18),    # expected_receipt_date
        14,                                     # lead_time_days
        120.0,                                  # current_qty_on_hand
        60.0,                                   # safety_stock
        60.0,                                   # reorder_point
        200.0,                                  # confirmed_inbound_qty
        228.2,                                  # lt_forecast_demand
        "2026-03",                              # plan_version
        0.950,                                  # confidence_score
        "all data sources available",           # confidence_reason
        False,                                  # is_past_due
        "proposed",                             # status
        datetime.datetime(2026, 3, 2, 8, 4),   # created_at
        None,                                   # approved_by
        None,                                   # approved_at
    )


# ---------------------------------------------------------------------------
# GET /supply/planned-orders/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_planned_orders_summary_success():
    """200 with KPI counts and values."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        1482, 203, 87, 44,         # status counts
        1842340.0, 284150.0,       # proposed_value, approved_value
        34, 41200.0,               # past_due_count, past_due_value
        0.873, 142,                # avg_confidence, low_confidence_count
        datetime.datetime(2026, 3, 2, 8, 4),  # generated_at
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/planned-orders/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status_counts"]["proposed"] == 1482
    assert data["total_proposed_value_usd"] == pytest.approx(1842340.0)
    assert data["past_due_proposed_count"] == 34
    assert data["avg_confidence_score"] == pytest.approx(0.873)


@pytest.mark.asyncio
async def test_get_planned_orders_summary_empty():
    """Returns zeroed payload when no data."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (None, None, None, None, None, None, None, None, None, None, None)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/planned-orders/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status_counts"]["proposed"] == 0


# ---------------------------------------------------------------------------
# GET /supply/planned-orders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_planned_orders_success():
    """200 with list of planned order rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (1, 3750.0, 0),  # COUNT, SUM(order_value), past_due_count
    ]
    cursor.fetchall.return_value = [_planned_order_row()]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/planned-orders")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert item["item_id"] == "100320"
    assert item["recommended_qty"] == 300.0
    assert item["status"] == "proposed"


@pytest.mark.asyncio
async def test_get_planned_orders_past_due_only_filter():
    """past_due_only=true filter passes correctly."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(0, 0.0, 0)]
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/planned-orders?past_due_only=true")

    assert resp.status_code == 200
    assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# PUT /supply/planned-orders/{id}/approve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_planned_order_success():
    """200 — status changed to 'approved'."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1001, "approved", "jane@example.com", datetime.datetime(2026, 3, 6, 9, 15))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/planned-orders/1001/approve",
                json={"approved_by": "jane@example.com"},
                headers={"X-API-Key": "test-key"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approved"
    assert data["approved_by"] == "jane@example.com"


@pytest.mark.asyncio
async def test_approve_planned_order_requires_auth():
    """401 when API_KEY env var is set and header is missing."""
    pool, conn, cursor = _make_pool()

    import os
    os.environ["API_KEY"] = "secret-key"
    try:
        with patch("api.core._get_pool", return_value=pool):
            from api.main import app
            transport = ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/supply/planned-orders/1001/approve",
                    json={"approved_by": "jane@example.com"},
                    # no X-API-Key header
                )
    finally:
        del os.environ["API_KEY"]

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_approve_planned_order_not_found():
    """404 when order_id not found or not in 'proposed' state."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/planned-orders/9999/approve",
                json={"approved_by": "jane@example.com"},
                headers={"X-API-Key": "test"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /supply/planned-orders/{id}/reject
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reject_planned_order_records_reason():
    """200 — rejection_reason stored, status = rejected."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1001, "rejected")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/planned-orders/1001/reject",
                json={"rejection_reason": "Demand expected to fall"},
                headers={"X-API-Key": "test"},
            )

    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# PUT /supply/planned-orders/{id}/release
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_release_planned_order_success():
    """200 — status changed to 'released'."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1001, "released", datetime.datetime(2026, 3, 6, 10, 0))

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/planned-orders/1001/release",
                headers={"X-API-Key": "test"},
            )

    assert resp.status_code == 200
    assert resp.json()["status"] == "released"


# ---------------------------------------------------------------------------
# POST /supply/planned-orders/generate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_planned_orders_async_202():
    """POST returns 202 accepted with job_id."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.generate_planned_orders.get_active_dfus_for_recommendation", return_value=[]), \
         patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start = lambda: None
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/planned-orders/generate",
                json={},
                headers={"X-API-Key": "test"},
            )

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert "job_id" in data
