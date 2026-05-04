"""
API tests for F2.4 — /supply/purchase-orders/* endpoints.
"""

import datetime
import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Helper: a realistic PO row (20 columns from list_purchase_orders query)
# ---------------------------------------------------------------------------

def _po_row(
    po_number="DS-2026-04-001",
    line_number=1,
    item_id="100320",
    item_description="Bulk Cleaning Solution",
    loc="1401-BULK",
    supplier_id="SUP-4821",
    supplier_name="ABC Trading Co",
    ordered_qty=316.0,
    unit_cost=24.00,
    total_value=7584.0,
    currency="USD",
    po_date=None,
    requested_delivery_date=None,
    confirmed_delivery_date=None,
    status="proposed",
    source_exception_id=7834,
    created_by="planner1",
    planner_approved_by=None,
    buyer_released_by=None,
    erp_po_number=None,
):
    return (
        po_number, line_number, item_id, item_description, loc,
        supplier_id, supplier_name, ordered_qty, unit_cost, total_value, currency,
        po_date or datetime.date(2026, 4, 15),
        requested_delivery_date or datetime.date(2026, 4, 28),
        confirmed_delivery_date,
        status, source_exception_id, created_by,
        planner_approved_by, buyer_released_by, erp_po_number,
    )


# ---------------------------------------------------------------------------
# GET /supply/purchase-orders
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_purchase_orders_list():
    """Returns list of POs with correct shape."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2, 15168.0)  # total, total_value
    cursor.fetchall.return_value = [_po_row(), _po_row(po_number="DS-2026-04-002", status="planner_approved")]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/purchase-orders")

    assert resp.status_code == 200
    data = resp.json()
    assert "orders" in data
    assert data["total"] == 2
    assert data["total_value"] == pytest.approx(15168.0)
    assert len(data["orders"]) == 2
    order = data["orders"][0]
    assert order["po_number"] == "DS-2026-04-001"
    assert order["item_id"] == "100320"
    assert order["status"] == "proposed"


@pytest.mark.asyncio
async def test_get_purchase_orders_empty():
    """Empty table returns zeros and empty list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0, 0)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/purchase-orders", params={"status": "proposed"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["orders"] == []


# ---------------------------------------------------------------------------
# POST /supply/purchase-orders/from-exception/{exception_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_po_from_exception_returns_201():
    """POST approve returns 201 with po_number."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (7584.0, datetime.date(2026, 4, 28))

    # Patch at the source module where the function is defined.
    # The endpoint does `from scripts.inventory.release_planned_orders import create_po_from_exception as _create_po`
    # inside the function body, so we must patch the original module.
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.inventory.release_planned_orders.create_po_from_exception",
               return_value="DS-2026-04-001"):

        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/purchase-orders/from-exception/7834",
                json={
                    "performed_by": "planner1",
                    "ordered_qty": 316.0,
                },
                headers={"x-api-key": ""},
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["po_number"] == "DS-2026-04-001"
    assert data["status"] == "proposed"


# ---------------------------------------------------------------------------
# POST /supply/purchase-orders/{po_number}/approve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_purchase_order_success():
    """Approve transitions proposed → planner_approved."""
    pool, conn, cursor = _make_pool()
    # RETURNING po_line_id, ordered_qty for UPDATE, then log INSERT
    cursor.fetchall.return_value = [(101, 316.0)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/purchase-orders/DS-2026-04-001/approve",
                json={"approved_by": "planner1"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["po_number"] == "DS-2026-04-001"
    assert data["status"] == "planner_approved"
    assert data["approved_by"] == "planner1"


@pytest.mark.asyncio
async def test_approve_purchase_order_not_found():
    """Approving a non-existent or already-approved PO returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []  # UPDATE returned no rows

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/purchase-orders/DS-9999-99-999/approve",
                json={"approved_by": "planner1"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /supply/purchase-orders/{po_number}/release
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_release_purchase_order_success():
    """Release transitions planner_approved → buyer_released."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [(101,)]  # RETURNING po_line_id

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/purchase-orders/DS-2026-04-001/release",
                json={"released_by": "buyer1"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["po_number"] == "DS-2026-04-001"
    assert data["status"] == "buyer_released"
    assert data["released_by"] == "buyer1"


@pytest.mark.asyncio
async def test_release_po_not_in_approved_state():
    """Releasing a PO not in planner_approved state returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/purchase-orders/DS-9999-99-999/release",
                json={"released_by": "buyer1"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /supply/purchase-orders/export-csv
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_export_csv_returns_filename():
    """Export returns filename and line_count."""
    pool, conn, cursor = _make_pool()
    # 2 PO lines in buyer_released state
    cursor.fetchall.return_value = [
        ("DS-2026-04-001", 1, "100320", "Item Desc", "1401-BULK",
         "SUP-001", "Supplier", 316.0, "EA", 24.0, 7584.0, "USD",
         datetime.date(2026, 4, 28), datetime.date(2026, 4, 15),
         None, None, None, 7834, None),
        ("DS-2026-04-001", 2, "204771", "Item 2", "2203-STD",
         "SUP-001", "Supplier", 80.0, "EA", 67.5, 5400.0, "USD",
         datetime.date(2026, 4, 28), datetime.date(2026, 4, 15),
         None, None, None, 7835, None),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/purchase-orders/export-csv",
                json={
                    "po_numbers": ["DS-2026-04-001"],
                    "exported_by": "bob.chen@company.com",
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "filename" in data
    assert data["filename"].startswith("PO_export_")
    assert data["filename"].endswith(".csv")
    assert data["line_count"] == 2
    assert "csv_content" in data
    assert "PO_NUMBER" in data["csv_content"]  # header row present


@pytest.mark.asyncio
async def test_export_csv_no_released_pos():
    """No released POs for the given po_numbers → 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/supply/purchase-orders/export-csv",
                json={"po_numbers": ["DS-9999-99-999"], "exported_by": "bob"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /supply/purchase-orders/{po_number}/timeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_timeline_returns_all_events():
    """Timeline includes all audit log entries."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("po_sent",)  # current_status
    cursor.fetchall.return_value = [
        ("proposed", "system", datetime.datetime(2026, 4, 15, 9, 14),
         None, "proposed", None, None, None, "Auto-created from exception EXC-7834"),
        ("planner_approved", "jane.smith@co.com", datetime.datetime(2026, 4, 15, 9, 22),
         "proposed", "planner_approved", None, 316.0, None, None),
        ("buyer_released", "bob.chen@co.com", datetime.datetime(2026, 4, 15, 11, 5),
         "planner_approved", "buyer_released", None, None, None, "Confirmed Apr 28"),
        ("po_sent", "system", datetime.datetime(2026, 4, 15, 11, 6),
         "buyer_released", "po_sent", None, None, None, "CSV exported"),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/purchase-orders/DS-2026-04-001/timeline")

    assert resp.status_code == 200
    data = resp.json()
    assert data["po_number"] == "DS-2026-04-001"
    assert data["current_status"] == "po_sent"
    assert len(data["timeline"]) == 4
    assert data["timeline"][0]["action"] == "proposed"
    assert data["timeline"][3]["action"] == "po_sent"


@pytest.mark.asyncio
async def test_get_timeline_po_not_found():
    """Non-existent PO number returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # PO not found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/purchase-orders/DS-9999-99-999/timeline")

    assert resp.status_code == 404
