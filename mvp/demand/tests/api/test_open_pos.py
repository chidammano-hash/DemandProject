"""API tests for /supply/* endpoints — F1.3 Open PO Integration."""

import datetime
import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# /supply/open-pos/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_empty():
    """Empty table returns zero summary without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0, None, None, None, None, None, None, None, None)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/open-pos/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_open_lines"] == 0
    assert data["total_open_value_usd"] == 0.0
    assert data["past_due_lines"] == 0
    assert data["suppliers_with_open_pos"] == 0
    assert data["last_loaded_at"] is None


@pytest.mark.asyncio
async def test_summary_with_data():
    """Summary returns correct KPIs when POs exist."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        8,           # total_open_lines
        125000.0,    # total_open_value_usd
        900.0,       # qty_open
        100.0,       # qty_partial
        2,           # past_due_lines
        8750.0,      # past_due_value_usd
        14.5,        # avg_days_past_due
        3,           # suppliers_with_open_pos
        datetime.datetime(2026, 3, 7, 6, 0, 0),  # last_loaded_at
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/open-pos/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_open_lines"] == 8
    assert data["total_open_value_usd"] == 125000.0
    assert data["total_open_qty_by_status"]["open"] == 900.0
    assert data["total_open_qty_by_status"]["partially_received"] == 100.0
    assert data["past_due_lines"] == 2
    assert data["past_due_value_usd"] == 8750.0
    assert data["avg_days_past_due"] == 14.5
    assert data["suppliers_with_open_pos"] == 3
    assert data["last_loaded_at"] is not None


# ---------------------------------------------------------------------------
# /supply/open-pos
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_open_pos_empty():
    """No PO data returns empty items list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (0,),                          # count
        (0,),                          # po_data_available check
        (None,),                       # max load_ts
    ]
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/open-pos")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["open_po_data_available"] is False


@pytest.mark.asyncio
async def test_open_pos_with_rows():
    """Open PO rows are returned with correct shape."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        (2,),                                          # count
        (8,),                                          # po_data_available
        (datetime.datetime(2026, 3, 7, 6, 0, 0),),    # max load_ts
    ]
    cursor.fetchall.return_value = [
        (
            "PO-4521", 1, "100320", "1401-BULK", "VENDOR-0042", "Acme Supply Co.",
            datetime.date(2026, 2, 15),  # po_date
            150.0, 150.0, 0.0,           # ordered, confirmed, received
            150.0,                        # open_qty
            12.50,                        # unit_cost
            1875.0,                       # line_value
            datetime.date(2026, 3, 14),  # promised
            datetime.date(2026, 3, 14),  # confirmed
            None,                         # revised
            datetime.date(2026, 3, 14),  # effective
            0,                            # days_past_due
            "open",
        ),
        (
            "PO-4604", 1, "400100", "1401-BULK", "VENDOR-0077", "Epsilon Trading Co.",
            datetime.date(2026, 2, 1),
            1000.0, 900.0, 100.0,
            800.0,
            3.20,
            2880.0,
            datetime.date(2026, 3, 10),
            datetime.date(2026, 3, 10),
            None,
            datetime.date(2026, 3, 10),
            0,
            "partially_received",
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/open-pos")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2
    item = data["items"][0]
    assert item["po_number"] == "PO-4521"
    assert item["item_no"] == "100320"
    assert item["open_qty"] == 150.0
    assert item["line_value"] == 1875.0
    assert item["days_past_due"] == 0
    assert item["line_status"] == "open"
    assert item["supplier_name"] == "Acme Supply Co."


@pytest.mark.asyncio
async def test_open_pos_page_size_capped():
    """page_size > 200 is capped without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(0,), (0,), (None,)]
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/open-pos?page_size=9999")

    assert resp.status_code == 200
    assert resp.json()["page_size"] == 200


# ---------------------------------------------------------------------------
# /supply/past-due-pos
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_past_due_empty():
    """No past-due POs returns empty list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/past-due-pos")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


@pytest.mark.asyncio
async def test_past_due_with_rows():
    """Past-due PO rows include severity classification."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (
            "PO-4605", "500050", "4000-WEST", "Gamma Wholesale LLC",
            200.0,                          # open_qty
            datetime.date(2026, 3, 5),      # confirmed_delivery_date
            31,                             # days_past_due
            3600.0,                         # line_value
            "critical",                     # severity
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/supply/past-due-pos")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    item = data["items"][0]
    assert item["po_number"] == "PO-4605"
    assert item["days_past_due"] == 31
    assert item["severity"] == "critical"
    assert item["open_qty"] == 200.0
