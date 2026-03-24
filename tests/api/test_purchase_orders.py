"""Tests for purchase orders API endpoints."""
import pytest
from unittest.mock import patch

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_po_rows_empty():
    pool, conn, cursor = _make_pool(
        fetchone_return=(0,),
        fetchall_return=[],
    )
    cursor.description = [("po_ck",), ("po_number",), ("site_id",), ("loc",),
                          ("source",), ("item_id",), ("ordered_qty",), ("net_price",),
                          ("gross_value",), ("closure_code",), ("po_hdr_status",),
                          ("po_line_status",), ("receipt_status",), ("supplier_id",),
                          ("supplier_name",), ("carrier_name",), ("delivery_date",),
                          ("original_delivery_date",), ("current_ship_date",),
                          ("original_ship_date",), ("po_type",), ("is_closed",),
                          ("lead_time_planned",), ("lead_time_actual",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/purchase-orders/rows")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["rows"] == []


@pytest.mark.asyncio
async def test_po_rows_with_filter():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (1, "3061972", "25", "6401-OTHR", "106037-700484", "540765",
         10.0, 524.0, 5240.0, "CLOSED", "INACTIVE", "INACTIVE", "DELIVERED",
         "106037", "FRUIT OF THE VINES", "ADVANTAGE TRANS",
         "2023-08-30", "2022-03-08", "2023-07-18", "2022-01-10",
         "YO", True, 100, 232),
    ]
    cursor.description = [("po_ck",), ("po_number",), ("site_id",), ("loc",),
                          ("source",), ("item_id",), ("ordered_qty",), ("net_price",),
                          ("gross_value",), ("closure_code",), ("po_hdr_status",),
                          ("po_line_status",), ("receipt_status",), ("supplier_id",),
                          ("supplier_name",), ("carrier_name",), ("delivery_date",),
                          ("original_delivery_date",), ("current_ship_date",),
                          ("original_ship_date",), ("po_type",), ("is_closed",),
                          ("lead_time_planned",), ("lead_time_actual",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/purchase-orders/rows?status=closed")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


@pytest.mark.asyncio
async def test_po_search():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    cursor.description = [("po_ck",), ("po_number",), ("item_id",), ("loc",),
                          ("source",), ("supplier_name",), ("closure_code",),
                          ("delivery_date",), ("ordered_qty",), ("net_price",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/purchase-orders/search?q=3061972")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_po_by_number():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    cursor.description = [("po_ck",), ("po_number",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/purchase-orders/by-po/3061972")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_po_summary():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5641373, 5354000, 287373, 100000, 2278, 50000, 1000000.0, 50000.0, 950000.0)
    cursor.description = [("total_lines",), ("closed_lines",), ("open_lines",),
                          ("distinct_pos",), ("distinct_suppliers",), ("distinct_items",),
                          ("total_value",), ("open_value",), ("closed_value",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/purchase-orders/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_lines"] == 5641373
    assert data["open_lines"] == 287373


@pytest.mark.asyncio
async def test_po_aging():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("0-30", 1000, 50000.0),
        ("30-60", 500, 25000.0),
    ]
    cursor.description = [("age_bucket",), ("line_count",), ("total_value",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/purchase-orders/aging")
    assert resp.status_code == 200
    assert len(resp.json()["buckets"]) == 2


@pytest.mark.asyncio
async def test_po_otd():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("106037", "FRUIT OF THE VINES", 100, 85, 85.0, 45.3),
    ]
    cursor.description = [("supplier_id",), ("supplier_name",), ("total_closed",),
                          ("on_time",), ("otd_pct",), ("avg_lead_time_days",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/purchase-orders/otd")
    assert resp.status_code == 200
    assert len(resp.json()["suppliers"]) == 1
    assert resp.json()["suppliers"][0]["otd_pct"] == 85.0
