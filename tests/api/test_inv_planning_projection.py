"""API tests for /inv-planning/projection/* endpoints — F1.2."""

import datetime
import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# GET /inv-planning/projection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_projection_404_no_data():
    """No projection data → 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # no run_id found

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/projection?item_id=100320&loc=1401-BULK")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_projection_success_all_scenarios():
    """200 with run data returns key_dates and projection array."""
    import uuid
    pool, conn, cursor = _make_pool()
    run_id = str(uuid.uuid4())
    today = datetime.date.today()

    cursor.fetchone.side_effect = [
        # run_id row
        (run_id, "production_forecast", "2026-03"),
        # inventory snapshot
        (120.0,),
        # safety stock
        (60.0,),
        # po count
        (8,),
    ]
    cursor.fetchall.side_effect = [
        # summary view rows (scenario × key dates)
        [
            ("no_order", today + datetime.timedelta(days=4), today + datetime.timedelta(days=12), None, 12, datetime.datetime(2026, 3, 7, 7)),
            ("with_open_po", today + datetime.timedelta(days=4), None, None, None, datetime.datetime(2026, 3, 7, 7)),
            ("with_planned_orders", today + datetime.timedelta(days=4), None, None, None, datetime.datetime(2026, 3, 7, 7)),
        ],
        # projection rows (one per scenario per day — minimal)
        [
            (today, "no_order", 103.7, 16.3, 0.0, False, False, False),
            (today, "with_open_po", 103.7, 16.3, 0.0, False, False, False),
            (today, "with_planned_orders", 103.7, 16.3, 0.0, False, False, False),
        ],
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/projection?item_id=100320&loc=1401-BULK")

    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == "100320"
    assert data["loc"] == "1401-BULK"
    assert data["current_qty_on_hand"] == 120.0
    assert data["safety_stock"] == 60.0
    assert data["forecast_source"] == "production_forecast"
    assert data["open_po_data_available"] is True
    assert "no_order" in data["key_dates"]
    assert "with_open_po" in data["key_dates"]
    assert len(data["projection"]) == 1  # one unique date


@pytest.mark.asyncio
async def test_get_projection_stockout_in_key_dates():
    """stockout_date is present for no_order scenario."""
    import uuid
    pool, conn, cursor = _make_pool()
    run_id = str(uuid.uuid4())
    today = datetime.date.today()
    stockout = today + datetime.timedelta(days=12)

    cursor.fetchone.side_effect = [
        (run_id, "production_forecast", "2026-03"),
        (120.0,),
        (60.0,),
        (8,),
    ]
    cursor.fetchall.side_effect = [
        [("no_order", today + datetime.timedelta(days=4), stockout, None, 12, datetime.datetime(2026, 3, 7, 7))],
        [],
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/projection?item_id=100320&loc=1401-BULK")

    assert resp.status_code == 200
    data = resp.json()
    assert data["key_dates"]["no_order"]["stockout_date"] == stockout.isoformat()
    assert data["key_dates"]["no_order"]["days_until_stockout"] == 12


@pytest.mark.asyncio
async def test_get_projection_no_po_data():
    """When fact_open_purchase_orders is empty, open_po_data_available = false."""
    import uuid
    pool, conn, cursor = _make_pool()
    run_id = str(uuid.uuid4())
    today = datetime.date.today()

    cursor.fetchone.side_effect = [
        (run_id, "fallback_avg", None),
        (50.0,),
        (0.0,),
        (0,),  # po count = 0
    ]
    cursor.fetchall.side_effect = [
        [("no_order", None, None, None, None, datetime.datetime(2026, 3, 7, 7))],
        [],
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/projection?item_id=100320&loc=1401-BULK")

    assert resp.status_code == 200
    data = resp.json()
    assert data["open_po_data_available"] is False
    assert data["forecast_source"] == "fallback_avg"


# ---------------------------------------------------------------------------
# GET /inv-planning/projection/at-risk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_projection_at_risk_empty():
    """No at-risk DFUs returns empty list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/projection/at-risk")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


@pytest.mark.asyncio
async def test_get_projection_at_risk_list():
    """At-risk list filters correctly by horizon_days."""
    import datetime
    pool, conn, cursor = _make_pool()
    today = datetime.date.today()

    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = [
        (
            "100320", "1401-BULK",
            today + datetime.timedelta(days=7),   # stockout_date
            7,                                     # days_until_stockout
            today + datetime.timedelta(days=3),   # reorder_trigger_date
            120.0,                                 # current_qty
            60.0,                                  # safety_stock
            "critical",
        ),
        (
            "200147", "1401-BULK",
            today + datetime.timedelta(days=20),
            20,
            today + datetime.timedelta(days=15),
            80.0,
            40.0,
            "high",
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/projection/at-risk?horizon_days=30")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["horizon_days"] == 30
    items = data["items"]
    assert len(items) == 2
    assert items[0]["item_id"] == "100320"
    assert items[0]["severity"] == "critical"
    assert items[0]["days_until_stockout"] == 7
    assert items[1]["severity"] == "high"


# ---------------------------------------------------------------------------
# POST /inv-planning/projection/refresh
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_projection_refresh():
    """POST refresh returns ok status and rows_written > 0."""
    pool, conn, cursor = _make_pool()
    today = datetime.date.today()

    # compute_dfu_projection calls: current_inventory, safety_stock,
    #   get_daily_demand_rates (fetchall → rows, fallback fetchone),
    #   get_open_po_receipts (fetchall),
    #   write_projection_rows (delete + executemany)

    # Patch at the script module level (imported inside function body)
    with patch("api.core._get_pool", return_value=pool), \
         patch("scripts.inventory.compute_inventory_projection.compute_dfu_projection",
               return_value=(270, "test-run-id")), \
         patch("scripts.inventory.compute_inventory_projection.refresh_summary_view"):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/projection/refresh",
                json={"item_id": "100320", "loc": "1401-BULK", "horizon_days": 90},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["rows_written"] == 270
    assert data["run_id"] == "test-run-id"
