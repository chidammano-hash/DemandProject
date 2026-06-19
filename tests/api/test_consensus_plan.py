"""
API tests for F2.3 — /forecast/overrides/* and /forecast/consensus-plan endpoints.
"""

import datetime
import pytest
from unittest.mock import patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# Helper: a realistic override row (27 columns from list_overrides query)
# ---------------------------------------------------------------------------

def _override_row(
    override_id=1,
    item_id="100320",
    loc="1401-BULK",
    override_month=None,
    override_type="PROMO",
    override_qty=None,
    override_multiplier=1.25,
    override_additive_qty=0.0,
    is_hard_override=False,
    override_reason="Summer promo",
    override_note=None,
    created_by="planner1",
    created_at=None,
    valid_from=None,
    valid_to=None,
    approved_by=None,
    approved_at=None,
    rejected_by=None,
    rejected_at=None,
    rejection_reason=None,
    status="pending_approval",
    requires_approval=True,
    priority_rank=5,
    statistical_qty_at_creation=400.0,
    estimated_impact_units=100.0,
    estimated_impact_value=500.0,
    currency="USD",
):
    return (
        override_id,
        item_id,
        loc,
        override_month or datetime.date(2026, 5, 1),
        override_type,
        override_qty,
        override_multiplier,
        override_additive_qty,
        is_hard_override,
        override_reason,
        override_note,
        created_by,
        created_at or datetime.datetime(2026, 3, 1, 8, 0, 0),
        valid_from or datetime.date(2026, 4, 1),
        valid_to or datetime.date(2026, 9, 30),
        approved_by,
        approved_at,
        rejected_by,
        rejected_at,
        rejection_reason,
        status,
        requires_approval,
        priority_rank,
        statistical_qty_at_creation,
        estimated_impact_units,
        estimated_impact_value,
        currency,
    )


# ---------------------------------------------------------------------------
# GET /forecast/overrides/summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_override_summary_returns_counts():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3, 5, 1, 0, 0, 4, 2400.0, 12000.0)
    cursor.fetchall.return_value = [("PROMO", 3), ("MANUAL", 2)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/overrides/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["by_status"]["pending_approval"] == 3
    assert data["by_status"]["approved"] == 5
    assert data["by_status"]["rejected"] == 1
    assert data["dfu_count_overridden"] == 4
    assert data["total_uplift_units"] == pytest.approx(2400.0)
    assert data["total_uplift_value"] == pytest.approx(12000.0)
    assert data["by_type"]["PROMO"] == 3
    assert data["by_type"]["MANUAL"] == 2


@pytest.mark.asyncio
async def test_get_override_summary_empty_db():
    """Empty table returns zeros."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0, 0, 0, 0, 0, 0, 0, 0)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/overrides/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["by_status"]["pending_approval"] == 0
    assert data["by_type"] == {}


# ---------------------------------------------------------------------------
# GET /forecast/overrides
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_overrides_returns_rows():
    pool, conn, cursor = _make_pool()
    # list_overrides uses fetchone for COUNT then fetchall for rows
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = [_override_row()]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/overrides")

    assert resp.status_code == 200
    data = resp.json()
    assert "overrides" in data
    assert data["total"] == 2
    assert len(data["overrides"]) == 1
    ov = data["overrides"][0]
    assert ov["item_id"] == "100320"
    assert ov["override_type"] == "PROMO"
    assert ov["status"] == "pending_approval"


@pytest.mark.asyncio
async def test_list_overrides_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/forecast/overrides", params={"status": "pending_approval"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["overrides"] == []


# ---------------------------------------------------------------------------
# POST /forecast/overrides
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_override_auto_approved():
    """Small override (no threshold breach) → auto-approved."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1, "approved", False, 30.0, 150.0)  # RETURNING

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.consensus_plan.record_override_approval") as mock_ledger, \
         patch.dict("os.environ", {}, clear=False):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/forecast/overrides",
                json={
                    "item_id": "100320",
                    "loc": "1401-BULK",
                    "override_month": "2026-05-01",
                    "override_type": "PROMO",
                    "override_multiplier": 1.05,
                    "override_reason": "Small promo lift",
                    "created_by": "planner1",
                    "valid_from": "2026-05-01",
                    "valid_to": "2026-05-31",
                    "statistical_qty": 400.0,
                },
            )

    assert resp.status_code == 201
    data = resp.json()
    assert "override_id" in data
    assert data["status"] == "approved"
    mock_ledger.assert_called_once()


@pytest.mark.asyncio
async def test_submit_override_requires_approval():
    """Large uplift → requires_approval=True."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2, "pending_approval", True, 500.0, 2500.0)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/forecast/overrides",
                json={
                    "item_id": "100320",
                    "loc": "1401-BULK",
                    "override_month": "2026-05-01",
                    "override_type": "PROMO",
                    "override_multiplier": 2.0,  # 100% uplift > 20% threshold
                    "override_reason": "Major promo event",
                    "created_by": "planner2",
                    "valid_from": "2026-05-01",
                    "valid_to": "2026-05-31",
                    "statistical_qty": 400.0,
                },
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["requires_approval"] is True
    assert data["status"] == "pending_approval"


@pytest.mark.asyncio
async def test_submit_override_invalid_type():
    """Invalid override_type → 422 validation error."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/forecast/overrides",
                json={
                    "item_id": "100320",
                    "loc": "1401-BULK",
                    "override_month": "2026-05-01",
                    "override_type": "INVALID_TYPE",
                    "override_reason": "Test",
                    "created_by": "planner1",
                    "valid_from": "2026-05-01",
                    "valid_to": "2026-05-31",
                },
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_submit_override_multiplier_out_of_bounds():
    """Multiplier > 5.0 (config max) → 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/forecast/overrides",
                json={
                    "item_id": "100320",
                    "loc": "1401-BULK",
                    "override_month": "2026-05-01",
                    "override_type": "PROMO",
                    "override_multiplier": 10.0,
                    "override_reason": "Test",
                    "created_by": "planner1",
                    "valid_from": "2026-05-01",
                    "valid_to": "2026-05-31",
                },
            )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# PUT /forecast/overrides/{id}/approve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_override_success():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        42, "approved", "manager", datetime.datetime(2026, 3, 7, 10, 0, 0),
        "100320", "1401-BULK", datetime.date(2026, 5, 1), "PROMO",
    )

    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.forecasting.consensus_plan.record_override_approval") as mock_ledger:
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/forecast/overrides/42/approve",
                json={"approved_by": "manager"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["override_id"] == 42
    assert data["status"] == "approved"
    assert data["approved_by"] == "manager"
    mock_ledger.assert_called_once()


@pytest.mark.asyncio
async def test_approve_override_not_found():
    """Override doesn't exist or not in pending_approval → 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/forecast/overrides/999/approve",
                json={"approved_by": "manager"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /forecast/overrides/{id}/reject
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reject_override_success():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (42, "rejected")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/forecast/overrides/42/reject",
                json={"rejected_by": "manager", "rejection_reason": "Not aligned with plan"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["override_id"] == 42
    assert data["status"] == "rejected"


@pytest.mark.asyncio
async def test_reject_override_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/forecast/overrides/999/reject",
                json={"rejected_by": "manager", "rejection_reason": "Test"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /forecast/overrides/{id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_override_soft_deletes():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (42, "superseded")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/forecast/overrides/42")

    assert resp.status_code == 200
    data = resp.json()
    assert data["override_id"] == 42
    assert data["status"] == "superseded"


@pytest.mark.asyncio
async def test_delete_override_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete("/forecast/overrides/999")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /forecast/consensus-plan
# ---------------------------------------------------------------------------

def _consensus_row(plan_month=None):
    return (
        plan_month or datetime.date(2026, 5, 1),  # plan_month
        450.0,    # statistical_qty
        320.0,    # statistical_p10
        580.0,    # statistical_p90
        50.0,     # override_qty (uplift applied)
        500.0,    # consensus_qty
        360.0,    # consensus_p10
        640.0,    # consensus_p90
        True,     # override_applied
        "PROMO",  # override_type
        1.10,     # override_multiplier
        False,    # is_hard_override
        "planner1",  # overrider
        "manager",   # approver
        11.11,    # uplift_pct
    )


@pytest.mark.asyncio
async def test_get_consensus_plan_returns_months():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("2026-04-01_production",)  # plan_version lookup
    cursor.fetchall.return_value = [
        _consensus_row(datetime.date(2026, 5, 1)),
        _consensus_row(datetime.date(2026, 6, 1)),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/consensus-plan",
                params={"item_id": "100320", "loc": "1401-BULK"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "months" in data
    assert len(data["months"]) == 2
    m = data["months"][0]
    assert m["statistical_qty"] == pytest.approx(450.0)
    assert m["consensus_qty"] == pytest.approx(500.0)
    assert m["override_applied"] is True
    assert m["override_type"] == "PROMO"
    assert m["uplift_pct"] == pytest.approx(11.11)


@pytest.mark.asyncio
async def test_get_consensus_plan_with_explicit_version():
    """When plan_version is provided, skip the version-lookup fetchone."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [_consensus_row()]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/consensus-plan",
                params={
                    "item_id": "100320",
                    "loc": "1401-BULK",
                    "plan_version": "2026-04-01_production",
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_version"] == "2026-04-01_production"


@pytest.mark.asyncio
async def test_get_consensus_plan_404_no_version():
    """No plan_version in DB → 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None  # version lookup fails

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/consensus-plan",
                params={"item_id": "UNKNOWN", "loc": "X"},
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_consensus_plan_404_no_rows():
    """plan_version resolves but no rows in consensus table → 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = ("2026-04-01_production",)
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/forecast/consensus-plan",
                params={"item_id": "100320", "loc": "1401-BULK"},
            )

    assert resp.status_code == 404
