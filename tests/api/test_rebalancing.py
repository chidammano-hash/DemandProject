"""API tests for inventory rebalancing endpoints.

Tests all 12 rebalancing REST endpoints using httpx AsyncClient with
ASGITransport -- no running server needed.
"""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


_NOW = datetime.datetime(2026, 3, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
_TODAY = datetime.date(2026, 3, 1)


# ---------------------------------------------------------------------------
# GET /inv-planning/rebalancing/kpis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kpis_200():
    """KPIs endpoint returns network balance stats + latest plan."""
    pool, conn, cursor = _make_pool()
    # First fetchone: mv_network_balance stats
    # Second fetchone: latest plan row
    cursor.fetchone.side_effect = [
        (150, 0.35, 42, 80, 60),  # total, avg_dos_cv, imbalanced, excess, shortage
        ("plan-001", 5000.0, 12000.0, 25000.0, 1.08, 42, "completed", _TODAY),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/rebalancing/kpis")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_multi_loc_items"] == 150
    assert data["imbalanced_items"] == 42
    assert data["total_excess_locs"] == 80
    assert data["total_shortage_locs"] == 60
    assert data["latest_plan"] is not None
    assert data["latest_plan"]["plan_id"] == "plan-001"
    assert data["latest_plan"]["status"] == "completed"
    assert data["latest_plan"]["computation_date"] == "2026-03-01"


# ---------------------------------------------------------------------------
# GET /inv-planning/rebalancing/network
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_network_list_200():
    """Network lanes endpoint returns total + rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = [
        ("lane-1", "LOC-A", "LOC-B", "truck", 1.5, 0.2, 3.0, 0.5, 50.0, 3, 10, 500, 1),
        ("lane-2", "LOC-C", "LOC-D", "rail", 0.8, 0.1, 5.0, 0.3, 100.0, 7, 50, None, 10),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/rebalancing/network")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["rows"]) == 2
    assert data["rows"][0]["lane_id"] == "lane-1"
    assert data["rows"][0]["source_loc"] == "LOC-A"
    assert data["rows"][1]["max_transfer_qty"] is None


# ---------------------------------------------------------------------------
# POST /inv-planning/rebalancing/network (auth required)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_network_create_requires_auth():
    """POST create lane without API key returns 401 when API_KEY is set."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/rebalancing/network",
                json={
                    "source_loc": "LOC-A",
                    "dest_loc": "LOC-B",
                    "cost_per_unit": 1.5,
                },
            )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_network_create_201():
    """POST create lane with auth returns lane_id."""
    pool, conn, cursor = _make_pool(fetchone_return=("lane-99",))

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/rebalancing/network",
                headers={"X-API-Key": "secret-key"},
                json={
                    "source_loc": "LOC-A",
                    "dest_loc": "LOC-B",
                    "cost_per_unit": 1.5,
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["lane_id"] == "lane-99"
    assert data["status"] == "created"


# ---------------------------------------------------------------------------
# DELETE /inv-planning/rebalancing/network/{lane_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deactivate_lane_404():
    """DELETE non-existent lane returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(
                "/inv-planning/rebalancing/network/nonexistent-lane",
                headers={"X-API-Key": "secret-key"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /inv-planning/rebalancing/imbalances
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_imbalances_200():
    """Imbalances endpoint returns count + rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3,)
    cursor.fetchall.return_value = [
        ("ITEM-001", 5, 1200.0, 30.5, 0.72, 2, 1),
        ("ITEM-002", 3, 800.0, 15.2, 0.85, 1, 2),
        ("ITEM-003", 4, 2000.0, 45.0, 0.60, 3, 1),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/rebalancing/imbalances")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert len(data["rows"]) == 3
    assert data["rows"][0]["item_no"] == "ITEM-001"
    assert data["rows"][0]["excess_loc_count"] == 2
    assert data["rows"][0]["shortage_loc_count"] == 1


# ---------------------------------------------------------------------------
# POST /inv-planning/rebalancing/compute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compute_202():
    """POST compute triggers background task, returns 202."""
    pool, conn, cursor = _make_pool()

    mock_executor = MagicMock()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}), \
         patch("concurrent.futures.ThreadPoolExecutor",
               return_value=mock_executor):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/rebalancing/compute",
                headers={"X-API-Key": "secret-key"},
                json={"solver": "greedy", "horizon_weeks": 4},
            )

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"


# ---------------------------------------------------------------------------
# GET /inv-planning/rebalancing/plans
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plans_list_200():
    """Plans list endpoint returns total + paginated rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = [
        ("plan-001", _TODAY, "greedy", "minimize_cost",
         5000.0, 12000.0, 25000.0, 1.08, 42, 8, "completed", 1500, _NOW),
        ("plan-002", _TODAY, "milp", "maximize_roi",
         3000.0, 8000.0, 18000.0, 1.25, 30, 5, "draft", 3200, _NOW),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/rebalancing/plans")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["rows"]) == 2
    assert data["rows"][0]["plan_id"] == "plan-001"
    assert data["rows"][0]["solver_method"] == "greedy"
    assert data["rows"][0]["status"] == "completed"


# ---------------------------------------------------------------------------
# GET /inv-planning/rebalancing/plans/{plan_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plan_detail_200():
    """Plan detail endpoint returns full plan info."""
    pool, conn, cursor = _make_pool(
        fetchone_return=(
            "plan-001", _TODAY, 4, "greedy", "minimize_cost",
            5000.0, 12000.0, 25000.0, 1.08,
            65.0, 82.0,      # balance before/after
            42, 8, "completed",
            "admin", _NOW, 1500, _NOW,
        ),
    )

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/rebalancing/plans/plan-001")

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_id"] == "plan-001"
    assert data["horizon_weeks"] == 4
    assert data["solver_method"] == "greedy"
    assert data["items_rebalanced"] == 42
    assert data["network_balance_before"] == 65.0
    assert data["network_balance_after"] == 82.0
    assert data["approved_by"] == "admin"


@pytest.mark.asyncio
async def test_plan_detail_404():
    """Plan detail for non-existent plan returns 404."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/rebalancing/plans/nonexistent")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /inv-planning/rebalancing/plans/{plan_id}/transfers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transfers_list_200():
    """Transfers list endpoint returns total + rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [
        (
            "txn-001", "ITEM-001", "LOC-A", "LOC-B", "truck",
            100.0, None,                              # recommended_qty, approved_qty
            500.0, 45.0, 200.0, 300.0,               # source: oh, dos, ss, excess
            50.0, 5.0, 200.0, 150.0,                 # dest: oh, dos, ss, shortage
            150.0, 80.0, 500.0,                       # costs: transfer, carrying saved, stockout avoided
            350.0, 2.33,                              # net_benefit, roi
            _TODAY, _TODAY, 3,                        # ship, arrival, lt
            0.85, "A", "high", "recommended",         # priority, abc, urgency, status
            None, None, None,                         # approved_by, rejection_reason, notes
        ),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/rebalancing/plans/plan-001/transfers")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert row["transfer_id"] == "txn-001"
    assert row["item_no"] == "ITEM-001"
    assert row["urgency"] == "high"
    assert row["status"] == "recommended"
    assert row["planned_ship_date"] == "2026-03-01"


# ---------------------------------------------------------------------------
# POST /inv-planning/rebalancing/transfers/{id}/approve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_transfer_200():
    """Approve transfer returns updated row with status=approved."""
    pool, conn, cursor = _make_pool(
        fetchone_return=("txn-001", "ITEM-001", "LOC-A", "LOC-B", 100.0, "approved"),
    )

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/rebalancing/transfers/txn-001/approve",
                headers={"X-API-Key": "secret-key"},
                json={"approved_by": "planner-1"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["transfer_id"] == "txn-001"
    assert data["status"] == "approved"
    assert data["approved_qty"] == 100.0


# ---------------------------------------------------------------------------
# POST /inv-planning/rebalancing/transfers/{id}/reject
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reject_transfer_422_missing_reason():
    """Reject transfer with empty reason returns 422."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/rebalancing/transfers/txn-001/reject",
                headers={"X-API-Key": "secret-key"},
                json={"rejection_reason": "   "},
            )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /inv-planning/rebalancing/plans/{plan_id}/approve-all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_all_200():
    """Bulk approve all recommended transfers in a plan."""
    pool, conn, cursor = _make_pool()
    cursor.rowcount = 5

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret-key"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/inv-planning/rebalancing/plans/plan-001/approve-all",
                headers={"X-API-Key": "secret-key"},
                json={"approved_by": "planner-1"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_id"] == "plan-001"
    assert data["approved_count"] == 5
    assert data["status"] == "approved"
