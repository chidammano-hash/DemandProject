"""API tests for /inv-planning/policies and /inv-planning/policy-assignments endpoints.

IPfeature5 — Replenishment Policy Management.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


# ---------------------------------------------------------------------------
# GET /inv-planning/policies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_policies_200():
    """GET /inv-planning/policies returns 200 with policies list."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("A_continuous_v1", "A-Class Continuous Review", "continuous_rop", "A",
         None, 0.98, True, True, True, 150),
        ("B_periodic_v1", "B-Class Periodic Review", "periodic_review", "B",
         28, 0.95, True, True, True, 80),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/policies")

    assert resp.status_code == 200
    data = resp.json()
    assert "policies" in data
    assert isinstance(data["policies"], list)


@pytest.mark.asyncio
async def test_get_policies_structure():
    """Policies response has expected keys per policy."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        ("A_continuous_v1", "A-Class Continuous Review", "continuous_rop", "A",
         None, 0.98, True, True, True, 42),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/policies")

    data = resp.json()
    assert len(data["policies"]) == 1
    p = data["policies"][0]
    for key in ("policy_id", "policy_name", "policy_type", "service_level", "use_eoq", "use_safety_stock", "dfu_count"):
        assert key in p


@pytest.mark.asyncio
async def test_get_policies_empty_db():
    """Empty DB returns empty policies list, not 500."""
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/policies")

    assert resp.status_code == 200
    assert resp.json()["policies"] == []


# ---------------------------------------------------------------------------
# POST /inv-planning/policies (auth required)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_policy_no_auth_returns_401():
    """POST /inv-planning/policies without API key returns 401 when API_KEY is set."""
    pool, conn, cursor = _make_pool()

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret123"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/inv-planning/policies", json={
                "policy_id": "test_v1",
                "policy_name": "Test Policy",
                "policy_type": "manual",
            })

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_create_policy_with_auth_returns_201():
    """POST /inv-planning/policies with valid auth returns 201."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (
        "test_v1", "Test Policy", "manual", None, None, None, False, False, True, None
    )
    cursor.description = [
        ("policy_id",), ("policy_name",), ("policy_type",), ("segment",),
        ("review_cycle_days",), ("service_level",), ("use_eoq",),
        ("use_safety_stock",), ("active",), ("notes",),
    ]

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": ""}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/inv-planning/policies", json={
                "policy_id": "test_v1",
                "policy_name": "Test Policy",
                "policy_type": "manual",
            })

    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# PUT /inv-planning/policies/{policy_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_policy_200():
    """PUT /inv-planning/policies/{policy_id} returns 200 with updated policy."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [
        # UPDATE RETURNING row
        ("A_continuous_v1", "A-Class Continuous Review", "continuous_rop", "A",
         None, 0.99, True, True, True, None),
        # COUNT for dfu_count
        (150,),
    ]
    cursor.description = [
        ("policy_id",), ("policy_name",), ("policy_type",), ("segment",),
        ("review_cycle_days",), ("service_level",), ("use_eoq",),
        ("use_safety_stock",), ("active",), ("notes",),
    ]

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": ""}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put("/inv-planning/policies/A_continuous_v1", json={
                "service_level": 0.99,
            })

    assert resp.status_code == 200
    data = resp.json()
    assert "policy_id" in data


# ---------------------------------------------------------------------------
# GET /inv-planning/policy-assignments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_policy_assignments_200():
    """GET /inv-planning/policy-assignments returns paginated rows."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (10,)
    cursor.fetchall.return_value = [
        ("ITEM001", "LOC1", "A_continuous_v1", "A-Class Continuous Review",
         "continuous_rop", None, "system", None),
    ]
    cursor.description = [
        ("item_no",), ("loc",), ("policy_id",), ("policy_name",),
        ("policy_type",), ("override_reason",), ("assigned_by",), ("effective_date",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/policy-assignments")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_get_policy_assignments_filter_params():
    """Filter query params accepted without error."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    cursor.description = [
        ("item_no",), ("loc",), ("policy_id",), ("policy_name",),
        ("policy_type",), ("override_reason",), ("assigned_by",), ("effective_date",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/policy-assignments?item=ITEM001&policy_id=A_continuous_v1")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /inv-planning/policy-assignments/assign
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assign_individual_returns_assigned_count():
    """POST assign (individual) returns assigned_count=1."""
    pool, conn, cursor = _make_pool()
    # Policy exists
    cursor.fetchone.return_value = (1,)
    cursor.rowcount = 1

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": ""}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/inv-planning/policy-assignments/assign", json={
                "item_no": "ITEM001",
                "loc": "LOC1",
                "policy_id": "A_continuous_v1",
            })

    assert resp.status_code == 200
    data = resp.json()
    assert "assigned_count" in data
    assert "failed_count" in data
    assert "already_assigned_count" in data


@pytest.mark.asyncio
async def test_assign_bulk_by_segment():
    """POST assign (bulk by segment) returns assigned_count >= 0."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.rowcount = 5

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": ""}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/inv-planning/policy-assignments/assign", json={
                "segment": "A",
                "policy_id": "A_continuous_v1",
            })

    assert resp.status_code == 200
    data = resp.json()
    assert "assigned_count" in data


@pytest.mark.asyncio
async def test_assign_invalid_body_returns_422():
    """POST assign with neither individual nor bulk fields returns 422."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": ""}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/inv-planning/policy-assignments/assign", json={})

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /inv-planning/policy-assignments/compliance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compliance_200():
    """GET /inv-planning/policy-assignments/compliance returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(500,), (450,)]
    cursor.fetchall.return_value = [
        ("A_continuous_v1", "A-Class Continuous Review", "continuous_rop", 150, None),
        ("B_periodic_v1", "B-Class Periodic Review", "periodic_review", 80, 45.0),
    ]
    cursor.description = [
        ("policy_id",), ("policy_name",), ("policy_type",), ("dfu_count",), ("avg_dos",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/policy-assignments/compliance")

    assert resp.status_code == 200
    data = resp.json()
    assert "total_dfus" in data
    assert "assigned_count" in data
    assert "unassigned_count" in data
    assert "assignment_pct" in data
    assert "by_policy" in data


@pytest.mark.asyncio
async def test_compliance_assignment_pct_range():
    """assignment_pct is between 0 and 100."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(200,), (180,)]
    cursor.fetchall.return_value = [
        ("A_continuous_v1", "A-Class Continuous Review", "continuous_rop", 100, None),
    ]
    cursor.description = [
        ("policy_id",), ("policy_name",), ("policy_type",), ("dfu_count",), ("avg_dos",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/policy-assignments/compliance")

    data = resp.json()
    assert 0.0 <= data["assignment_pct"] <= 100.0


@pytest.mark.asyncio
async def test_compliance_by_policy_structure():
    """by_policy dict has expected keys per policy."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(100,), (80,)]
    cursor.fetchall.return_value = [
        ("A_continuous_v1", "A-Class Continuous Review", "continuous_rop", 80, 30.5),
    ]
    cursor.description = [
        ("policy_id",), ("policy_name",), ("policy_type",), ("dfu_count",), ("avg_dos",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/policy-assignments/compliance")

    data = resp.json()
    assert "A_continuous_v1" in data["by_policy"]
    entry = data["by_policy"]["A_continuous_v1"]
    for key in ("policy_name", "policy_type", "dfu_count", "below_ss_pct", "avg_ss_coverage", "avg_dos"):
        assert key in entry
