"""API tests for IPfeature12 Supplier Performance endpoints."""
import pytest
from unittest.mock import MagicMock, patch
from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_supplier_summary_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(25, 72.0, 14.5, 0.35, 0.60, 0.20, 50000.0, 3),
    )
    cursor.description = [
        ("total_suppliers",), ("avg_reliability_score",), ("avg_lead_time_days",),
        ("avg_lt_cv",), ("avg_pct_stable",), ("avg_pct_volatile",),
        ("total_ss_value",), ("low_reliability_count",),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/supplier-performance/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_suppliers" in data
    assert "avg_reliability_score" in data


@pytest.mark.asyncio
async def test_supplier_detail_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(5,),
        fetchall_return=[
            ("SUP001", "Acme Corp", 120, 40, 14.5, 0.30, 2.1, 0.65, 0.15, 5000.0, 75000.0, 68),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/supplier-performance/detail")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert data["rows"][0]["supplier_no"] == "SUP001"


@pytest.mark.asyncio
async def test_supplier_detail_filter_min_score():
    pool, conn, cursor = _make_pool(
        fetchone_return=(2,),
        fetchall_return=[("SUP002", "TechParts", 50, 20, 10.0, 0.20, 1.5, 0.80, 0.05, 2000.0, 30000.0, 85)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/supplier-performance/detail?min_score=80")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_supplier_items_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(3,),
        fetchall_return=[
            ("ITEM1", "LOC1", 14.0, 2.0, 0.14, "stable", 12, "A", "high_volume"),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/supplier-performance/items?supplier_no=SUP001")
    assert resp.status_code == 200
    data = resp.json()
    assert "supplier_no" in data
    assert "total" in data
    assert "rows" in data


@pytest.mark.asyncio
async def test_supplier_items_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,), fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/supplier-performance/items?supplier_no=NONEXISTENT")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
