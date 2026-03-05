"""API tests for IPfeature11 ABC-XYZ endpoints."""
import pytest
from unittest.mock import MagicMock, patch


def _make_pool(fetchall_return=None, fetchone_return=None):
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return or (0,)
    cursor.description = [("col",)]
    cursor.rowcount = 1
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn, cursor


@pytest.mark.asyncio
async def test_abc_xyz_matrix_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("A", "X", 120, 0.98, 14.0, 21.0),
        ("B", "Y", 80, 0.93, 28.0, 35.0),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/abc-xyz/matrix")
    assert resp.status_code == 200
    data = resp.json()
    assert "cells" in data
    assert "total_classified" in data
    assert len(data["cells"]) == 2


@pytest.mark.asyncio
async def test_abc_xyz_matrix_cells_have_segment():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("A", "X", 50, 0.98, 14.0, 21.0),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/abc-xyz/matrix")
    assert resp.status_code == 200
    cell = resp.json()["cells"][0]
    assert cell["segment"] == "AX"


@pytest.mark.asyncio
async def test_abc_xyz_summary_200():
    pool, conn, cursor = _make_pool(fetchone_return=(1000, 850, 300, 350, 200, 0.45, 0.12))
    cursor.description = [
        ("total_dfus",), ("classified_dfus",), ("x_count",), ("y_count",), ("z_count",),
        ("avg_demand_cv",), ("avg_intermittency_ratio",),
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/abc-xyz/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_dfus" in data


@pytest.mark.asyncio
async def test_abc_xyz_detail_200():
    pool, conn, cursor = _make_pool(
        fetchone_return=(500,),
        fetchall_return=[
            ("ITEM1", "GRP1", "LOC1", "A", "X", "AX", 0.20, 0.10, 14.0, 21.0, 0.98),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/abc-xyz/detail")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data


@pytest.mark.asyncio
async def test_abc_xyz_detail_filter_segment():
    pool, conn, cursor = _make_pool(
        fetchone_return=(10,),
        fetchall_return=[("ITEM1", "GRP1", "LOC1", "B", "Z", "BZ", 0.90, 0.40, 35.0, 45.0, 0.90)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/abc-xyz/detail?segment=BZ")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_abc_xyz_detail_empty():
    pool, conn, cursor = _make_pool(fetchone_return=(0,), fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/abc-xyz/detail?segment=ZZ")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
