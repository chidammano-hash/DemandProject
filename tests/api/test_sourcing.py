"""Tests for sourcing API endpoints."""
import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_sourcing_rows_empty():
    pool, conn, cursor = _make_pool(
        fetchone_return=(0,),
        fetchall_return=[],
    )
    cursor.description = [("sourcing_ck",), ("site_id",), ("item_id",), ("loc",),
                          ("source_cd",), ("transit_mode",), ("supplier_id",), ("plant_id",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sourcing/rows")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["rows"] == []


@pytest.mark.asyncio
async def test_sourcing_rows_with_data():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = [
        (1, "56", "1040", "8701-1000", "104522-706901", "0-43500-WEIGHT", "104522", "706901"),
        (2, "56", "1041", "8701-1000", "104522-706901", "0-43500-WEIGHT", "104522", "706901"),
    ]
    cursor.description = [("sourcing_ck",), ("site_id",), ("item_id",), ("loc",),
                          ("source_cd",), ("transit_mode",), ("supplier_id",), ("plant_id",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sourcing/rows?item=1040")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["rows"]) == 2
    assert data["rows"][0]["item_id"] == "1040"


@pytest.mark.asyncio
async def test_sourcing_search():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = [
        (1, "56", "1040", "8701-1000", "104522-706901", "WEIGHT", "104522", "706901"),
    ]
    cursor.description = [("sourcing_ck",), ("site_id",), ("item_id",), ("loc",),
                          ("source_cd",), ("transit_mode",), ("supplier_id",), ("plant_id",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sourcing/search?q=1040")
    assert resp.status_code == 200
    assert len(resp.json()["rows"]) == 1


@pytest.mark.asyncio
async def test_sourcing_by_item():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    cursor.description = [("sourcing_ck",), ("site_id",), ("item_id",), ("loc",),
                          ("source_cd",), ("transit_mode",), ("supplier_id",), ("plant_id",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sourcing/by-item/1040")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_sourcing_by_supplier():
    pool, conn, cursor = _make_pool()
    cursor.fetchall.return_value = []
    cursor.description = [("sourcing_ck",), ("site_id",), ("item_id",), ("loc",),
                          ("source_cd",), ("transit_mode",), ("supplier_id",), ("plant_id",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sourcing/by-supplier/104522")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_sourcing_network():
    pool, conn, cursor = _make_pool()
    # network endpoint makes 5 fetchone + 1 fetchall calls
    cursor.fetchone.side_effect = [(1000,), (50,), (800,), (200,)]
    cursor.fetchall.return_value = [("WEIGHT", 500), ("AIR", 300)]
    cursor.description = [("transit_mode",), ("cnt",)]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/sourcing/network")
    assert resp.status_code == 200
    data = resp.json()
    assert "supplier_count" in data
    assert "single_source_count" in data
