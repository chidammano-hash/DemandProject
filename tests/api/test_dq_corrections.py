"""API tests for /data-quality/corrections and /data-quality/corrections/summary endpoints."""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import patch

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


_NOW = datetime.datetime(2026, 3, 22, 10, 0, 0)

_CORRECTION_ROW = (
    1,                                    # correction_id
    "sales",                              # domain
    "fact_sales_monthly",                 # table_name
    "1401-BULK",                          # item_id
    "101",                                # loc
    datetime.date(2024, 6, 1),            # period
    "qty",                                # column_name
    50000.0,                              # old_value
    1200.0,                               # new_value
    "outliers",                           # fix_type
    "iqr_per_sku",                        # fix_strategy
    3.0,                                  # threshold
    -500.0,                               # lower_bound
    1200.0,                               # upper_bound
    _NOW,                                 # applied_at
)


@pytest.mark.asyncio
async def test_corrections_by_item():
    pool, conn, cursor = _make_pool(fetchall_return=[_CORRECTION_ROW])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/corrections", params={
                "item_id": "1401-BULK",
                "loc": "101",
            })
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    c = body["corrections"][0]
    assert c["item_id"] == "1401-BULK"
    assert c["loc"] == "101"
    assert c["column_name"] == "qty"
    assert c["old_value"] == 50000.0
    assert c["new_value"] == 1200.0
    assert c["fix_type"] == "outliers"
    assert c["fix_strategy"] == "iqr_per_sku"
    assert c["threshold"] == 3.0
    assert c["period"] == "2024-06-01"


@pytest.mark.asyncio
async def test_corrections_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/corrections")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["corrections"] == []


@pytest.mark.asyncio
async def test_corrections_filters():
    """Verify that query params translate to WHERE clauses."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/corrections", params={
                "item_id": "ITEM1",
                "loc": "LOC1",
                "table_name": "fact_sales_monthly",
                "column_name": "qty",
                "fix_type": "outliers",
            })
    assert resp.status_code == 200
    # Verify the SQL was called with the right params
    call_args = cursor.execute.call_args
    sql = call_args[0][0]
    params = call_args[0][1]
    assert "item_id = %s" in sql
    assert "loc = %s" in sql
    assert "table_name = %s" in sql
    assert "column_name = %s" in sql
    assert "fix_type = %s" in sql
    assert "ITEM1" in params
    assert "LOC1" in params


# ---------------------------------------------------------------------------
# Summary endpoint tests
# ---------------------------------------------------------------------------

_SUMMARY_ROW = (
    "1401-BULK",                          # item_id
    "101",                                # loc
    5,                                    # correction_count
    ["sales"],                            # domains (array)
    ["fact_sales_monthly"],               # tables (array)
    ["qty", "qty_shipped"],               # columns (array)
    ["outliers"],                         # fix_types (array)
    ["iqr_per_sku"],                      # strategies (array)
    datetime.date(2024, 1, 1),            # earliest_period
    datetime.date(2024, 6, 1),            # latest_period
    _NOW,                                 # latest_at
)


@pytest.mark.asyncio
async def test_corrections_summary():
    pool, conn, cursor = _make_pool()
    # summary endpoint does 2 queries: count + aggregated rows
    cursor.fetchone.return_value = (3,)
    cursor.fetchall.return_value = [_SUMMARY_ROW]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/corrections/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["skus"]) == 1
    s = body["skus"][0]
    assert s["item_id"] == "1401-BULK"
    assert s["loc"] == "101"
    assert s["correction_count"] == 5
    assert s["domains"] == ["sales"]
    assert s["columns"] == ["qty", "qty_shipped"]
    assert s["fix_types"] == ["outliers"]
    assert s["earliest_period"] == "2024-01-01"
    assert s["latest_period"] == "2024-06-01"


@pytest.mark.asyncio
async def test_corrections_summary_empty():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/corrections/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["skus"] == []


@pytest.mark.asyncio
async def test_corrections_summary_with_filters():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/corrections/summary", params={
                "domain": "sales",
                "fix_type": "outliers",
            })
    assert resp.status_code == 200
    # Verify domain filter was applied in both queries
    calls = cursor.execute.call_args_list
    for call in calls:
        sql = call[0][0]
        if "GROUP BY" in sql:
            assert "domain = %s" in sql
            assert "fix_type = %s" in sql
