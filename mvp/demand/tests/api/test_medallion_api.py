"""API tests for medallion lineage/corrections/quarantine endpoints."""

import datetime
from unittest.mock import patch, MagicMock

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_list_batches():
    now = datetime.datetime(2026, 3, 17, 12, 0, 0)
    rows = [
        (1, "sales", "bronze", "sales_clean.csv", "abc123",
         1000, 950, 50, "completed", now, now, None),
    ]
    pool, conn, cur = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/lineage/batches")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["batches"][0]["batch_id"] == 1
    assert data["batches"][0]["domain"] == "sales"


@pytest.mark.asyncio
async def test_list_batches_filters():
    pool, conn, cur = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/lineage/batches?domain=item&status=completed")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_batch_detail():
    now = datetime.datetime(2026, 3, 17, 12, 0, 0)
    batch_row = (1, "sales", "bronze", "sales_clean.csv", "abc123",
                 1000, 950, 50, "completed", now, now, None, None)
    layer_rows = [("gold", 950), ("quarantined", 50)]

    pool, conn, cur = _make_pool()
    cur.fetchone.side_effect = [batch_row]
    cur.fetchall.return_value = layer_rows

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/lineage/batches/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["batch_id"] == 1
    assert data["layer_counts"]["gold"] == 950


@pytest.mark.asyncio
async def test_batch_detail_not_found():
    """B2: returns 404 HTTPException instead of tuple."""
    pool, conn, cur = _make_pool()
    cur.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/lineage/batches/999")
    assert resp.status_code == 404
    assert "batch not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_row_lineage():
    now = datetime.datetime(2026, 3, 17, 12, 0, 0)
    rows = [
        (1, 42, 100, 200, None, "gold", now),
    ]
    pool, conn, cur = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/lineage/row/sales/key123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["domain"] == "sales"
    assert len(data["lineage"]) == 1
    assert data["lineage"][0]["layer_reached"] == "gold"


@pytest.mark.asyncio
async def test_list_corrections():
    now = datetime.datetime(2026, 3, 17, 12, 0, 0)
    rows = [
        (1, "sales", "silver_sales", "key1", "qty", "100", "50",
         "clamp", "range", "system", now, 42),
    ]
    pool, conn, cur = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/lineage/corrections")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["corrections"][0]["fix_type"] == "clamp"


@pytest.mark.asyncio
async def test_list_corrections_filtered():
    pool, conn, cur = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                "/data-quality/lineage/corrections?domain=sales&fix_type=clamp&batch_id=1"
            )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_quarantine():
    now = datetime.datetime(2026, 3, 17, 12, 0, 0)
    rows = [
        (1, "sales", 100, 42, "null_pk", '{"column": "dmdunit"}',
         '{"dmdunit": null}', False, None, now),
    ]
    pool, conn, cur = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/quarantine")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["quarantine"][0]["rejection_reason"] == "null_pk"
    # B1: should use quarantined_at, not created_at
    assert "quarantined_at" in data["quarantine"][0]
    assert "created_at" not in data["quarantine"][0]


@pytest.mark.asyncio
async def test_quarantine_resolved_filter():
    pool, conn, cur = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/quarantine?resolved=false&domain=item")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_resolve_quarantine():
    pool, conn, cur = _make_pool()
    cur.fetchone.return_value = (1,)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/data-quality/quarantine/1/resolve")
    assert resp.status_code == 200
    data = resp.json()
    assert data["resolved"] is True


@pytest.mark.asyncio
async def test_resolve_quarantine_not_found():
    """B2: returns 404 HTTPException instead of tuple."""
    pool, conn, cur = _make_pool()
    cur.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/data-quality/quarantine/999/resolve")
    assert resp.status_code == 404
    assert "quarantine entry not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_quarantine_query_uses_quarantined_at():
    """B1: Verify the SQL uses quarantined_at column, not created_at."""
    pool, conn, cur = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/data-quality/quarantine")
    assert resp.status_code == 200
    # Verify the SQL sent to cursor uses quarantined_at
    sql = cur.execute.call_args[0][0]
    assert "quarantined_at" in sql
    assert "created_at" not in sql
