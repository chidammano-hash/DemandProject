"""API tests for data-quality batch listing endpoints."""

import datetime
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool


_NOW = datetime.datetime(2026, 3, 21, 10, 0, 0)
_NOW_DONE = datetime.datetime(2026, 3, 21, 10, 1, 0)

_BATCH_ROW = (
    1, "sales", "sales_clean.csv", "abc123", "completed",
    1000, 995, _NOW, _NOW_DONE, None,
)


@pytest.mark.asyncio
async def test_list_batches():
    pool, conn, cur = _make_pool(fetchall_return=[_BATCH_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/data-quality/batches")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    batch = data["batches"][0]
    assert batch["batch_id"] == 1
    assert batch["domain"] == "sales"
    assert batch["source_file"] == "sales_clean.csv"
    assert batch["row_count_in"] == 1000
    assert batch["row_count_out"] == 995
    assert batch["error_message"] is None


@pytest.mark.asyncio
async def test_list_batches_with_filters():
    pool, conn, cur = _make_pool(fetchall_return=[_BATCH_ROW])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/data-quality/batches?domain=sales&status=completed"
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1

    # Verify the SQL included both filter params
    sql = cur.execute.call_args[0][0]
    assert "domain = %s" in sql
    assert "status = %s" in sql
    params = cur.execute.call_args[0][1]
    assert "sales" in params
    assert "completed" in params


@pytest.mark.asyncio
async def test_batch_detail():
    pool, conn, cur = _make_pool()
    cur.fetchone.return_value = _BATCH_ROW

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/data-quality/batches/1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["batch_id"] == 1
    assert data["domain"] == "sales"
    assert data["status"] == "completed"
    assert data["started_at"] == _NOW.isoformat()
    assert data["completed_at"] == _NOW_DONE.isoformat()


@pytest.mark.asyncio
async def test_batch_detail_not_found():
    pool, conn, cur = _make_pool()
    cur.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/data-quality/batches/999")

    assert resp.status_code == 404
    assert "batch not found" in resp.json()["detail"]
