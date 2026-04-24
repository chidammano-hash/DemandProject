"""API tests for exception lifecycle + MTTR endpoints (Gen-4 Roadmap 1.9)."""
from __future__ import annotations

import datetime

import httpx
import pytest
from httpx import ASGITransport
from unittest.mock import patch

from tests.api.conftest import make_pool as _make_pool


@pytest.mark.asyncio
async def test_exception_lifecycle_200():
    rows = [
        (1, None, "open",          datetime.datetime(2026, 4, 1, 10, 0), "system", None),
        (2, "open", "acknowledged", datetime.datetime(2026, 4, 1, 11, 0), "alice",  "investigating"),
        (3, "acknowledged", "resolved", datetime.datetime(2026, 4, 1, 16, 0), "alice",  "closed, ordered expediter"),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions/abc-123/lifecycle")

    assert resp.status_code == 200
    data = resp.json()
    assert data["exception_id"] == "abc-123"
    assert len(data["transitions"]) == 3
    # First transition: open, no from_state
    assert data["transitions"][0]["from_state"] is None
    assert data["transitions"][0]["to_state"] == "open"
    assert data["transitions"][-1]["to_state"] == "resolved"


@pytest.mark.asyncio
async def test_exception_lifecycle_table_missing_returns_empty():
    """If the table doesn't exist yet (older env), endpoint still returns 200."""
    import psycopg

    pool, conn, cursor = _make_pool()
    cursor.execute.side_effect = psycopg.Error("relation does not exist")

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions/abc-123/lifecycle")

    assert resp.status_code == 200
    assert resp.json()["transitions"] == []


@pytest.mark.asyncio
async def test_exception_mttr_summary_200():
    rows = [
        ("stockout_risk", "critical", 12, 4.5,  3.2,  9.0),
        ("stockout_risk", "high",     20, 12.5, 10.0, 30.0),
        ("excess_risk",   "medium",    5, 72.0, 60.0, 120.0),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions/mttr")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["rows"]) == 3
    assert data["rows"][0]["exception_type"] == "stockout_risk"
    assert data["rows"][0]["mttr_hours_avg"] == 4.5


@pytest.mark.asyncio
async def test_exception_mttr_filter_severity():
    pool, conn, cursor = _make_pool(fetchall_return=[("stockout_risk", "critical", 12, 4.5, 3.2, 9.0)])

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/exceptions/mttr?severity=critical")

    assert resp.status_code == 200
    # Must have included severity param in SQL
    call_args_list = cursor.execute.call_args_list
    assert any("critical" in (args[0][1] or []) for args in call_args_list if len(args[0]) > 1)
