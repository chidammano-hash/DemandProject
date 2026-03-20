"""API tests for data quality endpoints (Spec 08-01).

Tests all 5 data quality REST endpoints using httpx AsyncClient with
ASGITransport -- no running server needed.
"""
from __future__ import annotations

import datetime
import pytest
from unittest.mock import patch, MagicMock

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


_NOW = datetime.datetime(2026, 3, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# GET /data-quality/dashboard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_dashboard_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        ("sales", 8, 1, 1, 10),
        ("forecast", 5, 3, 2, 10),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "domains" in data
    assert len(data["domains"]) == 2
    assert data["domains"][0]["domain"] == "sales"
    assert data["domains"][0]["score"] == 80.0  # 8/10 * 100
    assert data["domains"][0]["passed"] == 8
    assert data["domains"][0]["failed"] == 1
    assert data["domains"][0]["warnings"] == 1


@pytest.mark.asyncio
async def test_dq_dashboard_empty():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/dashboard")
    assert resp.status_code == 200
    assert resp.json()["domains"] == []


# ---------------------------------------------------------------------------
# GET /data-quality/checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_checks_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "freshness_sales", "freshness", "sales", "fact_sales_monthly",
         "critical", True, "pass", 1.0, _NOW),
        (2, "completeness_forecast", "completeness", "forecast",
         "fact_external_forecast_monthly", "warning", True, "warn", 0.95, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/checks")
    assert resp.status_code == 200
    checks = resp.json()["checks"]
    assert len(checks) == 2
    assert checks[0]["check_name"] == "freshness_sales"
    assert checks[0]["check_type"] == "freshness"
    assert checks[0]["severity"] == "critical"
    assert checks[0]["enabled"] is True
    assert checks[0]["last_status"] == "pass"
    assert checks[0]["last_value"] == 1.0
    assert checks[0]["last_run"] is not None


@pytest.mark.asyncio
async def test_dq_checks_null_metric():
    """Null metric_value and run_ts should be returned as None."""
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "orphan_check", "referential", "sales", "fact_sales_monthly",
         "warning", False, None, None, None),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/checks")
    assert resp.status_code == 200
    check = resp.json()["checks"][0]
    assert check["last_value"] is None
    assert check["last_run"] is None


# ---------------------------------------------------------------------------
# GET /data-quality/history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_history_200():
    pool, conn, cursor = _make_pool(fetchall_return=[
        (1, "freshness_sales", "sales", "fact_sales_monthly", "critical",
         "pass", 1.0, None, _NOW),
    ])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/history?days=7")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["check_name"] == "freshness_sales"
    assert entries[0]["status"] == "pass"


@pytest.mark.asyncio
async def test_dq_history_with_domain_filter():
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/history?domain=sales&days=3&limit=10")
    assert resp.status_code == 200
    assert resp.json()["entries"] == []
    # Verify domain filter was included in the SQL params
    cursor.execute.assert_called_once()
    call_args = cursor.execute.call_args
    assert "sales" in call_args[0][1]  # domain param in the params list


@pytest.mark.asyncio
async def test_dq_history_invalid_days():
    """days < 1 should fail validation."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/history?days=0")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /data-quality/freshness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_freshness_200():
    pool, conn, cursor = _make_pool(fetchone_return=(_NOW,))
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (_NOW,)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("api.core._get_pool", return_value=pool), \
         patch("psycopg.connect", return_value=mock_conn):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/freshness")
    assert resp.status_code == 200
    tables = resp.json()["tables"]
    assert len(tables) == 6  # 6 tables checked
    assert tables[0]["table"] == "dim_item"
    assert tables[0]["last_load"] is not None


@pytest.mark.asyncio
async def test_dq_freshness_no_data():
    """Tables with no rows return last_load=None."""
    pool, conn, cursor = _make_pool(fetchone_return=(None,))
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (None,)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("api.core._get_pool", return_value=pool), \
         patch("psycopg.connect", return_value=mock_conn):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/freshness")
    assert resp.status_code == 200
    for t in resp.json()["tables"]:
        assert t["last_load"] is None


# ---------------------------------------------------------------------------
# POST /data-quality/run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_run_200():
    """POST /data-quality/run triggers ad-hoc check run (manager+ role)."""
    pool, conn, cursor = _make_pool()
    mock_engine = MagicMock()
    mock_engine.run_all_checks.return_value = [
        {"check_name": "freshness_sales", "status": "pass"}
    ]
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.data_quality.DQEngine", return_value=mock_engine, create=True), \
         patch.dict("sys.modules", {"common.dq_engine": MagicMock(DQEngine=lambda: mock_engine)}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/run")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["total"] == 1
    assert data["triggered"] == 1
    assert data["message"] == "ok"


@pytest.mark.asyncio
async def test_dq_run_with_domain():
    """POST /data-quality/run?domain=sales passes domain to engine."""
    pool, conn, cursor = _make_pool()
    mock_engine = MagicMock()
    mock_engine.run_all_checks.return_value = []
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"common.dq_engine": MagicMock(DQEngine=lambda: mock_engine)}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/run?domain=sales")
    assert resp.status_code == 200
    mock_engine.run_all_checks.assert_called_once_with(domain="sales")


# ---------------------------------------------------------------------------
# GET /data-quality/fix/preview
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_fix_preview_200():
    """Preview returns indexed fix items."""
    pool, conn, cursor = _make_pool()
    mock_items = [
        {"id": 0, "fix_type": "range", "description": "Clamp t.col to [0, 100]",
         "affected_rows": 500, "recommendation": None, "status": "pending"},
        {"id": 1, "fix_type": "completeness", "description": "Impute t.col NULLs",
         "affected_rows": 100, "recommendation": None, "status": "pending"},
    ]
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"scripts.fix_dq_issues": MagicMock(
             preview_all_fixes=MagicMock(return_value=mock_items),
             FIX_REGISTRY={"range": None, "lead_time": None, "completeness": None,
                           "orphans": None, "outliers": None},
         )}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/fix/preview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2
    assert data["items"][0]["id"] == 0
    assert data["items"][0]["fix_type"] == "range"
    assert data["items"][0]["affected_rows"] == 500


@pytest.mark.asyncio
async def test_dq_fix_preview_invalid_type():
    """Preview with unknown fix_type returns error."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"scripts.fix_dq_issues": MagicMock(
             FIX_REGISTRY={"range": None, "lead_time": None, "completeness": None,
                           "orphans": None, "outliers": None},
         )}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/fix/preview?fix_type=bogus")
    assert resp.status_code == 200
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_dq_fix_preview_empty():
    """Preview with no fixable issues returns empty list."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"scripts.fix_dq_issues": MagicMock(
             preview_all_fixes=MagicMock(return_value=[]),
             FIX_REGISTRY={"range": None},
         )}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/data-quality/fix/preview")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
    assert resp.json()["items"] == []


# ---------------------------------------------------------------------------
# POST /data-quality/fix/apply
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dq_fix_apply_200():
    """Apply selected fixes returns applied results."""
    pool, conn, cursor = _make_pool()
    mock_result = {
        "applied": [
            {"id": 0, "fix_type": "range", "description": "Clamp t.col",
             "affected_rows": 500, "recommendation": None, "status": "applied",
             "rows_fixed": 500},
        ],
        "skipped": [
            {"id": 1, "fix_type": "completeness", "description": "Impute",
             "affected_rows": 100, "recommendation": None, "status": "skipped"},
        ],
        "total_applied": 1,
        "total_skipped": 1,
        "total_rows_fixed": 500,
    }
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("sys.modules", {"scripts.fix_dq_issues": MagicMock(
             apply_selected_fixes=MagicMock(return_value=mock_result),
         )}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/fix/apply",
                                     json={"fix_ids": [0]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_applied"] == 1
    assert data["total_rows_fixed"] == 500
    assert len(data["applied"]) == 1
    assert data["applied"][0]["status"] == "applied"


@pytest.mark.asyncio
async def test_dq_fix_apply_empty_ids():
    """Apply with empty fix_ids returns error."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/fix/apply",
                                     json={"fix_ids": []})
    assert resp.status_code == 200
    assert resp.json()["total_applied"] == 0
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_dq_fix_apply_missing_body():
    """Apply without body returns 422."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/data-quality/fix/apply")
    assert resp.status_code == 422
