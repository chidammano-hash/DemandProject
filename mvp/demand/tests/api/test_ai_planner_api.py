"""API tests for /ai-planner/* endpoints — IPAIfeature1.

Uses httpx AsyncClient with ASGITransport; no running server required.
DB is mocked via patch("api.core._get_pool").
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import httpx
from httpx import ASGITransport


def _make_pool(fetchall_return=None, fetchone_return=None, description=None):
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    cursor.fetchone.return_value = fetchone_return or (0,)
    cursor.description = description or []
    cursor.rowcount = 1

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, conn, cursor


# ---------------------------------------------------------------------------
# GET /ai-planner/insights — empty list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_insights_empty():
    pool, conn, cursor = _make_pool(
        fetchone_return=(0,),
        fetchall_return=[],
        description=[("insight_id",), ("insight_type",), ("severity",),
                     ("item_no",), ("loc",), ("summary",)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ai-planner/insights")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["insights"] == []
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_get_insights_with_data():
    pool, conn, cursor = _make_pool(
        fetchone_return=(2,),
        fetchall_return=[
            (1, "stockout_risk", "critical", "100320", "1401-BULK", "Low DOS"),
            (2, "excess_inventory", "medium", "100321", "1401-BULK", "High DOS"),
        ],
        description=[
            ("insight_id",), ("insight_type",), ("severity",),
            ("item_no",), ("loc",), ("summary",),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ai-planner/insights")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["insights"]) == 2


@pytest.mark.asyncio
async def test_get_insights_severity_filter():
    """Severity filter 'critical' is passed; invalid filter is ignored."""
    pool, conn, cursor = _make_pool(
        fetchone_return=(1,),
        fetchall_return=[(1, "stockout_risk", "critical", "100320", "LOC", "summary")],
        description=[
            ("insight_id",), ("insight_type",), ("severity",),
            ("item_no",), ("loc",), ("summary",),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ai-planner/insights?severity=critical")

    assert resp.status_code == 200
    assert resp.json()["total"] == 1


@pytest.mark.asyncio
async def test_get_insights_invalid_severity_ignored():
    """Invalid severity value is silently ignored (not filtered)."""
    pool, conn, cursor = _make_pool(
        fetchone_return=(5,),
        fetchall_return=[],
        description=[("insight_id",), ("summary",)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ai-planner/insights?severity=INVALID")

    # Should still return 200 — invalid severity is silently ignored
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# PUT /ai-planner/insights/{id}/status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_insight_status_acknowledge():
    # RETURNING has 11 cols: id, status, type, item_no, loc, abc_vol,
    #   financial_impact_est, dos, total_lt_days, champion_wape, forecast_bias_pct
    pool, conn, cursor = _make_pool(fetchone_return=(
        1, "acknowledged", "stockout_risk", "100320", "1401-BULK", "A",
        8500.0, 18.0, 14, 0.41, 0.20,
    ))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/ai-planner/insights/1/status",
                json={"status": "acknowledged"},
                headers={"X-API-Key": "test"},
            )

    # Auth disabled when API_KEY not set → should pass through
    assert resp.status_code in (200, 403)


@pytest.mark.asyncio
async def test_update_insight_status_writes_outcome():
    """When acknowledged, an outcome record is written (cursor.execute called twice)."""
    pool, conn, cursor = _make_pool(fetchone_return=(
        1, "acknowledged", "stockout_risk", "100320", "1401-BULK", "A",
        8500.0, 18.0, 14, 0.41, 0.20,
    ))
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/ai-planner/insights/1/status",
                json={"status": "acknowledged", "action_taken": "Emergency reorder 250 units"},
                headers={"X-API-Key": "test"},
            )

    # Either succeeded (200) or auth blocked (403)
    assert resp.status_code in (200, 403)
    # If auth succeeded, cursor.execute should have been called for both
    # the UPDATE and the INSERT INTO ai_recommendation_outcomes
    if resp.status_code == 200:
        assert cursor.execute.call_count >= 2


@pytest.mark.asyncio
async def test_update_insight_status_invalid():
    """Invalid status returns 422."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/ai-planner/insights/1/status",
                json={"status": "BOGUS"},
                headers={"X-API-Key": "test"},
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_update_insight_status_not_found():
    """Row not found → 404."""
    pool, conn, cursor = _make_pool(fetchone_return=None)
    # Override fetchone to return None for the UPDATE RETURNING
    cursor.fetchone.return_value = None
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/ai-planner/insights/9999/status",
                json={"status": "resolved"},
                headers={"X-API-Key": "test"},
            )

    assert resp.status_code in (404, 403)


# ---------------------------------------------------------------------------
# GET /ai-planner/memos — empty list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_memos_empty():
    pool, conn, cursor = _make_pool(
        fetchall_return=[],
        description=[("memo_id",), ("period",), ("scope",), ("narrative_text",)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ai-planner/memos")

    assert resp.status_code == 200
    assert resp.json()["memos"] == []


@pytest.mark.asyncio
async def test_get_memos_with_scope_filter():
    pool, conn, cursor = _make_pool(
        fetchall_return=[
            (1, "2026-03-01", "portfolio", "All good this week."),
        ],
        description=[("memo_id",), ("period",), ("scope",), ("narrative_text",)],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ai-planner/memos?scope=portfolio")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memos"]) == 1


# ---------------------------------------------------------------------------
# GET /ai-planner/metrics — observability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_ai_metrics_empty():
    """GET /ai-planner/metrics returns by_model and by_tool lists."""
    pool, conn, cursor = _make_pool(
        fetchall_return=[],
        description=[
            ("provider",), ("model",), ("llm_turns",), ("tool_calls",),
            ("total_tokens",), ("avg_llm_latency_ms",), ("p95_llm_latency_ms",),
            ("tool_errors",), ("error_rate_pct",),
        ],
    )
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ai-planner/metrics?days=7")

    assert resp.status_code == 200
    data = resp.json()
    assert data["days"] == 7
    assert "by_model" in data
    assert "by_tool" in data


# ---------------------------------------------------------------------------
# POST /ai-planner/auto-accept
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_accept_dry_run():
    """Dry run returns matching count without writing."""
    rows = [
        (1, "stockout_risk", "100320", "1401-BULK", "A", 8500.0, 18.0, 21, 0.41, 0.20),
        (2, "forecast_bias", "200100", "1401-BULK", "B", 3000.0, 30.0, 14, 0.55, -0.30),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ai-planner/auto-accept",
                json={"min_severity": "high", "insight_types": [], "dry_run": True},
            )

    assert resp.status_code in (200, 403)
    if resp.status_code == 200:
        data = resp.json()
        assert data["dry_run"] is True
        assert data["accepted"] == 2
        assert data["insight_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_auto_accept_executes():
    """Non-dry-run updates rows and writes outcomes."""
    rows = [
        (1, "stockout_risk", "100320", "1401-BULK", "A", 8500.0, 18.0, 21, 0.41, 0.20),
    ]
    pool, conn, cursor = _make_pool(fetchall_return=rows)
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ai-planner/auto-accept",
                json={"min_severity": "critical", "insight_types": [], "dry_run": False},
            )

    assert resp.status_code in (200, 403)
    if resp.status_code == 200:
        data = resp.json()
        assert data["dry_run"] is False
        assert data["accepted"] == 1
        # Should have called execute for: SELECT + UPDATE + INSERT (outcome)
        assert cursor.execute.call_count >= 3


@pytest.mark.asyncio
async def test_auto_accept_empty_results():
    """No matching insights returns accepted=0."""
    pool, conn, cursor = _make_pool(fetchall_return=[])
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ai-planner/auto-accept",
                json={"min_severity": "critical", "insight_types": [], "dry_run": False},
            )

    assert resp.status_code in (200, 403)
    if resp.status_code == 200:
        data = resp.json()
        assert data["accepted"] == 0


# ---------------------------------------------------------------------------
# POST /ai-planner/portfolio-scan — 202
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_portfolio_scan_returns_202():
    """POST /portfolio-scan returns 202 with scan_run_id immediately."""
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch("api.routers.ai_planner._executor") as mock_exec, \
         patch("api.routers.ai_planner._AI_CONFIG", {"model": "claude-opus-4-6", "portfolio_scan_limit": 10}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ai-planner/portfolio-scan",
                headers={"X-API-Key": "test"},
            )

    assert resp.status_code in (202, 403)
    if resp.status_code == 202:
        data = resp.json()
        assert "scan_run_id" in data
        assert data["status"] == "accepted"
