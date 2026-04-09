"""API tests for GET /inv-planning/daily-briefing endpoint.

Issue #23 — Daily Planner Summary Report.
"""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from tests.api.conftest import make_pool as _make_pool

# ---------------------------------------------------------------------------
# GET /inv-planning/daily-briefing — returns 200 with expected structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_daily_briefing_200():
    """GET /inv-planning/daily-briefing returns 200 with all sections."""
    pool, _conn, cursor = _make_pool()

    # The endpoint makes many independent fetchone/fetchall calls.
    # We use side_effect to return different values for each call.
    cursor.fetchone.side_effect = [
        # 1a. Critical exceptions: (count, total_value)
        (8, 42500.00),
        # 1b. Past-due orders: (count, total_value)
        (3, 15200.00),
        # 2a. Below safety stock: (count, recommended_value)
        (23, 1200000.00),
        # 2b. Forecast miss count
        (5,),
        # 3a. Health score: (avg_score, total_skus)
        (73.2, 13329),
        # 3c. Excess stats: (count, total_excess)
        (1794, 996593.57),
    ]
    cursor.fetchall.side_effect = [
        # 3b. ABC class health: [(abc_vol, avg_score), ...]
        [("A", 82.0), ("B", 68.5), ("C", 54.0)],
        # 4. Top 3 actions: [(source, item_id, loc, text, impact, urgency, deadline)]
        [
            ("exception", "ITEM-A1234", "LOC-01", "Resolve stockout for ITEM-A1234 @ LOC-01", 95000.0, 0.95, "Today"),
            ("planned_order", "ITEM-B5678", "LOC-02", "Approve order for ITEM-B5678 @ LOC-02", 12000.0, 0.9, "Today"),
            ("exception", "ITEM-C9012", "LOC-03", "Resolve low_stock for ITEM-C9012 @ LOC-03", 8500.0, 0.75, "This week"),
        ],
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/daily-briefing")

    assert resp.status_code == 200
    data = resp.json()

    # Top-level keys
    assert "date" in data
    assert "urgent" in data
    assert "this_week" in data
    assert "portfolio" in data
    assert "actions" in data
    assert "stats" in data

    # Urgent section
    assert data["urgent"]["label"] == "Act within 24 hours"
    assert len(data["urgent"]["items"]) == 2

    # This week section
    assert data["this_week"]["label"] == "Review this week"
    assert len(data["this_week"]["items"]) == 2

    # Portfolio section
    assert data["portfolio"]["label"] == "Portfolio Health"
    assert len(data["portfolio"]["items"]) == 4  # 1 overall + 3 ABC classes

    # Actions section
    assert data["actions"]["label"] == "Top 3 Recommended Actions"
    assert len(data["actions"]["items"]) == 3
    assert data["actions"]["items"][0]["priority"] == 1

    # Stats
    stats = data["stats"]
    assert stats["total_skus"] == 13329
    assert stats["below_ss_count"] == 23
    assert stats["excess_count"] == 1794
    assert stats["total_excess_value"] == 996593.57
    assert stats["total_stockout_risk_value"] == 42500.00
    assert stats["avg_health_score"] == 73.2


@pytest.mark.asyncio
async def test_daily_briefing_empty_db():
    """GET /inv-planning/daily-briefing returns graceful empty when DB has no data."""
    pool, _conn, cursor = _make_pool()

    # All queries return empty/zero results
    cursor.fetchone.side_effect = [
        (0, 0),   # 1a. No critical exceptions
        (0, 0),   # 1b. No past-due orders
        (0, 0),   # 2a. No below-SS items
        (0,),     # 2b. No forecast misses
        (None, 0),  # 3a. No health scores
        (0, 0),   # 3c. No excess
    ]
    cursor.fetchall.side_effect = [
        [],  # 3b. No ABC health data
        [],  # 4. No actions
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/daily-briefing")

    assert resp.status_code == 200
    data = resp.json()

    # All sections should be empty but present
    assert data["urgent"]["items"] == []
    assert data["this_week"]["items"] == []
    assert data["portfolio"]["items"] == []
    assert data["actions"]["items"] == []

    # Stats should be zeroed out
    stats = data["stats"]
    assert stats["total_skus"] == 0
    assert stats["below_ss_count"] == 0
    assert stats["avg_health_score"] is None


@pytest.mark.asyncio
async def test_daily_briefing_partial_failure():
    """Endpoint degrades gracefully when some DB queries fail."""
    import psycopg
    pool, _conn, cursor = _make_pool()

    # Simulate partial failures by making execute raise on specific calls.
    # Calls 2 (past-due orders) and 5 (forecast miss) will fail.
    call_count = {"n": 0}

    def execute_with_failures(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] in (2, 5):
            raise psycopg.Error("simulated failure")
        # Otherwise do nothing (mock default)

    cursor.execute.side_effect = execute_with_failures

    # fetchone/fetchall will be called only for queries that don't raise
    # Successful queries: 1a(critical exc), 2a(below-SS), 3a(health), 3c(excess)
    cursor.fetchone.side_effect = [
        (5, 30000.00),    # 1a. Critical exceptions succeeds
        # 1b skipped (execute raises)
        (10, 500000.00),  # 2a. Below-SS succeeds
        # 2b skipped (execute raises)
        (70.0, 5000),     # 3a. Health score succeeds
        (100, 50000.00),  # 3c. Excess succeeds
    ]
    cursor.fetchall.side_effect = [
        [("A", 80.0)],  # 3b. ABC health partial
        [],              # 4. No actions
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/daily-briefing")

    assert resp.status_code == 200
    data = resp.json()
    # Should still have the date and all section keys even with partial failures
    assert "date" in data
    assert "urgent" in data
    assert "this_week" in data
    assert "portfolio" in data
    assert "actions" in data
    assert "stats" in data
