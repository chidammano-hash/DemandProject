"""API tests for the Unified Action Feed (GET /inv-planning/action-feed).

Covers finding F1.1:
  * the feed surfaces open exceptions (regression guard for the
    ``created_at`` → ``exception_date`` column fix), and
  * each source query is isolated in its own SAVEPOINT so one failing
    source can't zero out the whole feed.
"""
from unittest.mock import patch

import psycopg
import pytest

from tests.api.conftest import make_async_pool as _make_async_pool


@pytest.mark.asyncio
async def test_action_feed_returns_open_exceptions():
    """Source 1 (open exceptions) populates the feed and summary counts."""
    # 7-column rows matching the SELECT: source, item_id, loc, action_type,
    # urgency_score, financial_impact, action_label, exception_date.
    # Source 1 now selects the raw exception_type as action_label; the router
    # humanizes it into the title (see test below).
    exc_rows = [
        ("exception", "627099", "1401-BULK", "stockout", 0.95, 571.98,
         "stockout", "2026-04-02"),
        ("exception", "664631", "1401-BULK", "below_ss", 0.95, 292.39,
         "below_ss", "2026-04-02"),
    ]
    # Source 1 returns exceptions; Sources 2 & 3 return nothing.
    # The full-population aggregate (fetchone) matches the 2 returned rows here.
    pool, _conn, cursor = _make_async_pool(fetchall_returns=[exc_rows, [], []])
    cursor.fetchone.side_effect = [(2, 2, 0, 864.37)]

    with patch("api.core._get_async_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/action-feed?limit=20")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["actions"]) == 2
    assert data["summary"]["total"] == 2
    assert data["summary"]["critical"] == 2
    assert data["summary"]["financial_at_risk"] == pytest.approx(864.37)


@pytest.mark.asyncio
async def test_action_feed_humanizes_exception_enum_in_title_and_detail():
    """U2.11: raw DB enum (e.g. ``below_ss``) must never leak into the UI.

    Both the action title and the detail subtitle render the friendly label
    "Below Safety Stock" rather than the raw enum or its naive title-case
    ("Below Ss").
    """
    exc_rows = [
        ("exception", "664631", "1401-BULK", "below_ss", 0.95, 292.39,
         "below_ss", "2026-04-02"),
    ]
    pool, _conn, _cursor = _make_async_pool(fetchall_returns=[exc_rows, [], []])

    with patch("api.core._get_async_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/action-feed?limit=20")

    assert resp.status_code == 200
    action = resp.json()["actions"][0]
    assert action["title"] == "Resolve Below Safety Stock"
    assert "Below Safety Stock" in action["detail"]
    # No raw enum / naive title-case leaks.
    assert "below_ss" not in action["title"]
    assert "Below Ss" not in action["detail"]


@pytest.mark.asyncio
async def test_action_feed_source_isolation():
    """F1.1: if Source 1 raises (e.g. a renamed column), Sources 2 & 3 still
    run and populate the feed — one failure must not blank the whole feed."""
    # Planned-order row (Source 2): source, item_id, loc, action_type,
    # urgency_score, financial_impact, action_label, created_at.
    order_rows = [
        ("planned_order", "900001", "1401-BULK", "approve_order", 0.9, 1000.0,
         "Approve order: 50 units", "2026-04-02"),
    ]

    pool, _conn, cursor = _make_async_pool()
    # execute: Source 1 raises; Sources 2 & 3 succeed; aggregate also runs.
    cursor.execute.side_effect = [
        psycopg.errors.UndefinedColumn('column "created_at" does not exist'),
        None,
        None,
        None,  # full-population aggregate query
    ]
    # fetchall is only reached by Sources 2 & 3.
    cursor.fetchall.side_effect = [order_rows, []]
    # Aggregate fetchone: 1 total, 1 critical, 0 high, $1000.
    cursor.fetchone.side_effect = [(1, 1, 0, 1000.0)]

    with patch("api.core._get_async_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/action-feed?limit=20")

    assert resp.status_code == 200
    data = resp.json()
    # Source 1 failed, but Source 2's planned order still made it through.
    assert len(data["actions"]) == 1
    assert data["actions"][0]["source"] == "planned_order"
    assert data["summary"]["total"] == 1


@pytest.mark.asyncio
async def test_action_feed_summary_reflects_full_population_not_display_page():
    """U9.1: the summary KPIs (total / critical / high / financial_at_risk) must
    be computed over the FULL candidate population, NOT just the truncated
    display page. A planner triaging on the headline numbers must see the real
    portfolio exposure, not the sum of the top-N rows shown.

    Here the display page returns only 2 rows (limit-sliced), but the dedicated
    full-population aggregate query reports 2,465 critical / 4,180 total /
    $9,338.30 — those are what the summary must reflect.
    """
    # Display page: only 2 rows survive the slice.
    exc_rows = [
        ("exception", "627099", "1401-BULK", "stockout", 0.95, 571.98,
         "stockout", "2026-04-02"),
        ("exception", "664631", "1401-BULK", "below_ss", 0.95, 292.39,
         "below_ss", "2026-04-02"),
    ]
    pool, _conn, cursor = _make_async_pool()
    # Three source detail queries (fetchall) then the aggregate (fetchone).
    cursor.fetchall.side_effect = [exc_rows, [], []]
    # Full-population aggregate: (total, critical, high, financial_at_risk)
    cursor.fetchone.side_effect = [(4180, 2465, 1715, 9338.30)]

    with patch("api.core._get_async_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/action-feed?limit=20")

    assert resp.status_code == 200
    data = resp.json()
    # Display page is still truncated...
    assert len(data["actions"]) == 2
    # ...but the summary reports the FULL population, not len(actions).
    assert data["summary"]["total"] == 4180
    assert data["summary"]["critical"] == 2465
    assert data["summary"]["high"] == 1715
    assert data["summary"]["financial_at_risk"] == pytest.approx(9338.30)
    # And the response declares the list is a truncated top-N.
    assert data["summary"]["displayed"] == 2


@pytest.mark.asyncio
async def test_action_feed_summary_falls_back_to_page_when_aggregate_fails():
    """If the full-population aggregate query fails (schema drift / missing MV),
    the summary degrades to counting the returned rows rather than 500-ing."""
    exc_rows = [
        ("exception", "627099", "1401-BULK", "stockout", 0.95, 571.98,
         "stockout", "2026-04-02"),
    ]
    pool, _conn, cursor = _make_async_pool()
    cursor.fetchall.side_effect = [exc_rows, [], []]
    # Aggregate query raises -> fall back to page-level counts.
    cursor.fetchone.side_effect = psycopg.errors.UndefinedColumn("boom")

    with patch("api.core._get_async_pool", return_value=pool):
        from httpx import ASGITransport, AsyncClient

        from api.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/inv-planning/action-feed?limit=20")

    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["total"] == 1
    assert data["summary"]["critical"] == 1
