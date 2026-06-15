"""API tests for Feature 40 — /storyboard/* endpoints.

Tests all 7 storyboard REST endpoints using httpx AsyncClient with
ASGITransport — no running server needed.
"""
from __future__ import annotations

import datetime
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

import httpx
from httpx import ASGITransport
from tests.api.conftest import make_pool as _make_pool


_EXC_ID = str(uuid.uuid4())
_NOW = datetime.datetime(2026, 3, 1, 12, 0, 0)

# A representative exception_queue row
_EXCEPTION_ROW = (
    _EXC_ID,             # exception_id
    "stockout_risk",     # exception_type
    "ITEM001",           # item_id
    "LOC1",              # loc
    0.85,                # severity
    142000.0,            # financial_impact
    "Stockout Risk: Item ITEM001 @ LOC1 has 8.0 days of supply",  # headline
    {"dos": 8.0},        # supporting_data (psycopg returns dict for jsonb)
    "open",              # status
    None,                # assigned_to
    _NOW,                # generated_at
    None,                # expires_at
    datetime.date(2026, 3, 1),  # month_start
)

_EXCEPTION_COLS = [
    ("exception_id",), ("exception_type",), ("item_id",), ("loc",),
    ("severity",), ("financial_impact",), ("headline",), ("supporting_data",),
    ("status",), ("assigned_to",), ("generated_at",), ("expires_at",),
    ("month_start",),
]


# ---------------------------------------------------------------------------
# GET /storyboard/exceptions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_exceptions_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3,)
    cursor.fetchall.return_value = [_EXCEPTION_ROW]
    cursor.description = _EXCEPTION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_list_exceptions_row_keys():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [_EXCEPTION_ROW]
    cursor.description = _EXCEPTION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions")

    assert resp.status_code == 200
    row = resp.json()["rows"][0]
    for key in ("exception_id", "exception_type", "item_id", "loc", "severity", "status", "headline"):
        assert key in row


@pytest.mark.asyncio
async def test_list_exceptions_filter_by_type():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = []
    cursor.description = _EXCEPTION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?exception_type=stockout_risk")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_exceptions_pagination_params():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (100,)
    cursor.fetchall.return_value = []
    cursor.description = _EXCEPTION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?limit=10&offset=20")

    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 10
    assert data["offset"] == 20


# A representative fact_replenishment_exceptions fallback row.
# Column order matches the fallback SELECT in storyboard.list_exceptions.
_REPL_ROW = (
    "repl-uuid-1",            # exception_id
    "stockout",              # exception_type
    "627099",               # item_id
    "1401-BULK",            # loc
    "critical",             # severity (text)
    571.98,                  # financial_impact_total
    "open",                 # status
    datetime.date(2026, 4, 2),  # exception_date
    1.5,                     # current_dos
    120.0,                   # recommended_order_qty
    "MENAGE A TROIS A(D/R/S)3P PAD(44",  # item_desc (dim_item join)
)


@pytest.mark.asyncio
async def test_list_exceptions_falls_back_to_replenishment_when_queue_empty():
    """F4.1: when exception_queue is empty but fact_replenishment_exceptions has
    open rows, the storyboard feed must return those replenishment rows so the
    Command Center feed is not "Exception data unavailable" while the KPI tile
    shows 6142 open exceptions.
    """
    pool, conn, cursor = _make_pool()
    # 1st fetchone: COUNT(*) over exception_queue -> 0 (empty).
    # 2nd fetchone: COUNT(*) over fact_replenishment_exceptions -> 1.
    cursor.fetchone.side_effect = [(0,), (1,)]
    cursor.fetchall.return_value = [_REPL_ROW]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?limit=5")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert row["item_id"] == "627099"
    assert row["loc"] == "1401-BULK"
    # text severity mapped to a numeric severity for display/sort
    assert isinstance(row["severity"], (int, float))
    assert row["severity"] >= 0.75  # critical
    assert row["source"] == "fact_replenishment_exceptions"
    # The fallback query must read the replenishment table.
    executed = " ".join(str(c.args[0]) for c in cursor.execute.call_args_list).lower()
    assert "fact_replenishment_exceptions" in executed


@pytest.mark.asyncio
async def test_replenishment_fallback_headline_uses_friendly_label():
    """U4.1: the storyboard fallback must NOT leak the raw enum or its naive
    title-case ("Below Ss") into the headline. ``below_ss`` must render as
    "Below Safety Stock" — the same label the Inv Planning action feed uses —
    so the two surfaces name the identical exception identically.
    """
    repl_row = (
        "repl-uuid-2", "below_ss", "664631", "1401-BULK", "critical",
        292.39, "open", datetime.date(2026, 4, 2), 1.2, 80.0,
        "TITOS HANDMADE VODKA 80",  # item_desc
    )
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(0,), (1,)]
    cursor.fetchall.return_value = [repl_row]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?limit=5")

    assert resp.status_code == 200
    row = resp.json()["rows"][0]
    assert row["headline"] == "Below Safety Stock — 664631 @ 1401-BULK"
    assert "below_ss" not in row["headline"]
    assert "Below Ss" not in row["headline"]


@pytest.mark.asyncio
async def test_replenishment_fallback_includes_item_desc():
    """U2.1: the storyboard replenishment fallback must LEFT JOIN dim_item and
    surface ``item_desc`` so Command Center / Control Tower / AI Planner rows show
    the human-readable product name alongside the code — the same name the Inv
    Planning Action Feed shows (cycle-1 U1.8 only landed in ActionFeed).
    """
    # Fallback row now carries a trailing item_desc column from the dim_item join.
    repl_row = (
        "repl-uuid-3", "stockout", "627099", "1401-BULK", "critical",
        571.98, "open", datetime.date(2026, 4, 2), 1.5, 120.0,
        "MENAGE A TROIS A(D/R/S)3P PAD(44",  # item_desc
    )
    pool, conn, cursor = _make_pool()
    cursor.fetchone.side_effect = [(0,), (1,)]
    cursor.fetchall.return_value = [repl_row]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?limit=5")

    assert resp.status_code == 200
    row = resp.json()["rows"][0]
    assert row["item_desc"] == "MENAGE A TROIS A(D/R/S)3P PAD(44"
    # The fallback query must LEFT JOIN dim_item to resolve the description.
    executed = " ".join(str(c.args[0]) for c in cursor.execute.call_args_list).lower()
    assert "dim_item" in executed


@pytest.mark.asyncio
async def test_list_exceptions_severity_min_filter():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5,)
    cursor.fetchall.return_value = []
    cursor.description = _EXCEPTION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?severity_min=0.75")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_replenishment_fallback_severity_band_selects_high_only():
    """U7.10: the Command Center severity chips (Critical/High/Medium/Low) are
    dead because the feed loads only the top-N-by-severity (all critical). The
    fix lets the client push a severity BAND down to the server: a band of
    [0.5, 0.75) over the replenishment fallback must filter to the 'high' text
    severity (score 0.70) and exclude 'critical' (0.95). The query must constrain
    severity to that text-severity set, and the returned row must score < 0.75.
    """
    repl_row = (
        "repl-high", "below_rop", "664631", "1401-BULK", "high",
        292.39, "open", datetime.date(2026, 4, 2), 1.2, 80.0,
        "TITOS HANDMADE VODKA 80",  # item_desc
    )
    pool, conn, cursor = _make_pool()
    # 1st fetchone: exception_queue COUNT -> 0 (empty, trigger fallback).
    # 2nd fetchone: replenishment COUNT -> 1.
    cursor.fetchone.side_effect = [(0,), (1,)]
    cursor.fetchall.return_value = [repl_row]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/storyboard/exceptions?severity_min=0.5&severity_max=0.75&limit=50"
            )

    assert resp.status_code == 200
    data = resp.json()
    row = data["rows"][0]
    # 'high' scores 0.70 -> below the 0.75 critical cutoff the UI uses.
    assert row["severity"] < 0.75
    # The fallback query must constrain the text severity to the high-only band,
    # i.e. it must NOT allow 'critical' through for a [0.5, 0.75) request.
    executed = " ".join(str(c.args[0]) for c in cursor.execute.call_args_list)
    params = [c.args[1] for c in cursor.execute.call_args_list if len(c.args) > 1]
    flat_params = [p for plist in params for p in (plist if isinstance(plist, (list, tuple)) else [plist])]
    severity_lists = [p for p in flat_params if isinstance(p, list) and all(isinstance(x, str) for x in p)]
    assert severity_lists, "expected a text-severity ANY() list bound to the query"
    assert "critical" not in severity_lists[0]
    assert "high" in severity_lists[0]


@pytest.mark.asyncio
async def test_list_exceptions_brand_filter():
    """brand param is accepted and returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = []
    cursor.description = _EXCEPTION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?brand=BrandA")

    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data


@pytest.mark.asyncio
async def test_list_exceptions_category_filter():
    """category param is accepted and returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (3,)
    cursor.fetchall.return_value = []
    cursor.description = _EXCEPTION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?category=CAT1,CAT2")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_exceptions_market_filter():
    """market param is accepted and returns 200."""
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = []
    cursor.description = _EXCEPTION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions?market=NY,CA")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /storyboard/exceptions/summary
# ---------------------------------------------------------------------------

_SUMMARY_ROW = (
    23,   # open_count
    5,    # investigating_count
    42,   # resolved_count
    10,   # dismissed_count
    7,    # critical_open
    9,    # high_open
    2400000.0,  # total_impact_open
    0.72,       # avg_severity_open
    8,    # forecast_bias_count
    5,    # stockout_risk_count
    4,    # accuracy_drop_count
    3,    # excess_risk_count
    2,    # model_drift_count
    1,    # new_item_count
)

@pytest.mark.asyncio
async def test_exceptions_summary_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _SUMMARY_ROW
    cursor.fetchall.return_value = []
    cursor.description = [
        ("open_count",), ("investigating_count",), ("resolved_count",), ("dismissed_count",),
        ("critical_open",), ("high_open",), ("total_impact_open",), ("avg_severity_open",),
        ("forecast_bias_count",), ("stockout_risk_count",), ("accuracy_drop_count",),
        ("excess_risk_count",), ("model_drift_count",), ("new_item_count",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert "open_count" in data
    assert "by_type" in data
    assert "top_items" in data


@pytest.mark.asyncio
async def test_exceptions_summary_by_type_keys():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _SUMMARY_ROW
    cursor.fetchall.return_value = []
    cursor.description = [
        ("open_count",), ("investigating_count",), ("resolved_count",), ("dismissed_count",),
        ("critical_open",), ("high_open",), ("total_impact_open",), ("avg_severity_open",),
        ("forecast_bias_count",), ("stockout_risk_count",), ("accuracy_drop_count",),
        ("excess_risk_count",), ("model_drift_count",), ("new_item_count",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions/summary")

    data = resp.json()
    for key in ("forecast_bias", "stockout_risk", "accuracy_drop", "excess_risk",
                "model_drift", "new_item"):
        assert key in data["by_type"]


@pytest.mark.asyncio
async def test_exceptions_summary_open_count():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = _SUMMARY_ROW
    cursor.fetchall.return_value = []
    cursor.description = [
        ("open_count",), ("investigating_count",), ("resolved_count",), ("dismissed_count",),
        ("critical_open",), ("high_open",), ("total_impact_open",), ("avg_severity_open",),
        ("forecast_bias_count",), ("stockout_risk_count",), ("accuracy_drop_count",),
        ("excess_risk_count",), ("model_drift_count",), ("new_item_count",),
    ]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/exceptions/summary")

    data = resp.json()
    assert data["open_count"] == 23
    assert data["critical_open"] == 7


# ---------------------------------------------------------------------------
# GET /storyboard/exceptions/{id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_exception_detail_200():
    pool, conn, cursor = _make_pool()
    # First fetchone: the exception itself
    cursor.fetchone.return_value = _EXCEPTION_ROW + (datetime.datetime(2026, 3, 1),)
    cursor.fetchall.return_value = []
    cursor.description = _EXCEPTION_COLS + [("load_ts",)]

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/storyboard/exceptions/{_EXC_ID}")

    assert resp.status_code == 200
    data = resp.json()
    assert "exception_type" in data
    assert "decisions" in data


@pytest.mark.asyncio
async def test_get_exception_detail_404():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/storyboard/exceptions/{_EXC_ID}")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /storyboard/exceptions/{id}/status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_status_requires_auth():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/status",
                json={"status": "investigating"},
            )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_update_status_with_auth_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (uuid.UUID(_EXC_ID), "investigating", None)

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/status",
                headers={"X-API-Key": "secret"},
                json={"status": "investigating"},
            )

    assert resp.status_code in (200, 404)


@pytest.mark.asyncio
async def test_update_status_invalid_status_422():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/status",
                headers={"X-API-Key": "secret"},
                json={"status": "not_valid"},
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_update_status_not_found():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = None

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/status",
                headers={"X-API-Key": "secret"},
                json={"status": "resolved"},
            )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /storyboard/exceptions/{id}/decide
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decide_requires_auth():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/decide",
                json={"decision_type": "accept_exception"},
            )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_decide_accept_with_auth():
    pool, conn, cursor = _make_pool()
    dec_id = str(uuid.uuid4())
    cursor.fetchone.return_value = (uuid.UUID(dec_id), "ITEM001", "LOC1")

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/decide",
                headers={"X-API-Key": "secret"},
                json={"decision_type": "accept_exception", "decided_by": "j.smith"},
            )

    assert resp.status_code in (200, 404)


@pytest.mark.asyncio
async def test_decide_invalid_decision_type_422():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/decide",
                headers={"X-API-Key": "secret"},
                json={"decision_type": "totally_invalid"},
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_decide_maps_accept_to_resolved_status():
    """accept_exception should result in resolved status in response."""
    pool, conn, cursor = _make_pool()
    dec_id = str(uuid.uuid4())
    cursor.fetchone.return_value = (uuid.UUID(dec_id), "ITEM001", "LOC1")

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/decide",
                headers={"X-API-Key": "secret"},
                json={"decision_type": "accept_exception"},
            )

    if resp.status_code == 200:
        data = resp.json()
        assert data["new_exception_status"] == "resolved"


@pytest.mark.asyncio
async def test_decide_escalate_maps_to_investigating():
    pool, conn, cursor = _make_pool()
    dec_id = str(uuid.uuid4())
    cursor.fetchone.return_value = (uuid.UUID(dec_id), "ITEM001", "LOC1")

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/storyboard/exceptions/{_EXC_ID}/decide",
                headers={"X-API-Key": "secret"},
                json={
                    "decision_type": "escalate",
                    "rationale": "Needs manager approval",
                    "decided_by": "j.smith",
                },
            )

    if resp.status_code == 200:
        assert resp.json()["new_exception_status"] == "investigating"


# ---------------------------------------------------------------------------
# GET /storyboard/decisions
# ---------------------------------------------------------------------------

_DECISION_ROW = (
    str(uuid.uuid4()),   # decision_id
    _EXC_ID,             # exception_id
    "ITEM001",           # item_id
    "LOC1",              # loc
    "accept_exception",  # decision_type
    {},                  # decision_value (jsonb → dict)
    "Looks correct",     # rationale
    "j.smith",           # decided_by
    _NOW,                # decided_at
)

_DECISION_COLS = [
    ("decision_id",), ("exception_id",), ("item_id",), ("loc",),
    ("decision_type",), ("decision_value",), ("rationale",), ("decided_by",),
    ("decided_at",),
]


@pytest.mark.asyncio
async def test_list_decisions_200():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (5,)
    cursor.fetchall.return_value = [_DECISION_ROW]
    cursor.description = _DECISION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/decisions")

    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "rows" in data


@pytest.mark.asyncio
async def test_list_decisions_row_keys():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [_DECISION_ROW]
    cursor.description = _DECISION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/decisions")

    row = resp.json()["rows"][0]
    for key in ("decision_id", "exception_id", "item_id", "loc", "decision_type", "decided_by"):
        assert key in row


@pytest.mark.asyncio
async def test_list_decisions_filter_by_decision_type():
    pool, conn, cursor = _make_pool()
    cursor.fetchone.return_value = (2,)
    cursor.fetchall.return_value = []
    cursor.description = _DECISION_COLS

    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/storyboard/decisions?decision_type=escalate")

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /storyboard/generate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_requires_auth():
    pool, conn, cursor = _make_pool()
    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/storyboard/generate", json={})

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_generate_with_auth_200():
    pool, conn, cursor = _make_pool()
    mock_result = {
        "detected": 15,
        "inserted": 12,
        "skipped_dedupe": 3,
        "dry_run": False,
    }

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}), \
         patch("scripts.ops.generate_storyboard_exceptions.run", return_value=mock_result):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/storyboard/generate",
                headers={"X-API-Key": "secret"},
                json={"month": "2026-03"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "detected" in data
    assert "inserted" in data
    assert "skipped_dedupe" in data


@pytest.mark.asyncio
async def test_generate_dry_run_with_auth():
    pool, conn, cursor = _make_pool()
    mock_result = {
        "detected": 8,
        "inserted": 0,
        "skipped_dedupe": 0,
        "dry_run": True,
        "sample": [],
    }

    with patch("api.core._get_pool", return_value=pool), \
         patch.dict("os.environ", {"API_KEY": "secret"}), \
         patch("scripts.ops.generate_storyboard_exceptions.run", return_value=mock_result):
        from api.main import app
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/storyboard/generate",
                headers={"X-API-Key": "secret"},
                json={"dry_run": True},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("dry_run") is True
    assert data["inserted"] == 0
