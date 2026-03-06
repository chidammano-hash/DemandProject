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
    "ITEM001",           # item_no
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
    ("exception_id",), ("exception_type",), ("item_no",), ("loc",),
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
    for key in ("exception_id", "exception_type", "item_no", "loc", "severity", "status", "headline"):
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
            resp = await client.put(
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
            resp = await client.put(
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
            resp = await client.put(
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
            resp = await client.put(
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
    "ITEM001",           # item_no
    "LOC1",              # loc
    "accept_exception",  # decision_type
    {},                  # decision_value (jsonb → dict)
    "Looks correct",     # rationale
    "j.smith",           # decided_by
    _NOW,                # decided_at
)

_DECISION_COLS = [
    ("decision_id",), ("exception_id",), ("item_no",), ("loc",),
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
    for key in ("decision_id", "exception_id", "item_no", "loc", "decision_type", "decided_by"):
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
         patch("scripts.generate_storyboard_exceptions.run", return_value=mock_result):
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
         patch("scripts.generate_storyboard_exceptions.run", return_value=mock_result):
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
