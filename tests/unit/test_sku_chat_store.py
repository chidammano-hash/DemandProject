"""Unit tests for Phase 3 persistence (common/ai/sku_chat/store.py)."""
from __future__ import annotations

from unittest.mock import MagicMock

import psycopg

from common.ai.sku_chat import store


def _mock_pool(*, fetchone=None, fetchall=None, description=None):
    cursor = MagicMock()
    cursor.description = description if description is not None else [("col",)]
    cursor.fetchone.return_value = fetchone
    cursor.fetchall.return_value = fetchall if fetchall is not None else []

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, cursor


def test_ensure_session_executes_upsert():
    pool, cur = _mock_pool()
    store.ensure_session(pool, "sess-1", "100320", "RETAIL", "DC1", "alice")
    cur.execute.assert_called_once()
    sql, params = cur.execute.call_args.args
    assert "ON CONFLICT" in sql
    assert params == ["sess-1", "100320", "RETAIL", "DC1", "alice"]


def test_save_message_returns_id():
    pool, _ = _mock_pool(fetchone=(42,))
    msg_id = store.save_message(pool, "sess-1", "user", "hi")
    assert msg_id == 42


def test_save_message_returns_none_when_no_row():
    pool, _ = _mock_pool(fetchone=None)
    assert store.save_message(pool, "sess-1", "user", "hi") is None


def test_log_call_maps_usage_fields():
    pool, cur = _mock_pool()
    store.log_call(
        pool,
        "sess-1",
        42,
        model="claude-opus-4-8",
        tier="deep",
        usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 30},
        cost_usd=0.02,
        tool_calls=3,
        latency_ms=1200,
        truncated=False,
    )
    _, params = cur.execute.call_args.args
    # order: session_id, message_id, model, tier, input, output, cache_read, cost, tools, latency, truncated
    assert params[4] == 100
    assert params[5] == 50
    assert params[6] == 30
    assert params[8] == 3
    assert params[10] is False


def test_get_session_with_messages():
    pool, cur = _mock_pool(
        fetchone=("sess-1", "100320", "RETAIL", "DC1", None, "t0", "t1"),
        fetchall=[(1, "user", "hi", None, None, "t2")],
    )
    session_desc = [
        ("session_id",), ("item_id",), ("customer_group",), ("loc",),
        ("created_by",), ("created_at",), ("last_active_at",),
    ]
    msg_desc = [("id",), ("role",), ("content",), ("model",), ("tier",), ("created_at",)]
    descs = [session_desc, msg_desc]
    cur.execute.side_effect = lambda *a, **k: setattr(cur, "description", descs.pop(0))

    out = store.get_session(pool, "sess-1")
    assert out is not None
    assert out["item_id"] == "100320"
    assert out["messages"][0]["content"] == "hi"


def test_get_session_not_found():
    pool, _ = _mock_pool(fetchone=None)
    assert store.get_session(pool, "missing") is None


def test_writes_are_best_effort_on_db_error():
    pool, cur = _mock_pool()
    cur.execute.side_effect = psycopg.OperationalError("boom")
    # None of these should raise:
    store.ensure_session(pool, "s", "i", "c", "l")
    assert store.save_message(pool, "s", "user", "x") is None
    store.log_call(
        pool, "s", None, model=None, tier=None, usage=None,
        cost_usd=None, tool_calls=0, latency_ms=0, truncated=False,
    )
    assert store.get_session(pool, "s") is None
