"""Tests for common.ai.memory (working/episodic/semantic tiers)."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from common.ai.memory import (
    EpisodeRecord,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
)
from common.ai.rag import RagChunk


# ─────── WorkingMemory ───────

def test_working_memory_set_get_delete():
    wm = WorkingMemory()
    wm.set("x", 42)
    assert wm.get("x") == 42
    assert "x" in wm
    wm.delete("x")
    assert wm.get("x") is None


def test_working_memory_default_missing():
    wm = WorkingMemory()
    assert wm.get("missing", default="fallback") == "fallback"


def test_working_memory_ttl_expires(monkeypatch):
    wm = WorkingMemory(default_ttl=0.01)
    wm.set("k", "v")
    assert wm.get("k") == "v"
    # Advance monotonic clock past expiry.
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base + 1.0)
    assert wm.get("k") is None


def test_working_memory_clear_and_len():
    wm = WorkingMemory()
    wm.set("a", 1)
    wm.set("b", 2)
    assert len(wm) == 2
    wm.clear()
    assert len(wm) == 0


# ─────── EpisodicMemory ───────

def _cf(cursor: MagicMock):
    return lambda: cursor


def test_episodic_record_inserts_row():
    cursor = MagicMock()
    cursor.fetchone.return_value = (99,)
    mem = EpisodicMemory(_cf(cursor))
    new_id = mem.record(EpisodeRecord(
        decision_id=7, succeeded=True, outcome={"wape_delta": -0.03}, reward=0.4, notes="ok",
    ))
    assert new_id == 99
    args = cursor.execute.call_args.args
    assert "INSERT INTO fact_decision" in args[0]
    assert args[1][0] == 7
    assert json.loads(args[1][1]) == {"wape_delta": -0.03}
    assert args[1][2] is True
    assert args[1][3] == 0.4


def test_episodic_record_validates_decision_id():
    cursor = MagicMock()
    mem = EpisodicMemory(_cf(cursor))
    with pytest.raises(ValueError):
        mem.record(EpisodeRecord(decision_id=0, succeeded=False))
    with pytest.raises(ValueError):
        mem.record(EpisodeRecord(decision_id=-1, succeeded=True))


def test_episodic_record_raises_when_returning_empty():
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    mem = EpisodicMemory(_cf(cursor))
    with pytest.raises(ValueError):
        mem.record(EpisodeRecord(decision_id=1, succeeded=True))


def test_episodic_recent_parses_rows():
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (1, 10, "2026-04-20", json.dumps({"a": 1}), True, 0.5, "note"),
        (2, 11, "2026-04-21", {"b": 2}, False, -0.2, None),
    ]
    mem = EpisodicMemory(_cf(cursor))
    rows = mem.recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["outcome"] == {"a": 1}
    assert rows[1]["outcome"] == {"b": 2}
    assert rows[1]["succeeded"] is False


def test_episodic_recent_only_successes_adds_where():
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    mem = EpisodicMemory(_cf(cursor))
    mem.recent(limit=5, only_successes=True)
    sql = cursor.execute.call_args.args[0]
    assert "WHERE succeeded = TRUE" in sql


# ─────── SemanticMemory ───────

def test_semantic_remember_and_recall(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.side_effect = [[], []]  # vec + lex branches

    sem = SemanticMemory(_cf(cursor))
    chunk = RagChunk(doc_id="d", chunk_index=0, source="spec", text="t", embedding=[0.1] * 4)
    n = sem.remember([chunk])
    assert n == 1
    result = sem.recall("query", [0.1] * 4, k=3)
    assert result == []
