"""Tests for common.ai.dry_run — preview + confirm pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from common.ai import dry_run as dry_run_mod
from common.ai.dry_run import (
    DryRunAction,
    DryRunResult,
    confirm,
    dry_run,
    register_handler,
)


def _mk_action(**overrides) -> DryRunAction:
    base = {
        "action_type": "adjust_safety_stock",
        "target_kind": "item_id",
        "target_id": "ITEM_A",
        "params": {"new_ss": 100},
    }
    base.update(overrides)
    return DryRunAction(**base)


def test_dry_run_default_handler_returns_stub_flag():
    # No handler registered -> default returns empty plan with stub flag.
    result = dry_run(MagicMock(), _mk_action(action_type="no_handler_yet"))
    assert result.proposed_changes == []
    assert "no_handler_registered" in result.risk_flags
    assert result.estimated_impact == {}


def test_dry_run_dispatches_to_registered_handler():
    calls = []

    def handler(conn, action):
        calls.append((conn, action))
        return DryRunResult(
            action=action,
            proposed_changes=[{"table": "foo", "op": "UPDATE", "row": {"id": 1}}],
            risk_flags=["large_change"],
            estimated_impact={"fill_rate_delta": 0.02},
        )

    register_handler("some_action", handler)

    conn_sentinel = object()
    action = _mk_action(action_type="some_action")
    result = dry_run(conn_sentinel, action)

    assert len(calls) == 1
    assert calls[0][0] is conn_sentinel
    assert result.proposed_changes == [{"table": "foo", "op": "UPDATE", "row": {"id": 1}}]
    assert result.risk_flags == ["large_change"]


def test_dry_run_is_deterministic():
    def handler(_conn, action):
        return DryRunResult(
            action=action,
            proposed_changes=[{"table": "x", "op": "INSERT"}],
            risk_flags=[],
            estimated_impact={"revenue": 100.0},
        )

    register_handler("determ", handler)
    action = _mk_action(action_type="determ")
    a = dry_run(None, action).to_dict()
    b = dry_run(None, action).to_dict()
    assert a == b


def test_confirm_requires_approver():
    result = DryRunResult(action=_mk_action())
    with pytest.raises(ValueError):
        confirm(MagicMock(), result, approver="")


def test_confirm_writes_ledger_row(monkeypatch):
    # Stub append_decision so confirm stays unit-scoped.
    captured = {}

    def fake_append(_cur, record):
        captured["record"] = record
        return (77, "deadbeef" * 8)

    monkeypatch.setattr(dry_run_mod, "append_decision", fake_append)

    # Minimal connection with a cursor context manager.
    cur = MagicMock()
    cur.__enter__ = lambda self: self  # noqa: E731
    cur.__exit__ = lambda self, *a: False  # noqa: E731
    conn = MagicMock()
    conn.cursor = lambda: cur

    result = DryRunResult(
        action=_mk_action(),
        proposed_changes=[{"table": "t", "op": "UPDATE"}],
        risk_flags=["high_blast_radius"],
        estimated_impact={"cost": -500.0},
    )

    out = confirm(conn, result, approver="alice", policy_id="supply.auto_transfer")

    assert out["applied"] is True
    assert out["ledger_id"] == 77
    # Ledger received the approver as actor and the full plan as payload.
    rec = captured["record"]
    assert rec.actor == "alice"
    assert rec.outcome == "applied"
    assert rec.policy_id == "supply.auto_transfer"
    assert rec.payload["proposed_changes"] == [{"table": "t", "op": "UPDATE"}]
    assert rec.payload["risk_flags"] == ["high_blast_radius"]
