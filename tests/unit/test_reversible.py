"""Tests for common.ai.reversible — apply + sweeper mechanics."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from common.ai import reversible
from common.ai.reversible import (
    ReversibleAction,
    apply,
    detect_kpi_regressions,
    rollback_pending,
)


def test_apply_inserts_applied_row_with_expiry():
    cursor = MagicMock()
    cursor.fetchone.return_value = (42,)
    action = ReversibleAction(
        action_type="expedite_transfer",
        target_kind="exception_id",
        target_id="EX-1",
        rollback_payload={"qty": 5},
        applied_by="exception_agent",
    )

    new_id = apply(cursor, action)

    assert new_id == 42
    cursor.execute.assert_called_once()
    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO fact_reversible_action" in sql
    assert "'applied'" in sql
    # applied_at, expires_at are params 3,4; ensure expires_at > applied_at
    applied_at, expires_at = params[3], params[4]
    assert expires_at > applied_at
    # rollback_payload serialized as JSON string
    assert json.loads(params[5]) == {"qty": 5}
    assert params[6] == "exception_agent"


def test_apply_rejects_empty_fields():
    cursor = MagicMock()
    with pytest.raises(ValueError):
        apply(cursor, ReversibleAction(action_type="", target_kind="k", target_id="id"))


def test_apply_rejects_nonpositive_quiet_period():
    cursor = MagicMock()
    bad = ReversibleAction(
        action_type="a",
        target_kind="k",
        target_id="id",
        quiet_period_hours=0,
    )
    with pytest.raises(ValueError):
        apply(cursor, bad)


def test_detect_kpi_regressions_empty_returns_empty():
    assert detect_kpi_regressions(MagicMock(), []) == set()


def test_detect_kpi_regressions_stub_returns_empty():
    # Stub returns empty for now — the real detector is a TODO.
    assert detect_kpi_regressions(MagicMock(), [1, 2, 3]) == set()


def test_rollback_pending_marks_expired_when_no_regression():
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (1, "expedite_transfer", "exception_id", "EX-1", {"qty": 5}),
        (2, "emergency_po", "exception_id", "EX-2", {"qty": 10}),
    ]

    result = rollback_pending(cursor)

    # No regressions -> no rollbacks.
    assert result == []
    # Two UPDATE calls to mark 'expired' (after the SELECT).
    update_calls = [c for c in cursor.execute.call_args_list if "UPDATE" in c.args[0]]
    assert len(update_calls) == 2
    for call in update_calls:
        assert call.args[1][0] == "expired"


def test_rollback_pending_rolls_back_on_regression(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (1, "expedite_transfer", "exception_id", "EX-1", {"qty": 5}),
        (2, "emergency_po", "exception_id", "EX-2", {"qty": 10}),
    ]
    # Pretend action id=1 regressed; id=2 didn't.
    monkeypatch.setattr(reversible, "detect_kpi_regressions", lambda _c, ids: {1})

    result = rollback_pending(cursor)

    assert result == [1]
    update_calls = [c for c in cursor.execute.call_args_list if "UPDATE" in c.args[0]]
    assert len(update_calls) == 2
    statuses = [c.args[1][0] for c in update_calls]
    assert "rolled_back" in statuses
    assert "expired" in statuses


def test_rollback_pending_empty_is_noop():
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    assert rollback_pending(cursor) == []
