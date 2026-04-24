"""Tests for common.ai.orchestrator — detect/simulate/rank/route scaffold."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from common.ai import orchestrator as orch_mod
from common.ai.orchestrator import (
    Exception as ExceptionRow,
    ExceptionOrchestrator,
    Option,
)


def _mk_exception(**overrides) -> ExceptionRow:
    base = {
        "exception_id": "EX-1",
        "item_id": "ITEM_A",
        "loc": "LOC1",
        "severity": "critical",
        "exception_type": "stockout",
        "current_qty_on_hand": 0.0,
        "recommended_order_qty": 100.0,
    }
    base.update(overrides)
    return ExceptionRow(**base)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executes: list[tuple] = []
        self.fetchone_results: list = []

    def execute(self, sql, params=None):
        self.executes.append((sql, params))

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        # Support the append_decision / reversible.apply sequence:
        # both of those call fetchone exactly once per INSERT so we pop
        # sequential ids from a queue populated per-test.
        return self.fetchone_results.pop(0) if self.fetchone_results else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_detect_returns_exception_rows():
    rows = [
        ("EX-1", "ITEM_A", "LOC1", "critical", "stockout", 0.0, 100.0),
        ("EX-2", "ITEM_B", "LOC2", "high", "low_stock", 5.0, 20.0),
    ]
    cur = _FakeCursor(rows)
    conn = _FakeConn(cur)

    exceptions = ExceptionOrchestrator().detect(conn)

    assert len(exceptions) == 2
    assert exceptions[0].exception_id == "EX-1"
    assert exceptions[0].current_qty_on_hand == 0.0
    assert exceptions[1].severity == "high"


def test_simulate_options_returns_three_ranked_scored_candidates():
    # Mock TwinState: simulate() returns a deterministic array for each call.
    twin = MagicMock()
    # Each option sets a unique extra_stock; return arrays that encode it.
    call_values = iter([
        np.array([80.0, 80.0]),   # expedite_transfer (extra=80)
        np.array([100.0, 100.0]), # emergency_po (extra=100)
        np.array([50.0, 50.0]),   # reallocate (extra=50)
    ])
    twin.simulate = lambda scenario=None, n_iter=2_000: next(call_values)

    exc = _mk_exception()
    options = ExceptionOrchestrator().simulate_options(exc, twin)

    assert len(options) == 3
    types = [o.action_type for o in options]
    assert set(types) == {"expedite_transfer", "emergency_po", "reallocate"}
    for o in options:
        assert isinstance(o.score, float)
        assert isinstance(o.projected_stock_at_horizon, float)


def test_rank_sorts_by_score_desc():
    exc = _mk_exception()
    a = Option(action_type="a", policy_id="p", exception=exc,
               projected_stock_at_horizon=10.0, cost=1.0, score=9.0)
    b = Option(action_type="b", policy_id="p", exception=exc,
               projected_stock_at_horizon=5.0, cost=1.0, score=4.0)
    c = Option(action_type="c", policy_id="p", exception=exc,
               projected_stock_at_horizon=20.0, cost=1.0, score=19.0)

    ranked = ExceptionOrchestrator.rank([a, b, c])

    assert [o.action_type for o in ranked] == ["c", "a", "b"]


def test_route_auto_applies_when_policy_permits(monkeypatch):
    # Policy permits + tier auto_within_policy.
    from common.ai.policy_engine import PolicyDecision
    monkeypatch.setattr(
        orch_mod,
        "evaluate",
        lambda _ctx: PolicyDecision(
            permitted=True,
            effective_tier="auto_within_policy",
            policy_id="p",
            reasons=[],
        ),
    )

    cur = _FakeCursor(rows=[])
    # reversible.apply fetches id=11; append_decision fetches latest-hash=None then id=12.
    cur.fetchone_results = [(11,), None, (12,)]
    conn = _FakeConn(cur)

    exc = _mk_exception()
    option = Option(
        action_type="expedite_transfer", policy_id="supply.auto_transfer",
        exception=exc, projected_stock_at_horizon=80.0, cost=10.0, score=70.0,
        payload={"order_qty": 100.0, "extra_stock": 80.0,
                 "item_id": exc.item_id, "loc": exc.loc},
    )

    result = ExceptionOrchestrator().route(conn, option)

    assert result["applied"] is True
    assert result["reversible_action_id"] == 11
    assert result["ledger_id"] == 12


def test_route_enqueues_when_policy_denies(monkeypatch):
    from common.ai.policy_engine import PolicyDecision
    monkeypatch.setattr(
        orch_mod,
        "evaluate",
        lambda _ctx: PolicyDecision(
            permitted=False,
            effective_tier="advisory",
            policy_id="p",
            reasons=["blast_radius_skus"],
        ),
    )

    cur = _FakeCursor(rows=[])
    # append_decision only: latest-hash lookup -> None; insert -> id=99.
    cur.fetchone_results = [None, (99,)]
    conn = _FakeConn(cur)

    exc = _mk_exception()
    option = Option(
        action_type="emergency_po", policy_id="supply.auto_transfer",
        exception=exc, projected_stock_at_horizon=50.0, cost=25.0, score=25.0,
        payload={"order_qty": 100.0},
    )

    result = ExceptionOrchestrator().route(conn, option)

    assert result["applied"] is False
    assert result["reversible_action_id"] is None
    assert result["ledger_id"] == 99


def test_route_with_permitted_but_suggestive_tier_does_not_auto_apply(monkeypatch):
    # Permitted but effective_tier below auto_within_policy -> still queued.
    from common.ai.policy_engine import PolicyDecision
    monkeypatch.setattr(
        orch_mod,
        "evaluate",
        lambda _ctx: PolicyDecision(
            permitted=True,
            effective_tier="suggestive",
            policy_id="p",
            reasons=[],
        ),
    )
    cur = _FakeCursor(rows=[])
    cur.fetchone_results = [None, (7,)]
    conn = _FakeConn(cur)

    exc = _mk_exception()
    option = Option(
        action_type="reallocate", policy_id="supply.auto_transfer",
        exception=exc, projected_stock_at_horizon=40.0, cost=5.0, score=35.0,
        payload={"order_qty": 100.0},
    )

    result = ExceptionOrchestrator().route(conn, option, requested_tier="suggestive")

    assert result["applied"] is False
    assert result["ledger_id"] == 7
