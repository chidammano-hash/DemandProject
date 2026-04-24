"""Tests for common.twin.state (Gen-4 Cross-cutting #7)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from common.twin import TwinState


class _FakeCursor:
    """Plays back fetch results in the same order they will be requested."""

    def __init__(self, fetchone_queue: list, fetchall_queue: list):
        self._fetchone = list(fetchone_queue)
        self._fetchall = list(fetchall_queue)
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, tuple(params)))

    def fetchone(self):
        return self._fetchone.pop(0) if self._fetchone else None

    def fetchall(self):
        return self._fetchall.pop(0) if self._fetchall else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_from_db_hydrates_state_and_simulate_returns_array():
    # Three queries happen in TwinState.from_db:
    #   1) fetchall -> demand rows (24 months)
    #   2) fetchone -> lead time profile row
    #   3) fetchone -> safety stock row
    demand_rows = [(30.0,) for _ in range(12)]  # 1 unit/day after /30.44
    cur = _FakeCursor(
        fetchone_queue=[(14.0, 2.0), (100.0,)],  # lt_mean/lt_std, ss_combined
        fetchall_queue=[demand_rows],
    )
    conn = _FakeConn(cur)

    state = TwinState.from_db(conn, "ITEM_A", "LOC1")
    assert state.item_id == "ITEM_A"
    assert state.loc == "LOC1"
    assert state.on_hand == pytest.approx(100.0)
    assert state.lt_mean_days == pytest.approx(14.0)
    assert state.demand_mean > 0
    assert state.demand_pool.size == 12
    assert state.lt_pool.size == 100

    out = state.simulate(scenario={"extra_stock": 50.0}, n_iter=500)
    assert isinstance(out, np.ndarray)
    assert out.shape == (500,)
    # Simulated end-of-horizon stock = on_hand + extra_stock - demand_during_lt.
    # With on_hand=100, extra=50, expected mean < 150.
    assert out.mean() < 150.0


def test_simulate_rejects_nonzero_iter_and_mismatched_id():
    state = TwinState(
        item_id="A",
        loc="L",
        on_hand=10.0,
        demand_pool=np.array([1.0, 2.0, 3.0]),
        lt_pool=np.array([5.0, 7.0]),
    )
    with pytest.raises(ValueError):
        state.simulate(n_iter=0)
    with pytest.raises(ValueError):
        state.simulate(item_id="B", n_iter=10)


def test_simulate_deterministic_with_fixed_seed():
    state = TwinState(
        item_id="A",
        loc="L",
        on_hand=20.0,
        demand_pool=np.array([1.0, 2.0, 3.0]),
        lt_pool=np.array([3.0, 4.0, 5.0]),
    )
    a = state.simulate(n_iter=200, random_seed=7)
    b = state.simulate(n_iter=200, random_seed=7)
    assert np.array_equal(a, b)


def test_simulate_honors_scenario_pool_override():
    state = TwinState(
        item_id="A",
        loc="L",
        on_hand=100.0,
        demand_pool=np.array([10.0]),
        lt_pool=np.array([5.0]),
    )
    # Override demand pool with zero demand — stock should stay at on_hand.
    out = state.simulate(
        scenario={"demand_pool": [0.0], "lt_pool": [3.0], "horizon_days": 3},
        n_iter=50,
    )
    assert np.allclose(out, 100.0)


def test_from_db_raises_when_no_demand():
    cur = _FakeCursor(fetchone_queue=[], fetchall_queue=[[]])
    conn = _FakeConn(cur)
    with pytest.raises(ValueError):
        TwinState.from_db(conn, "ITEM_X", "LOC_Z")
