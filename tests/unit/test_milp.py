"""Tests for common.ml.milp and scripts.ml.milp_rebalancer (scaffold)."""

from __future__ import annotations

import pytest

from common.ml.milp import (
    ExceptionProblem,
    MilpSolution,
    TransferCandidate,
    solve_rebalance,
)


def test_solve_rebalance_returns_greedy_solution_shape():
    exceptions = [
        ExceptionProblem(item_id="A", loc="L1", shortfall_qty=50, severity="critical"),
    ]
    pool = [
        TransferCandidate(item_id="A", source_loc="L2", dest_loc="L1",
                          available_qty=30, transfer_cost=1.0),
    ]

    sol = solve_rebalance(exceptions, pool, emergency_po_unit_cost=2.0)

    assert isinstance(sol, MilpSolution)
    assert sol.solver == "greedy_fallback"
    # One transfer for qty=30, one emergency PO for the residual 20.
    assert len(sol.transfers) == 1
    assert sol.transfers[0]["qty"] == 30
    assert len(sol.emergency_pos) == 1
    assert sol.emergency_pos[0]["qty"] == 20
    # Cost = 30 * 1.0 + 20 * 2.0 = 70.
    assert sol.total_cost == pytest.approx(70.0)


def test_solve_rebalance_prefers_cheapest_source():
    exceptions = [
        ExceptionProblem(item_id="A", loc="L1", shortfall_qty=10, severity="high"),
    ]
    pool = [
        TransferCandidate(item_id="A", source_loc="EXPENSIVE", dest_loc="L1",
                          available_qty=10, transfer_cost=5.0),
        TransferCandidate(item_id="A", source_loc="CHEAP", dest_loc="L1",
                          available_qty=10, transfer_cost=1.0),
    ]
    sol = solve_rebalance(exceptions, pool)
    assert sol.transfers[0]["source_loc"] == "CHEAP"


def test_solve_rebalance_orders_critical_first():
    exceptions = [
        ExceptionProblem(item_id="A", loc="L1", shortfall_qty=10, severity="low"),
        ExceptionProblem(item_id="A", loc="L2", shortfall_qty=10, severity="critical"),
    ]
    pool = [
        TransferCandidate(item_id="A", source_loc="S", dest_loc="L2",
                          available_qty=10, transfer_cost=1.0),
    ]
    sol = solve_rebalance(exceptions, pool)
    # The only candidate pairs with the critical (L2) exception first.
    assert sol.transfers[0]["dest_loc"] == "L2"


def test_solve_rebalance_empty_inputs():
    sol = solve_rebalance([], [])
    assert sol.transfers == []
    assert sol.emergency_pos == []
    assert sol.total_cost == 0.0


def test_milp_rebalancer_cli_dry_run_returns_zero():
    from scripts.ml.milp_rebalancer import run
    exit_code = run(["--dry-run"])
    assert exit_code == 0
