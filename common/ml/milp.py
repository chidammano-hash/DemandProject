"""MILP transfer/replenishment optimizer scaffold.

Gen-4 Roadmap AI-8 (Stream H Phase 2). The production implementation
will solve a mixed-integer linear program that assigns expedite /
transfer / PO decisions across every open exception while respecting
capacity, cost, and service-level constraints.

Formulation sketch (TODO — implement with PuLP + highspy once we install
the solver):

    Decision vars:
        x[i, j, t]  integer transfer qty from loc j to loc i at time t
        y[i, t]     binary emergency-PO flag for SKU i at time t
        z[i, j, t]  integer reallocation qty from loc j to loc i at time t

    Objective:
        minimize  sum_{i,j,t} (cost_transfer * x[i,j,t])
                + sum_{i,t}   (cost_po * y[i,t])
                + sum_{i,t}   (stockout_penalty * backorder[i,t])

    Constraints:
        - Balance: inventory[i, t] = inventory[i, t-1] + inflow - outflow - demand
        - Capacity: sum_j x[i, j, t] <= truck_capacity[t]
        - Budget:   sum cost <= budget_ceiling
        - Service-level floor on inventory[i, t]

This scaffold returns a greedy fallback so the rest of the pipeline can
be wired end-to-end before the real solver lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExceptionProblem:
    """One shortage row fed into the optimizer."""

    item_id: str
    loc: str
    shortfall_qty: float
    severity: str = "medium"
    lead_time_days: float | None = None


@dataclass
class TransferCandidate:
    """A candidate source location with spare stock to draw from."""

    item_id: str
    source_loc: str
    dest_loc: str
    available_qty: float
    transfer_cost: float = 0.0


@dataclass
class MilpSolution:
    """Solver output.

    Attributes:
        transfers: list of ``{item_id, source_loc, dest_loc, qty, cost}``
            entries describing recommended transfers.
        emergency_pos: list of ``{item_id, loc, qty}`` entries for POs.
        unsatisfied: shortfalls that could not be covered.
        total_cost: sum of transfer + PO cost.
        solver: label of the solver that produced the solution
            (``"greedy_fallback"`` until PuLP/highspy lands).
    """

    transfers: list[dict[str, Any]] = field(default_factory=list)
    emergency_pos: list[dict[str, Any]] = field(default_factory=list)
    unsatisfied: list[dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    solver: str = "greedy_fallback"


def solve_rebalance(
    exceptions: list[ExceptionProblem],
    transfer_pool: list[TransferCandidate],
    *,
    emergency_po_unit_cost: float = 1.0,
) -> MilpSolution:
    """Greedy fallback: fill each exception from cheapest available source.

    TODO(gen-4 AI-8): replace with a real MILP once ``highspy`` is on
    the allow-list. The greedy version is intentionally simple: sort
    shortages by severity, then for each exception take the cheapest
    matching ``TransferCandidate``; any residual becomes an emergency PO.
    """
    _severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    ordered = sorted(
        exceptions, key=lambda e: (_severity_rank.get(e.severity, 99), -e.shortfall_qty)
    )

    # Pool of spare stock keyed by item_id; sorted cheapest first so each
    # exception greedily drains the cheapest source available.
    pool_by_item: dict[str, list[TransferCandidate]] = {}
    for cand in transfer_pool:
        pool_by_item.setdefault(cand.item_id, []).append(cand)
    for item_id in pool_by_item:
        pool_by_item[item_id].sort(key=lambda c: c.transfer_cost)

    solution = MilpSolution()
    for exc in ordered:
        remaining = float(exc.shortfall_qty)
        if remaining <= 0:
            continue
        candidates = pool_by_item.get(exc.item_id, [])
        for cand in candidates:
            if remaining <= 0:
                break
            if cand.available_qty <= 0 or cand.dest_loc != exc.loc:
                continue
            take = min(cand.available_qty, remaining)
            cost = take * cand.transfer_cost
            solution.transfers.append(
                {
                    "item_id": exc.item_id,
                    "source_loc": cand.source_loc,
                    "dest_loc": cand.dest_loc,
                    "qty": take,
                    "cost": cost,
                }
            )
            solution.total_cost += cost
            cand.available_qty -= take
            remaining -= take

        if remaining > 0:
            po_cost = remaining * emergency_po_unit_cost
            solution.emergency_pos.append(
                {"item_id": exc.item_id, "loc": exc.loc, "qty": remaining, "cost": po_cost}
            )
            solution.total_cost += po_cost

    # Anything still unfilled becomes 'unsatisfied' (empty when emergency
    # POs are allowed, but the field exists so a future budget-capped
    # solver can populate it).
    return solution


__all__ = [
    "ExceptionProblem",
    "TransferCandidate",
    "MilpSolution",
    "solve_rebalance",
]
