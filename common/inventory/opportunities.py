"""Overlap-safe inventory-reduction opportunities and financial semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OpportunityKind = Literal["open_po_reduction", "rebalance_transfer", "excess_stock_reduction"]


@dataclass(frozen=True, slots=True)
class ReductionOpportunity:
    kind: OpportunityKind
    current_qty: float
    remaining_qty: float
    reducible_qty: float
    current_book_value: float
    purchase_avoidance_value: float
    annual_carrying_cost_savings: float
    recoverable_cash_value: float
    enterprise_reduction_value: float


@dataclass(frozen=True, slots=True)
class ReductionOpportunityResult:
    excess_pool_qty: float
    allocated_qty: float
    opportunities: tuple[ReductionOpportunity, ...]


def _finite(name: str, value: float, *, positive: bool = False) -> None:
    valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    if not valid:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _floor_qty(value: float) -> float:
    """Floor at the persisted four-decimal quantity scale."""
    return math.floor(max(0.0, value) * 10_000) / 10_000


def _money(value: float) -> float:
    return round(value, 2)


def _opportunity(
    *,
    kind: OpportunityKind,
    current_qty: float,
    reducible_qty: float,
    unit_cost: float,
    annual_holding_rate: float,
    recovery_rate: float,
) -> ReductionOpportunity:
    reducible = _floor_qty(reducible_qty)
    current = _floor_qty(current_qty)
    remaining = _floor_qty(current - reducible)
    value = _money(reducible * unit_cost)
    is_po = kind == "open_po_reduction"
    is_on_hand = kind == "excess_stock_reduction"
    enterprise_value = value if is_po or is_on_hand else 0.0
    return ReductionOpportunity(
        kind=kind,
        current_qty=current,
        remaining_qty=remaining,
        reducible_qty=reducible,
        current_book_value=value if is_on_hand else 0.0,
        purchase_avoidance_value=value if is_po else 0.0,
        annual_carrying_cost_savings=_money(enterprise_value * annual_holding_rate),
        recoverable_cash_value=_money(value * recovery_rate) if is_on_hand else 0.0,
        enterprise_reduction_value=enterprise_value,
    )


def build_reduction_opportunities(
    *,
    usable_on_hand_qty: float,
    eligible_open_po_qty: float,
    target_stock_qty: float,
    transfer_reservation_qty: float,
    unit_cost: float,
    annual_holding_rate: float,
    recovery_rate: float,
) -> ReductionOpportunityResult:
    """Allocate one physical excess pool once in deterministic priority order."""
    if not math.isfinite(usable_on_hand_qty):
        raise ValueError("usable_on_hand_qty must be finite")
    for name, value in (
        ("eligible_open_po_qty", eligible_open_po_qty),
        ("target_stock_qty", target_stock_qty),
        ("transfer_reservation_qty", transfer_reservation_qty),
        ("annual_holding_rate", annual_holding_rate),
    ):
        _finite(name, value)
    _finite("unit_cost", unit_cost, positive=True)
    if not math.isfinite(recovery_rate) or not 0 <= recovery_rate <= 1:
        raise ValueError("recovery_rate must be between zero and one")

    excess_pool = _floor_qty(max(0.0, usable_on_hand_qty + eligible_open_po_qty - target_stock_qty))
    local_on_hand_excess = _floor_qty(max(0.0, usable_on_hand_qty - target_stock_qty))
    remaining_pool = excess_pool
    rows: list[ReductionOpportunity] = []

    po_qty = _floor_qty(min(eligible_open_po_qty, remaining_pool))
    if po_qty > 0:
        rows.append(
            _opportunity(
                kind="open_po_reduction",
                current_qty=eligible_open_po_qty,
                reducible_qty=po_qty,
                unit_cost=unit_cost,
                annual_holding_rate=annual_holding_rate,
                recovery_rate=recovery_rate,
            )
        )
        remaining_pool = _floor_qty(remaining_pool - po_qty)

    transfer_qty = _floor_qty(min(transfer_reservation_qty, local_on_hand_excess, remaining_pool))
    if transfer_qty > 0:
        rows.append(
            _opportunity(
                kind="rebalance_transfer",
                current_qty=local_on_hand_excess,
                reducible_qty=transfer_qty,
                unit_cost=unit_cost,
                annual_holding_rate=annual_holding_rate,
                recovery_rate=recovery_rate,
            )
        )
        remaining_pool = _floor_qty(remaining_pool - transfer_qty)
        local_on_hand_excess = _floor_qty(local_on_hand_excess - transfer_qty)

    on_hand_qty = _floor_qty(min(local_on_hand_excess, remaining_pool))
    if on_hand_qty > 0:
        rows.append(
            _opportunity(
                kind="excess_stock_reduction",
                current_qty=local_on_hand_excess,
                reducible_qty=on_hand_qty,
                unit_cost=unit_cost,
                annual_holding_rate=annual_holding_rate,
                recovery_rate=recovery_rate,
            )
        )

    allocated = _floor_qty(sum(row.reducible_qty for row in rows))
    if allocated > excess_pool:
        raise ValueError("opportunity allocation exceeds the physical excess pool")
    return ReductionOpportunityResult(
        excess_pool_qty=excess_pool,
        allocated_qty=allocated,
        opportunities=tuple(rows),
    )
