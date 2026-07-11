"""Financial and physical invariants for inventory-reduction opportunities."""

from __future__ import annotations

import pytest

from common.inventory.opportunities import build_reduction_opportunities
from common.inventory.quantile_targets import quantile_protection_target


def test_shared_pool_allocates_po_then_transfer_then_on_hand_without_overlap() -> None:
    result = build_reduction_opportunities(
        usable_on_hand_qty=140,
        eligible_open_po_qty=80,
        target_stock_qty=100,
        transfer_reservation_qty=30,
        unit_cost=10,
        annual_holding_rate=0.25,
        recovery_rate=0.60,
    )

    assert result.excess_pool_qty == 120
    assert [row.kind for row in result.opportunities] == [
        "open_po_reduction",
        "rebalance_transfer",
        "excess_stock_reduction",
    ]
    assert [row.reducible_qty for row in result.opportunities] == [80, 30, 10]
    assert sum(row.reducible_qty for row in result.opportunities) == 120
    assert result.opportunities[0].purchase_avoidance_value == 800
    assert result.opportunities[1].enterprise_reduction_value == 0
    assert result.opportunities[2].current_book_value == 100
    assert result.opportunities[2].recoverable_cash_value == 60


def test_reduction_financials_remain_separate() -> None:
    result = build_reduction_opportunities(
        usable_on_hand_qty=160,
        eligible_open_po_qty=20,
        target_stock_qty=100,
        transfer_reservation_qty=0,
        unit_cost=5,
        annual_holding_rate=0.20,
        recovery_rate=0.50,
    )
    po, on_hand = result.opportunities

    assert po.purchase_avoidance_value == 100
    assert po.current_book_value == 0
    assert po.recoverable_cash_value == 0
    assert po.annual_carrying_cost_savings == 20
    assert on_hand.current_book_value == 300
    assert on_hand.purchase_avoidance_value == 0
    assert on_hand.recoverable_cash_value == 150
    assert on_hand.annual_carrying_cost_savings == 60


def test_fractional_allocations_floor_without_exceeding_physical_pool() -> None:
    result = build_reduction_opportunities(
        usable_on_hand_qty=100.00009,
        eligible_open_po_qty=0,
        target_stock_qty=100,
        transfer_reservation_qty=0,
        unit_cost=10,
        annual_holding_rate=0.25,
        recovery_rate=0.60,
    )

    assert result.opportunities == ()
    assert result.allocated_qty <= result.excess_pool_qty


@pytest.mark.parametrize(
    "field,value",
    [
        ("eligible_open_po_qty", -1),
        ("target_stock_qty", -1),
        ("transfer_reservation_qty", -1),
        ("unit_cost", 0),
        ("annual_holding_rate", -0.1),
        ("recovery_rate", 1.1),
    ],
)
def test_invalid_opportunity_inputs_fail_loudly(field: str, value: float) -> None:
    kwargs = {
        "usable_on_hand_qty": 100,
        "eligible_open_po_qty": 20,
        "target_stock_qty": 80,
        "transfer_reservation_qty": 0,
        "unit_cost": 10,
        "annual_holding_rate": 0.25,
        "recovery_rate": 0.60,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        build_reduction_opportunities(**kwargs)


def test_quantile_target_preserves_p50_p90_protection_semantics() -> None:
    target = quantile_protection_target(
        lead_window=((100, 130), (120, 150)),
        review_window_p50=(90, 110),
    )

    assert target.protection_p50_qty == 220
    assert target.protection_p90_qty == pytest.approx(220 + (30**2 + 30**2) ** 0.5)
    assert target.safety_stock_qty == pytest.approx((30**2 + 30**2) ** 0.5)
    assert target.reorder_point_qty == target.protection_p90_qty
    assert target.target_stock_qty == pytest.approx(target.protection_p90_qty + 200)


def test_quantile_target_rejects_crossing_or_empty_quantiles() -> None:
    with pytest.raises(ValueError, match="lead window"):
        quantile_protection_target(lead_window=(), review_window_p50=(10,))
    with pytest.raises(ValueError, match="P90"):
        quantile_protection_target(lead_window=((100, 90),), review_window_p50=(10,))
