# Inventory Reduction Opportunities

## Outcome

Inventory planning can produce auditable, overlap-safe reduction opportunities without double
counting the same units. The initial opportunity types are open-PO reduction, rebalance transfer,
and excess-stock reduction.

## Quantity waterfall

For each item/location pool, usable on-hand and eligible open-PO quantity are compared with a
protected target. The excess pool is allocated in this fixed order:

1. open-PO reduction;
2. transfer reservation;
3. residual on-hand reduction.

The invariant is `current_qty = remaining_qty + reducible_qty`. A transfer changes location but has
zero enterprise inventory reduction. Negative quantities, costs, carrying rates, and targets are
rejected. Results are rounded down to four decimals so reporting never overstates savings.

Book value, purchase commitment, annual carrying cost, and recoverable value are separate fields;
they are never added into one ambiguous savings number.

## Lineage and review

Migration 204 adds `inventory_planning_run`, expands replenishment rows with forecast-release and
inventory-run lineage, and adds `fact_inventory_opportunity`. Immutable, versioned
`planning_decision_event` rows capture review decisions with idempotency keys.

New replenishment runs select only the verified active forecast release. Lexicographic selection of
the latest plan version is prohibited.

## Quantile shadow target

`common/inventory/quantile_targets.py` provides a shadow-only P50/P90 protection target. It is not an
automatic replacement for the production safety-stock policy; planners can compare it before any
future governed activation.
