# 14 - Inventory Algorithm Comparison

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Backtest |
| **Key Files** | `scripts/inventory/compare_inventory_algorithms.py`, `api/routers/inventory/inv_planning_algorithm_comparison.py`, `sql/127_create_inventory_algorithm_comparison.sql` |

## Problem

Planners need to know how much a forecast *model* choice (LGBM, CatBoost, NBEATS, etc.) actually moves inventory targets - safety stock, EOQ, and reorder point - not just forecast accuracy. Two models with similar accuracy can still imply very different safety stock and ordering behavior once run through the same inventory formulas, and that gap is invisible from accuracy metrics alone.

This is a forecast-model comparison, not a rebalancing-solver comparison - it does not compare rebalancing algorithm variants (e.g., greedy vs. LP transfer solvers).

## Solution

`scripts/inventory/compare_inventory_algorithms.py` computes safety stock (combined variability), EOQ (Wilson formula), and reorder point for every forecast `model_id` present in `fact_production_forecast_staging`, using the same downstream inventory formulas (ABC-based service level/z-score, lead time profile per DFU) for each model so the comparison isolates the forecast's effect. Results are written per `model_id` (DELETE-then-INSERT) to `fact_inventory_algorithm_comparison`. The API exposes aggregate and per-DFU views for the Inv. Backtest tab's model comparison charts.

## API

Prefix `/inv-planning/algorithm-comparison`, router `api/routers/inventory/inv_planning_algorithm_comparison.py`:

| Method | Path | Params | Purpose |
|---|---|---|---|
| GET | `/inv-planning/algorithm-comparison/summary` | none | Aggregate avg SS/EOQ/ROP and total SS units/cycle stock per `model_id` |
| GET | `/inv-planning/algorithm-comparison/detail` | `item_id`, `loc`, `model_id`, `limit` (1-500, default 50), `offset` | Per-DFU comparison rows across models |
| GET | `/inv-planning/algorithm-comparison/models` | none | Distinct `model_id` values present, with SKU counts |

Pipeline: `make algo-comparison` runs `scripts/inventory/compare_inventory_algorithms.py` (`--models lgbm_cluster,nbeats` to restrict, `--dry-run` to skip writes).
