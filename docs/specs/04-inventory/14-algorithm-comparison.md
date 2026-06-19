# 14 — Inventory Algorithm Comparison

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Backtest |
| **Key Files** | `api/routers/inventory/inv_planning_algorithm_comparison.py` |

## Problem

Operators need to compare inventory planning algorithm variants (greedy vs LP rebalancer, policy assignments) on the same SKU sample.

## Solution

API endpoints expose side-by-side algorithm comparison metrics from inventory backtest runs. Used by the Inv. Backtest tab model comparison charts.

## API

Prefix `/inv-planning/algorithm-comparison` — summary and detail endpoints for comparing algorithm run results.
