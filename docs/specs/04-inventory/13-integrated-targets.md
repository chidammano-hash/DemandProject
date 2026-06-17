# 13 — Integrated Targets

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Planning (Insights / Action Feed) |
| **Key Files** | `api/routers/inventory/integrated_targets.py` |

## Problem

Exception queue and insights need a unified view of safety stock, ROP, and replenishment targets per DFU without querying each planning sub-system separately.

## Solution

`GET /inv-planning/integrated-targets` returns per-DFU target rows (SS, ROP, EOQ, policy) with summary endpoint for portfolio rollups. Consumed by the async insights action feed.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/inv-planning/integrated-targets` | Paginated DFU targets |
| GET | `/inv-planning/integrated-targets/summary` | Portfolio KPI rollup |
