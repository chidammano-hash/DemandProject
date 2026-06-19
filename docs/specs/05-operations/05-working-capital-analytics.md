# 05 — Working Capital Analytics

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Planning (Financial Plan panel) |
| **Key Files** | `api/routers/inventory/working_capital.py`, `frontend/src/tabs/inv-planning/FinancialPlanPanel.tsx` |

## Problem

Finance and planning need cash-to-cash cycle visibility (DIO, DSO, DPO, inventory turns) alongside inventory planning.

## Solution

`GET /analytics/working-capital` computes working-capital metrics from inventory snapshots, sales, and PO data with graceful NULL fallbacks when source tables are empty. `GET /analytics/rolling-13-week` exposes weekly sales trend.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/analytics/working-capital` | DIO, DSO, DPO, cash-to-cash, turns |
| GET | `/analytics/rolling-13-week` | 13-week rolling sales |

## UI

`FinancialPlanPanel` charts working-capital trend via `fetchWorkingCapitalTrend` in `frontend/src/api/queries/evolution.ts`.
