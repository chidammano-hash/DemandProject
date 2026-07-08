# 05 — Working Capital Analytics

| | |
|---|---|
| **Status** | Implemented (backend); unconsumed by the frontend |
| **UI Tab** | None -- see UI note below |
| **Key Files** | `api/routers/inventory/working_capital.py` |

## Problem

Finance and planning need cash-to-cash cycle visibility (DIO, DSO, DPO, inventory turns) alongside inventory planning.

## Solution

`GET /analytics/working-capital` computes working-capital metrics from inventory snapshots, sales, and PO data with graceful NULL fallbacks when source tables are empty. `GET /analytics/rolling-13-week` exposes weekly sales trend.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/analytics/working-capital` | DIO, DSO, DPO, cash-to-cash, turns |
| GET | `/analytics/rolling-13-week` | 13-week rolling sales |

Both endpoints are correctly documented above and both exist as written -- the gap is on the consumer side, not the API.

## UI

`FinancialPlanPanel` (`frontend/src/tabs/inv-planning/FinancialPlanPanel.tsx`) does **not** call this router. Its
working-capital chart calls `fetchWorkingCapitalTrend`, and its budget table calls `fetchBudgetStatus` -- both
defined in `frontend/src/api/queries/evolution.ts` -- which hit `GET /finance/working-capital-trend` and
`GET /finance/budget-status` on `api/routers/operations/financial_plan.py` (see `05-operations/02-financial-planning.md`),
not `/analytics/working-capital` or `/analytics/rolling-13-week` on this router.

As of this writing, no component under `frontend/src/` calls `/analytics/working-capital` or
`/analytics/rolling-13-week` -- confirmed by searching for both paths across `frontend/src/` (the only hits are
the auto-generated `frontend/src/api/generated/schema.ts`, not a live call site). `working_capital.py`'s
endpoints are implemented and correct but currently orphaned.
