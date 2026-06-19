# 29 — Consensus Plan & Planner Overrides

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Planning (Override Queue, Demand Forecast panels) |
| **Key Files** | `api/routers/forecasting/consensus_plan.py`, `common/engines/override_ledger.py`, `frontend/src/tabs/inv-planning/OverrideQueuePanel.tsx` |

## Problem

Statistical forecasts need planner adjustments for promos, launches, and known events. Overrides must be approved, merged into a consensus plan, and audited for future FVA measurement.

## Solution

Override workflow: submit → (optional) manager approval → merge into `fact_consensus_plan`. Approved overrides append a `DecisionRecord` to `ai_decision_ledger` via `record_override_approval()` (best-effort; does not block the workflow).

## API

| Method | Path | Description |
|---|---|---|
| GET | `/forecast/overrides/summary` | Portfolio override counts |
| GET | `/forecast/overrides` | Paginated override list |
| POST | `/forecast/overrides` | Submit override (`require_api_key`) |
| PUT | `/forecast/overrides/{id}/approve` | Manager approve |
| PUT | `/forecast/overrides/{id}/reject` | Manager reject |
| DELETE | `/forecast/overrides/{id}` | Soft-delete (supersede) |
| GET | `/forecast/consensus-plan` | Merged statistical + approved overrides |

## Ledger

On auto-approve (submit) or manager approve, writes `action_type=forecast_override_approved`, `agent_id=consensus_planner`, `policy_id=consensus_override`. Sets up future planner-FVA linkage (ladder stage still `planned`).

## Configuration

`config/forecasting/consensus_config.yaml` — valid override types, multiplier bounds, approval thresholds.
