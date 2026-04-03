# Forecast Value Added (FVA) & ROI Tracking

> Measures whether human interventions (forecast overrides, policy changes, AI insight actions) improve or degrade forecast accuracy, and calculates the financial return on planning effort.

| | |
|---|---|
| **Status** | ✅ Implemented |
| **UI Tab** | FVA |
| **Key Files** | `api/routers/fva.py`, `frontend/src/tabs/FVATab.tsx`, `config/fva_config.yaml`, `sql/068_create_fva_tracking.sql`, `frontend/src/api/queries/platform.ts` |

---

## Problem

Planners spend hours adjusting forecasts, changing safety stock targets, and responding to AI recommendations. But nobody knows if these interventions actually help. Without measurement, low-value overrides persist while high-value adjustments go unrecognized. Management cannot justify planning headcount or tool investment without ROI data.

## Solution

Two complementary views answer "Is the planning process adding value?"

**FVA Ladder** compares accuracy across the planning handoff sequence to show where the process adds value: `Naive Seasonal -> External -> Champion`, with reserved placeholders for future `AI Adjusted` and `Planner Adjusted` stages.

**Intervention Tracker** records every planner action (forecast override, policy change, safety stock adjustment, AI insight response, S&OP approval) with before/after metric snapshots and financial impact estimates. After a configurable measurement window, actuals are compared to show whether the intervention helped or hurt.

## How It Works

1. When a planner takes an action, the source module records it in `fact_intervention_metrics` with the current metric snapshot and estimated financial impact
2. The intervention starts with status `pending` and a measurement window (e.g., 3 months for overrides, 6 months for policy changes)
3. The FVA ladder endpoint queries `fact_external_forecast_monthly`, grouping accuracy by `model_id` to compare `seasonal_naive`, `external`, and `champion`, then appends planned `AI Adjusted` and `Planner Adjusted` stages for the UI roadmap
4. When the measurement window expires and actuals are available, a background job computes the actual metric outcome and financial impact
5. The intervention status flips to `measured`, and the FVA tab shows the realized ROI
6. The ROI summary aggregates all interventions to show total estimated vs. actual financial impact

## Data Model

### `fact_intervention_metrics`

| Column | Type | Description |
|---|---|---|
| `intervention_id` | `BIGSERIAL PK` | Auto-increment ID |
| `user_id` | `UUID` | Planner who took the action (nullable) |
| `intervention_type` | `TEXT` | forecast_override, ss_change, policy_change, ai_insight_action, sop_approval |
| `resource_type` | `TEXT` | Target entity type (sku, policy, plan) |
| `resource_id` | `TEXT` | Target entity identifier (e.g., "100320-1401") |
| `metric_before` | `JSONB` | Key metrics snapshot before intervention |
| `metric_after` | `JSONB` | Key metrics snapshot after measurement window |
| `financial_impact_estimate` | `NUMERIC` | Estimated dollar impact at time of action |
| `actual_financial_impact` | `NUMERIC` | Measured dollar impact after window expires |
| `measurement_window_start` | `DATE` | Start of measurement period |
| `measurement_window_end` | `DATE` | End of measurement period |
| `status` | `TEXT` | pending (within window) or measured (actuals available) |
| `created_at` | `TIMESTAMPTZ` | When the intervention was recorded |

Indexes on `user_id`, `status`, `intervention_type`, and `measurement_window_end`.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/fva/waterfall` | Ordered ladder stages (`seasonal_naive`, `external`, `champion`, `ai_adjusted`, `planner_adjusted`) plus a separate `ceiling` benchmark |
| GET | `/fva/interventions` | Paginated intervention log with before/after metrics |
| GET | `/fva/roi-summary` | Aggregate ROI: total interventions, estimated vs. actual impact |

`/fva/waterfall` and `/fva/roi-summary` accept a `months` query parameter (default 12, range 1-36) for the lookback window. `/fva/interventions` uses pagination and filter parameters. Accuracy uses the standard formula: `100 - (100 * SUM(|F-A|) / |SUM(A)|)`.

## Configuration

`config/fva_config.yaml`:

```yaml
measurement_windows:
  forecast_override: 3    # months to wait before measuring outcome
  ss_change: 6
  policy_change: 6
  ai_insight_action: 3
  sop_approval: 6
financial_impact:
  carrying_cost_pct: 0.25           # annual carrying cost as % of inventory value
  stockout_cost_per_unit: 50        # estimated cost per unit stockout
  service_level_target: 0.95        # target fill rate for impact calculations
```

Short-horizon actions (forecast overrides, AI responses) use 3-month windows. Structural changes (policies, safety stock, S&OP approvals) use 6-month windows.

## Frontend

The FVA tab has four sections:

1. **Header** -- "Forecast Value Added" title + month selector dropdown (3, 6, 12, 24 months)
2. **ROI KPI Cards** -- Total Interventions, Measured count, Estimated Impact ($), Actual Impact ($)
3. **Forecast Value Ladder** -- ordered stage cards for `Naive Seasonal -> External -> Champion -> AI Adjusted -> Planner Adjusted`, showing stage accuracy, delta vs prior stage, and a separate **Ceiling Benchmark** card
4. **Recent Interventions** -- Last 10 interventions with status dot (green=measured, amber=pending), type, resource, and financial impact

## Integration Points

Interventions are recorded by upstream features:

| Source | Intervention Type |
|---|---|
| Override Queue Panel | `forecast_override` |
| Policy Management Panel | `policy_change` |
| Safety Stock Panel | `ss_change` |
| AI Planner (accept/resolve) | `ai_insight_action` |
| S&OP Tab (approve plan) | `sop_approval` |

## Current vs Planned Stages

Currently measured ladder stages:

- `seasonal_naive`
- `external`
- `champion`

Roadmap stages already rendered in the UI as placeholders:

- `ai_adjusted`
- `planner_adjusted`

The `ceiling` value is shown as a benchmark card rather than part of the main ladder so it acts as a reference target, not a production handoff.

## Dependencies

- No external dependencies beyond existing stack
- Queries use `STALE_PLATFORM` (5 min) stale time in frontend

## See Also

- [Champion Selection](../02-forecasting/05-champion-selection.md) -- model tiers shown in waterfall
- [AI Planning Agent](../05-ai-platform/01-ai-planning-agent.md) -- insight actions tracked as interventions
- [Collaboration](./05-collaboration.md) -- annotations provide qualitative context for interventions
