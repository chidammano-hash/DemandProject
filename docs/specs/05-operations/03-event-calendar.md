# Event Calendar

> A structured calendar of promotions, product launches, phase-outs, and seasonal events that applies uplift or dampening adjustments to the statistical demand forecast before it enters the S&OP cycle.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning (EventCalendarPanel) |
| **Key Files** | `inv-planning/EventCalendarPanel.tsx`, `api/routers/operations/events.py`, `scripts/apply_event_adjustments.py`, `config/operations/event_planning_config.yaml`, `sql/058_create_event_calendar.sql` |

---

## Problem

Statistical forecasts assume the future looks like the past. Promotions, new product launches, and planned phase-outs break that assumption. Without a formal event calendar, planners make ad-hoc adjustments in spreadsheets that are invisible to the rest of the supply chain. When a 30% promotional uplift is applied inconsistently, downstream safety stock and replenishment calculations are wrong.

---

## Solution

A centralized event calendar stores planned events with their affected scope (items, locations, date range) and quantified impact (uplift or dampening multiplier). An adjustment pipeline applies these multipliers to the base statistical forecast, producing an event-adjusted forecast that flows into the S&OP demand review. Events have an approval workflow so managers can review adjustments before they take effect.

---

## How It Works

### Event Types

| Type | Impact Direction | Typical Multiplier Range | Example |
|---|---|---|---|
| Promotion | Uplift | 1.1 -- 2.0 | Buy-one-get-one campaign |
| Product Launch | Uplift | Varies widely | New SKU introduction |
| Phase-Out | Dampening | 0.0 -- 0.8 | End-of-life SKU wind-down |
| Seasonal Peak | Uplift | 1.2 -- 1.8 | Back-to-school or holiday surge |
| Seasonal Trough | Dampening | 0.5 -- 0.9 | Post-holiday slowdown |
| Supply Disruption | Dampening | 0.0 -- 0.7 | Known supplier shutdown |

### Adjustment Formula

For each item-location-month covered by an event:

`adjusted_forecast = base_forecast x event_multiplier`

When multiple events overlap the same item-location-month, multipliers are applied multiplicatively:

`adjusted_forecast = base_forecast x multiplier_1 x multiplier_2 x ...`

### Approval Workflow

| Status | Meaning |
|---|---|
| `draft` | Event created but not yet reviewed |
| `pending_approval` | Submitted for manager review |
| `approved` | Adjustment will be applied in next pipeline run |
| `rejected` | Adjustment will not be applied |
| `expired` | Event end date has passed |

Only events with `approved` status are applied by the adjustment pipeline.

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `dim_event` | Event definitions | `event_id`, `event_type`, `name`, `start_date`, `end_date`, `status` |
| `fact_event_scope` | Items and locations affected | `event_id`, `item_id`, `loc`, `multiplier` |
| `fact_event_adjusted_forecast` | Adjusted forecast output | `item_id`, `loc`, `month`, `base_qty`, `adjusted_qty`, `event_ids` |
| `fact_event_audit` | Change history | `event_id`, `changed_by`, `old_status`, `new_status`, `changed_at` |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/events` | List events with filtering by type, status, date range |
| POST | `/events` | Create a new event with scope and multipliers |
| PUT | `/events/{id}` | Update event details or scope |
| PUT | `/events/{id}/status` | Advance event through approval workflow |
| POST | `/events/apply` | Run the adjustment pipeline for a given period |
| GET | `/events/impact` | Preview aggregate impact of approved events on forecast |

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make events-schema` | Creates event tables |
| Apply | `make events-apply` | Runs adjustment pipeline for current planning period |
| Full | `make events-all` | Schema + apply |

The adjustment script (`scripts/apply_event_adjustments.py`) reads approved events, joins with the base production forecast, computes adjusted quantities, and writes to `fact_event_adjusted_forecast`. Supports `--dry-run` for preview.

---

## Configuration

File: `config/operations/event_planning_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `max_multiplier` | Cap on any single event multiplier | `3.0` |
| `overlap_strategy` | How to combine overlapping events | `multiplicative` |
| `auto_expire_days` | Days after end date to mark expired | `7` |
| `require_approval` | Whether events need manager approval | `true` |
| `default_event_types` | Pre-defined event type list | 6 types as listed above |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Production forecast (02-08) | Base forecast that events adjust |
| S&OP cycle (05-01) | Adjusted forecast feeds into Stage 1 demand review |
| Item dimension (`dim_item`) | Scope events to specific items |
| Location dimension (`dim_location`) | Scope events to specific locations |

---

## See Also

- `05-operations/01-sop-cycle.md` -- event-adjusted forecast enters S&OP demand review
- `05-operations/04-scenario-planning.md` -- events can be modeled as scenarios for what-if analysis
- `02-forecasting/08-production-forecast.md` -- base forecast that events modify
