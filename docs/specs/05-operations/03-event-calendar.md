# Event Calendar

> A structured calendar of promotions, product launches, phase-outs, and seasonal events that applies uplift or dampening adjustments to the statistical demand forecast before it enters the S&OP cycle.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning (EventCalendarPanel) |
| **Key Files** | `inv-planning/EventCalendarPanel.tsx`, `api/routers/operations/events.py`, `scripts/forecasting/apply_event_adjustments.py`, `config/operations/event_planning_config.yaml`, `sql/057_create_event_planning.sql` |

---

## Problem

Statistical forecasts assume the future looks like the past. Promotions, new product launches, and planned phase-outs break that assumption. Without a formal event calendar, planners make ad-hoc adjustments in spreadsheets that are invisible to the rest of the supply chain. When a 30% promotional uplift is applied inconsistently, downstream safety stock and replenishment calculations are wrong.

---

## Solution

A centralized event calendar stores planned events with their affected scope (items, locations, date range) and quantified impact (uplift or dampening multiplier). An adjustment pipeline applies these multipliers to the base statistical forecast, producing an event-adjusted forecast that flows into the S&OP demand review. Events have an approval workflow so managers can review adjustments before they take effect.

---

## How It Works

### Event Types

`event_type` is a free-text `VARCHAR(30)` on `fact_event_calendar` -- there is no DB-enforced enum. The
documented convention (from `sql/057_create_event_planning.sql`) is:

| Type | `event_type` value | Impact Direction | Example |
|---|---|---|---|
| Promotion | `promo` | Uplift | Discount / buy-one-get-one campaign |
| New Launch | `new_launch` | Uplift | New SKU introduction |
| Phase-Out | `phase_out` | Dampening | End-of-life SKU wind-down |
| Holiday | `holiday` | Uplift | Seasonal or holiday-driven surge |
| Cannibalization | `cannibalization` | Dampening | Demand shifted to a replacing SKU (`cannibalized_item_id`) |

### Adjustment Model

Each event carries an `uplift_pct`, a `ramp_profile` (`linear`\|`s_curve`\|`immediate`) applied over `ramp_weeks`,
an optional `pantry_loading_pct` over `pantry_loading_weeks` (pre-event stock-up demand), and an optional flat
`override_multiplier`. Per item-location-month, `fact_event_adjusted_forecast` stores the resulting quantities:
`base_forecast_qty`, `event_adjustment_qty` (the uplift), `post_promo_dip_qty` (post-promotion demand dip), and
`adjusted_forecast_qty`.

When events overlap the same item-location-month, `conflict_resolution` (default `highest_priority`) on the
event row decides which one wins based on `priority`; overlaps are also logged as rows in `fact_event_conflicts`.

### Approval Workflow

| Status | Meaning |
|---|---|
| `draft` | Event created but not yet approved |
| `approved` | Approved via `PUT /events/calendar/{id}/approve` |
| `active` | Event window is currently in progress |
| `completed` | Event window has ended |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `fact_event_calendar` | Event definitions, scope, and uplift parameters | `event_id`, `event_type`, `event_name`, `event_start`, `event_end`, `uplift_pct`, `ramp_profile`, `ramp_weeks`, `peak_qty_weekly`, `decay_rate`, `pantry_loading_pct`, `pantry_loading_weeks`, `last_order_date`, `cannibalized_item_id`, `override_multiplier`, `target_items`/`target_locations`/`target_categories` (JSONB), `status`, `conflict_resolution`, `priority` |
| `fact_event_adjusted_forecast` | Adjusted forecast output per item-location-month | `item_id`, `loc`, `plan_month`, `event_id`, `base_forecast_qty`, `event_adjustment_qty`, `post_promo_dip_qty`, `adjusted_forecast_qty`, `adjustment_type`, `order_deadline` |
| `fact_event_performance` | Post-event lift accuracy and calibration | `event_id`, `item_id`, `loc`, `plan_month`, `forecasted_lift_qty`, `actual_sales_qty`, `actual_lift_qty`, `lift_accuracy_pct`, `uplift_calibration_factor` |
| `fact_event_conflicts` | Overlapping-event log | `conflict_id`, `event_id_a`, `event_id_b`, `overlap_start`, `overlap_end`, `resolution_status` |

Event scope (which items/locations/categories an event applies to) is stored as JSONB arrays directly on
`fact_event_calendar` -- there is no separate scope-fact table.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/events/calendar` | List events, filterable by `event_type`, `status`, date range |
| POST | `/events/calendar` | Create a new event with scope and uplift parameters (auth) |
| GET | `/events/calendar/{event_id}` | Event detail |
| PUT | `/events/calendar/{event_id}/approve` | Approve an event (auth) |
| GET | `/events/impact-preview` | Adjusted-forecast rows for an event, item, or location |
| GET | `/events/performance` | Post-event forecasted vs. actual lift and calibration factor |

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make events-schema` | Creates event tables |
| Apply | `make events-apply` | Runs adjustment pipeline for current planning period |
| Apply (dry) | `make events-apply-dry` | Computes adjustments without writing |
| Full | `make events-all` | Schema + apply |

The adjustment script (`scripts/forecasting/apply_event_adjustments.py`) reads approved events, joins with the
base statistical forecast (`fact_external_forecast_monthly`, champion model), computes adjusted quantities, and
writes to `fact_event_adjusted_forecast`. Supports `--event-id`, `--month`, and `--dry-run`.

Caveat: the script's write path (`uplift_multiplier`, `additive_qty`, `uplift_delta_units`, `impact_value_usd`)
does not match the columns on `fact_event_adjusted_forecast` in the Data Model above -- verify schema alignment
before relying on `make events-apply` in production.

---

## Configuration

File: `config/operations/event_planning_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `event_planning.event_types` | Allowed event type codes (uppercase; a separate convention from the lowercase `event_type` values written by the live router -- see caveat below) | `PROMOTION`, `HOLIDAY`, `PRODUCT_LAUNCH`, `PHASE_OUT`, `DISRUPTION`, `TRADE_SHOW` |
| `event_planning.max_uplift_multiplier` | Guard rail on uplift multiplier | `5.0` |
| `event_planning.min_uplift_multiplier` | Floor on uplift multiplier (0 = zero demand, e.g. phase-out) | `0.0` |
| `event_planning.require_approval_above_impact_value` | USD impact above which an event requires approval | `5000.0` |
| `event_planning.require_approval_above_uplift_pct` | Uplift % above which an event requires approval | `20.0` |
| `event_planning.post_event_lag_weeks` | Weeks to wait after `event_end` before computing post-event accuracy | `2` |
| `event_planning.min_advance_days` | Minimum advance notice required to create a new event | `3` |
| `event_planning.conflict_window_days` | Window for flagging near-overlapping events for review | `7` |
| `scheduler.cron` | Daily adjustment-pipeline schedule | `0 6 * * *` |

Caveat: this config file's `event_types` list and its header comment (`Used by: scripts/apply_event_adjustments.py, api/routers/events.py`) reference the old, pre-move script/router paths and an uppercase event-type vocabulary that does not match the `event_type` values the live `events.py` router actually writes (see Event Types above). Treat the config as only loosely wired to the current implementation.

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
