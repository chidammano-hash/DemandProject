# Exception Queue

> Scans inventory, forecast, and policy data to detect 6 anomaly types, scores them 0-100 by severity, deduplicates within a 7-day window, and presents a prioritized work queue with recommended actions and a 4-state workflow.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inventory Planning |
| **Key Files** | `scripts/generate_replenishment_exceptions.py`, `api/routers/inventory/inv_planning_exceptions.py`, `config/operations/exception_config.yaml`, `sql/027_create_replenishment_exceptions.sql` |

---

## Problem

Planners cannot monitor every DFU daily. They need the system to surface only the items that require human intervention -- stockouts, policy violations, unusual demand spikes -- ranked by severity so they can work the highest-impact exceptions first.

---

## Solution

An exception detection engine that scans inventory, forecast, and policy data to identify 6 types of anomalies, scores them by severity, deduplicates within a 7-day window, and presents them as a prioritized work queue with recommended actions.

---

## How It Works

### Exception Types

| Type | Trigger | Recommendation |
|---|---|---|
| `stockout_risk` | Projected DOS < lead time | Expedite order or reallocate |
| `excess_inventory` | DOS > 2x target | Reduce next order, consider transfer |
| `below_safety_stock` | Current qty < SS target | Place replenishment order |
| `forecast_deviation` | Actual vs forecast > threshold | Review forecast, check market signals |
| `policy_violation` | Ordering outside policy params | Align to assigned policy |
| `lead_time_change` | LT shifted > 2 std dev | Update LT assumptions, adjust SS |

### Severity Scoring

Each exception is scored 0-100 based on:

| Factor | Weight | Logic |
|---|---|---|
| Financial impact | 40% | Revenue at risk (ABC-A items score higher) |
| Time urgency | 30% | How close to stockout (days) |
| Recurrence | 15% | Repeated exceptions score higher |
| Forecast confidence | 15% | Low confidence = higher severity |

Severity levels: `critical` (>= 80), `high` (>= 60), `medium` (>= 40), `low` (< 40).

### Deduplication

Same exception type + same DFU within 7 days is suppressed. The existing exception's severity is updated if the new score is higher.

### Workflow States

```
open -> acknowledged -> ordered -> resolved
```

Planners move exceptions through states as they take action. Resolved exceptions remain in history for trend analysis.

---

## Data Model

| Table | Grain | Key Columns |
|---|---|---|
| `fact_replenishment_exceptions` | exception_id | item_id, loc, exception_type, severity, severity_score, status, recommendation, detected_at |

6 indexes cover: status filtering, severity ranking, DFU lookup, type grouping, date range, and composite queries.

DDL: `sql/027_create_replenishment_exceptions.sql`

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/inv-planning/exceptions` | Paginated exception queue (default: open, sorted by severity) |
| GET | `/inv-planning/exceptions/summary` | Counts by type and severity |
| POST | `/inv-planning/exceptions/{exception_id}/acknowledge` | Mark as acknowledged (body: `acknowledged_by`, `notes`; auth required) |
| POST | `/inv-planning/exceptions/{exception_id}/status` | Advance status to `ordered` or `resolved` (body: `status`, `notes`; auth required) |
| GET | `/inv-planning/exceptions/{exception_id}/lifecycle` | Append-only state-transition history for one exception |
| GET | `/inv-planning/exceptions/mttr` | Mean time to resolve, by type and severity |
| POST | `/inv-planning/exceptions/generate` | Trigger exception scan (auth required) |

Router: `inv_planning_exceptions.py`

---

## Pipeline

```
make exceptions-generate       # Detect + write exceptions
make exceptions-generate-dry   # Preview without writing (--dry-run)
```

| Step | Script | Output |
|---|---|---|
| Detect | `scripts/generate_replenishment_exceptions.py` | `fact_replenishment_exceptions` rows |

Supports `--dry-run` for previewing exceptions before committing.

---

## Configuration

File: `config/operations/exception_config.yaml`

```yaml
dedup_window_days: 7
severity_weights:
  financial_impact: 0.4
  time_urgency: 0.3
  recurrence: 0.15
  forecast_confidence: 0.15
thresholds:
  stockout_risk:
    dos_below_lt_pct: 1.0      # DOS < 100% of lead time
  excess_inventory:
    dos_above_target_multiple: 2.0
  forecast_deviation:
    abs_pct_threshold: 0.30    # >30% deviation
```

---

## Dependencies

- **Upstream:** `fact_safety_stock_targets`, `agg_inventory_monthly`, `fact_external_forecast_monthly`, `fact_dfu_policy_assignment`, `dim_item_lead_time_profile`
- **Downstream:** Control tower (aggregates exception counts), AI planner (consumes exceptions for analysis), storyboard (exception-based planner workflow)

---

## See Also

- [04-replenishment](04-replenishment.md) -- Policy violations feed exception queue
- [03-safety-stock](03-safety-stock.md) -- Below-SS triggers exceptions
- [../06-ai-platform/01-ai-planning-agent](../06-ai-platform/01-ai-planning-agent.md) -- AI agent analyzes exceptions
- [../06-ai-platform/03-control-tower](../06-ai-platform/03-control-tower.md) -- Exception counts on control tower dashboard
