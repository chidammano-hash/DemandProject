# Feature: Planning Date Configuration

## Objective

Allow the system to operate against **stale data** without requiring a data refresh when the system clock has advanced beyond the data's coverage period. All date-sensitive operations use a configurable "planning date" instead of the live system clock.

## Motivation

Demand Studio ingests data snapshots (sales, inventory, forecasts) that may be weeks or months old. When the system date advances beyond the data's coverage:

- Projection scripts compute `start_date = date.today()` — past all available inventory snapshots
- Exception detection looks for open orders and signals "near today" — finds nothing
- Champion forecast lookups use date windows anchored to "now" — return empty results
- Safety stock computations drift from the data's reference frame

Example: data covers through **Feb 24, 2026**, system date is **Mar 9, 2026** — a 13-day gap where all date-anchored queries return empty or misleading results.

## Design

### Single Source of Truth

A new module `common/planning_date.py` provides `get_planning_date() -> datetime.date`.
All production code imports and calls this function instead of `date.today()`.

### Config File

`config/planning_config.yaml`:

```yaml
planning:
  planning_date: "2026-02-24"    # frozen reference date for dev/testing
  use_system_date: false          # true = use real date.today() (production)
```

### Precedence (highest → lowest)

| Source | How to set |
|---|---|
| `USE_SYSTEM_DATE` env var | `USE_SYSTEM_DATE=true uv run python ...` |
| `PLANNING_DATE` env var | `PLANNING_DATE=2026-01-15 uv run python ...` |
| Config `use_system_date: true` | Edit `planning_config.yaml` |
| Config `planning_date: "..."` | Edit `planning_config.yaml` |
| Fallback | `date.today()` |

### Caching

Config is loaded lazily on the **first call** and cached for the process lifetime (same pattern as `common/db.py`). No config file I/O on subsequent calls. Use `_reset_cache()` (test-only helper) to clear between test cases.

### Environment Variable Patterns

```bash
# Development: use frozen date from config (default when use_system_date: false)
uv run python scripts/compute_inventory_projection.py --dfu 100320 1401-BULK

# One-off override via env var
PLANNING_DATE=2026-01-01 uv run python scripts/compute_inventory_projection.py

# Production / live data
USE_SYSTEM_DATE=true uv run python scripts/compute_inventory_projection.py

# Or set permanently for production in .env
USE_SYSTEM_DATE=true
```

## Files

### New Files

| File | Purpose |
|---|---|
| `config/planning_config.yaml` | Planning date config (frozen date + use_system_date flag) |
| `common/planning_date.py` | `get_planning_date()` module |
| `tests/unit/test_planning_date.py` | 12 unit tests |

### Modified Files (25 production files)

All scripts, API routers, and common modules that previously called `date.today()` or `datetime.date.today()` now call `get_planning_date()`.

**Scripts (19):**
`compute_inventory_projection.py`, `generate_replenishment_exceptions.py`, `generate_storyboard_exceptions.py`, `generate_planned_orders.py`, `compute_demand_signals.py`, `run_ss_simulation.py`, `compute_investment_plan.py`, `assign_replenishment_policies.py`, `generate_clustering_features.py`, `run_clustering_scenario.py`, `load_open_pos.py`, `release_planned_orders.py`, `compute_echelon_targets.py`, `compute_bias_corrections.py`, `compute_blended_forecast.py`, `compute_financial_plan.py`, `generate_consensus_plan.py`, `generate_quantile_forecasts.py`, `run_supply_chain_scenario.py`

**API Routers (3):**
`api/routers/inv_planning_policy.py`, `api/routers/consensus_plan.py`, `api/routers/dashboard.py` (new `GET /dashboard/planning-date` endpoint)

**Common (1):**
`common/ai_planner.py`

**Frontend (2):**
`frontend/src/components/GlobalFilterBar.tsx` (planning date chip), `frontend/src/tabs/inv-planning/ProjectionPanel.tsx` (chart reference line)

**Frontend query additions:**
`frontend/src/api/queries/core.ts` — `PlanningDateInfo` type, `fetchPlanningDate()`, `queryKeys.planningDate()`

### Not Modified

- Test files — tests remain clock-relative (they mock at the DB/pool level, not date level)
- Frontend `new Date()` calls — cosmetic only (CSV export filenames, scan timestamps); no data logic

## UI Design

### GlobalFilterBar Chip

A planning date badge is always visible on the right side of the global filter bar:

- **Frozen mode** (`is_frozen: true`): amber border + background, `CalendarClock` icon, "Plan: 24 Feb 2026 ⚠" label. Tooltip shows days behind system date.
- **Live mode** (`is_frozen: false`): muted styling, shows current system date. No warning indicator.

### Chart Reference Line (ProjectionPanel)

When the planning date is frozen, a vertical amber `ReferenceLine` is drawn at `planning_date` on the projection chart. Labeled "Plan Date". This visually anchors the projection start point on the timeline, making it clear the forecast starts from the frozen date rather than the chart's leftmost data point.

## API

```
GET /dashboard/planning-date
```

Response:
```json
{
  "planning_date": "2026-02-24",
  "system_date": "2026-03-09",
  "is_frozen": true,
  "days_behind": 13
}
```

## Module API

```python
from common.planning_date import get_planning_date, _reset_cache

# Production usage
today = get_planning_date()  # returns date(2026, 2, 24) per config

# Test usage — clear cache between test cases
_reset_cache()
```

## Testing

12 unit tests in `tests/unit/test_planning_date.py`:

1. Returns configured date when `use_system_date: false`
2. Returns `date.today()` when `use_system_date: true`
3. `PLANNING_DATE` env var overrides config file
4. `USE_SYSTEM_DATE` env var overrides config file
5. `USE_SYSTEM_DATE` beats `PLANNING_DATE` env var
6. Invalid `PLANNING_DATE` env var raises `ValueError`
7. Missing config file falls back to `date.today()`
8. Config missing `planning_date` key falls back to `date.today()`
9. Config is cached after first call
10. `_reset_cache()` causes re-read on next call
11. `USE_SYSTEM_DATE` accepts multiple truthy values (`true`, `1`, `yes`, etc.)
12. Invalid `planning_date` string in config raises `ValueError`
