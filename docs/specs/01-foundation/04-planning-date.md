# Planning Date Configuration

> Replaces `date.today()` with a configurable planning date so the entire system works correctly against stale data — no more empty queries when the calendar moves past the data's coverage period.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (visible as badge in Global Filter Bar) |
| **Key Files** | `common/planning_date.py`, `config/planning_config.yaml`, `api/routers/dashboard.py` |

---

## Problem

Supply Chain Command Center ingests data snapshots (sales, inventory, forecasts) that may be weeks or months old. When the system date advances past the data's last month, date-anchored queries return empty or misleading results. Projection scripts start after all inventory snapshots. Exception detection finds nothing. Champion lookups return empty. For example: data covers through Feb 24, 2026, but the system date is Mar 9, 2026 — a 13-day gap where every date-sensitive feature breaks silently.

## Solution

A single module (`common/planning_date.py`) provides `get_planning_date()` that every script, router, and common module calls instead of `date.today()`. The date can be frozen in config for development, overridden via environment variables for one-off runs, or set to live system time for production. Twenty-five production files were migrated to use this function.

## How It Works

1. On first call, `get_planning_date()` loads `config/planning_config.yaml` and caches the result.
2. It checks environment variables first, then config file values, then falls back to `date.today()`.
3. All 19 scripts, 3 API routers, and 1 common module call this function instead of `date.today()`.
4. The Global Filter Bar shows a planning date badge — amber when frozen, muted when live.
5. The Projection Panel draws a vertical reference line at the planning date on its chart.

## Configuration

`config/planning_config.yaml`:

```yaml
planning:
  planning_date: "2026-02-24"    # frozen reference date for dev/testing
  use_system_date: false          # true = use real date.today() (production)
```

### Precedence (highest wins)

| Priority | Source | Example |
|---|---|---|
| 1 | `USE_SYSTEM_DATE` env var | `USE_SYSTEM_DATE=true uv run python ...` |
| 2 | `PLANNING_DATE` env var | `PLANNING_DATE=2026-01-15 uv run python ...` |
| 3 | Config `use_system_date: true` | Set in planning_config.yaml |
| 4 | Config `planning_date` value | Set in planning_config.yaml |
| 5 | Fallback | `date.today()` |

The config is loaded once and cached for the process lifetime. Use `_reset_cache()` in tests to clear between test cases.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/dashboard/planning-date` | Returns planning_date, system_date, is_frozen, days_behind |

## Dependencies

- [Infrastructure](01-infrastructure.md) — config file must be accessible at `config/planning_config.yaml`

## See Also

- [Data Quality](03-data-quality.md) — freshness checks use the planning date instead of wall-clock time
- [Inventory Projection](../03-inventory/01-inventory-snapshot.md) — projection start date anchored to planning date
