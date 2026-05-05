# Data Quality Engine

> Automatically checks every table for stale data, null keys, broken references, and statistical anomalies — so planners can trust the numbers before making inventory and forecasting decisions.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Data Quality |
| **Key Files** | `common/engines/dq_engine.py`, `config/platform/data_quality_config.yaml`, `scripts/fix_dq_issues.py`, `api/routers/platform/data_quality.py`, `frontend/src/tabs/DataQualityTab.tsx` |

---

## Problem

Supply Chain Command Center manages 11 domain datasets totaling hundreds of millions of rows. Without automated quality monitoring, issues like stale loads, null keys, duplicate records, and broken foreign keys go undetected until they corrupt downstream analytics — wrong accuracy metrics, incorrect safety stock, and misleading exception queues. A planner who acts on bad data makes bad decisions.

## Solution

A config-driven Data Quality Engine runs 12 check types against any table or column. Checks are defined in YAML (no code changes needed to add new ones), results are stored for trend analysis, and a dedicated UI tab shows per-domain health scores with an interactive Self-Heal workflow for fixing issues. The engine also provides statistical auto-fix capabilities for common problems like out-of-range values and null imputation.

## How It Works

1. Check definitions are declared in `config/platform/data_quality_config.yaml` — checks span all 11 domains.
2. The `DQEngine` loads the config, flattens it into a list of checks, and runs each one against PostgreSQL.
3. Each check returns a status (pass/fail/warn/error/skip) with a metric value and details.
4. Results are written to `fact_dq_check_results` for historical tracking.
5. The dashboard materialized view aggregates pass/fail counts by domain and day.
6. The UI shows domain health scores, pipeline freshness, a check catalog, and recent issues with AI-generated root cause analysis.
7. The Self-Heal panel lets users preview fixable issues, then selectively accept or reject each fix.

## Data Model

| Table | Purpose |
|---|---|
| `dim_dq_check_catalog` | Reusable check definitions (name, type, domain, threshold, severity) |
| `fact_dq_check_results` | Every check execution result (status, metric, details, timestamp) |
| `mv_dq_dashboard` | Aggregated pass/fail/warn counts by domain and day |

DDL: `sql/063_create_data_quality.sql`

## Check Types

| # | Type | What It Checks | Pass Condition |
|---|---|---|---|
| 1 | `freshness` | Hours since last data load | Within configured max hours |
| 2 | `completeness` | Null percentage for a column | Below max null threshold |
| 3 | `uniqueness` | Duplicate key groups | Zero duplicates |
| 4 | `row_count` | Minimum row count | Above minimum threshold |
| 5 | `range` | Values outside min/max bounds | Zero out-of-range rows |
| 6 | `volume_delta` | Row count change between loads | Below max % change |
| 7 | `referential_integrity` | Orphan foreign key values | Zero orphans |
| 8 | `statistical_outlier` | IQR or Z-score outliers | 5% or fewer outliers |
| 9 | `distribution_drift` | Mean/stddev/median shift between loads | Below drift threshold |
| 10 | `temporal_gaps` | Missing time periods in date columns | Zero gaps |
| 11 | `cross_column` | Custom SQL rule violations (e.g., "lag must match date diff") | Zero violations |
| 12 | `cardinality_anomaly` | Sudden change in distinct value count | Below max % change |

Freshness checks use the planning date (`get_planning_date()`), not wall-clock time, for consistent results in development.

## Self-Heal Auto-Fix

Five fix strategies handle common data quality issues:

| Strategy | What It Fixes | Method |
|---|---|---|
| `range` | Out-of-range values | Clamp to configured min/max bounds |
| `lead_time` | Extreme lead time values | Replace with per-item median, fallback to global median |
| `completeness` | NULL values | Numeric columns get column median; categorical get column mode |
| `orphans` | Broken foreign keys | Report only (recommends re-running the load pipeline) |
| `outliers` | Statistical outliers | Winsorize to IQR/Z-score computed bounds |

The Self-Heal UI workflow: Scan for Fixes -> Preview each fix with affected row count -> Accept or Reject individually or in bulk -> Applied fixes execute in a single transaction.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/data-quality/dashboard` | Domain health scores (last 24h) |
| GET | `/data-quality/checks` | Check catalog with last-run status |
| GET | `/data-quality/history` | Historical results (filterable by domain, days) |
| POST | `/data-quality/run` | Trigger ad-hoc check run (requires auth) |
| GET | `/data-quality/freshness` | Per-table last-load timestamps |
| POST | `/data-quality/fix` | Run auto-fix (dry-run or apply) |
| GET | `/data-quality/fix/preview` | Preview all fixable issues with IDs |
| POST | `/data-quality/fix/apply` | Apply selected fixes by preview IDs |

## Pipeline

| Step | Command | Description |
|---|---|---|
| Create tables | `make dq-schema` | Apply DDL (one-time) |
| Populate catalog | `make dq-populate` | Sync check catalog from config |
| Run all checks | `make dq-run` | Execute all 73 checks |
| Full pipeline | `make dq-all` | schema + populate + run |

CLI auto-fix:

```bash
uv run python scripts/fix_dq_issues.py              # Preview all fixes (dry-run)
uv run python scripts/fix_dq_issues.py --apply       # Apply all fixes
uv run python scripts/fix_dq_issues.py --fix range   # Fix only range issues
```

## Configuration

`config/platform/data_quality_config.yaml` — nested structure: `checks.<check_type>.<domain_key>`.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `schedule.cron` | `"0 */4 * * *"` | Run every 4 hours |
| `schedule.on_load` | `true` | Also run after each data load |
| Freshness: fact tables | 48 hours | Updated with each data load cycle |
| Freshness: inventory | 24 hours | Daily operational data |
| Freshness: dimensions | 168 hours (7 days) | Master data, less frequent updates |
| Freshness: time dimension | 720 hours (30 days) | Static/generated, rarely changes |

## Dependencies

- [Data Models](02-data-models.md) — all 11 domain tables must exist
- [Planning Date](04-planning-date.md) — freshness checks use the planning date
- [Infrastructure](01-infrastructure.md) — PostgreSQL connection

## See Also

- [Storyboard](../06-ai-platform/04-storyboard.md) — exception-based planner workflow that depends on data quality
- [Pipeline Orchestrator](../../../docs/RUNBOOK.md) — unified ETL pipeline with DQ checks after each load phase
