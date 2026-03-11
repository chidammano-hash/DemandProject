# 07-02 Data Quality Engine

## Overview

Automated data quality monitoring framework that validates incoming data against configurable rules, tracks quality scores over time, and surfaces issues via a dashboard materialized view.

## Components

| Component | Path | Purpose |
|---|---|---|
| Engine | `common/dq_engine.py` | `DQEngine` class: rule evaluation, scoring, check orchestration |
| Router | `api/routers/data_quality.py` | 5 REST endpoints for quality monitoring |
| DDL | `sql/062_create_data_quality.sql` | `fact_dq_check_results`, `dim_dq_check_catalog`, `mv_dq_dashboard` |
| Config | `config/data_quality_config.yaml` | Rule definitions, thresholds, schedule |

## DQEngine (`common/dq_engine.py`)

- `DQEngine` class with `run_checks(dataset: str) -> list[CheckResult]`
- Built-in check types: `not_null`, `unique`, `range`, `referential_integrity`, `freshness`
- Each check returns `CheckResult(check_id, dataset, column, status, score, details, checked_at)`
- Score: 0.0 (fail) to 1.0 (pass), with configurable warning threshold (default 0.8)
- Batch execution: run all checks for a dataset or all datasets in one call
- Results written to `fact_dq_check_results`; catalog of available checks in `dim_dq_check_catalog`

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/data-quality/dashboard` | Aggregated quality scores from `mv_dq_dashboard` |
| GET | `/data-quality/checks` | List check catalog with last-run status |
| POST | `/data-quality/run` | Trigger check execution for a dataset (returns 202) |
| GET | `/data-quality/results` | Historical check results with filters (dataset, status, date range) |
| GET | `/data-quality/trends` | Quality score trend over time by dataset |

## Database Schema

### `dim_dq_check_catalog`
- `check_id SERIAL PRIMARY KEY`
- `check_name TEXT NOT NULL`
- `check_type TEXT NOT NULL` (not_null, unique, range, referential_integrity, freshness)
- `dataset TEXT NOT NULL`, `column_name TEXT`
- `params JSONB` (type-specific parameters: min/max, ref_table, max_age_hours)
- `severity TEXT DEFAULT 'warning'` (info, warning, critical)
- `is_active BOOLEAN DEFAULT true`
- `created_at TIMESTAMPTZ DEFAULT NOW()`

### `fact_dq_check_results`
- `result_id BIGSERIAL PRIMARY KEY`
- `check_id INTEGER REFERENCES dim_dq_check_catalog(check_id)`
- `status TEXT NOT NULL` (pass, warn, fail)
- `score NUMERIC(5,4)`, `details JSONB`
- `rows_checked BIGINT`, `rows_failed BIGINT`
- `checked_at TIMESTAMPTZ DEFAULT NOW()`
- Indexes: `(check_id, checked_at DESC)`, `(status)`

### `mv_dq_dashboard`
- Materialized view aggregating latest check results per dataset
- Columns: `dataset`, `total_checks`, `passed`, `warned`, `failed`, `overall_score`, `last_checked`

## Config (`config/data_quality_config.yaml`)

```yaml
checks:
  - name: sales_not_null_qty
    type: not_null
    dataset: sales
    column: qty
    severity: critical
  - name: forecast_range_basefcst
    type: range
    dataset: forecast
    column: basefcst_pref
    params: { min: 0, max: 1000000 }
    severity: warning
  - name: inventory_freshness
    type: freshness
    dataset: inventory
    params: { max_age_hours: 48 }
    severity: critical
thresholds:
  warning: 0.8
  critical: 0.5
schedule:
  cron: "0 6 * * *"  # Daily at 6 AM
```

## Make Targets

```bash
make dq-schema          # Apply DDL (one-time)
make dq-run             # Run all data quality checks
make dq-refresh         # Refresh mv_dq_dashboard
make dq-all             # dq-schema + dq-run + dq-refresh
```

## Dependencies

- PostgreSQL 16 (existing)
- `common/db.py` for `get_db_params()`
- APScheduler integration via `common/job_registry.py` (optional scheduled runs)
