# 07-02 Data Quality & Pipeline Observability

## EPIC
Platform Integration

## Status
Implemented

## Priority
P2 — Should Have

## Effort
M (Medium)

## Expert Perspectives
- **Data Engineering Expert** (lead) — check types, SQL-based validation, pipeline freshness
- **UI/UX Expert** — dashboard layout, score visualization, check catalog table
- **Platform Architect** — scheduling strategy, config-driven rule definitions

---

## Overview

Demand Studio manages 8 domain datasets (5 dimensions + 2 fact tables + inventory) totaling hundreds of millions of rows. Without automated data quality monitoring, issues like stale loads, null keys, duplicate records, out-of-range values, and broken foreign keys go undetected until they corrupt downstream analytics (accuracy metrics, safety stock computations, exception queues).

The Data Quality Engine provides a config-driven framework that:

1. Defines checks declaratively in YAML (no code changes to add new checks)
2. Runs 12 check types against any table/column combination (including statistical outlier, distribution drift, temporal gaps, cross-column consistency, cardinality anomaly)
3. Records every result in `fact_dq_check_results` for trend analysis
4. Exposes per-domain health scores and pipeline freshness via REST API
5. Surfaces issues in a dedicated Data Quality tab with AI-powered issue analysis and severity filtering
6. Provides statistical auto-fix capabilities (Winsorization, median imputation, per-item median replacement) via CLI and API

---

## Architecture

```
config/data_quality_config.yaml
         │
         ▼
  common/dq_engine.py (DQEngine)
    │ run_all_checks(domain?)
    │   ├─ _flatten_checks()    → config → flat list of check dicts
    │   ├─ _run_single(conn, check) → dispatch to check function
    │   └─ _record_result(conn, result) → INSERT into fact_dq_check_results
    │
    ▼
  PostgreSQL 16
    ├── dim_dq_check_catalog      (check definitions)
    ├── fact_dq_check_results     (execution results)
    └── mv_dq_dashboard           (aggregated pass/fail/warn by domain+day)
    │
    ▼
  api/routers/data_quality.py
    ├── GET  /data-quality/dashboard   → domain health scores
    ├── GET  /data-quality/checks      → catalog + last-run status
    ├── GET  /data-quality/history     → historical results
    ├── POST /data-quality/run         → trigger ad-hoc run
    ├── GET  /data-quality/freshness   → per-table last-load timestamps
    ├── POST /data-quality/fix         → statistical auto-fix (dry-run/apply)
    ├── GET  /data-quality/fix/preview → indexed preview of all fixable issues
    └── POST /data-quality/fix/apply   → apply selected fixes by preview IDs
    │
    ▼
  frontend/src/tabs/DataQualityTab.tsx
    ├── Domain health score cards (4-column grid)
    ├── Recent Issues panel (severity filter, check type badges)
    ├── Self-Heal panel (scan → preview → bulk/single accept/reject)
    ├── Pipeline freshness panel (per-table last-load)
    ├── Check catalog table (scrollable, last-run status)
    └── "Run Checks Now" button (mutation → invalidates all queries)
```

---

## Check Types

The engine supports 12 check types. Each is a pure function that receives a `psycopg` connection and check-specific parameters, returning a `{status, metric_value, details}` dict.

| # | Check Type | Function | Parameters | Pass Condition | Metric Value |
|---|---|---|---|---|---|
| 1 | `freshness` | `_check_freshness` | `table_name`, `max_hours` | `hours_since_load <= max_hours` | Hours since last `load_ts` (compared to planning date) |
| 2 | `completeness` | `_check_completeness` | `table_name`, `column`, `max_null_pct` | `null_pct <= max_null_pct` | Null percentage (0-100) |
| 3 | `uniqueness` | `_check_uniqueness` | `table_name`, `key_columns[]` | `duplicate_groups == 0` | Count of duplicate key groups |
| 4 | `row_count` | `_check_row_count` | `table_name`, `min_rows` | `count >= min_rows` | Row count |
| 5 | `range` | `_check_range` | `table_name`, `column`, `min`, `max` | `outliers == 0` | Count of out-of-range rows |
| 6 | `volume_delta` | `_check_volume_delta` | `table_name`, `max_pct_change` | `pct_change <= max_pct_change` | Absolute % change between last 2 loads |
| 7 | `referential_integrity` | `_check_referential_integrity` | `source_table`, `source_columns[]`, `target_table`, `target_columns[]` | `orphan_keys == 0` | Count of orphan FK values |
| 8 | `statistical_outlier` | `_check_statistical_outlier` | `table_name`, `column`, `method` (iqr/zscore), `threshold` | `outlier_pct <= 5%` | Count of statistical outliers |
| 9 | `distribution_drift` | `_check_distribution_drift` | `table_name`, `column`, `max_drift` | `drift <= max_drift` | Normalized drift score (0-1) |
| 10 | `temporal_gaps` | `_check_temporal_gaps` | `table_name`, `date_column`, `grain` (month/day) | `gap_count == 0` | Count of missing periods |
| 11 | `cross_column` | `_check_cross_column` | `table_name`, `rule` (SQL expression), `description` | `violations == 0` | Count of rows violating the rule |
| 12 | `cardinality_anomaly` | `_check_cardinality_anomaly` | `table_name`, `column`, `max_change_pct` | `change_pct <= max_change_pct` | Absolute % change in distinct values |

### Status Values

Each check returns one of:

| Status | Meaning |
|---|---|
| `pass` | Check passed — metric within threshold |
| `fail` | Check failed — metric exceeds threshold |
| `warn` | Indeterminate — empty table or zero-row edge case |
| `error` | Exception during check execution (caught, not re-raised) |
| `skip` | Check not applicable (no min/max specified, fewer than 2 loads) |

### Check Type Details

**Freshness** — Queries `max(load_ts)` from the target table. Computes hours elapsed since that timestamp relative to the **planning date** (`get_planning_date()` from `common/planning_date.py`, frozen at 2026-02-24), not wall-clock time. This ensures consistent results in development. Tables without any `load_ts` values immediately fail with `"No load_ts found"`.

**Completeness** — Counts total rows and null rows for the specified column using `count(*) FILTER (WHERE column IS NULL)`. Computes null percentage. Empty tables return `warn` status.

**Uniqueness** — Groups by the specified key columns and counts groups with `HAVING count(*) > 1`. Zero duplicates = pass.

**Row Count** — Simple `count(*)` against minimum threshold. Default `min_rows=1`.

**Range** — Counts rows where the column value is below `min` or above `max` (either bound is optional). Only checks non-null rows. Returns outlier count and percentage.

**Volume Delta** — Groups rows by `load_ts::date`, takes the two most recent load dates, and computes the absolute percentage change in row count. Requires at least 2 distinct load dates; otherwise returns `skip`.

**Referential Integrity** — For single-column FKs: LEFT JOIN + NULL check on target. For multi-column FKs: EXCEPT query between DISTINCT source keys and DISTINCT target keys. Counts orphan key combinations.

**Statistical Outlier** — Computes statistical bounds using IQR or Z-score method. IQR: `lower = Q1 - threshold * IQR`, `upper = Q3 + threshold * IQR`. Z-score: `lower = mean - threshold * stddev`, `upper = mean + threshold * stddev`. Counts rows outside bounds. Status: pass (0 outliers), warn (≤5% outliers or empty table), fail (>5% outliers).

**Distribution Drift** — Compares distribution statistics (mean, stddev, median) between the two most recent load batches (`GROUP BY load_ts::date`). Computes a normalized drift score as the max of mean shift / overall stddev, stddev ratio deviation, and median shift / overall stddev. Requires at least 2 distinct load dates; otherwise returns `skip`.

**Temporal Gaps** — Queries distinct periods (`date_trunc(grain, date_column)`) from the table, sorted ascending. Walks expected periods using `dateutil.relativedelta` and identifies missing ones. Status: pass (0 gaps), warn (≤3 gaps), fail (>3 gaps). Reports first 20 missing periods.

**Cross Column** — Evaluates a SQL boolean expression per row: `count(*) FILTER (WHERE NOT (expression))`. Counts violations. Status: pass (0 violations), warn (≤100 violations or empty table), fail (>100 violations). Details include the rule expression and description.

**Cardinality Anomaly** — Counts distinct values in the column for the two most recent load batches. Computes absolute percentage change. Requires at least 2 distinct load dates; otherwise returns `skip`. Reports new and dropped values (first 20 each).

---

## Statistical Auto-Fix

### `mvp/demand/scripts/fix_dq_issues.py`

A statistical remediation script that applies safe, data-driven fixes for common DQ issues. Supports dry-run mode (default) to preview changes before applying.

### Fix Strategies

| # | Fix Type | Strategy | Details |
|---|---|---|---|
| 1 | `range` | Winsorization | Clamp out-of-range values to configured min/max bounds from YAML |
| 2 | `lead_time` | Per-item median | Replace extreme `lead_time_days` (< 0 or > 730) with per-item median via CTE; fallback to global median |
| 3 | `completeness` | Statistical imputation | Numeric NULLs → column median; categorical NULLs → column mode. Skips PK columns (threshold=0) |
| 4 | `orphans` | Report only | Non-destructive — reports orphan FK counts and recommends `make normalize-all && make load-all` |
| 5 | `outliers` | Statistical Winsorization | Compute IQR/Z-score bounds from data; clamp values outside to computed bounds |

### API Endpoint

`POST /data-quality/fix` — Triggers auto-fix with dry-run/apply control.

| Param | Type | Default | Description |
|---|---|---|---|
| `fix_type` | string | `""` | Specific fix type (range, lead_time, completeness, orphans, outliers) |
| `apply` | bool | `false` | Set true to apply fixes; false for dry-run preview |

### Self-Heal Preview/Apply Endpoints

Two additional endpoints support the interactive Self-Heal workflow where users preview all fixable issues and selectively accept or reject each fix:

#### `GET /data-quality/fix/preview`

Returns an indexed list of all fixable issues (dry-run only). Each item has a unique `id` for selective application.

| Param | Type | Default | Description |
|---|---|---|---|
| `fix_type` | string | `""` | Filter by fix type (range, lead_time, completeness, orphans, outliers) |

Response: `{ "items": [{ "id": 0, "fix_type": "range", "description": "...", "affected_rows": 500, "recommendation": null, "status": "pending" }], "total": N }`

#### `POST /data-quality/fix/apply`

Apply selected fixes by their preview IDs. Only the specified fixes are executed; all others are skipped.

| Body Field | Type | Required | Description |
|---|---|---|---|
| `fix_ids` | `list[int]` | Yes | Array of fix IDs from the preview response |

Response: `{ "applied": [...], "skipped": [...], "total_applied": N, "total_skipped": N, "total_rows_fixed": N }`

#### Supporting Functions

- `preview_all_fixes(fix_type=None)` — Runs all fix strategies in dry-run mode, assigns sequential IDs, returns indexed list
- `apply_selected_fixes(fix_ids)` — Re-runs preview, partitions by selected IDs, applies only selected fixes in a single transaction

### CLI Usage

```bash
uv run python scripts/fix_dq_issues.py                    # Preview all fixes (dry-run)
uv run python scripts/fix_dq_issues.py --apply             # Apply all fixes
uv run python scripts/fix_dq_issues.py --fix range         # Fix only range issues
uv run python scripts/fix_dq_issues.py --fix lead_time     # Fix only lead time
uv run python scripts/fix_dq_issues.py --fix completeness  # Fix only NULLs
uv run python scripts/fix_dq_issues.py --fix orphans       # Quarantine orphan keys
uv run python scripts/fix_dq_issues.py --fix outliers      # Winsorise statistical outliers
```

---

## Self-Heal UI Panel

The DataQualityTab includes an interactive Self-Heal panel (Section 4) that provides a user-driven fix workflow:

### Workflow

1. User clicks "Scan for Fixes" button to open the panel
2. Panel fetches `GET /data-quality/fix/preview` (TanStack Query with `enabled: healOpen`)
3. Fix items displayed as cards with: fix_type badge (color-coded), description, affected rows count, recommendation (if any)
4. User can select/deselect individual fixes via checkboxes, or use bulk actions
5. "Accept Selected" applies chosen fixes via `POST /data-quality/fix/apply`
6. "Reject Selected" marks fixes as rejected (client-side only, no DB write)
7. Success banner shows total applied count and rows fixed

### Toolbar Actions

| Button | Action |
|---|---|
| Select All | Check all pending fix items |
| Deselect All | Uncheck all fix items |
| Accept Selected | POST selected fix IDs to `/fix/apply`, mark as applied (green) |
| Reject Selected | Mark selected fixes as rejected (dimmed, strikethrough) |
| Reset | Clear all applied/rejected state, re-fetch preview |

### Fix Item States

| State | Visual |
|---|---|
| Pending | Default card with checkbox |
| Applied | Green background, check icon, "Applied" badge |
| Rejected | Dimmed/muted, X icon, "Rejected" badge |

### Single-Item Actions

Each fix item also has individual Accept/Reject buttons for one-at-a-time workflows.

---

## Database Schema

### DDL: `mvp/demand/sql/063_create_data_quality.sql`

### `dim_dq_check_catalog`

Reusable check definitions. Each row describes one configurable check.

```sql
CREATE TABLE IF NOT EXISTS dim_dq_check_catalog (
    check_id    SERIAL PRIMARY KEY,
    check_name  TEXT NOT NULL UNIQUE,
    check_type  TEXT NOT NULL,          -- freshness | completeness | uniqueness |
                                        -- range | volume_delta | referential_integrity
    domain      TEXT NOT NULL,          -- item | location | customer | time | dfu |
                                        -- sales | forecast | inventory
    sql_template TEXT,                  -- parameterized SQL (reserved for future use)
    threshold   NUMERIC,               -- pass/fail threshold
    severity    TEXT NOT NULL DEFAULT 'warning',  -- info | warning | critical
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Indexes:**
- `idx_dq_catalog_domain` — B-tree on `domain`
- `idx_dq_catalog_type` — B-tree on `check_type`
- `idx_dq_catalog_enabled` — Partial index `WHERE enabled = TRUE`

### `fact_dq_check_results`

Every check execution writes one row here. This is the primary audit trail.

```sql
CREATE TABLE IF NOT EXISTS fact_dq_check_results (
    check_id      BIGSERIAL PRIMARY KEY,
    check_name    TEXT NOT NULL,
    domain        TEXT NOT NULL,
    table_name    TEXT NOT NULL,
    severity      TEXT NOT NULL DEFAULT 'warning',
    status        TEXT NOT NULL DEFAULT 'pass',   -- pass | fail | warn | error
    metric_value  NUMERIC,
    threshold     NUMERIC,
    details       JSONB,
    run_ts        TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Indexes:**
- `idx_dq_results_run_ts` — B-tree on `run_ts DESC` (recent results first)
- `idx_dq_results_domain` — B-tree on `domain`
- `idx_dq_results_status` — B-tree on `status`
- `idx_dq_results_check_name` — B-tree on `check_name`
- `idx_dq_results_severity` — B-tree on `severity`
- `idx_dq_results_domain_ts` — Composite B-tree on `(domain, run_ts DESC)` (domain health score queries)

### `mv_dq_dashboard`

Materialized view aggregating pass/fail/warn/error counts by domain and calendar day.

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_dq_dashboard AS
SELECT
    domain,
    date_trunc('day', run_ts)::DATE AS run_date,
    COUNT(*) FILTER (WHERE status = 'pass')  AS pass_count,
    COUNT(*) FILTER (WHERE status = 'fail')  AS fail_count,
    COUNT(*) FILTER (WHERE status = 'warn')  AS warn_count,
    COUNT(*) FILTER (WHERE status = 'error') AS error_count,
    COUNT(*)                                  AS total_count
FROM fact_dq_check_results
GROUP BY domain, date_trunc('day', run_ts)::DATE;
```

**Indexes:**
- `idx_mv_dq_dashboard_pk` — Unique on `(domain, run_date)` (enables `REFRESH CONCURRENTLY`)
- `idx_mv_dq_dashboard_date` — B-tree on `run_date DESC`

---

## API Endpoints

**Router:** `mvp/demand/api/routers/data_quality.py`
**Prefix:** `/data-quality`
**Tag:** `data-quality`

All endpoints use `get_conn()` directly (consistent with the `inv_planning_*.py` pattern).

### GET `/data-quality/dashboard`

Domain health scores from the last 24 hours of check results.

**Response:**
```json
{
  "domains": [
    {
      "domain": "sales",
      "score": 80.0,
      "passed": 8,
      "failed": 1,
      "warnings": 1,
      "total": 10
    }
  ]
}
```

Score formula: `round(100.0 * passed / total, 1)`. Domains with no recent checks are omitted.

### GET `/data-quality/checks`

Full check catalog with last-run status via `LATERAL` subquery join.

**Response:**
```json
{
  "checks": [
    {
      "check_id": 1,
      "check_name": "freshness_sales",
      "check_type": "freshness",
      "domain": "sales",
      "table_name": "fact_sales_monthly",
      "severity": "critical",
      "enabled": true,
      "last_status": "pass",
      "last_value": 1.0,
      "last_run": "2026-03-01T12:00:00"
    }
  ]
}
```

### GET `/data-quality/history`

Historical check results with filtering and pagination.

**Query Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `domain` | string | `""` | Filter by domain (empty = all) |
| `days` | int | 7 | Lookback window (1-90) |
| `limit` | int | 100 | Max results (1-1000) |

**Response:**
```json
{
  "entries": [
    {
      "check_id": 1,
      "check_name": "freshness_sales",
      "domain": "sales",
      "table_name": "fact_sales_monthly",
      "severity": "critical",
      "status": "pass",
      "metric_value": 1.0,
      "details": {"hours_since_load": 1.0},
      "run_ts": "2026-03-01T12:00:00"
    }
  ]
}
```

### POST `/data-quality/run`

Trigger an ad-hoc check run. Requires `manager+` role and API key authentication.

**Query Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `domain` | string | `""` | Run checks for specific domain only (empty = all) |

**Auth:** `require_role("manager")` + `require_api_key`

**Response:**
```json
{
  "results": [
    {
      "check_name": "freshness_sales",
      "check_type": "freshness",
      "domain": "sales",
      "table_name": "fact_sales_monthly",
      "severity": "critical",
      "status": "pass",
      "metric_value": 1.23,
      "details": {"hours_since_load": 1.23}
    }
  ],
  "total": 42
}
```

### GET `/data-quality/freshness`

Per-table last-load timestamps for the 6 core tables.

**Tables checked:** `dim_item`, `dim_location`, `dim_customer`, `dim_dfu`, `fact_sales_monthly`, `fact_external_forecast_monthly`

**Response:**
```json
{
  "tables": [
    {"table": "dim_item", "last_load": "2026-03-01T06:00:00"},
    {"table": "dim_location", "last_load": "2026-03-01T06:00:00"},
    {"table": "fact_sales_monthly", "last_load": null}
  ]
}
```

Tables that error during the query return `last_load: null` (best-effort, non-fatal).

---

## Configuration Schema

### `mvp/demand/config/data_quality_config.yaml`

The config uses a nested structure: `checks.<check_type>.<domain_key>`. The `DQEngine._flatten_checks()` method normalizes this into a flat list of individual check dicts at runtime.

```yaml
schedule:
  cron: "0 */4 * * *"          # Run every 4 hours
  on_load: true                 # Also run after each data load

global_defaults:
  severity: warning
  enabled: true

checks:
  freshness:
    <domain_key>:
      table: <table_name>
      max_hours_since_load: <int>
      severity: <info|warning|critical>

  completeness:
    <domain_key>:
      table: <table_name>
      columns:
        - column: <column_name>
          null_pct_threshold: <float>
          severity: <info|warning|critical>

  uniqueness:
    <domain_key>:
      table: <table_name>
      key_columns: [<col1>, <col2>, ...]
      severity: <info|warning|critical>

  range:
    <domain_key>:
      table: <table_name>
      columns:
        - column: <column_name>
          min: <number|null>
          max: <number|null>
          severity: <info|warning|critical>

  volume_delta:
    <domain_key>:
      table: <table_name>
      max_pct_change: <float>
      severity: <info|warning|critical>

  referential_integrity:
    <domain_key>:
      source_table: <table_name>
      source_columns: [<col1>, <col2>, ...]
      target_table: <table_name>
      target_columns: [<col1>, <col2>, ...]
      severity: <info|warning|critical>
```

### Current Check Inventory

| Check Type | Domain Checks | Total Individual Checks |
|---|---|---|
| Freshness | 8 (item, location, customer, time, dfu, sales, forecast, inventory) | 8 |
| Completeness | 7 domains with 18 column checks total | 18 |
| Uniqueness | 7 (item, location, customer, dfu, sales, forecast, inventory) | 7 |
| Range | 3 domains with 6 column checks total | 6 |
| Volume Delta | 6 (sales, forecast, inventory, item, location, dfu) | 6 |
| Referential Integrity | 8 FK relationships | 8 |
| Statistical Outlier | 3 domains with 5 column checks | 5 |
| Distribution Drift | 3 domains with 4 column checks | 4 |
| Temporal Gaps | 2 (sales, forecast) | 2 |
| Cross Column | 2 domains with 5 rules total | 5 |
| Cardinality Anomaly | 3 domains with 4 column checks | 4 |
| **Total** | | **73** |

### Severity Distribution

| Severity | Typical Usage |
|---|---|
| `critical` | Null primary keys, stale fact tables (<48h), duplicate keys |
| `warning` | Null descriptive fields (>5-10%), range violations, volume spikes |
| `info` | Static table freshness (dim_time), informational FK checks |

### Freshness Thresholds

| Table Type | Max Hours | Rationale |
|---|---|---|
| Fact tables (sales, forecast) | 48 | Updated with each data load cycle |
| Inventory snapshot | 24 | Daily operational data |
| Dimension tables | 168 (7 days) | Master data, less frequent updates |
| Time dimension | 720 (30 days) | Static/generated, rarely changes |

---

## DQEngine Class

### `mvp/demand/common/dq_engine.py`

The `DQEngine` class is the core orchestrator. Key methods:

| Method | Description |
|---|---|
| `run_all_checks(domain=None)` | Run all enabled checks (optionally filtered by domain). Opens a single `psycopg.connect()` connection, runs checks sequentially, records each result, commits at the end. Returns list of result dicts. |
| `_flatten_checks()` | Normalizes the nested YAML config into a flat list of check dicts. Handles per-column expansion for `completeness` and `range` check types. Supports backward-compatible flat list format. |
| `_run_single(conn, check)` | Dispatches a single check to the appropriate check function. Catches all exceptions and returns `status="error"` on failure. |
| `_record_result(conn, result)` | INSERTs the result into `fact_dq_check_results`. Best-effort (silent exception catch). |
| `get_domain_score(domain)` | Queries last 24h results for a domain. Returns `{domain, score, total_checks, passed, failed, warnings}`. |
| `get_pipeline_health()` | Runs freshness checks against 5 core tables (`dim_item`, `dim_location`, `dim_dfu`, `fact_sales_monthly`, `fact_external_forecast_monthly`). Returns `{tables: [...]}`. |

### Config Loading

Uses `common.utils.load_config("data_quality_config.yaml")` — thread-safe, cached per-process. `_reset_config()` exposed for test isolation.

### Check Function Registry

```python
CHECK_FUNCTIONS = {
    "freshness": _check_freshness,
    "completeness": _check_completeness,
    "row_count": _check_row_count,
    "uniqueness": _check_uniqueness,
    "range": _check_range,
    "volume_delta": _check_volume_delta,
    "referential_integrity": _check_referential_integrity,
    "statistical_outlier": _check_statistical_outlier,
    "distribution_drift": _check_distribution_drift,
    "temporal_gaps": _check_temporal_gaps,
    "cross_column": _check_cross_column,
    "cardinality_anomaly": _check_cardinality_anomaly,
}
```

All 12 check types are fully dispatched in `_run_single()` with appropriate parameter extraction per type.

---

## Frontend Components

### `mvp/demand/frontend/src/tabs/DataQualityTab.tsx`

Three sections in a vertical `space-y-6` layout:

**1. Header + Action Button**
- Title: "Data Quality & Observability"
- Subtitle: "Monitor pipeline health and data quality across all domains"
- "Run Checks Now" button — triggers `POST /data-quality/run` via `useMutation`
- Button shows "Running..." and is disabled while mutation is pending
- On success, invalidates all 3 DQ query keys to force refetch

**2. Domain Health Score Cards** (responsive 2-col / 4-col grid)
- One card per domain with:
  - Domain name (capitalized)
  - Score badge: green (>=90%), amber (>=70%), red (<70%)
  - Pass/fail/warn counts in colored text
- Empty state: "No data quality checks have been run yet"

**3. Pipeline Freshness Panel** (bordered card)
- Lists 6 core tables with `font-mono` table names
- Shows `last_load` as localized datetime, or "Never loaded" in red

**4. Check Catalog Table** (bordered card, scrollable max-h-80)
- Columns: Check name, Domain, Type, Severity badge, Status, Last Run
- Severity badges: `critical` = red, others = amber
- Status coloring: `pass` = green, `fail` = red, others = muted

**5. Recent Issues Panel** (with severity filter)
- Shows non-pass entries from history, sorted by most recent
- Severity dropdown filter (All / Critical / Warning / Info)
- Each issue includes AI-generated analysis:
  - Summary: human-readable description of what was detected
  - Root cause: probable explanation based on check type and metric values
  - Fix steps: SQL investigation queries and auto-fix commands
- `analyzeIssue()` function handles all 12 check types with type-specific analysis

### TypeScript Types

Defined in `frontend/src/api/queries/platform.ts`:

```typescript
interface DQDomainScore {
  domain: string;
  score: number;
  passed: number;
  failed: number;
  warnings: number;
  total: number;
}

interface DQCheck {
  check_id: number;
  check_name: string;
  check_type: string;
  domain: string;
  table_name: string;
  severity: string;
  enabled: boolean;
  last_status: string | null;
  last_value: number | null;
  last_run: string | null;
}
```

### Query Keys & Fetch Functions

```typescript
export const dqKeys = {
  dashboard: ["dq", "dashboard"],
  checks: ["dq", "checks"],
  freshness: ["dq", "freshness"],
};

fetchDQDashboard()  → GET /data-quality/dashboard
fetchDQChecks()     → GET /data-quality/checks
fetchDQFreshness()  → GET /data-quality/freshness
runDQChecks(domain?) → POST /data-quality/run
```

Stale time: `STALE_PLATFORM = 5 * 60 * 1000` (5 minutes).

---

## Scheduling Strategy

### Cron: Every 4 Hours

```yaml
schedule:
  cron: "0 */4 * * *"
```

The DQ engine is designed to run as a scheduled job via the APScheduler job registry (`common/job_registry.py`). The cron expression triggers a full check run 6 times per day.

### On-Load Trigger

```yaml
schedule:
  on_load: true
```

When `on_load: true`, data pipeline scripts (normalize + load) can invoke `DQEngine().run_all_checks()` at the end of each load cycle to validate freshly loaded data.

### Ad-Hoc Runs

The `POST /data-quality/run` endpoint allows managers to trigger checks on demand. The frontend "Run Checks Now" button uses this endpoint. The run is synchronous — all checks execute within the HTTP request and results are returned immediately.

---

## Makefile Targets

```makefile
dq-schema:            # Apply sql/063_create_data_quality.sql (one-time)
dq-run:               # Run all data quality checks (DQEngine.run_all_checks)
dq-refresh:           # REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dq_dashboard
dq-all:               # dq-schema + dq-run + dq-refresh (full pipeline)
```

---

## Testing Strategy

### Backend API Tests: `mvp/demand/tests/api/test_data_quality.py`

11 tests covering all 5 endpoints:

| Test | Endpoint | Assertion |
|---|---|---|
| `test_dq_dashboard_200` | GET /dashboard | 2 domains returned, score=80.0 computed correctly |
| `test_dq_dashboard_empty` | GET /dashboard | Empty domains list when no results |
| `test_dq_checks_200` | GET /checks | 2 checks with correct fields (name, type, severity, enabled, last_status) |
| `test_dq_checks_null_metric` | GET /checks | Null metric_value and run_ts returned as None |
| `test_dq_history_200` | GET /history | Single entry with correct check_name and status |
| `test_dq_history_with_domain_filter` | GET /history | Domain param included in SQL params list |
| `test_dq_history_invalid_days` | GET /history | days=0 returns 422 validation error |
| `test_dq_freshness_200` | GET /freshness | 6 tables returned, first has last_load set |
| `test_dq_freshness_no_data` | GET /freshness | All tables return last_load=None |
| `test_dq_run_200` | POST /run | DQEngine mocked, returns results with total=1 |
| `test_dq_run_with_domain` | POST /run?domain=sales | Engine called with domain="sales" |

Test pattern: `httpx.AsyncClient(transport=ASGITransport(app))` with `_make_pool()` factory from `tests/api/conftest.py`.

### Backend Unit Tests: `mvp/demand/tests/unit/test_dq_engine.py`

103 tests covering all 12 check types, flatten logic, dispatch routing, domain scoring, and pipeline health. All freshness tests mock `get_planning_date()`.

### Frontend Component Tests: `mvp/demand/frontend/src/tabs/__tests__/DataQualityTab.test.tsx`

20 tests covering rendering, interaction, severity filtering, and AI analysis:

| Test | Assertion |
|---|---|
| Renders without crashing | "Data Quality & Observability" text present |
| Shows empty state | "No data quality checks have been run yet" message |
| Renders Pipeline Freshness section | Section header present |
| Renders Check Catalog section | "Check Catalog" text present |
| Renders Run Checks Now button | Button role found |
| Calls runDQChecks on click | Mock called once |
| Shows loading state | Button text changes to "Running...", button disabled |
| Invalidates queries on success | All 3 fetch functions re-called after mutation |

Test pattern: `TestQueryWrapper` + `vi.mock("@/api/queries")` for API layer.

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| PostgreSQL 16 | Existing | All check queries run against existing tables |
| `common/db.py` | Existing | `get_db_params()` for engine connections |
| `common/utils.py` | Existing | `load_config()` for thread-safe YAML loading |
| `common/auth.py` | Existing | `require_role("manager")` for POST /run |
| `api/auth.py` | Existing | `require_api_key` for POST /run |
| `@tanstack/react-query` | Existing | `useQuery` + `useMutation` in frontend |

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/063_create_data_quality.sql` | Create — DDL for 2 tables + 1 materialized view |
| `mvp/demand/common/dq_engine.py` | Create — DQEngine class + 12 check functions; freshness uses planning date |
| `mvp/demand/scripts/fix_dq_issues.py` | Create — Statistical auto-fix script with 5 strategies + CLI + API integration |
| `mvp/demand/config/data_quality_config.yaml` | Create — 73 check definitions across 12 types |
| `mvp/demand/api/routers/data_quality.py` | Create — 5 REST endpoints |
| `mvp/demand/frontend/src/tabs/DataQualityTab.tsx` | Create — Data Quality dashboard tab |
| `mvp/demand/frontend/src/api/queries/platform.ts` | Modify — DQ types, query keys, fetch functions |
| `mvp/demand/tests/api/test_data_quality.py` | Create — 11 API tests |
| `mvp/demand/frontend/src/tabs/__tests__/DataQualityTab.test.tsx` | Create — 8 component tests |
| `mvp/demand/api/main.py` | Modify — mount `data_quality.router` |
| `mvp/demand/frontend/src/components/AppSidebar.tsx` | Modify — add Data Quality nav item |
| `mvp/demand/frontend/vite.config.ts` | Modify — add `/data-quality` proxy entry |
| `docs/specs/07-platform-integration/07-02-data-quality.md` | Create (this file) |
