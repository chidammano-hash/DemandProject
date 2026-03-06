# IPfeature2 — Lead Time Variability Profiling

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P1 — Must Have (Foundation)

## Effort
M (Medium)

## Expert Perspectives
- **Statistical Analyst** (lead) — LT distribution fitting, change-point detection
- **Inventory Planning Expert** — σ_LT role in safety stock formula
- **Supply Chain Control Tower Expert** — supplier unreliability visibility

---

## Problem Statement

The existing system stores only `latest_lead_time_days` — a single scalar point-in-time value per item-location in `agg_inventory_monthly`. There is no history of lead time variation.

The combined safety stock formula requires **both** demand sigma AND lead time sigma:

```
SS_combined = Z × sqrt(LT_mean × σ_D² + D̄² × σ_LT²)
```

Without σ_LT, the second term is zero. Safety stock will be systematically **understated** for all items with unreliable suppliers — the items that need the most protection. This is a fundamental data gap that must be filled before IPfeature3 can produce correct results.

---

## User Story

> As an inventory planner, I want each item-location's lead time distribution (mean, std, CV, min, max, p95) derived from the daily inventory snapshot history, so that I can compute statistically correct safety stock that accounts for supplier delivery unreliability — not just average performance.

---

## Business Value

- Enables the full combined SS formula in IPfeature3
- Quantifies the "SS cost of supplier unreliability" per IPfeature12
- Identifies which suppliers' LT variance is driving excess safety stock
- Provides data for supplier performance negotiations

---

## Data Requirements

### New DDL: `mvp/demand/sql/023_create_lead_time_profile.sql`

New table `dim_item_lead_time_profile`:

```sql
CREATE TABLE IF NOT EXISTS dim_item_lead_time_profile (
    lt_profile_sk        BIGSERIAL PRIMARY KEY,
    item_no              TEXT NOT NULL,
    loc                  TEXT NOT NULL,
    as_of_date           DATE NOT NULL,
    lt_mean_days         NUMERIC(10,2),
    lt_std_days          NUMERIC(10,2),
    lt_cv                NUMERIC(10,4),
    lt_min_days          NUMERIC(10,2),
    lt_max_days          NUMERIC(10,2),
    lt_p25_days          NUMERIC(10,2),
    lt_p50_days          NUMERIC(10,2),
    lt_p75_days          NUMERIC(10,2),
    lt_p95_days          NUMERIC(10,2),
    observation_months   INTEGER,
    lt_variability_class TEXT,        -- 'stable' | 'moderate' | 'volatile'
    load_ts              TIMESTAMPTZ DEFAULT NOW(),
    modified_ts          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (item_no, loc)
);
CREATE INDEX IF NOT EXISTS idx_lt_profile_item_loc ON dim_item_lead_time_profile (item_no, loc);
CREATE INDEX IF NOT EXISTS idx_lt_profile_class ON dim_item_lead_time_profile (lt_variability_class);
```

### Classification Rules

```
lt_variability_class:
  stable    → lt_cv < 0.15
  moderate  → lt_cv 0.15 – 0.40
  volatile  → lt_cv > 0.40
```

### Config Extension: `mvp/demand/config/variability_config.yaml`

Add to existing file:
```yaml
lead_time:
  history_months: 12
  min_observations: 3        # need ≥3 distinct LT values to compute std
  stable_cv_threshold: 0.15
  volatile_cv_threshold: 0.40
```

---

## Method: Change-Point Detection from Daily Snapshots

`fact_inventory_snapshot` contains `lead_time_days` for every daily row. When this value **changes** between consecutive daily snapshots for the same item-loc, it signals a new lead time observation (e.g., a new PO receipt with updated vendor LT).

**Algorithm:**
```
For each item_no + loc:
  1. ORDER BY snapshot_date ASC
  2. Detect rows where lead_time_days != lag(lead_time_days) OVER (...)
     → each such row = one LT observation
  3. Collect all distinct observed LT values
  4. If < min_observations → mark as insufficient, store only mean = latest_lt, all others NULL
  5. Compute: mean, std, cv, min, max, p25, p50, p75, p95
  6. Classify lt_variability_class
  7. Upsert into dim_item_lead_time_profile
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py` (extends IPfeature1's router)

```
GET /inv-planning/lead-time/profile
  Query params: item, location, lt_variability_class, limit (1–1000, default 50),
                offset, sort_by (lt_mean_days | lt_std_days | lt_cv), sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, lt_mean_days, lt_std_days, lt_cv, lt_min_days, lt_max_days,
             lt_p50_days, lt_p95_days, observation_months, lt_variability_class} ]
  }
  Cache: max-age=120s

GET /inv-planning/lead-time/summary
  Query params: item, location
  Response: {
    total_item_locs: int,
    with_lt_data: int,
    insufficient_data: int,
    by_class: { stable: int, moderate: int, volatile: int },
    avg_lt_mean_days: float,
    avg_lt_cv: float,
    most_volatile: [ {item_no, loc, lt_cv, lt_std_days, lt_mean_days} × 10 ]
  }
  Cache: max-age=300s

GET /inv-planning/lead-time/histogram
  Query params: metric (lt_mean_days | lt_std_days | lt_cv), bins (5–50, default 20)
  Response: {
    metric: str,
    buckets: [ {bin_start, bin_end, count} ]
  }
  Cache: max-age=300s
```

---

## Frontend UI

### Location
New collapsible card "Lead Time Profile" in `mvp/demand/frontend/src/tabs/InventoryTab.tsx`

### Components

**1. LT Variability Class Donut**
- 3 segments: stable (green) / moderate (yellow) / volatile (red)
- + insufficient data (gray) for items with < min_observations

**2. LT Mean vs. LT Std Scatter Plot**
- X-axis: `lt_mean_days`
- Y-axis: `lt_std_days`
- Point color: `lt_variability_class` (stable=green, moderate=yellow, volatile=red)
- Tooltip: item, loc, lt_mean, lt_std, lt_cv, lt_variability_class

**3. Top-10 Most Unreliable Item-Locs Table**
- Columns: item, loc, lt_mean_days, lt_std_days, lt_cv, lt_p95_days, lt_variability_class
- Sorted: lt_cv descending

### Item Detail Drill-Down Extension
When a row is clicked in the main inventory position table, the existing item-detail panel now also shows:
- `lt_mean_days` ± `lt_std_days` (e.g., "14.2 ± 3.1 days")
- `lt_p95_days` (95th percentile worst-case LT)
- `lt_variability_class` badge (color-coded)

---

## Backend Script

### `mvp/demand/scripts/compute_lead_time_variability.py`

```python
# Pseudocode:
# 1. Query: SELECT item_no, loc, snapshot_date, lead_time_days
#    FROM fact_inventory_snapshot
#    WHERE snapshot_date >= CURRENT_DATE - INTERVAL '{history_months} months'
#    ORDER BY item_no, loc, snapshot_date

# 2. For each (item_no, loc) group:
#    a. Collect series of lead_time_days values
#    b. Detect change-points: index where value differs from previous
#    c. observed_lt_values = [lt at each change-point]
#    d. IF len(observed_lt_values) < min_observations:
#         lt_mean = series[-1] (latest value), all std/cv = NULL
#         lt_variability_class = NULL
#         observation_months = len(observed_lt_values)
#       ELSE:
#         lt_mean = mean(observed_lt_values)
#         lt_std  = std(observed_lt_values)
#         lt_cv   = lt_std / lt_mean if lt_mean > 0 else NULL
#         lt_min, lt_max, p25, p50, p75, p95 = percentiles
#         classify lt_variability_class

# 3. Batch upsert into dim_item_lead_time_profile ON CONFLICT (item_no, loc) DO UPDATE
```

**CLI Usage:**
```bash
uv run python scripts/compute_lead_time_variability.py
uv run python scripts/compute_lead_time_variability.py --history-months 12
```

---

## Makefile Targets

```makefile
lt-profile-schema:
	# apply sql/023_create_lead_time_profile.sql

lt-profile-compute:
	uv run python scripts/compute_lead_time_variability.py

lt-profile-all: lt-profile-schema lt-profile-compute
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_inventory_snapshot` | Existing table | 190M rows — query with date filter |
| `dim_item` | Existing table | For supplier join in IPfeature12 |
| `config/variability_config.yaml` | New (IPfeature1) | Shares config file, adds lead_time section |
| IPfeature1 | Prior IPfeature | Shares variability_config.yaml structure |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_lead_time_variability.py`

Minimum 12 tests:
- Change-point detection: series [7,7,7,10,10,14] → 3 observations [7,10,14]
- Single value series: `observation_months=1 < min_observations` → NULL std/cv
- LT std formula verification: known series → expected std
- CV classification: cv=0.10→stable, cv=0.25→moderate, cv=0.55→volatile
- Boundary: cv exactly 0.15 → stable (not moderate)
- Boundary: cv exactly 0.40 → moderate (not volatile)
- All same value (zero variance): std=0, cv=0, class=stable
- NULL lead_time_days rows: dropped before computation
- p95 ≥ p75 ≥ p50 ≥ p25 ≥ lt_min always
- lt_cv = lt_std / lt_mean verified numerically

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_lead_time.py`

Minimum 8 tests:
- `GET /inv-planning/lead-time/summary` → 200 OK, by_class has 3 keys
- `GET /inv-planning/lead-time/profile` → 200 OK, rows with lt_mean_days
- `GET /inv-planning/lead-time/profile?lt_variability_class=volatile` → filtered
- `GET /inv-planning/lead-time/histogram?metric=lt_cv` → buckets list
- `GET /inv-planning/lead-time/histogram?metric=bad_metric` → 422
- Pagination: limit=5 returns ≤5 rows with total count
- Sort by lt_cv desc → first row has highest CV

### Frontend Tests
- Extend `InventoryTab.test.tsx` — LT profile card renders
- LT donut shows stable/moderate/volatile segments
- Item detail drill-down shows lt_mean ± lt_std

---

## Acceptance Criteria

- [ ] `dim_item_lead_time_profile` populated for all item-locs with ≥3 distinct LT observations
- [ ] Items with < 3 observations have `lt_std_days = NULL`, `lt_variability_class = NULL`
- [ ] `lt_cv = lt_std_days / lt_mean_days` verified numerically for known cases
- [ ] LT CV and variability class visible in InventoryTab item drill-down
- [ ] `GET /inv-planning/lead-time/summary` returns non-zero `with_lt_data` count
- [ ] `make test-all` passes with no regressions

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/023_create_lead_time_profile.sql` | Create |
| `mvp/demand/config/variability_config.yaml` | Modify — add lead_time section |
| `mvp/demand/scripts/compute_lead_time_variability.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add lead-time endpoints |
| `mvp/demand/frontend/src/tabs/InventoryTab.tsx` | Modify — add LT profile panel + drill-down |
| `mvp/demand/tests/unit/test_lead_time_variability.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_lead_time.py` | Create |
| `mvp/demand/Makefile` | Modify — add lt-profile-* targets |
| `docs/design-specs/IPfeature2.md` | Create (this file) |
