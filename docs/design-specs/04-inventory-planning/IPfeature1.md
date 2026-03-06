# IPfeature1 — Demand Variability & Statistical Profiling Engine

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P1 — Must Have (Foundation)

## Effort
M (Medium)

## Expert Perspectives
- **Statistical Analyst** (lead) — CV computation, outlier capping, distribution shape
- **Inventory Planning Expert** — variability classification thresholds, downstream SS use
- **Demand Planning Expert** — intermittency detection, lumpy demand identification

---

## Problem Statement

The current system classifies DFUs only by demand **volume** (`abc_vol`: A/B/C from the source system). It has zero statistical spread measures — no standard deviation, no coefficient of variation, no intermittency ratio.

Safety stock math is mathematically impossible without σ_D (standard deviation of demand) and CV (coefficient of variation). Every downstream IPfeature (safety stock, EOQ, ABC-XYZ matrix, simulation) depends on this statistical foundation.

---

## User Story

> As an inventory planner, I want each SKU-location's demand variability profile — CV, sigma, percentiles, skewness, and intermittency ratio — computed from sales history, so that I understand the true uncertainty I must protect against before setting safety stock targets.

---

## Business Value

- Enables IPfeature3 (Safety Stock Engine), IPfeature10 (Monte Carlo), IPfeature11 (ABC-XYZ)
- Identifies lumpy/intermittent demand items that need special handling (Croston's method)
- Surfaces high-CV items that carry disproportionate inventory risk
- Replaces gut-feel variability assessment with statistically defensible numbers

---

## Data Requirements

### New DDL: `mvp/demand/sql/022_add_demand_variability_columns.sql`

Adds the following columns to `dim_dfu`:

| Column | Type | Description |
|---|---|---|
| `demand_mean` | NUMERIC(15,4) | Mean monthly demand over history window |
| `demand_std` | NUMERIC(15,4) | Standard deviation of monthly demand |
| `demand_cv` | NUMERIC(10,4) | Coefficient of variation = std / mean |
| `demand_mad` | NUMERIC(15,4) | Median absolute deviation (robust σ) |
| `demand_p50` | NUMERIC(15,4) | Median monthly demand |
| `demand_p90` | NUMERIC(15,4) | 90th percentile monthly demand |
| `demand_skewness` | NUMERIC(10,4) | Distribution skewness |
| `demand_kurtosis` | NUMERIC(10,4) | Distribution kurtosis (excess) |
| `zero_demand_months` | INTEGER | Count of months with zero sales |
| `total_demand_months` | INTEGER | Total months with any history |
| `intermittency_ratio` | NUMERIC(10,4) | zero_demand_months / total_demand_months |
| `variability_class` | TEXT | `'low'` / `'medium'` / `'high'` / `'lumpy'` |
| `demand_profile_ts` | TIMESTAMPTZ | Timestamp of last computation |

### Classification Rules

```
variability_class:
  low    → demand_cv < 0.3
  medium → demand_cv 0.3 – 0.8
  high   → demand_cv 0.8 – 1.5
  lumpy  → demand_cv > 1.5  OR  intermittency_ratio > 0.30
```

### New Config: `mvp/demand/config/variability_config.yaml`

```yaml
variability:
  history_months: 24
  outlier_sigma_threshold: 3       # winsorize at mean ± 3σ before computing std
  cv_thresholds:
    low: 0.3
    medium: 0.8
    high: 1.5                      # above = lumpy/intermittent
  intermittency_threshold: 0.30    # >30% zero months = lumpy
  min_nonzero_months: 6            # skip DFU if fewer than this
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py` (new file, mounted at `/inv-planning`)

```
GET /inv-planning/variability/summary
  Query params: cluster_assignment, abc_vol, variability_class, item, location
  Response: {
    total_dfus: int,
    by_class: { low: int, medium: int, high: int, lumpy: int },
    cv_percentiles: { p25, p50, p75, p90 },
    top_volatile: [ {item_no, loc, demand_cv, demand_std, variability_class} × 10 ]
  }
  Cache: max-age=300s

GET /inv-planning/variability/detail
  Query params: item, location, variability_class, abc_vol, cluster_assignment,
                limit (1–1000, default 50), offset, sort_by, sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, demand_mean, demand_std, demand_cv, demand_mad,
             demand_p50, demand_p90, demand_skewness, demand_kurtosis,
             zero_demand_months, total_demand_months, intermittency_ratio,
             variability_class, demand_profile_ts} ]
  }
  Cache: max-age=120s

GET /inv-planning/variability/histogram
  Query params: metric (demand_cv | demand_std | demand_mean), bins (5–50, default 20)
  Response: {
    metric: str,
    buckets: [ {bin_start, bin_end, count} ]
  }
  Cache: max-age=300s
```

**Vite proxy:** Add `/inv-planning` entry to `mvp/demand/frontend/vite.config.ts`

---

## Frontend UI

### Location
New collapsible card "Demand Variability Profile" appended to `mvp/demand/frontend/src/tabs/InventoryTab.tsx`

### Components

**1. Variability Class Donut Chart**
- 4 segments: low (green) / medium (yellow) / high (orange) / lumpy (red)
- Counts from `GET /inv-planning/variability/summary`

**2. CV vs. Mean Demand Scatter Plot**
- X-axis: `demand_cv` (coefficient of variation)
- Y-axis: `demand_mean` (avg monthly demand)
- Point color: `abc_vol` (A=blue, B=green, C=gray)
- Tooltip: item, loc, cv, mean, variability_class

**3. Top-20 Most Volatile Items Table**
- Columns: item, loc, abc_vol, demand_cv, demand_std, intermittency_ratio, variability_class
- Sorted: demand_cv descending
- Color-coded rows: lumpy=red-50, high=orange-50, medium=yellow-50

### Filter Controls
- `variability_class` dropdown (all / low / medium / high / lumpy)
- `abc_vol` dropdown (all / A / B / C)
- `cluster_assignment` text input

---

## Backend Script

### `mvp/demand/scripts/compute_demand_variability.py`

**Algorithm:**
```
1. Query fact_sales_monthly WHERE type=1
   GROUP BY dfu_ck, dmdunit, loc
   Filter to last history_months (from variability_config.yaml)

2. For each DFU:
   a. Collect monthly demand series (fill 0 for missing months within window)
   b. Count zero_demand_months, total_demand_months → intermittency_ratio
   c. Winsorize: cap values at mean ± outlier_sigma_threshold × std
      (iterative: 2 passes to handle bootstrap)
   d. Compute: mean, std, CV = std/mean (NULL if mean=0)
   e. Compute: MAD = median(|x - median(x)|)
   f. Compute: p50, p90 (numpy percentile)
   g. Compute: skewness, kurtosis (scipy.stats)
   h. Classify variability_class per thresholds
   i. Skip if total non-zero months < min_nonzero_months

3. UPDATE dim_dfu SET demand_mean=..., demand_std=..., ..., demand_profile_ts=NOW()
   WHERE dmdunit=... AND loc=...
   (uses executemany for batch update)
```

**CLI Usage:**
```bash
uv run python scripts/compute_demand_variability.py
uv run python scripts/compute_demand_variability.py --config config/variability_config.yaml
```

---

## Makefile Targets

```makefile
variability-schema:
	uv run python -c "from common.db import get_conn; ..."  # apply sql/022_...

variability-compute:
	uv run python scripts/compute_demand_variability.py

variability-all: variability-schema variability-compute
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_sales_monthly` | Existing table | Source of demand history; `type=1` filter |
| `dim_dfu` | Existing table | Target for variability columns |
| `config/variability_config.yaml` | New file | Thresholds and window size |
| None from other IPfeatures | — | This is the foundation feature |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_demand_variability.py`

Minimum 15 tests covering:
- CV formula: `cv = std / mean`, verified numerically
- Outlier winsorization: values beyond ±3σ are capped
- Classification boundaries: CV=0.29→low, CV=0.31→medium, CV=0.79→medium, CV=0.81→high, CV=1.51→lumpy
- Intermittency override: intermittency=0.31 forces lumpy even if CV<0.3
- All-zero demand: handled gracefully (cv=NULL, variability_class=lumpy)
- Single-month history: skipped if < min_nonzero_months
- Constant demand (std=0): cv=0.0, variability_class=low
- Skewness / kurtosis sign checks
- p90 ≥ p50 always

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_variability.py`

Minimum 8 tests:
- `GET /inv-planning/variability/summary` → 200 OK, by_class has 4 keys
- `GET /inv-planning/variability/summary?variability_class=lumpy` → filtered
- `GET /inv-planning/variability/detail` → 200 OK, rows non-empty
- `GET /inv-planning/variability/detail?limit=5` → ≤5 rows
- `GET /inv-planning/variability/histogram?metric=demand_cv` → buckets list
- `GET /inv-planning/variability/histogram?metric=invalid` → 422
- Empty DB scenario → returns zeros, not 500
- Sort by demand_cv desc → first row has highest CV

### Frontend Tests: extend `mvp/demand/frontend/src/tabs/__tests__/InventoryTab.test.tsx`
- Variability panel renders when data available
- Donut chart shows 4 class labels
- Table renders with demand_cv column

---

## Acceptance Criteria

- [ ] All DFUs in `dim_dfu` have `demand_cv`, `demand_std`, `variability_class` populated after `make variability-compute`
- [ ] `variability_class = 'lumpy'` for any DFU with `intermittency_ratio > 0.30`
- [ ] `demand_cv` is NULL (not 0) for DFUs with `demand_mean = 0`
- [ ] `GET /inv-planning/variability/summary` returns non-empty `by_class` breakdown
- [ ] Scatter plot renders in InventoryTab variability panel
- [ ] `make test-all` passes with no regressions

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/022_add_demand_variability_columns.sql` | Create |
| `mvp/demand/config/variability_config.yaml` | Create |
| `mvp/demand/scripts/compute_demand_variability.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Create (variability endpoints first) |
| `mvp/demand/api/main.py` | Modify — mount inv_planning router |
| `mvp/demand/frontend/vite.config.ts` | Modify — add `/inv-planning` proxy |
| `mvp/demand/frontend/src/tabs/InventoryTab.tsx` | Modify — add variability panel |
| `mvp/demand/tests/unit/test_demand_variability.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_variability.py` | Create |
| `mvp/demand/frontend/src/tabs/__tests__/InventoryTab.test.tsx` | Modify |
| `mvp/demand/Makefile` | Modify — add variability-* targets |
| `docs/design-specs/IPfeature1.md` | Create (this file) |
