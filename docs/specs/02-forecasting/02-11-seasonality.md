<!-- SOURCE: feature30.md (Seasonality Detection) -->
# Feature 30: DFU Seasonality Detection & Profile Assignment

## Objective

Build a statistical seasonality detection pipeline that analyzes each DFU's monthly sales history, classifies it into a seasonality profile (`none`, `low`, `medium`, `high`), determines whether the DFU exhibits a yearly seasonal cycle, and stores the results in `dim_dfu` for downstream use in model selection, feature engineering, and UI analytics.

## Motivation

- **Model selection:** Seasonal DFUs benefit from models with explicit seasonality components (Prophet, NeuralProphet, SeasonalNaive) while non-seasonal DFUs perform better with trend-focused models (LGBM, XGBoost). Per-DFU seasonality profiles can inform champion model selection logic.
- **Feature engineering:** Backtesting scripts can condition lag/rolling window features on seasonality strength — e.g., including month-of-year indicators only for seasonal DFUs.
- **Clustering refinement:** The current clustering pipeline uses `seasonality_strength` as one of 8 features. A dedicated seasonality pipeline produces richer, validated metrics that can replace or augment the clustering signal.
- **Business visibility:** Planners need to know which DFUs are seasonal to set appropriate safety stock, review frequency, and exception thresholds. Currently this information exists only as a raw float inside the clustering feature matrix — not as a first-class DFU attribute.

## Scope

**In scope:**
- New script: `scripts/detect_seasonality.py` — computes per-DFU seasonality metrics from `fact_sales_monthly`
- New script: `scripts/update_seasonality_profiles.py` — writes results to `dim_dfu` columns in PostgreSQL
- New config: `config/seasonality_config.yaml` — thresholds and parameters
- New columns on `dim_dfu`: `seasonality_profile`, `seasonality_strength`, `is_yearly_seasonal`, `peak_month`, `trough_month`, `peak_trough_ratio`
- New DDL: `sql/015_add_seasonality_columns.sql`
- Update `DFU_SPEC` in `common/domain_specs.py` with new columns
- New Makefile targets: `seasonality-detect`, `seasonality-update`, `seasonality-all`

**Out of scope:**
- UI panel for seasonality visualization (future feature)
- API endpoint for seasonality profiles (accessible via existing `/domains/dfu/rows` and `/domains/dfu/search` once columns are added)
- Automatic re-run on data refresh (manual pipeline execution)
- Sub-annual periodicity detection (e.g., quarterly, biweekly) — only yearly cycles

## Architecture Overview

```
fact_sales_monthly (qty per DFU per month)
        ↓
detect_seasonality.py
  1. Load monthly sales per DFU
  2. Filter DFUs with sufficient history (≥ min_months)
  3. Per-DFU: compute seasonality metrics
  4. Classify into profile tiers (none / low / medium / high)
  5. Detect yearly seasonal cycle (boolean)
  6. Output: seasonality_results.csv
        ↓
update_seasonality_profiles.py
  1. Read seasonality_results.csv
  2. UPDATE dim_dfu SET seasonality columns WHERE dfu_ck = ...
  3. DFUs with insufficient history → seasonality_profile = 'insufficient_history'
        ↓
dim_dfu (enriched with seasonality attributes)
```

## Seasonality Detection Algorithm

### Step 1: Monthly Demand Aggregation

For each DFU, pivot the sales time series into a `month_of_year × year` matrix:

```
         Year1   Year2   Year3
Jan       120     135     128
Feb        95     102      98
Mar       110     118     115
...
Dec       145     160     152
```

Requires at least `min_months_history` (default: 24) months of non-zero data. DFUs below this threshold are labeled `insufficient_history`.

### Step 2: Seasonality Strength (Coefficient of Variation of Monthly Means)

Compute the average demand for each calendar month across all available years, then measure the dispersion:

```
monthly_means = [mean(Jan), mean(Feb), ..., mean(Dec)]
seasonality_strength = std(monthly_means) / mean(monthly_means)
```

- A perfectly flat series has `seasonality_strength ≈ 0`
- A highly seasonal series (e.g., sunscreen, holiday items) has `seasonality_strength > 1.0`

This metric is already computed in `generate_clustering_features.py` — the seasonality pipeline reuses the same formula but stores it as a persistent DFU attribute rather than a transient clustering feature.

### Step 3: Year-over-Year Correlation

Confirm that the seasonal pattern is repeatable (not one-time spikes):

```
Construct a (months_per_year × num_years) matrix
Compute pairwise Pearson correlation between year columns
yoy_correlation = mean of off-diagonal correlations
```

- `yoy_correlation > threshold` → the pattern repeats across years (true seasonality)
- `yoy_correlation ≤ threshold` → the pattern may be noise or one-time events

### Step 4: Autocorrelation at Lag 12

Compute the autocorrelation of the raw monthly time series at lag 12 (one year):

```
acf_lag12 = autocorrelation(series, lag=12)
```

- `acf_lag12 > threshold` → strong evidence of a 12-month cycle
- This catches cases where `seasonality_strength` is moderate but the cycle is consistent

### Step 5: Peak-to-Trough Ratio

Identify the peak and trough months and compute the demand ratio:

```
peak_month = argmax(monthly_means) + 1       # 1-indexed (1=Jan, 12=Dec)
trough_month = argmin(monthly_means) + 1     # 1-indexed
peak_trough_ratio = monthly_means[peak] / monthly_means[trough]
```

- Ratio close to 1.0 → flat demand
- Ratio > 2.0 → peak month has 2x+ demand vs. trough — strong seasonal swing

Guard against division by zero: if `monthly_means[trough] == 0`, set `peak_trough_ratio = NULL`.

### Step 6: Profile Classification

Combine the metrics into a composite seasonality profile using configurable thresholds:

| Profile | Criteria |
|---------|----------|
| `high` | `seasonality_strength ≥ high_threshold` AND (`yoy_correlation ≥ yoy_threshold` OR `acf_lag12 ≥ acf_threshold`) |
| `medium` | `seasonality_strength ≥ medium_threshold` AND (`yoy_correlation ≥ yoy_threshold` OR `acf_lag12 ≥ acf_threshold`) |
| `low` | `seasonality_strength ≥ low_threshold` BUT fails yoy/acf confirmation |
| `none` | `seasonality_strength < low_threshold` |
| `insufficient_history` | Fewer than `min_months_history` months of data |

The `yoy_correlation` and `acf_lag12` checks serve as **confirmation gates** — a DFU needs both amplitude (strength) and repeatability (correlation/autocorrelation) to qualify as `medium` or `high`.

### Step 7: Yearly Seasonal Flag

A DFU is flagged as `is_yearly_seasonal = TRUE` when:

```
(seasonality_strength ≥ low_threshold)
AND (yoy_correlation ≥ yoy_threshold OR acf_lag12 ≥ acf_threshold)
AND (peak_trough_ratio ≥ peak_trough_min_ratio)
```

This is effectively `profile IN ('medium', 'high')` plus the peak-trough ratio check. It provides a simple boolean for downstream consumers that just need "is this DFU seasonal — yes or no?"

## Output Metrics (per DFU)

| Metric | Type | Description |
|--------|------|-------------|
| `seasonality_profile` | TEXT | `none`, `low`, `medium`, `high`, or `insufficient_history` |
| `seasonality_strength` | NUMERIC(10,4) | CV of monthly means (0 = flat, >1 = strongly seasonal) |
| `is_yearly_seasonal` | BOOLEAN | TRUE if DFU exhibits a confirmed yearly cycle |
| `peak_month` | INTEGER | Month with highest average demand (1–12) |
| `trough_month` | INTEGER | Month with lowest average demand (1–12) |
| `peak_trough_ratio` | NUMERIC(10,4) | Peak demand / trough demand ratio |

Additional metrics computed and saved to CSV but **not** stored in `dim_dfu` (available for analysis):

| Metric | Type | Description |
|--------|------|-------------|
| `yoy_correlation` | FLOAT | Mean year-over-year Pearson correlation |
| `acf_lag12` | FLOAT | Autocorrelation at lag 12 |
| `months_available` | INTEGER | Number of months with non-zero demand |
| `monthly_means` | ARRAY[12] | Average demand per calendar month (JSON) |

## Implementation Components

### 1. Seasonality Detection Script

**File:** `mvp/demand/scripts/detect_seasonality.py`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `config/seasonality_config.yaml` | Path to config file |
| `--min-months` | (from config: 24) | Override minimum months required |
| `--output` | `data/seasonality_results.csv` | Output CSV path |
| `--verbose` | `false` | Print per-DFU diagnostics |

**Logic:**
1. Connect to PostgreSQL, load `fact_sales_monthly` (qty, dmdunit, dmdgroup, loc, startdate)
2. Construct `dfu_ck` = `dmdunit|dmdgroup|loc` to match `dim_dfu`
3. Group by `dfu_ck` + `startdate`, aggregate `qty`
4. For each DFU:
   a. Check history length ≥ `min_months_history`
   b. Pivot to month-of-year × year matrix
   c. Compute `seasonality_strength`, `yoy_correlation`, `acf_lag12`, `peak_month`, `trough_month`, `peak_trough_ratio`, `monthly_means`
   d. Classify `seasonality_profile`
   e. Set `is_yearly_seasonal` flag
5. Write results to CSV
6. Print summary: count per profile tier, top 10 most seasonal DFUs

### 2. Seasonality Update Script

**File:** `mvp/demand/scripts/update_seasonality_profiles.py`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `data/seasonality_results.csv` | Input CSV from detection step |
| `--dry-run` | `false` | Print SQL without executing |

**Logic:**
1. Read `seasonality_results.csv`
2. For each DFU row, execute:
   ```sql
   UPDATE dim_dfu
   SET seasonality_profile = %s,
       seasonality_strength = %s,
       is_yearly_seasonal = %s,
       peak_month = %s,
       trough_month = %s,
       peak_trough_ratio = %s,
       modified_ts = NOW()
   WHERE dfu_ck = %s
   ```
3. Use batch updates (executemany) for performance
4. Print update count and any unmatched DFU keys

## Database Schema

**File:** `mvp/demand/sql/015_add_seasonality_columns.sql`

```sql
-- Feature 30: DFU Seasonality Detection
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS seasonality_profile TEXT;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS seasonality_strength NUMERIC(10,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS is_yearly_seasonal BOOLEAN;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS peak_month INTEGER;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS trough_month INTEGER;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS peak_trough_ratio NUMERIC(10,4);

CREATE INDEX IF NOT EXISTS idx_dim_dfu_seasonality_profile
    ON dim_dfu (seasonality_profile);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_is_yearly_seasonal
    ON dim_dfu (is_yearly_seasonal);

COMMENT ON COLUMN dim_dfu.seasonality_profile IS 'none | low | medium | high | insufficient_history';
COMMENT ON COLUMN dim_dfu.seasonality_strength IS 'CV of monthly means (0=flat, >1=strongly seasonal)';
COMMENT ON COLUMN dim_dfu.is_yearly_seasonal IS 'TRUE if confirmed 12-month cycle';
COMMENT ON COLUMN dim_dfu.peak_month IS 'Month with highest avg demand (1=Jan, 12=Dec)';
COMMENT ON COLUMN dim_dfu.trough_month IS 'Month with lowest avg demand (1=Jan, 12=Dec)';
COMMENT ON COLUMN dim_dfu.peak_trough_ratio IS 'Peak month avg / trough month avg';
```

## Domain Spec Update

Add new columns to `DFU_SPEC` in `common/domain_specs.py`:

```python
# In DFU_SPEC.columns list:
"seasonality_profile",
"seasonality_strength",
"is_yearly_seasonal",
"peak_month",
"trough_month",
"peak_trough_ratio",

# In search_fields:
"seasonality_profile",

# In float_fields:
"seasonality_strength", "peak_trough_ratio",

# In int_fields:
"peak_month", "trough_month",
```

This enables immediate querying via the existing generic API:
- `GET /domains/dfu/search?q=high` — finds DFUs with `seasonality_profile = 'high'`
- `GET /domains/dfu/rows?sort=seasonality_strength&order=desc` — rank by seasonality

## Configuration

**File:** `mvp/demand/config/seasonality_config.yaml`

```yaml
seasonality:
  # Minimum months of non-zero history required
  min_months_history: 24

  # Seasonality strength thresholds (CV of monthly means)
  thresholds:
    low: 0.15          # Below this → profile = 'none'
    medium: 0.35        # Between low and medium with confirmation → 'medium'
    high: 0.70          # Above this with confirmation → 'high'

  # Confirmation gates (at least one must pass for medium/high)
  confirmation:
    yoy_correlation: 0.40     # Year-over-year Pearson correlation
    acf_lag12: 0.30           # Autocorrelation at lag 12

  # Peak-trough ratio minimum for is_yearly_seasonal flag
  peak_trough_min_ratio: 1.3

  # Output
  output_path: "data/seasonality_results.csv"
```

### Threshold Rationale

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| `low` | 0.15 | Below 15% CV, month-to-month variation is within noise for most demand series |
| `medium` | 0.35 | 35% CV indicates a detectable seasonal swing (e.g., 30% above/below annual mean) |
| `high` | 0.70 | 70% CV means peak months have ~2x the demand of trough months |
| `yoy_correlation` | 0.40 | Moderate positive correlation confirms the pattern repeats across years |
| `acf_lag12` | 0.30 | Positive autocorrelation at 12-month lag confirms annual periodicity |
| `peak_trough_min_ratio` | 1.3 | Peak month must be ≥30% above trough to call it "yearly seasonal" |

These are starting defaults — tune after running on the actual dataset. The config file makes iteration easy without code changes.

## Makefile Targets

```makefile
# Seasonality detection pipeline
seasonality-detect:
	$(UV) python scripts/detect_seasonality.py

seasonality-update:
	$(UV) python scripts/update_seasonality_profiles.py

seasonality-all: seasonality-detect seasonality-update

# Schema (run once)
seasonality-schema:
	psql "$(DB_URL)" -f sql/015_add_seasonality_columns.sql
```

## API Integration

No new endpoints required. Once the columns are added to `DFU_SPEC`, the existing generic domain API automatically supports:

- **Listing:** `GET /domains/dfu/rows` returns seasonality columns in response
- **Search:** `GET /domains/dfu/search?q=high` matches `seasonality_profile` via trigram index
- **Exact filter:** `GET /domains/dfu/rows?seasonality_profile==high` via B-tree (uses `=exact` prefix convention)
- **Sorting:** `GET /domains/dfu/rows?sort=seasonality_strength&order=desc`
- **Typeahead:** Column typeahead on `seasonality_profile` returns distinct values (`none`, `low`, `medium`, `high`, `insufficient_history`)

## Relationship to Existing Features

| Feature | Relationship |
|---------|-------------|
| Feature 7 (Clustering) | Clustering already computes `seasonality_strength` as a transient feature. This pipeline promotes it to a persistent DFU attribute with richer validation (yoy correlation, autocorrelation, profile tiers). The clustering pipeline can optionally consume the stored `seasonality_profile` as a categorical feature. |
| Feature 15 (Champion Selection) | Champion selection picks the best model per DFU via WAPE. `seasonality_profile` can be added as a stratification dimension — e.g., "which model wins most often for `high` seasonal DFUs?" |
| Feature 17 (DFU Analysis) | The DFU Analysis tab already shows sales vs. forecast overlays. `seasonality_profile` can be added as a filter/badge to help planners focus on seasonal DFUs. |
| Feature 8/9/12/13 (Backtesting) | Backtest scripts can use `is_yearly_seasonal` to conditionally include month-of-year features or select seasonal model variants. |

## Dependencies

- Feature 3 (`dim_dfu` table)
- Feature 4 (`fact_sales_monthly` table)
- Python packages (all already in stack): `numpy`, `pandas`, `scipy`, `psycopg`, `pyyaml`
- No new package dependencies required

## Expected Output Distribution

Based on typical demand planning datasets with ~3 years of monthly history:

| Profile | Expected % | Description |
|---------|-----------|-------------|
| `none` | 30–40% | Staple/commodity items with flat demand |
| `low` | 20–30% | Slight seasonal variation but not actionable |
| `medium` | 15–25% | Clear seasonal pattern, benefits from seasonal models |
| `high` | 5–15% | Strong seasonality (holiday items, weather-dependent) |
| `insufficient_history` | 5–10% | New items or sparse DFUs with < 24 months |

Actual distribution will vary — the detection script prints a summary table after each run.

## Testing & Validation

1. **Sanity check:** Run on the full dataset and verify profile distribution roughly matches expected ranges
2. **Known seasonal items:** Identify 3–5 DFUs known to be seasonal (e.g., holiday or summer products) and confirm they are classified as `medium` or `high`
3. **Known flat items:** Identify 3–5 staple DFUs and confirm they are classified as `none` or `low`
4. **Threshold sensitivity:** Re-run with `low: 0.10` and `low: 0.20` to check that profile counts shift predictably
5. **Consistency with clustering:** Compare `seasonality_strength` values from this pipeline vs. `generate_clustering_features.py` — they should be identical (same formula, same data)
6. **Database verification:** After `seasonality-update`, query `SELECT seasonality_profile, COUNT(*) FROM dim_dfu GROUP BY 1` and confirm totals match the CSV output

---

## Implementation Details

### Makefile
- `seasonality-all` target includes `seasonality-schema` step (schema + detect + update)

### Update Script Performance
- Uses temp table + `COPY ... FROM STDIN` + single `UPDATE ... FROM` join (not `executemany` batch updates)

### Automated Tests
- `tests/unit/test_seasonality.py` — 12 tests: `compute_acf_lag12` (4), `compute_seasonality_metrics` (7), `DfuSpecSeasonalityColumns` (4)
- `tests/api/test_seasonality.py` — 7 tests: DFU meta includes seasonality columns, numeric fields, typeahead, page filtering, sorting, seasonality-profiles endpoint, empty profiles

---

## Examples

### Example: Run seasonality detection pipeline

```bash
make seasonality-all
# Step 1: detect_seasonality.py → computes strength, profile, peak/trough for 18,432 DFUs
# Step 2: update_seasonality_profiles.py → writes to dim_dfu
# Updated 18,432 rows

# Verify distribution:
psql -h localhost -p 5440 -U demand -d demand \
  -c "SELECT seasonality_profile, COUNT(*) FROM dim_dfu GROUP BY 1 ORDER BY 2 DESC;"
# yearly_strong    | 4821
# yearly_moderate  | 6103
# non_seasonal     | 7508
```

### Example: Seasonality config

```yaml
# config/seasonality_config.yaml
seasonality:
  min_months_history: 24
  strength_thresholds:
    strong:   0.60
    moderate: 0.30
  labeling:
    yearly_strong:   {min_strength: 0.60, is_yearly_seasonal: true}
    yearly_moderate: {min_strength: 0.30, is_yearly_seasonal: true}
    non_seasonal:    {max_strength: 0.30, is_yearly_seasonal: false}
```

### Example: Query seasonality attributes for a DFU

```sql
SELECT dmdunit, loc, seasonality_profile, seasonality_strength,
       peak_month, trough_month, peak_trough_ratio, is_yearly_seasonal
FROM dim_dfu WHERE dmdunit='100320' AND loc='1401-BULK';
-- 100320 | 1401-BULK | yearly_strong | 0.78 | 11 | 2 | 3.4 | true
-- Peak in November (holiday season), trough in February
```


---

<!-- SOURCE: feature32.md (Seasonality Profile Filtering) -->
# Feature 32: Seasonality Profile Filtering

## Summary

Expose `seasonality_profile` (from `dim_dfu`) as a filter and slice dimension across the Accuracy Tab, DFU Analysis Tab, and Data Explorer. Allows users to compare forecast accuracy by seasonality tier (e.g., high_seasonal vs non_seasonal) and filter DFU-level analysis to specific seasonality profiles.

## Prerequisites

- Feature 30 (DFU Seasonality Detection) — populates `dim_dfu.seasonality_profile`
- Feature 10 (Multi-dimensional accuracy slicing) — accuracy materialized views
- Feature 17 (DFU Analysis tab) — DFU analysis endpoint

## Database Changes

### SQL Migration: `sql/016_add_seasonality_to_accuracy_views.sql`

Drops and recreates all 4 accuracy materialized views with `seasonality_profile` as a new GROUP BY dimension:

| View | Change |
|------|--------|
| `agg_accuracy_by_dim` | Added `COALESCE(d.seasonality_profile, '(unknown)') AS seasonality_profile` |
| `agg_accuracy_lag_archive` | Same |
| `agg_dfu_coverage` | Same |
| `agg_dfu_coverage_lag_archive` | Same |

Each view gets a new B-tree index on `seasonality_profile`.

After applying, run `REFRESH MATERIALIZED VIEW` for each view.

## API Changes

### New Endpoint

**`GET /domains/dfu/seasonality-profiles`**

Returns distinct seasonality profiles with DFU counts, used to populate filter dropdowns.

Response:
```json
{
  "profiles": [
    {"profile": "high_seasonal", "count": 1200},
    {"profile": "non_seasonal", "count": 800}
  ]
}
```

### Modified Endpoints

**`GET /forecast/accuracy/slice`**
- New query param: `seasonality_profile` (string, optional)
- New `group_by` value: `"seasonality_profile"`
- When provided as filter, adds WHERE clause on the pre-aggregated view or raw fact table

**`GET /forecast/accuracy/lag-curve`**
- New query param: `seasonality_profile` (string, optional)
- When provided, filters by seasonality profile

**`GET /dfu/analysis`**
- New query param: `seasonality_profile` (string, optional)
- When provided, adds subquery filter: `(dmdunit, loc) IN (SELECT dmdunit, loc FROM dim_dfu WHERE seasonality_profile = %s)`
- Response `dfu_attributes` now includes 6 seasonality columns

## Frontend Changes

### Types (`types/index.ts`)
- `SeasonalityProfile`: `{ profile: string; count: number }`
- `SeasonalityProfilesPayload`: `{ profiles: SeasonalityProfile[] }`

### Query Layer (`api/queries.ts`)
- `queryKeys.seasonalityProfiles()` — query key factory
- `fetchSeasonalityProfiles()` — fetches profiles endpoint
- `SliceParams.seasonality_profile?` — optional param
- `LagCurveParams.seasonality_profile?` — optional param
- `DfuAnalysisParams.seasonality_profile?` — optional param

### Accuracy Tab (`tabs/AccuracyTab.tsx`)
- Seasonality Profile dropdown filter (populated from profiles endpoint)
- "Seasonality Profile" option in "Slice by" dropdown
- `seasonality_profile` wired into both `sliceParams` and `lagCurveParams`

### DFU Analysis Tab (`tabs/DfuAnalysisTab.tsx`)
- Seasonality Profile dropdown filter in controls row
- `seasonality_profile` passed in analysis query params and query key

### Data Explorer (`tabs/ExplorerTab.tsx`)
- Seasonality Profile dropdown shown when `domain === "dfu"`
- Injects `{"seasonality_profile": "=<value>"}` into `effectiveFilters`

## Tests

### Backend (pytest)
- `test_forecast_accuracy.py`: 3 new tests — seasonality as filter param, as group_by, and on lag-curve
- `test_dfu_analysis.py`: 1 new test — seasonality_profile filter param
- `test_seasonality.py`: 2 new tests — seasonality-profiles endpoint (with data + empty)

### Frontend (vitest)
- Updated mocks in `AccuracyTab.test.tsx`, `DfuAnalysisTab.test.tsx`, `ExplorerTab.test.tsx` to include `fetchSeasonalityProfiles` and `seasonalityProfiles` query key

## Verification

1. Apply migration: `psql -f sql/016_add_seasonality_to_accuracy_views.sql`
2. Refresh views: `REFRESH MATERIALIZED VIEW agg_accuracy_by_dim` (and 3 others)
3. Backend tests: `make test` — 189 tests pass
4. Frontend tests: `make ui-test` — 108 tests pass
5. Manual: Accuracy Tab → Slice by "Seasonality Profile" shows buckets; filter dropdown constrains data
6. Manual: DFU Analysis → seasonality dropdown filters aggregated chart data
7. Manual: Data Explorer → DFU domain shows seasonality dropdown, filters rows

---

## Implementation Details

### Endpoint Locations
- Accuracy endpoints: both `api/main.py` (inline) and `api/routers/accuracy.py` (router)
- DFU Analysis endpoint: both `api/main.py` and `api/routers/analysis.py` (returns 6 seasonality columns in `dfu_attributes`)
- Seasonality-profiles endpoint: `api/routers/clusters.py` (not a dedicated router)

### Additional View Dimensions
- `sql/016` also adds `brand_desc` as a dimension in all 4 views
- `dfu_execution_lag` added to `agg_accuracy_by_dim` and `agg_dfu_coverage`
- Full `agg_accuracy_by_dim` grain: 11 columns (model_id, lag, month_start, cluster_assignment, ml_cluster, supplier_desc, abc_vol, region, brand_desc, dfu_execution_lag, seasonality_profile)


---

## Examples

### Example: Accuracy slice by seasonality profile

```bash
curl -s "http://localhost:8000/forecast/accuracy/slice?lag=2&dim=seasonality_profile&model=lgbm_global" \
  | jq '.rows[] | {seasonality_profile, accuracy_pct, n_dfus}'
# {"seasonality_profile": "yearly_strong",   "accuracy_pct": 88.4, "n_dfus": 4821}
# {"seasonality_profile": "yearly_moderate", "accuracy_pct": 91.2, "n_dfus": 6103}
# {"seasonality_profile": "non_seasonal",    "accuracy_pct": 94.1, "n_dfus": 7508}
# Key insight: non-seasonal DFUs have 5.7pp better accuracy than seasonal ones
```

### Example: Filter DFU explorer by seasonality profile

```bash
# Show only strongly seasonal DFUs in data explorer
curl -s "http://localhost:8000/domains/dfu/page?seasonality_profile==yearly_strong&limit=20" \
  | jq '{total_rows, first_item: .rows[0] | {dmdunit, loc, peak_month}}'
# {"total_rows": 4821, "first_item": {"dmdunit": "100320", "loc": "1401-BULK", "peak_month": 11}}
```

### Example: Seasonality filter dropdown in AccuracyTab

```typescript
// Seasonality profile filter (uses B-tree exact match with = prefix)
const [seasonalityFilter, setSeasonalityFilter] = useState<string>('')

const { data } = useQuery({
  queryKey: ['accuracy-slice', 'seasonality_profile', seasonalityFilter, model, lag],
  queryFn: () => fetchAccuracySlice({
    dim: 'seasonality_profile',
    filter: seasonalityFilter ? `seasonality_profile==${seasonalityFilter}` : undefined,
    model, lag
  }),
})
```
