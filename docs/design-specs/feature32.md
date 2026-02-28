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
