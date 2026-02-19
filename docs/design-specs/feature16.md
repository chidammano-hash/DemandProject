# Feature 16 — Data Explorer Performance & UX

## Overview

A suite of performance optimizations and UX improvements for the Data Explorer, enabling fast interactive filtering on tables with 60M+ rows, typeahead suggestions on column filters, and a visible chemistry-themed loading indicator.

## Problem

1. **Slow column filters on large tables:** The forecast table has 66M+ rows. Column filters used `::text` casts on every column, preventing PostgreSQL from using indexes. `ILIKE` substring search triggered full table scans. `COUNT(*)` on filtered queries scanned millions of rows.
2. **Invisible loading state:** The original spinner was a tiny 4×4px Loader2 icon that replaced all table rows — virtually invisible on wide tables. Users couldn't tell if the UI was working or stuck.
3. **No column-level autocomplete:** Users typed filter values blind, not knowing what values exist in a column until results came back (or didn't).
4. **Debounce instability:** The `useDebounce` hook used reference equality for objects, causing infinite re-render loops when column suggestions triggered state updates.

## Solution

### 1. Type-Aware SQL Filtering

**File:** `api/main.py`

Added type-aware helpers that generate native-type SQL instead of universal `::text` casts:

- **`_col_type(spec, col)`** — returns `"text"`, `"int"`, `"float"`, or `"date"` using `DomainSpec` field lists
- **`_typed_eq_clause()`** — exact match (`=`) with native types → uses B-tree indexes
- **`_typed_like_clause()`** — substring match (`ILIKE`) without `::text` cast on text columns → uses GIN trigram indexes

Applied to: `build_where()`, `domain_suggest()`, `build_agg_trend_source()`

### 2. GIN Trigram Indexes

**File:** `sql/008_perf_indexes_and_agg.sql`

Added GIN `gin_trgm_ops` indexes on all text columns in fact tables:

```sql
-- Forecast table
CREATE INDEX IF NOT EXISTS idx_fact_forecast_model_id_trgm ON fact_external_forecast_monthly USING gin (model_id gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_forecast_dmdunit_trgm  ON fact_external_forecast_monthly USING gin (dmdunit gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_forecast_loc_trgm      ON fact_external_forecast_monthly USING gin (loc gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_forecast_dmdgroup_trgm ON fact_external_forecast_monthly USING gin (dmdgroup gin_trgm_ops);

-- Sales table
CREATE INDEX IF NOT EXISTS idx_fact_sales_dmdunit_trgm  ON fact_sales_monthly USING gin (dmdunit gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_sales_loc_trgm      ON fact_sales_monthly USING gin (loc gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_sales_dmdgroup_trgm ON fact_sales_monthly USING gin (dmdgroup gin_trgm_ops);
```

Prerequisites: `pg_trgm` extension (already enabled). Apply via `make db-apply-sql`.

### 3. Capped COUNT Optimization

**File:** `api/main.py` — `fetch_page()`

- **Unfiltered queries:** Use `pg_class.reltuples` catalog estimate (instant, no scan)
- **Filtered queries on large tables:** Cap the count scan at 100,001 rows via `SELECT count(*) FROM (SELECT 1 FROM ... WHERE ... LIMIT 100001) _sub`
- Response includes `total_approximate: true` when the count was capped
- Frontend displays `"100,000+"` badge instead of exact count

### 4. Column-Level Typeahead Suggestions

**File:** `frontend/src/App.tsx`

Extends the existing `/domains/{domain}/suggest` endpoint to column filter inputs:

- **State:** `columnSuggestions: Record<string, string[]>` — suggestions per column
- **Trigger:** `useEffect` watching `debouncedColumnFilters` — fires for text columns with non-empty, non-exact-match values
- **API call:** `GET /domains/{domain}/suggest?field={col}&q={value}&limit=12` with 180ms delay after debounce
- **Scoped:** Passes other active column filters as context so suggestions reflect the filtered subset
- **Rendering:** Native HTML `<datalist>` per column (same pattern as item/location filters)
- **Cleanup:** Clears suggestions when filter is emptied; returns `prev` reference when nothing is stale to avoid unnecessary re-renders

### 5. Chemistry-Themed Loading Overlay

**File:** `frontend/src/App.tsx`, `frontend/tailwind.config.ts`

Replaced the invisible spinner with a frosted-glass overlay showing a periodic table element tile:

- **Overlay:** Semi-transparent `bg-background/70` with `backdrop-blur` on top of existing data (doesn't replace table rows)
- **Element tile:** Domain-specific symbol (e.g., "Fc" for Forecast, "Sl" for Sales) with atomic number, styled as a periodic table element
- **Animation:** Custom `pulse-glow` keyframe with indigo box-shadow
- **Caption:** "Querying {Domain}..." below the tile
- Same pattern applied to analytics trend chart loading state

### 6. Debounce Stability Fix

**File:** `frontend/src/App.tsx` — `useDebounce()`

Fixed infinite re-render loop by using `JSON.stringify` for deep comparison of object values:

```typescript
const serialized = typeof value === "object" ? JSON.stringify(value) : undefined;
useEffect(() => {
  // ...
}, [serialized ?? value, delay]);
```

Primitives (strings) still use reference equality. Objects (like `columnFilters` record) compare by content, preventing re-renders from resetting the debounce timer.

## Filter Syntax

Users can use two modes in column filter inputs:

| Syntax | Mode | SQL | Index Used |
|---|---|---|---|
| `kraft` | Substring | `col ILIKE '%kraft%'` | GIN trigram |
| `=Kraft` | Exact match | `col = 'Kraft'` | B-tree |

The placeholder text `"Filter (=exact)"` hints at this syntax. Dropdown selectors (Model, Cluster) auto-prefix with `=`.

## Files Modified

| File | Change |
|---|---|
| `api/main.py` | `_col_type()`, `_typed_eq_clause()`, `_typed_like_clause()`, capped COUNT in `fetch_page()` |
| `sql/008_perf_indexes_and_agg.sql` | 7 GIN trigram indexes on fact table text columns |
| `frontend/src/App.tsx` | Column typeahead, chemistry overlay, debounce fix, approximate count display |
| `frontend/tailwind.config.ts` | `pulse-glow` animation keyframe |

## Dependencies

- No new Python or npm packages
- `pg_trgm` extension (already enabled)
- GIN indexes require one-time `make db-apply-sql` (5–15 min for 66M rows)

## Performance Impact

| Operation | Before | After |
|---|---|---|
| Substring filter on 66M rows | Full table scan (5–30s) | GIN index scan (<1s) |
| Exact match filter | `::text` cast prevents B-tree | Native type → B-tree (<100ms) |
| Unfiltered row count | `COUNT(*)` scan | `pg_class.reltuples` (instant) |
| Filtered row count (large table) | Full count scan | Capped at 100K rows |
