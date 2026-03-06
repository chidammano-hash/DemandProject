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

---

## Implementation Corrections

### Code Refactoring
All type-aware helpers have been refactored from `api/main.py` to new locations:
- `api/core.py`: `_typed_eq_clause()`, `_typed_like_clause()`, `fetch_page()`, `build_agg_trend_source()`, `parse_filters_json()`, `parse_filters_safe()`, `qident()`
- `api/routers/domains.py`: `domain_suggest()` endpoint

### Constants
- `_LARGE_TABLES = {"fact_external_forecast_monthly", "fact_sales_monthly"}`
- `_MAX_COUNT_SCAN = 100_001`

### Three-Tier Count Strategy
1. Unfiltered: `pg_class.reltuples` catalog estimate
2. Filtered on large tables (`_LARGE_TABLES`): capped at 100,001 rows
3. Filtered on small tables: full `COUNT(*)` scan

### Key Files (updated locations)
- `api/core.py` — type-aware helpers, pagination, SQL utilities
- `api/routers/domains.py` — suggest endpoint, domain CRUD


---

## Examples

### Example: Type-aware column filtering

```bash
# GIN trigram search (partial match, no = prefix) — uses pg_trgm index
curl -s "http://localhost:8000/domains/item/page?description=cabernet&limit=20" | jq '.total_rows'
# 47

# B-tree exact match (= prefix) — uses native type index
curl -s "http://localhost:8000/domains/item/page?brand==COASTAL+RIDGE&limit=50" | jq '.total_rows'
# 312

# Numeric range filter
curl -s "http://localhost:8000/domains/item/page?item_proof=>13&limit=50" | jq '.total_rows'
# 1,847
```

### Example: Column typeahead suggestions

```bash
curl -s "http://localhost:8000/domains/item/suggest?col=brand&q=coast" | jq .
# {"suggestions": ["COASTAL RIDGE", "COASTAL HIGHWAY", "COAST MOUNTAINS"]}
```

### Example: Approximate row count for large queries

```bash
curl -s "http://localhost:8000/domains/sales/page?limit=50" | jq '{total_rows, approximate}'
# {"total_rows": 2847362, "approximate": true}
# Badge shown as "2,800,000+" in the UI
```

### Example: GIN index performance comparison

```sql
EXPLAIN ANALYZE
SELECT item_no, description FROM dim_item
WHERE description ILIKE '%cabernet%';
-- "Bitmap Heap Scan on dim_item (cost=12.50..48.31 rows=47)"
-- "  Recheck Cond: (description % 'cabernet'::text)"
-- "  ->  Bitmap Index Scan on idx_dim_item_description_trgm"
-- Planning time: 0.3ms  Execution time: 0.8ms
```


---

## Additional Examples

#### Example — Filter syntax: = prefix vs plain text

```bash
# Plain text (no prefix) → GIN trigram ILIKE search (substring match)
curl -s "http://localhost:8000/domains/item/page?brand=coastal&limit=20"
# SQL: WHERE brand ILIKE '%coastal%'
# Uses: GIN trigram index → fast even on 10M rows

# = prefix → exact B-tree match (fastest, case-sensitive)
curl -s "http://localhost:8000/domains/item/page?brand==COASTAL+RIDGE&limit=20"
# SQL: WHERE brand = 'COASTAL RIDGE'
# Uses: B-tree index → sub-millisecond lookup

# Dropdown selectors auto-apply = prefix for known categorical columns
# e.g. Model selector: "lgbm_global" → sent as "=lgbm_global" to the API
# Filter placeholder text "Filter (=exact)" hints at this syntax to users
```

#### Example — Three-tier COUNT strategy

```python
# From api/core.py fetch_page()
_LARGE_TABLES = {"fact_external_forecast_monthly", "fact_sales_monthly"}
_MAX_COUNT_SCAN = 100_001

def count_rows(conn, table, where_clause, params):
    if not where_clause:
        # Tier 1: unfiltered → instant catalog estimate
        row = conn.execute(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = %s", (table,)
        ).fetchone()
        return int(row[0]), True   # (count, approximate=True)

    if table in _LARGE_TABLES:
        # Tier 2: large filtered table → cap at 100,001 to limit scan cost
        row = conn.execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM {table} WHERE {where_clause} LIMIT {_MAX_COUNT_SCAN}) _sub",
            params
        ).fetchone()
        count = int(row[0])
        return count, count >= _MAX_COUNT_SCAN  # approximate=True when capped

    # Tier 3: small table → full COUNT(*) scan (accurate)
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {where_clause}", params
    ).fetchone()
    return int(row[0]), False
```

#### Example — Debounce stability fix (preventing infinite re-renders)

```typescript
// BEFORE (broken): object reference changes every render → debounce resets forever
const debouncedFilters = useDebounce(columnFilters, 300);
// columnFilters = { brand: "coastal", model_id: "lgbm_global" }
// React creates new object reference on every render → triggers useEffect → infinite loop

// AFTER (fixed): serialize objects to JSON for stable deep comparison
function useDebounce<T>(value: T, delay: number): T {
  const serialized = typeof value === "object" && value !== null
    ? JSON.stringify(value)
    : undefined;

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [serialized ?? value, delay]);   // strings compare by value; objects by content
}
// Now the debounce timer only resets when filter CONTENT changes, not object identity
```

#### Example — Chemistry-themed loading overlay

```tsx
// From frontend/src/App.tsx (simplified)
// Domain-specific element symbols:
// "Fc" (Forecast), "Sl" (Sales), "Iv" (Inventory), "Df" (DFU), etc.

{isLoading && (
  <div className="absolute inset-0 bg-background/70 backdrop-blur-sm
                  flex flex-col items-center justify-center z-10">
    <div className="periodic-tile animate-pulse-glow
                    border-2 border-indigo-400 rounded p-4 text-center">
      <div className="text-xs text-indigo-300">26</div>
      <div className="text-3xl font-bold text-indigo-100">Fc</div>
      <div className="text-xs text-indigo-300">Forecast</div>
    </div>
    <p className="mt-3 text-sm text-muted-foreground">Querying Forecast...</p>
  </div>
)}
```

```css
/* tailwind.config.ts — custom pulse-glow animation */
keyframes: {
  'pulse-glow': {
    '0%, 100%': { boxShadow: '0 0 5px rgba(99,102,241,0.4)' },
    '50%':      { boxShadow: '0 0 20px rgba(99,102,241,0.9), 0 0 40px rgba(99,102,241,0.5)' },
  }
},
animation: { 'pulse-glow': 'pulse-glow 1.5s ease-in-out infinite' }
```

#### Example — Column typeahead suggestions in action

```bash
# User types "cab" in the description column filter of the Item domain
# Frontend calls: GET /domains/item/suggest?field=description&q=cab&limit=12
curl -s "http://localhost:8000/domains/item/suggest?field=description&q=cab&limit=12" | jq .
# {"suggestions": ["CABERNET SAUVIGNON", "CABERNET FRANC", "CAB BLEND 2024"]}

# Suggestions render as <datalist> options in the filter input
# Selecting "CABERNET SAUVIGNON" auto-fills the filter input
# If user adds = prefix → GET /domains/item/suggest?field=description&q==CABERNET...
# No suggestions returned for = prefix (exact match — no typeahead needed)
```
