# IPfeature12 — Supplier Performance Intelligence

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P2 — Should Have

## Effort
M (Medium)

## Expert Perspectives
- **Supply Chain Control Tower Expert** (lead) — OTIF, supplier reliability, risk scoring
- **Inventory Planning Expert** — LT variance impact on SS, buffer quantification
- **UI/UX Expert** — supplier ranking, scatter bubble chart, drill-down

---

## Problem Statement

Safety stock is fundamentally a tax on supplier unreliability. When a supplier's lead time varies from 7 to 21 days, the planner must carry extra SS to cover the worst-case scenario.

The current system computes SS correctly (using σ_LT from IPfeature2), but makes this **invisible** to management. There is no supplier-level view showing:

- Which suppliers have the most variable lead times?
- How many units of SS are we carrying because of that specific supplier's unreliability?
- What is the dollar cost of each supplier's delivery variability?

This feature surfaces those answers by joining `dim_item_lead_time_profile` (IPfeature2) with `dim_item.supplier_no` and `fact_safety_stock_targets` (IPfeature3).

---

## User Story

> As a supply chain manager, I want to see each supplier's lead time reliability metrics — avg LT, LT CV, % items with stable delivery — and the associated safety stock units and dollars I'm carrying because of their variability, so I have data to drive supplier negotiations and identify which suppliers require buffer increases.

---

## Business Value

- Quantifies "SS cost of supplier unreliability" — makes a business case for supplier development
- Identifies suppliers where tighter SLAs would reduce inventory investment
- Provides objective data for procurement/supplier management conversations
- Feeds IPfeature15 (Control Tower) with supplier risk KPIs

---

## Data Requirements

### New DDL: `mvp/demand/sql/032_create_supplier_performance.sql`

New materialized view `mv_supplier_performance`:

```sql
CREATE MATERIALIZED VIEW mv_supplier_performance AS
WITH lt_by_supplier AS (
    SELECT
        i.supplier_no,
        i.supplier_name,
        ltp.item_no,
        ltp.loc,
        ltp.lt_mean_days,
        ltp.lt_std_days,
        ltp.lt_cv,
        ltp.lt_variability_class,
        ltp.observation_months
    FROM dim_item_lead_time_profile ltp
    INNER JOIN dim_item i ON ltp.item_no = i.item_no
    WHERE i.supplier_no IS NOT NULL
),
ss_by_supplier AS (
    SELECT
        i.supplier_no,
        SUM(s.ss_combined)                                   AS total_safety_stock_units,
        SUM(s.ss_lt_only)                                    AS ss_from_lt_variance,
        AVG(s.ss_coverage)                                   AS avg_ss_coverage,
        COUNT(*)                                             AS sku_loc_count,
        SUM(CASE WHEN s.is_below_ss THEN 1 ELSE 0 END)      AS below_ss_count,
        SUM(s.ss_combined * COALESCE(s.unit_cost, 1.0))     AS total_ss_value
    FROM fact_safety_stock_targets s
    INNER JOIN dim_item i ON s.item_no = i.item_no
    WHERE s.policy_version = 'v1'
    GROUP BY i.supplier_no
)
SELECT
    l.supplier_no,
    l.supplier_name,
    COUNT(DISTINCT l.item_no || '_' || l.loc)  AS sku_loc_count,
    COUNT(DISTINCT l.item_no)                  AS distinct_items,
    AVG(l.lt_mean_days)                        AS avg_lt_mean_days,
    STDDEV(l.lt_mean_days)                     AS stddev_lt_mean_cross_skus,
    AVG(l.lt_std_days)                         AS avg_lt_std_days,
    AVG(l.lt_cv)                               AS avg_lt_cv,
    MIN(l.lt_mean_days)                        AS min_lt_days,
    MAX(l.lt_mean_days)                        AS max_lt_days,
    SUM(CASE WHEN l.lt_variability_class = 'stable' THEN 1 ELSE 0 END)::float
        / NULLIF(COUNT(*), 0)                  AS pct_stable_lt,
    SUM(CASE WHEN l.lt_variability_class = 'volatile' THEN 1 ELSE 0 END)::float
        / NULLIF(COUNT(*), 0)                  AS pct_volatile_lt,
    -- SS attribution
    s.total_safety_stock_units,
    s.ss_from_lt_variance,                     -- the SS units driven purely by LT variability
    s.avg_ss_coverage,
    s.below_ss_count,
    s.total_ss_value,
    -- Reliability score (0–100): higher = more reliable
    LEAST(100, GREATEST(0,
        50 * COALESCE(
            SUM(CASE WHEN l.lt_variability_class='stable' THEN 1 ELSE 0 END)::float
            / NULLIF(COUNT(*), 0),
        0.5)
        + 50 * GREATEST(0, 1 - COALESCE(AVG(l.lt_cv), 0))
    ))::INTEGER AS supplier_reliability_score
FROM lt_by_supplier l
LEFT JOIN ss_by_supplier s ON l.supplier_no = s.supplier_no
GROUP BY l.supplier_no, l.supplier_name,
         s.total_safety_stock_units, s.ss_from_lt_variance, s.avg_ss_coverage,
         s.below_ss_count, s.total_ss_value
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_supplier_perf_pk
    ON mv_supplier_performance (supplier_no);
CREATE INDEX IF NOT EXISTS idx_supplier_perf_score
    ON mv_supplier_performance (supplier_reliability_score);
CREATE INDEX IF NOT EXISTS idx_supplier_perf_lt_cv
    ON mv_supplier_performance (avg_lt_cv DESC);
```

**Reliability Score formula:**
```
supplier_reliability_score = 50 × pct_stable_lt + 50 × (1 − avg_lt_cv)
Clamped to [0, 100]

Interpretation:
  100 = all items stable, avg LT CV = 0 (perfectly reliable)
  50  = 50% stable, avg LT CV = 0.5 (moderate)
  0   = no stable items, avg LT CV ≥ 1.0 (very unreliable)
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/supplier-performance/summary
  Response: {
    total_suppliers: int,
    avg_reliability_score: float,
    most_reliable: { supplier_no, supplier_name, supplier_reliability_score },
    least_reliable: { supplier_no, supplier_name, supplier_reliability_score },
    pct_with_volatile_lt: float,
    total_ss_from_lt_variance: float,       -- SS units attributable to LT variability
    total_ss_value_from_lt_variance: float  -- $ value of LT-driven SS
  }
  Cache: max-age=300s

GET /inv-planning/supplier-performance/detail
  Query params: supplier_no, lt_variability_class (stable|moderate|volatile),
                limit, offset, sort_by (supplier_reliability_score | avg_lt_cv | total_ss_value), sort_dir
  Response: {
    total: int,
    rows: [ {supplier_no, supplier_name, sku_loc_count, distinct_items,
             avg_lt_mean_days, avg_lt_std_days, avg_lt_cv, pct_stable_lt, pct_volatile_lt,
             total_safety_stock_units, ss_from_lt_variance, total_ss_value,
             avg_ss_coverage, below_ss_count, supplier_reliability_score} ]
  }
  Cache: max-age=300s

GET /inv-planning/supplier-performance/items
  Query params: supplier_no (required), limit, offset,
                sort_by (lt_cv | lt_mean_days | ss_combined)
  Response: {
    supplier_no: str,
    supplier_name: str,
    total: int,
    rows: [ {item_no, loc, lt_mean_days, lt_std_days, lt_cv, lt_variability_class,
             ss_combined, ss_lt_only, ss_coverage, is_below_ss} ]
  }
  Cache: max-age=120s
```

---

## Frontend UI

### Panel: "Supplier Intelligence" in `InvPlanningTab.tsx`

**KPI Cards (row of 4):**
| Card | Value | Color |
|---|---|---|
| Tracked Suppliers | total_suppliers | neutral |
| Avg Reliability Score | avg_reliability_score / 100 | green ≥ 80, amber 60–79, red < 60 |
| Volatile LT Suppliers | count with pct_volatile_lt > 50% | red if > 0 |
| SS Value from LT Variance | total_ss_value_from_lt_variance $ | amber if large |

**Reliability Score Bar Chart (horizontal):**
- One bar per supplier, sorted ascending (worst at top → easiest to see who needs attention)
- X-axis: supplier_reliability_score (0–100)
- Color: green ≥ 80, amber 60–79, red < 60

**LT Mean vs. LT CV Scatter (Bubble Chart):**
- X-axis: avg_lt_mean_days
- Y-axis: avg_lt_cv
- Bubble size: total_ss_value (larger bubble = more SS investment at risk)
- Bubble color: pct_stable_lt (green = mostly stable, red = mostly volatile)
- Tooltip: supplier_name, avg_lt, avg_lt_cv, sku_loc_count, total_ss_value

**Drill-Down: Click Supplier → Item Detail Panel**
- Shows `GET /supplier-performance/items?supplier_no=X`
- Columns: item_no, loc, lt_mean_days, lt_cv, lt_variability_class badge, ss_combined, ss from LT variance
- Sorted by lt_cv descending (most unreliable items for that supplier first)

---

## Backend Script

No separate computation script needed — `mv_supplier_performance` is a materialized view joining existing tables from IPfeature2 and IPfeature3. Refresh with:

```makefile
supplier-perf-schema:
	# apply sql/032_create_supplier_performance.sql

supplier-perf-refresh:
	uv run python -c "
	# REFRESH MATERIALIZED VIEW CONCURRENTLY mv_supplier_performance
	"
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `dim_item_lead_time_profile` | IPfeature2 | LT variability per item-loc |
| `dim_item.supplier_no`, `supplier_name` | Existing | Supplier identifier |
| `fact_safety_stock_targets` | IPfeature3 | SS quantities for attribution |
| `fact_safety_stock_targets.unit_cost` | IPfeature4 | For total_ss_value computation |

---

## Testing Requirements

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_supplier.py`

Minimum 8 tests:
- `GET /inv-planning/supplier-performance/summary` → 200 OK, total_suppliers > 0
- `GET /inv-planning/supplier-performance/detail` → rows with supplier_reliability_score 0–100
- `GET /inv-planning/supplier-performance/detail?sort_by=avg_lt_cv&sort_dir=desc` → first row has highest CV
- `GET /inv-planning/supplier-performance/items?supplier_no=X` → rows for that supplier only
- `GET /inv-planning/supplier-performance/items` (no supplier_no) → 422
- `supplier_reliability_score` is always 0–100 (clamped)
- `ss_from_lt_variance ≤ total_safety_stock_units` (LT component ≤ total)
- Empty supplier_no not in `dim_item` → empty results, not 500

### Unit Tests
Include reliability score formula verification:
- pct_stable=1.0, avg_lt_cv=0 → score=100
- pct_stable=0, avg_lt_cv=1.0 → score=0
- pct_stable=0.5, avg_lt_cv=0.5 → score=50

---

## Acceptance Criteria

- [ ] `mv_supplier_performance` populated after `make supplier-perf-refresh`
- [ ] `supplier_reliability_score` is 0–100 for all rows
- [ ] `ss_from_lt_variance` = sum of `ss_lt_only` from IPfeature3 for that supplier's items
- [ ] Bubble chart renders with size proportional to `total_ss_value`
- [ ] Drill-down shows item-level LT profiles for selected supplier
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/032_create_supplier_performance.sql` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add supplier-performance endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Supplier Intelligence panel |
| `mvp/demand/tests/api/test_inv_planning_supplier.py` | Create |
| `mvp/demand/Makefile` | Modify — add supplier-perf-* targets |
| `docs/design-specs/IPfeature12.md` | Create (this file) |
