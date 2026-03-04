# IPfeature8 — Fill Rate & Demand Fulfillment Analytics

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P2 — Should Have

## Effort
M (Medium)

## Expert Perspectives
- **Demand Planning Expert** (lead) — fill rate definition, partial fulfillment interpretation
- **Statistical Analyst** — fill rate formula (β service level), shortage imputation
- **Supply Chain Control Tower Expert** — fulfillment KPIs, ABC-class service tracking

---

## Problem Statement

The existing inventory backtest (Feature 37) measures **Cycle Service Level (CSL)** — whether EOM stock is positive. CSL is a binary yes/no per month. It misses the severity of shortfalls:

- A DFU that shipped 80 of 100 units ordered has CSL=1 (no stockout at EOM) but Fill Rate=80%
- CSL doesn't measure **how much** demand went unfulfilled
- Planners and commercial teams care about Fill Rate, not just whether the warehouse had any stock at month-end

**Fill Rate (β service level)** = units shipped / units ordered — the customer-facing metric.

The data already exists in `fact_sales_monthly`: both `qty_ordered` and `qty_shipped` are loaded. This feature is primarily a materialized view computation and API layer on top of existing data.

---

## User Story

> As a supply chain manager, I want Fill Rate (units shipped / units ordered) per item-location per month, split by ABC class, cluster, and seasonality — so I can measure how well we actually serve demand rather than just whether we avoided end-of-month stockouts.

---

## Business Value

- Fills the **customer-facing service level gap** that CSL leaves open
- Enables portfolio-wide "we ship X% of what customers order" KPI
- Identifies high-shortage items whose stockout impact was invisible to CSL
- Feeds IPfeature15 (Control Tower) with fill rate trend

---

## Key Formulas

```
# Per item-loc-month
total_ordered  = SUM(qty_ordered)  WHERE type=1 AND qty_ordered > 0
total_shipped  = SUM(qty_shipped)
fill_rate      = total_shipped / total_ordered      (NULL if total_ordered = 0)
shortage_qty   = GREATEST(total_ordered - total_shipped, 0)
had_partial    = total_shipped < total_ordered

# Portfolio aggregate
portfolio_fill_rate = SUM(total_shipped) / SUM(total_ordered)
```

**Note:** `fill_rate` is NULL (not 0 or 1) when `total_ordered = 0`. Items with no orders in a month are excluded from portfolio averages.

---

## Data Requirements

### New DDL: `mvp/demand/sql/028_create_fill_rate_monthly.sql`

New materialized view `mv_fill_rate_monthly`:

```sql
CREATE MATERIALIZED VIEW mv_fill_rate_monthly AS
SELECT
    s.dmdunit                              AS item_no,
    s.loc,
    s.startdate                            AS month_start,
    SUM(s.qty_ordered)                     AS total_ordered,
    SUM(s.qty_shipped)                     AS total_shipped,
    CASE WHEN SUM(s.qty_ordered) > 0
         THEN SUM(s.qty_shipped) / SUM(s.qty_ordered)
         ELSE NULL END                     AS fill_rate,
    GREATEST(SUM(s.qty_ordered) - SUM(s.qty_shipped), 0) AS shortage_qty,
    (SUM(s.qty_shipped) < SUM(s.qty_ordered)) AS had_partial_fulfillment,
    -- DFU attributes for slicing (from dim_dfu)
    COALESCE(d.abc_vol, '(unknown)')              AS abc_vol,
    COALESCE(d.cluster_assignment, '(unassigned)') AS cluster_assignment,
    COALESCE(d.region, '(unknown)')               AS region,
    d.is_yearly_seasonal,
    d.seasonality_profile,
    d.variability_class
FROM fact_sales_monthly s
LEFT JOIN dim_dfu d
    ON s.dmdunit = d.dmdunit
    AND s.dmdgroup = d.dmdgroup
    AND s.loc = d.loc
WHERE s.type = 1
  AND s.qty_ordered IS NOT NULL
  AND s.qty_ordered > 0
GROUP BY
    s.dmdunit, s.loc, s.startdate,
    d.abc_vol, d.cluster_assignment, d.region,
    d.is_yearly_seasonal, d.seasonality_profile, d.variability_class
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_fill_rate_pk
    ON mv_fill_rate_monthly (item_no, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_month
    ON mv_fill_rate_monthly (month_start);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_abc
    ON mv_fill_rate_monthly (abc_vol);
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_partial
    ON mv_fill_rate_monthly (had_partial_fulfillment)
    WHERE had_partial_fulfillment = TRUE;
CREATE INDEX IF NOT EXISTS idx_mv_fill_rate_low
    ON mv_fill_rate_monthly (fill_rate)
    WHERE fill_rate < 0.95;
```

---

## API Endpoints

**New Router:** `mvp/demand/api/routers/fill_rate.py` (mounted at `/fill-rate`)

```
GET /fill-rate/summary
  Query params: month_from, month_to, item, location, abc_vol,
                cluster_assignment, region
  Response: {
    portfolio_fill_rate: float,
    total_ordered: float,
    total_shipped: float,
    total_shortage_qty: float,
    partial_fulfillment_events: int,
    by_abc: {
      A: { avg_fill_rate, total_shortage_qty, events },
      B: { ... },
      C: { ... }
    },
    worst_items: [ {item_no, loc, fill_rate, shortage_qty, abc_vol} × 10 ],
    trend: [ {month_start, portfolio_fill_rate, total_shortage_qty} ]
  }
  Cache: max-age=300s

GET /fill-rate/trend
  Query params: month_from, month_to, item, location, abc_vol
  Response: {
    months: [ {month_start, fill_rate, total_ordered, total_shipped, shortage_qty} ]
  }
  Cache: max-age=300s

GET /fill-rate/detail
  Query params: month_from, month_to, item, location, abc_vol,
                had_partial_fulfillment (bool), limit, offset,
                sort_by (fill_rate | shortage_qty | total_ordered), sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, month_start, total_ordered, total_shipped,
             fill_rate, shortage_qty, had_partial_fulfillment,
             abc_vol, cluster_assignment, region} ]
  }
  Cache: max-age=120s
```

**Vite proxy:** Add `/fill-rate` entry to `mvp/demand/frontend/vite.config.ts`

---

## Frontend UI

### Panel: "Fill Rate Analytics" in `InvPlanningTab.tsx`

**KPI Cards (row of 4):**
| Card | Value | Threshold |
|---|---|---|
| Portfolio Fill Rate | portfolio_fill_rate % | Green ≥95%, amber 90–95%, red <90% |
| Total Shortage Units | total_shortage_qty | Red if >0 |
| A-Class Fill Rate | by_abc.A.avg_fill_rate | Higher standard for A items |
| Partial Fulfillment Events | partial_fulfillment_events count | Amber if >100 |

**Monthly Trend Chart (dual Y-axis):**
- Line: fill_rate % (left axis, 0–100%)
- Bar: shortage_qty (right axis)
- X-axis: month_start
- Reference line at 95% (target fill rate)

**ABC Class Fill Rate Horizontal Bar Chart:**
- 3 bars: A, B, C classes
- X-axis: avg_fill_rate (0–100%)
- Color: green if ≥95%, amber if 90–95%, red if <90%

**Worst Items Table:**
- Columns: item, loc, abc_vol, month, fill_rate, shortage_qty, total_ordered, total_shipped
- Sorted by shortage_qty descending
- Filter: had_partial_fulfillment checkbox

---

## Backend Script

**Makefile Targets:**
```makefile
fill-rate-schema:
	# apply sql/028_create_fill_rate_monthly.sql (CREATE MAT VIEW WITH NO DATA)

fill-rate-refresh:
	uv run python -c "
	import asyncio
	from common.db import get_conn
	async def refresh():
	    conn = await get_conn()
	    await conn.execute('REFRESH MATERIALIZED VIEW CONCURRENTLY mv_fill_rate_monthly')
	asyncio.run(refresh())
	"
```

No separate script needed — refresh is a single SQL command.

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_sales_monthly` | Existing | qty_ordered + qty_shipped (type=1 rows) |
| `dim_dfu` | Existing | abc_vol, cluster_assignment, region for slicing |
| IPfeature1 `variability_class` | IPfeature1 | For variability-based slicing in summary |
| No other IPfeatures required | — | Independent of SS/EOQ pipeline |

---

## Testing Requirements

### Backend API Tests: `mvp/demand/tests/api/test_fill_rate.py`

Minimum 10 tests:
- `GET /fill-rate/summary` → 200 OK, portfolio_fill_rate between 0 and 1
- `GET /fill-rate/summary?abc_vol=A` → filtered to A-class rows only
- `GET /fill-rate/trend` → months list, each with fill_rate and shortage_qty
- `GET /fill-rate/trend?month_from=2024-01-01&month_to=2024-06-30` → 6 months max
- `GET /fill-rate/detail` → rows with fill_rate between 0 and 1
- `GET /fill-rate/detail?had_partial_fulfillment=true` → all rows had_partial_fulfillment=True
- Pagination: limit=5 returns ≤5 rows
- Sort by shortage_qty desc → first row has highest shortage
- Empty DB → zeros returned, not 500
- `fill_rate = total_shipped / total_ordered` verified (assert abs(row.fill_rate - row.total_shipped/row.total_ordered) < 0.001)

### Unit Test Verification

In `tests/unit/test_fill_rate.py` (or inline):
- `fill_rate = NULL` when `total_ordered = 0` (not 0 or 1)
- `shortage_qty = GREATEST(ordered - shipped, 0)`: always ≥ 0
- `had_partial_fulfillment = True` when `shipped < ordered`
- `had_partial_fulfillment = False` when `shipped >= ordered`

### Frontend Tests: extend `InvPlanningTab.test.tsx`
- Fill rate panel renders with mock data
- Portfolio fill rate KPI card visible
- Trend chart renders with at least 1 month of data

---

## Acceptance Criteria

- [ ] `mv_fill_rate_monthly` populated after `make fill-rate-refresh`
- [ ] `fill_rate = NULL` for all rows where `qty_ordered = 0` (not present in view since filtered by WHERE)
- [ ] `fill_rate` is always between 0 and 1.0 (no negative, no >1 — implies returns > orders, edge case to handle)
- [ ] `shortage_qty ≥ 0` always (GREATEST enforces this)
- [ ] `portfolio_fill_rate = SUM(shipped) / SUM(ordered)` computed at summary level
- [ ] `GET /fill-rate/summary` shows by_abc with A/B/C keys
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/028_create_fill_rate_monthly.sql` | Create |
| `mvp/demand/api/routers/fill_rate.py` | Create |
| `mvp/demand/api/main.py` | Modify — mount fill_rate router |
| `mvp/demand/frontend/vite.config.ts` | Modify — add `/fill-rate` proxy |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Fill Rate panel |
| `mvp/demand/tests/api/test_fill_rate.py` | Create |
| `mvp/demand/Makefile` | Modify — add fill-rate-* targets |
| `docs/design-specs/IPfeature8.md` | Create (this file) |
