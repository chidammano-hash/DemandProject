# IPfeature14 — Intra-Month Stockout Detection (Daily Granularity)

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P2 — Should Have

## Effort
L (Large)

## Expert Perspectives
- **Supply Chain Control Tower Expert** (lead) — daily visibility gaps, stockout duration KPIs
- **Demand Planning Expert** — lost sales imputation methodology, MTD sales derivation
- **UI/UX Expert** — daily on-hand trajectory drill-down, heatmap visualization

---

## Problem Statement

Feature 37 (inventory backtest) measures **end-of-month (EOM) stock**. An item with `eom_qty_on_hand > 0` is classified as "no stockout" — even if it was out of stock for 20 of 30 days in that month.

This is the **CSL vs. fill-rate vs. intra-month duration gap**:

> A DFU that depletes to zero on the 5th of the month, receives replenishment on the 28th, and ends with 500 units on hand shows: EOM stockout = FALSE, CSL = 1.0 — but the customer-facing impact was enormous.

Daily granularity from `fact_inventory_snapshot` (190M rows) enables:
- `stockout_days` = days with `qty_on_hand <= 0`
- `stockout_day_rate` = fraction of the month in stockout
- `est_lost_sales` = imputed from daily MTD sales reconstruction via LAG

This is the **true operational service level** — what the warehouse actually experienced.

---

## User Story

> As a planner, I want to see which item-locations were out of stock at any point during a month — including the number of stockout days, stockout day rate, and estimated lost sales — so I can see the true service impact that EOM snapshots hide.

---

## Business Value

- Surfaces hidden service failures invisible to EOM-based CSL measurement
- Enables "stockout days per month per DFU" as a KPI alongside fill rate
- Identifies items with chronic short-duration stockouts (e.g. weekly stockout then replenish)
- Feeds IPfeature15 (Control Tower) with intra-month stockout KPIs
- Supports commercial conversations: "We were out of stock 40% of the month for this item"

---

## Key Metrics & Formulas

```
# Per item-loc-month from daily fact_inventory_snapshot

snapshot_days    = COUNT(DISTINCT snapshot_date)
stockout_days    = COUNT(DISTINCT snapshot_date WHERE qty_on_hand <= 0)
stockout_day_rate = stockout_days / snapshot_days     (0.0 to 1.0)

# Daily MTD sales reconstruction (cumulative to incremental)
daily_sls[day]  = mtd_sales[day] - mtd_sales[day-1]    (LAG window)
                   -- first day of month: daily_sls = mtd_sales[1]
                   -- clamp to >= 0 (handle data corrections)

est_lost_sales   = SUM(daily_sls WHERE qty_on_hand <= 0)
                   -- imputed NOT actual: daily sales on stockout days may be
                   -- underreported; this captures only known demand gaps

had_full_stockout     = TRUE if stockout_days >= 1 (any stockout day)
had_extended_stockout = TRUE if stockout_days >= 7 (week or more)

min_qty_on_hand  = MIN(qty_on_hand) per month    -- lowest point
max_qty_on_hand  = MAX(qty_on_hand) per month
avg_qty_on_hand  = AVG(qty_on_hand) per month
```

**Imputation note:** `est_lost_sales` is an underestimate. When an item is out of stock, `mtd_sales` may still accrue from backorders or manual adjustments. The LAG difference represents the observable demand signal on that day — not a perfect lost sales figure. It should be labeled "Estimated" in the UI.

---

## Data Requirements

### New DDL: `mvp/demand/sql/034_create_intramonth_stockout.sql`

New materialized view `mv_intramonth_stockout`:

```sql
CREATE MATERIALIZED VIEW mv_intramonth_stockout AS
WITH daily_with_lag AS (
    SELECT
        item_no,
        loc,
        DATE_TRUNC('month', snapshot_date)::DATE AS month_start,
        snapshot_date,
        qty_on_hand,
        mtd_sales,
        -- Daily incremental sales from cumulative MTD (LAG within same item-loc-month)
        GREATEST(
            mtd_sales
            - LAG(mtd_sales, 1, 0::NUMERIC) OVER (
                PARTITION BY item_no, loc, DATE_TRUNC('month', snapshot_date)
                ORDER BY snapshot_date
            ),
            0
        ) AS daily_sls
    FROM fact_inventory_snapshot
),
monthly_agg AS (
    SELECT
        item_no,
        loc,
        month_start,
        COUNT(*)                                          AS snapshot_days,
        COUNT(*) FILTER (WHERE qty_on_hand <= 0)         AS stockout_days,
        CASE WHEN COUNT(*) > 0
             THEN COUNT(*) FILTER (WHERE qty_on_hand <= 0)::NUMERIC / COUNT(*)
             ELSE NULL END                               AS stockout_day_rate,
        MIN(qty_on_hand)                                 AS min_qty_on_hand,
        MAX(qty_on_hand)                                 AS max_qty_on_hand,
        AVG(qty_on_hand)                                 AS avg_qty_on_hand,
        SUM(daily_sls) FILTER (WHERE qty_on_hand <= 0)  AS est_lost_sales,
        (COUNT(*) FILTER (WHERE qty_on_hand <= 0)) >= 1  AS had_full_stockout,
        (COUNT(*) FILTER (WHERE qty_on_hand <= 0)) >= 7  AS had_extended_stockout
    FROM daily_with_lag
    GROUP BY item_no, loc, month_start
)
SELECT
    m.item_no,
    m.loc,
    m.month_start,
    m.snapshot_days,
    m.stockout_days,
    m.stockout_day_rate,
    m.min_qty_on_hand,
    m.max_qty_on_hand,
    m.avg_qty_on_hand,
    m.est_lost_sales,
    m.had_full_stockout,
    m.had_extended_stockout,
    -- DFU attributes for slicing
    COALESCE(d.abc_vol, '(unknown)')              AS abc_vol,
    COALESCE(d.abc_xyz_segment, '(unknown)')      AS abc_xyz_segment,
    COALESCE(d.cluster_assignment, '(unassigned)') AS cluster_assignment,
    d.variability_class
FROM monthly_agg m
LEFT JOIN dim_dfu d
    ON m.item_no = d.dmdunit
    AND m.loc = d.loc
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_intramonth_pk
    ON mv_intramonth_stockout (item_no, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_intramonth_month
    ON mv_intramonth_stockout (month_start DESC);
CREATE INDEX IF NOT EXISTS idx_intramonth_abc
    ON mv_intramonth_stockout (abc_vol, month_start);
CREATE INDEX IF NOT EXISTS idx_intramonth_extended
    ON mv_intramonth_stockout (had_extended_stockout)
    WHERE had_extended_stockout = TRUE;
CREATE INDEX IF NOT EXISTS idx_intramonth_high_rate
    ON mv_intramonth_stockout (stockout_day_rate DESC)
    WHERE stockout_day_rate > 0.1;
```

**⚠ Performance Warning:** This view processes 190M rows across a LAG window partitioned by item_no + loc + month. Full refresh takes 10–30 minutes. Implement **incremental refresh** — refreshing only the current month + prior month each daily run:

```sql
-- Incremental refresh approach (run from scripts/refresh_intramonth_stockout.py):
-- 1. DELETE from mv_intramonth_stockout WHERE month_start >= CURRENT_DATE - INTERVAL '35 days'
-- 2. INSERT INTO mv_intramonth_stockout (re-compute only recent months from raw table)
-- OR: Use REFRESH MATERIALIZED VIEW CONCURRENTLY but limit source table by date filter
-- The script implements a manual upsert refresh for the rolling 2-month window.
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/intramonth-stockouts/summary
  Query params: month_from, month_to, item, location, abc_vol, had_extended_stockout (bool)
  Response: {
    total_item_months: int,
    total_stockout_days: int,             -- sum across all items
    items_with_any_stockout: int,         -- distinct item-locs with stockout_days >= 1
    items_with_extended_stockout: int,    -- stockout_days >= 7
    est_total_lost_sales: float,
    avg_stockout_day_rate: float,         -- portfolio avg across items with stockouts
    by_abc: {
      A: { items_with_stockout, avg_stockout_day_rate, est_lost_sales },
      B: { ... },
      C: { ... }
    },
    worst_month: { month_start, total_stockout_days, items_affected },
    trend: [ {month_start, items_with_stockout, total_stockout_days, est_lost_sales} ]
  }
  Cache: max-age=3600s

GET /inv-planning/intramonth-stockouts/detail
  Query params: month_from, month_to, item, location, abc_vol,
                had_extended_stockout (bool), min_stockout_day_rate (float),
                limit, offset, sort_by (stockout_day_rate | stockout_days | est_lost_sales), sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, month_start, abc_vol, abc_xyz_segment,
             snapshot_days, stockout_days, stockout_day_rate,
             min_qty_on_hand, avg_qty_on_hand, est_lost_sales,
             had_full_stockout, had_extended_stockout} ]
  }
  Cache: max-age=3600s

GET /inv-planning/intramonth-stockouts/daily
  Query params: item (required), location (required), month (required, format: YYYY-MM)
  Response: {
    item_no: str,
    loc: str,
    month_start: date,
    days: [
      { date, qty_on_hand, daily_sls, is_stockout }
      -- one row per snapshot_date in that month for this item-loc
    ],
    summary: { stockout_days, stockout_day_rate, est_lost_sales }
  }
  Cache: max-age=3600s
  -- NOTE: Queries fact_inventory_snapshot directly (not the mat view) for daily detail
  -- Uses LAG() inline for daily_sls reconstruction
```

---

## Frontend UI

### Panel: "Intra-Month Stockouts" in `InvPlanningTab.tsx`

**KPI Cards (row of 4):**
| Card | Value | Threshold |
|---|---|---|
| Total Stockout Days | sum(stockout_days) across portfolio in period | Red if > 0 |
| Extended Stockouts | items_with_extended_stockout count (≥7 days) | Red if > 0, amber if > 10 |
| Est. Lost Sales | est_total_lost_sales units | Red if > 0 |
| Avg Stockout Day Rate | avg(stockout_day_rate) × 100% | Green <5%, amber 5-20%, red >20% |

**Monthly Trend Bar Chart:**
- X-axis: month_start
- Bars: total_stockout_days (stacked by abc_vol: A=red, B=orange, C=yellow)
- Line (secondary Y): items_with_stockout count
- Reference: 0 stockout days = ideal

**ABC Class × Month Heatmap:**
- Rows: month_start (last 6 months)
- Columns: A, B, C, (unknown)
- Cell: avg_stockout_day_rate × 100%
- Color: white (0%) → amber (10%) → red (>20%)
- Click cell → filter detail table by that ABC + month

**Intra-Month Detail Table:**
- Sort by: stockout_day_rate descending (worst items first)
- Columns: item, loc, month, abc_vol, stockout_days, stockout_day_rate (%), est_lost_sales, min_on_hand, extended? (boolean badge)
- Row color: red if had_extended_stockout, amber if had_full_stockout, white otherwise
- Filter: had_extended_stockout checkbox, abc_vol dropdown, month range picker

**Daily Drill-Down Chart (on row click):**
- Line chart: daily `qty_on_hand` over the selected month
- Stockout zones: days with qty_on_hand <= 0 highlighted with red background band
- Bar overlay (secondary Y): `daily_sls` — estimated daily demand
- Tooltip: date, qty_on_hand, daily_sls, is_stockout flag
- Shows recovery: when replenishment arrived and inventory recovered
- Breadcrumb: "Back to summary" link

---

## Backend Script

### `mvp/demand/scripts/refresh_intramonth_stockout.py`

```python
# Incremental refresh: recompute only the rolling 2-month window
# (current month + prior month), avoiding full 190M-row scan every day.
#
# Algorithm:
# 1. Determine current_month = DATE_TRUNC('month', CURRENT_DATE)
# 2. prior_month = current_month - INTERVAL '1 month'
# 3. DELETE FROM mv_intramonth_stockout
#    WHERE month_start >= prior_month
# 4. INSERT INTO mv_intramonth_stockout
#    (run the full materialized view SQL but with WHERE clause:
#     DATE_TRUNC('month', snapshot_date) >= prior_month)
# 5. Log: rows inserted, rows deleted, timing
#
# Full refresh (one-time or monthly):
# REFRESH MATERIALIZED VIEW mv_intramonth_stockout;
# (takes 10-30 min but produces complete history)
#
# CLI:
# uv run python scripts/refresh_intramonth_stockout.py              # incremental (default)
# uv run python scripts/refresh_intramonth_stockout.py --full       # full refresh
# uv run python scripts/refresh_intramonth_stockout.py --months 2   # rolling N months
```

**Makefile Targets:**
```makefile
intramonth-schema:
	# apply sql/034_create_intramonth_stockout.sql (CREATE MAT VIEW WITH NO DATA)

intramonth-refresh:
	uv run python scripts/refresh_intramonth_stockout.py
	# Default: incremental (current + prior month, fast)

intramonth-refresh-full:
	uv run python scripts/refresh_intramonth_stockout.py --full
	# Warning: 10-30 min, touches 190M rows
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_inventory_snapshot` | Existing | 190M daily rows — primary source |
| `dim_dfu` | Existing | abc_vol, abc_xyz_segment, variability_class for slicing |
| `mv_inventory_forecast_monthly` | Feature 37 | EOM CSL comparison context |
| IPfeature11 `abc_xyz_segment` | IPfeature11 | For cell-level slicing |
| No SS/EOQ pipeline required | — | Independent of SS computation |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_intramonth_stockout.py`

Minimum 8 tests:
- `daily_sls = mtd_sales[day] - mtd_sales[day-1]`: correct incremental derivation
- First day of month: `daily_sls = mtd_sales` (no LAG available → defaults to 0 LAG)
- Negative daily_sls clamped to 0 (data correction scenario)
- `stockout_days = COUNT(days where qty_on_hand <= 0)`: verified numerically
- `stockout_day_rate = stockout_days / snapshot_days`: in range [0.0, 1.0]
- `had_extended_stockout = True` iff `stockout_days >= 7`
- `had_extended_stockout = False` if `stockout_days = 6` (exactly at boundary)
- `est_lost_sales`: only daily_sls on stockout days summed (non-stockout days excluded)

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_intramonth.py`

Minimum 10 tests:
- `GET /inv-planning/intramonth-stockouts/summary` → 200 OK, trend is list
- `GET /inv-planning/intramonth-stockouts/summary?abc_vol=A` → by_abc.A present
- `GET /inv-planning/intramonth-stockouts/detail` → rows with stockout_day_rate in [0, 1]
- `GET /inv-planning/intramonth-stockouts/detail?had_extended_stockout=true` → all rows had_extended_stockout=True
- Sort by stockout_day_rate desc → first row has highest rate
- Pagination: limit=5 returns ≤5 rows
- `GET /inv-planning/intramonth-stockouts/daily?item=X&location=Y&month=2024-03` → days list
- Daily response: is_stockout = True for days where qty_on_hand <= 0
- Empty DB → zeros returned, not 500
- `stockout_day_rate` verified: `assert abs(row.stockout_day_rate - row.stockout_days/row.snapshot_days) < 0.001`

### Frontend Tests: extend `InvPlanningTab.test.tsx`
- Intra-month panel renders 4 KPI cards
- Monthly trend bar chart renders
- Detail table renders with stockout_day_rate column
- Row click drill-down shows daily chart

---

## Acceptance Criteria

- [ ] `stockout_day_rate = stockout_days / snapshot_days` (between 0 and 1)
- [ ] `had_extended_stockout = TRUE` iff `stockout_days >= 7`
- [ ] `est_lost_sales >= 0` always (no negative values — clamped by GREATEST)
- [ ] Daily drill-down chart renders with stockout zones highlighted red
- [ ] Incremental refresh (default) processes only 2 months, completes in < 2 min
- [ ] Full refresh available via `--full` flag for complete history rebuild
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/034_create_intramonth_stockout.sql` | Create |
| `mvp/demand/scripts/refresh_intramonth_stockout.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add intramonth endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Intra-Month Stockouts panel |
| `mvp/demand/tests/unit/test_intramonth_stockout.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_intramonth.py` | Create |
| `mvp/demand/Makefile` | Modify — add intramonth-* targets |
| `docs/design-specs/IPfeature14.md` | Create (this file) |
