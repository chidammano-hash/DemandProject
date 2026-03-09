<!-- SOURCE: IPfeature8.md (Fill Rate) -->
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


---

<!-- SOURCE: IPfeature9.md (Demand Signals) -->
# IPfeature9 — Demand Sensing & Short-Horizon Signal Integration

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P3 — Nice to Have

## Effort
L (Large)

## Expert Perspectives
- **Demand Planning Expert** (lead) — intra-month sensing, velocity detection, demand signals
- **Inventory Planning Expert** — translating demand signals into replenishment urgency
- **Statistical Analyst** — projection math, signal strength, false-positive suppression

---

## Problem Statement

All current analytics (SS targets, health scores, exceptions) are based on **month-end snapshots**. By the time a month-end stockout is detected, it is too late to act.

The `fact_inventory_snapshot` table has **daily MTD sales data** (`mtd_sales`). If a DFU is running 40% above its forecast pace by day 15 of the month, a planner who acts today can place an emergency order. A planner who waits for month-end data sees a stockout 15 days from now.

Demand sensing converts daily MTD data into actionable forward-looking signals: "This item is tracking 35% above plan — projected stockout in 8 days."

---

## User Story

> As an inventory planner, I want intra-month demand velocity signals — above/below plan, projected month-end total, and days until projected stockout — computed from daily MTD sales and compared against the champion model forecast, so I can intervene before month-end reveals a stockout.

---

## Business Value

- Reduces stockout frequency by enabling proactive replenishment 2–4 weeks earlier
- Converts 190M-row daily snapshot table into actionable signals
- Provides early warning for promotions, seasonality spikes, and demand surges
- Feeds IPfeature15 (Control Tower) with urgent demand signal count

---

## Key Math

```
# Daily sense
days_in_month = calendar days in month_start
days_elapsed  = EXTRACT(day FROM signal_date)
days_remaining = days_in_month - days_elapsed

# MTD projection
mtd_actual        = latest mtd_sales for item-loc in current month
projected_monthly = mtd_actual × (days_in_month / days_elapsed)

# vs. plan
demand_vs_forecast_pct = (projected_monthly - champion_forecast) / champion_forecast × 100

# Signal type
signal_type:
  demand_vs_forecast_pct > +10%  → 'above_plan'
  demand_vs_forecast_pct < -10%  → 'below_plan'
  else                           → 'on_plan'

# Signal strength
signal_strength = ABS(demand_vs_forecast_pct) / 100

# Urgency
projected_stockout = projected_monthly > (eom_qty_on_hand / days_remaining × days_in_month)
  (simplified: if projected demand exceeds what's available at current rate)
alert_priority:
  projected_stockout AND is_below_ss → 'urgent'
  ABS(demand_vs_forecast_pct) > 20%  → 'watch'
  else                               → 'none'
```

---

## Data Requirements

### New DDL: `mvp/demand/sql/029_create_demand_signals.sql`

New table `fact_demand_signals`:

```sql
CREATE TABLE IF NOT EXISTS fact_demand_signals (
    signal_sk               BIGSERIAL PRIMARY KEY,
    item_no                 TEXT NOT NULL,
    loc                     TEXT NOT NULL,
    signal_date             DATE NOT NULL,
    month_start             DATE NOT NULL,
    day_of_month            INTEGER NOT NULL,
    days_elapsed            INTEGER NOT NULL,
    days_remaining          INTEGER NOT NULL,
    -- Current month progress
    mtd_actual              NUMERIC(15,4),
    mtd_expected            NUMERIC(15,4),      -- forecast_daily × days_elapsed
    projected_monthly       NUMERIC(15,4),
    historical_avg_monthly  NUMERIC(15,4),      -- same calendar month, 3-yr avg
    forecast_monthly        NUMERIC(15,4),      -- champion forecast for this month
    -- Signal metrics
    demand_vs_forecast_pct  NUMERIC(10,2),
    demand_acceleration     NUMERIC(10,4),      -- daily rate this month vs prior 30d
    signal_type             TEXT,               -- 'above_plan' | 'below_plan' | 'on_plan'
    signal_strength         NUMERIC(10,4),
    -- Inventory implication
    current_on_hand         NUMERIC(15,4),
    ss_combined             NUMERIC(15,4),
    is_below_ss             BOOLEAN,
    projected_stockout      BOOLEAN,
    projected_excess        BOOLEAN,
    alert_priority          TEXT,               -- 'urgent' | 'watch' | 'none'
    load_ts                 TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_demand_signals_pk
    ON fact_demand_signals (item_no, loc, signal_date);
CREATE INDEX IF NOT EXISTS idx_demand_signals_month
    ON fact_demand_signals (month_start);
CREATE INDEX IF NOT EXISTS idx_demand_signals_type
    ON fact_demand_signals (signal_type);
CREATE INDEX IF NOT EXISTS idx_demand_signals_urgent
    ON fact_demand_signals (alert_priority)
    WHERE alert_priority = 'urgent';
CREATE INDEX IF NOT EXISTS idx_demand_signals_date
    ON fact_demand_signals (signal_date DESC);
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/demand-signals
  Query params: signal_date (default: latest available), signal_type, alert_priority,
                item, location, abc_vol, limit, offset,
                sort_by (alert_priority | demand_vs_forecast_pct | signal_strength)
  Response: {
    signal_date: date,
    total: int,
    rows: [ {item_no, loc, signal_type, alert_priority, mtd_actual, projected_monthly,
             forecast_monthly, demand_vs_forecast_pct, projected_stockout, projected_excess,
             current_on_hand, is_below_ss, days_remaining} ]
  }
  Cache: max-age=3600s (computed once daily)

GET /inv-planning/demand-signals/summary
  Query params: signal_date (default: latest)
  Response: {
    signal_date: date,
    total_items_with_signals: int,
    above_plan: int,
    below_plan: int,
    on_plan: int,
    urgent_alerts: int,
    watch_alerts: int,
    projected_stockouts: int
  }
  Cache: max-age=3600s

GET /inv-planning/demand-signals/item
  Query params: item (required), location (required)
  Response: {
    item_no: str,
    loc: str,
    signal_date: date,
    signal_type: str,
    alert_priority: str,
    mtd_actual: float,
    projected_monthly: float,
    forecast_monthly: float,
    demand_vs_forecast_pct: float,
    days_elapsed: int,
    days_remaining: int,
    -- Daily MTD series for chart
    daily_series: [ {date, mtd_actual, mtd_expected_pace} ]
  }
  Cache: max-age=3600s
```

---

## Frontend UI

### Panel: "Demand Sensing" in `InvPlanningTab.tsx`

**Date Picker:** Select `signal_date` (defaults to latest date with signals)

**KPI Cards (row of 4):**
| Card | Value | Color |
|---|---|---|
| Above Plan | count (above_plan) | amber if >50 |
| Below Plan | count (below_plan) | blue |
| Urgent Alerts | count (alert_priority='urgent') | red if >0 |
| Projected Stockouts | count (projected_stockout=True) | red if >0 |

**Signal Scatter Plot:**
- X-axis: `demand_vs_forecast_pct` (-100% to +100%)
- Y-axis: `current_dos` (days of supply)
- Point color: urgent=red, watch=amber, none=gray
- Tooltip: item, loc, signal_type, projected_monthly vs. forecast_monthly
- Quadrant lines: x=0, y=safety_stock_days (reference)

**Urgent Items List:**
- Filtered to `alert_priority='urgent'`, sorted by `projected_stockout` then `demand_vs_forecast_pct` desc
- Columns: item, loc, mtd_actual, projected_monthly, forecast_monthly, demand_vs_forecast_pct, days_remaining, current_on_hand, alert_priority

**Per-Item Drill-Down (on row click):**
- Line chart: `mtd_actual` (solid) vs. `mtd_expected_pace` (dashed) — daily series
- Projected month-end marker on day 30/31
- Champion forecast as horizontal reference line
- Stockout zone: days where projected on-hand ≤ 0 highlighted red

---

## Backend Script

### `mvp/demand/scripts/compute_demand_signals.py`

```python
# Run once daily (scheduled via make demand-signals-compute or APScheduler)
# Algorithm:
# 1. Determine signal_date = CURRENT_DATE (or latest snapshot date)
# 2. Determine month_start = date_trunc('month', signal_date)
# 3. Query fact_inventory_snapshot:
#    SELECT item_no, loc, snapshot_date, mtd_sales, qty_on_hand
#    WHERE snapshot_date >= month_start AND snapshot_date <= signal_date
#    ORDER BY item_no, loc, snapshot_date
# 4. For each item-loc:
#    a. mtd_actual = max(mtd_sales) for the month (latest cumulative value)
#    b. day_of_month = EXTRACT(day FROM latest snapshot_date)
#    c. days_in_month = calendar days in month_start
#    d. projected_monthly = mtd_actual × (days_in_month / day_of_month)
#    e. Load champion forecast from fact_external_forecast_monthly
#       WHERE startdate = month_start AND model_id = 'champion'
#    f. demand_vs_forecast_pct = (projected - champion) / champion × 100
#    g. signal_type classification (±10% threshold)
#    h. Load ss_combined, is_below_ss from fact_safety_stock_targets
#    i. current_on_hand = latest qty_on_hand from snapshot
#    j. projected_stockout: projected daily demand × days_remaining > current_on_hand
#    k. alert_priority classification
#    l. historical_avg: query fact_sales_monthly for same calendar month in prior 3 years
#    m. demand_acceleration: daily_rate = mtd_actual/day_of_month vs prior 30-day avg
# 5. INSERT OR REPLACE INTO fact_demand_signals (replace today's signals if re-run)
```

**CLI Usage:**
```bash
uv run python scripts/compute_demand_signals.py
uv run python scripts/compute_demand_signals.py --signal-date 2026-02-15
```

**Makefile Targets:**
```makefile
demand-signals-schema:
	# apply sql/029_create_demand_signals.sql

demand-signals-compute:
	uv run python scripts/compute_demand_signals.py
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_inventory_snapshot` | Existing | Daily MTD sales data |
| `fact_external_forecast_monthly` | Existing | Champion forecast for month |
| `fact_safety_stock_targets` | IPfeature3 | ss_combined, is_below_ss |
| `fact_sales_monthly` | Existing | Historical avg same calendar month |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_demand_signals.py`

Minimum 12 tests:
- Projection: mtd_actual=50, day_of_month=15, days_in_month=31 → projected=50×(31/15)≈103.3
- Signal type: demand_vs_forecast_pct=+15% → 'above_plan'
- Signal type: demand_vs_forecast_pct=-5% → 'on_plan'
- Signal type: demand_vs_forecast_pct=-25% → 'below_plan'
- Alert priority: projected_stockout=True AND is_below_ss=True → 'urgent'
- Alert priority: demand_vs_forecast_pct=+22%, no stockout → 'watch'
- Alert priority: demand_vs_forecast_pct=+5% → 'none'
- champion forecast = 0 → demand_vs_forecast_pct = NULL (division by zero guard)
- Day 1 of month: day_of_month=1, projection is noisy — consider minimum day threshold (day ≥ 5)
- days_remaining = days_in_month - day_of_month

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_demand_signals.py`

Minimum 8 tests:
- `GET /inv-planning/demand-signals` → 200 OK, rows with demand_vs_forecast_pct
- `GET /inv-planning/demand-signals?alert_priority=urgent` → filtered
- `GET /inv-planning/demand-signals/summary` → all counts present and ≥ 0
- `GET /inv-planning/demand-signals/item?item=X&location=Y` → single item with daily_series
- `GET /inv-planning/demand-signals/item` (no params) → 422
- Empty day (no snapshots) → returns empty, not 500

---

## Acceptance Criteria

- [ ] `projected_monthly = mtd_actual × (days_in_month / day_of_month)` verified numerically
- [ ] Signal not computed for day_of_month < 5 (insufficient data)
- [ ] `demand_vs_forecast_pct > 10%` → `signal_type = 'above_plan'`
- [ ] `projected_stockout = True` AND `is_below_ss = True` → `alert_priority = 'urgent'`
- [ ] Daily MTD series chart renders in item drill-down
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/029_create_demand_signals.sql` | Create |
| `mvp/demand/scripts/compute_demand_signals.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add demand-signals endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Demand Sensing panel |
| `mvp/demand/tests/unit/test_demand_signals.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_demand_signals.py` | Create |
| `mvp/demand/Makefile` | Modify — add demand-signals-* targets |
| `docs/design-specs/IPfeature9.md` | Create (this file) |


---

<!-- SOURCE: IPfeature14.md (Intramonth Stockout) -->
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
