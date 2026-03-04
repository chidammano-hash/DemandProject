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
