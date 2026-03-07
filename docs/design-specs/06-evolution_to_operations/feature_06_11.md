# Feature F3.4 — Demand Sensing Integration (Short-Horizon Override)

**Phase:** 3 — Operational Feedback Loop
**Feature Number:** F3.4 (file: feature_06_11)
**Status:** Design / Not Started
**Depends On:** IPfeature9 (Demand Signals), IPfeature8 (Fill Rate), Safety Stock (IPfeature3)

---

## 1. Problem Statement

The current `fact_demand_signals` table (IPfeature9) computes a short-horizon demand signal from daily inventory velocity. However, this signal is informational only — it is **never used to override the monthly statistical forecast** in the planning engine. The monthly ML model (LGBM/CatBoost/XGBoost, trained on 2+ years of history) is blind to a sales velocity spike that happened two weeks ago.

### What Fails Today

**Concrete example:** Item 100320, April 2026. The statistical model (champion) forecasts 450 units for April, based on historical April averages. By April 8th, actual sales are 145 units in 8 days — a run rate of 18.1 units/day vs the historical 15 units/day (+21% velocity).

```
┌────────────────────────────────────────────────────────────────────────┐
│  APRIL 2026 — DAY 8 OF MONTH                                           │
│                                                                         │
│  Statistical forecast (April total):  450 units  ◄── ML model sees     │
│                                                       2023-2025 history │
│                                                                         │
│  Actual sales so far:                 145 units  ◄── FACT: happening   │
│  Run rate:                           18.1 u/day       right now        │
│  Historical run rate:                15.0 u/day                        │
│  Velocity deviation:                   +21%                            │
│                                                                         │
│  Projected April total (sensing):    543 units                         │
│                                                                         │
│  GAP vs statistical forecast:         +93 units (+21%)                 │
│                                                                         │
│  PROBLEM: The planning engine still uses 450 units for:                 │
│    - Reorder point calculation:  ROP = 450/30 × LT_days + SS          │
│    - Replenishment order trigger: current inventory vs 450-unit demand  │
│    - Exception generation:        no alert because 450 units looks fine │
│                                                                         │
│  RESULT: Stockout risk is NOT detected until mid-April when actual      │
│          inventory drops below the 450-unit-based ROP. By then it is   │
│          too late — lead time is 14 days.                               │
└────────────────────────────────────────────────────────────────────────┘
```

The system currently stores the sensing signal (IPfeature9) but the demand planning pipeline, reorder points, and exception engine all read from the statistical forecast table. The sensing signal is disconnected.

---

## 2. Demand Sensing Theory and Blending Approach

### Why Sensing Beats Statistical in the Near Term

Statistical models optimize for long-horizon accuracy (minimize WAPE over 1-12 months). At horizon T+1 weeks, a simple velocity extrapolation from current MTD sales consistently outperforms the statistical model because:

1. The ML model cannot see sales that occurred AFTER its training cutoff
2. Demand regime changes (promotions, competitor stockouts, weather, viral trends) take months to appear in training data
3. Inventory velocity is a leading indicator of demand surge — it is measured daily

**Accuracy comparison by horizon (empirical from demand sensing literature):**

| Horizon | Sensing Signal WAPE | Statistical Model WAPE |
|---------|--------------------|-----------------------|
| Week 1  | ~8-12%             | ~22-28%               |
| Week 2  | ~10-15%            | ~20-25%               |
| Week 3  | ~14-18%            | ~18-23%               |
| Week 4  | ~18-22%            | ~17-22%               |
| Week 5+ | ~22-28%            | ~15-20%               |

Sensing is superior through week ~4. Beyond that, the statistical model is at parity or better.

### The Blending Formula

```
blended_forecast[week t] = α(t) × sensing_signal[t] + (1 - α(t)) × statistical_forecast[t]

where:
  α(t) = max(0, 1 - (t - 1) / (sensing_horizon_weeks - 1))

  t = 1 (current week):  α = 1.00  →  100% sensing
  t = 2:                 α = 0.67  →   67% sensing, 33% statistical
  t = 3:                 α = 0.33  →   33% sensing, 67% statistical
  t = 4:                 α = 0.00  →  100% statistical (sensing horizon ends)
  t = 5+:                α = 0.00  →  100% statistical
```

**Visual representation of blend weights:**

```
  α (sensing weight)
  1.0 │●
      │  ●
  0.6 │     ●
      │
  0.3 │         ●
      │
  0.0 │─────────────●──────────────────────────────────  statistical only
      └──────────────────────────────────────────────
        W1   W2   W3   W4   W5   W6   W7   W8

  ░░░ = Sensing contribution  ████ = Statistical contribution

  W1: ████████████████████████  100% sensing
  W2: ████████████████░░░░░░░░   67% sensing, 33% statistical
  W3: ████████░░░░░░░░░░░░░░░░   33% sensing, 67% statistical
  W4: ░░░░░░░░░░░░░░░░░░░░░░░░    0% sensing, 100% statistical
```

---

## 3. Velocity Signal Computation

### From Daily Inventory Snapshots to Weekly Sensing Signal

**Step 1: Compute daily run rate from MTD sales**

```python
# Available from fact_inventory_snapshot at any point in month
days_elapsed   = current_day_of_month      # e.g., 8 on April 8
mtd_sales      = 145 units                 # from mtd_sales column
daily_run_rate = mtd_sales / days_elapsed  # = 145 / 8 = 18.125 units/day
```

**Step 2: Validate run rate (outlier guard)**

```python
# Compare to historical average daily demand
historical_avg_daily = 15.0 units/day  # from agg_sales_monthly / 30
spike_ratio          = 18.125 / 15.0   # = 1.208

# Apply outlier guard: if spike_ratio > velocity_outlier_threshold (3.0σ), cap it
# 1.208 < 3.0 → sensing signal is valid, no capping needed
```

**Step 3: Project sensing signal forward**

```python
# Sensing signal for the REMAINDER of the current month
days_remaining_in_month = 30 - days_elapsed  # = 22 days
remaining_demand_sensing = daily_run_rate * days_remaining_in_month  # = 18.125 × 22 = 398.75

# Full-month projection
april_sensing_total = mtd_sales + remaining_demand_sensing  # = 145 + 398.75 = 543.75 units
```

**Step 4: Disaggregate to weekly grain**

Monthly sensing projection → weekly by applying day-of-week proportions:

```
Historical DOW proportions for this item-location:
  Monday:    12%   Tuesday:  15%   Wednesday: 16%
  Thursday:  14%   Friday:   18%   Saturday:  15%   Sunday:  10%

Weekly sensing signal for week starting April 7 (Mon-Sun):
  Week 1 sensing = 543.75 / 4.33 weeks × DOW_factor ≈ 125.6 units/week
```

---

## 4. Data Model

### 4.1 `fact_blended_demand_plan` — Weekly Blended Forecast

**Grain:** item_no + loc + week_start + plan_version (one row per item-location per week per version)

```sql
CREATE TABLE fact_blended_demand_plan (
    blend_id                BIGSERIAL        PRIMARY KEY,
    item_no                 VARCHAR(50)      NOT NULL,
    loc                     VARCHAR(50)      NOT NULL,
    week_start              DATE             NOT NULL,  -- Monday of the week (ISO 8601 week start)
    plan_version            VARCHAR(50)      NOT NULL,  -- e.g., "2026-04-08_v1"
    sensing_horizon_weeks   INTEGER          NOT NULL DEFAULT 4,
    week_offset             INTEGER          NOT NULL,  -- 1=current week, 2=next week, ...
    alpha_weight            NUMERIC(5,4)     NOT NULL,  -- 0.0000 to 1.0000
    sensing_signal_qty      NUMERIC(12,2),              -- NULL if outside sensing horizon
    statistical_forecast_qty NUMERIC(12,2)  NOT NULL,
    blended_qty             NUMERIC(12,2)   NOT NULL,
    sensing_active          BOOLEAN          NOT NULL DEFAULT FALSE,
    sensing_override_reason VARCHAR(100),              -- velocity_spike / promotion / manual
    velocity_run_rate       NUMERIC(10,4),             -- units/day at time of computation
    velocity_spike_ratio    NUMERIC(6,3),              -- sensing / historical rate
    days_elapsed_in_month   INTEGER,
    mtd_sales_actual        NUMERIC(12,2),
    dow_adjustment_factor   NUMERIC(5,4),             -- day-of-week seasonality applied
    is_outlier_capped       BOOLEAN         NOT NULL DEFAULT FALSE,  -- spike was capped
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_blend_plan_item_loc_week_version
    ON fact_blended_demand_plan(item_no, loc, week_start, plan_version);

CREATE INDEX idx_blend_plan_item_loc
    ON fact_blended_demand_plan(item_no, loc);

CREATE INDEX idx_blend_plan_week
    ON fact_blended_demand_plan(week_start DESC);

CREATE INDEX idx_blend_plan_sensing_active
    ON fact_blended_demand_plan(sensing_active, week_start)
    WHERE sensing_active = TRUE;

CREATE INDEX idx_blend_plan_version
    ON fact_blended_demand_plan(plan_version);
```

### 4.2 `mv_sensing_overrides_active` — Current DFUs Where Sensing Is Overriding Statistical

```sql
CREATE MATERIALIZED VIEW mv_sensing_overrides_active AS
WITH latest_version AS (
    SELECT item_no, loc, MAX(plan_version) AS latest_version
    FROM fact_blended_demand_plan
    GROUP BY item_no, loc
)
SELECT
    b.item_no, b.loc,
    b.week_start AS current_week,
    b.sensing_signal_qty,
    b.statistical_forecast_qty,
    b.blended_qty,
    b.alpha_weight,
    b.velocity_spike_ratio,
    b.sensing_override_reason,
    d.abc_class,
    d.cluster_assignment,
    ABS(b.blended_qty - b.statistical_forecast_qty) / NULLIF(b.statistical_forecast_qty, 0) * 100
        AS pct_deviation_from_statistical
FROM fact_blended_demand_plan b
JOIN latest_version lv USING (item_no, loc)
JOIN dim_dfu d USING (item_no, loc)
WHERE b.plan_version = lv.latest_version
  AND b.week_offset = 1  -- current week only
  AND b.sensing_active = TRUE
  AND b.velocity_spike_ratio > 1.10;  -- only show meaningful overrides

CREATE UNIQUE INDEX uq_sensing_active ON mv_sensing_overrides_active(item_no, loc);
CREATE INDEX idx_sensing_active_spike ON mv_sensing_overrides_active(velocity_spike_ratio DESC);
```

---

## 5. Python Script

### `scripts/compute_blended_forecast.py`

```python
#!/usr/bin/env python3
"""
compute_blended_forecast.py

Reads fact_demand_signals (short-horizon sensing) and fact_external_forecast_monthly
(statistical model) to produce fact_blended_demand_plan with weekly blending.

Usage:
    uv run scripts/compute_blended_forecast.py
    uv run scripts/compute_blended_forecast.py --item-no 100320 --loc 1401-BULK
    uv run scripts/compute_blended_forecast.py --dry-run
"""

import argparse
import logging
import math
import yaml
from datetime import date, timedelta
from typing import Optional
import psycopg
from common.db import get_db_params

CONFIG_PATH = "config/demand_sensing_config.yaml"
log = logging.getLogger(__name__)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_alpha(week_offset: int, sensing_horizon_weeks: int) -> float:
    """
    Linear decay of sensing weight from 1.0 at week 1 to 0.0 at sensing_horizon_weeks.

    α(t) = max(0, 1 - (t - 1) / (sensing_horizon_weeks - 1))

    Examples (4-week horizon):
        week_offset=1 → α=1.000 (100% sensing)
        week_offset=2 → α=0.667
        week_offset=3 → α=0.333
        week_offset=4 → α=0.000 (100% statistical)
        week_offset=5 → α=0.000

    Args:
        week_offset: 1 = current week, 2 = next week, etc.
        sensing_horizon_weeks: Number of weeks sensing is active (default 4).

    Returns:
        Alpha weight between 0.0 and 1.0.
    """
    if sensing_horizon_weeks <= 1:
        return 1.0 if week_offset == 1 else 0.0
    alpha = max(0.0, 1.0 - (week_offset - 1) / (sensing_horizon_weeks - 1))
    return round(alpha, 4)


def compute_velocity_signal(
    mtd_sales: float,
    days_elapsed: int,
    days_in_month: int,
    historical_daily_avg: float,
    outlier_threshold: float,
) -> tuple[float, float, float, bool]:
    """
    Compute daily run rate, projected monthly total from sensing, spike ratio,
    and whether an outlier cap was applied.

    Args:
        mtd_sales: Month-to-date actual sales units.
        days_elapsed: Number of days elapsed in the current month.
        days_in_month: Total days in the month (28/29/30/31).
        historical_daily_avg: Historical average daily demand for this DFU.
        outlier_threshold: Spike ratio above which sensing is capped.

    Returns:
        Tuple of (projected_monthly_total, daily_run_rate, spike_ratio, is_capped).

    Raises:
        ValueError: If days_elapsed < 1.
    """
    if days_elapsed < 1:
        raise ValueError("days_elapsed must be >= 1")

    daily_run_rate = mtd_sales / days_elapsed
    spike_ratio    = daily_run_rate / max(historical_daily_avg, 0.001)
    is_capped      = False

    if spike_ratio > outlier_threshold:
        # Cap at outlier_threshold × historical rate to avoid noise driving orders
        daily_run_rate = historical_daily_avg * outlier_threshold
        is_capped      = True
        spike_ratio    = outlier_threshold

    days_remaining         = days_in_month - days_elapsed
    remaining_sensing      = daily_run_rate * days_remaining
    projected_monthly_total = mtd_sales + remaining_sensing

    return projected_monthly_total, daily_run_rate, spike_ratio, is_capped


def monthly_to_weekly(monthly_qty: float, n_weeks_in_month: float = 4.33) -> float:
    """Convert monthly quantity to average weekly quantity."""
    return monthly_qty / n_weeks_in_month


def apply_dow_factor(weekly_qty: float, dow_factor: float = 1.0) -> float:
    """
    Apply day-of-week seasonality adjustment to weekly qty.
    dow_factor is the ratio of this week's expected demand to average weekly demand,
    derived from historical sales proportions. Default 1.0 = no adjustment.
    """
    return round(weekly_qty * dow_factor, 2)


def fetch_sensing_data(conn, item_no: Optional[str], loc: Optional[str]) -> list[dict]:
    """Pull latest demand signals from fact_demand_signals."""
    sql = """
        SELECT ds.item_no, ds.loc, ds.signal_date,
               ds.velocity_7d, ds.mtd_sales,
               agg.avg_daily_sales AS historical_daily_avg,
               d.abc_class
        FROM fact_demand_signals ds
        JOIN (
            SELECT item_no, loc, AVG(avg_daily_sales) AS avg_daily_sales
            FROM agg_inventory_monthly
            WHERE month_start >= CURRENT_DATE - INTERVAL '6 months'
            GROUP BY item_no, loc
        ) agg USING (item_no, loc)
        LEFT JOIN dim_dfu d USING (item_no, loc)
        WHERE ds.signal_date = (
            SELECT MAX(signal_date) FROM fact_demand_signals
        )
          AND (%s IS NULL OR ds.item_no = %s)
          AND (%s IS NULL OR ds.loc     = %s)
    """
    cur = conn.execute(sql, (item_no, item_no, loc, loc))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_statistical_forecast(conn, item_no: str, loc: str, months_ahead: int = 3) -> list[dict]:
    """Pull champion statistical forecast for the next N months."""
    sql = """
        SELECT startdate, basefcst_pref AS statistical_qty
        FROM fact_external_forecast_monthly
        WHERE dmdunit = %s AND loc = %s
          AND model_id = 'champion'
          AND startdate >= DATE_TRUNC('month', CURRENT_DATE)
          AND startdate < DATE_TRUNC('month', CURRENT_DATE) + (%s * INTERVAL '1 month')
        ORDER BY startdate
    """
    cur = conn.execute(sql, (item_no, loc, months_ahead))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def upsert_blend_rows(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO fact_blended_demand_plan (
            item_no, loc, week_start, plan_version, sensing_horizon_weeks,
            week_offset, alpha_weight, sensing_signal_qty, statistical_forecast_qty,
            blended_qty, sensing_active, sensing_override_reason,
            velocity_run_rate, velocity_spike_ratio, days_elapsed_in_month,
            mtd_sales_actual, is_outlier_capped, computed_at
        ) VALUES (
            %(item_no)s, %(loc)s, %(week_start)s, %(plan_version)s, %(sensing_horizon_weeks)s,
            %(week_offset)s, %(alpha_weight)s, %(sensing_signal_qty)s, %(statistical_forecast_qty)s,
            %(blended_qty)s, %(sensing_active)s, %(sensing_override_reason)s,
            %(velocity_run_rate)s, %(velocity_spike_ratio)s, %(days_elapsed_in_month)s,
            %(mtd_sales_actual)s, %(is_outlier_capped)s, NOW()
        )
        ON CONFLICT (item_no, loc, week_start, plan_version)
        DO UPDATE SET
            blended_qty              = EXCLUDED.blended_qty,
            alpha_weight             = EXCLUDED.alpha_weight,
            sensing_signal_qty       = EXCLUDED.sensing_signal_qty,
            sensing_active           = EXCLUDED.sensing_active,
            velocity_spike_ratio     = EXCLUDED.velocity_spike_ratio,
            computed_at              = NOW()
    """
    for row in rows:
        conn.execute(sql, row)
    return len(rows)


def run(
    item_no: Optional[str] = None,
    loc: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    cfg = load_config()
    sensing_horizon_weeks   = cfg.get("sensing_horizon_weeks", 4)
    outlier_threshold       = cfg.get("velocity_outlier_threshold", 3.0)
    min_velocity_days       = cfg.get("min_velocity_days", 5)
    alert_deviation_pct     = cfg.get("alert_spike_threshold_pct", 25.0)
    plan_version            = date.today().isoformat() + "_v1"
    today                   = date.today()
    # ISO Monday of current week
    current_week_start      = today - timedelta(days=today.weekday())

    log.info("Computing blended demand plan (version=%s, dry_run=%s)", plan_version, dry_run)

    with psycopg.connect(**get_db_params()) as conn:
        sensing_rows = fetch_sensing_data(conn, item_no, loc)
        log.info("Processing %d DFUs with sensing signals", len(sensing_rows))

        total_written = 0
        for sr in sensing_rows:
            item    = sr["item_no"]
            loc_val = sr["loc"]

            # Compute days elapsed in current month
            days_elapsed = today.day
            if days_elapsed < min_velocity_days:
                log.debug("Skipping %s/%s — only %d days elapsed (min=%d)",
                          item, loc_val, days_elapsed, min_velocity_days)
                continue

            # Days in current month
            next_month      = date(today.year, today.month % 12 + 1, 1) if today.month < 12 else date(today.year + 1, 1, 1)
            days_in_month   = (next_month - date(today.year, today.month, 1)).days

            historical_daily_avg = float(sr.get("historical_daily_avg") or 0)
            if historical_daily_avg <= 0:
                continue

            mtd_sales = float(sr.get("mtd_sales") or 0)
            projected_monthly, daily_rate, spike_ratio, is_capped = compute_velocity_signal(
                mtd_sales=mtd_sales,
                days_elapsed=days_elapsed,
                days_in_month=days_in_month,
                historical_daily_avg=historical_daily_avg,
                outlier_threshold=outlier_threshold,
            )

            # Fetch statistical forecast
            stat_rows = fetch_statistical_forecast(conn, item, loc_val, months_ahead=3)
            stat_map  = {r["startdate"]: float(r["statistical_qty"] or 0) for r in stat_rows}

            # Current month statistical forecast
            current_month_stat = stat_map.get(date(today.year, today.month, 1), historical_daily_avg * 30)
            sensing_weekly     = monthly_to_weekly(projected_monthly)
            stat_weekly        = monthly_to_weekly(current_month_stat)

            # Generate 8 weeks of blended plan
            blend_rows = []
            for week_offset in range(1, 9):
                week_start = current_week_start + timedelta(weeks=week_offset - 1)
                alpha       = compute_alpha(week_offset, sensing_horizon_weeks)
                sensing_qty = sensing_weekly if alpha > 0 else None
                stat_qty    = stat_weekly  # could extend to next month stat for weeks 5+
                blended_qty = round(
                    (alpha * (sensing_qty or 0)) + ((1 - alpha) * stat_qty), 2
                )
                sensing_active = alpha > 0
                override_reason = None
                if sensing_active and spike_ratio > 1.10:
                    override_reason = "velocity_spike"
                elif sensing_active and spike_ratio < 0.90:
                    override_reason = "velocity_slowdown"

                blend_rows.append({
                    "item_no": item, "loc": loc_val,
                    "week_start": week_start,
                    "plan_version": plan_version,
                    "sensing_horizon_weeks": sensing_horizon_weeks,
                    "week_offset": week_offset,
                    "alpha_weight": alpha,
                    "sensing_signal_qty": sensing_qty,
                    "statistical_forecast_qty": stat_qty,
                    "blended_qty": blended_qty,
                    "sensing_active": sensing_active,
                    "sensing_override_reason": override_reason,
                    "velocity_run_rate": round(daily_rate, 4),
                    "velocity_spike_ratio": round(spike_ratio, 3),
                    "days_elapsed_in_month": days_elapsed,
                    "mtd_sales_actual": round(mtd_sales, 2),
                    "is_outlier_capped": is_capped,
                })

            if dry_run:
                log.info("[DRY-RUN] %s/%s spike_ratio=%.2f proj_monthly=%.0f stat=%.0f w1_blended=%.0f",
                         item, loc_val, spike_ratio, projected_monthly, current_month_stat,
                         blend_rows[0]["blended_qty"] if blend_rows else 0)
                # Alert check
                deviation_pct = abs(projected_monthly - current_month_stat) / max(current_month_stat, 1) * 100
                if deviation_pct > alert_deviation_pct:
                    log.warning("ALERT: %s/%s sensing deviation %.1f%% vs statistical",
                                item, loc_val, deviation_pct)
            else:
                total_written += upsert_blend_rows(conn, blend_rows)

        if not dry_run:
            conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sensing_overrides_active")
            conn.commit()
        log.info("Done. Rows written: %d", total_written)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-no")
    parser.add_argument("--loc")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(item_no=args.item_no, loc=args.loc, dry_run=args.dry_run)
```

---

## 6. Config File

### `config/demand_sensing_config.yaml`

```yaml
# Demand Sensing Integration Configuration — Feature F3.4

# How many weeks from today that sensing overrides statistical
sensing_horizon_weeks: 4

# Alpha decay function: "linear" | "exponential" | "step"
alpha_decay: linear

# Minimum days of actual sales data required before sensing overrides statistical
min_velocity_days: 5

# If sensing/historical ratio exceeds this, cap the signal (noise guard)
# 3.0 = sensing rate must be < 3× historical average to be applied uncapped
velocity_outlier_threshold: 3.0

# Alert threshold: deviation between sensing and statistical (percentage)
# Generates exception alert in Control Tower when exceeded
alert_spike_threshold_pct: 25.0

# Minimum ABC class for sensing to be active
# "C" = all classes; "B" = A and B only; "A" = A-class only
min_abc_class_for_sensing: C

# Weeks ahead to compute blended plan
plan_horizon_weeks: 8

# How many weeks of blended plan to store per version
# Older versions are archived (not deleted) for accuracy backtesting
max_plan_versions_per_dfu: 4
```

---

## 7. API Endpoints

### `GET /forecast/blended?item_no=100320&loc=1401-BULK&weeks=8`

Returns blended weekly forecast for a specific DFU.

**Response:**
```json
{
  "item_no": "100320",
  "loc": "1401-BULK",
  "plan_version": "2026-04-08_v1",
  "sensing_horizon_weeks": 4,
  "velocity_spike_ratio": 1.208,
  "sensing_override_reason": "velocity_spike",
  "alert": "Sensing signal +20.8% above statistical forecast",
  "weeks": [
    {
      "week_start": "2026-04-07",
      "week_offset": 1,
      "alpha_weight": 1.0,
      "sensing_signal_qty": 125.6,
      "statistical_forecast_qty": 103.7,
      "blended_qty": 125.6,
      "sensing_active": true
    },
    {
      "week_start": "2026-04-14",
      "week_offset": 2,
      "alpha_weight": 0.667,
      "sensing_signal_qty": 125.6,
      "statistical_forecast_qty": 103.7,
      "blended_qty": 118.3,
      "sensing_active": true
    },
    {
      "week_start": "2026-04-21",
      "week_offset": 3,
      "alpha_weight": 0.333,
      "sensing_signal_qty": 125.6,
      "statistical_forecast_qty": 103.7,
      "blended_qty": 111.1,
      "sensing_active": true
    },
    {
      "week_start": "2026-04-28",
      "week_offset": 4,
      "alpha_weight": 0.0,
      "sensing_signal_qty": null,
      "statistical_forecast_qty": 103.7,
      "blended_qty": 103.7,
      "sensing_active": false
    }
  ],
  "monthly_summary": {
    "sensing_projected_monthly": 543.0,
    "statistical_forecast_monthly": 450.0,
    "blended_april_total": 503.7,
    "deviation_pct": 11.9
  }
}
```

### `GET /forecast/blended/summary`

Aggregated view of sensing status across portfolio.

**Response:**
```json
{
  "computed_at": "2026-04-08T09:00:00Z",
  "total_dfus_with_sensing": 892,
  "dfus_with_active_override": 143,
  "dfus_with_spike_alert": 28,
  "dfus_with_slowdown_alert": 14,
  "avg_spike_ratio_where_active": 1.31,
  "top_spikes": [
    {
      "item_no": "100320",
      "loc": "1401-BULK",
      "velocity_spike_ratio": 1.208,
      "sensing_qty_w1": 125.6,
      "statistical_qty_w1": 103.7,
      "abc_class": "A"
    }
  ]
}
```

### `GET /forecast/sensing-active`

Lists DFUs where sensing is currently overriding the statistical forecast.

**Response:**
```json
{
  "total": 143,
  "items": [
    {
      "item_no": "100320",
      "loc": "1401-BULK",
      "abc_class": "A",
      "velocity_spike_ratio": 1.208,
      "pct_deviation_from_statistical": 20.8,
      "sensing_override_reason": "velocity_spike"
    }
  ]
}
```

---

## 8. Frontend Components

### Location: New "Blended Demand Plan" panel in InvPlanningTab

```
┌──────────────────────────────────────────────────────────────────────────┐
│  DEMAND SENSING                              [Sensing Active: 143 DFUs] │
│                                                                           │
│  Item: [100320 ▼]  Location: [1401-BULK ▼]   Weeks ahead: [8 ▼]       │
│                                                                           │
│  SENSING OVERRIDE ACTIVE — Velocity +20.8% vs statistical forecast      │
│  Run rate: 18.1 units/day (historical: 15.0 u/day)                      │
│                                                                           │
│  Weekly Blended Forecast:                                                 │
│                                                                           │
│   Units/week                                                              │
│    130 ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← Sensing (Week 1)                       │
│    120 ┤░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓  ← Blended (Week 2)                       │
│    110 ┤░░░░░░░░░░░░▓▓▓▓▓▓▓  ← Blended (Week 3)                       │
│    103 ┤░░░░░░░░░░░░░░░░░░░  ← Statistical only (Week 4+)              │
│        └──────────────────────────────────                               │
│          W1    W2    W3    W4    W5    W6    W7    W8                   │
│                                                                           │
│  ▓ Sensing contribution  ░ Statistical contribution                      │
│                                                                           │
│  Sensing Horizon: [══════════════] 4 weeks   [Adjust ▲▼]               │
│                                                                           │
│  Active Spike Alerts (28 DFUs with >25% sensing deviation):             │
│  Item     Loc          Spike  W1 Sensing  W1 Statistical  ABC           │
│  ───────  ──────────   ─────  ──────────  ─────────────   ──            │
│  100320   1401-BULK    +21%   125.6       103.7           A             │
│  200145   DC-EAST      +34%   88.3        65.9            A             │
└──────────────────────────────────────────────────────────────────────────┘
```

**"Sensing Active" badge** on the Inv. Planning sidebar nav item shows count of DFUs where sensing is actively overriding statistical (from `mv_sensing_overrides_active`).

**Manual blend slider:** Planner can drag to adjust `sensing_horizon_weeks` from 1-8 and see the blend recalculate in real time (client-side calculation using current week's signal + stat forecast already loaded).

---

## 9. Worked Example — Full End-to-End Numbers

**Item 100320, April 2026, Day 8**

### Step 1: Inputs
```
mtd_sales              = 145 units  (from fact_demand_signals / fact_inventory_snapshot)
days_elapsed           = 8
days_in_month          = 30
historical_daily_avg   = 15.0 units/day  (6-month trailing average from agg_inventory_monthly)
```

### Step 2: compute_velocity_signal()
```
daily_run_rate  = 145 / 8           = 18.125 units/day
spike_ratio     = 18.125 / 15.0     = 1.208  (21% above historical)
outlier check   = 1.208 < 3.0 ✓ (no cap applied)
days_remaining  = 30 - 8            = 22 days
remaining       = 18.125 × 22       = 398.75 units
projected_total = 145 + 398.75      = 543.75 units
```

### Step 3: Weekly conversion
```
sensing_weekly_avg     = 543.75 / 4.33   = 125.6 units/week
statistical_weekly_avg = 450.0   / 4.33  = 103.9 units/week  (champion model: 450/month)
```

### Step 4: Alpha-weighted blending (4-week horizon)
```
Week 1 (Apr 7-13):   α=1.000, blended = 1.000×125.6 + 0.000×103.9 = 125.6 units
Week 2 (Apr 14-20):  α=0.667, blended = 0.667×125.6 + 0.333×103.9 = 118.3 units
Week 3 (Apr 21-27):  α=0.333, blended = 0.333×125.6 + 0.667×103.9 = 111.1 units
Week 4 (Apr 28-30):  α=0.000, blended = 0.000×125.6 + 1.000×103.9 = 103.9 units
Week 5+:             α=0.000, blended = 103.9 units (pure statistical)
```

### Step 5: April blended total
```
April blended (4 weeks × weekly + partial):
  ≈ (125.6 + 118.3 + 111.1 + 103.9) / 4 × 30/7
  Simplified: 4-week blended monthly total ≈ 503 units

  vs statistical = 450 units
  vs sensing-only = 543 units
  Blended = 503 units (+11.8% vs statistical, -7.4% vs pure sensing)
```

### Step 6: Earlier reorder trigger comparison
```
Current ROP (using static 450/month):
  Daily demand rate = 450/30 = 15.0 units/day
  ROP = 15.0 × 14 days LT + 60 SS = 270 units
  → Reorder triggered when on-hand drops to 270 units

Updated ROP (using blended 503/month, week 1 sensing active):
  Daily demand rate (week 1) = 125.6/7 = 17.9 units/day
  ROP = 17.9 × 14 + 60 SS = 311 units
  → Reorder triggered when on-hand drops to 311 units (41 units earlier)

  Days of earlier warning: 41 units / 15 units/day ≈ 2.7 days earlier signal
  With 14-day LT: this 2.7-day signal translates to a meaningful stockout avoidance
```

### Step 7: Alert generated
```
deviation_pct = |543 - 450| / 450 × 100 = 20.7% > threshold (25% not exceeded → no hard alert)
spike_ratio   = 1.208 > 1.10 → sensing_override_reason = "velocity_spike"
→ "Sensing Override Active" badge shows in UI
→ No hard exception alert (below 25% threshold)

If April Day 10 shows 200 units in 10 days (rate=20/day, spike_ratio=1.333):
  projected = 200 + 20×20 = 600 units
  deviation = |600-450|/450 = 33.3% > 25% threshold
  → ALERT: "Velocity spike on Item 100320 — sensing signal +33% vs statistical forecast"
  → Exception generated in fact_replenishment_exceptions (IPfeature7)
```

---

## 10. Dependencies

| Dependency | Required For | Status |
|---|---|---|
| IPfeature9 — Demand Signals | `fact_demand_signals` with `mtd_sales` and velocity | Implemented |
| IPfeature3 — Safety Stock | ROP recalculation using blended daily rate | Implemented |
| IPfeature7 — Exception Queue | Sensing spike alert → exception record | Implemented |
| IPfeature8 — Fill Rate | Backtest: did blended plan improve fill rate vs static? | Implemented |
| Champion model (feature15) | Statistical forecast source via `model_id='champion'` | Implemented |

---

## 11. Out of Scope

- Machine learning-based alpha decay (ML-optimized sensing/statistical mix per DFU)
- Real-time (sub-daily) sensing — current design is daily batch computation
- POS (point-of-sale) data integration — current sensing uses inventory velocity as proxy
- Multi-echelon sensing propagation (store velocity → DC sensing) — covered in feature_06_12
- External signals (weather, social media, Google Trends) as sensing inputs

---

## 12. Test Requirements

### Backend Unit Tests — `tests/unit/test_demand_sensing.py`

```
test_compute_alpha_week1()                      — α=1.0 for week 1
test_compute_alpha_week4_four_horizon()         — α=0.0 at sensing boundary
test_compute_alpha_mid_horizon()                — α=0.667 week 2 of 4-week horizon
test_compute_alpha_beyond_horizon()             — α=0.0 for week 5+
test_compute_alpha_single_week_horizon()        — degenerate case
test_compute_velocity_signal_normal()           — 145 sales / 8 days → 543 projected
test_compute_velocity_signal_outlier_capped()   — spike_ratio > 3.0 → capped
test_compute_velocity_signal_min_days_guard()   — raises ValueError when days_elapsed < 1
test_monthly_to_weekly_conversion()             — 450 / 4.33 ≈ 103.9
test_blended_qty_pure_sensing_week1()           — α=1.0, blended = sensing only
test_blended_qty_pure_stat_week4()             — α=0.0, blended = statistical only
test_blended_qty_mixed_week2()                  — α=0.667 gives correct weighted average
test_alert_threshold_triggered()               — deviation > 25% → alert flag
test_alert_threshold_not_triggered()           — deviation = 20% → no alert
```

### Backend API Tests — `tests/api/test_blended_forecast.py`

```
test_blended_endpoint_returns_8_weeks()
test_blended_endpoint_unknown_item()
test_blended_week1_is_pure_sensing()
test_blended_week4_is_pure_statistical()
test_summary_counts_active_overrides()
test_sensing_active_endpoint_filters_correctly()
test_blended_monthly_summary_totals()
```

### Make Targets to Add

```makefile
sensing-schema:     # Apply DDL (fact_blended_demand_plan, mv_sensing_overrides_active)
sensing-compute:    # Run compute_blended_forecast.py for all DFUs
sensing-compute-dfu: ITEM=100320 LOC=1401-BULK  # Single DFU
sensing-compute-dry: # --dry-run preview
sensing-all:        # sensing-schema + sensing-compute
```
