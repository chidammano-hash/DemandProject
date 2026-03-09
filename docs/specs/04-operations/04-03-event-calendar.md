# Feature F4.3 — Promotion & Event Planning

**Phase:** 4 — Evolution to Operations
**Feature Number:** F4.3 (Internal: feature_06_15)
**Status:** Specification
**Author:** Supply Chain Systems Architecture
**Date:** 2026-03-06

---

## 1. Problem Statement

Statistical forecasting models excel at learning repeating demand patterns from history — seasonality, trends, and item velocity. They cannot, by definition, anticipate events that have not yet happened: a 3-for-2 promotional bundle, a new product launch, a competitor going out of stock, or a Category 3 hurricane shutting down Southeast distribution.

Every major commercial event creates one of two failure modes:

**Stockout (under-stocked):** The demand surge arrives and inventory is exhausted. Service level drops, customers switch to competitors, and revenue is permanently lost (not just deferred). The promotion that was supposed to drive growth instead drives customer attrition.

**Excess (over-ordered):** The event was smaller than expected, or the pantry-loading dip after a promotion was deeper than anticipated. Unsold inventory sits in the warehouse for 60–90 days, consuming carrying cost and warehouse space, and often must be marked down or written off.

### What Fails Today

**Scenario A — Easter promotion blindspot:**
Marketing plans a 30% price reduction on "Energy Drink 500ml" for 7 days at all locations starting April 15. The statistical model sees no signal — it forecasts the same 450 units for April as March. The warehouse ships 450 units to the DC. Actual demand in the promo week: 720 units. Stockout occurs on day 4. Lost sales: ~270 units × $36 shelf price = $9,720. Recovery impossible: the promotional window has passed.

**Scenario B — New product launch failure:**
A new SKU "Protein Bar Chocolate 80g" launches on June 1. The statistical model has zero history for this item. It either forecasts zero (no data) or inherits the cluster average. The launch ramp (slow start → 8-week acceleration → plateau) is not modeled. Week 1: ordered 500 units, sold 80. Week 8: reordered 500 units (same as week 1), sold 1,200 — stockout.

**Scenario C — Phase-out inventory bloat:**
SKU "Lemon Iced Tea 500ml" is being discontinued. Last production run is March 31. Without explicit phase-out planning, the system continues to recommend replenishment orders based on historical demand. Final orders arrive after the last production run, leaving 2,800 units stranded at the DC with zero future demand.

**Scenario D — No post-event learning:**
After each promotional event, no structured comparison between forecasted lift and actual lift is captured. The same promotional uplift estimate (35%) is reused for every event regardless of actual performance, preventing calibration over time.

---

## 2. Objectives

1. Maintain a structured **event calendar** covering promotions, launches, phase-outs, and market events.
2. Apply **mathematical uplift adjustments** to the statistical base forecast for each event type.
3. Generate **adjusted demand signals** that flow into the inventory planning and ordering pipeline.
4. Detect **event conflicts** when overlapping events affect the same item-location.
5. Track **post-event performance** by comparing forecasted vs. actual lift, enabling calibration.
6. Support **order deadline calculation** — given an event start date and lead time, when must the order be placed?

---

## 3. Event Types and Uplift Models

### 3.1 Event Types

| Event Type | Description | Key Parameters | Typical Duration |
|---|---|---|---|
| `PROMOTION` | Price reduction or buy-more-get-more offer | uplift_pct, pantry_loading_pct | 1 day–4 weeks |
| `NEW_LAUNCH` | New product introduction | ramp_profile, peak_qty, ramp_weeks | 8–26 weeks |
| `PHASE_OUT` | Product discontinuation | last_order_date, target_ending_stock | 4–12 weeks |
| `MARKET_EVENT` | External uplift or suppression (sport season, weather, competitor) | uplift_pct, uplift_type [additive/multiplicative] | 1 day–3 months |
| `CANNIBALIZATION` | New product reduces demand for existing product | new_item_no, cannibalized_item_no, cannibalization_pct | Permanent |
| `SEASONALITY_OVERRIDE` | Manual correction to model seasonality for a specific period | override_multiplier | 1–4 weeks |

### 3.2 Uplift Formulas

**PROMOTION — Weekly Adjustment:**

```
promo_week_qty[t] = base_forecast_weekly[t] * (1 + uplift_pct / 100)

post_promo_qty[t] = base_forecast_weekly[t] * (1 - pantry_loading_pct / 100)
  for each week in [event_end, event_end + pantry_loading_weeks]

monthly_adjusted = sum(week_qty for all weeks in month)
                   where week_qty = promo_week_qty (if promo week) else post_promo_qty (if dip week) else base_week_qty
```

**NEW_LAUNCH — S-Curve Ramp:**

```python
def s_curve_qty(t: int, peak_qty: float, ramp_weeks: int) -> float:
    """
    t          = weeks since launch (0-indexed)
    peak_qty   = weekly quantity at plateau
    ramp_weeks = number of weeks to reach 80% of peak
    """
    k = 8.0 / ramp_weeks      # Steepness coefficient (calibrated so t_mid = ramp_weeks/2 → 50% of peak)
    t_mid = ramp_weeks / 2.0
    return peak_qty / (1 + math.exp(-k * (t - t_mid)))
```

**PHASE_OUT — Exponential Decay:**

```python
def phase_out_qty(t: int, initial_qty: float, decay_rate: float, target_ending_stock: float) -> float:
    """
    t                  = weeks since phase-out announcement
    initial_qty        = current weekly demand rate
    decay_rate         = weekly decay factor (e.g., 0.15 = 15% demand drops per week)
    target_ending_stock = stop ordering when projected stock ≥ this value
    """
    return max(0.0, initial_qty * (1 - decay_rate) ** t)
```

**CANNIBALIZATION:**

```
cannibalized_item_adjusted[t] = base_forecast[t] * (1 - cannibalization_pct / 100)
  for t >= cannibalization_start_date
```

---

## 4. Data Model

### 4.1 New Table: `fact_event_calendar`

**Grain:** event_id (one row per event definition)
**Purpose:** Master event registry — the source of truth for all planned events.

```sql
CREATE TABLE fact_event_calendar (
    event_id              SERIAL PRIMARY KEY,
    event_type            VARCHAR(30)   NOT NULL,
    -- Allowed: PROMOTION / NEW_LAUNCH / PHASE_OUT / MARKET_EVENT / CANNIBALIZATION / SEASONALITY_OVERRIDE
    event_name            VARCHAR(200)  NOT NULL,
    event_description     TEXT,
    event_start           DATE          NOT NULL,
    event_end             DATE          NOT NULL,
    -- Uplift parameters
    uplift_pct            NUMERIC(6,2),   -- % uplift (positive = increase, negative = suppression)
    uplift_type           VARCHAR(20),    -- additive / multiplicative
    ramp_profile          VARCHAR(20),    -- linear / s_curve / step / custom (for NEW_LAUNCH)
    ramp_weeks            INTEGER,        -- Weeks to 80% of peak (for s_curve)
    peak_qty_weekly       NUMERIC(12,2),  -- Target peak weekly quantity (NEW_LAUNCH)
    decay_rate            NUMERIC(6,4),   -- Weekly decay fraction (PHASE_OUT)
    pantry_loading_pct    NUMERIC(5,2),   -- Post-promo dip % (PROMOTION)
    pantry_loading_weeks  INTEGER DEFAULT 2, -- Weeks of post-promo dip
    last_order_date       DATE,           -- PHASE_OUT: last replenishment order allowed
    target_ending_stock   NUMERIC(12,2),  -- PHASE_OUT: target residual stock at EOL
    cannibalized_item_no  VARCHAR(50),    -- CANNIBALIZATION: item being cannibalized
    cannibalization_pct   NUMERIC(5,2),   -- CANNIBALIZATION: % demand transfer
    override_multiplier   NUMERIC(6,4),   -- SEASONALITY_OVERRIDE: multiply base forecast
    -- Scope parameters
    target_items          JSONB,          -- ["100320", "100321"] or null for ALL
    target_locations      JSONB,          -- ["1401-BULK", "2201-DC"] or null for ALL
    target_categories     JSONB,          -- ["Beverages"] or null for ALL
    -- Workflow
    status                VARCHAR(20)     NOT NULL DEFAULT 'draft',
    -- Allowed: draft / approved / active / completed / cancelled
    conflict_resolution   VARCHAR(30),    -- none / override_first / combine_multiplicative / manual
    priority              INTEGER         DEFAULT 50,  -- Higher number = higher priority in conflicts
    created_by            VARCHAR(100),
    approved_by           VARCHAR(100),
    approved_at           TIMESTAMPTZ,
    created_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_event_cal_type    ON fact_event_calendar (event_type);
CREATE INDEX idx_event_cal_dates   ON fact_event_calendar (event_start, event_end);
CREATE INDEX idx_event_cal_status  ON fact_event_calendar (status);
CREATE INDEX idx_event_cal_active  ON fact_event_calendar (event_start, event_end)
    WHERE status IN ('approved', 'active');
```

**Example rows:**

| event_id | event_type | event_name | event_start | event_end | uplift_pct | pantry_loading_pct | status |
|---|---|---|---|---|---|---|---|
| 1 | PROMOTION | Easter Sale 2026 | 2026-04-15 | 2026-04-21 | 35.0 | 10.0 | approved |
| 2 | NEW_LAUNCH | Protein Bar Chocolate Launch | 2026-06-01 | 2026-09-30 | NULL | NULL | approved |
| 3 | PHASE_OUT | Lemon Iced Tea 500ml EOL | 2026-03-01 | 2026-05-31 | NULL | NULL | approved |
| 4 | CANNIBALIZATION | Protein Bar → replaces Cereal Bar | 2026-06-01 | 2026-12-31 | NULL | NULL | approved |

---

### 4.2 New Table: `fact_event_adjusted_forecast`

**Grain:** item_no + loc + plan_month + event_id
**Purpose:** Per-item-location adjusted forecast output after applying all event adjustments.

```sql
CREATE TABLE fact_event_adjusted_forecast (
    id                       BIGSERIAL PRIMARY KEY,
    item_no                  VARCHAR(50)   NOT NULL,
    loc                      VARCHAR(50)   NOT NULL,
    plan_month               DATE          NOT NULL,
    event_id                 INTEGER       NOT NULL REFERENCES fact_event_calendar(event_id),
    base_forecast_qty        NUMERIC(12,2) NOT NULL,
    event_adjustment_qty     NUMERIC(12,2) NOT NULL,
    post_promo_dip_qty       NUMERIC(12,2) DEFAULT 0,
    adjusted_forecast_qty    NUMERIC(12,2) NOT NULL,
    -- adjusted = base + event_adjustment + post_promo_dip (dip is typically negative)
    adjustment_type          VARCHAR(30)   NOT NULL,
    -- Allowed: promotional_uplift / launch_ramp / phase_out_decay / market_uplift
    --          cannibalization_reduction / seasonality_override
    order_deadline           DATE,         -- Latest order date to have stock for event_start
    plan_version             VARCHAR(50)   NOT NULL DEFAULT 'latest',
    computed_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT event_adj_fcst_unique UNIQUE (item_no, loc, plan_month, event_id, plan_version)
);

CREATE INDEX idx_event_adj_item  ON fact_event_adjusted_forecast (item_no, loc);
CREATE INDEX idx_event_adj_month ON fact_event_adjusted_forecast (plan_month);
CREATE INDEX idx_event_adj_event ON fact_event_adjusted_forecast (event_id);
```

---

### 4.3 New Table: `fact_event_performance`

**Grain:** event_id + item_no + loc + plan_month
**Purpose:** Post-event tracking — compare forecasted vs. actual lift for calibration.

```sql
CREATE TABLE fact_event_performance (
    id                     BIGSERIAL PRIMARY KEY,
    event_id               INTEGER       NOT NULL REFERENCES fact_event_calendar(event_id),
    item_no                VARCHAR(50)   NOT NULL,
    loc                    VARCHAR(50)   NOT NULL,
    plan_month             DATE          NOT NULL,
    base_forecast_qty      NUMERIC(12,2) NOT NULL,
    forecasted_lift_qty    NUMERIC(12,2) NOT NULL,  -- event_adjustment_qty from plan
    actual_sales_qty       NUMERIC(12,2),            -- From fact_sales_monthly after close
    actual_lift_qty        NUMERIC(12,2),            -- actual_sales_qty - base_forecast_qty
    lift_accuracy_pct      NUMERIC(6,2),             -- (1 - |forecasted_lift - actual_lift| / actual_lift) * 100
    forecasted_sell_through_pct NUMERIC(5,2),
    actual_sell_through_pct     NUMERIC(5,2),
    ending_stock_qty       NUMERIC(12,2),
    uplift_calibration_factor NUMERIC(6,4),          -- actual_lift / forecasted_lift (for future calibration)
    computed_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT event_perf_unique UNIQUE (event_id, item_no, loc, plan_month)
);

CREATE INDEX idx_event_perf_event ON fact_event_performance (event_id);
CREATE INDEX idx_event_perf_item  ON fact_event_performance (item_no, loc);
```

---

### 4.4 New Table: `fact_event_conflicts`

**Grain:** conflict_id (one row per detected overlap)
**Purpose:** Track overlapping events that affect the same item-location and require resolution.

```sql
CREATE TABLE fact_event_conflicts (
    conflict_id       SERIAL PRIMARY KEY,
    event_id_a        INTEGER       NOT NULL REFERENCES fact_event_calendar(event_id),
    event_id_b        INTEGER       NOT NULL REFERENCES fact_event_calendar(event_id),
    item_no           VARCHAR(50),   -- NULL means category-wide conflict
    loc               VARCHAR(50),
    overlap_start     DATE          NOT NULL,
    overlap_end       DATE          NOT NULL,
    resolution        VARCHAR(30)   NOT NULL DEFAULT 'unresolved',
    -- Allowed: unresolved / event_a_priority / event_b_priority / combine_multiplicative / manual
    resolved_by       VARCHAR(100),
    resolved_at       TIMESTAMPTZ,
    detected_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_event_conflict_events ON fact_event_conflicts (event_id_a, event_id_b);
```

---

## 5. Python Scripts

### 5.1 `scripts/apply_event_adjustments.py`

**Purpose:** Read event calendar → apply adjustments to base statistical forecast → write to `fact_event_adjusted_forecast`.

```python
# scripts/apply_event_adjustments.py

import math
import yaml
import psycopg
import pandas as pd
from datetime import date, timedelta
from typing import Literal
from common.db import get_db_params

CONFIG_PATH = "config/event_planning_config.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_active_events(conn, as_of: date) -> pd.DataFrame:
    """Load all approved/active events that overlap with the planning horizon."""
    sql = """
        SELECT *
        FROM fact_event_calendar
        WHERE status IN ('approved', 'active')
          AND event_end >= %s
        ORDER BY priority DESC, event_start
    """
    return pd.read_sql(sql, conn, params=(as_of,))

def load_base_forecasts(conn, plan_months: list[date], model_id: str = "external") -> pd.DataFrame:
    """Load statistical base forecasts for the planning horizon."""
    sql = """
        SELECT f.dmdunit AS item_no, f.loc, f.startdate AS plan_month,
               f.qty AS base_forecast_qty,
               i.category, d.abc_class
        FROM fact_external_forecast_monthly f
        JOIN dim_item i ON i.item_no = f.dmdunit
        JOIN dim_dfu d ON d.item_no = f.dmdunit AND d.loc = f.loc
        WHERE f.startdate = ANY(%s)
          AND f.model_id = %s
          AND f.lag = 0
    """
    return pd.read_sql(sql, conn, params=(plan_months, model_id))

def weeks_in_month_overlap(event_start: date, event_end: date, plan_month: date) -> float:
    """
    Compute fractional weeks of event overlap within a given plan month.
    Returns a float representing weeks (e.g., 0.5 = half a week).
    """
    month_start = plan_month
    month_end = (plan_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    overlap_start = max(event_start, month_start)
    overlap_end   = min(event_end, month_end)
    overlap_days  = max(0, (overlap_end - overlap_start).days + 1)
    return overlap_days / 7.0

def apply_promotion_adjustment(
    base_qty: float,
    uplift_pct: float,
    pantry_loading_pct: float,
    pantry_loading_weeks: int,
    event_weeks_in_month: float,
    post_promo_weeks_in_month: float,
    total_weeks_in_month: float = 4.33,
) -> tuple[float, float]:
    """
    Compute promotional uplift and post-promo dip for one item-location-month.

    Returns: (event_adjustment_qty, post_promo_dip_qty)
    Both values are deltas relative to the base forecast.
    """
    base_weekly = base_qty / total_weeks_in_month

    promo_uplift_per_week = base_weekly * (uplift_pct / 100)
    event_adjustment_qty  = promo_uplift_per_week * event_weeks_in_month

    pantry_dip_per_week   = base_weekly * (pantry_loading_pct / 100)
    post_promo_dip_qty    = -pantry_dip_per_week * post_promo_weeks_in_month

    return round(event_adjustment_qty, 2), round(post_promo_dip_qty, 2)

def apply_new_launch_adjustment(
    event_start: date,
    plan_month: date,
    peak_qty_weekly: float,
    ramp_weeks: int,
    ramp_profile: str,
) -> float:
    """
    Compute total new launch quantity for a plan_month.
    Returns the total launch-attributed quantity (this is the full forecast, not just the delta,
    since there is no base forecast for a new item).
    """
    month_start = plan_month
    monthly_total = 0.0
    for day_offset in range(0, 31):
        day = month_start + timedelta(days=day_offset)
        if day.month != plan_month.month:
            break
        if day < event_start:
            continue
        t = (day - event_start).days / 7.0  # Weeks since launch

        if ramp_profile == "s_curve":
            k = 8.0 / max(ramp_weeks, 1)
            t_mid = ramp_weeks / 2.0
            daily_qty = (peak_qty_weekly / 7.0) / (1 + math.exp(-k * (t - t_mid)))
        elif ramp_profile == "linear":
            daily_qty = (peak_qty_weekly / 7.0) * min(1.0, t / max(ramp_weeks, 1))
        else:  # step
            daily_qty = peak_qty_weekly / 7.0

        monthly_total += daily_qty

    return round(monthly_total, 2)

def apply_phase_out_adjustment(
    base_qty: float,
    event_start: date,
    plan_month: date,
    decay_rate: float,
) -> float:
    """
    Compute phase-out demand reduction for a plan_month.
    Returns the REDUCTION amount (negative delta from base).
    """
    weeks_since_start = max(0, (plan_month - event_start).days / 7.0)
    remaining_fraction = (1 - decay_rate) ** weeks_since_start
    adjusted_qty = base_qty * remaining_fraction
    return round(adjusted_qty - base_qty, 2)  # Negative value

def compute_order_deadline(event_start: date, lead_time_days: int) -> date:
    """Latest date by which an order must be placed to receive stock before event_start."""
    return event_start - timedelta(days=lead_time_days)

def detect_conflicts(events: pd.DataFrame) -> list[dict]:
    """
    Detect overlapping events that affect overlapping item/location scopes.
    Returns a list of conflict dicts.
    """
    conflicts = []
    event_list = events.to_dict("records")
    for i, ea in enumerate(event_list):
        for eb in event_list[i + 1:]:
            # Check date overlap
            if ea["event_end"] < eb["event_start"] or eb["event_end"] < ea["event_start"]:
                continue
            # Check scope overlap (simplified: flag all date overlaps — fine-grained scope TBD)
            overlap_start = max(ea["event_start"], eb["event_start"])
            overlap_end   = min(ea["event_end"],   eb["event_end"])
            conflicts.append({
                "event_id_a":    ea["event_id"],
                "event_id_b":    eb["event_id"],
                "overlap_start": overlap_start,
                "overlap_end":   overlap_end,
                "resolution":    "unresolved",
            })
    return conflicts

def write_adjusted_forecasts(conn, rows: list[dict], plan_version: str) -> int:
    """Bulk upsert adjusted forecast rows."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM fact_event_adjusted_forecast WHERE plan_version = %s",
            (plan_version,)
        )
        cur.executemany("""
            INSERT INTO fact_event_adjusted_forecast (
                item_no, loc, plan_month, event_id,
                base_forecast_qty, event_adjustment_qty, post_promo_dip_qty,
                adjusted_forecast_qty, adjustment_type, order_deadline, plan_version
            ) VALUES (
                %(item_no)s, %(loc)s, %(plan_month)s, %(event_id)s,
                %(base_forecast_qty)s, %(event_adjustment_qty)s, %(post_promo_dip_qty)s,
                %(adjusted_forecast_qty)s, %(adjustment_type)s, %(order_deadline)s, %(plan_version)s
            )
        """, rows)
        conn.commit()
    return len(rows)

def run(horizon_months: int = 3, plan_version: str = "latest") -> None:
    cfg = load_config()
    default_lead_time = cfg.get("default_lead_time_days", 14)

    today = date.today().replace(day=1)
    plan_months = [
        (today + timedelta(days=32 * i)).replace(day=1)
        for i in range(horizon_months)
    ]

    with psycopg.connect(**get_db_params()) as conn:
        events      = load_active_events(conn, today)
        forecasts   = load_base_forecasts(conn, plan_months)
        conflicts   = detect_conflicts(events)

        if conflicts:
            print(f"[events] WARNING: {len(conflicts)} event conflict(s) detected. Check fact_event_conflicts.")
            # Write conflicts to DB for UI display (omitted for brevity)

        rows = []
        for _, evt in events.iterrows():
            for plan_month in plan_months:
                event_weeks = weeks_in_month_overlap(evt["event_start"], evt["event_end"], plan_month)
                if event_weeks == 0:
                    continue

                scope_mask = pd.Series([True] * len(forecasts))
                if evt["target_items"] is not None:
                    scope_mask &= forecasts["item_no"].isin(evt["target_items"])
                if evt["target_locations"] is not None:
                    scope_mask &= forecasts["loc"].isin(evt["target_locations"])
                if evt["target_categories"] is not None:
                    scope_mask &= forecasts["category"].isin(evt["target_categories"])

                scoped = forecasts[scope_mask & (forecasts["plan_month"] == plan_month)]

                for _, frow in scoped.iterrows():
                    base = float(frow["base_forecast_qty"])
                    adj  = 0.0
                    dip  = 0.0
                    adj_type = "promotional_uplift"

                    if evt["event_type"] == "PROMOTION":
                        post_weeks = weeks_in_month_overlap(
                            evt["event_end"] + timedelta(days=1),
                            evt["event_end"] + timedelta(days=7 * int(evt.get("pantry_loading_weeks", 2))),
                            plan_month,
                        )
                        adj, dip = apply_promotion_adjustment(
                            base,
                            float(evt["uplift_pct"]),
                            float(evt.get("pantry_loading_pct", 0)),
                            int(evt.get("pantry_loading_weeks", 2)),
                            event_weeks,
                            post_weeks,
                        )
                        adj_type = "promotional_uplift"

                    elif evt["event_type"] == "NEW_LAUNCH":
                        adj = apply_new_launch_adjustment(
                            evt["event_start"], plan_month,
                            float(evt.get("peak_qty_weekly", 0)),
                            int(evt.get("ramp_weeks", 8)),
                            str(evt.get("ramp_profile", "s_curve")),
                        )
                        base = 0.0  # New item — no base forecast
                        adj_type = "launch_ramp"

                    elif evt["event_type"] == "PHASE_OUT":
                        adj = apply_phase_out_adjustment(
                            base, evt["event_start"], plan_month,
                            float(evt.get("decay_rate", 0.15)),
                        )
                        adj_type = "phase_out_decay"

                    elif evt["event_type"] == "CANNIBALIZATION":
                        adj = -base * float(evt.get("cannibalization_pct", 0)) / 100
                        adj_type = "cannibalization_reduction"

                    rows.append({
                        "item_no":               frow["item_no"],
                        "loc":                   frow["loc"],
                        "plan_month":            plan_month,
                        "event_id":              int(evt["event_id"]),
                        "base_forecast_qty":     round(base, 2),
                        "event_adjustment_qty":  adj,
                        "post_promo_dip_qty":    dip,
                        "adjusted_forecast_qty": round(base + adj + dip, 2),
                        "adjustment_type":       adj_type,
                        "order_deadline":        compute_order_deadline(
                            evt["event_start"], default_lead_time
                        ),
                        "plan_version":          plan_version,
                    })

        n = write_adjusted_forecasts(conn, rows, plan_version)
        print(f"[events] Wrote {n:,} adjusted forecast rows for version '{plan_version}'")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--version", default="latest")
    args = p.parse_args()
    run(args.horizon, args.version)
```

### 5.2 Config: `config/event_planning_config.yaml`

```yaml
# Event planning configuration
default_lead_time_days: 14         # Default LT for order deadline calculation
default_pantry_loading_weeks: 2    # Default weeks of post-promo dip
default_ramp_weeks: 8              # Default weeks to 80% of peak for s_curve launches
default_decay_rate: 0.15           # Default weekly demand decay for phase-outs
default_plan_version: "latest"

# Conflict detection
auto_resolve_same_type: false      # Combine same-type events multiplicatively
conflict_resolution_default: "override_first"  # Higher priority event wins

# Post-event performance tracking
performance_window_weeks: 4        # Weeks after event end to compute actual lift

# S-curve calibration
s_curve_calibration:
  k_factor_base: 8.0               # Larger k = steeper ramp (faster to plateau)
  calibrate_from_history: false    # True = infer k from past launch data per category
```

---

## 6. API Endpoints

### `GET /events/calendar`

Returns list of all events with optional filters.

**Query params:** `event_type=PROMOTION`, `status=approved`, `from_date=2026-04-01`, `to_date=2026-06-30`

**Response:**
```json
{
  "events": [
    {
      "event_id": 1,
      "event_type": "PROMOTION",
      "event_name": "Easter Sale 2026",
      "event_start": "2026-04-15",
      "event_end": "2026-04-21",
      "uplift_pct": 35.0,
      "pantry_loading_pct": 10.0,
      "status": "approved",
      "affected_skus": 847,
      "estimated_uplift_value": 184300.00
    }
  ]
}
```

---

### `POST /events/calendar`

Create a new event.

**Request body:**
```json
{
  "event_type": "PROMOTION",
  "event_name": "Easter Sale 2026",
  "event_start": "2026-04-15",
  "event_end": "2026-04-21",
  "uplift_pct": 35.0,
  "pantry_loading_pct": 10.0,
  "pantry_loading_weeks": 2,
  "target_items": ["100320", "100321", "100322"],
  "target_locations": null,
  "status": "draft",
  "created_by": "mktg_alice"
}
```

---

### `GET /events/impact-preview`

Preview the demand adjustment for a specific event and item-location before committing.

**Query params:** `event_id=1`, `item_no=100320`, `loc=1401-BULK`

**Response:**
```json
{
  "event_id": 1,
  "item_no": "100320",
  "loc": "1401-BULK",
  "lead_time_days": 14,
  "order_deadline": "2026-04-01",
  "months": [
    {
      "plan_month": "2026-04-01",
      "base_forecast_qty": 450,
      "event_adjustment_qty": 56.2,
      "post_promo_dip_qty": -20.9,
      "adjusted_forecast_qty": 485.3,
      "adjustment_type": "promotional_uplift"
    }
  ],
  "chart_data": {
    "weeks": ["Apr 1", "Apr 8", "Apr 15", "Apr 22", "Apr 29"],
    "base": [105, 105, 105, 105, 105],
    "adjusted": [105, 105, 141.75, 94.5, 105]
  }
}
```

---

### `GET /events/performance`

Returns post-event performance comparison.

**Query params:** `event_id=`, `min_lift_accuracy=`, `status=completed`

**Response:**
```json
{
  "events": [
    {
      "event_id": 1,
      "event_name": "Easter Sale 2026",
      "total_forecasted_lift": 12400,
      "total_actual_lift": 10850,
      "portfolio_lift_accuracy_pct": 87.5,
      "avg_uplift_calibration_factor": 0.875,
      "recommendation": "Reduce Easter promo uplift estimate from 35% to 30.6% for future events"
    }
  ]
}
```

---

## 7. Frontend Components

### 7.1 Event Calendar Panel

```
┌─────────────────────────────────────────────────────────────────────┐
│  EVENT CALENDAR                     [+ New Event]  [Import CSV]     │
├─────────────────────────────────────────────────────────────────────┤
│  APRIL 2026                                                         │
│  Mon  Tue  Wed  Thu  Fri  Sat  Sun                                  │
│   1    2    3    4    5    6    7                                    │
│   8    9   10   11   12   13   14                                   │
│  15   16   17   18   19   20   21  ████ EASTER SALE (35% uplift)   │
│  22   23   24   25   26   27   28  ░░░░ Post-promo dip (-10%)      │
│  29   30                                                            │
├─────────────────────────────────────────────────────────────────────┤
│  EVENT LIST                                                         │
│  ID  Type       Name                    Dates         Status  Scope │
│   1  PROMOTION  Easter Sale 2026        Apr 15-21     Approved 847  │
│   2  NEW_LAUNCH Protein Bar Choc Launch Jun 1 - Sep30 Approved NEW  │
│   3  PHASE_OUT  Lemon Iced Tea EOL      Mar 1 - May31 Active  34   │
│   4  CANNIB.    Protein Bar→Cereal Bar  Jun 1 - Dec31 Approved 89  │
├─────────────────────────────────────────────────────────────────────┤
│  IMPACT PREVIEW: Easter Sale 2026 on Item 100320 @ 1401-BULK        │
│                                                                     │
│  250 ┤                          ╭──╮                               │
│  200 ┤             Base ────────╯  ╰──────────────────             │
│  150 ┤             Adjusted ════╗  ╔══════════════════             │
│  100 ┤                          ╚╗╔╝                               │
│   50 ┤                           ╔╝                                │
│      └──────────────────────────────────────────────               │
│       Apr 1  Apr 8  Apr 15  Apr 22  Apr 29                         │
│                     ↑ Promo  ↑ Dip                                  │
│  Order deadline: April 1 (14-day lead time)                        │
│  ⚠ Order 56 extra units before April 1 to cover promo uplift       │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Event Creation Modal

Modal with:
- Event type selector (PROMOTION / NEW_LAUNCH / PHASE_OUT / MARKET_EVENT / CANNIBALIZATION)
- Date range picker
- Uplift % slider with live preview (adjusts the impact chart in real-time)
- Item/location multi-select with count display
- Conflict warning banner if overlapping events detected
- "Preview Impact" button shows chart before saving

---

## 8. Worked Example — Easter Sale 2026, Item 100320

**Setup:**
- Item: 100320 — Energy Drink 500ml
- Location: 1401-BULK
- Base April statistical forecast: 450 units
- Promo period: April 15–21 (1 week out of 4.33 weeks in April)
- Uplift: +35%
- Post-promo pantry loading: -10% for 2 weeks after promo

**Step 1: Weekly base rate**
```
base_weekly = 450 / 4.33 = 103.9 units/week ≈ 104 units/week
```

**Step 2: Promo week uplift**
```
promo_week_qty = 104 × (1 + 0.35) = 104 × 1.35 = 140.4 units
promo_uplift_delta = 140.4 - 104 = 36.4 units (for the 1 promo week)
```

**Step 3: Post-promo dip (2 weeks in April 22–30 = ~1.3 weeks in April)**
```
dip_per_week = 104 × 0.10 = 10.4 units/week
post_promo_dip = -10.4 × 1.3 = -13.5 units
```

**Step 4: Adjusted April total**
```
adjusted_april = base_april + promo_uplift + post_promo_dip
               = 450 + 36.4 - 13.5
               = 472.9 units ≈ 473 units
```

**Step 5: Order deadline**
```
Lead time: 14 days
Promo start: April 15
Order deadline: April 15 - 14 = April 1
Additional stock needed above current plan: 36 units × unit_cost $24.50 = $882
```

**Month-by-month adjusted forecast table:**

| Month | Base Qty | Event Adjustment | Post-Promo Dip | Adjusted Qty | Order Deadline |
|---|---|---|---|---|---|
| Apr 2026 | 450 | +36.4 | -13.5 | 473 | Apr 1 (extra 36 units) |
| May 2026 | 450 | -9.1 (residual dip) | 0 | 441 | — |
| Jun 2026 | 450 | 0 | 0 | 450 | — |

**Post-event performance (computed after April actuals loaded):**

```
Forecasted lift:   36 units
Actual lift:       28 units (actual April sales = 478, base = 450, actual lift = 28)
Lift accuracy:     (1 - |36-28|/28) × 100 = 71.4%
Calibration factor: 28/36 = 0.78
Recommendation: Reduce Easter promo uplift from 35% to 27% for future events
```

---

## 9. Dependencies

| Dependency | Feature | Status |
|---|---|---|
| Statistical base forecast | fact_external_forecast_monthly | Implemented |
| Item lead time data | dim_dfu / fact_lead_time_profile | Partially implemented |
| Commercial forecast overrides | F2.3 — Forecast Override | Not yet implemented |
| S&OP approved plan (events become inputs) | F4.2 — S&OP Module | Specified in feature_06_14 |
| ABC classification (for scope rules) | IPfeature11 | Implemented |

---

## 10. Out of Scope

- Automated promotion import from marketing calendar system (API integration with marketing platforms is a separate feature)
- Multi-wave promotion modeling (sequential discounts changing within the promo period)
- Promotion funding / trade spend tracking (belongs to Trade Promotion Management, not supply chain)
- Weather-driven demand prediction (external data source integration, separate feature)
- Competitor intelligence integration (separate market intelligence feature)
- SKU proliferation from promotions (pack changes, bundles treated as separate item_nos)

---

## 11. Test Requirements

### Backend Unit Tests (`tests/unit/test_event_planning.py`)

- `test_weeks_overlap_no_overlap()` — events with no overlap return 0.0 weeks
- `test_weeks_overlap_partial()` — partial month overlap returns correct fractional weeks
- `test_weeks_overlap_full_month()` — event spanning entire month returns ~4.33 weeks
- `test_promotion_adjustment_zero_uplift()` — uplift_pct=0 returns (0.0, 0.0)
- `test_promotion_adjustment_35pct()` — 450 base, 35% uplift, 1 week → adj ≈ 36.4
- `test_promotion_post_promo_dip()` — 10% pantry dip × 2 weeks → dip ≈ -20.8
- `test_s_curve_midpoint_50pct()` — at t=ramp_weeks/2, output ≈ peak_qty/2
- `test_s_curve_plateau_at_peak()` — at t >> ramp_weeks, output ≈ peak_qty
- `test_phase_out_full_decay()` — high decay_rate leads to near-zero after sufficient weeks
- `test_cannibalization_reduction()` — 25% cannibalization of 400 units = -100 adjustment
- `test_order_deadline_calculation()` — event_start April 15 - 14 days = April 1
- `test_conflict_detection_no_overlap()` — non-overlapping events return empty conflicts list
- `test_conflict_detection_overlap()` — overlapping events return correct conflict records

### Backend API Tests (`tests/api/test_events.py`)

- `test_get_calendar_200()` — returns events list with correct fields
- `test_get_calendar_type_filter()` — event_type=PROMOTION filter works
- `test_get_calendar_status_filter()` — status=approved filter works
- `test_post_event_created_201()` — returns event_id and status=draft
- `test_get_event_detail_200()` — returns single event with all parameters
- `test_get_event_404()` — returns 404 for non-existent event_id
- `test_approve_event_200()` — sets status=approved
- `test_get_impact_preview_200()` — returns chart_data and order_deadline
- `test_get_performance_200()` — returns lift_accuracy_pct and calibration_factor

### Frontend Tests (`src/tabs/__tests__/EventCalendarPanel.test.tsx`)

- Calendar grid renders correct day count for month
- Event blocks rendered on correct calendar days
- Impact preview chart renders with base and adjusted lines
- Event creation modal opens on "+ New Event" click
- Conflict warning renders when overlapping events detected
- Order deadline computed and displayed correctly
- Post-promo dip shown in a different color/pattern on the chart
