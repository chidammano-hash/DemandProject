# Feature F3.3 — Supplier Performance & Lead Time Learning

**Phase:** 3 — Operational Feedback Loop
**Feature Number:** F3.3 (file: feature_06_10)
**Status:** Design / Not Started
**Depends On:** IPfeature3 (Safety Stock), IPfeature12 (Supplier Performance), feature_06_09 (Service Level Tracking)

---

## 1. Problem Statement

The current `mv_supplier_performance` view (IPfeature12) aggregates receipt data to show high-level on-time delivery rates per supplier. However it does NOT:

- (A) Track promised vs. actual delivery dates at the **PO line level** — the view only knows when goods arrived, not when they were promised
- (B) Build a **statistical model** of each supplier's lead time distribution (mean, standard deviation, 90th percentile)
- (C) **Feed lead time actuals back into safety stock calculation** — SS is static after initial computation and never reacts when a supplier becomes unreliable

### The Lead Time Feedback Loop — Where It Breaks Today

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INTENDED LOOP (not working today)                                       │
│                                                                          │
│  ERP: PO issued         ──────────────────────────────────────────────► │
│  (promised_delivery_date)                                                │
│                                                                          │
│  ERP: Goods received    ──────────────────────────────────────────────► │
│  (actual_receipt_date)             fact_lead_time_actuals               │
│                                           │                             │
│                                           ▼                             │
│                                   dim_lead_time_profile                 │
│                                   (mean_lt, σ_lt, p90_lt)              │
│                                           │                             │
│                                           ▼                             │
│                           SS = Z × √(LT × σ_d² + μ_d² × σ_LT²)       │
│                                           │                             │
│                                           ▼                             │
│                                   fact_safety_stock_targets             │
│                                   (updated when σ_LT changes)          │
│                                                                          │
│  BREAKS HERE TODAY: promised_delivery_date is NOT captured. Actuals     │
│  arrive in inventory snapshot only as qty_on_hand change, with no PO   │
│  linkage, no promised date, no supplier attribution per receipt.        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Concrete Example of What Fails

Supplier "ABC Trading Co." has been delivering Item 100320 progressively later each month:

| Month | PO Issued | Promised Delivery | Actual Receipt | LT Promised | LT Actual | Variance |
|-------|-----------|------------------|----------------|-------------|-----------|----------|
| Jan   | Jan 3     | Jan 15           | Jan 14         | 12 days     | 11 days   | -1 day   |
| Feb   | Feb 2     | Feb 14           | Feb 16         | 12 days     | 14 days   | +2 days  |
| Mar   | Mar 3     | Mar 15           | Mar 21         | 12 days     | 18 days   | +6 days  |

The safety stock for Item 100320 was computed assuming LT=12 days and σ_LT≈0 (treated as fixed). Now σ_LT has increased to 3.6 days. The system has no idea. Stockouts result. The planner only finds out when the fill rate report shows 91% in March (feature_06_09).

---

## 2. Input Data Required

### Currently Available
- `mv_supplier_performance` — high-level supplier on-time rate (no PO-line detail)
- `fact_inventory_snapshot` — receipt quantities embedded in on-hand deltas
- `fact_safety_stock_targets` — current SS targets with assumed_lt_days (IPfeature3)
- `dim_dfu` — item-location-supplier mappings (partial — supplier_id must be added)

### Currently Missing — Must Be Sourced from ERP/WMS

| Data Element | Source System | Gap | Priority |
|---|---|---|---|
| `po_number` per receipt | ERP (SAP, Oracle, NetSuite) | Not linked to inventory snapshot | CRITICAL |
| `promised_delivery_date` per PO line | ERP purchase order module | Not captured at all | CRITICAL |
| `actual_receipt_date` per PO line | WMS or ERP goods receipt | Only visible as qty_on_hand change | CRITICAL |
| `supplier_id` per item-location | ERP vendor master | dim_dfu has no supplier_id | HIGH |
| `item_category` for supplier profiling | ERP item master | Partially in dim_item | MEDIUM |
| `ordered_qty` vs `received_qty` per line | ERP PO management | Not tracked | HIGH |

**For environments WITHOUT ERP integration:** A CSV import template will be provided to allow planners to manually upload PO receipt history. The script will process either source.

---

## 3. Data Model

### 3.1 `fact_lead_time_actuals` — PO Receipt History at Line Level

**Grain:** po_number + line_number (one row per purchase order line receipt)

```sql
CREATE TABLE fact_lead_time_actuals (
    receipt_id                  BIGSERIAL        PRIMARY KEY,
    po_number                   VARCHAR(50)      NOT NULL,
    line_number                 INTEGER          NOT NULL,
    item_no                     VARCHAR(50)      NOT NULL,
    loc                         VARCHAR(50)      NOT NULL,
    supplier_id                 VARCHAR(50)      NOT NULL,
    item_category               VARCHAR(50),
    ordered_qty                 NUMERIC(12,2)    NOT NULL,
    received_qty                NUMERIC(12,2)    NOT NULL,
    po_issue_date               DATE             NOT NULL,
    promised_delivery_date      DATE             NOT NULL,
    actual_receipt_date         DATE,            -- NULL if not yet received
    lead_time_days_promised     INTEGER          GENERATED ALWAYS AS
                                    (promised_delivery_date - po_issue_date) STORED,
    lead_time_days_actual       INTEGER          GENERATED ALWAYS AS
                                    (actual_receipt_date - po_issue_date) STORED,
    lead_time_variance_days     INTEGER          GENERATED ALWAYS AS
                                    (actual_receipt_date - promised_delivery_date) STORED,
    on_time                     BOOLEAN          GENERATED ALWAYS AS
                                    (actual_receipt_date <= promised_delivery_date) STORED,
    partial_receipt             BOOLEAN          GENERATED ALWAYS AS
                                    (received_qty < ordered_qty * 0.98) STORED,
    source_system               VARCHAR(30)      NOT NULL DEFAULT 'manual_import',
    created_at                  TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_lt_actuals_po_line
    ON fact_lead_time_actuals(po_number, line_number);

CREATE INDEX idx_lt_actuals_item_loc
    ON fact_lead_time_actuals(item_no, loc);

CREATE INDEX idx_lt_actuals_supplier
    ON fact_lead_time_actuals(supplier_id);

CREATE INDEX idx_lt_actuals_receipt_date
    ON fact_lead_time_actuals(actual_receipt_date DESC)
    WHERE actual_receipt_date IS NOT NULL;

CREATE INDEX idx_lt_actuals_supplier_category
    ON fact_lead_time_actuals(supplier_id, item_category);
```

### 3.2 `dim_lead_time_profile` — Statistical LT Profile per Supplier × Category × Location

**Grain:** supplier_id + item_category + loc (one row per supplier-category-location combination; loc can be NULL for supplier-wide aggregate)

```sql
CREATE TABLE dim_lead_time_profile (
    profile_id              SERIAL           PRIMARY KEY,
    supplier_id             VARCHAR(50)      NOT NULL,
    item_category           VARCHAR(50)      NOT NULL DEFAULT 'ALL',
    loc                     VARCHAR(50),     -- NULL = applies to all locations for this supplier
    mean_lt_days            NUMERIC(8,2)     NOT NULL,
    stddev_lt_days          NUMERIC(8,2)     NOT NULL,
    p50_lt_days             NUMERIC(8,2)     NOT NULL,
    p90_lt_days             NUMERIC(8,2)     NOT NULL,
    p95_lt_days             NUMERIC(8,2)     NOT NULL,
    min_lt_days             INTEGER          NOT NULL,
    max_lt_days             INTEGER          NOT NULL,
    on_time_delivery_rate   NUMERIC(5,2)     NOT NULL,  -- 0.00-100.00%
    partial_receipt_rate    NUMERIC(5,2)     NOT NULL,
    sample_size             INTEGER          NOT NULL,   -- number of PO lines
    sample_window_months    INTEGER          NOT NULL DEFAULT 12,
    last_computed           DATE             NOT NULL,
    -- Prior period for change detection
    prior_mean_lt_days      NUMERIC(8,2),
    prior_stddev_lt_days    NUMERIC(8,2),
    prior_otdr              NUMERIC(5,2),
    lt_mean_change_pct      NUMERIC(6,2),   -- (new-old)/old × 100
    lt_stddev_change_pct    NUMERIC(6,2),
    -- Review flags
    flagged_for_ss_review   BOOLEAN          NOT NULL DEFAULT FALSE,
    flag_reason             VARCHAR(100),   -- e.g. "stddev_increased_200pct"
    flag_resolved           BOOLEAN          NOT NULL DEFAULT FALSE,
    flag_resolved_at        TIMESTAMPTZ,
    created_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_lt_profile_supplier_cat_loc
    ON dim_lead_time_profile(supplier_id, item_category, COALESCE(loc, ''));

CREATE INDEX idx_lt_profile_flagged
    ON dim_lead_time_profile(flagged_for_ss_review)
    WHERE flagged_for_ss_review = TRUE;

CREATE INDEX idx_lt_profile_supplier
    ON dim_lead_time_profile(supplier_id);
```

### 3.3 `fact_lt_review_triggers` — Automatic SS Review Event Log

**Grain:** supplier_id + item_category + trigger_date (one row per trigger event)

```sql
CREATE TABLE fact_lt_review_triggers (
    trigger_id              SERIAL           PRIMARY KEY,
    supplier_id             VARCHAR(50)      NOT NULL,
    item_category           VARCHAR(50)      NOT NULL DEFAULT 'ALL',
    loc                     VARCHAR(50),
    trigger_date            DATE             NOT NULL,
    trigger_type            VARCHAR(50)      NOT NULL,  -- mean_lt_change / stddev_change / otdr_degradation
    old_mean_lt_days        NUMERIC(8,2),
    new_mean_lt_days        NUMERIC(8,2),
    old_stddev_lt_days      NUMERIC(8,2),
    new_stddev_lt_days      NUMERIC(8,2),
    stddev_change_pct       NUMERIC(6,2),
    old_otdr                NUMERIC(5,2),
    new_otdr                NUMERIC(5,2),
    affected_dfu_count      INTEGER          NOT NULL DEFAULT 0,
    affected_dfus           JSONB,           -- array of {item_no, loc, old_ss, new_ss}
    review_status           VARCHAR(20)      NOT NULL DEFAULT 'open',  -- open / acknowledged / resolved
    reviewed_by             VARCHAR(100),
    reviewed_at             TIMESTAMPTZ,
    auto_ss_updated         BOOLEAN          NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lt_triggers_supplier
    ON fact_lt_review_triggers(supplier_id, trigger_date DESC);

CREATE INDEX idx_lt_triggers_open
    ON fact_lt_review_triggers(review_status)
    WHERE review_status = 'open';
```

---

## 4. Python Script

### `scripts/update_lead_time_actuals.py`

```python
#!/usr/bin/env python3
"""
update_lead_time_actuals.py

Reads PO receipt history (from CSV import or ERP staging table),
computes LT statistics per supplier per item category,
updates dim_lead_time_profile, and triggers SS review flags when
supplier reliability degrades significantly.

Usage:
    uv run scripts/update_lead_time_actuals.py --input data/po_receipts.csv
    uv run scripts/update_lead_time_actuals.py --erp-source postgres_staging
    uv run scripts/update_lead_time_actuals.py --supplier-id "ABC Trading Co." --dry-run
"""

import argparse
import logging
import math
import yaml
import json
from datetime import date, timedelta
from typing import Optional
import psycopg
import pandas as pd
from common.db import get_db_params

CONFIG_PATH = "config/lead_time_config.yaml"
log = logging.getLogger(__name__)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_lt_statistics(lt_days_series: list[int]) -> dict:
    """
    Compute descriptive statistics for a series of lead time observations.

    Args:
        lt_days_series: List of actual lead time days for a supplier-category-loc group.

    Returns:
        Dict with mean, stddev, p50, p90, p95, min, max.
    """
    if not lt_days_series:
        return None
    s = sorted(lt_days_series)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    stddev = math.sqrt(variance)

    def percentile(data, p):
        idx = (p / 100) * (len(data) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
        return data[lo] + (idx - lo) * (data[hi] - data[lo])

    return {
        "mean_lt_days":  round(mean,  2),
        "stddev_lt_days": round(stddev, 2),
        "p50_lt_days":   round(percentile(s, 50), 2),
        "p90_lt_days":   round(percentile(s, 90), 2),
        "p95_lt_days":   round(percentile(s, 95), 2),
        "min_lt_days":   s[0],
        "max_lt_days":   s[-1],
        "sample_size":   n,
    }


def compute_recalculated_ss(
    mean_demand_daily: float,
    sigma_demand_daily: float,
    mean_lt_days: float,
    sigma_lt_days: float,
    z_score: float,
) -> float:
    """
    Full safety stock formula accounting for both demand and lead time variability.

    SS = Z × sqrt(LT × σ_demand² + μ_demand² × σ_LT²)

    Args:
        mean_demand_daily: Average daily demand (units/day)
        sigma_demand_daily: Standard deviation of daily demand
        mean_lt_days: Mean replenishment lead time (days)
        sigma_lt_days: Standard deviation of lead time (days)
        z_score: Service level Z-score (e.g., 2.054 for 98%)

    Returns:
        Recommended safety stock in units.
    """
    demand_variance_component  = mean_lt_days * (sigma_demand_daily ** 2)
    lt_variance_component      = (mean_demand_daily ** 2) * (sigma_lt_days ** 2)
    ss = z_score * math.sqrt(demand_variance_component + lt_variance_component)
    return round(ss, 2)


def detect_ss_review_trigger(
    old_stats: dict,
    new_stats: dict,
    cfg: dict,
) -> Optional[str]:
    """
    Returns a trigger_type string if the new stats represent a significant change
    vs the prior period that warrants SS review. Returns None if no trigger.

    Trigger conditions (all configurable in lead_time_config.yaml):
    - stddev_increased_pct > stddev_change_threshold (default 20%)
    - mean_lt_increased_days > mean_lt_change_threshold_days (default 3 days)
    - otdr_degraded_pct > otdr_change_threshold (default 5 percentage points)
    """
    if old_stats is None:
        return None

    stddev_threshold = cfg.get("stddev_change_trigger_pct", 20.0)
    mean_lt_threshold = cfg.get("mean_lt_change_trigger_days", 3.0)
    otdr_threshold = cfg.get("otdr_degradation_trigger_ppt", 5.0)

    if old_stats.get("stddev_lt_days", 0) > 0:
        stddev_change = (
            (new_stats["stddev_lt_days"] - old_stats["stddev_lt_days"])
            / old_stats["stddev_lt_days"]
        ) * 100
        if stddev_change > stddev_threshold:
            return "stddev_change"

    mean_change = new_stats["mean_lt_days"] - old_stats.get("mean_lt_days", new_stats["mean_lt_days"])
    if mean_change > mean_lt_threshold:
        return "mean_lt_change"

    old_otdr = old_stats.get("on_time_delivery_rate", 100.0)
    new_otdr = new_stats.get("on_time_delivery_rate", 100.0)
    if (old_otdr - new_otdr) > otdr_threshold:
        return "otdr_degradation"

    return None


def fetch_affected_dfus(conn, supplier_id: str, item_category: str) -> list[dict]:
    """Find all DFUs supplied by this supplier in this category for SS recalculation."""
    sql = """
        SELECT d.item_no, d.loc, t.ss_combined, t.service_level_target, t.z_score,
               t.mean_demand_daily, t.sigma_demand_daily, t.lead_time_days
        FROM dim_dfu d
        LEFT JOIN fact_safety_stock_targets t USING (item_no, loc)
        WHERE d.supplier_id = %s
          AND (%s = 'ALL' OR d.item_category = %s)
        LIMIT 500
    """
    cur = conn.execute(sql, (supplier_id, item_category, item_category))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def run(
    input_csv: Optional[str] = None,
    supplier_filter: Optional[str] = None,
    dry_run: bool = False,
    window_months: int = 12,
) -> None:
    cfg = load_config()
    log.info("Updating lead time profiles (dry_run=%s)", dry_run)

    with psycopg.connect(**get_db_params()) as conn:
        # Step 1: Load PO receipt actuals into fact_lead_time_actuals (if CSV provided)
        if input_csv:
            df = pd.read_csv(input_csv, parse_dates=["po_issue_date", "promised_delivery_date", "actual_receipt_date"])
            log.info("Loaded %d receipt rows from CSV", len(df))
            for _, row_data in df.iterrows():
                if supplier_filter and row_data["supplier_id"] != supplier_filter:
                    continue
                sql = """
                    INSERT INTO fact_lead_time_actuals
                        (po_number, line_number, item_no, loc, supplier_id, item_category,
                         ordered_qty, received_qty, po_issue_date,
                         promised_delivery_date, actual_receipt_date, source_system)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'csv_import')
                    ON CONFLICT (po_number, line_number) DO UPDATE SET
                        actual_receipt_date = EXCLUDED.actual_receipt_date,
                        received_qty        = EXCLUDED.received_qty,
                        updated_at          = NOW()
                """
                if not dry_run:
                    conn.execute(sql, (
                        row_data["po_number"], int(row_data["line_number"]),
                        row_data["item_no"], row_data["loc"], row_data["supplier_id"],
                        row_data.get("item_category", "ALL"),
                        float(row_data["ordered_qty"]), float(row_data["received_qty"]),
                        row_data["po_issue_date"].date(),
                        row_data["promised_delivery_date"].date(),
                        row_data["actual_receipt_date"].date() if pd.notna(row_data["actual_receipt_date"]) else None,
                    ))

        # Step 2: Compute LT statistics per supplier-category-loc
        window_start = date.today() - timedelta(days=window_months * 30)
        sql = """
            SELECT supplier_id, item_category, loc,
                   ARRAY_AGG(lead_time_days_actual) FILTER (WHERE lead_time_days_actual IS NOT NULL) AS lt_days,
                   COUNT(*)::INT                                          AS n_receipts,
                   AVG(on_time::INT)::NUMERIC(5,2) * 100                 AS otdr_pct,
                   AVG(partial_receipt::INT)::NUMERIC(5,2) * 100         AS partial_receipt_rate
            FROM fact_lead_time_actuals
            WHERE actual_receipt_date >= %s
              AND (%s IS NULL OR supplier_id = %s)
            GROUP BY supplier_id, item_category, loc
        """
        cur = conn.execute(sql, (window_start, supplier_filter, supplier_filter))
        groups = cur.fetchall()
        log.info("Computing LT profiles for %d supplier-category-loc groups", len(groups))

        for (supplier_id, item_category, loc, lt_days_arr, n_receipts, otdr_pct, partial_rate) in groups:
            stats = compute_lt_statistics(lt_days_arr or [])
            if stats is None or stats["sample_size"] < cfg.get("min_sample_size", 5):
                log.debug("Skipping %s/%s/%s — insufficient samples", supplier_id, item_category, loc)
                continue

            stats["on_time_delivery_rate"] = float(otdr_pct or 0)
            stats["partial_receipt_rate"]  = float(partial_rate or 0)

            # Fetch prior profile for change detection
            prior = conn.execute(
                """SELECT mean_lt_days, stddev_lt_days, on_time_delivery_rate
                   FROM dim_lead_time_profile
                   WHERE supplier_id=%s AND item_category=%s AND COALESCE(loc,'')=COALESCE(%s,'')""",
                (supplier_id, item_category, loc)
            ).fetchone()
            prior_dict = {"mean_lt_days": prior[0], "stddev_lt_days": prior[1],
                          "on_time_delivery_rate": prior[2]} if prior else None

            trigger_type = detect_ss_review_trigger(prior_dict, stats, cfg)
            flagged = trigger_type is not None

            # Compute change percentages
            mean_change_pct   = None
            stddev_change_pct = None
            if prior_dict:
                if prior_dict["mean_lt_days"] and prior_dict["mean_lt_days"] > 0:
                    mean_change_pct = round(
                        (stats["mean_lt_days"] - prior_dict["mean_lt_days"]) / prior_dict["mean_lt_days"] * 100, 2
                    )
                if prior_dict["stddev_lt_days"] and prior_dict["stddev_lt_days"] > 0:
                    stddev_change_pct = round(
                        (stats["stddev_lt_days"] - prior_dict["stddev_lt_days"]) / prior_dict["stddev_lt_days"] * 100, 2
                    )

            if not dry_run:
                # Upsert profile
                conn.execute("""
                    INSERT INTO dim_lead_time_profile
                        (supplier_id, item_category, loc, mean_lt_days, stddev_lt_days,
                         p50_lt_days, p90_lt_days, p95_lt_days, min_lt_days, max_lt_days,
                         on_time_delivery_rate, partial_receipt_rate, sample_size,
                         sample_window_months, last_computed,
                         prior_mean_lt_days, prior_stddev_lt_days, prior_otdr,
                         lt_mean_change_pct, lt_stddev_change_pct,
                         flagged_for_ss_review, flag_reason, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON CONFLICT (supplier_id, item_category, COALESCE(loc,'')) DO UPDATE SET
                        mean_lt_days            = EXCLUDED.mean_lt_days,
                        stddev_lt_days          = EXCLUDED.stddev_lt_days,
                        p90_lt_days             = EXCLUDED.p90_lt_days,
                        on_time_delivery_rate   = EXCLUDED.on_time_delivery_rate,
                        flagged_for_ss_review   = EXCLUDED.flagged_for_ss_review,
                        flag_reason             = EXCLUDED.flag_reason,
                        lt_mean_change_pct      = EXCLUDED.lt_mean_change_pct,
                        lt_stddev_change_pct    = EXCLUDED.lt_stddev_change_pct,
                        last_computed           = EXCLUDED.last_computed,
                        updated_at              = NOW()
                """, (
                    supplier_id, item_category, loc,
                    stats["mean_lt_days"], stats["stddev_lt_days"],
                    stats["p50_lt_days"], stats["p90_lt_days"], stats["p95_lt_days"],
                    stats["min_lt_days"], stats["max_lt_days"],
                    stats["on_time_delivery_rate"], stats["partial_receipt_rate"],
                    stats["sample_size"], window_months, date.today(),
                    prior_dict["mean_lt_days"] if prior_dict else None,
                    prior_dict["stddev_lt_days"] if prior_dict else None,
                    prior_dict["on_time_delivery_rate"] if prior_dict else None,
                    mean_change_pct, stddev_change_pct,
                    flagged, trigger_type,
                ))

                # Log review trigger
                if flagged and trigger_type:
                    affected = fetch_affected_dfus(conn, supplier_id, item_category)
                    conn.execute("""
                        INSERT INTO fact_lt_review_triggers
                            (supplier_id, item_category, loc, trigger_date, trigger_type,
                             old_mean_lt_days, new_mean_lt_days,
                             old_stddev_lt_days, new_stddev_lt_days,
                             stddev_change_pct,
                             affected_dfu_count, affected_dfus, review_status)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'open')
                    """, (
                        supplier_id, item_category, loc, date.today(), trigger_type,
                        prior_dict["mean_lt_days"] if prior_dict else None,
                        stats["mean_lt_days"],
                        prior_dict["stddev_lt_days"] if prior_dict else None,
                        stats["stddev_lt_days"],
                        stddev_change_pct,
                        len(affected),
                        json.dumps([{"item_no": d["item_no"], "loc": d["loc"]} for d in affected[:50]]),
                    ))
                    log.warning("SS REVIEW TRIGGERED: supplier=%s category=%s trigger=%s affected_dfus=%d",
                                supplier_id, item_category, trigger_type, len(affected))
            else:
                log.info("[DRY-RUN] supplier=%s category=%s mean_lt=%.1f σ_lt=%.2f otdr=%.1f%% flagged=%s",
                         supplier_id, item_category,
                         stats["mean_lt_days"], stats["stddev_lt_days"],
                         stats["on_time_delivery_rate"], flagged)

        if not dry_run:
            conn.commit()
        log.info("Lead time profile update complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to PO receipt CSV")
    parser.add_argument("--supplier-id", help="Filter to one supplier")
    parser.add_argument("--window-months", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(
        input_csv=args.input,
        supplier_filter=args.supplier_id,
        dry_run=args.dry_run,
        window_months=args.window_months,
    )
```

---

## 5. Config File

### `config/lead_time_config.yaml`

```yaml
# Lead Time Learning Configuration — Feature F3.3

# Minimum receipts before computing statistics for a supplier-category pair
min_sample_size: 5

# Rolling window for LT statistics
sample_window_months: 12

# Trigger thresholds for SS review
stddev_change_trigger_pct: 20.0       # If σ_LT increases by >20%, trigger review
mean_lt_change_trigger_days: 3.0      # If mean LT increases by >3 days, trigger review
otdr_degradation_trigger_ppt: 5.0     # If OTDR drops by >5 percentage points, trigger review

# Partial receipt threshold: received_qty < ordered_qty * (1 - threshold) = partial
partial_receipt_threshold_pct: 2.0

# Which ABC classes require SS recalculation on trigger
auto_review_abc_classes: [A, B]       # C-class triggers alert but no auto-SS-recalc

# Alert severity mapping for OTDR
otdr_thresholds:
  green: 95.0    # OTDR >= 95% = healthy
  amber: 85.0    # OTDR 85-95% = watch
  red: 0.0       # OTDR < 85% = critical
```

---

## 6. API Endpoints

### `GET /supply/supplier-lead-times?supplier_id=ABC+Trading+Co.`

Returns LT profile + trend for one supplier.

**Response:**
```json
{
  "supplier_id": "ABC Trading Co.",
  "profiles": [
    {
      "item_category": "ALL",
      "loc": null,
      "mean_lt_days": 14.3,
      "stddev_lt_days": 3.6,
      "p90_lt_days": 18.0,
      "on_time_delivery_rate": 71.4,
      "sample_size": 21,
      "last_computed": "2026-03-06",
      "prior_mean_lt_days": 12.5,
      "prior_stddev_lt_days": 1.2,
      "lt_mean_change_pct": 14.4,
      "lt_stddev_change_pct": 200.0,
      "flagged_for_ss_review": true,
      "flag_reason": "stddev_change"
    }
  ],
  "monthly_trend": [
    { "month": "2026-01-01", "avg_lt_days": 11.0, "otdr_pct": 100.0, "n_receipts": 7 },
    { "month": "2026-02-01", "avg_lt_days": 14.0, "otdr_pct": 71.4,  "n_receipts": 7 },
    { "month": "2026-03-01", "avg_lt_days": 18.0, "otdr_pct": 42.9,  "n_receipts": 7 }
  ]
}
```

### `GET /supply/supplier-lead-times/summary`

Aggregated across all suppliers.

**Response:**
```json
{
  "total_suppliers": 47,
  "flagged_for_review": 3,
  "avg_otdr_pct": 89.2,
  "suppliers_by_rag": { "green": 38, "amber": 6, "red": 3 },
  "top_degraded": [
    {
      "supplier_id": "ABC Trading Co.",
      "mean_lt_change_pct": 14.4,
      "stddev_change_pct": 200.0,
      "affected_dfus": 23,
      "rag": "red"
    }
  ]
}
```

### `GET /supply/lead-time-alerts`

Returns open SS review triggers.

**Response:**
```json
{
  "total_open": 3,
  "triggers": [
    {
      "trigger_id": 14,
      "supplier_id": "ABC Trading Co.",
      "item_category": "ALL",
      "trigger_date": "2026-03-06",
      "trigger_type": "stddev_change",
      "old_stddev_lt_days": 1.2,
      "new_stddev_lt_days": 3.6,
      "stddev_change_pct": 200.0,
      "affected_dfu_count": 23,
      "review_status": "open"
    }
  ]
}
```

### `POST /supply/lead-time-review/{trigger_id}/acknowledge`

Marks a review trigger as acknowledged (requires auth).

---

## 7. Frontend Components

### Location: Supplier Performance Panel in InvPlanningTab — Enhanced

The existing `SupplierPanel` gains a "Lead Time Reliability" sub-panel.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SUPPLIER PERFORMANCE — LEAD TIME RELIABILITY                           │
│                                                                          │
│  Alert: 3 suppliers have significant LT degradation                     │
│  23 DFUs need SS review — Supplier "ABC Trading Co." LT +6 days        │
│  [Review All Open Triggers]                                             │
│                                                                          │
│  Supplier               OTDR    Mean LT   σ_LT   P90 LT  LT Trend      │
│  ─────────────────────  ──────  ──────    ─────  ──────  ──────────     │
│  ABC Trading Co.  [RED] 71.4%   14.3d     3.6d   18.0d   ↗↗↗ +6d      │
│  XYZ Logistics  [AMBER] 88.1%   8.2d      1.8d   10.5d   ↗ +1.2d      │
│  Global Supply  [GREEN] 96.3%   7.0d      0.9d   8.1d    → stable      │
│                                                                          │
│  ABC Trading Co. — 12-Month LT Distribution:                           │
│                                                                          │
│   Days  ┤                              ▄▄▄                              │
│   20    ┤                          ▄▄▄███                              │
│   16    ┤              ▄▄▄     ▄▄▄████████                             │
│   12    ┤  ▄▄▄▄▄▄▄▄▄███████████████████████                            │
│    8    ┤──────────────────────────────────────────  Target=12d ───     │
│         └────────────────────────────────────────                       │
│           Jan    Feb   Mar   Apr   May   Jun   Jul                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Worked Example — Full Numbers

### Supplier: "ABC Trading Co." — Item 100320, Category ALL

**Step 1: Raw PO receipt data (fact_lead_time_actuals)**

| PO #    | Line | Promised | Actual   | LT Promised | LT Actual | On-Time |
|---------|------|----------|----------|-------------|-----------|---------|
| PO-1001 | 1    | Jan 15   | Jan 14   | 12          | 11        | TRUE    |
| PO-1001 | 2    | Jan 15   | Jan 15   | 12          | 12        | TRUE    |
| PO-1002 | 1    | Feb 14   | Feb 16   | 12          | 14        | FALSE   |
| PO-1003 | 1    | Mar 15   | Mar 21   | 12          | 18        | FALSE   |
| ... (21 receipts total over 3 months)

**Step 2: compute_lt_statistics()**
```
All LT actuals: [11, 12, 14, 18, 11, 12, 12, 13, 18, 17, 18, ...]
mean_lt_days   = 14.3 days
stddev_lt_days = 3.6 days
p90_lt_days    = 18.0 days
on_time_delivery_rate = 71.4% (15/21 on time)
```

**Step 3: Change detection vs prior period**
```
Prior (Q4 2025):
  prior_mean_lt_days   = 12.5 days
  prior_stddev_lt_days = 1.2 days
  prior_otdr           = 95.2%

Changes:
  mean_lt_change_pct   = (14.3 - 12.5) / 12.5 × 100 = +14.4%
  stddev_change_pct    = (3.6 - 1.2) / 1.2 × 100     = +200%  ← TRIGGERS REVIEW

detect_ss_review_trigger() returns "stddev_change"
```

**Step 4: SS recalculation for affected DFUs**

For Item 100320 (A-class, Z=2.054):
```
Given:
  mean_demand_daily    = 3.0 units/day
  sigma_demand_daily   = 0.8 units/day
  mean_lt_days_old     = 12.5 days (prior)
  sigma_lt_days_old    = 1.2 days  (prior)
  mean_lt_days_new     = 14.3 days (current)
  sigma_lt_days_new    = 3.6 days  (current)

OLD SS (with σ_LT = 1.2):
  SS_old = 2.054 × sqrt(12.5 × 0.8² + 3.0² × 1.2²)
         = 2.054 × sqrt(8.0 + 12.96)
         = 2.054 × sqrt(20.96)
         = 2.054 × 4.578
         = 9.40 days × 3.0 units/day = 28 units (simplified to full SS=60 in DB)

NEW SS (with σ_LT = 3.6):
  SS_new = 2.054 × sqrt(14.3 × 0.8² + 3.0² × 3.6²)
         = 2.054 × sqrt(9.152 + 116.64)
         = 2.054 × sqrt(125.79)
         = 2.054 × 11.215
         = 23.04 days × 3.0 units/day = 69 units

  Scale factor applied to recorded SS baseline (60 units):
  SS_recommended = 60 × (23.04 / 9.40) = 60 × 2.45 = 147 units

  Cap at max_eoq_months_supply guideline:
  Monthly demand = 90 units → 2 months supply = 180 units cap → 147 accepted
```

**Step 5: Alert generated**
```
fact_lt_review_triggers row:
  supplier_id       = "ABC Trading Co."
  trigger_type      = "stddev_change"
  old_stddev_lt     = 1.2 days → new = 3.6 days (+200%)
  affected_dfu_count = 23
  review_status     = "open"

Alert in UI: "23 DFUs need SS review — Supplier 'ABC Trading Co.' lead time σ increased
              from 1.2 to 3.6 days (+200%). Recommend SS increase of ~145% for A-class items."
```

---

## 9. Historical LT Trend Chart

The 12-month LT trend chart shows per-supplier LT drift:

```
Mean LT (days)
  20 │                                           ●
  18 │                                       ●
  16 │
  14 │                                   ●
  12 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ●─ Target (12d)
  10 │                   ●   ●   ●
   8 │   ●   ●   ●   ●
     └────────────────────────────────────────────
       Apr May Jun Jul Aug Sep Oct Nov Dec Jan Feb Mar

σ_LT (days)
   4 │                                           ●
   3 │                                       ●
   2 │
   1 │─●──●──●──●──●──●──●──●──●──●──●────  Historical σ
   0 └────────────────────────────────────────────
```

---

## 10. Dependencies

| Dependency | Required For | Status |
|---|---|---|
| IPfeature3 — Safety Stock | `ss_units_planned`, SS recalculation formula | Implemented |
| IPfeature12 — Supplier Performance | Existing supplier aggregation to extend | Implemented |
| feature_06_09 — SL Tracking | LT actuals feed into miss reason attribution | Planned (F3.2) |
| ERP/WMS Integration (F1.3) | `promised_delivery_date`, `actual_receipt_date` | Future |

---

## 11. Out of Scope

- Real-time PO tracking (push notifications when PO is delayed) — requires ERP webhook integration
- Multi-tier supplier (tier 2/3 vendor risk) — out of scope for this phase
- Currency/landed cost changes in supplier switch scenarios
- Automated purchase order generation — planning recommendation only
- Supplier scorecards with weighted multi-metric scoring — covered by IPfeature12

---

## 12. Test Requirements

### Backend Unit Tests — `tests/unit/test_lead_time_learning.py`

```
test_compute_lt_statistics_basic()               — mean, stddev, p90 from sample
test_compute_lt_statistics_single_value()        — n=1 → stddev=0
test_compute_lt_statistics_empty_returns_none()  — empty list → None
test_compute_recalculated_ss_formula()           — Z × sqrt(LT×σd² + μd²×σLT²)
test_compute_recalculated_ss_zero_lt_variance()  — σ_LT=0 reduces to simple formula
test_detect_trigger_stddev_change()              — 200% σ increase → "stddev_change"
test_detect_trigger_mean_lt_change()             — +3 days mean → "mean_lt_change"
test_detect_trigger_otdr_degradation()           — OTDR drops 5+ points → trigger
test_detect_trigger_no_change()                  — small changes → None
test_detect_trigger_no_prior_returns_none()      — first computation, no prior
test_on_time_flag_correct()                      — actual <= promised → on_time=TRUE
test_partial_receipt_flag()                      — received < 98% ordered → partial=TRUE
```

### Backend API Tests — `tests/api/test_lead_time.py`

```
test_supplier_lt_profile_200()
test_supplier_lt_trend_12_months()
test_supplier_summary_rag_distribution()
test_lead_time_alerts_open_only()
test_acknowledge_trigger_requires_auth()
test_acknowledge_trigger_updates_status()
test_unknown_supplier_returns_empty()
```

### Make Targets to Add

```makefile
lt-schema:        # Apply DDL (fact_lead_time_actuals, dim_lead_time_profile, fact_lt_review_triggers)
lt-import:        # Import from CSV: make lt-import INPUT=data/po_receipts.csv
lt-update:        # Run update_lead_time_actuals.py
lt-update-dry:    # --dry-run preview
lt-all:           # lt-schema + lt-update
```
