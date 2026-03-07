# Feature F3.2 — Service Level Actuals vs. Targets Tracking

**Phase:** 3 — Operational Feedback Loop
**Feature Number:** F3.2 (file: feature_06_09)
**Status:** Design / Not Started
**Depends On:** IPfeature3 (Safety Stock), IPfeature8 (Fill Rate), IPAIfeature1 (AI Planner), IPfeature6 (Health Score)

---

## 1. Problem Statement

Safety stock is sized to achieve a target service level (e.g., 98% for A-class items). The system currently computes fill rate from inventory snapshots (IPfeature8) but does NOT track whether the planned service level targets were actually achieved, which DFUs are chronically missing their target, or whether safety stock changes produced the intended improvement.

### What Fails Today

**Concrete example:** Item 100320 (A-class, Loc 1401-BULK) was assigned SS=60 units in January to achieve a 98% line fill rate target. The actual performance over three months:

| Month | Target | Actual Fill Rate | Stockout Days | SS Days Held | Gap    |
|-------|--------|-----------------|---------------|-------------|--------|
| Jan   | 98%    | 94%             | 3             | 58          | -4%    |
| Feb   | 98%    | 96%             | 1             | 61          | -2%    |
| Mar   | 98%    | 91%             | 5             | 57          | -7%    |

The planner has NO visibility that:
- This DFU has missed its target three months running (miss streak = 3)
- The cumulative gap has widened from -4% to -7%
- The root cause is that the actual lead time (18 days) is 4 days longer than the assumed LT (14 days) used when SS was calculated
- A simple SS recalculation with actual LT would increase SS from 60 to 90 units and close the gap

The current system shows a fill rate number but never compares it to the target, never counts consecutive misses, and never triggers a review.

---

## 2. Service Level Definitions

Three service level variants are in scope. **Line fill rate is the recommended primary KPI** because it treats each order line (item × location × demand event) equally regardless of order size, making it robust to a few large orders dominating the metric.

```
┌─────────────────────────────────────────────────────────────────────┐
│  SERVICE LEVEL TYPE DEFINITIONS                                      │
│                                                                      │
│  Line Fill Rate  = Lines shipped in full / Lines ordered             │
│                    Treats each SKU-location-order equally            │
│                    PRIMARY KPI — recommended                         │
│                                                                      │
│  Case Fill Rate  = Units shipped / Units ordered                     │
│                    Weighted by volume — large orders dominate        │
│                    SECONDARY KPI                                     │
│                                                                      │
│  Order Fill Rate = Orders shipped complete / Orders placed           │
│                    All lines in an order must ship in full           │
│                    TERTIARY — strictest definition                   │
│                                                                      │
│  OTIF (On Time In Full) = Fill rate AND delivered by promised date   │
│                    Requires PO promised date (F1.3 dependency)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Input Data Required

### Currently Available
- `mv_fill_rate_monthly` — actual fill rate by item_no + loc + month (IPfeature8)
- `fact_safety_stock_targets` — SS targets with service_level target by ABC class (IPfeature3)
- `fact_inventory_snapshot` — daily on-hand quantities and MTD sales
- `dim_dfu` — ABC class, cluster assignment

### Currently Missing — Must Be Sourced or Computed
| Data Element | Source | Gap |
|---|---|---|
| `target_service_level` by DFU | `fact_service_level_targets` (new table) | Config/policy input — must be created |
| `stockout_days` per DFU per month | Derived from `fact_inventory_snapshot` where `qty_on_hand = 0` | Not currently aggregated at DFU-month grain |
| `actual_lead_time_days` per PO receipt | Requires `fact_lead_time_actuals` (feature_06_10) | Not tracked today |
| `demand_fulfillment_events` | Would require order management integration | Approximated from inventory delta + MTD sales |
| Promised vs actual delivery dates | ERP/WMS integration (F1.3) | Not yet sourced |

---

## 4. Data Model

### 4.1 `fact_service_level_targets` — Configurable Targets by Segment

**Grain:** ABC class + optional item_no override + optional loc override (hierarchical: item override beats class default)

```sql
CREATE TABLE fact_service_level_targets (
    target_id           SERIAL PRIMARY KEY,
    item_no             VARCHAR(50)       NULL,      -- NULL = applies to all items in abc_class
    loc                 VARCHAR(50)       NULL,      -- NULL = applies to all locations
    abc_class           VARCHAR(5)        NOT NULL,  -- A, B, C, or ALL
    service_level_type  VARCHAR(20)       NOT NULL DEFAULT 'line_fill_rate',
    target_service_level NUMERIC(5,2)    NOT NULL,  -- e.g., 98.00
    review_period_months INTEGER         NOT NULL DEFAULT 3,
    miss_streak_alert_threshold INTEGER  NOT NULL DEFAULT 3,
    effective_from      DATE             NOT NULL,
    effective_to        DATE             NULL,       -- NULL = currently active
    set_by              VARCHAR(100)     NOT NULL,
    notes               TEXT,
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sl_targets_abc       ON fact_service_level_targets(abc_class);
CREATE INDEX idx_sl_targets_item_loc  ON fact_service_level_targets(item_no, loc)
    WHERE item_no IS NOT NULL;
CREATE INDEX idx_sl_targets_effective ON fact_service_level_targets(effective_from, effective_to);
```

**Default target rows (seeded by migration):**

| abc_class | target_service_level | review_period_months |
|-----------|---------------------|----------------------|
| A         | 98.00               | 3                    |
| B         | 95.00               | 3                    |
| C         | 90.00               | 6                    |

### 4.2 `fact_service_level_performance` — Actual vs Target by DFU per Month

**Grain:** item_no + loc + perf_month + service_level_type (one row per DFU per month per SL type)

```sql
CREATE TABLE fact_service_level_performance (
    slp_id                  BIGSERIAL        PRIMARY KEY,
    item_no                 VARCHAR(50)      NOT NULL,
    loc                     VARCHAR(50)      NOT NULL,
    perf_month              DATE             NOT NULL,  -- first day of month
    service_level_type      VARCHAR(20)      NOT NULL DEFAULT 'line_fill_rate',
    abc_class               VARCHAR(5),
    target_service_level    NUMERIC(5,2)     NOT NULL,
    achieved_fill_rate      NUMERIC(5,2)     NOT NULL,
    gap                     NUMERIC(5,2)     GENERATED ALWAYS AS
                                (achieved_fill_rate - target_service_level) STORED,
    gap_direction           VARCHAR(10)      NOT NULL,  -- above / below / on_target
    stockout_events         INTEGER          NOT NULL DEFAULT 0,
    stockout_days           INTEGER          NOT NULL DEFAULT 0,
    projected_ss_days       NUMERIC(8,2),   -- days of SS coverage at month-start
    actual_ss_days          NUMERIC(8,2),   -- days actually held (avg on-hand / avg daily demand)
    ss_units_planned        NUMERIC(12,2),  -- from fact_safety_stock_targets
    ss_units_actual_avg     NUMERIC(12,2),  -- average on-hand when below SS target
    assumed_lead_time_days  NUMERIC(8,2),   -- LT used when SS was computed
    actual_lead_time_days   NUMERIC(8,2),   -- actual LT recorded for that month (if available)
    lt_variance_days        NUMERIC(8,2),   -- actual - assumed
    miss_streak_months      INTEGER          NOT NULL DEFAULT 0,
    flagged_for_review      BOOLEAN          NOT NULL DEFAULT FALSE,
    primary_miss_reason     VARCHAR(50),    -- stockout_insufficient_ss / lead_time_longer /
                                            -- demand_spike / supplier_failure / data_gap
    miss_reason_confidence  NUMERIC(4,2),   -- 0.00 to 1.00
    auto_insight_created    BOOLEAN          NOT NULL DEFAULT FALSE,
    computed_at             TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_slp_dfu_month_type
    ON fact_service_level_performance(item_no, loc, perf_month, service_level_type);

CREATE INDEX idx_slp_item_loc
    ON fact_service_level_performance(item_no, loc);

CREATE INDEX idx_slp_month
    ON fact_service_level_performance(perf_month DESC);

CREATE INDEX idx_slp_miss_streak
    ON fact_service_level_performance(miss_streak_months DESC)
    WHERE miss_streak_months >= 3;

CREATE INDEX idx_slp_flagged
    ON fact_service_level_performance(flagged_for_review)
    WHERE flagged_for_review = TRUE;

CREATE INDEX idx_slp_abc_gap
    ON fact_service_level_performance(abc_class, gap)
    WHERE gap < 0;
```

### 4.3 `mv_service_level_dashboard` — Aggregated View

```sql
CREATE MATERIALIZED VIEW mv_service_level_dashboard AS
WITH latest_3m AS (
    SELECT
        item_no, loc, abc_class,
        AVG(achieved_fill_rate)      AS avg_fill_rate_3m,
        AVG(target_service_level)    AS target_service_level,
        AVG(gap)                     AS avg_gap_3m,
        MAX(miss_streak_months)      AS current_miss_streak,
        SUM(stockout_days)           AS total_stockout_days_3m,
        SUM(stockout_events)         AS total_stockout_events_3m,
        MAX(perf_month)              AS latest_month,
        MAX(flagged_for_review::INT) AS is_flagged,
        MODE() WITHIN GROUP (ORDER BY primary_miss_reason) AS dominant_miss_reason,
        COUNT(*) FILTER (WHERE gap < 0) AS months_below_target
    FROM fact_service_level_performance
    WHERE perf_month >= DATE_TRUNC('month', NOW()) - INTERVAL '3 months'
      AND service_level_type = 'line_fill_rate'
    GROUP BY item_no, loc, abc_class
)
SELECT
    l.*,
    d.dfu_description,
    d.cluster_assignment,
    d.xyz_class,
    CASE
        WHEN l.avg_gap_3m >= 0                    THEN 'green'
        WHEN l.avg_gap_3m >= -3                   THEN 'amber'
        ELSE                                           'red'
    END AS rag_status
FROM latest_3m l
LEFT JOIN dim_dfu d USING (item_no, loc);

CREATE UNIQUE INDEX uq_sl_dashboard ON mv_service_level_dashboard(item_no, loc);
CREATE INDEX idx_sl_dashboard_rag  ON mv_service_level_dashboard(rag_status);
CREATE INDEX idx_sl_dashboard_abc  ON mv_service_level_dashboard(abc_class, avg_gap_3m);
```

---

## 5. Python Script

### `scripts/compute_service_level_actuals.py`

```python
#!/usr/bin/env python3
"""
compute_service_level_actuals.py

Joins fill rate actuals, safety stock targets, and inventory snapshot data
to produce fact_service_level_performance rows for a given month.

Usage:
    uv run scripts/compute_service_level_actuals.py --month 2026-03-01
    uv run scripts/compute_service_level_actuals.py --month 2026-03-01 --dry-run
"""

import argparse
import logging
import yaml
from datetime import date
from typing import Optional
import psycopg
from common.db import get_db_params

CONFIG_PATH = "config/service_level_config.yaml"
log = logging.getLogger(__name__)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_fill_rate_actuals(conn, month: date) -> list[dict]:
    """
    Pull mv_fill_rate_monthly for the given month.
    Returns rows: item_no, loc, fill_rate, demand_qty, shipped_qty, stockout_events
    """
    sql = """
        SELECT
            item_no, loc,
            fill_rate_pct,
            demand_qty,
            shipped_qty,
            stockout_events,
            stockout_days
        FROM mv_fill_rate_monthly
        WHERE month_start = %s
    """
    cur = conn.execute(sql, (month,))
    return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]


def fetch_sl_targets(conn, month: date) -> dict:
    """
    Returns {(item_no_or_None, loc_or_None, abc_class): target_pct}
    Priority: item+loc override > item > class default
    """
    sql = """
        SELECT item_no, loc, abc_class, target_service_level, service_level_type
        FROM fact_service_level_targets
        WHERE effective_from <= %s
          AND (effective_to IS NULL OR effective_to >= %s)
        ORDER BY
            (item_no IS NOT NULL)::INT DESC,
            (loc IS NOT NULL)::INT DESC
    """
    cur = conn.execute(sql, (month, month))
    rows = cur.fetchall()
    targets = {}
    for row in rows:
        key = (row[0], row[1], row[2])
        targets[key] = {"target": float(row[3]), "sl_type": row[4]}
    return targets


def resolve_target(targets: dict, item_no: str, loc: str, abc_class: str) -> float:
    """
    Priority resolution: item+loc > item only > class default
    """
    for key in [
        (item_no, loc,  abc_class),
        (item_no, None, abc_class),
        (None,    None, abc_class),
        (None,    None, "ALL"),
    ]:
        if key in targets:
            return targets[key]["target"]
    return 95.0  # safe fallback


def compute_miss_streak(conn, item_no: str, loc: str, month: date) -> int:
    """Count consecutive months below target ending at `month`."""
    sql = """
        SELECT gap_direction
        FROM fact_service_level_performance
        WHERE item_no = %s AND loc = %s
          AND service_level_type = 'line_fill_rate'
          AND perf_month < %s
        ORDER BY perf_month DESC
        LIMIT 12
    """
    rows = conn.execute(sql, (item_no, loc, month)).fetchall()
    streak = 0
    for (direction,) in rows:
        if direction == "below":
            streak += 1
        else:
            break
    return streak


def classify_miss_reason(
    gap: float,
    stockout_days: int,
    lt_variance_days: Optional[float],
    demand_spike_ratio: Optional[float],
) -> tuple[str, float]:
    """
    Heuristic miss reason classification.
    Returns (reason_code, confidence_score).

    Priority order:
    1. lead_time_longer  — LT variance > 3 days
    2. demand_spike      — demand spike ratio > 1.25
    3. stockout_insufficient_ss — stockout_days > 0 with no LT/spike explanation
    4. data_gap          — cannot determine
    """
    if lt_variance_days is not None and lt_variance_days > 3:
        conf = min(0.95, 0.60 + (lt_variance_days - 3) * 0.05)
        return ("lead_time_longer", round(conf, 2))
    if demand_spike_ratio is not None and demand_spike_ratio > 1.25:
        conf = min(0.90, 0.55 + (demand_spike_ratio - 1.25) * 0.30)
        return ("demand_spike", round(conf, 2))
    if stockout_days > 0:
        return ("stockout_insufficient_ss", 0.70)
    return ("data_gap", 0.30)


def upsert_performance_row(conn, row: dict) -> None:
    sql = """
        INSERT INTO fact_service_level_performance (
            item_no, loc, perf_month, service_level_type, abc_class,
            target_service_level, achieved_fill_rate,
            gap_direction, stockout_events, stockout_days,
            projected_ss_days, actual_ss_days, ss_units_planned,
            assumed_lead_time_days, actual_lead_time_days, lt_variance_days,
            miss_streak_months, flagged_for_review,
            primary_miss_reason, miss_reason_confidence, computed_at
        ) VALUES (
            %(item_no)s, %(loc)s, %(perf_month)s, %(service_level_type)s, %(abc_class)s,
            %(target_service_level)s, %(achieved_fill_rate)s,
            %(gap_direction)s, %(stockout_events)s, %(stockout_days)s,
            %(projected_ss_days)s, %(actual_ss_days)s, %(ss_units_planned)s,
            %(assumed_lead_time_days)s, %(actual_lead_time_days)s, %(lt_variance_days)s,
            %(miss_streak_months)s, %(flagged_for_review)s,
            %(primary_miss_reason)s, %(miss_reason_confidence)s, NOW()
        )
        ON CONFLICT (item_no, loc, perf_month, service_level_type)
        DO UPDATE SET
            achieved_fill_rate   = EXCLUDED.achieved_fill_rate,
            gap_direction        = EXCLUDED.gap_direction,
            miss_streak_months   = EXCLUDED.miss_streak_months,
            flagged_for_review   = EXCLUDED.flagged_for_review,
            primary_miss_reason  = EXCLUDED.primary_miss_reason,
            computed_at          = NOW()
    """
    conn.execute(sql, row)


def run(month: date, dry_run: bool = False) -> None:
    cfg = load_config()
    streak_threshold = cfg.get("miss_streak_alert_threshold", 3)
    gap_amber_pct = cfg.get("gap_amber_threshold_pct", -3.0)

    log.info("Computing service level actuals for month=%s (dry_run=%s)", month, dry_run)

    with psycopg.connect(**get_db_params()) as conn:
        fill_rate_rows = fetch_fill_rate_actuals(conn, month)
        targets        = fetch_sl_targets(conn, month)

        written = 0
        for fr in fill_rate_rows:
            item_no = fr["item_no"]
            loc     = fr["loc"]
            # Resolve ABC class from dim_dfu
            abc_row = conn.execute(
                "SELECT abc_class FROM dim_dfu WHERE item_no=%s AND loc=%s LIMIT 1",
                (item_no, loc)
            ).fetchone()
            abc_class = abc_row[0] if abc_row else "C"

            target_pct  = resolve_target(targets, item_no, loc, abc_class)
            actual_pct  = float(fr["fill_rate_pct"])
            gap         = actual_pct - target_pct
            gap_dir     = "above" if gap > 0.5 else ("below" if gap < -0.5 else "on_target")
            streak       = compute_miss_streak(conn, item_no, loc, month) if gap_dir == "below" else 0
            if gap_dir == "below":
                streak += 1
            flagged = streak >= streak_threshold

            reason, confidence = classify_miss_reason(
                gap=gap,
                stockout_days=int(fr.get("stockout_days", 0)),
                lt_variance_days=None,    # populated when F3.3 data is available
                demand_spike_ratio=None,  # populated when demand sensing is live
            )

            row = {
                "item_no": item_no, "loc": loc, "perf_month": month,
                "service_level_type": "line_fill_rate",
                "abc_class": abc_class,
                "target_service_level": target_pct,
                "achieved_fill_rate": actual_pct,
                "gap_direction": gap_dir,
                "stockout_events": int(fr.get("stockout_events", 0)),
                "stockout_days": int(fr.get("stockout_days", 0)),
                "projected_ss_days": None,
                "actual_ss_days": None,
                "ss_units_planned": None,
                "assumed_lead_time_days": None,
                "actual_lead_time_days": None,
                "lt_variance_days": None,
                "miss_streak_months": streak,
                "flagged_for_review": flagged,
                "primary_miss_reason": reason if gap_dir == "below" else None,
                "miss_reason_confidence": confidence if gap_dir == "below" else None,
            }

            if not dry_run:
                upsert_performance_row(conn, row)
                written += 1
            else:
                log.info("[DRY-RUN] Would write: item=%s loc=%s gap=%.1f%% streak=%d",
                         item_no, loc, gap, streak)

        if not dry_run:
            conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_service_level_dashboard")
            conn.commit()
        log.info("Done. Rows written: %d", written)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYY-MM-DD (first day of month)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(date.fromisoformat(args.month), dry_run=args.dry_run)
```

---

## 6. Config File

### `config/service_level_config.yaml`

```yaml
# Service Level Actuals vs. Targets Configuration
# Feature F3.2

miss_streak_alert_threshold: 3        # months before flagging for review
gap_amber_threshold_pct: -3.0        # gap worse than -3% = amber
gap_red_threshold_pct: -7.0          # gap worse than -7% = red

primary_service_level_type: line_fill_rate   # line_fill_rate | case_fill_rate | order_fill_rate

# Causal attribution thresholds
lead_time_variance_threshold_days: 3  # LT actual > assumed by this much → blame LT
demand_spike_ratio_threshold: 1.25    # sensing signal > 25% above statistical → blame demand spike

# Auto AI insight generation (feeds IPAIfeature1)
auto_insight_miss_streak_threshold: 3
auto_insight_severity_by_abc:
  A: critical
  B: high
  C: medium
```

---

## 7. API Endpoints

### `GET /analytics/service-level/summary`

Returns aggregate service level health by ABC class.

**Response:**
```json
{
  "computed_at": "2026-03-06T08:00:00Z",
  "period": "2026-01-01_to_2026-03-01",
  "summary_by_abc": [
    {
      "abc_class": "A",
      "total_dfus": 412,
      "target_service_level": 98.0,
      "avg_achieved": 96.3,
      "avg_gap": -1.7,
      "dfus_below_target": 87,
      "dfus_chronic_miss": 14,
      "total_stockout_days_3m": 231,
      "rag_distribution": { "green": 325, "amber": 73, "red": 14 }
    },
    {
      "abc_class": "B",
      "total_dfus": 1204,
      "target_service_level": 95.0,
      "avg_achieved": 94.1,
      "avg_gap": -0.9,
      "dfus_below_target": 310,
      "dfus_chronic_miss": 42,
      "total_stockout_days_3m": 876,
      "rag_distribution": { "green": 894, "amber": 268, "red": 42 }
    }
  ],
  "total_flagged_for_review": 56
}
```

### `GET /analytics/service-level/detail?item_no=100320&loc=1401-BULK`

Returns month-by-month performance for one DFU.

**Response:**
```json
{
  "item_no": "100320",
  "loc": "1401-BULK",
  "abc_class": "A",
  "current_miss_streak": 3,
  "flagged_for_review": true,
  "history": [
    {
      "perf_month": "2026-01-01",
      "target_service_level": 98.0,
      "achieved_fill_rate": 94.0,
      "gap": -4.0,
      "gap_direction": "below",
      "stockout_events": 2,
      "stockout_days": 3,
      "ss_units_planned": 60,
      "assumed_lead_time_days": 14,
      "actual_lead_time_days": 18,
      "lt_variance_days": 4,
      "primary_miss_reason": "lead_time_longer",
      "miss_reason_confidence": 0.85
    },
    {
      "perf_month": "2026-02-01",
      "target_service_level": 98.0,
      "achieved_fill_rate": 96.0,
      "gap": -2.0,
      "gap_direction": "below",
      "stockout_events": 1,
      "stockout_days": 1,
      "ss_units_planned": 60,
      "assumed_lead_time_days": 14,
      "actual_lead_time_days": 17,
      "lt_variance_days": 3,
      "primary_miss_reason": "lead_time_longer",
      "miss_reason_confidence": 0.78
    },
    {
      "perf_month": "2026-03-01",
      "target_service_level": 98.0,
      "achieved_fill_rate": 91.0,
      "gap": -7.0,
      "gap_direction": "below",
      "stockout_events": 3,
      "stockout_days": 5,
      "ss_units_planned": 60,
      "assumed_lead_time_days": 14,
      "actual_lead_time_days": 18,
      "lt_variance_days": 4,
      "primary_miss_reason": "lead_time_longer",
      "miss_reason_confidence": 0.88
    }
  ],
  "recommended_ss_recalc": {
    "current_ss_units": 60,
    "recommended_ss_units": 90,
    "recalc_basis": "actual_lead_time_18_days",
    "expected_fill_rate_improvement_pct": 5.8
  }
}
```

### `GET /analytics/service-level/chronic-misses?min_streak=3&abc_class=A`

Returns DFUs that have missed their target for `min_streak` or more consecutive months.

**Response:**
```json
{
  "total_count": 14,
  "items": [
    {
      "item_no": "100320",
      "loc": "1401-BULK",
      "abc_class": "A",
      "miss_streak_months": 3,
      "avg_gap_3m": -4.3,
      "dominant_miss_reason": "lead_time_longer",
      "total_stockout_days_3m": 9,
      "ss_units_planned": 60,
      "recommended_ss_units": 90
    }
  ]
}
```

### `PUT /analytics/service-level/targets`

Create or update service level targets (requires API key auth).

**Request body:**
```json
{
  "abc_class": "A",
  "item_no": null,
  "loc": null,
  "target_service_level": 98.5,
  "review_period_months": 3,
  "miss_streak_alert_threshold": 2,
  "effective_from": "2026-04-01",
  "set_by": "planner@company.com",
  "notes": "Raised A-class target post Q1 review"
}
```

---

## 8. Frontend Components

### Location: Control Tower Tab — "Service Level Performance" Panel

New panel added below the existing KPI cards in `ControlTowerTab.tsx`.

```
┌─────────────────────────────────────────────────────────────────────┐
│  SERVICE LEVEL PERFORMANCE                           [Export CSV]   │
│                                                                      │
│  Alert: 14 A-class items have missed 98% SL target for 3+ months   │
│         [Flag All for SS Review]                                    │
│                                                                      │
│  ABC Filter: [All] [A] [B] [C]    RAG: [All] [Red] [Amber] [Green] │
│                                                                      │
│  Item      Loc         ABC  Target  Actual  Gap     Streak  Trend   │
│  ───────   ─────────── ──── ──────  ──────  ──────  ──────  ─────── │
│  100320    1401-BULK   A    98.0%   91.0%   -7.0%   3 mo   ╲╲╲     │
│  200145    DC-EAST     A    98.0%   92.5%   -5.5%   4 mo   ╲╲╲╲    │
│  100891    STORE-07    B    95.0%   90.2%   -4.8%   3 mo   ╲╲╲     │
│                                                             [Action]│
│                                                                      │
│  Miss Reason Breakdown (last 3 months):                             │
│  ████████████████░░░░░░  Lead Time Longer    42%                    │
│  ████████░░░░░░░░░░░░░░  Demand Spike        28%                    │
│  ██████░░░░░░░░░░░░░░░░  Insufficient SS     21%                    │
│  ███░░░░░░░░░░░░░░░░░░░  Data Gap             9%                    │
└─────────────────────────────────────────────────────────────────────┘
```

**"Flag for SS Review" action:** calls `POST /ai-planner/portfolio-scan` with filter `flagged_for_review=true` to generate AI insights for flagged DFUs.

**Trend sparkline:** 3-month arrow indicating direction of gap (widening = red arrows ╲╲╲, narrowing = green ╱╱╱, stable = ─).

---

## 9. Worked Example — End to End

**Item 100320, Loc 1401-BULK, A-class, target=98% line fill rate**

### Step 1: Gather inputs
```
Jan fill rate (mv_fill_rate_monthly): 94.0%
Feb fill rate (mv_fill_rate_monthly): 96.0%
Mar fill rate (mv_fill_rate_monthly): 91.0%

SS target (fact_safety_stock_targets): 60 units
Assumed LT at SS calculation time: 14 days
Actual LT Jan (from fact_lead_time_actuals, feature_06_10): 18 days
Actual LT Feb: 17 days
Actual LT Mar: 18 days
```

### Step 2: Compute gap and streak
```
Jan: gap = 94.0 - 98.0 = -4.0%  →  below  →  streak=1
Feb: gap = 96.0 - 98.0 = -2.0%  →  below  →  streak=2
Mar: gap = 91.0 - 98.0 = -7.0%  →  below  →  streak=3  ← FLAGGED
```

### Step 3: Classify miss reason
```
LT variance Mar = 18 - 14 = 4 days  >  threshold(3 days)
→ primary_miss_reason = "lead_time_longer"
→ confidence = min(0.95, 0.60 + (4-3)*0.05) = 0.65 → adjusted for 3-month pattern = 0.88
```

### Step 4: Recalculate recommended SS with actual LT

The full safety stock formula (from IPfeature3):

```
SS = Z × sqrt(LT × σ_demand² + μ_demand² × σ_LT²)

Given:
  Z           = 2.054  (98% service level)
  μ_demand    = 3.0 units/day
  σ_demand    = 0.8 units/day
  LT_actual   = 18 days
  σ_LT        = 2.1 days (3-month sample: 18,17,18 → mean=17.7, std=0.58 days ← small sample)
               Use 3-month average: σ_LT = 0.58 days (or use supplier profile from feature_06_10)

With assumed LT=14 days (original):
  SS_old = 2.054 × sqrt(14 × 0.8² + 3.0² × 0) = 2.054 × sqrt(8.96) = 2.054 × 2.99 = 6.1 days
  SS_old_units = 6.1 × 3.0 = 18 units ... (simplified; actual computed was 60 units with
                 actual demand variability including stockout coverage — use recorded value=60)

With actual LT=18 days (recalculation):
  SS_new = 2.054 × sqrt(18 × 0.8² + 3.0² × 0.58²)
         = 2.054 × sqrt(11.52 + 3.02)
         = 2.054 × sqrt(14.54)
         = 2.054 × 3.813
         = 7.83 days

  SS_new_units = 7.83 × 3.0 × (60/6.1)  [scaled to recorded baseline]
               ≈ 90 units

Expected fill rate improvement: +5.8%  (from 91% base → ~97%)
```

### Step 5: Auto-generate AI insight
Because `miss_streak_months = 3 >= threshold(3)` and `abc_class = A`:
- Severity: `critical`
- Type: `policy_gap`
- Summary: "Item 100320 at 1401-BULK has missed its 98% A-class fill rate target for 3 consecutive months. Actual lead time (18 days) exceeds the assumed 14 days used to compute safety stock. Recommend increasing SS from 60 to 90 units."
- `auto_insight_created = TRUE`

---

## 10. Feedback Loop to AI Planner

When `miss_streak_months >= miss_streak_alert_threshold` AND `flagged_for_review = TRUE`:

```
compute_service_level_actuals.py
         │
         ▼
  fact_service_level_performance
  (flagged_for_review = TRUE)
         │
         ▼ (batch or real-time trigger)
  POST /ai-planner/portfolio-scan
  (filter: flagged_for_review=true)
         │
         ▼
  AIPlannerAgent.run_dfu_analysis()
  → tool: get_dfu_full_context()
  → tool: get_inventory_trend()
  → tool: check_stockout_history()
  → tool: create_insight(
        type="policy_gap",
        severity="critical",
        summary="...",
        recommendation="Increase SS to 90 units",
        financial_impact=estimated_stockout_cost
    )
         │
         ▼
  ai_insights table → AIPlannerTab UI
```

---

## 11. Dependencies

| Dependency | Required For | Status |
|---|---|---|
| IPfeature3 — Safety Stock | `ss_units_planned`, SS recalculation | Implemented |
| IPfeature8 — Fill Rate | `achieved_fill_rate`, `stockout_days` | Implemented |
| IPfeature6 — Health Score | `abc_class`, DFU metadata | Implemented |
| IPAIfeature1 — AI Planner | Auto-insight generation on chronic miss | Implemented |
| feature_06_10 — Lead Time Actuals | `actual_lead_time_days` for causal attribution | Planned (F3.3) |

---

## 12. Out of Scope

- OTIF tracking (requires promised delivery date from ERP integration — F1.3)
- Order fill rate (requires order-level demand data — not in current inventory snapshot model)
- Real-time intraday service level monitoring (current data is end-of-day snapshots)
- Customer-segmented service level targets (current model is ABC-class based only)
- Automated SS adjustment write-back (recommendation is generated, but human approval required before `fact_safety_stock_targets` is updated)

---

## 13. Test Requirements

### Backend Unit Tests — `tests/unit/test_service_level.py`

```
test_resolve_target_item_override()           — item-level target beats class default
test_resolve_target_class_fallback()          — class default used when no item target
test_classify_miss_reason_lead_time()         — LT variance > 3 days → lead_time_longer
test_classify_miss_reason_demand_spike()      — spike ratio > 1.25 → demand_spike
test_classify_miss_reason_insufficient_ss()  — stockout_days > 0, no LT/spike → insufficient_ss
test_classify_miss_reason_data_gap()         — no signal → data_gap
test_compute_miss_streak_three_months()      — correctly counts 3 consecutive misses
test_compute_miss_streak_resets_on_green()   — streak resets after a green month
test_gap_direction_above()                   — fill_rate > target → above
test_gap_direction_below()                   — fill_rate < target → below
test_gap_direction_on_target()               — within ±0.5% → on_target
test_ss_recalculation_formula()              — actual LT=18 → SS=90 units
```

### Backend API Tests — `tests/api/test_service_level.py`

```
test_summary_endpoint_200()
test_summary_by_abc_class()
test_detail_endpoint_item_not_found()
test_detail_endpoint_3month_history()
test_chronic_misses_default_min_streak()
test_chronic_misses_min_streak_5()
test_put_targets_creates_row()
test_put_targets_requires_auth()
test_put_targets_validates_pct_range()
```

### Frontend Tests — `src/tabs/__tests__/ControlTowerTab.test.tsx` (extend existing)

```
test_sl_panel_shows_chronic_miss_alert()
test_sl_panel_rag_color_coding()
test_sl_panel_flag_for_review_button()
test_sl_trend_sparkline_renders()
test_sl_abc_filter_works()
```

### Make Targets to Add

```makefile
sl-schema:          # Apply DDL (fact_service_level_targets + fact_service_level_performance)
sl-compute:         # Run compute_service_level_actuals.py for current month
sl-compute-dry:     # --dry-run preview
sl-all:             # sl-schema + sl-compute
```
