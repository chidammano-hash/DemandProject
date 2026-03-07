# Feature F3.1 — Forecast Bias Correction Engine

**Phase:** Evolution to Operations — Phase 3 (Closed-Loop Learning)
**Feature Number:** F3.1
**Status:** Design — Not Implemented
**Depends On:** Feature 15 (Champion Model Selection), Feature 44 (Algorithm Config), IPfeature3 (Safety Stock), F2.2 (Quantile Forecasts)

---

## 1. Problem Statement

### What Fails Today

The Accuracy tab in Demand Studio measures bias using the standard formula:

```
bias = (SUM(forecast) / SUM(actuals)) - 1
```

A positive bias means the system is systematically over-forecasting. A negative bias means it is systematically under-forecasting.

Today, this information is displayed on a dashboard and acted on manually — if a planner notices high bias, they can file an override (F2.3) for individual DFUs. But:

1. There is no automated mechanism to correct the next planning cycle's forecast
2. Bias at the cluster or segment level (e.g., "Cluster 3 consistently over-forecasts by 18%") is not used to adjust individual DFU forecasts
3. The cycle of observe → correct → re-evaluate is entirely manual and inconsistent

**Concrete failure example:**

Cluster 3 (high-volume seasonal items, 312 DFUs) has been over-forecasting for 3 consecutive months:

```
January 2026:   bias = +0.22   (over-forecast by 22%)
February 2026:  bias = +0.17   (over-forecast by 17%)
March 2026:     bias = +0.15   (over-forecast by 15%)
```

Item 100320 raw April 2026 forecast = 450 units. Without correction, the system orders based on 450. The actual demand comes in at 385, leaving 65 units as excess inventory (worth $1,560 at $24/unit). Multiplied across 312 DFUs in Cluster 3, the total excess inventory held in April due to uncorrected bias is estimated at 65 × 312 / DFU-average-ratio ≈ significant capital tie-up.

The system knows the bias is +17% (rolling average). It could have corrected the April forecast to 385 before the order was placed. It did not, because no automated correction mechanism exists.

### What Good Looks Like

```
Observe:  Rolling 3-month bias for Cluster 3 = +0.170
Compute:  correction_factor = 1 / (1 + 0.170) = 0.855
Apply:    corrected_forecast = 450 * 0.855 = 385
Order:    Based on 385 units (not 450)
Evaluate: Next month: actual = 380, corrected forecast = 385, error = 1.3%
           vs. uncorrected: actual = 380, raw forecast = 450, error = 18.4%
```

The feedback loop closes: bias is observed, a correction is applied, the correction's impact is measured in the next cycle.

---

## 2. Bias Segments

Bias is computed and tracked at four hierarchical levels. Corrections can be applied at any level, with DFU-level taking precedence over cluster-level.

```
Level 1 — DFU (item_no + loc)
    Highest precision. Computed only for DFUs with ≥ 3 months of history.
    Correction applied if |bias| > threshold (default: 0.10).

Level 2 — Cluster (ml_cluster)
    Applied to DFUs within a cluster when DFU-level correction is not available.
    Represents systematic model error in how a cluster's demand pattern is learned.

Level 3 — ABC Class × Seasonality Profile
    Cross-segment: e.g., "A-class items with strong seasonality".
    Useful when model struggles on a pattern type (e.g., always over-shoots peak months
    for seasonal items with lumpy demand).

Level 4 — Product Lifecycle Stage
    (launch, growth, mature, decline) — requires lifecycle stage flag in dim_dfu.
    New products consistently under-forecasted (model has no history).
    Declining products consistently over-forecasted (model hasn't seen the drop yet).
```

### Precedence Rules

When multiple bias corrections exist for a DFU, the most specific level takes precedence:

```
DFU-level correction > Cluster-level > ABC×Seasonality > Lifecycle
```

If DFU-level bias is computed (≥ 3 months data) AND cluster-level correction exists, the DFU-level correction is used. The cluster correction is held as a fallback.

---

## 3. Rolling Bias Computation

### Exponential Decay Weighting

The system uses a 3-month rolling window with exponential decay weighting. More recent months carry more weight because they reflect current model behavior more accurately.

**Default weights:**

```
Most recent month:     weight = 0.50
Prior month:           weight = 0.30
Oldest month in window: weight = 0.20
Total:                             1.00
```

**Formula:**

```
rolling_bias = w1 * bias_m1 + w2 * bias_m2 + w3 * bias_m3

where:
  bias_mN = (SUM(forecast_mN) / SUM(actual_mN)) - 1  [for the segment]
  w1 = 0.50 (most recent month)
  w2 = 0.30
  w3 = 0.20
```

**Worked Example — Cluster 3:**

```
January 2026:   bias = +0.220
February 2026:  bias = +0.170
March 2026:     bias = +0.150

rolling_bias = 0.50 * 0.150 + 0.30 * 0.170 + 0.20 * 0.220
             = 0.075 + 0.051 + 0.044
             = 0.170
```

### Correction Factor Derivation

```
correction_factor = 1 / (1 + rolling_bias)

Cluster 3 example:
  correction_factor = 1 / (1 + 0.170) = 1 / 1.170 = 0.855

Item 100320 April 2026:
  raw_forecast     = 450.00
  corrected_qty    = 450.00 * 0.855 = 384.75 → rounded to 384.75
  correction_pct   = (384.75 - 450.00) / 450.00 * 100 = -14.5%
```

### Guard Rails

```
correction_factor CLIPPED TO [0.70, 1.30]
```

This prevents extreme corrections. Maximum downward correction = 30% (correction_factor = 0.70). Maximum upward correction = 30% (correction_factor = 1.30).

If the computed correction_factor falls outside this range, the DFU is flagged for human review:

```
flagged_for_review = TRUE when |1 - correction_factor_raw| > 0.20
```

Any DFU requiring correction > 20% is flagged regardless of whether the guard rail clips it. This captures situations where the model is fundamentally broken for a specific DFU or segment and human intervention is warranted.

---

## 4. Data Model

### 4.1 New Table: `fact_bias_corrections`

**Grain:** `item_no + loc + plan_month + segment_type`

One row per DFU per plan month per correction segment (DFU-level and cluster-level corrections both stored; DFU takes precedence at application time).

```sql
CREATE TABLE fact_bias_corrections (
    id                      BIGSERIAL       PRIMARY KEY,
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    plan_month              DATE            NOT NULL,       -- Month being corrected (future month)
    segment_type            VARCHAR(30)     NOT NULL,       -- 'dfu', 'cluster', 'abc_seasonality', 'lifecycle'
    segment_value           VARCHAR(100)    NOT NULL,       -- e.g., '3' (cluster_id), 'A_strong_seasonal'
    rolling_bias_3m         NUMERIC(8,4)    NOT NULL,       -- Weighted 3-month rolling bias
    rolling_wape_3m         NUMERIC(8,4),                   -- 3-month rolling WAPE for context
    bias_month1             NUMERIC(8,4),                   -- Most recent month bias (raw)
    bias_month2             NUMERIC(8,4),                   -- Prior month bias
    bias_month3             NUMERIC(8,4),                   -- Oldest month in window
    wape_month1             NUMERIC(8,4),
    wape_month2             NUMERIC(8,4),
    wape_month3             NUMERIC(8,4),
    correction_factor_raw   NUMERIC(6,4)    NOT NULL,       -- 1/(1+rolling_bias) before guard rails
    correction_factor       NUMERIC(6,4)    NOT NULL,       -- After guard rail clipping [0.70, 1.30]
    correction_was_clipped  BOOLEAN         NOT NULL DEFAULT FALSE,
    raw_forecast_qty        NUMERIC(12,2),                  -- Point forecast before correction
    corrected_forecast_qty  NUMERIC(12,2),                  -- After correction applied
    correction_pct          NUMERIC(6,2),                   -- (corrected - raw) / raw * 100
    flagged_for_review      BOOLEAN         NOT NULL DEFAULT FALSE,
    correction_applied      BOOLEAN         NOT NULL DEFAULT FALSE,  -- TRUE once merged into consensus/demand plan
    applied_at              TIMESTAMPTZ,
    applied_to_version      VARCHAR(50),                    -- Which plan version consumed this correction
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    months_of_data          INTEGER,                        -- How many months of actuals were available
    CONSTRAINT uq_bias_correction UNIQUE (item_no, loc, plan_month, segment_type)
);

CREATE INDEX idx_bias_correction_item_loc_month
    ON fact_bias_corrections (item_no, loc, plan_month);

CREATE INDEX idx_bias_correction_segment
    ON fact_bias_corrections (segment_type, segment_value, plan_month);

CREATE INDEX idx_bias_correction_flagged
    ON fact_bias_corrections (flagged_for_review, plan_month)
    WHERE flagged_for_review = TRUE;

CREATE INDEX idx_bias_correction_applied
    ON fact_bias_corrections (correction_applied, plan_month);
```

### 4.2 New Table: `fact_bias_correction_history`

Tracks correction factors over time for trend analysis and model health monitoring. One row per segment per month computed.

```sql
CREATE TABLE fact_bias_correction_history (
    id                      BIGSERIAL       PRIMARY KEY,
    segment_type            VARCHAR(30)     NOT NULL,
    segment_value           VARCHAR(100)    NOT NULL,
    computation_month       DATE            NOT NULL,       -- When was this correction computed (plan run date)
    rolling_bias_3m         NUMERIC(8,4)    NOT NULL,
    correction_factor       NUMERIC(6,4)    NOT NULL,
    dfu_count_in_segment    INTEGER,
    avg_raw_wape            NUMERIC(6,4),
    avg_corrected_wape      NUMERIC(6,4),                  -- NULL until next month actuals available
    correction_improved_accuracy BOOLEAN,                   -- Set retroactively when actuals arrive
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_bias_history_segment_month
    ON fact_bias_correction_history (segment_type, segment_value, computation_month DESC);
```

### 4.3 New Columns on `fact_demand_plan` (from F2.2)

The demand plan table is extended with correction metadata:

```sql
ALTER TABLE fact_demand_plan
    ADD COLUMN IF NOT EXISTS bias_correction_factor    NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS bias_correction_segment   VARCHAR(30),
    ADD COLUMN IF NOT EXISTS raw_forecast_qty          NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS bias_corrected            BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS flagged_for_review        BOOLEAN NOT NULL DEFAULT FALSE;
```

---

## 5. Python Scripts

### 5.1 `scripts/compute_bias_corrections.py`

```python
"""
Compute rolling 3-month forecast bias by segment and derive correction factors.
Writes results to fact_bias_corrections.
Optionally applies corrections to fact_demand_plan (fact_demand_plan.forecast_qty updated).

Usage:
    uv run scripts/compute_bias_corrections.py \
        --plan-version 2026-04-01_production \
        --apply-to-plan \
        --dry-run

Config: config/bias_correction_config.yaml
Output:
    - fact_bias_corrections (correction factors per DFU per month)
    - fact_bias_correction_history (segment-level trend tracking)
    - fact_demand_plan (updated if --apply-to-plan)
"""

import yaml
import pandas as pd
import numpy as np
import psycopg
from datetime import date
from dateutil.relativedelta import relativedelta
from common.db import get_db_params

# Weights for 3-month rolling bias (most recent first)
DEFAULT_WEIGHTS = [0.50, 0.30, 0.20]

# Guard rail: correction factor clipped to [min, max]
CORRECTION_MIN = 0.70
CORRECTION_MAX = 1.30

# Flag for review if raw correction > this threshold (before clipping)
REVIEW_THRESHOLD = 0.20


def load_historical_bias(
    reference_months: list[date],
    segment_type: str,
    conn: psycopg.Connection,
) -> pd.DataFrame:
    """
    Load historical bias per segment for the specified months.
    Bias is computed from backtest actuals vs predictions in backtest_lag_archive.

    Args:
        reference_months: List of 3 past month-start dates (most recent first)
        segment_type: 'dfu', 'cluster', 'abc_seasonality', 'lifecycle'
        conn: DB connection

    Returns:
        DataFrame: segment_value, month, forecast_sum, actual_sum, bias
    """
    if segment_type == "dfu":
        groupby_cols = "f.dmdunit AS item_no, f.loc"
        join_clause = ""
        segment_col = "CONCAT(f.dmdunit, '@', f.loc)"
    elif segment_type == "cluster":
        groupby_cols = "d.ml_cluster::TEXT AS segment_value"
        join_clause = "JOIN dim_dfu d ON d.dmdunit = f.dmdunit AND d.loc = f.loc"
        segment_col = "d.ml_cluster::TEXT"
    elif segment_type == "abc_seasonality":
        groupby_cols = "CONCAT(d.abc_class, '_', d.seasonality_profile) AS segment_value"
        join_clause = "JOIN dim_dfu d ON d.dmdunit = f.dmdunit AND d.loc = f.loc"
        segment_col = "CONCAT(d.abc_class, '_', d.seasonality_profile)"
    else:
        raise ValueError(f"Unknown segment_type: {segment_type}")

    sql = f"""
        SELECT
            {segment_col} AS segment_value,
            f.startdate AS plan_month,
            SUM(f.basefcst_pref)   AS forecast_sum,
            SUM(f.qty)             AS actual_sum,
            CASE
                WHEN SUM(f.qty) = 0 THEN NULL
                ELSE (SUM(f.basefcst_pref) / NULLIF(SUM(f.qty), 0)) - 1
            END AS bias
        FROM fact_external_forecast_monthly f
        {join_clause}
        WHERE f.startdate = ANY(%s)
          AND f.lag = 0                -- Use lag-0 (execution month) for bias measurement
          AND f.model_id = 'champion'  -- Use champion model as baseline
        GROUP BY {segment_col}, f.startdate
        HAVING SUM(f.qty) > 0
    """
    with conn.cursor() as cur:
        cur.execute(sql, (reference_months,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["segment_value", "plan_month", "forecast_sum", "actual_sum", "bias"])


def compute_rolling_bias(
    bias_by_month: pd.Series,
    weights: list[float] = DEFAULT_WEIGHTS,
) -> float:
    """
    Compute exponentially weighted rolling bias from a Series of 3 bias values
    ordered most-recent-first.

    Args:
        bias_by_month: pd.Series of length ≤ 3, most recent bias first
        weights: Decay weights summing to 1.0 (default [0.50, 0.30, 0.20])

    Returns:
        Weighted rolling bias (float)

    Example:
        bias_by_month = [0.150, 0.170, 0.220]  (Mar, Feb, Jan)
        weights       = [0.50,  0.30,  0.20]
        result        = 0.50*0.150 + 0.30*0.170 + 0.20*0.220 = 0.170
    """
    values = bias_by_month.values[:len(weights)]
    active_weights = np.array(weights[:len(values)])
    active_weights = active_weights / active_weights.sum()  # Normalize if fewer than 3 months
    return float(np.dot(values, active_weights))


def derive_correction_factor(
    rolling_bias: float,
    clip_min: float = CORRECTION_MIN,
    clip_max: float = CORRECTION_MAX,
    review_threshold: float = REVIEW_THRESHOLD,
) -> tuple[float, float, bool, bool]:
    """
    Derive correction factor from rolling bias with guard rail clipping.

    Args:
        rolling_bias: Weighted rolling bias (e.g., +0.170 means 17% over-forecast)
        clip_min: Minimum allowed correction factor (default 0.70)
        clip_max: Maximum allowed correction factor (default 1.30)
        review_threshold: Absolute correction > this → flagged_for_review

    Returns:
        Tuple of:
            correction_factor_raw (before clipping)
            correction_factor (after clipping)
            correction_was_clipped (bool)
            flagged_for_review (bool)

    Example:
        rolling_bias = +0.170
        correction_factor_raw = 1 / (1 + 0.170) = 0.855
        correction_factor     = max(0.70, min(1.30, 0.855)) = 0.855
        correction_was_clipped = False
        flagged_for_review     = False (|1 - 0.855| = 0.145 < 0.20)

        rolling_bias = +0.45
        correction_factor_raw = 1 / (1 + 0.45) = 0.690
        correction_factor     = max(0.70, 0.690) = 0.70  [CLIPPED]
        correction_was_clipped = True
        flagged_for_review     = True (|1 - 0.690| = 0.31 > 0.20)
    """
    raw = 1.0 / (1.0 + rolling_bias)
    clipped = max(clip_min, min(clip_max, raw))
    was_clipped = abs(raw - clipped) > 1e-6
    flagged = abs(1.0 - raw) > review_threshold
    return raw, clipped, was_clipped, flagged


def apply_correction_to_forecast(
    raw_qty: float,
    correction_factor: float,
) -> float:
    """
    Apply the correction factor to a raw forecast quantity.

    Args:
        raw_qty: Point forecast (e.g., 450.00)
        correction_factor: Derived correction factor (e.g., 0.855)

    Returns:
        Corrected forecast quantity (e.g., 384.75)

    Note: Result is floored at 0. Negative forecasts are not allowed.
    """
    return round(max(0.0, raw_qty * correction_factor), 2)


def write_bias_corrections(
    corrections: list[dict],
    conn: psycopg.Connection,
    dry_run: bool = False,
) -> int:
    """
    Bulk upsert bias correction rows into fact_bias_corrections.

    Args:
        corrections: List of correction dicts (see uq_bias_correction constraint)
        conn: DB connection
        dry_run: If True, print rows but do not write

    Returns:
        Number of rows written
    """
    if dry_run:
        for c in corrections[:5]:
            print(f"[DRY RUN] {c['item_no']}@{c['loc']} {c['plan_month']}: "
                  f"bias={c['rolling_bias_3m']:.4f} → factor={c['correction_factor']:.4f} "
                  f"raw={c['raw_forecast_qty']} → corrected={c['corrected_forecast_qty']}")
        return 0

    sql = """
        INSERT INTO fact_bias_corrections (
            item_no, loc, plan_month, segment_type, segment_value,
            rolling_bias_3m, rolling_wape_3m,
            bias_month1, bias_month2, bias_month3,
            correction_factor_raw, correction_factor,
            correction_was_clipped, raw_forecast_qty, corrected_forecast_qty,
            correction_pct, flagged_for_review, months_of_data, computed_at
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s, NOW()
        )
        ON CONFLICT (item_no, loc, plan_month, segment_type)
        DO UPDATE SET
            rolling_bias_3m       = EXCLUDED.rolling_bias_3m,
            correction_factor_raw = EXCLUDED.correction_factor_raw,
            correction_factor     = EXCLUDED.correction_factor,
            correction_was_clipped = EXCLUDED.correction_was_clipped,
            corrected_forecast_qty = EXCLUDED.corrected_forecast_qty,
            correction_pct        = EXCLUDED.correction_pct,
            flagged_for_review    = EXCLUDED.flagged_for_review,
            computed_at           = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, [
            (c["item_no"], c["loc"], c["plan_month"], c["segment_type"], c["segment_value"],
             c["rolling_bias_3m"], c.get("rolling_wape_3m"),
             c.get("bias_month1"), c.get("bias_month2"), c.get("bias_month3"),
             c["correction_factor_raw"], c["correction_factor"],
             c["correction_was_clipped"], c.get("raw_forecast_qty"), c.get("corrected_forecast_qty"),
             c.get("correction_pct"), c["flagged_for_review"], c.get("months_of_data"))
            for c in corrections
        ])
    conn.commit()
    return len(corrections)


def apply_corrections_to_demand_plan(
    plan_version: str,
    conn: psycopg.Connection,
) -> int:
    """
    Apply stored correction factors from fact_bias_corrections to fact_demand_plan.

    Precedence: DFU-level correction overrides cluster-level overrides.
    Updates fact_demand_plan.forecast_qty with corrected values.
    Sets bias_corrected=TRUE and records the correction_factor and segment used.

    Args:
        plan_version: Target plan version to update
        conn: DB connection

    Returns:
        Number of rows updated in fact_demand_plan
    """
    sql = """
        WITH ranked_corrections AS (
            SELECT
                bc.item_no, bc.loc, bc.plan_month,
                bc.correction_factor,
                bc.segment_type,
                bc.flagged_for_review,
                ROW_NUMBER() OVER (
                    PARTITION BY bc.item_no, bc.loc, bc.plan_month
                    ORDER BY
                        CASE bc.segment_type
                            WHEN 'dfu'             THEN 1
                            WHEN 'cluster'         THEN 2
                            WHEN 'abc_seasonality' THEN 3
                            WHEN 'lifecycle'       THEN 4
                        END
                ) AS rn
            FROM fact_bias_corrections bc
            WHERE bc.plan_month >= CURRENT_DATE
        )
        UPDATE fact_demand_plan dp
        SET
            raw_forecast_qty       = dp.forecast_qty,
            forecast_qty           = ROUND(dp.forecast_qty * rc.correction_factor, 2),
            bias_correction_factor = rc.correction_factor,
            bias_correction_segment = rc.segment_type,
            bias_corrected         = TRUE,
            flagged_for_review     = rc.flagged_for_review
        FROM ranked_corrections rc
        WHERE dp.item_no    = rc.item_no
          AND dp.loc        = rc.loc
          AND dp.plan_month = rc.plan_month
          AND dp.plan_version = %s
          AND rc.rn = 1
          AND dp.quantile = 0.50   -- Apply correction to P50; P10/P90 scaled proportionally below
    """
    with conn.cursor() as cur:
        cur.execute(sql, (plan_version,))
        updated = cur.rowcount
    conn.commit()
    return updated


def run(
    plan_version: str,
    plan_run_date: date,
    apply_to_plan: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Main entry point. Compute rolling bias by segment, derive corrections,
    write to fact_bias_corrections, optionally apply to demand plan.

    Returns summary dict: segments_computed, dfus_corrected, dfus_flagged
    """
    cfg = yaml.safe_load(open("config/bias_correction_config.yaml"))
    reference_months = [
        (plan_run_date - relativedelta(months=i)).replace(day=1)
        for i in range(1, 4)  # [last month, 2 months ago, 3 months ago]
    ]
    weights = cfg["rolling_bias"]["decay_weights"]

    with psycopg.connect(**get_db_params()) as conn:
        corrections = []

        for seg_type in cfg["segments"]["enabled"]:
            bias_df = load_historical_bias(reference_months, seg_type, conn)
            # ... pivot, compute rolling bias, derive correction factors
            # ... build correction rows and append to corrections list
            pass

        total_written = write_bias_corrections(corrections, conn, dry_run=dry_run)

        if apply_to_plan and not dry_run:
            updated = apply_corrections_to_demand_plan(plan_version, conn)
        else:
            updated = 0

    flagged = sum(1 for c in corrections if c["flagged_for_review"])
    return {
        "segments_computed": len(corrections),
        "dfus_corrected": updated,
        "dfus_flagged": flagged,
        "dry_run": dry_run,
    }
```

### 5.2 `config/bias_correction_config.yaml`

```yaml
bias_correction:
  rolling_bias:
    window_months: 3
    decay_weights: [0.50, 0.30, 0.20]    # Most recent month first
    min_months_required: 2               # Need at least 2 months of actuals

  correction_guard_rails:
    min_correction_factor: 0.70          # Never correct down more than 30%
    max_correction_factor: 1.30          # Never correct up more than 30%
    review_threshold: 0.20               # Flag DFUs needing > 20% correction

  segments:
    enabled:
      - dfu                              # Per-DFU bias (highest precision)
      - cluster                          # Per-cluster bias (fallback)
      - abc_seasonality                  # ABC class × seasonality profile
    dfu_min_history_months: 3            # Skip DFU-level correction if fewer months

  bias_measurement:
    model_id: champion                   # Which model's forecasts to measure bias against
    lag: 0                               # Use lag-0 (execution month) for bias
    exclude_zero_actuals: true           # Exclude months with 0 actual demand

  application:
    apply_to_quantiles: [0.10, 0.50, 0.90]   # Scale all quantiles by same factor
    apply_to_consensus: true             # Also apply to fact_consensus_plan

  flagging:
    notify_on_flag: true
    flag_threshold_percent: 20           # Same as review_threshold (% form for config clarity)
    max_flagged_pct_before_alert: 0.10   # Alert if > 10% of DFUs are flagged in a run
```

---

## 6. Feedback Loop

The bias correction engine is not a one-time operation — it improves iteratively.

```
CYCLE N (April 2026 plan run):
  ┌─────────────────────────────────────────────────────┐
  │ 1. Load Jan/Feb/Mar 2026 actuals vs champion preds  │
  │ 2. Compute rolling bias per cluster/DFU             │
  │ 3. Derive correction factors                        │
  │ 4. Apply corrections to April 2026 demand plan      │
  │ 5. Write fact_bias_correction_history snapshot      │
  └─────────────────────────────────────────────────────┘
              ↓ (April 2026 passes, actuals arrive in May)
CYCLE N+1 (May 2026 plan run):
  ┌─────────────────────────────────────────────────────┐
  │ 1. Load Feb/Mar/Apr 2026 actuals vs champion preds  │
  │    *** April 2026 used corrected forecast as plan   │
  │    *** Did correction improve accuracy?             │
  │ 2. Compute new rolling bias (now including Apr)     │
  │ 3. In fact_bias_correction_history: retroactively   │
  │    set correction_improved_accuracy for Apr entry   │
  │    by comparing avg_raw_wape vs avg_corrected_wape  │
  └─────────────────────────────────────────────────────┘
```

The `fact_bias_correction_history.avg_corrected_wape` field is populated one month after the correction was applied, when actuals for the corrected month become available. This enables trend charts showing "correction effectiveness over time" in the UI.

---

## 7. API Endpoints

### 7.1 `GET /forecast/bias-corrections`

Get bias corrections for a specific item/location.

**Parameters:** `item_no`, `loc`, `plan_version`, `segment_type`

**Response:**

```json
{
  "item_no": "100320",
  "loc": "1401-BULK",
  "corrections": [
    {
      "plan_month": "2026-04-01",
      "segment_type": "cluster",
      "segment_value": "3",
      "rolling_bias_3m": 0.1700,
      "rolling_wape_3m": 0.1842,
      "bias_month1": 0.1500,
      "bias_month2": 0.1700,
      "bias_month3": 0.2200,
      "correction_factor_raw": 0.8547,
      "correction_factor": 0.8547,
      "correction_was_clipped": false,
      "raw_forecast_qty": 450.00,
      "corrected_forecast_qty": 384.75,
      "correction_pct": -14.50,
      "flagged_for_review": false,
      "correction_applied": true,
      "applied_to_version": "2026-04-01_production"
    }
  ]
}
```

### 7.2 `GET /forecast/bias-corrections/summary`

Portfolio-level bias correction summary for a plan version.

**Parameters:** `plan_version`, `segment_type` (optional)

**Response:**

```json
{
  "plan_version": "2026-04-01_production",
  "computation_date": "2026-04-01",
  "total_dfus": 4823,
  "dfus_corrected": 3241,
  "dfus_uncorrected": 1582,
  "dfus_flagged_for_review": 47,
  "pct_dfus_corrected": 67.2,
  "pct_dfus_flagged": 0.97,
  "avg_correction_factor": 0.921,
  "avg_rolling_bias": 0.084,
  "by_segment": {
    "dfu": { "count": 2140, "avg_correction_factor": 0.934, "avg_bias": 0.071 },
    "cluster": { "count": 1101, "avg_correction_factor": 0.899, "avg_bias": 0.112 }
  },
  "correction_direction": {
    "downward_corrections": 2890,
    "upward_corrections": 351,
    "no_correction": 1582
  },
  "alert": null
}
```

### 7.3 `GET /forecast/bias-corrections/flagged`

Items requiring human review (correction > 20% threshold).

**Parameters:** `plan_version`, `page`, `page_size`

**Response:**

```json
{
  "total_flagged": 47,
  "page": 1,
  "items": [
    {
      "item_no": "307821",
      "loc": "2203-STD",
      "plan_month": "2026-04-01",
      "segment_type": "dfu",
      "rolling_bias_3m": 0.4200,
      "correction_factor_raw": 0.7042,
      "correction_factor": 0.7100,
      "correction_was_clipped": true,
      "raw_forecast_qty": 920.00,
      "corrected_forecast_qty": 653.20,
      "correction_pct": -29.0,
      "reason_hint": "Correction clipped at guard rail minimum (0.70). Manual review required."
    }
  ]
}
```

### 7.4 `GET /forecast/bias-corrections/history`

Segment-level correction history over time (for trend chart).

**Parameters:** `segment_type`, `segment_value`, `months_back` (default: 12)

**Response:**

```json
{
  "segment_type": "cluster",
  "segment_value": "3",
  "history": [
    {
      "computation_month": "2026-04-01",
      "rolling_bias_3m": 0.1700,
      "correction_factor": 0.8547,
      "dfu_count_in_segment": 312,
      "avg_raw_wape": 0.1842,
      "avg_corrected_wape": null,
      "correction_improved_accuracy": null
    },
    {
      "computation_month": "2026-03-01",
      "rolling_bias_3m": 0.1650,
      "correction_factor": 0.8584,
      "dfu_count_in_segment": 308,
      "avg_raw_wape": 0.1798,
      "avg_corrected_wape": 0.1021,
      "correction_improved_accuracy": true
    }
  ]
}
```

---

## 8. Frontend Components

### 8.1 Bias Correction Report Panel

Located in: `frontend/src/tabs/accuracy/BiasCorrectionsPanel.tsx`

```
┌─────────────────────────────────────────────────────────────────────┐
│  BIAS CORRECTION ENGINE                  [Version: 2026-04-01 ▼]    │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  3,241   │  │  0.921   │  │   47     │  │  +8.4%   │           │
│  │ DFUs     │  │ Avg Corr │  │ Flagged  │  │ Avg Bias │           │
│  │ Corrected│  │ Factor   │  │ Review   │  │ (rolling)│           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│                                                                     │
│  TOP 10 MOST BIASED DFUs                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Item      Loc       Cluster  Bias   Correction  Flagged      │  │
│  │ 307821   2203-STD     7      +42%    0.71 [↓]   ⚠ YES       │  │
│  │ 100320   1401-BULK    3      +17%    0.855       No          │  │
│  │ 204771   2203-STD     3      +22%    0.820       No          │  │
│  │ 415302   1102-STD     5      -19%    1.235       No          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  CLUSTER BIAS HEATMAP                                              │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │         Jan    Feb    Mar    Apr (corrected)                 │  │
│  │ C1      +4%    +3%    +2%   → 0%                           │  │
│  │ C2      -8%    -6%    -5%   → 0%                           │  │
│  │ C3     +22%   +17%   +15%   → -14.5% correction applied   │  │
│  │ C4      +1%    +2%    +1%   → No correction (< threshold)  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  CORRECTION EFFECTIVENESS (Cluster 3 — last 6 months)             │
│  [Line chart: avg_raw_wape vs avg_corrected_wape over time]        │
│  Before correction: 18.4%  After correction: 10.2%  (-44% error)  │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Bias Corrected Badge in Demand Plan Panel

In `DemandPlanPanel.tsx`, rows where `bias_corrected=TRUE` display:

```
Apr '26  Stat P50: 450 → Corrected: 384.75  [-14.5%] [📊 Bias Corrected: Cluster 3, -17% rolling]
```

The correction chip is teal/cyan colored to distinguish from orange planner overrides.

If a DFU is `flagged_for_review=TRUE`:

```
Apr '26  Stat P50: 920 → Corrected: 653.20  [-29%]  [⚠ Review Required: Clipped at guard rail]
```

Yellow warning chip. Clicking the chip opens a detail popover showing the 3-month bias history.

---

## 9. Worked Example: End-to-End

### Cluster 3 — April 2026 Plan Run

**Step 1: Load Historical Bias (load_historical_bias, segment_type='cluster')**

Reference months: March 2026, February 2026, January 2026

```
Query fact_external_forecast_monthly WHERE model_id='champion' AND lag=0:

Cluster 3, March 2026:    forecast_sum=47,250   actual_sum=41,085   bias=(47250/41085)-1 = +0.1500
Cluster 3, February 2026: forecast_sum=44,820   actual_sum=38,308   bias=(44820/38308)-1 = +0.1700
Cluster 3, January 2026:  forecast_sum=42,120   actual_sum=34,525   bias=(42120/34525)-1 = +0.2200
```

**Step 2: Compute Rolling Bias (compute_rolling_bias)**

```
rolling_bias = 0.50 * 0.1500 + 0.30 * 0.1700 + 0.20 * 0.2200
             = 0.0750        + 0.0510        + 0.0440
             = 0.1700
```

**Step 3: Derive Correction Factor (derive_correction_factor)**

```
correction_factor_raw = 1 / (1 + 0.1700) = 1 / 1.1700 = 0.8547

Clipping: max(0.70, min(1.30, 0.8547)) = 0.8547  [not clipped]
correction_was_clipped = False

Review check: |1 - 0.8547| = 0.1453 < 0.20  → flagged_for_review = False
```

**Step 4: Apply to Item 100320 Forecast**

```
raw_forecast_qty    = 450.00  (from fact_demand_plan, P50, April 2026)
corrected_qty       = 450.00 * 0.8547 = 384.75
correction_pct      = (384.75 - 450.00) / 450.00 * 100 = -14.5%
```

**Step 5: fact_bias_corrections Row Written**

```
item_no              = '100320'
loc                  = '1401-BULK'
plan_month           = 2026-04-01
segment_type         = 'cluster'
segment_value        = '3'
rolling_bias_3m      = 0.1700
bias_month1          = 0.1500   (March)
bias_month2          = 0.1700   (February)
bias_month3          = 0.2200   (January)
correction_factor    = 0.8547
correction_was_clipped = FALSE
raw_forecast_qty     = 450.00
corrected_forecast_qty = 384.75
correction_pct       = -14.50
flagged_for_review   = FALSE
months_of_data       = 3
```

**Step 6: fact_demand_plan Updated**

```sql
UPDATE fact_demand_plan
SET
    raw_forecast_qty       = 450.00,
    forecast_qty           = 384.75,
    bias_correction_factor = 0.8547,
    bias_correction_segment = 'cluster',
    bias_corrected         = TRUE
WHERE item_no = '100320' AND loc = '1401-BULK'
  AND plan_month = '2026-04-01'
  AND plan_version = '2026-04-01_production'
  AND quantile = 0.50;
```

**Step 7: Impact Measurement (May 2026 plan run)**

April 2026 actuals arrive. `avg_corrected_wape` for Cluster 3 is computed:

```
Cluster 3, April 2026 actuals:
  Forecast (corrected P50, avg across Cluster 3 DFUs): 385.2
  Actual demand: 381.4
  Corrected WAPE = |385.2 - 381.4| / 381.4 = 1.0%

vs. uncorrected WAPE:
  Forecast (raw P50, avg): 450.0
  Actual demand: 381.4
  Raw WAPE = |450.0 - 381.4| / 381.4 = 18.0%

correction_improved_accuracy = TRUE  (1.0% < 18.0%)
```

`fact_bias_correction_history` updated: `avg_corrected_wape=0.010`, `correction_improved_accuracy=TRUE`.

---

## 10. Dependencies

| Dependency | Type | Status |
|---|---|---|
| Feature 15 — Champion Model | Hard (bias measured against champion forecasts) | Implemented |
| Feature 44 — Backtest Framework | Hard (actuals sourced from backtest_lag_archive) | Implemented |
| F2.2 — Quantile Forecasts | Hard (target of correction application) | Design |
| `dim_dfu.ml_cluster` | Hard (cluster-level segmentation) | Implemented |
| `dim_dfu.abc_class` | Hard (ABC×seasonality segment) | Implemented |
| `dim_dfu.seasonality_profile` | Hard (ABC×seasonality segment) | Implemented |
| IPfeature3 — Safety Stock | Soft (SS re-computation recommended post-correction) | Implemented |

---

## 11. Out of Scope

- Online/real-time bias correction (corrections are computed in batch, not per-request)
- Separate correction models per product lifecycle stage (lifecycle flag not in dim_dfu yet)
- Automatic retraining of ML models based on bias patterns (this corrects outputs; retraining is separate)
- Bias-adjusted prediction intervals (P10/P90 shrinkage based on bias history)
- Supplier-level bias correction (e.g., supplier systematically ships 5% short)
- Customer-level demand correction (requires customer-level actuals from CRM)

---

## 12. Makefile Targets

```makefile
bias-schema:
    @echo "Applying bias correction schema..."
    uv run python -c "..." sql/043_create_bias_corrections.sql

bias-compute:
    uv run scripts/compute_bias_corrections.py \
        --plan-version $(VERSION)

bias-compute-dry:
    uv run scripts/compute_bias_corrections.py \
        --plan-version $(VERSION) \
        --dry-run

bias-apply:
    uv run scripts/compute_bias_corrections.py \
        --plan-version $(VERSION) \
        --apply-to-plan

bias-all:
    make bias-schema && make bias-compute && make bias-apply
```

---

## 13. Test Requirements

### Backend Unit Tests (`tests/unit/test_bias_corrections.py`)

- `test_compute_rolling_bias_three_months`: 3-month weighted sum correct (0.170)
- `test_compute_rolling_bias_two_months`: Normalizes weights when only 2 months available
- `test_compute_rolling_bias_one_month`: Single month → that month's bias returned
- `test_derive_correction_factor_positive_bias`: +0.170 → factor=0.8547
- `test_derive_correction_factor_negative_bias`: -0.120 → factor=1.1364
- `test_derive_correction_factor_clipping_high_bias`: bias=+0.50 → raw=0.667, clipped=0.70, was_clipped=True
- `test_derive_correction_factor_clipping_low_bias`: bias=-0.40 → raw=1.667, clipped=1.30, was_clipped=True
- `test_derive_correction_factor_review_flag_triggered`: |1-raw|>0.20 → flagged_for_review=True
- `test_derive_correction_factor_review_flag_not_triggered`: |1-raw|<0.20 → flagged_for_review=False
- `test_apply_correction_to_forecast_standard`: 450 * 0.8547 = 384.75
- `test_apply_correction_to_forecast_floor_at_zero`: Negative raw_qty → 0.0
- `test_precedence_dfu_over_cluster`: DFU correction takes precedence over cluster-level
- `test_precedence_cluster_over_abc_seasonality`: Cluster beats ABC×seasonality when DFU not available

### Backend API Tests (`tests/api/test_bias_corrections.py`)

- `test_get_bias_corrections_item_loc`: Returns correction rows with correct fields
- `test_get_bias_corrections_summary_counts`: Total DFUs, flagged, corrected counts correct
- `test_get_bias_corrections_flagged_only`: Returns only flagged rows
- `test_get_bias_corrections_history_monthly`: Returns 12 monthly history rows
- `test_get_bias_corrections_unknown_version_404`: Unknown plan_version → 404
- `test_get_bias_corrections_summary_no_corrections`: Returns zeroes when no corrections computed

### Frontend Tests (`frontend/src/tabs/__tests__/BiasCorrectionsPanel.test.tsx`)

- KPI cards render: DFUs corrected, avg correction factor, flagged count
- Cluster bias heatmap renders with mock data
- Correction effectiveness line chart renders (ECharts mock)
- Flagged item row shows warning chip with Review Required text
- Demand Plan Panel badge shows "Bias Corrected" for corrected DFUs
- Clicking bias chip opens popover with 3-month history
