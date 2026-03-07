# Feature F2.2 — Multi-Horizon Demand Plan (Quantile Forecasts)

**Phase:** Evolution to Operations — Phase 2 (Demand Planning)
**Feature Number:** F2.2
**Status:** Design — Not Implemented
**Depends On:** F1.1 (Production Forecast Infrastructure), existing backtest framework (Features 8–13, 44)

---

## 1. Problem Statement

### What Fails Today

The current ML backtest pipeline (LightGBM, CatBoost, XGBoost) produces a **single point estimate** per DFU per month — the expected value of future demand. This works for measuring historical accuracy but is structurally inadequate for inventory planning decisions.

**Concrete failure example:**

Item 100320, Location 1401-BULK. The model predicts 450 units for April 2026.

The current safety stock engine (`scripts/compute_safety_stock.py`) uses this 450 as the demand input to the Z-score formula:

```
SS = Z * σ_demand * sqrt(lead_time_days / 30)
```

But σ_demand is computed from historical variability alone. It ignores the forecast uncertainty — the fact that the model itself may be wrong by ±130 units at the 90th percentile.

**The math that fails:**

If demand is truly Normally distributed with μ=450 and σ=80 (from history), the planner sizes SS for a 95% service level:

```
SS = 1.645 * 80 * sqrt(2) = 186 units
```

But the forecast itself carries uncertainty. If the model's prediction interval is [320, 580] at P10/P90, then the effective σ of the demand plan is much larger:

```
σ_forecast = (P90 - P10) / (2 * 1.28) = (580 - 320) / 2.56 = 101.6 units
```

The combined uncertainty is:

```
σ_combined = sqrt(σ_demand² + σ_forecast²) = sqrt(80² + 101.6²) = sqrt(6400 + 10323) = 129.3 units
```

Sizing SS on P50 with only historical σ produces a safety stock of 186 units, but the correct combined-uncertainty SS is:

```
SS_correct = 1.645 * 129.3 * sqrt(2) = 301 units
```

The difference is 115 units — a 62% underestimate in safety stock. The planner sized for 95% service level but is actually achieving approximately 74% due to the ignored forecast uncertainty.

### Three Planning Horizons Currently Missing

| Horizon | Use Case | Currently Available |
|---|---|---|
| 1–4 weeks | Execution: daily pick plans, DC labor | No |
| 1–4 months | Replenishment ordering, safety stock sizing | Point estimate only |
| 12–18 months | Supplier capacity booking, SIOP, S&OP | No |

---

## 2. Input Data Required

### Available Today

| Source | Table | Data |
|---|---|---|
| Backtest pipeline | `backtest_lag_archive` | Historical model predictions (point) |
| Sales history | `fact_sales_monthly` | Actuals by DFU by month |
| DFU dimension | `dim_dfu` | Cluster, ABC class, seasonality profile |
| Safety stock | `fact_safety_stock_targets` | Current SS values |

### Missing — Must Be Sourced or Generated

| Data | Source | Gap |
|---|---|---|
| Quantile model weights | New training run with quantile loss | Requires `train_quantile_model()` (new script) |
| Weekly sales breakdown | Historical `fact_sales_monthly` disaggregation | Formula derivable from existing data |
| Day-of-week shipment distribution | WMS shipment log | Not currently ingested |
| Long-horizon external signals | S&OP inputs, management targets | Manual or ERP integration (future) |

---

## 3. Data Model

### 3.1 New Table: `fact_demand_plan`

**Grain:** `item_no + loc + plan_month + quantile + plan_version`

```sql
CREATE TABLE fact_demand_plan (
    id                  BIGSERIAL PRIMARY KEY,
    item_no             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    plan_month          DATE            NOT NULL,        -- First day of month (YYYY-MM-01)
    quantile            NUMERIC(4,2)    NOT NULL,        -- 0.10, 0.50, or 0.90
    forecast_qty        NUMERIC(12,2)   NOT NULL,
    lower_bound         NUMERIC(12,2),                  -- P10 (populated on all rows for reference)
    upper_bound         NUMERIC(12,2),                  -- P90 (populated on all rows for reference)
    model_id            VARCHAR(100)    NOT NULL,        -- e.g. 'lgbm_quantile_cluster'
    plan_version        VARCHAR(50)     NOT NULL,        -- e.g. '2026-04-01_production'
    horizon_months      INTEGER         NOT NULL,        -- 1–18, how far ahead this row is
    sigma_forecast      NUMERIC(10,4),                  -- Estimated forecast std dev
    sigma_demand        NUMERIC(10,4),                  -- Historical demand std dev
    sigma_combined      NUMERIC(10,4),                  -- sqrt(σ_f² + σ_d²)
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    cluster_id          INTEGER,
    abc_class           VARCHAR(5),
    seasonality_profile VARCHAR(50),
    CONSTRAINT uq_demand_plan UNIQUE (item_no, loc, plan_month, quantile, plan_version)
);

CREATE INDEX idx_demand_plan_item_loc_month
    ON fact_demand_plan (item_no, loc, plan_month);

CREATE INDEX idx_demand_plan_version
    ON fact_demand_plan (plan_version, plan_month);

CREATE INDEX idx_demand_plan_quantile
    ON fact_demand_plan (quantile, plan_version);

CREATE INDEX idx_demand_plan_horizon
    ON fact_demand_plan (horizon_months, plan_version);
```

### 3.2 New Table: `fact_demand_plan_weekly`

**Grain:** `item_no + loc + plan_week + quantile + plan_version`

```sql
CREATE TABLE fact_demand_plan_weekly (
    id                  BIGSERIAL PRIMARY KEY,
    item_no             VARCHAR(50)     NOT NULL,
    loc                 VARCHAR(50)     NOT NULL,
    plan_week           DATE            NOT NULL,        -- Monday of the ISO week
    iso_week            INTEGER         NOT NULL,        -- 1–53
    iso_year            INTEGER         NOT NULL,
    plan_month          DATE            NOT NULL,        -- Parent month (FK reference)
    quantile            NUMERIC(4,2)    NOT NULL,
    forecast_qty        NUMERIC(12,2)   NOT NULL,
    weekly_weight       NUMERIC(6,4)    NOT NULL,        -- Fraction of monthly qty assigned to this week
    plan_version        VARCHAR(50)     NOT NULL,
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_demand_plan_weekly UNIQUE (item_no, loc, plan_week, quantile, plan_version)
);

CREATE INDEX idx_demand_plan_weekly_item_loc
    ON fact_demand_plan_weekly (item_no, loc, plan_week);
```

### 3.3 New Table: `fact_plan_versions`

**Grain:** `plan_version` (one row per generation run)

```sql
CREATE TABLE fact_plan_versions (
    plan_version        VARCHAR(50)     PRIMARY KEY,
    plan_date           DATE            NOT NULL,
    plan_label          VARCHAR(100),                   -- 'production', 'scenario_1', 'what_if_promo'
    model_id            VARCHAR(100)    NOT NULL,
    horizon_months      INTEGER         NOT NULL,
    dfu_count           INTEGER,
    generated_by        VARCHAR(100),
    generated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    status              VARCHAR(20)     NOT NULL DEFAULT 'draft',  -- draft, active, archived
    notes               TEXT,
    parent_version      VARCHAR(50)                     -- For scenario branching
);
```

---

## 4. Quantile Forecast Methods

### Option A: Quantile Regression Loss (Recommended)

LightGBM natively supports quantile loss via `objective='quantile'` and `alpha=0.1 / 0.5 / 0.9`. Three models are trained per cluster (one per quantile). This is the recommended approach because:

- No distributional assumption required (non-parametric)
- Integrates directly with the existing per-cluster backtest framework
- LightGBM quantile training is 2–3x slower than MSE but still fast (< 5 min for all clusters)
- Consistent with existing `run_tree_backtest()` orchestrator

**Quantile loss function (pinball loss):**

```
L_alpha(y, f) = alpha * (y - f)          if y >= f   (under-prediction)
              = (alpha - 1) * (y - f)    if y < f    (over-prediction)
```

For alpha=0.9: the model is penalized 9x more for under-prediction than over-prediction, so it learns to predict the 90th percentile.

### Option B: Monte Carlo Sampling (Alternative)

Train a standard point model, then sample from the residual distribution N=1000 times per DFU. This requires storing residuals per cluster and is more expensive. Not recommended for production use due to storage and latency.

### Sigma Derivation from Quantile Outputs

Once P10 and P90 are available:

```
σ_forecast = (P90 - P10) / (2 * 1.2816)   -- 1.2816 = Z-score for 80% interval
```

Example: P10=320, P90=580
```
σ_forecast = (580 - 320) / 2.5632 = 101.6 units
```

---

## 5. Short-Horizon Weekly Disaggregation

Monthly forecast is disaggregated to weekly using the **historical week-weight distribution** from `fact_sales_monthly`.

Since `fact_sales_monthly` is at monthly grain, the week-weight is estimated from the proportion of weeks in each month falling in each ISO week bucket. For items with a strong day-of-week shipment pattern (future WMS integration), actual weights would be used.

**Default formula (equal-weight within month):**

```python
def get_weekly_weights(plan_month: date) -> list[tuple[date, float]]:
    """
    Returns (week_start, weight) tuples for all ISO weeks that overlap
    the given month. Weight proportional to days-overlap / days-in-month.
    """
    weeks = []
    month_start = plan_month.replace(day=1)
    month_end = (month_start + relativedelta(months=1)) - timedelta(days=1)
    days_in_month = month_end.day

    current = month_start - timedelta(days=month_start.weekday())  # Monday
    while current <= month_end:
        week_end = current + timedelta(days=6)
        overlap_start = max(current, month_start)
        overlap_end = min(week_end, month_end)
        overlap_days = (overlap_end - overlap_start).days + 1
        weight = overlap_days / days_in_month
        weeks.append((current, weight))
        current += timedelta(days=7)

    return weeks
```

**Example: April 2026 disaggregation for Item 100320, P50=450**

| ISO Week | Week Start | Days in Month | Weight | Weekly Qty |
|---|---|---|---|---|
| W14 | Mar 30 | 2 days in Apr | 0.067 | 30.1 |
| W15 | Apr 6 | 7 days | 0.233 | 104.9 |
| W16 | Apr 13 | 7 days | 0.233 | 104.9 |
| W17 | Apr 20 | 7 days | 0.233 | 104.9 |
| W18 | Apr 27 | 5 days | 0.167 | 75.1 |
| **Total** | | **28 days** | **0.933** | **420.0** |

Note: 2 days of W14 that fall in March are excluded. The remaining 2 days of W18 fall in May and belong to that month's plan. Total within-month allocation = 420 ≈ 450 × (28/30).

---

## 6. Python Scripts

### 6.1 `scripts/generate_quantile_forecasts.py`

```python
"""
Generate multi-horizon quantile forecasts (P10/P50/P90) for all active DFUs.

Usage:
    uv run scripts/generate_quantile_forecasts.py \
        --horizon 12 \
        --plan-version 2026-04-01_production \
        --n-timeframes 10

Config: config/quantile_forecast_config.yaml
Output:
    - fact_demand_plan (monthly quantile rows)
    - fact_demand_plan_weekly (weekly disaggregation)
    - fact_plan_versions (version metadata)
"""

import yaml
import pandas as pd
import lightgbm as lgb
import psycopg
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from common.db import get_db_params
from common.backtest_framework import load_feature_grid
from common.feature_engineering import build_feature_matrix

QUANTILES = [0.10, 0.50, 0.90]

def train_quantile_model(
    alpha: float,
    cluster_id: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> lgb.Booster:
    """
    Train a single LightGBM quantile model for a given alpha and cluster.

    Args:
        alpha: Quantile level (0.10, 0.50, 0.90)
        cluster_id: Cluster identifier for logging
        X_train: Feature matrix (n_samples, n_features)
        y_train: Target series (actual demand qty)
        params: LightGBM hyperparameters (from config)

    Returns:
        Trained lgb.Booster
    """
    quantile_params = {
        **params,
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        quantile_params,
        train_data,
        num_boost_round=params.get("n_estimators", 300),
    )
    return model


def generate_quantile_predictions(
    models: dict[float, lgb.Booster],
    predict_data: pd.DataFrame,
    feature_cols: list[str],
    plan_month: date,
    plan_version: str,
    horizon_months: int,
) -> list[dict]:
    """
    Generate P10/P50/P90 predictions for a single cluster's prediction data.

    Args:
        models: dict mapping alpha -> trained Booster
        predict_data: Feature DataFrame for prediction period
        feature_cols: Column names used in training
        plan_month: The month being forecast (DATE)
        plan_version: Version string for this planning run
        horizon_months: How many months ahead this prediction is

    Returns:
        List of row dicts ready for bulk insert into fact_demand_plan
    """
    rows = []
    X_pred = predict_data[feature_cols].fillna(0)

    quantile_preds = {
        alpha: model.predict(X_pred)
        for alpha, model in models.items()
    }

    for i, (_, row) in enumerate(predict_data.iterrows()):
        p10 = max(0, quantile_preds[0.10][i])
        p50 = max(0, quantile_preds[0.50][i])
        p90 = max(0, quantile_preds[0.90][i])
        sigma_f = (p90 - p10) / 2.5632 if p90 > p10 else 0.0

        rows.append({
            "item_no": row["item_no"],
            "loc": row["loc"],
            "plan_month": plan_month,
            "quantile": 0.10,
            "forecast_qty": round(p10, 2),
            "lower_bound": round(p10, 2),
            "upper_bound": round(p90, 2),
            "sigma_forecast": round(sigma_f, 4),
            "horizon_months": horizon_months,
            "plan_version": plan_version,
            "model_id": "lgbm_quantile_cluster",
        })
        # ... repeat for 0.50 and 0.90 quantiles

    return rows


def write_demand_plan(rows: list[dict], plan_version: str) -> int:
    """
    Bulk upsert quantile forecast rows into fact_demand_plan.

    Args:
        rows: List of row dicts (from generate_quantile_predictions)
        plan_version: Version string for conflict-resolution scope

    Returns:
        Number of rows written
    """
    sql = """
        INSERT INTO fact_demand_plan
            (item_no, loc, plan_month, quantile, forecast_qty, lower_bound,
             upper_bound, model_id, plan_version, horizon_months,
             sigma_forecast, generated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (item_no, loc, plan_month, quantile, plan_version)
        DO UPDATE SET
            forecast_qty   = EXCLUDED.forecast_qty,
            lower_bound    = EXCLUDED.lower_bound,
            upper_bound    = EXCLUDED.upper_bound,
            sigma_forecast = EXCLUDED.sigma_forecast,
            generated_at   = NOW()
    """
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, [
                (r["item_no"], r["loc"], r["plan_month"], r["quantile"],
                 r["forecast_qty"], r["lower_bound"], r["upper_bound"],
                 r["model_id"], r["plan_version"], r["horizon_months"],
                 r["sigma_forecast"])
                for r in rows
            ])
        conn.commit()
    return len(rows)


def disaggregate_to_weekly(
    monthly_rows: list[dict],
    plan_version: str,
) -> list[dict]:
    """
    Convert monthly quantile forecasts to weekly using day-overlap weights.

    Args:
        monthly_rows: Output from generate_quantile_predictions (monthly grain)
        plan_version: Plan version string

    Returns:
        List of weekly row dicts for fact_demand_plan_weekly
    """
    weekly_rows = []
    for row in monthly_rows:
        weights = get_weekly_weights(row["plan_month"])
        for week_start, weight in weights:
            weekly_rows.append({
                "item_no": row["item_no"],
                "loc": row["loc"],
                "plan_week": week_start,
                "iso_week": week_start.isocalendar()[1],
                "iso_year": week_start.isocalendar()[0],
                "plan_month": row["plan_month"],
                "quantile": row["quantile"],
                "forecast_qty": round(row["forecast_qty"] * weight, 2),
                "weekly_weight": round(weight, 4),
                "plan_version": plan_version,
            })
    return weekly_rows


def run(horizon: int, plan_version: str, n_timeframes: int) -> None:
    """Main entry point: train quantile models and generate multi-horizon plan."""
    cfg = yaml.safe_load(open("config/quantile_forecast_config.yaml"))
    # ... load feature grid, train per-cluster quantile models,
    #     generate predictions for each future month (1..horizon),
    #     disaggregate to weekly, write all to DB
    pass
```

### 6.2 `config/quantile_forecast_config.yaml`

```yaml
quantile_forecast:
  quantiles: [0.10, 0.50, 0.90]
  default_horizon_months: 12
  max_horizon_months: 18
  weekly_disaggregation: true

  model:
    objective: quantile
    n_estimators: 300
    learning_rate: 0.05
    num_leaves: 31
    min_child_samples: 20
    subsample: 0.8
    colsample_bytree: 0.8

  sigma_guard_rails:
    min_sigma: 0.0
    max_sigma_multiplier: 3.0   # Never allow σ > 3x the P50 value

  plan_versioning:
    production_label: production
    max_active_versions: 5      # Archive older versions automatically
    retention_days: 180
```

---

## 7. API Endpoints

### 7.1 `GET /forecast/demand-plan`

Retrieve quantile forecast rows for a specific item/location.

**Parameters:**
- `item_no` (required): VARCHAR(50)
- `loc` (required): VARCHAR(50)
- `plan_version` (optional, default: latest active): VARCHAR(50)
- `quantile` (optional, default: all): 0.10 | 0.50 | 0.90
- `horizon` (optional, default: 12): INTEGER 1–18

**Response:**

```json
{
  "plan_version": "2026-04-01_production",
  "generated_at": "2026-04-01T06:00:00Z",
  "item_no": "100320",
  "loc": "1401-BULK",
  "horizon_months": 12,
  "rows": [
    {
      "plan_month": "2026-04-01",
      "horizon_months": 1,
      "p10": 320.0,
      "p50": 450.0,
      "p90": 580.0,
      "sigma_forecast": 101.6,
      "sigma_demand": 80.0,
      "sigma_combined": 129.3
    },
    {
      "plan_month": "2026-05-01",
      "horizon_months": 2,
      "p10": 290.0,
      "p50": 420.0,
      "p90": 560.0,
      "sigma_forecast": 105.4,
      "sigma_demand": 80.0,
      "sigma_combined": 132.0
    }
  ]
}
```

### 7.2 `GET /forecast/demand-plan/versions`

List all plan versions with metadata.

**Response:**

```json
{
  "versions": [
    {
      "plan_version": "2026-04-01_production",
      "plan_date": "2026-04-01",
      "plan_label": "production",
      "model_id": "lgbm_quantile_cluster",
      "horizon_months": 12,
      "dfu_count": 4823,
      "status": "active",
      "generated_at": "2026-04-01T06:03:14Z"
    },
    {
      "plan_version": "2026-03-01_production",
      "plan_date": "2026-03-01",
      "plan_label": "production",
      "model_id": "lgbm_quantile_cluster",
      "horizon_months": 12,
      "dfu_count": 4790,
      "status": "archived",
      "generated_at": "2026-03-01T06:01:52Z"
    }
  ]
}
```

### 7.3 `GET /forecast/demand-plan/comparison`

Compare two plan versions for a specific item/location.

**Parameters:** `v1`, `v2`, `item_no`, `loc`

**Response:**

```json
{
  "item_no": "100320",
  "loc": "1401-BULK",
  "v1": "2026-04-01_production",
  "v2": "2026-03-01_production",
  "months": [
    {
      "plan_month": "2026-04-01",
      "v1_p50": 450.0,
      "v2_p50": 410.0,
      "delta_p50": 40.0,
      "delta_pct": 9.76,
      "v1_p10": 320.0,
      "v1_p90": 580.0,
      "v2_p10": 290.0,
      "v2_p90": 540.0
    }
  ]
}
```

### 7.4 `GET /forecast/demand-plan/weekly`

Retrieve weekly disaggregated forecast.

**Parameters:** `item_no`, `loc`, `plan_version`, `weeks_ahead` (default: 8)

**Response:**

```json
{
  "item_no": "100320",
  "loc": "1401-BULK",
  "weeks": [
    {
      "plan_week": "2026-03-30",
      "iso_week": 14,
      "iso_year": 2026,
      "parent_month": "2026-04-01",
      "p10_weekly": 21.4,
      "p50_weekly": 30.1,
      "p90_weekly": 38.9
    },
    {
      "plan_week": "2026-04-06",
      "iso_week": 15,
      "iso_year": 2026,
      "parent_month": "2026-04-01",
      "p10_weekly": 75.3,
      "p50_weekly": 104.9,
      "p90_weekly": 135.1
    }
  ]
}
```

---

## 8. Frontend Components

### 8.1 Demand Plan Sub-Panel in Inv. Planning Tab

Located in: `frontend/src/tabs/inv-planning/DemandPlanPanel.tsx`

**Layout:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  DEMAND PLAN                    [Version: 2026-04-01_production ▼]  │
│  Item: 100320  Loc: 1401-BULK   [Horizon: 12M ▼]  [Compare ▼]      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Qty │  580 ┤                           ╭───╮                     │
│       │  540 ┤                     ╭─────╯   ╰──╮                  │
│       │  490 ┤              ╭──────╯             ╰──╮              │
│  P90  │  450 ┤────────────────────────────────────────  P50        │
│       │  410 ┤           ╰──╮                        ╰──           │
│       │  370 ┤         ╰────╯                                      │
│  P10  │  320 ┤──────────────────────────────────────               │
│       └──────┴──────────────────────────────────────               │
│              Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Combined σ:  129.3   Forecast σ: 101.6   Historical σ: 80.0 │   │
│  │ Recommended SS (95% SL): 301 units  [vs current: 186 units] │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  WEEKLY VIEW (next 8 weeks)                                         │
│  W14: 30.1  W15: 104.9  W16: 104.9  W17: 104.9  W18: 75.1  ...   │
└─────────────────────────────────────────────────────────────────────┘
```

**Key UI behaviors:**
- Fan chart rendered with ECharts using `areaStyle` bands: P10→P90 fills light blue, P50 line solid blue
- Version selector populated from `GET /forecast/demand-plan/versions`
- "Compare" toggle shows version N-1 as a dashed line on the same chart
- Horizon selector: 3M / 6M / 12M / 18M
- SS recommendation chip: shows current SS vs recommended based on σ_combined. Chip is red if current < recommended by > 20%.
- Weekly bar chart below the main fan chart, showing 8-week horizon

---

## 9. Worked Example: End-to-End for Item 100320

### Step 1: Train Three Quantile Models for Cluster 3

Cluster 3 contains 312 DFUs including Item 100320. Training data: 36 months of sales history (Jan 2023 – Dec 2025).

Three LightGBM models are trained with identical features but different loss functions:
- `model_q10`: `alpha=0.10`, learns to predict the 10th percentile
- `model_q50`: `alpha=0.50`, equivalent to MAE minimization
- `model_q90`: `alpha=0.90`, learns to predict the 90th percentile

### Step 2: Generate April 2026 Predictions

Item 100320, Cluster 3:

| Quantile | Raw Prediction | Clipped (floor 0) | Stored |
|---|---|---|---|
| P10 | 317.4 | 317.4 | 317.40 |
| P50 | 452.1 | 452.1 | 452.10 |
| P90 | 583.7 | 583.7 | 583.70 |

Sigma computations:
```
σ_forecast = (583.7 - 317.4) / 2.5632 = 103.9
σ_demand   = 80.0  (from fact_safety_stock_targets.sigma_demand)
σ_combined = sqrt(103.9² + 80.0²) = sqrt(10795 + 6400) = sqrt(17195) = 131.1
```

### Step 3: SS Recommendation

For 95% service level (Z=1.645), lead time = 2 months:
```
SS_old = 1.645 * 80.0 * sqrt(2) = 186.3 units   (using historical σ only)
SS_new = 1.645 * 131.1 * sqrt(2) = 305.0 units  (using combined σ)
```

The system displays a warning: "Safety stock undersized by 118 units (63%). Recommend updating from 186 to 305."

### Step 4: Weekly Disaggregation

April 2026 P50 = 452.1 units. Weekly breakdown:

| Week | Start | Days in Apr | Weight | P50 Weekly |
|---|---|---|---|---|
| W14 | Mar 30 | 2 | 0.067 | 30.3 |
| W15 | Apr 6 | 7 | 0.233 | 105.3 |
| W16 | Apr 13 | 7 | 0.233 | 105.3 |
| W17 | Apr 20 | 7 | 0.233 | 105.3 |
| W18 | Apr 27 | 5 | 0.167 | 75.5 |

---

## 10. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    QUANTILE FORECAST PIPELINE                │
│                                                              │
│  fact_sales_monthly ──► Feature Engineering ──► X_train     │
│  dim_dfu (cluster)  ──►                                      │
│                                                              │
│  X_train ──► train_quantile_model(alpha=0.10) ──► model_q10 │
│           ──► train_quantile_model(alpha=0.50) ──► model_q50 │
│           ──► train_quantile_model(alpha=0.90) ──► model_q90 │
│                                                              │
│  [model_q10, q50, q90] ──► generate_quantile_predictions()  │
│                                    │                         │
│                          ┌─────────▼──────────┐             │
│                          │  fact_demand_plan   │             │
│                          │  (monthly, P10/50/90│             │
│                          └─────────┬──────────┘             │
│                                    │                         │
│                          disaggregate_to_weekly()            │
│                                    │                         │
│                          ┌─────────▼─────────────┐          │
│                          │ fact_demand_plan_weekly│          │
│                          │ (weekly, 8-week horizon│          │
│                          └───────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

---

## 11. Dependencies

| Dependency | Type | Status |
|---|---|---|
| F1.1 — Production Forecast Infrastructure | Hard | Not implemented |
| Feature 44 — Algorithm Config | Hard | Implemented |
| `common/backtest_framework.py` | Hard | Implemented |
| `fact_safety_stock_targets` (IPfeature3) | Soft | Implemented |
| `dim_dfu.cluster_assignment` | Hard | Implemented |
| `config/quantile_forecast_config.yaml` | New | Must be created |

---

## 12. Out of Scope

- Neural network quantile methods (NGBoost, DeepAR confidence intervals)
- Empirical prediction intervals from conformal prediction
- Bayesian demand modelling
- Daily granularity forecasting (below weekly)
- Integration with S&OP planning tools (future F3.x)
- External demand signal ingestion (market data, macro indices)

---

## 13. Makefile Targets

```makefile
quantile-schema:
    uv run python -m psycopg ... sql/039_create_demand_plan.sql

quantile-train:
    uv run scripts/generate_quantile_forecasts.py --horizon 12 --plan-version $(VERSION)

quantile-weekly:
    uv run scripts/generate_quantile_forecasts.py --weekly-only --plan-version $(VERSION)

quantile-all:
    make quantile-schema && make quantile-train && make quantile-weekly
```

---

## 14. Test Requirements

### Backend Unit Tests (`tests/unit/test_quantile_forecasts.py`)

- `test_train_quantile_model_alpha_10`: Assert P10 predictions < P50 for held-out data
- `test_train_quantile_model_alpha_90`: Assert P90 predictions > P50 for held-out data
- `test_prediction_interval_ordering`: P10 <= P50 <= P90 for all rows (no crossings)
- `test_sigma_forecast_computation`: Verify formula `(P90 - P10) / 2.5632`
- `test_sigma_combined_computation`: Verify `sqrt(σ_f² + σ_d²)`
- `test_weekly_disaggregation_sums_to_monthly`: Weekly sum ≈ monthly (within rounding)
- `test_weekly_weights_sum_to_one`: Sum of weights for a full month = 1.0
- `test_write_demand_plan_upsert`: Confirm ON CONFLICT DO UPDATE behavior

### Backend API Tests (`tests/api/test_demand_plan.py`)

- `test_get_demand_plan_returns_three_quantiles`: 3 rows per month in response
- `test_get_demand_plan_versions_list`: Returns list with plan_version and status
- `test_get_demand_plan_comparison_delta`: v1_p50 - v2_p50 matches delta_p50
- `test_get_demand_plan_weekly_8_weeks`: Returns exactly 8 week rows
- `test_get_demand_plan_unknown_version_404`: Returns 404 for unknown plan_version

### Frontend Tests (`frontend/src/tabs/__tests__/DemandPlanPanel.test.tsx`)

- Renders fan chart container (ECharts mock)
- Version selector populated with mock versions
- Horizon selector changes query parameter
- SS recommendation chip renders correct values
- Weekly bar chart renders for 8-week data
