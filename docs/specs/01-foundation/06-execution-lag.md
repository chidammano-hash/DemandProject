# Spec 06 — Execution Lag

> Defines what execution lag is, how it flows through every layer of the system
> (source data, ETL, backtesting, champion selection, accuracy reporting, and UI),
> and the critical distinction between a DFU's **execution lag** and a prediction's
> **forecast lag**.

| | |
|---|---|
| **Status** | Implemented |
| **Key Files** | `dim_sku.execution_lag`, `backtest_framework.py`, `run_champion_selection.py`, `seed_production_baselines.py`, `accuracy_budget.py`, `LagFilterBar.tsx` |
| **UI Surfaces** | Accuracy Tab, Algorithm Experiments, Champion Experiments, DFU Analysis, Inventory Backtest |

---

## 1. What Is Execution Lag?

Execution lag is **the number of months between when a forecast is generated and
when the demand actually occurs**. It is a fixed property of each DFU
(Demand Forecast Unit = item + customer group + location), derived from the
item's total lead time.

| Total lead time | Execution lag | Forecast horizon | Practical meaning |
|---|---|---|---|
| < 30 days | **0** | 1 month ahead | Orders placed in the same month as demand |
| 30–59 days | **1** | 2 months ahead | Orders committed 1 month before demand |
| 60–89 days | **2** | 3 months ahead | Orders committed 2 months before demand |
| 90–119 days | **3** | 4 months ahead | Orders committed 3 months before demand |
| >= 120 days | **4** | 5 months ahead | Orders committed 4+ months before demand |

**A single DFU has exactly one execution lag.** It does not change from month to
month — it is a structural attribute of the supply chain for that item-location
combination, determined by supplier lead times, transit times, and internal
processing windows.

### Why It Matters

The execution lag determines **how far ahead a planner must commit orders**.
A DFU with execution_lag = 0 only needs a 1-month-ahead forecast, so its
accuracy bar is high. A DFU with execution_lag = 4 requires a 5-month-ahead
forecast — inherently less accurate, but that's the reality of long-lead-time
items. Measuring accuracy at the correct execution lag gives a truthful picture
of forecast value for planning decisions.

---

## 2. Where Execution Lag Lives

### 2.1 Source: `dim_sku` (the DFU dimension)

```
Table:  dim_sku
Column: execution_lag INTEGER
Grain:  one row per item_id + customer_group + loc
Source: loaded from SKU master CSV via load_dataset_postgres.py
```

Every DFU row carries its execution lag. Downstream pipelines join to `dim_sku`
to look up a DFU's execution lag.

### 2.2 Forecast tables

Both forecast tables carry the execution lag for query-time filtering:

| Table | Columns | Notes |
|---|---|---|
| `fact_external_forecast_monthly` | `lag INTEGER`, `execution_lag INTEGER` | `lag` is the forecast lag (0-4). For external forecasts, only the row where `lag = execution_lag` is loaded. ML backtests load all 5 lags. |
| `backtest_lag_archive` | `lag INTEGER`, `execution_lag INTEGER` | Always stores all 5 lags (0-4) per DFU-month for every model. |

---

## 3. Two Different "Lags" — The Critical Distinction

The system has **two lag concepts** that share the 0-4 integer range but mean
fundamentally different things:

### 3.1 Execution Lag (DFU property)

- **Stored in:** `dim_sku.execution_lag`
- **Set by:** lead time from source data
- **Cardinality:** one value per DFU, never changes
- **Meaning:** "This DFU's orders are committed N months before demand"
- **Example:** Item ABC at warehouse 01 has `execution_lag = 2` because its
  supplier is overseas with 75-day lead time

### 3.2 Forecast Lag (prediction property)

- **Stored in:** `fact_external_forecast_monthly.lag`, `backtest_lag_archive.lag`
- **Set by:** `assign_natural_lags()` in the backtest framework, computed from
  the timeframe's training cutoff: `lag = months(startdate - train_end) - 1`
- **Cardinality:** up to 5 rows (0-4) per DFU-month per model
- **Meaning:** "This prediction was made N months before the demand month"
- **Key property:** Each lag comes from a **different timeframe** (different
  model training cutoff), so each lag has a genuinely different `basefcst_pref`
- **Example:** Item ABC at warehouse 01 for Feb 2026: lag 0 is the prediction
  from Timeframe J (trained through Jan), lag 2 is from Timeframe H (trained
  through Nov) — these are different model outputs

### 3.3 The Intersection

The **production-relevant prediction** for a DFU is the one where:

```sql
forecast_lag = execution_lag
-- i.e., fact_external_forecast_monthly.lag = dim_sku.execution_lag
```

This is the prediction that would have been available to the planner at the time
they needed to commit orders. All other lag rows are analytical — useful for
understanding model behavior but not operationally relevant.

The canonical SQL pattern used across the entire codebase:

```sql
WHERE f.lag = COALESCE(d.execution_lag, 0)
```

This appears in: `accuracy_budget.py` (12 queries), `mv_inventory_forecast_monthly`,
`agg_dfu_coverage` (partial index), and every production forecast join.

---

## 4. Data Flow Through the System

### 4.1 Backtest Framework

```
backtest_framework.py
│
├── assign_execution_lag(predictions, execution_lag_map)
│   │
│   │  For each prediction row:
│   │    1. Look up DFU's execution_lag from dim_sku
│   │    2. Set  lag = execution_lag
│   │    3. Set  fcstdate = startdate - execution_lag months
│   │
│   └── Output: ONE row per DFU-month (at its execution lag)
│       → Used for: portfolio accuracy, production forecast table
│
├── assign_natural_lags(predictions, timeframes, execution_lag_map, max_lag=4)
│   │
│   │  For each prediction row:
│   │    1. Compute natural lag from timeframe:
│   │       lag = months_between(startdate, train_end) - 1
│   │    2. Keep only rows where 0 <= lag <= 4
│   │    3. Each lag comes from a DIFFERENT timeframe (different model)
│   │
│   │  Example for demand month Feb 2026 (10 timeframes A-J):
│   │    Lag 0: Timeframe J (trained through Jan 2026, 1mo ahead)
│   │    Lag 1: Timeframe I (trained through Dec 2025, 2mo ahead)
│   │    Lag 2: Timeframe H (trained through Nov 2025, 3mo ahead)
│   │    Lag 3: Timeframe G (trained through Oct 2025, 4mo ahead)
│   │    Lag 4: Timeframe F (trained through Sep 2025, 5mo ahead)
│   │
│   └── Output: up to FIVE rows per DFU-month (one per forecast lag)
│       → Used for: backtest_lag_archive, per-lag accuracy analysis
│       → Each lag has genuinely different basefcst_pref values
│
└── compute_accuracy_metrics(execution_lag_predictions)
    └── Portfolio accuracy = accuracy at each DFU's execution lag
```

### 4.2 Forecast Loading (Dual-Path)

```
load_backtest_forecasts.py
│
├── Phase 3b (Archive): Insert ALL 5 lag rows into backtest_lag_archive
│   └── Preserves the full horizon range for analysis
│
├── Phase 3c (Mutation): Set execution_lag from dim_sku on staging data
│   └── Prepares for execution-lag filtering
│
└── Phase 5 (Main): Insert only WHERE lag = execution_lag
    └── fact_external_forecast_monthly gets one row per DFU-month
```

### 4.3 Accuracy Reporting

All accuracy queries in the platform use the execution-lag join:

```sql
-- "How accurate is this model for planning purposes?"
SELECT ... accuracy ...
FROM fact_external_forecast_monthly f
JOIN dim_sku d ON d.item_id = f.item_id AND d.loc = f.loc
WHERE f.lag = COALESCE(d.execution_lag, 0)   -- ← the canonical pattern
```

This ensures every accuracy number reflects the **planning-relevant horizon**
for each DFU.

### 4.4 Champion Selection

```
run_champion_selection.py / run_champion_experiment.py
│
├── load_monthly_errors_df(models, lag_mode="execution")
│   │
│   │  SQL: WHERE lag::text = execution_lag::text
│   │  → Only loads rows where forecast lag matches DFU's execution lag
│   │  → Each DFU contributes exactly one row per month
│   │
│   └── Used for: picking the best model per DFU for operational use
│
└── Champion = model that wins most DFU-months at execution lag
    └── This is the model that would minimize error for actual orders
```

---

## 5. The Lag Filter Bar in Experiment Tabs

The UI provides a segmented control (`LagFilterBar`) with 6 buttons:

```
┌──────────┬───────┬───────┬───────┬───────┬───────┐
│ Exec Lag │ Lag 0 │ Lag 1 │ Lag 2 │ Lag 3 │ Lag 4 │
│(Portfolio)│ (1mo) │ (2mo) │ (3mo) │ (4mo) │ (5mo) │
└──────────┴───────┴───────┴───────┴───────┴───────┘
```

### 5.1 What Each Button Means

| Button | Population | Horizon | Use case |
|---|---|---|---|
| **Exec Lag** | ALL DFUs | Each at its own execution lag | "What's the real planning accuracy?" |
| **Lag 0** | ALL DFUs | 1 month ahead | "How good is the model at the shortest horizon?" |
| **Lag 1** | ALL DFUs | 2 months ahead | "How does accuracy degrade at 2mo?" |
| **Lag 2** | ALL DFUs | 3 months ahead | "How does accuracy degrade at 3mo?" |
| **Lag 3** | ALL DFUs | 4 months ahead | "How does accuracy degrade at 4mo?" |
| **Lag 4** | ALL DFUs | 5 months ahead | "Worst case — longest horizon accuracy?" |

**Key:** Every button shows **ALL DFUs**. The difference is **which forecast
horizon** is used to measure accuracy. Lag 0-4 buttons do NOT filter to a
subset of DFUs — they show the entire portfolio evaluated at a specific horizon.

### 5.2 Expected Accuracy Pattern

Accuracy should **decrease monotonically** from Lag 0 to Lag 4:

```
Lag 0 (1mo):  ~78%   ← easiest to predict, closest to demand
Lag 1 (2mo):  ~75%
Lag 2 (3mo):  ~72%
Lag 3 (4mo):  ~70%
Lag 4 (5mo):  ~67%   ← hardest to predict, farthest from demand
Exec Lag:     ~73%   ← weighted mix (depends on DFU lead-time distribution)
```

Exec Lag accuracy is typically between Lag 1 and Lag 3, because the DFU
population has a mix of lead times (some short, some long).

### 5.3 n_dfu_months Consistency Check

When the lag filter is working correctly:

- **n_dfu_months should be identical across Lag 0-4** — every DFU has a
  prediction at every lag
- **n_dfu_months for Exec Lag** should also equal the same number (same DFUs,
  just each measured at its own lag)
- If n_dfu_months varies by lag, something is wrong (likely filtering by DFU
  execution lag cohort instead of forecast lag)

---

## 6. How Each Experiment Tab Handles Lags

### 6.1 Algorithm Experiments (Model Tuning) — Correct

**Data source:** `lgbm_tuning_lag`, populated by `seed_production_baselines.py`

```python
# seed_production_baselines.py
all_lags_df = pd.read_csv("backtest_predictions_all_lags.csv")
for lag_val, grp in all_lags_df.groupby("lag"):    # ← groups by FORECAST lag
    metrics = compute_accuracy(grp)                 # ← ALL DFUs at this horizon
    INSERT INTO lgbm_tuning_lag (run_id, exec_lag, accuracy_pct, ...)
```

- `exec_lag = 0` → all DFUs at forecast lag 0 (1-month-ahead, highest accuracy) ✓
- `exec_lag = 4` → all DFUs at forecast lag 4 (5-month-ahead, lowest accuracy) ✓
- Portfolio → accuracy at each DFU's execution lag (from tuning run) ✓
- Each lag has genuinely different accuracy because `assign_natural_lags()` maps
  each lag to a different timeframe's prediction (different model training cutoff) ✓

### 6.2 Champion Experiments — Fixed

**Data source:** `champion_experiment_lag`, populated by `run_champion_experiment.py`

```python
# run_champion_experiment.py (FIXED)
# Load all-lags data from backtest_lag_archive
all_lags_df = load_monthly_errors_df(db, models, lag_mode="all")
for lag_val, lag_group in all_lags_df.groupby("lag"):
    # ← groups by FORECAST LAG — all DFUs at each horizon
    lag_winners = strategy_fn(lag_group)
```

- `exec_lag = 0` → all DFUs at forecast lag 0 (1-month-ahead) ✓
- `exec_lag = 3` → all DFUs at forecast lag 3 (4-month-ahead) ✓
- Each lag has genuinely different predictions from different timeframes ✓
- Portfolio → champion accuracy at execution lag (unchanged) ✓

---

## 7. Execution Lag in Other Contexts

### 7.1 Accuracy Tab

All accuracy panels use `f.lag = COALESCE(d.execution_lag, 0)` — they always
show accuracy at execution lag. There is no lag selector here because the
accuracy tab always reports the planning-relevant metric.

### 7.2 DFU Analysis (Item-Level Drilldown)

Per-DFU accuracy charts show the forecast at that DFU's specific execution lag.
The chart title includes the execution lag value:
"Forecast vs Actuals (Execution Lag: 2 = 3mo ahead)"

### 7.3 Inventory Backtest

Root cause attribution and trend charts use execution-lag-matched forecasts via
`mv_inventory_forecast_monthly`, which joins with `f.lag = COALESCE(d.execution_lag, 0)`.

### 7.4 Production Forecast Generation

`generate_production_forecasts.py` only generates predictions at each DFU's
execution lag — there is no reason to generate 5 predictions per DFU in
production since only one horizon is operationally relevant.

---

## 8. Common Mistakes to Avoid

### 8.1 Confusing the two lags

```
WRONG:  "Lag 3 shows DFUs with execution_lag=3"
RIGHT:  "Lag 3 shows ALL DFUs measured at the 4-month-ahead forecast horizon"
```

### 8.2 Filtering DFUs by execution lag in per-lag views

```python
# WRONG — this creates a cohort filter, not a horizon filter
for lag_val, grp in df.groupby("execution_lag"):

# RIGHT — this shows all DFUs at each forecast horizon
for lag_val, grp in all_lags_df.groupby("lag"):
```

### 8.3 Forgetting the COALESCE

```sql
-- WRONG — NULLs drop out silently
WHERE f.lag = d.execution_lag

-- RIGHT — default to lag 0 when execution_lag is NULL
WHERE f.lag = COALESCE(d.execution_lag, 0)
```

### 8.4 Using `lag` and `execution_lag` interchangeably in column names

The `lgbm_tuning_lag` and `champion_experiment_lag` tables both have a column
called `exec_lag` that stores the **forecast lag** (0-4, all DFUs at that
horizon). This naming is misleading — it would be more accurate to call it
`forecast_lag`. Be aware of this when reading the code.

---

## 9. Summary

```
dim_sku.execution_lag
    │
    │  "This DFU commits orders N months ahead"
    │   Fixed per DFU. Driven by lead time.
    │   Values: 0, 1, 2, 3, 4
    │
    │                        backtest_lag_archive.lag
    │                            │
    │                            │  "This prediction was made N months ahead"
    │                            │   5 rows per DFU-month (one per horizon)
    │                            │   Values: 0, 1, 2, 3, 4
    │                            │
    ▼                            ▼
┌─────────────────────────────────────────────┐
│         lag = execution_lag                  │
│                                             │
│  The production-relevant prediction.        │
│  The one that would have been available     │
│  to the planner at order-commit time.       │
│                                             │
│  This is what "Exec Lag" button shows.      │
│  This is what all accuracy queries use.     │
│  This is what champion selection optimizes. │
└─────────────────────────────────────────────┘

UI Lag Filter:
  "Exec Lag"  = each DFU at its own execution lag (production accuracy)
  "Lag 0-4"   = ALL DFUs at a specific forecast horizon (diagnostic view)
```
