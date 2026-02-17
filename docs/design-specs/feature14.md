# Feature 14: Backtesting Framework — Expanding Window Timeframes for Forecast Models

## Objective

Build a generic backtesting framework that trains forecast models (LGBM, CatBoost, etc.) across multiple expanding-window timeframes, generates multi-lag predictions, stores results in `fact_external_forecast_monthly`, and measures accuracy at each DFU's execution lag.

## Motivation

- **Model comparison**: Evaluate multiple forecast algorithms (LGBM, CatBoost, ensemble) side-by-side using the same backtesting structure.
- **Lag-aware accuracy**: Forecast accuracy depends on lead time — a 1-month-ahead forecast is easier than 4 months ahead. Measuring at execution lag reflects operational reality.
- **Temporal robustness**: Expanding windows test whether a model holds up across different forecast origins, not just one snapshot.

## Core Concept: Expanding Window Timeframes

### Generic Logic

Given:
- `latest_month` = max(startdate) from `fact_sales_monthly` (detected from data)
- `earliest_month` = min(startdate) from `fact_sales_monthly` (detected from data)
- `N` = number of timeframes (default: 10, labeled A–J)
- `max_lag` = maximum forecast lag to store (default: 4, so lags 0–4)

For each timeframe `i` (0-indexed, A=0, B=1, ..., J=9):

```
train_end   = latest_month - (N - i) months
train_start = earliest_month
predict_start = train_end + 1 month
predict_end   = latest_month
```

Only predictions at lag 0–4 are stored. That means each timeframe produces at most `max_lag + 1 = 5` forecast records per DFU.

### Concrete Example (current data)

Sales history: Feb 2023 – Jan 2026 (36 months). N=10.

| Timeframe | Train Period          | Predict Period         | Lags Generated |
|-----------|-----------------------|------------------------|----------------|
| A         | Feb 2023 – Mar 2025   | Apr 2025 – Jan 2026    | 0–4 (cap at 4) |
| B         | Feb 2023 – Apr 2025   | May 2025 – Jan 2026    | 0–4            |
| C         | Feb 2023 – May 2025   | Jun 2025 – Jan 2026    | 0–4            |
| D         | Feb 2023 – Jun 2025   | Jul 2025 – Jan 2026    | 0–4            |
| E         | Feb 2023 – Jul 2025   | Aug 2025 – Jan 2026    | 0–4            |
| F         | Feb 2023 – Aug 2025   | Sep 2025 – Jan 2026    | 0–4            |
| G         | Feb 2023 – Sep 2025   | Oct 2025 – Jan 2026    | 0–4            |
| H         | Feb 2023 – Oct 2025   | Nov 2025 – Jan 2026    | 0–3            |
| I         | Feb 2023 – Nov 2025   | Dec 2025 – Jan 2026    | 0–1            |
| J         | Feb 2023 – Dec 2025   | Jan 2026               | 0              |

### Lag Matrix

Each predicted month accumulates forecasts from multiple timeframes at different lags:

| Predicted Month | Lag 0 (from TF) | Lag 1 | Lag 2 | Lag 3 | Lag 4 |
|-----------------|-----------------|-------|-------|-------|-------|
| Apr 2025        | A               | —     | —     | —     | —     |
| May 2025        | B               | A     | —     | —     | —     |
| Jun 2025        | C               | B     | A     | —     | —     |
| Jul 2025        | D               | C     | B     | A     | —     |
| Aug 2025        | E               | D     | C     | B     | A     |
| Sep 2025        | F               | E     | D     | C     | B     |
| Oct 2025        | G               | F     | E     | D     | C     |
| Nov 2025        | H               | G     | F     | E     | D     |
| Dec 2025        | I               | H     | G     | F     | E     |
| Jan 2026        | J               | I     | H     | G     | F     |

**Months with all 5 lags available**: Aug 2025 – Jan 2026 (last `N - max_lag` = 6 months).

### Generic Shift

When the latest sales month moves from Jan 2026 to Feb 2026, everything shifts by 1 month automatically. The script detects `latest_month` from the data — no hardcoded dates.

## Forecast Storage

### Mapping to `fact_external_forecast_monthly`

Each forecast record maps to the existing schema:

| Column         | Value                                                       |
|----------------|-------------------------------------------------------------|
| `forecast_ck`  | `{dmdunit}_{dmdgroup}_{loc}_{fcstdate}_{startdate}`         |
| `dmdunit`      | DFU item                                                    |
| `dmdgroup`     | DFU group                                                   |
| `loc`          | DFU location                                                |
| `fcstdate`     | `predict_start` of the timeframe (month forecast was generated) |
| `startdate`    | Predicted month                                             |
| `lag`          | `month_diff(startdate, fcstdate)` — 0 to 4                  |
| `execution_lag`| From `dim_dfu.execution_lag` for this DFU                    |
| `basefcst_pref`| Model's predicted demand quantity                           |
| `tothist_dmd`  | Actual demand (from `fact_sales_monthly` for that month)     |
| `model_id`     | Model identifier, e.g. `lgbm_v1`, `catboost_v1`            |

**Constraint**: `UNIQUE(forecast_ck, model_id)` already exists — each (DFU, fcstdate, startdate, model_id) combination stored once.

### What Gets Stored

**All lag 0–4 forecasts** are stored for every timeframe. This enables:
- Lag-by-lag accuracy analysis (how does accuracy degrade with lead time?)
- Filtering to execution lag for operational accuracy
- Model comparison at each lag independently

### Actuals Attachment

`tothist_dmd` is populated by looking up `fact_sales_monthly.qty` for the same DFU and `startdate`. This enables the existing feature10 KPI engine to compute accuracy without any changes.

## Accuracy Measurement

### At Execution Lag

Each DFU has an `execution_lag` value in `dim_dfu` (e.g., 1, 2, 3 months). This represents how far in advance the organization needs the forecast for planning.

**Accuracy at execution lag** = filter forecast records where `lag = execution_lag` for each DFU.

Only months where the execution lag forecast exists are used. Generically, months from `predict_start + execution_lag` of the earliest qualifying timeframe through `latest_month` will have execution-lag forecasts.

**Full-lag accuracy months**: Aug 2025 – Jan 2026 (months where all lags 0–4 exist). Every DFU with `execution_lag` between 0 and 4 will have accuracy data for these 6 months.

### Accuracy Formulas

Uses the same formulas from feature10:
- **WAPE**: `100 * SUM(ABS(F - A)) / ABS(SUM(A))`
- **Bias**: `(SUM(F) / SUM(A)) - 1`
- **Accuracy %**: `100 - WAPE`

Filtered by `model_id` and optionally by `lag` or by matching `lag = execution_lag`.

## Cluster Integration (Feature 13)

Clusters from feature13 (`dim_dfu.ml_cluster`) drive the training strategy:

### Option A: Global Model + Cluster Feature
- Train one model for all DFUs
- `ml_cluster` is a categorical input feature
- Simpler, fewer models to manage

### Option B: Per-Cluster Models
- Train separate model per cluster
- Each cluster gets homogeneous training data
- Potentially better accuracy, more models to manage

### Option C: Hybrid
- Global model for large clusters
- Specialized models for distinct clusters (seasonal, intermittent)

The backtesting framework supports all options — the `model_id` distinguishes them (e.g., `lgbm_global`, `lgbm_cluster_0`, `catboost_global`).

## Feature Engineering for Tree Models

### Lag Features (per DFU per month)
- `qty_lag_1` through `qty_lag_12`: Demand from 1–12 months prior
- `qty_rolling_3m`, `qty_rolling_6m`, `qty_rolling_12m`: Rolling mean demand
- `qty_rolling_std_3m`, `qty_rolling_std_6m`: Rolling standard deviation

### Calendar Features
- `month`: Month of year (1–12)
- `quarter`: Quarter (1–4)
- `month_sin`, `month_cos`: Cyclical month encoding

### DFU / Item Attributes (from dim_dfu, dim_item)
- `ml_cluster`: Cluster assignment (categorical)
- `execution_lag`, `total_lt`: Lead times
- `region`, `brand`, `prod_cat_desc`: Categorical attributes
- `alcoh_pct`, `proof`, `case_weight`: Numeric attributes

### Target
- `qty`: Next month's demand (lag 0 prediction)
- For lag N: predict demand N months ahead using features available at forecast origin

**Important**: Feature engineering must be **causal** — only use data available at the forecast origin (train_end). No future leakage.

## Implementation

### Script: `mvp/demand/scripts/run_backtest.py`

**Responsibilities:**
1. Detect `latest_month` and `earliest_month` from `fact_sales_monthly`
2. Generate N timeframes with expanding window
3. For each timeframe:
   a. Split data into train/predict periods
   b. Engineer features (lag features, rolling stats, calendar, DFU attributes)
   c. Train model(s) on training set
   d. Generate predictions for predict period (lag 0–4)
   e. Attach actuals (`tothist_dmd`) from sales data
4. Combine all timeframe predictions
5. Load into `fact_external_forecast_monthly` with specified `model_id`
6. Log training run to MLflow

**Parameters:**
- `--model`: Model type (`lgbm`, `catboost`, `xgboost`; default: `lgbm`)
- `--model-id`: Model identifier for storage (default: auto-generated, e.g., `lgbm_v1`)
- `--n-timeframes`: Number of expanding windows (default: 10)
- `--max-lag`: Maximum forecast lag (default: 4)
- `--cluster-strategy`: `global`, `per_cluster`, or `global_with_feature` (default: `global_with_feature`)
- `--cluster-source`: Column for cluster assignments: `ml_cluster` or `cluster_assignment` (default: `ml_cluster`)
- `--dry-run`: Generate predictions but don't load to database
- `--output-dir`: Directory for backtest artifacts (default: `data/backtest`)

**Output Files** (saved to `--output-dir`):
- `backtest_predictions.csv`: All predictions with actuals and lags
- `backtest_accuracy.csv`: Per-month accuracy metrics at each lag
- `backtest_metadata.json`: Timeframe definitions, model params, run summary
- `feature_importance.csv`: Feature importance from the tree model

### Script: `mvp/demand/scripts/load_backtest_forecasts.py`

**Responsibilities:**
1. Load `backtest_predictions.csv`
2. Filter to months with all required lags (or configurable: load all)
3. Map to `fact_external_forecast_monthly` schema
4. COPY-insert into PostgreSQL (bulk load, same pattern as `load_dataset_postgres.py`)
5. Refresh `agg_forecast_monthly` materialized view

**Parameters:**
- `--input`: Predictions file (default: `data/backtest/backtest_predictions.csv`)
- `--model-id`: Model ID to use (must match predictions file)
- `--replace`: Delete existing records for this `model_id` before loading

### Makefile Targets

```makefile
backtest-lgbm:
	$(UV) python scripts/run_backtest.py --model lgbm --model-id lgbm_v1

backtest-catboost:
	$(UV) python scripts/run_backtest.py --model catboost --model-id catboost_v1

backtest-load:
	$(UV) python scripts/load_backtest_forecasts.py --replace

backtest-all: backtest-lgbm backtest-load
```

## Data Flow

```
fact_sales_monthly ──┐
dim_dfu ─────────────┤
dim_item ────────────┤
                     ▼
            run_backtest.py
           (N expanding windows)
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    Timeframe A  Timeframe B ... Timeframe J
    (train/predict) (train/predict)  (train/predict)
         │           │           │
         └───────────┼───────────┘
                     ▼
        backtest_predictions.csv
                     │
                     ▼
      load_backtest_forecasts.py
                     │
                     ▼
    fact_external_forecast_monthly
        (model_id = 'lgbm_v1')
                     │
                     ▼
         Existing KPI engine (feature10)
    Filter: model_id + lag = execution_lag
```

## API Integration

### No new endpoints required

The existing multi-model support (feature11) handles everything:
- `GET /domains/forecast/models` — returns `['external', 'lgbm_v1', 'catboost_v1']`
- `GET /domains/forecast/analytics?model=lgbm_v1` — KPIs for that model
- Model selector dropdown in UI works automatically

### Optional: Backtest summary endpoint

`GET /domains/forecast/backtest/summary?model_id=lgbm_v1`

Returns:
```json
{
  "model_id": "lgbm_v1",
  "n_timeframes": 10,
  "accuracy_at_execution_lag": {
    "wape": 22.4,
    "bias": 0.03,
    "accuracy_pct": 77.6,
    "months_evaluated": 6,
    "month_range": ["2025-08-01", "2026-01-01"]
  },
  "accuracy_by_lag": {
    "0": {"wape": 15.2, "bias": 0.01},
    "1": {"wape": 18.7, "bias": 0.02},
    "2": {"wape": 22.4, "bias": 0.03},
    "3": {"wape": 26.1, "bias": -0.01},
    "4": {"wape": 31.5, "bias": -0.04}
  },
  "accuracy_by_cluster": {
    "low_volume_steady": {"wape": 12.3, "n_dfus": 28722},
    "medium_volume_seasonal": {"wape": 25.1, "n_dfus": 82287}
  }
}
```

## MLflow Integration

**Experiment**: `dfu_backtest`

**Per-run logging:**
- **Parameters**: model type, n_timeframes, max_lag, cluster_strategy, hyperparameters
- **Metrics**: WAPE, bias, accuracy % (at execution lag), per-lag metrics, per-cluster metrics
- **Artifacts**: feature_importance.csv, backtest_accuracy.csv, accuracy plots
- **Model**: Serialized model artifact (per-cluster or global)

## Schema Changes

**None required.** The existing `fact_external_forecast_monthly` schema with `model_id` support (feature11) handles backtest forecasts natively.

## Dependencies

New packages (add to `pyproject.toml`):
- `lightgbm>=4.0.0` — LightGBM tree model
- `catboost>=1.2.0` — CatBoost tree model (optional)
- `xgboost>=2.0.0` — XGBoost tree model (optional)

Existing:
- `scikit-learn` — preprocessing, metrics
- `mlflow` — experiment tracking
- `pandas`, `numpy` — data manipulation

## Validation & Quality Checks

1. **No future leakage**: Features for each timeframe use only data available at train_end
2. **Lag correctness**: `lag = month_diff(startdate, fcstdate)` matches dates, constraint-checked by DB
3. **Actuals match**: `tothist_dmd` for each prediction month matches `fact_sales_monthly.qty`
4. **Execution lag coverage**: Warn if a DFU's execution_lag > max_lag (no accuracy measurement possible)
5. **Model ID uniqueness**: `UNIQUE(forecast_ck, model_id)` prevents duplicate inserts
6. **Cluster coverage**: Report how many DFUs per cluster were included in training

## Error Handling

- **Insufficient training data**: DFUs with < 12 months of history in a timeframe are excluded from that timeframe's training (but may still appear in later timeframes)
- **Missing execution_lag**: DFUs without `execution_lag` in `dim_dfu` default to lag 0 for accuracy measurement
- **Missing actuals**: Prediction months without sales data get `tothist_dmd = NULL`; accuracy metrics skip these rows
- **Model convergence**: If a model fails to train for a timeframe, log warning and continue with remaining timeframes

## Usage Workflow

```bash
# 1. Ensure clustering is up to date
make cluster-all

# 2. Run LGBM backtest (generates predictions for 10 timeframes)
make backtest-lgbm

# 3. Load predictions into forecast table
make backtest-load

# 4. View accuracy in UI
#    - Select model "lgbm_v1" in model dropdown
#    - KPIs show accuracy at execution lag
#    - Compare with "external" model side-by-side

# 5. (Optional) Run CatBoost for comparison
make backtest-catboost
make backtest-load  # loads with model_id=catboost_v1
```

## Future Enhancements

- **Hyperparameter tuning**: Grid/random search per timeframe with cross-validation
- **Ensemble models**: Weighted average of LGBM + CatBoost predictions
- **Champion/challenger selection**: Auto-select best model per cluster based on execution-lag accuracy
- **Incremental backtesting**: When new month of sales arrives, only retrain the shifted timeframes
- **Forecast reconciliation**: Top-down/bottom-up reconciliation across hierarchy levels
- **Exception reporting**: Flag DFUs where model accuracy is significantly worse than baseline
