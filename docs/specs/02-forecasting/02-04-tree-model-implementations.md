<!-- SOURCE: feature9.md (LGBM) -->
# Feature 9: LGBM Backtesting Implementation

## Objective
Implement the backtesting framework (Feature 8) with LightGBM models, supporting both global and per-cluster training strategies.

## Scope
- **Models**: LightGBM regressors for monthly demand forecasting
- **Strategies**: Global model (one LGBM, `ml_cluster` as feature) and per-cluster (separate LGBM per cluster)
- **Timeframes**: 10 expanding windows (A-J), auto-detected from data
- **Main table**: Each prediction stored at the DFU's `execution_lag` from `dim_dfu`
- **Archive table**: All lags 0-4 preserved in `backtest_lag_archive` for accuracy reporting at any horizon

## Model IDs

| Strategy | model_id | Description |
|----------|----------|-------------|
| Global | `lgbm_global` | One model for all DFUs, `ml_cluster` as feature |
| Per-cluster | `lgbm_cluster` | Separate model per `ml_cluster` |

## Feature Engineering

All features are **strictly causal** — only data available before the target month is used.

### Lag Features (12)
- `qty_lag_1` through `qty_lag_12`: Historical demand shifted by N months

### Rolling Statistics (6)
- `rolling_mean_3m`, `rolling_mean_6m`, `rolling_mean_12m` (shifted by 1)
- `rolling_std_3m`, `rolling_std_6m`, `rolling_std_12m` (shifted by 1)

### Calendar Features (4)
- `month` (1-12), `quarter` (1-4), `month_sin`, `month_cos`

### DFU Attributes
- `ml_cluster` (categorical), `execution_lag`, `total_lt`, `region`, `brand`, `abc_vol`

### Item Attributes
- `case_weight`, `item_proof`, `bpc`

### Grid Construction
- Complete (DFU x month) grid ensures lag features work for zero-demand months
- Sales data masked at `train_end` cutoff to prevent future leakage

#### Example

```python
import pandas as pd
import numpy as np

# Build lag features for item 100320 at 1401-BULK
df = sales_df[sales_df.dmdunit == '100320'].sort_values('startdate')
for lag in [1, 2, 3, 6, 12]:
    df[f'qty_lag_{lag}'] = df['qty_shipped'].shift(lag)

# Rolling stats — ALWAYS shift(1) before rolling to prevent data leakage
df['rolling_mean_3m'] = df['qty_shipped'].shift(1).rolling(3).mean()
df['rolling_std_6m']  = df['qty_shipped'].shift(1).rolling(6).std()

# Calendar cyclical features
df['month_sin'] = np.sin(2 * np.pi * df['startdate'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['startdate'].dt.month / 12)
```

## Lag Strategy

### Main Table (`fact_external_forecast_monthly`)
Predictions stored **only at execution lag**:
- `fcstdate = startdate - execution_lag months`
- `lag = execution_lag` for every row
- ~5x fewer rows than full lag 0-4 expansion

### Archive Table (`backtest_lag_archive`)
All lags 0-4 preserved:
- Same prediction expanded to 5 rows (lag 0, 1, 2, 3, 4)
- Includes `timeframe` column (A-J) for traceability
- Unique on `(forecast_ck, model_id, lag)`

## Implementation

### Scripts

| Script | Purpose |
|--------|---------|
| `mvp/demand/scripts/run_backtest.py` | LGBM-specific training functions + argparse (imports shared framework from `common/`) |
| `mvp/demand/common/backtest_framework.py` | Shared orchestrator: data loading, timeframes, feature engineering, output saving, MLflow |
| `mvp/demand/common/feature_engineering.py` | Shared feature matrix construction, lag/rolling features, future masking |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load predictions into Postgres (main + archive) |

The script contains only three LGBM-specific functions (`train_and_predict_global`, `train_and_predict_per_cluster`, `train_and_predict_transfer`) passed to `run_tree_backtest()` from the shared framework.

### run_backtest.py
Parameters: `--cluster-strategy`, `--model-id`, `--n-timeframes`, `--output-dir`, `--n-estimators`, `--learning-rate`, `--num-leaves`, `--min-child-samples`

Output:
- `backtest_predictions.csv`: Execution-lag only (for main table)
- `backtest_predictions_all_lags.csv`: All lags 0-4 (for archive)
- `backtest_metadata.json`, `feature_importance.csv`

### load_backtest_forecasts.py
Parameters: `--input`, `--model-id`, `--replace`

Pattern: COPY -> temp staging -> INSERT with upsert -> refresh agg view. Auto-loads archive CSV if present.

## Makefile Targets

```makefile
backtest-lgbm:          # Global LGBM backtest
backtest-lgbm-cluster:  # Per-cluster LGBM backtest
backtest-load:          # Load predictions into Postgres (main + archive)
backtest-all:           # backtest-lgbm + backtest-load
```

#### Example

```bash
# Run the global LGBM backtest (10 expanding timeframes A-J)
cd mvp/demand
make backtest-lgbm
# Output: data/backtest_predictions.csv (~95k rows at execution lag)
#         data/backtest_predictions_all_lags.csv (~475k rows, lags 0-4)

# Load into Postgres and refresh materialized views
make backtest-load
# Inserts into fact_external_forecast_monthly WHERE model_id='lgbm_global'
# Refreshes: agg_forecast_monthly, agg_accuracy_by_dim, agg_dfu_coverage,
#            agg_accuracy_lag_archive, agg_dfu_coverage_lag_archive

# Verify predictions loaded
curl -s "http://localhost:8000/domains/forecast/rows?model_id==lgbm_global&limit=5" | python3 -m json.tool
```

## Schema

### Main table
Uses existing `fact_external_forecast_monthly` with `model_id` support (Feature 6).

### Archive table (`backtest_lag_archive`)
New table in `mvp/demand/sql/010_create_backtest_lag_archive.sql`:

| Column | Type | Description |
|--------|------|-------------|
| `archive_sk` | BIGSERIAL PK | Surrogate key |
| `forecast_ck` | TEXT NOT NULL | Composite business key |
| `dmdunit` | TEXT NOT NULL | Item |
| `dmdgroup` | TEXT NOT NULL | Product group |
| `loc` | TEXT NOT NULL | Location |
| `fcstdate` | DATE NOT NULL | Forecast creation date |
| `startdate` | DATE NOT NULL | Actual month being forecast |
| `lag` | INTEGER NOT NULL | 0-4 |
| `execution_lag` | INTEGER | DFU's execution lag |
| `basefcst_pref` | NUMERIC(18,4) | Forecast value |
| `tothist_dmd` | NUMERIC(18,4) | Actual demand |
| `model_id` | TEXT NOT NULL | Model identifier |
| `timeframe` | TEXT | Backtest timeframe (A-J) |
| `load_ts` | TIMESTAMPTZ | Record load timestamp |

Constraints: `UNIQUE(forecast_ck, model_id, lag)`, lag 0-4, month-start checks.

## Verification

```bash
cd mvp/demand && uv sync          # Install dependencies
make db-apply-sql                  # Create backtest_lag_archive
make backtest-lgbm                 # Run global backtest
make backtest-load                 # Load main + archive
curl "http://localhost:8000/domains/forecast/models"
curl "http://localhost:8000/domains/forecast/analytics?model=lgbm_global"
make backtest-lgbm-cluster         # Per-cluster backtest
make backtest-load                 # Reload
```

#### Example: Archive Table Query

```sql
-- Verify all 5 lags are stored for lgbm_global
SELECT lag, COUNT(*) AS rows, ROUND(AVG(basefcst_pref),2) AS avg_fcst
FROM backtest_lag_archive
WHERE model_id = 'lgbm_global'
GROUP BY lag ORDER BY lag;
-- lag | rows   | avg_fcst
--   0 | 95621  |  1243.50
--   1 | 95621  |  1231.80
--   2 | 95621  |  1218.40
--   3 | 95621  |  1205.20
--   4 | 95621  |  1198.70

-- Check per-cluster model coverage
SELECT model_id, COUNT(DISTINCT dmdunit || dmdgroup || loc) AS dfus, COUNT(*) AS rows
FROM fact_external_forecast_monthly
WHERE model_id IN ('lgbm_global', 'lgbm_cluster')
GROUP BY model_id;
```

## Dependencies
- Feature 8 (backtesting framework)
- Feature 7 (clustering)
- Feature 4 (fact tables)
- lightgbm >= 4.0.0, python-dateutil >= 2.8.0

---

## Implementation Details

### Additional Model ID
- Transfer strategy: `lgbm_transfer` — global base model fine-tuned per cluster via `init_model`

### Additional CLI Parameters
- `--transfer-n-estimators` (default: 100) — additional trees for per-cluster fine-tuning
- `--transfer-min-rows` (default: 20) — minimum cluster rows for fine-tuning
- `--verbosity` (default: -1) — LightGBM verbosity level

### Apple GPU (OpenCL) Auto-Detection
- macOS: auto-detects OpenCL GPU availability; if found, `device="gpu"` added to model params

### Shared Module Dependencies
- `common/constants.py`: `CAT_FEATURES`, `LAG_RANGE`, `ROLLING_WINDOWS`, `OUTPUT_COLS`, `ARCHIVE_COLS`, `MIN_TRAINING_MONTHS` (13), `MIN_CLUSTER_ROWS` (50), `MAX_ARCHIVE_LAG` (4)
- `common/metrics.py`: `compute_accuracy_metrics()`
- `common/mlflow_utils.py`: `log_backtest_run()`
- `common/db.py`: `get_db_params()`

### Makefile Target
- `backtest-lgbm-transfer` — LGBM transfer learning backtest

### Load Script Details (`load_backtest_forecasts.py`)
- `BATCH_SIZE = 2,000,000` for batched inserts
- Drops secondary indexes/constraints before bulk insert (`--replace`), recreates after
- Refreshes 5 materialized views: `agg_forecast_monthly`, `agg_accuracy_by_dim`, `agg_dfu_coverage`, `agg_accuracy_lag_archive`, `agg_dfu_coverage_lag_archive`
- Uses upsert (`ON CONFLICT DO UPDATE`) when not in replace mode


---

## Additional Examples

#### Example — Lag Strategy Row Expansion

```python
import pandas as pd

# For a DFU with execution_lag=2, one prediction becomes:
#   Main table : 1 row  (lag=2, fcstdate = startdate - 2 months)
#   Archive    : 5 rows (lag=0 through lag=4)

base_row = {
    "dmdunit": "100320", "loc": "1401-BULK",
    "startdate": pd.Timestamp("2025-04-01"),
    "basefcst_pref": 1200, "tothist_dmd": 1100,
    "model_id": "lgbm_global", "timeframe": "G",
}

archive_rows = [
    {**base_row,
     "lag": lag,
     "fcstdate": base_row["startdate"] - pd.DateOffset(months=lag),
     "execution_lag": lag}        # original lag preserved per row in archive
    for lag in range(5)            # 0, 1, 2, 3, 4
]
# Phase 3b loads archive BEFORE staging mutation → each row keeps its own lag
# Phase 5 inserts main table WHERE lag = execution_lag (only lag=2 row enters)
```

#### Example — Apple GPU (OpenCL) Auto-Detection

```python
import platform, subprocess

def _detect_lgbm_device() -> str:
    """Return 'gpu' if OpenCL GPU is available on macOS, else 'cpu'."""
    if platform.system() != "Darwin":
        return "cpu"
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5
        )
        if "Metal" in result.stdout or "OpenCL" in result.stdout:
            return "gpu"
    except Exception:
        pass
    return "cpu"

# Usage in run_backtest.py
device = _detect_lgbm_device()
model_params = {"device": device, "verbosity": -1, "n_estimators": 500}
# LightGBM silently falls back to CPU if GPU driver is missing
```

#### Example — Load Script Batch Insert

```bash
# Bulk-load 475k archive rows using BATCH_SIZE=2,000,000 with index drop/recreate
uv run python scripts/load_backtest_forecasts.py \
  --input data/backtest_predictions_all_lags.csv \
  --model-id lgbm_global \
  --replace
# Drops secondary indexes before COPY (faster bulk insert)
# Recreates indexes + refreshes 5 materialized views on completion
# Elapsed: ~45 seconds for 475k archive rows on local Postgres
```


---

<!-- SOURCE: feature12.md (CatBoost) -->
# Feature 12: CatBoost Backtesting Implementation

## Objective
Implement the backtesting framework (Feature 8) with CatBoost models, supporting both global and per-cluster training strategies. Mirrors the LGBM implementation (Feature 9) with CatBoost-specific optimizations.

## Scope
- **Models**: CatBoost regressors for monthly demand forecasting
- **Strategies**: Global model (one CatBoost, `ml_cluster` as feature) and per-cluster (separate CatBoost per cluster)
- **Timeframes**: 10 expanding windows (A-J), auto-detected from data
- **Main table**: Each prediction stored at the DFU's `execution_lag` from `dim_dfu`
- **Archive table**: All lags 0-4 preserved in `backtest_lag_archive` for accuracy reporting at any horizon

## Model IDs

| Strategy | model_id | Description |
|----------|----------|-------------|
| Global | `catboost_global` | One model for all DFUs, `ml_cluster` as feature |
| Per-cluster | `catboost_cluster` | Separate model per `ml_cluster` |

## CatBoost-Specific Details

### Categorical Feature Handling
- CatBoost handles categorical features natively via ordered target encoding
- Categorical columns passed as string dtype; column indices provided via `cat_features` parameter
- No manual one-hot encoding required

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 500 | Number of boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage |
| `depth` | 6 | Tree depth |
| `l2_leaf_reg` | 3.0 | L2 regularization |
| `loss_function` | RMSE | Regression objective |
| `random_seed` | 42 | Reproducibility |

### GPU Support
- CatBoost supports GPU training via `task_type="GPU"`
- Auto-detected at runtime; falls back to CPU if unavailable

## Feature Engineering

Identical to Feature 9 (LGBM) — all features are **strictly causal**.

### Lag Features (12)
- `qty_lag_1` through `qty_lag_12`: Historical demand shifted by N months

### Rolling Statistics (6)
- `rolling_mean_3m`, `rolling_mean_6m`, `rolling_mean_12m` (shifted by 1)
- `rolling_std_3m`, `rolling_std_6m`, `rolling_std_12m` (shifted by 1)

### Calendar Features (4)
- `month` (1-12), `quarter` (1-4), `month_sin`, `month_cos`

### DFU Attributes
- `ml_cluster` (categorical), `execution_lag`, `total_lt`, `region`, `brand`, `abc_vol`

### Item Attributes
- `case_weight`, `item_proof`, `bpc`

### Grid Construction
- Complete (DFU x month) grid ensures lag features work for zero-demand months
- Sales data masked at `train_end` cutoff to prevent future leakage

## Lag Strategy

### Main Table (`fact_external_forecast_monthly`)
Predictions stored **only at execution lag**:
- `fcstdate = startdate - execution_lag months`
- `lag = execution_lag` for every row

### Archive Table (`backtest_lag_archive`)
All lags 0-4 preserved:
- Same prediction expanded to 5 rows (lag 0, 1, 2, 3, 4)
- Includes `timeframe` column (A-J) for traceability
- Unique on `(forecast_ck, model_id, lag)`

## Implementation

### Scripts

| Script | Purpose |
|--------|---------|
| `mvp/demand/scripts/run_backtest_catboost.py` | CatBoost-specific training functions + argparse (imports shared framework from `common/`) |
| `mvp/demand/common/backtest_framework.py` | Shared orchestrator: data loading, timeframes, feature engineering, output saving, MLflow |
| `mvp/demand/common/feature_engineering.py` | Shared feature matrix construction (uses `cat_dtype="str"` for CatBoost's index-based categoricals) |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load predictions into Postgres (main + archive) — shared with LGBM/XGBoost |

The script contains only three CatBoost-specific functions (`train_and_predict_global`, `train_and_predict_per_cluster`, `train_and_predict_transfer`) passed to `run_tree_backtest()` from the shared framework.

### run_backtest_catboost.py
Parameters: `--cluster-strategy`, `--model-id`, `--n-timeframes`, `--output-dir`, `--iterations`, `--learning-rate`, `--depth`, `--l2-leaf-reg`, `--random-seed`

Output:
- `backtest_predictions.csv`: Execution-lag only (for main table)
- `backtest_predictions_all_lags.csv`: All lags 0-4 (for archive)
- `backtest_metadata.json`, `feature_importance.csv`

### load_backtest_forecasts.py
Shared loader — same script used for LGBM, CatBoost, and XGBoost. Model-agnostic; operates on `model_id` column in CSV.

## Makefile Targets

```makefile
backtest-catboost:          # Global CatBoost backtest
backtest-catboost-cluster:  # Per-cluster CatBoost backtest
backtest-load:              # Load predictions into Postgres (main + archive) — shared
backtest-all:               # backtest-lgbm + backtest-load (unchanged)
```

## Schema

No schema changes required. Uses existing:
- `fact_external_forecast_monthly` with `model_id` support (Feature 6)
- `backtest_lag_archive` (Feature 8/9)

## Verification

```bash
cd mvp/demand && uv sync          # Install dependencies (includes catboost)
make db-apply-sql                  # Ensure tables exist
make backtest-catboost             # Run global backtest
make backtest-load                 # Load main + archive
curl "http://localhost:8000/domains/forecast/models"
curl "http://localhost:8000/domains/forecast/analytics?model=catboost_global"
make backtest-catboost-cluster     # Per-cluster backtest
make backtest-load                 # Reload
```

## Dependencies
- Feature 8 (backtesting framework)
- Feature 7 (clustering)
- Feature 4 (fact tables)
- catboost >= 1.2.0, python-dateutil >= 2.8.0

---

## Implementation Details

### Additional Model ID
- Transfer strategy: `catboost_transfer` — global base model fine-tuned per cluster via `init_model`

### Additional CLI Parameters
- `--transfer-iterations` (default: 100) — additional iterations for fine-tuning
- `--transfer-min-rows` (default: 20) — minimum cluster rows

### Default Hyperparameters
- `verbose: 0` (CatBoost silent mode, not in original spec)

### `cat_dtype` Implementation
- `cat_dtype="str"` passed to `run_tree_backtest()`, causing categoricals to use `.astype(str)`
- CatBoost receives column indices via `cat_features=cat_indices` (not column names)

### Makefile Target
- `backtest-catboost-transfer` — CatBoost transfer learning backtest

### Shared Modules
- Same as Feature 9: `common/constants.py`, `common/metrics.py`, `common/mlflow_utils.py`, `common/db.py`


---

## Examples

### Example: Run CatBoost backtest

```bash
make backtest-catboost
# CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05)
# cat_dtype='str' for CatBoost (uses ordered target encoding internally)
# Training on 10 expanding timeframes A-J
# Output: data/backtest_catboost_global.csv  (95,621 rows)
make backtest-load  # load into Postgres as model_id='catboost_global'
```

### Example: CatBoost training function

```python
from catboost import CatBoostRegressor
import numpy as np

cat_cols = [X_train.columns.get_loc(c) for c in CAT_FEATURES if c in X_train.columns]
model = CatBoostRegressor(
    iterations=500, depth=6, learning_rate=0.05,
    loss_function='RMSE', verbose=0
)
model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val))
preds = np.maximum(model.predict(X_test), 0)  # clip negatives
```

### Example: Compare CatBoost vs LGBM accuracy

```bash
curl -s "http://localhost:8000/forecast/accuracy/slice?lag=2&dim=model_id" \
  | jq '.rows[] | select(.model_id | test("catboost|lgbm")) | {model_id, accuracy_pct}'
# {"model_id": "catboost_global", "accuracy_pct": 92.1}
# {"model_id": "lgbm_global",     "accuracy_pct": 91.5}
```


---

## Additional Examples

#### Example — Categorical feature handling (CatBoost-specific)

```python
from catboost import CatBoostRegressor, Pool
from common.constants import CAT_FEATURES

# CatBoost requires column INDICES (not names) for cat_features
# cat_dtype='str' ensures columns are string dtype before index lookup
X_train = X_train.astype({c: str for c in CAT_FEATURES if c in X_train.columns})

cat_indices = [X_train.columns.get_loc(c) for c in CAT_FEATURES if c in X_train.columns]
# e.g. [3, 7, 12, 15] — positional indices for ml_cluster, region, brand, abc_vol

pool = Pool(X_train, y_train, cat_features=cat_indices)
model = CatBoostRegressor(iterations=500, depth=6, verbose=0)
model.fit(pool)
# CatBoost applies ordered target encoding internally — no one-hot encoding needed
```

#### Example — GPU auto-detection for CatBoost

```python
import subprocess, platform

def _detect_catboost_task_type() -> str:
    """Return 'GPU' if a CUDA-capable device is found, else 'CPU'."""
    if platform.system() == "Linux":
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            return "GPU"
    # macOS: CatBoost GPU via Metal is experimental; default to CPU
    return "CPU"

task_type = _detect_catboost_task_type()
model = CatBoostRegressor(task_type=task_type, iterations=500, verbose=0)
# Falls back to CPU silently if GPU driver unavailable
```

#### Example — Lag Strategy archive expansion for CatBoost

```sql
-- Verify CatBoost archive has all 5 lags
SELECT lag, COUNT(*) AS rows, ROUND(AVG(basefcst_pref), 2) AS avg_fcst
FROM backtest_lag_archive
WHERE model_id = 'catboost_global'
GROUP BY lag ORDER BY lag;
-- lag | rows   | avg_fcst
--   0 | 95621  |  1251.30
--   1 | 95621  |  1243.10
--   2 | 95621  |  1229.50
--   3 | 95621  |  1214.80
--   4 | 95621  |  1208.20
-- Each prediction is expanded to 5 rows in the archive (shared load script)
```

#### Example — Transfer learning for CatBoost

```python
from catboost import CatBoostRegressor

# Phase 1: global base model
base_model = CatBoostRegressor(iterations=500, depth=6, verbose=0)
base_model.fit(X_all, y_all, cat_features=cat_indices)

# Phase 2: per-cluster fine-tune (CatBoost uses init_model=regressor directly)
for cluster_id, (X_c, y_c) in cluster_data.items():
    if len(X_c) < transfer_min_rows:   # default: 20
        continue
    fine_tuned = CatBoostRegressor(iterations=100, depth=6, verbose=0)
    fine_tuned.fit(X_c, y_c, cat_features=cat_indices,
                   init_model=base_model)   # <-- regressor object, not booster_
    predictions[cluster_id] = fine_tuned.predict(X_test_c)
```


---

<!-- SOURCE: feature13.md (XGBoost) -->
# Feature 13: XGBoost Backtesting Implementation

## Objective
Implement the backtesting framework (Feature 8) with XGBoost models, supporting both global and per-cluster training strategies. Mirrors the LGBM implementation (Feature 9) with XGBoost-specific optimizations.

## Scope
- **Models**: XGBoost regressors for monthly demand forecasting
- **Strategies**: Global model (one XGBoost, `ml_cluster` as feature) and per-cluster (separate XGBoost per cluster)
- **Timeframes**: 10 expanding windows (A-J), auto-detected from data
- **Main table**: Each prediction stored at the DFU's `execution_lag` from `dim_dfu`
- **Archive table**: All lags 0-4 preserved in `backtest_lag_archive` for accuracy reporting at any horizon

## Model IDs

| Strategy | model_id | Description |
|----------|----------|-------------|
| Global | `xgboost_global` | One model for all DFUs, `ml_cluster` as feature |
| Per-cluster | `xgboost_cluster` | Separate model per `ml_cluster` |

## XGBoost-Specific Details

### Categorical Feature Handling
- XGBoost supports native categorical features via `enable_categorical=True` with `tree_method="hist"`
- Categorical columns stored as pandas `category` dtype
- No manual one-hot encoding required (XGBoost >= 2.0)

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 500 | Number of boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `min_child_weight` | 5 | Minimum sum of instance weight in a child |
| `subsample` | 0.8 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Column subsampling ratio per tree |
| `tree_method` | hist | Histogram-based tree method (enables categorical support) |
| `enable_categorical` | True | Native categorical feature support |
| `random_state` | 42 | Reproducibility |

### GPU Support
- XGBoost supports GPU training via `device="cuda"`
- Auto-detected at runtime; falls back to CPU if unavailable

## Feature Engineering

Identical to Feature 9 (LGBM) — all features are **strictly causal**.

### Lag Features (12)
- `qty_lag_1` through `qty_lag_12`: Historical demand shifted by N months

### Rolling Statistics (6)
- `rolling_mean_3m`, `rolling_mean_6m`, `rolling_mean_12m` (shifted by 1)
- `rolling_std_3m`, `rolling_std_6m`, `rolling_std_12m` (shifted by 1)

### Calendar Features (4)
- `month` (1-12), `quarter` (1-4), `month_sin`, `month_cos`

### DFU Attributes
- `ml_cluster` (categorical), `execution_lag`, `total_lt`, `region`, `brand`, `abc_vol`

### Item Attributes
- `case_weight`, `item_proof`, `bpc`

### Grid Construction
- Complete (DFU x month) grid ensures lag features work for zero-demand months
- Sales data masked at `train_end` cutoff to prevent future leakage

## Lag Strategy

### Main Table (`fact_external_forecast_monthly`)
Predictions stored **only at execution lag**:
- `fcstdate = startdate - execution_lag months`
- `lag = execution_lag` for every row

### Archive Table (`backtest_lag_archive`)
All lags 0-4 preserved:
- Same prediction expanded to 5 rows (lag 0, 1, 2, 3, 4)
- Includes `timeframe` column (A-J) for traceability
- Unique on `(forecast_ck, model_id, lag)`

## Implementation

### Scripts

| Script | Purpose |
|--------|---------|
| `mvp/demand/scripts/run_backtest_xgboost.py` | XGBoost-specific training functions + argparse (imports shared framework from `common/`) |
| `mvp/demand/common/backtest_framework.py` | Shared orchestrator: data loading, timeframes, feature engineering, output saving, MLflow |
| `mvp/demand/common/feature_engineering.py` | Shared feature matrix construction (uses `cat_dtype="category"` for XGBoost's native categoricals) |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Bulk load predictions into Postgres (main + archive) — shared with LGBM/CatBoost |

The script contains only three XGBoost-specific functions (`train_and_predict_global`, `train_and_predict_per_cluster`, `train_and_predict_transfer`) passed to `run_tree_backtest()` from the shared framework.

### run_backtest_xgboost.py
Parameters: `--cluster-strategy`, `--model-id`, `--n-timeframes`, `--output-dir`, `--n-estimators`, `--learning-rate`, `--max-depth`, `--min-child-weight`, `--subsample`, `--colsample-bytree`, `--verbosity`

Output:
- `backtest_predictions.csv`: Execution-lag only (for main table)
- `backtest_predictions_all_lags.csv`: All lags 0-4 (for archive)
- `backtest_metadata.json`, `feature_importance.csv`

### load_backtest_forecasts.py
Shared loader — same script used for LGBM, CatBoost, and XGBoost. Model-agnostic; operates on `model_id` column in CSV.

## Makefile Targets

```makefile
backtest-xgboost:          # Global XGBoost backtest
backtest-xgboost-cluster:  # Per-cluster XGBoost backtest
backtest-load:             # Load predictions into Postgres (main + archive) — shared
backtest-all:              # backtest-lgbm + backtest-load (unchanged)
```

## Schema

No schema changes required. Uses existing:
- `fact_external_forecast_monthly` with `model_id` support (Feature 6)
- `backtest_lag_archive` (Feature 8/9)

## Verification

```bash
cd mvp/demand && uv sync          # Install dependencies (includes xgboost)
make db-apply-sql                  # Ensure tables exist
make backtest-xgboost              # Run global backtest
make backtest-load                 # Load main + archive
curl "http://localhost:8000/domains/forecast/models"
curl "http://localhost:8000/domains/forecast/analytics?model=xgboost_global"
make backtest-xgboost-cluster      # Per-cluster backtest
make backtest-load                 # Reload
```

## Dependencies
- Feature 8 (backtesting framework)
- Feature 7 (clustering)
- Feature 4 (fact tables)
- xgboost >= 2.0.0, python-dateutil >= 2.8.0

---

## Implementation Details

### Additional Model ID
- Transfer strategy: `xgboost_transfer` — global base model fine-tuned per cluster

### Missing Default Hyperparameters
- `n_jobs: -1` (all CPU cores)
- `verbosity: 0` (silent)

### Additional CLI Parameters
- `--transfer-n-estimators` (default: 100) — additional trees for fine-tuning
- `--transfer-min-rows` (default: 20) — minimum cluster rows

### Makefile Target
- `backtest-xgboost-transfer`

### Shared Module Dependencies
- `common/constants.py`: `MIN_CLUSTER_ROWS` (50), `CAT_FEATURES`, `LAG_RANGE`
- `common/metrics.py`: `compute_accuracy_metrics()`
- `common/mlflow_utils.py`: `log_backtest_run()`
- `common/db.py`: `get_db_params()`

### Framework Integration
- `model_params_key="xgboost_params"`, `model_type_tag="xgboost_backtest"`, `cat_dtype="category"`
- Prediction clipping: `np.maximum(preds, 0)` — demand cannot be negative


---

## Examples

### Example: Run XGBoost backtest

```bash
make backtest-xgboost
# XGBRegressor(tree_method='hist', enable_categorical=True, n_estimators=500)
# cat_dtype='category' — pandas Category dtype for native XGBoost categorical support
# Output: data/backtest_xgboost_global.csv
make backtest-load  # load as model_id='xgboost_global'
```

### Example: XGBoost training function

```python
from xgboost import XGBRegressor
import numpy as np

# cat_dtype='category' needed for enable_categorical=True
model = XGBRegressor(
    tree_method='hist',
    enable_categorical=True,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    device='cuda' if gpu_available else 'cpu',
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
preds = np.maximum(model.predict(X_test), 0)
```

### Example: Per-cluster XGBoost

```bash
make backtest-xgboost-cluster
# Trains separate XGBoost model per cluster (7 clusters)
# Each model only sees DFUs from its cluster → better fit for cluster-specific patterns
make backtest-load
```


---

## Additional Examples

#### Example — Categorical feature handling (XGBoost-specific)

```python
from xgboost import XGBRegressor
from common.constants import CAT_FEATURES
import pandas as pd

# cat_dtype='category' — XGBoost >= 2.0 reads pandas Category dtype natively
# No manual encoding needed when tree_method='hist' + enable_categorical=True
X_train = X_train.copy()
for col in CAT_FEATURES:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype("category")

model = XGBRegressor(
    tree_method="hist",
    enable_categorical=True,
    n_estimators=500,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
# XGBoost handles category dtype directly — no cat_features index list needed
```

#### Example — GPU auto-detection for XGBoost

```python
import subprocess, platform

def _detect_xgboost_device() -> str:
    """Return 'cuda' if NVIDIA GPU available, else 'cpu'."""
    if platform.system() == "Linux":
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            return "cuda"
    # macOS: XGBoost Metal support is experimental; default to cpu
    return "cpu"

device = _detect_xgboost_device()
model = XGBRegressor(tree_method="hist", enable_categorical=True,
                     device=device, verbosity=0)
```

#### Example — Lag Strategy archive verification for XGBoost

```sql
-- Confirm all 5 lags are stored for xgboost_global
SELECT lag, COUNT(*) AS rows
FROM backtest_lag_archive
WHERE model_id = 'xgboost_global'
GROUP BY lag ORDER BY lag;
-- lag | rows
--   0 | 95621
--   1 | 95621
--   2 | 95621
--   3 | 95621
--   4 | 95621

-- Main table has only execution-lag rows
SELECT COUNT(*) FROM fact_external_forecast_monthly
WHERE model_id = 'xgboost_global';
-- 95621  (one row per DFU-month at its execution_lag)
```

#### Example — Three model strategies comparison

```bash
make backtest-xgboost          # model_id=xgboost_global
make backtest-xgboost-cluster  # model_id=xgboost_cluster
make backtest-xgboost-transfer # model_id=xgboost_transfer
make backtest-load             # load all three into Postgres

# Compare accuracy across strategies
curl -s "http://localhost:8000/forecast/accuracy/slice?group_by=model_id&models=xgboost_global,xgboost_cluster,xgboost_transfer&lag=2" \
  | jq '.rows[] | {model_id, accuracy_pct, wape}'
# {"model_id": "xgboost_transfer", "accuracy_pct": 93.1, "wape": 6.9}
# {"model_id": "xgboost_cluster",  "accuracy_pct": 93.0, "wape": 7.0}
# {"model_id": "xgboost_global",   "accuracy_pct": 91.8, "wape": 8.2}
```
