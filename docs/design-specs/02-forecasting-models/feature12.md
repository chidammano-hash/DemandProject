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
