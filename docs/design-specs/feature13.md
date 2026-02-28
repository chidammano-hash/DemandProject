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
