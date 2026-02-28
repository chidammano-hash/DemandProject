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
