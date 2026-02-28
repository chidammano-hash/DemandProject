# Feature 25 — NeuralProphet Backtesting Implementation

## Overview

NeuralProphet is a PyTorch-based successor to Facebook Prophet that supports GPU acceleration (Apple MPS, NVIDIA CUDA) while maintaining a Prophet-compatible API. Like Prophet, it fits per-DFU individual time series models with native trend and seasonality decomposition, but uses neural network components for potentially better non-linear pattern capture. It is ~2-3x faster than Prophet and supports hardware acceleration.

## Problem

Prophet is CPU-only (Stan backend) and cannot leverage the Apple MPS GPU already available in the project's MacBook environment. NeuralProphet replaces Stan with PyTorch, enabling:

1. **GPU acceleration** — Apple MPS support for faster per-DFU fitting
2. **Neural components** — LSTM-based autoregressive features and neural network trend
3. **Modern optimizer** — AdamW with learning rate scheduling instead of Stan's L-BFGS
4. **Global model option** — Can train one model across all DFUs (vs Prophet's per-DFU only)

#### Example

```python
# Verify PyTorch device detection before running backtest
import torch

if torch.backends.mps.is_available():
    print("Apple MPS GPU available — workers use CPU via PYTORCH_MPS_FORCE_CPU=1")
elif torch.cuda.is_available():
    print(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected — using CPU only")
```

## Architecture

### Why NeuralProphet Differs from Prophet

| Aspect | Prophet | NeuralProphet |
|--------|---------|---------------|
| Backend | Stan/C++ (CPU only) | PyTorch (CPU/MPS/CUDA) |
| GPU support | No | Yes (Apple MPS, NVIDIA CUDA) |
| Trend model | Piecewise linear/logistic | Neural network linear/discontinuous |
| Seasonality | Fourier series | Fourier series (same) |
| Autoregression | Not supported | `n_lags` parameter for AR features |
| Training | L-BFGS optimization | AdamW + learning rate scheduling |
| Speed (per DFU) | ~0.5-2s (Stan compile overhead) | ~0.2-0.5s (PyTorch) |
| Speed (276K × 10 TFs) | ~9-23 hours | ~4-8 hours |
| Data format | (ds, y) per DFU | (ds, y) per DFU (compatible) |

### Data Flow

```
fact_sales_monthly + dim_dfu
          ↓
  For each timeframe (A-J):
    ┌────────────────────────┐
    │ Mask future sales      │
    │ (same causality logic) │
    └──────────┬─────────────┘
               ↓
    For each DFU (parallel via Pool):
      ┌──────────────────────┐
      │ Extract (ds, y) pair │
      │ from sales history   │
      │       ↓              │
      │ Fit NeuralProphet    │
      │ (PyTorch + MPS GPU)  │
      │       ↓              │
      │ Predict future months│
      └──────────┬───────────┘
               ↓
    Collect all DFU predictions
          ↓
  Assign execution_lag, forecast_ck
          ↓
  Deduplicate across timeframes
          ↓
  Output: backtest_predictions.csv
          backtest_predictions_all_lags.csv
```

## Model IDs

| Strategy | model_id | Description |
|----------|----------|-------------|
| Global | `neuralprophet_global` | Per-DFU independent fits (all DFUs) |
| Per-cluster | `neuralprophet_cluster` | Per-DFU fits, clustered DFUs only |
| Pooled | `neuralprophet_pooled` | Aggregate by cluster → fit → disaggregate |

## NeuralProphet Configuration

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `growth` | `linear` | Trend model: linear, discontinuous, or off |
| `yearly_seasonality` | `auto` | Auto-detect yearly seasonality |
| `weekly_seasonality` | `False` | Disabled (monthly grain) |
| `daily_seasonality` | `False` | Disabled (monthly grain) |
| `epochs` | `100` | Training epochs per model |
| `learning_rate` | `0.1` | AdamW learning rate |
| `batch_size` | `64` | Training batch size |
| `n_lags` | `0` | AR lags (0 = pure decomposition like Prophet) |
| `accelerator` | `auto` | Hardware: auto, cpu, gpu, mps |

#### Example

```python
from neuralprophet import NeuralProphet
import pandas as pd

# Minimal single-DFU fit for item 100320 at loc 1401-BULK
train_df = pd.DataFrame({
    "ds": pd.date_range("2022-01-01", periods=36, freq="MS"),
    "y": [120, 95, 110, 130, 140, 155, 160, 148, 135, 125, 115, 145,
          130, 100, 118, 138, 148, 163, 168, 155, 142, 132, 122, 153,
          128,  98, 115, 135, 146, 161, 165, 152, 140, 130, 120, 150],
})

model = NeuralProphet(
    growth="linear",
    yearly_seasonality="auto",
    weekly_seasonality=False,
    daily_seasonality=False,
    epochs=100,
    learning_rate=0.1,
    batch_size=64,
    n_lags=0,
    trainer_config={"accelerator": "cpu"},  # always CPU in workers
)
model.fit(train_df, freq="MS")
future = model.make_future_dataframe(train_df, periods=5)
forecast = model.predict(future)

# NeuralProphet uses yhat1 (not yhat like Prophet)
yhat_col = "yhat1" if "yhat1" in forecast.columns else "yhat"
preds = forecast[["ds", yhat_col]].tail(5)
preds[yhat_col] = preds[yhat_col].clip(lower=0)  # floor at zero
print(preds)
# ds       yhat1
# 2025-01-01  147.3
# 2025-02-01  102.1
```

## GPU / Device Support

| Device | Detection | Performance |
|--------|-----------|-------------|
| Apple MPS | `torch.backends.mps.is_available()` | ~2-3x faster than CPU |
| NVIDIA CUDA | `torch.cuda.is_available()` | ~5-10x faster than CPU |
| CPU | Fallback | Baseline (still faster than Prophet/Stan) |

NeuralProphet uses PyTorch Lightning's `accelerator` parameter for device selection. The `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable is set in workers to handle any unsupported MPS operations.

### Worker Device Override

Despite the `--accelerator` CLI flag accepting `auto`, `cpu`, `gpu`, or `mps`, the actual implementation **forces CPU in all spawned worker processes**. This is because:

1. **MPS hangs in spawned subprocesses** -- PyTorch's MPS backend does not work reliably in `spawn`-context child processes
2. **Per-DFU series are tiny** -- GPU overhead exceeds any acceleration benefit for small time series

Workers set these environment variables:
- `PYTORCH_MPS_FORCE_CPU=1`
- `CUDA_VISIBLE_DEVICES=""`
- `trainer_config={"accelerator": "cpu"}` is passed to NeuralProphet constructor

The `--accelerator` flag is preserved in kwargs and metadata but not used for actual training in the current multiprocessing implementation.

#### Example

```python
import os
import multiprocessing

def worker_init():
    """Called once per worker process at pool startup."""
    os.environ["PYTORCH_MPS_FORCE_CPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    import logging
    for logger_name in ("neuralprophet", "pytorch_lightning", "lightning"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

ctx = multiprocessing.get_context("spawn")  # Required on macOS (not fork)
with ctx.Pool(processes=4, initializer=worker_init) as pool:
    results = list(pool.imap_unordered(fit_one_dfu, dfu_tasks, chunksize=10))
```

## CLI Interface

```bash
uv run python scripts/run_backtest_neuralprophet.py \
    --cluster-strategy global \
    --epochs 100 \
    --learning-rate 0.1 \
    --growth linear \
    --accelerator auto \
    --n-workers 4 \
    --n-timeframes 10
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--cluster-strategy` | `global` | Strategy: global, per_cluster, or pooled |
| `--model-id` | auto | Override model_id |
| `--epochs` | `100` | Training epochs |
| `--learning-rate` | `0.1` | AdamW learning rate |
| `--batch-size` | `64` | Training batch size |
| `--n-lags` | `0` | Autoregressive lags (0 = decomposition only) |
| `--yearly-seasonality` | `auto` | Yearly seasonality (accepts: `auto`, `True`/`False`, or integer Fourier order) |
| `--weekly-seasonality` | `False` | Enable weekly seasonality (boolean flag, `store_true`) |
| `--daily-seasonality` | `False` | Enable daily seasonality (boolean flag, `store_true`) |
| `--growth` | `linear` | Trend: linear, discontinuous, off |
| `--accelerator` | `auto` | Device: auto, cpu, gpu, mps (note: workers forced to CPU -- see Worker Device Override) |
| `--n-workers` | all CPUs | Parallel workers |
| `--n-timeframes` | `10` | Expanding windows |
| `--output-dir` | `data/backtest` | Output directory |

#### Example — Running the Global Backtest

```bash
cd mvp/demand

# Run global NeuralProphet backtest (10 timeframes, all DFUs)
make backtest-neuralprophet

# Sample console output:
# [NeuralProphet Global] Strategy: global | Model ID: neuralprophet_global
# [NeuralProphet Global] Workers: 8 | Timeframes: 10
# [TF A] Training 4821 DFUs...
# [TF A] Timeframe A complete: 4821 predictions
# ...
# [NeuralProphet Global] All timeframes complete in 4h 23m
# Saved: data/backtest/backtest_predictions.csv (48,210 rows)
# WAPE: 28.4% | Accuracy: 71.6% | Bias: -0.03

# Per-cluster strategy
make backtest-neuralprophet-cluster

# Pooled cluster strategy
make backtest-neuralprophet-pooled

# Load predictions into Postgres
make backtest-load
```

## Cluster Strategies

| Strategy | Model ID | Behavior |
|----------|----------|----------|
| `global` | `neuralprophet_global` | Fit independent NeuralProphet per DFU (all DFUs) |
| `per_cluster` | `neuralprophet_cluster` | Same but only DFUs with cluster assignment |
| `pooled` | `neuralprophet_pooled` | Aggregate sales by cluster, fit per cluster, disaggregate proportionally |

## Parallelization Strategy

NeuralProphet fits are parallelized via Python `multiprocessing.Pool`:

- Default: all CPU cores
- Each worker fits one DFU at a time
- Progress reported every 10,000 DFUs
- `imap_unordered` for maximum throughput (`chunksize=10` for per-DFU, `chunksize=2` for pooled cluster fits)
- Worker initialization suppresses PyTorch/Lightning logging (`neuralprophet`, `pytorch_lightning`, `lightning` loggers set to WARNING)
- **Spawn context required:** Uses `multiprocessing.get_context("spawn")` instead of default `fork()` because PyTorch MPS backend crashes with `fork()` on macOS (`+[MPSGraphObject initialize] may have been in progress in another thread when fork() was called`)
- **Workers forced to CPU:** All spawned workers set `PYTORCH_MPS_FORCE_CPU=1`, `CUDA_VISIBLE_DEVICES=""`, and pass `trainer_config={"accelerator": "cpu"}` -- MPS hangs in spawned subprocesses and adds overhead for tiny per-DFU time series
- Single-worker fallback: when `actual_workers <= 1`, runs sequentially without multiprocessing pool

## Prediction Details

### NeuralProphet Output Column

NeuralProphet uses `yhat1` as the forecast column name (not `yhat` like Prophet). The implementation checks for `yhat1` first with fallback to `yhat`:
```python
yhat_col = "yhat1" if "yhat1" in forecast.columns else "yhat"
```

### Prediction Flooring

All predictions are floored to zero: `max(float(row[yhat_col]), 0)`. Negative forecasts are replaced with 0.

### Minimum Training Data

DFUs with fewer than 2 historical data points (`len(train_series) < 2`) skip NeuralProphet fitting entirely and emit zero forecasts for all predict months.

### Error Handling

Failed DFU fits emit zero forecasts (not skipped) to maintain complete coverage. Only the first error per worker is logged (via `_logged_error` attribute on the fitting function) to avoid log flooding.

#### Example — Zero Fallback for Failed/Short DFUs

```python
def fit_one_dfu(args):
    dfu_ck, train_series, predict_months, np_kwargs = args
    if len(train_series) < 2:
        # Insufficient history — emit zero forecasts (not skipped)
        return [(dfu_ck, month, 0.0) for month in predict_months]
    try:
        df = pd.DataFrame({"ds": train_series.index, "y": train_series.values})
        model = NeuralProphet(**np_kwargs)
        model.fit(df, freq="MS")
        future = model.make_future_dataframe(df, periods=len(predict_months))
        forecast = model.predict(future)
        yhat_col = "yhat1" if "yhat1" in forecast.columns else "yhat"
        preds = forecast.tail(len(predict_months))
        return [(dfu_ck, row["ds"], max(float(row[yhat_col]), 0))
                for _, row in preds.iterrows()]
    except Exception as e:
        if not getattr(fit_one_dfu, "_logged_error", False):
            print(f"[WARN] DFU {dfu_ck} failed: {e}")
            fit_one_dfu._logged_error = True
        return [(dfu_ck, month, 0.0) for month in predict_months]
```

## Output Format (Identical to All Models)

**Main CSV (`backtest_predictions.csv`):**
```
forecast_ck,dmdunit,dmdgroup,loc,fcstdate,startdate,lag,execution_lag,basefcst_pref,tothist_dmd,model_id
```

**Archive CSV (`backtest_predictions_all_lags.csv`):** Same + `timeframe` column, expanded to lags 0-4.

**Metadata JSON (`backtest_metadata.json`):** Model ID, strategy, timeframes, neuralprophet_kwargs, n_workers, date range, and accuracy metrics (WAPE, bias, accuracy_pct at execution lag).

#### Example — Output Files

```bash
ls -lh data/backtest/
# -rw-r--r--  18M  data/backtest/backtest_predictions.csv
# -rw-r--r--  91M  data/backtest/backtest_predictions_all_lags.csv
# -rw-r--r--  2.1K data/backtest/backtest_metadata.json

cat data/backtest/backtest_metadata.json
```

```json
{
  "model_id": "neuralprophet_global",
  "strategy": "global",
  "n_timeframes": 10,
  "n_workers": 8,
  "neuralprophet_kwargs": {
    "epochs": 100,
    "learning_rate": 0.1,
    "batch_size": 64,
    "growth": "linear",
    "n_lags": 0,
    "yearly_seasonality": "auto",
    "weekly_seasonality": false,
    "daily_seasonality": false
  },
  "date_range": {"train_start": "2022-01-01", "train_end": "2024-12-01"},
  "metrics": {
    "wape": 28.4,
    "bias": -0.03,
    "accuracy_pct": 71.6,
    "n_rows": 48210
  }
}
```

```bash
head -3 data/backtest/backtest_predictions.csv
# forecast_ck,dmdunit,dmdgroup,loc,fcstdate,startdate,lag,execution_lag,basefcst_pref,tothist_dmd,model_id
# 100320|A|1401-BULK|2025-01-01|2025-01-01,100320,A,1401-BULK,2025-01-01,2025-01-01,0,0,147.3,1820.0,neuralprophet_global
```

## Shared Infrastructure Reuse

| Component | Source | Reuse |
|-----------|--------|-------|
| `generate_timeframes()` | `common/backtest_framework.py` | Identical expanding windows |
| `load_backtest_data()` | `common/backtest_framework.py` | Same Postgres queries (called with `include_item_attrs=False`) |
| `postprocess_predictions()` | `common/backtest_framework.py` | Same dedup + lag expansion |
| `save_backtest_output()` | `common/backtest_framework.py` | Same CSV + metadata output |
| `compute_accuracy_metrics()` | `common/metrics.py` | WAPE/bias/accuracy at execution lag |
| `log_backtest_run()` | `common/mlflow_utils.py` | Same MLflow logging |
| `get_db_params()` | `common/db.py` | Shared DB connection parameters |
| `load_backtest_forecasts.py` | Shared loader | Same COPY+upsert pattern |

**Note:** Unlike tree-based models (LGBM, CatBoost, XGBoost), NeuralProphet does NOT use `run_tree_backtest()` from the shared framework. It orchestrates its own per-DFU fitting loop (like Prophet and StatsForecast) but reuses the shared data loading, postprocessing, and output functions.

## NeuralProphet vs Prophet: When to Use Each

| Scenario | Recommended |
|----------|-------------|
| Large-scale (100K+ DFUs) | NeuralProphet (faster) or StatsForecast (fastest) |
| GPU available (MPS/CUDA) | NeuralProphet |
| Complex non-linear patterns | NeuralProphet (neural components) |
| Minimal dependencies | Prophet (simpler install) |
| Interpretability priority | Prophet (mature decomposition plots) |
| Autoregressive features needed | NeuralProphet (`n_lags` parameter) |

## Makefile Targets

```makefile
backtest-neuralprophet:
	$(UV) python scripts/run_backtest_neuralprophet.py --cluster-strategy global
backtest-neuralprophet-cluster:
	$(UV) python scripts/run_backtest_neuralprophet.py --cluster-strategy per_cluster
backtest-neuralprophet-pooled:
	$(UV) python scripts/run_backtest_neuralprophet.py --cluster-strategy pooled
```

## Model Competition Integration

```yaml
competition:
  models:
  - neuralprophet_global    # new
  - neuralprophet_cluster   # new
```

## Dependencies

Add to `mvp/demand/pyproject.toml`:
```toml
"neuralprophet>=0.9.0",
```

NeuralProphet depends on PyTorch (already installed for PatchTST/DeepAR) and PyTorch Lightning.

#### Example — Install and Verify

```bash
cd mvp/demand
uv sync   # picks up neuralprophet from pyproject.toml

python -c "import neuralprophet; print('NeuralProphet', neuralprophet.__version__)"
# NeuralProphet 0.9.1

# Quick smoke test: fit 24-month series in ~5 epochs
python -c "
from neuralprophet import NeuralProphet
import pandas as pd
m = NeuralProphet(epochs=5, trainer_config={'accelerator': 'cpu'})
df = pd.DataFrame({'ds': pd.date_range('2023-01-01', periods=24, freq='MS'), 'y': range(24)})
m.fit(df, freq='MS')
print('NeuralProphet smoke test passed')
"
```

## Key Files

| File | Purpose |
|------|---------|
| `mvp/demand/scripts/run_backtest_neuralprophet.py` | NeuralProphet backtest script |
| `mvp/demand/common/backtest_framework.py` | Shared backtest orchestration (timeframes, data loading, postprocessing, output) |
| `mvp/demand/common/mlflow_utils.py` | Shared MLflow logging |
| `mvp/demand/common/metrics.py` | Shared accuracy metrics (WAPE, bias, accuracy) |
| `mvp/demand/common/db.py` | Shared DB connection parameters |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Shared loader (unchanged) |
| `mvp/demand/config/model_competition.yaml` | Add neuralprophet models |
| `mvp/demand/pyproject.toml` | Add `neuralprophet>=0.9.0` dependency |
| `mvp/demand/Makefile` | Add 3 new targets |

## Verification

```bash
cd mvp/demand
uv sync                                              # Install neuralprophet
python -c "import neuralprophet; print(neuralprophet.__version__)"

make backtest-neuralprophet                           # Global backtest (~4-8 hours)
make backtest-load                                    # Load into Postgres
curl "http://localhost:8000/domains/forecast/models"  # Verify neuralprophet_global

make backtest-neuralprophet-cluster                   # Per-cluster
make backtest-neuralprophet-pooled                    # Pooled
make backtest-load                                    # Reload
make champion-select                                  # Include in competition
```

## Dependencies on Other Features

| Feature | Dependency |
|---------|------------|
| Feature 8 | Backtesting framework (expanding windows) |
| Feature 7 | Clustering (for per_cluster and pooled strategies) |
| Feature 4 | Fact tables (sales data source) |
| Feature 15 | Champion selection (model competition) |
| Feature 19 | PyTorch already in dependencies (for PatchTST) |


#### Example — Pooled Cluster Strategy

```bash
# Pooled strategy: aggregates sales by cluster, fits one NeuralProphet model per cluster,
# then disaggregates predictions back to individual DFUs proportionally.

cd mvp/demand
make backtest-neuralprophet-pooled

# Sample console output:
# [NeuralProphet Pooled] Strategy: pooled | Model ID: neuralprophet_pooled
# [NeuralProphet Pooled] Workers: 8 | Timeframes: 10
# [TF A] Pooling sales for 6 clusters...
# [TF A] Fitting NeuralProphet for cluster high_volume_steady (1203 DFUs)...
# [TF A] Fitting NeuralProphet for cluster medium_volume_seasonal (892 DFUs)...
# [TF A] Disaggregating predictions → 4,821 DFU rows
# [TF A] Timeframe A complete: 4821 predictions
# ...
# [NeuralProphet Pooled] All timeframes complete in 1h 12m
# Saved: data/backtest/backtest_predictions.csv (48,210 rows)
# WAPE: 31.2% | Accuracy: 68.8% | Bias: -0.05
# Note: pooled WAPE higher than global (31.2% vs 28.4%) — aggregation loses DFU-level nuance

# The disaggregation step: each DFU gets a share of its cluster's pooled forecast
# proportional to its historical share of cluster total demand:
#   dfu_forecast = cluster_forecast * (dfu_hist_avg / cluster_hist_avg)
```

```python
# Disaggregation logic in run_backtest_neuralprophet.py (pooled strategy):
# 1. Compute each DFU's share of cluster total demand (historical mean)
# 2. Fit one NeuralProphet per cluster on aggregated demand
# 3. Multiply cluster forecast by each DFU's share ratio

# cluster_sums = sales.groupby(['cluster', 'startdate'])['qty'].sum()
# dfu_shares   = sales.groupby(['dfu_ck'])['qty'].mean() / cluster_avgs
# dfu_forecast = cluster_forecast[cluster] * dfu_shares[dfu_ck]
```
