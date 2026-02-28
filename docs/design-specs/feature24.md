# Feature 24 — StatsForecast Backtesting Implementation

## Overview

StatsForecast (Nixtla) provides vectorized statistical models that process ALL time series as a single DataFrame — no per-DFU fitting loop needed. This makes it ~100x faster than Prophet for large-scale backtesting (276K+ DFUs). Uses AutoARIMA and AutoETS as the primary models, with Numba JIT compilation for native-speed execution.

## Problem

Prophet takes 9-23 hours for 276K DFUs because it compiles and fits one Stan model per DFU sequentially. The pooled strategy is faster but loses DFU-level accuracy by disaggregating proportionally. StatsForecast solves this by fitting all series in a single vectorized batch call.

## Architecture

### Why StatsForecast Differs from Prophet

| Aspect | Prophet | StatsForecast |
|--------|---------|---------------|
| Approach | Per-DFU Stan model (sequential) | Batch vectorized (all DFUs at once) |
| Backend | Stan/C++ (CPU only) | Numba JIT (CPU, compiled to machine code) |
| GPU support | No | No (but doesn't need it — already fast) |
| Speed (276K DFUs × 10 TFs) | ~9-23 hours | ~15-30 minutes |
| Parallelism | multiprocessing.Pool | Built-in `n_jobs=-1` |
| Models | Prophet decomposition | AutoARIMA, AutoETS, SeasonalNaive |
| Data format | (ds, y) per DFU | (unique_id, ds, y) all DFUs |
| Seasonality | Fourier series decomposition | Automatic detection via AIC/BIC |

### Data Flow

```
fact_sales_monthly + dim_dfu
          ↓
  Reshape to StatsForecast format:
    unique_id = dfu_ck
    ds = startdate
    y = qty
          ↓
  For each timeframe (A-J):
    ┌──────────────────────────┐
    │ Mask future sales        │
    │ (same causality logic)   │
    └──────────┬───────────────┘
               ↓
    StatsForecast.forecast(df, h=N)
    → fits AutoARIMA + AutoETS per series
    → returns predictions for all DFUs
               ↓
    Collect all predictions
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
| Global | `statsforecast_global` | Fit all DFUs at once (batch vectorized) |
| Per-cluster | `statsforecast_cluster` | Fit only DFUs within assigned clusters |
| Pooled | `statsforecast_pooled` | Aggregate by cluster → fit → disaggregate proportionally |

## StatsForecast Models

| Model | Description | Best For |
|-------|-------------|----------|
| AutoARIMA | Automatic ARIMA order selection via AIC | Trend + seasonality, stationary series |
| AutoETS | Automatic Exponential Smoothing selection | Strong seasonality, level/trend/season decomposition |
| SeasonalNaive | Last season's value as forecast (baseline) | Benchmark comparison |

Default configuration: `AutoARIMA(season_length=12)` + `AutoETS(season_length=12)`. AutoARIMA is used as the primary prediction; AutoETS serves as fallback.

## CLI Interface

```bash
uv run python scripts/run_backtest_statsforecast.py \
    --cluster-strategy global \
    --models AutoARIMA,AutoETS \
    --season-length 12 \
    --n-jobs -1 \
    --n-timeframes 10
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--cluster-strategy` | `global` | Strategy: global, per_cluster, or pooled |
| `--model-id` | auto | Override model_id |
| `--models` | `AutoARIMA,AutoETS` | Comma-separated model names |
| `--season-length` | `12` | Seasonal period (12 for monthly) |
| `--n-jobs` | `-1` | Parallel jobs (-1 = all CPUs) |
| `--n-timeframes` | `10` | Number of expanding windows |
| `--output-dir` | `data/backtest` | Output directory |

## Cluster Strategies

| Strategy | Model ID | Behavior |
|----------|----------|----------|
| `global` | `statsforecast_global` | Fit all DFUs in single batch call |
| `per_cluster` | `statsforecast_cluster` | Filter to clustered DFUs, fit in single batch |
| `pooled` | `statsforecast_pooled` | Aggregate sales by cluster, fit one model per cluster, disaggregate to DFU level proportionally |

## Output Format (Identical to All Models)

**Main CSV (`backtest_predictions.csv`):**
```
forecast_ck,dmdunit,dmdgroup,loc,fcstdate,startdate,lag,execution_lag,basefcst_pref,tothist_dmd,model_id
```

**Archive CSV (`backtest_predictions_all_lags.csv`):** Same + `timeframe` column, expanded to lags 0-4.

**Metadata JSON (`backtest_metadata.json`):** Model ID, strategy, timeframes, statsforecast_kwargs, accuracy metrics.

## Shared Infrastructure Reuse

| Component | Source | Reuse |
|-----------|--------|-------|
| `generate_timeframes()` | `common/backtest_framework.py` | Identical expanding windows |
| `load_backtest_data()` | `common/backtest_framework.py` | Same Postgres queries |
| `postprocess_predictions()` | `common/backtest_framework.py` | Same dedup + lag expansion |
| `save_backtest_output()` | `common/backtest_framework.py` | Same CSV + metadata output |
| `log_backtest_run()` | `common/mlflow_utils.py` | Same MLflow logging |
| `load_backtest_forecasts.py` | Shared loader | Same COPY+upsert pattern |

## Performance Comparison

| Model | ~276K DFUs × 10 TFs | Parallelism | GPU |
|-------|---------------------|-------------|-----|
| LGBM global | ~2 minutes | Single model | No |
| CatBoost global | ~3 minutes | Single model | No |
| XGBoost global | ~2 minutes | Single model | No |
| Prophet global | ~9-23 hours | 4 workers | No |
| **StatsForecast global** | **~15-30 min** | **All CPUs (Numba)** | **No** |
| NeuralProphet global | ~4-8 hours | 4 workers | Yes (MPS) |

## Makefile Targets

```makefile
backtest-statsforecast:
	$(UV) python scripts/run_backtest_statsforecast.py --cluster-strategy global
backtest-statsforecast-cluster:
	$(UV) python scripts/run_backtest_statsforecast.py --cluster-strategy per_cluster
backtest-statsforecast-pooled:
	$(UV) python scripts/run_backtest_statsforecast.py --cluster-strategy pooled
```

## Model Competition Integration

```yaml
competition:
  models:
  - statsforecast_global    # new
  - statsforecast_cluster   # new
```

## Dependencies

Add to `mvp/demand/pyproject.toml`:
```toml
"statsforecast>=2.0.0",
```

StatsForecast depends on `numba` (JIT compiler) and `scipy`. Both are installed automatically.

## Key Files

| File | Purpose |
|------|---------|
| `mvp/demand/scripts/run_backtest_statsforecast.py` | StatsForecast backtest script (new) |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Shared loader (unchanged) |
| `mvp/demand/config/model_competition.yaml` | Add statsforecast models |
| `mvp/demand/pyproject.toml` | Add `statsforecast>=2.0.0` dependency |
| `mvp/demand/Makefile` | Add 3 new targets |

## Verification

```bash
cd mvp/demand
uv sync                                              # Install statsforecast

make backtest-statsforecast                           # Global backtest (~15-30 min)
make backtest-load                                    # Load into Postgres
curl "http://localhost:8000/domains/forecast/models"  # Verify statsforecast_global

make backtest-statsforecast-cluster                   # Per-cluster
make backtest-statsforecast-pooled                    # Pooled
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

---

## Implementation Details

### Batch Fitting Details
- Minimum observation threshold: series with < 3 observations filtered out
- Zero-prediction fill: filtered DFUs get zero predictions added back (ensuring all DFUs have output rows)
- Model column selection priority: AutoARIMA → AutoETS → SeasonalNaive → first available
- Exception handling: batch fitting errors caught, return empty DataFrame with warning

### Pooled Strategy
- `__unknown__` clusters excluded
- Clusters with < 3 observations filtered out
- Proportion: `np.where` for safe division (zero cluster total yields zero)

### Additional Shared Modules
- `compute_accuracy_metrics()` from `common/metrics.py`
- `OUTPUT_COLS`, `ARCHIVE_COLS`, `MAX_ARCHIVE_LAG` (4) from `common/constants.py`
- `timeframe_idx` column added per prediction DataFrame

### MLflow
- `model_type`: `"statsforecast_backtest"`

### Model Competition
- Only `statsforecast_global` in actual `model_competition.yaml` (not `statsforecast_cluster`)

### Logging Suppression
- `logging.disable(logging.INFO)` and `warnings.filterwarnings("ignore")`
