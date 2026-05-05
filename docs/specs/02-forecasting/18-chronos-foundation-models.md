# Chronos Foundation Models — Backtest Variants

> Four zero-shot Amazon Chronos variants (T5 ~46M, Bolt ~205M, Chronos 2 ~821M, Chronos 2 Enriched with 31 covariates) integrated into the expanding-window backtest framework — no training required, producing competing forecasts that participate in champion selection.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `scripts/run_backtest_chronos.py`, `scripts/run_backtest_bolt.py`, `scripts/run_backtest_chronos2.py`, `adv_algorithm_testing/foundation_models.py`, `config/forecasting/forecast_pipeline_config.yaml` |

---

## 1. Overview

The platform supports **four Amazon Chronos foundation model variants** for demand forecasting, each representing a different generation, architecture, and capability level. All are zero-shot pretrained models — they require no training on our data and produce forecasts directly from raw historical demand.

| Variant | Model ID | HuggingFace ID | Params | Architecture | Covariates | Make Target |
|---|---|---|---|---|---|---|
| Chronos T5 | `chronos` | `amazon/chronos-t5-small` | ~46M | T5 decoder + tokenization | None | `make backtest-chronos` |
| Chronos Bolt | `chronos_bolt` | `amazon/chronos-bolt-base` | ~205M | Native encoder (v2) | None | `make backtest-bolt` |
| Chronos 2 | `chronos2` | `amazon/chronos-2` | ~821M | Latest generation | None | `make backtest-chronos2` |
| Chronos 2 Enriched | `chronos2_enriched` | `amazon/chronos-2` | ~821M | Latest generation | 31 covariates | `make backtest-chronos2e` |

All variants share:
- Zero-shot inference (no per-dataset training)
- Expanding-window backtest with 10 timeframes (same as tree models)
- Execution-lag assignment and all-lag archive (lags 0-4)
- Checkpoint/resume support via `BacktestCheckpointer`
- Apple MPS GPU acceleration (auto-detected)
- Pipeline caching (model loaded once, reused across timeframes)

---

## 2. Chronos T5 (`chronos`)

### What It Is
The original Amazon Chronos model (March 2024). Based on Google's T5 architecture, it tokenizes time series into discrete bins and uses the T5 decoder to generate probabilistic forecasts via sampling.

### How It Works
1. Historical demand per DFU is converted to a `torch.Tensor`
2. The T5 model generates `num_samples` (default: 20) stochastic forecast paths
3. The **median** across samples becomes the point forecast
4. DFUs with < 3 months of history are skipped

### Architecture Details
- **Tokenization**: Continuous values are binned into a vocabulary of ~4096 tokens
- **Sampling**: Each call produces `num_samples` independent forecast paths, then takes the median
- **Batching**: Manual — we batch DFUs into groups of `batch_size` and call `pipeline.predict()` per batch

### Configuration (`config/forecasting/forecast_pipeline_config.yaml`)
```yaml
chronos:
  enabled: true
  model_id: chronos
  model_size: small        # tiny | mini | small | base | large
  device: auto             # auto | cpu | mps | cuda
  batch_size: 1024         # DFUs per GPU dispatch
  num_samples: 20          # stochastic paths for median
  prediction_length: 6     # max months ahead (overridden by timeframe)
  num_workers: 1           # parallel timeframe workers
```

### Available Model Sizes
| Size | Params | HuggingFace ID | Accuracy | Speed |
|---|---|---|---|---|
| tiny | ~8M | `amazon/chronos-t5-tiny` | Lowest | Fastest |
| mini | ~20M | `amazon/chronos-t5-mini` | Low | Fast |
| **small** | **~46M** | **`amazon/chronos-t5-small`** | **Default** | **Moderate** |
| base | ~200M | `amazon/chronos-t5-base` | Good | Slow |
| large | ~710M | `amazon/chronos-t5-large` | Best | Slowest |

### Performance (231K DFUs, M4 Max, 10 timeframes)
- **GPU utilization**: ~84%
- **Per-timeframe**: ~13 min
- **Total**: ~2.5 hours

### Key Files
- `adv_algorithm_testing/foundation_models.py` — `_run_chronos()`, `_get_chronos_pipeline()`
- `scripts/run_backtest_chronos.py` — backtest orchestration
- `config/forecasting/forecast_pipeline_config.yaml` — `algorithms.chronos`

### Commands
```bash
make backtest-chronos          # run backtest
make backtest-load-chronos     # load to DB
make backtest-chronos-full     # both
# Resume after crash:
uv run python -m scripts.run_backtest_chronos --resume
```

---

## 3. Chronos Bolt (`chronos_bolt`)

### What It Is
Chronos Bolt (late 2024) is a complete architectural redesign. Instead of T5's tokenize-then-decode approach, Bolt uses a **native encoder architecture** purpose-built for time series. It returns **quantile forecasts directly** — no sampling needed. Up to 250x faster than Chronos T5 Large with comparable accuracy.

### How It Works
1. Historical demand per DFU is converted to a `torch.Tensor`
2. The model produces a tensor of shape `(n_series, n_quantiles, prediction_length)`
3. The **middle quantile** (median) is extracted as the point forecast
4. No sampling step — deterministic output

### Architecture Differences from T5
| Aspect | Chronos T5 | Chronos Bolt |
|---|---|---|
| Architecture | T5 decoder | Native encoder |
| Output | Sampled paths | Direct quantile forecasts |
| Pipeline class | `ChronosPipeline` | `ChronosBoltPipeline` |
| Speed | Baseline | Up to 250x faster |
| `num_samples` param | Required | Not used |

### Configuration
```yaml
chronos_bolt:
  enabled: true
  model_id: chronos_bolt
  model_size: base          # tiny | mini | small | base
  device: auto
  batch_size: 1024
  num_samples: 12           # passed but Bolt ignores internally
  prediction_length: 6
  num_workers: 1
```

### Available Model Sizes
| Size | Params | HuggingFace ID | Notes |
|---|---|---|---|
| tiny | ~9M | `amazon/chronos-bolt-tiny` | Fastest |
| mini | ~21M | `amazon/chronos-bolt-mini` | |
| small | ~48M | `amazon/chronos-bolt-small` | Matches T5-large accuracy |
| **base** | **~205M** | **`amazon/chronos-bolt-base`** | **Default — best variant** |

### Performance (231K DFUs, M4 Max, 10 timeframes)
- **GPU utilization**: ~93%
- **Per-timeframe**: ~1 min
- **Total**: ~12 min (12x faster than Chronos T5)

### Key Files
- `adv_algorithm_testing/foundation_models.py` — `_run_chronos_bolt()`
- `scripts/run_backtest_chronos_bolt.py` — backtest orchestration
- `config/forecasting/forecast_pipeline_config.yaml` — `algorithms.chronos_bolt`

### Commands
```bash
make backtest-bolt            # run backtest
make backtest-load-bolt       # load to DB
make backtest-bolt-full       # both
# Resume after crash:
uv run python -m scripts.run_backtest_chronos_bolt --resume
```

---

## 4. Chronos 2 (`chronos2`)

### What It Is
Amazon's latest generation foundation model (2025). Significantly larger (821M params) with a new architecture that supports **covariates**, **multivariate targets**, and **cross-series learning**. The zero-shot variant uses only raw demand history — same inputs as T5 and Bolt.

### How It Works
1. Historical demand per DFU is converted to a `torch.Tensor`
2. Inputs are chunked into batches (to avoid memory issues with 200K+ series)
3. `Chronos2Pipeline.predict()` returns a list of tensors, each `(1, 21_quantiles, prediction_length)`
4. Quantile index 10 (median of 21 quantiles) is extracted as the point forecast

### Architecture Details
- **Pipeline class**: `Chronos2Pipeline` (distinct from T5 and Bolt)
- **21 quantiles**: [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, **0.5**, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
- **Context length**: Up to 8192 time steps (our monthly data uses ~30)
- **Built-in batching**: `batch_size` parameter handled internally
- **Cross-learning**: Supports `cross_learning=True` for inter-series attention (not used in zero-shot variant)

### Configuration
```yaml
chronos2:
  enabled: true
  model_id: chronos2
  device: auto
  batch_size: 2048
  prediction_length: 6
  num_workers: 1
```

### Performance (231K DFUs, M4 Max, 10 timeframes)
- **GPU utilization**: ~100%
- **Per-timeframe**: ~35 min
- **Total**: ~5.5 hours

### Key Files
- `adv_algorithm_testing/foundation_models.py` — `_run_chronos2()`
- `scripts/run_backtest_chronos2.py` — backtest orchestration
- `config/forecasting/forecast_pipeline_config.yaml` — `algorithms.chronos2`

### Commands
```bash
make backtest-chronos2          # run backtest
make backtest-load-chronos2     # load to DB
make backtest-chronos2-full     # both
# Resume after crash:
uv run python -m scripts.run_backtest_chronos2 --resume
```

---

## 5. Chronos 2 Enriched (`chronos2_enriched`)

### What It Is
The same Chronos 2 model but with **31 covariate features** passed via the `past_covariates` and `future_covariates` API. This is the only foundation model variant that uses our feature engineering pipeline — combining the power of a pretrained foundation model with domain-specific signals.

### How It Works
1. The full feature matrix is built once via `build_feature_matrix()` (same as tree models)
2. Per timeframe, future sales are masked via `mask_future_sales()` to prevent leakage
3. For each DFU, an input dict is constructed:
   ```python
   {
       "target": torch.Tensor([...sales history...]),
       "past_covariates": {
           "qty_lag_1": np.array([...]),
           "brand": np.array(["BrandA", "BrandA", ...]),  # categorical
           "fourier_sin_12": np.array([...]),
           ...
       },
       "future_covariates": {
           "month": np.array([3, 4, 5, ...]),
           "fourier_sin_12": np.array([...]),
           ...
       }
   }
   ```
4. Chronos 2 attends to both the target history and covariates jointly
5. Median quantile extracted as point forecast

### Covariate Details

#### Past-Only Covariates (17 numeric)
Features known only for history — computed from lagged demand, cannot be projected forward.

| Feature | Source | Description |
|---|---|---|
| `qty_lag_1` .. `qty_lag_12` (5 used) | `feature_engineering.py` | Demand at 1, 2, 3, 6, 12 month lags |
| `qty_rolling_mean_3/6/12` | `feature_engineering.py` | Rolling average demand |
| `mom_growth` | `feature_engineering.py` | Month-over-month growth rate |
| `demand_accel` | `feature_engineering.py` | Second derivative of demand |
| `volatility_ratio` | `feature_engineering.py` | Short-term vs long-term volatility |
| `croston_demand_size` | `feature_engineering.py` | Croston decomposition: average non-zero demand |
| `croston_demand_interval` | `feature_engineering.py` | Croston decomposition: average interval between demands |
| `croston_probability` | `feature_engineering.py` | Croston decomposition: demand occurrence probability |
| `cluster_mean_lag1` | `feature_engineering.py` | Cluster-level average lag-1 demand |
| `cluster_total_lag1` | `feature_engineering.py` | Cluster-level total lag-1 demand |
| `cluster_demand_trend` | `feature_engineering.py` | Cluster-level demand trend direction |

#### Past + Future Covariates (13 numeric)
Calendar/seasonal features computable for any date — passed as both past and future covariates.

| Feature | Description |
|---|---|
| `month` | Month of year (1-12) |
| `quarter` | Quarter (1-4) |
| `is_quarter_end` | Binary: March, June, September, December |
| `is_year_end` | Binary: December |
| `days_in_month` | 28-31 |
| `fourier_sin_12`, `fourier_cos_12` | Annual seasonality (period 12) |
| `fourier_sin_6`, `fourier_cos_6` | Semi-annual seasonality (period 6) |
| `fourier_sin_4`, `fourier_cos_4` | Quarterly seasonality (period 4) |
| `fourier_sin_3`, `fourier_cos_3` | Tri-annual seasonality (period 3) |

#### Categorical Past Covariates (4)
String-valued features (Chronos 2 supports these natively as numpy string arrays).

| Feature | Source | Description |
|---|---|---|
| `ml_cluster` | `dim_sku` | Demand cluster assignment (e.g. "L2_1", "L2_3S") |
| `brand` | `dim_item` | Product brand |
| `region` | `dim_sku` | Geographic region |
| `abc_vol` | `dim_sku` | ABC volume classification |

### Vectorized Input Construction
Building 214K input dicts with covariates is optimized:
1. **Pre-compute row boundaries** for each DFU with a single pass over sorted arrays
2. **Extract all covariate columns once** as contiguous numpy arrays
3. **Per-DFU**: array slicing (`arr[start:end]`) — no pandas `groupby.get_group()` overhead
4. Result: 214K inputs built in ~2.3 seconds (vs ~2 minutes with naive approach)

### Configuration
```yaml
chronos2_enriched:
  enabled: true
  model_id: chronos2_enriched
  device: auto
  batch_size: 1024          # lower than zero-shot due to covariate memory
  prediction_length: 6
  num_workers: 1
```

### Batch Size Considerations
Smaller batch sizes (1024) outperform larger ones (8192) for the enriched variant because:
- **Padding overhead**: Variable-length histories + covariates are padded to the longest series in the batch. Larger batch = more wasted padding
- **Memory**: Each input dict carries 34 covariate arrays alongside the target
- **Sweet spot**: 1024 keeps pad waste low while maintaining GPU saturation

### Performance (214K DFUs, M4 Max, 10 timeframes)
- **GPU utilization**: ~100%
- **Per-timeframe**: ~35 min (similar to zero-shot — covariates add minimal GPU overhead)
- **Feature matrix build**: ~2 min (one-time)
- **Total**: ~6 hours

### Key Files
- `adv_algorithm_testing/foundation_models.py` — `_run_chronos2_enriched()`, `_build_future_calendar()`
- `scripts/run_backtest_chronos2_enriched.py` — backtest orchestration (builds feature matrix)
- `common/ml/feature_engineering.py` — `build_feature_matrix()`, `mask_future_sales()`
- `config/forecasting/forecast_pipeline_config.yaml` — `algorithms.chronos2_enriched`

### Commands
```bash
make backtest-chronos2e          # run backtest
make backtest-load-chronos2e     # load to DB
make backtest-chronos2e-full     # both
# Resume after crash:
uv run python -m scripts.run_backtest_chronos2_enriched --resume
```

---

## 6. Comparison Matrix

### Speed vs Accuracy Tradeoff

| Variant | Relative Speed | Expected Accuracy | Covariates | Best For |
|---|---|---|---|---|
| Chronos T5 | 1x (baseline) | Moderate | None | Baseline comparison |
| Chronos Bolt | **12x faster** | Good (matches T5-large) | None | Fast iteration, production default |
| Chronos 2 | 0.4x (slower) | Higher | None | Maximum zero-shot accuracy |
| Chronos 2 Enriched | 0.4x (slower) | **Highest** | 31 features | Best accuracy with domain knowledge |

### Input Comparison

| Input | T5 | Bolt | Chronos 2 | Chronos 2 Enriched |
|---|---|---|---|---|
| Sales history | Yes | Yes | Yes | Yes |
| Lag features | - | - | - | 5 lags |
| Rolling means | - | - | - | 3 windows |
| Calendar features | - | - | - | 13 features |
| Croston decomposition | - | - | - | 3 features |
| Cluster aggregates | - | - | - | 3 features |
| Categorical features | - | - | - | 4 features |
| Future covariates | - | - | - | 13 calendar features |

### Output Comparison

| Aspect | T5 | Bolt | Chronos 2 | Chronos 2 Enriched |
|---|---|---|---|---|
| Output type | Sampled paths | Quantiles | Quantiles | Quantiles |
| Number of quantiles | N/A (20 samples) | 9 | 21 | 21 |
| Point forecast | Median of samples | Middle quantile | Quantile idx 10 | Quantile idx 10 |
| Uncertainty | Via sample spread | Via quantile range | Via quantile range | Via quantile range |

---

## 7. Infrastructure

### Pipeline Caching
All variants share a module-level cache (`_chronos_pipeline_cache`) keyed by `"{model_name}:{device}"`. The model is loaded once and reused across all 10 timeframes within a run. Different variants use different cache keys and can coexist.

### Checkpoint/Resume
All backtest scripts use `BacktestCheckpointer` from the framework:
- Each timeframe's predictions are saved to `data/backtest/{model_id}/_checkpoints/tf_XXX.parquet` immediately after inference
- Default: starts fresh (clears old checkpoints). Use `--resume` to pick up from a crash
- Checkpoints are cleaned up after successful completion

### GPU Detection
```
auto → MPS (Apple Silicon) → CUDA (NVIDIA) → CPU
```
Controlled by `device` config param and `DEMAND_GPU` env var (`on`/`off`/`auto`).

### Parallel vs Sequential
Foundation model backtests support `--workers N` for parallel timeframe processing (useful for CPU-only runs). On GPU, keep `--workers 1` — the GPU is the bottleneck.

---

## 8. Future Enhancements

### Cross-Learning (not yet implemented)
Chronos 2 supports `cross_learning=True` which enables inter-series attention within a batch. Planned enhancement:
- Group DFUs by `ml_cluster` into batches
- Enable cross-learning so sparse DFUs borrow signal from active neighbors
- Expected benefit: improved accuracy for intermittent/cold-start DFUs

### Multivariate Targets (not yet implemented)
Chronos 2 accepts 2-d target tensors `(n_variates, history_length)`:
- Same item across multiple locations as co-targets
- Model learns cross-location demand correlations
- Requires restructuring from per-DFU to per-item inputs

### Additional Covariates (potential)
Features available in our data but not yet passed:
- `ext_fcst_ratio` — external forecast accuracy signal
- `n_zero_last_6m` — recent zero-demand count
- `cv_demand`, `seasonal_amplitude`, `mean_demand` — DFU profile features
- Price data (if available in source system)
