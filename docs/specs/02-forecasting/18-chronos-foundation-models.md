# Chronos Foundation Models

> Chronos 2 Enriched - Amazon's Chronos 2 foundation model (~821M params) driven with 31 covariate
> features - is the sole foundation-model variant in production. It combines a pretrained,
> zero-shot-capable architecture with our own feature engineering pipeline.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `scripts/ml/run_backtest_chronos2_enriched.py`, `common/ml/expert_panel/foundation_models.py`, `common/ml/feature_engineering.py`, `config/forecasting/forecast_pipeline_config.yaml` |

---

**Historical note:** Commit `5ab8d593` removed the Chronos T5, Chronos Bolt, and non-enriched Chronos 2
zero-shot variants, along with their backtest scripts, config sections, and Make targets
(`backtest-chronos`, `backtest-bolt`, `backtest-chronos2`, and the `bolt_hierarchical` variant). Only
Chronos 2 Enriched (`chronos2_enriched`) remains in production; this doc covers that variant only.

## Installation (optional `foundation` extra)

Chronos 2 Enriched requires the `chronos-forecasting` package (which pulls `torch`). It ships as the
optional **`foundation`** dependency group in `pyproject.toml` and is **not** part of the default
install (torch is heavy). Enable it with:

```bash
uv sync --extra foundation
```

The Jobs tab launches this backtest with `uv run --extra foundation …` (`common/services/job_state.py`,
`_MODEL_EXTRAS`), so the extra is ensured at run time even if a plain `uv sync` would otherwise strip
it. Without the package the backtest logs `chronos-forecasting not installed; skipping` and produces no
predictions - a graceful no-op, not a crash.

## What It Is

The same Chronos 2 model, but with **31 covariate features** passed via the `past_covariates` and
`future_covariates` API. This is the only foundation-model variant that uses our feature engineering
pipeline - combining a pretrained foundation model with domain-specific signals.

## How It Works

1. The full feature matrix is built once via `build_feature_matrix()` (same as the tree models)
2. Per timeframe, future sales are masked via `mask_future_sales()` to prevent leakage
3. For each DFU, an input dict is constructed with `target` (sales history), `past_covariates`, and
   `future_covariates`
4. Chronos 2 attends to both the target history and covariates jointly
5. The median quantile is extracted as the point forecast

## Covariate Details

**Past-only covariates (17 numeric):** known only for history, cannot be projected forward -
`qty_lag_1`..`qty_lag_12` (5 used), `qty_rolling_mean_3/6/12`, `mom_growth`, `demand_accel`,
`volatility_ratio`, Croston decomposition (`croston_demand_size`, `croston_demand_interval`,
`croston_probability`), and cluster aggregates (`cluster_mean_lag1`, `cluster_total_lag1`,
`cluster_demand_trend`).

**Past + future covariates (13 numeric):** calendar/seasonal features computable for any date -
`month`, `quarter`, `is_quarter_end`, `is_year_end`, `days_in_month`, and Fourier terms
(`fourier_sin/cos_12/6/4/3`).

**Categorical past covariates (4):** `ml_cluster`, `brand`, `region`, `abc_vol` - passed as numpy
string arrays (Chronos 2 supports these natively).

## Configuration

```yaml
chronos2_enriched:
  enabled: true
  model_id: chronos2_enriched
  device: auto
  batch_size: 1024          # lower than a zero-shot variant would use, due to covariate memory
  prediction_length: 6
  num_workers: 1
```

Smaller batch sizes (1024) outperform larger ones for the enriched variant: variable-length histories
plus 34 covariate arrays per input dict pad to the longest series in the batch, so a larger batch means
more wasted padding relative to the memory saved.

## Performance (214K DFUs, M4 Max, 10 timeframes)

- GPU utilization: ~100%
- Per-timeframe: ~35 min
- Feature matrix build: ~2 min (one-time)
- Total: ~6 hours

## Commands

```bash
make backtest-chronos2e          # run backtest
make backtest-load-chronos2e     # load to DB
make backtest-chronos2e-full     # both
# Resume after crash:
uv run python -m scripts.ml.run_backtest_chronos2_enriched --resume
```

## Infrastructure

- **Pipeline caching:** a module-level cache (`_chronos_pipeline_cache` in `foundation_models.py`),
  keyed by `"{model_name}:{device}"`, loads the model once and reuses it across all 10 timeframes.
- **Checkpoint/resume:** `BacktestCheckpointer` saves each timeframe's predictions to
  `data/backtest/chronos2_enriched/_checkpoints/tf_XXX.parquet` immediately after inference; `--resume`
  picks up from a crash.
- **GPU detection:** `auto → MPS (Apple Silicon) → CUDA (NVIDIA) → CPU`, controlled by the `device`
  config param and the `DEMAND_GPU` env var (`on`/`off`/`auto`).
