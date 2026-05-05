# 23 — Multi-Stage Feature Selection Pipeline

**Status**: Implemented
**Module**: `common/ml/shap_selector.py`
**Config**: `config/forecasting/forecast_pipeline_config.yaml` (per-algorithm params)

## Overview

Tree-based backtest models (LightGBM, CatBoost, XGBoost) use a 4-stage feature selection pipeline that runs per-timeframe to maintain causal correctness. Each stage operates only on training data up to the backtest cutoff — no future information leaks into feature selection.

## Pipeline Stages

### Stage 0: Duplicate Alias Removal (static)

Removes exact-duplicate features created by backward-compatibility aliases. Defined in `common/core/constants.py` as `DUPLICATE_FEATURE_ALIASES`:

| Alias (dropped) | Canonical (kept) |
|---|---|
| `year_over_year_correlation` | `yoy_correlation` |
| `sparsity_score` | `zero_demand_pct` |
| `growth_rate` | `cagr` |
| `recent_vs_historical` | `recency_ratio` |
| `demand_stability` | `cv_demand` |

**Cost**: Zero — static list, no data access.

### Stage 1: Near-Zero Variance Filter (per-timeframe)

Removes numeric features with relative variance below threshold. Relative variance = `var / (max - min)^2`. Features with zero range are always dropped.

- **Threshold**: `variance_threshold` (default: 0.01 = 1% of range)
- **Protected**: `PROTECTED_FEATURES` and categorical features are never removed
- **Cost**: O(n) single pass over training data

### Stage 2: Correlation Pre-Filter (per-timeframe)

For each pair of numeric features with absolute Pearson correlation exceeding threshold, drops the member with lower variance. Uses a 5,000-row sample for efficiency.

- **Threshold**: `correlation_threshold` (default: 0.95)
- **Tiebreaker**: Higher-variance feature survives
- **Protected**: `PROTECTED_FEATURES` always survive tiebreaks
- **Cost**: O(p^2) on sampled data (p = number of numeric features)

### Stage 3: SHAP Cumulative Selection (per-timeframe)

Existing SHAP-based selection, now operating on the reduced feature set from stages 0-2:

1. Compute SHAP values using the trained model (full feature set for dimensional consistency)
2. Pre-excluded features (from stages 0-2) are masked out of the cumulative selection pool
3. Select features covering `shap_threshold` (default: 95%) of total SHAP mass
4. `PROTECTED_FEATURES` are always kept regardless of SHAP rank

## Configuration

Per-algorithm in `config/forecasting/forecast_pipeline_config.yaml`:

```yaml
algorithms:
  lgbm_cluster:
    params:
      shap_select: true
      shap_threshold: 0.95
      shap_sample_size: 500
      correlation_filter: true
      correlation_threshold: 0.95
      variance_filter: true
      variance_threshold: 0.01
```

All stages are independently toggleable. Setting `correlation_filter: false` and `variance_filter: false` reverts to the previous SHAP-only behavior.

## Design Decisions

### Why not PCA?

PCA is dimensionality reduction, not feature selection. For tree models:
- Destroys interpretability (PC1 is a linear combination — meaningless for demand planning)
- Trees split on individual features and cannot exploit PCA components efficiently
- Trees already handle correlations natively via `colsample_bytree`

### Why same logic for all tree algorithms?

SHAP values are model-agnostic in interpretation (Lundberg et al. 2020). Feature importance rankings show >90% overlap across LGBM/XGBoost/CatBoost on the same data. Only the SHAP extractors differ (native `pred_contribs` vs CatBoost `ShapValues`).

### Why per-timeframe, not global?

Per-timeframe selection respects the causal boundary. The correlation structure and feature importance can shift across timeframes as the training window grows. Global selection using all history would leak future information — the same reason TS profile features are recomputed per cutoff.

### ml_cluster removed as a feature

`ml_cluster` was removed from `CAT_FEATURES` and `PROTECTED_FEATURES` because cluster assignments are computed from all available history, creating leakage when used as a feature in early backtest timeframes. It remains used for:
- Per-cluster model partitioning (splitting DFUs into separate training pools)
- SKU Features tab and clustering UI
- Inventory planning (ABC-XYZ segmentation)

See [Known Gaps section 1](../01-foundation/08-known-gaps.md) for full analysis.

## TS Profile Features

Per-DFU static features computed from historical demand in `_compute_ts_profile_features()`:

| Feature | Description |
|---|---|
| `mean_demand` | Mean of all qty values |
| `cv_demand` | Coefficient of variation (std / mean) |
| `zero_demand_pct` | Fraction of months with zero demand |
| `trend_slope_norm` | Scale-invariant linear trend slope |
| `recency_ratio` | Mean of last 6 months / prior mean |
| `seasonal_amplitude` | (max monthly mean - min monthly mean) / overall mean |
| `adi` | Average demand interval (mean gap between non-zero months) |
| `yoy_correlation` | Pearson correlation between demand and its 12-month lag |
| `periodicity` | Dominant cycle length (2-12 months) via autocorrelation peak. Returns the lag with highest autocorrelation if > 0.2, else 0. E.g., 12 = annual, 6 = semi-annual, 3 = quarterly cycle. |

All features are recomputed per backtest cutoff to prevent future leakage.

## Files

| File | Role |
|---|---|
| `common/ml/shap_selector.py` | Pipeline implementation (stages 0-3) |
| `common/core/constants.py` | `DUPLICATE_FEATURE_ALIASES`, `PROTECTED_FEATURES`, `CAT_FEATURES` |
| `scripts/run_backtest.py` | Builds `feature_selector_fn` closure with config params |
| `common/ml/backtest_framework.py` | Calls `feature_selector_fn` per timeframe (unchanged) |
| `config/forecasting/forecast_pipeline_config.yaml` | Per-algorithm feature selection config |
