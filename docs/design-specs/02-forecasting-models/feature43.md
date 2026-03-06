# Feature 43: Recursive Multi-Step Forecasting for Tree-Based Backtests

## Overview

Feature 43 adds a `--recursive` CLI flag to the LGBM, CatBoost, and XGBoost backtest scripts. When enabled, each month within the prediction window is forecast one step at a time, and the model's own prediction for month T is fed back as the `qty_lag_1` (and subsequently higher-order lags) feature for month T+1. This is recursive multi-step inference — a direct contrast to the default "direct multi-output" approach where all future months are predicted simultaneously from the same lag-1-zero baseline.

The feature is compute-side only: no API endpoints, no frontend changes, no database schema changes. Output files, loading, and downstream accuracy views are identical to direct mode.

---

## Motivation

### The lag_1=0 Problem

In the default backtest approach, when predicting future months (e.g., months 3–12 of the prediction window), the feature matrix is constructed by masking all sales after the training cutoff to zero and then computing lag features. This means:

- For the first predict month (month T+1): `qty_lag_1 = sales[T]` — correct, this is observed training data.
- For the second predict month (month T+2): `qty_lag_1 = sales[T+1]` — but `sales[T+1]` was masked to 0, so `qty_lag_1 = 0`.
- For month T+3 onward: the same pattern continues. All months beyond the first use `qty_lag_1 = 0`.

For tree-based models that heavily weight the most recent lag (`lag_1` consistently ranks #1 by SHAP importance), providing zero as the near-horizon signal is systematically distorting. This means multi-period accuracy metrics (e.g., lag 2, lag 3 accuracy in the backtest archive) are evaluated under artificially poor feature conditions.

### Recursive Multi-Step as the Alternative

In recursive mode, after predicting month T+1 the model's output (rather than 0) is written back into the grid as `qty[T+1]`, and all lag/rolling features are recomputed. Month T+2 therefore sees `qty_lag_1 = prediction[T+1]`, which is a richer signal than zero — even though it carries prediction error.

**Trade-offs:**
- **Advantage:** Near-horizon lag features reflect model state rather than zero, improving the realism of the forecast horizon simulation and the quality of multi-period accuracy metrics in the archive.
- **Disadvantage:** Prediction error from month T+1 propagates into month T+2's features, and so on — a form of error compounding. For DFUs with high volatility or large bias, this compounding can degrade accuracy for later months in the window.

Recursive mode is therefore most valuable for near-horizon analysis (lag 1–2) and for understanding how a model performs in a real deployment scenario where it would feed its own outputs forward.

---

## Architecture

### New Function: `update_grid_with_predictions()` in `common/feature_engineering.py`

**Signature:**

```python
def update_grid_with_predictions(
    grid: pd.DataFrame,
    month: pd.Timestamp,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
```

**Purpose:**

Writes the predicted `basefcst_pref` values for one month back into the grid's `qty` column, then recomputes all lag and rolling features using the same vectorized `groupby().shift()` and `rolling()` operations as `mask_future_sales()`.

**Args:**
- `grid`: Full masked feature grid (all DFUs × all months). The grid must already have future sales masked (call `mask_future_sales()` before the recursive loop).
- `month`: The prediction month being written back (e.g., `pd.Timestamp("2024-03-01")`).
- `predictions`: DataFrame with columns `dfu_ck` (index) and `basefcst_pref` (predicted quantity). Rows are matched to the grid by `dfu_ck` for the target month.

**Returns:** A new DataFrame with updated `qty` for the target month and recomputed lag/rolling features for all subsequent months.

**Why recompute all lags?** Because `qty_lag_1` for month T+2 depends on `qty[T+1]`, `qty_lag_2` for T+3 depends on `qty[T+1]`, and so on. After writing back the prediction for month T+1, a full recompute of the lag vector via `groupby().shift()` propagates the updated value into all future lag slots in a single vectorized pass.

---

### New Helper: `_fill_predict_nans()` in `common/backtest_framework.py`

**Signature:**

```python
def _fill_predict_nans(
    predict_data: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
```

**Purpose:**

Fills NaN values in numeric feature columns of `predict_data` with 0, skipping categorical columns (which are handled separately). This is a DRY extraction of the inline NaN-fill loop that was already present in `run_tree_backtest()` for the direct predict path. In recursive mode it is called once per predict month (instead of once over the full predict window).

---

### New Helper: `_predict_single_month()` in `common/backtest_framework.py`

**Signature:**

```python
def _predict_single_month(
    model_or_models: Any,
    predict_data: pd.DataFrame,
    feature_cols: list[str],
    cluster_strategy: str,
) -> pd.DataFrame:
```

**Purpose:**

Routes a single-month batch of predict rows to the correct model(s) without any retraining. Supports all three cluster strategies uniformly:

- **`"global"`**: Calls `model.predict(predict_data[feature_cols])`, clips to ≥ 0, returns rows with metadata columns plus `basefcst_pref`.
- **`"per_cluster"` / `"transfer"`**: `model_or_models` is a `dict[cluster_label → model]` (plus `"__base__"` for transfer). Iterates over `predict_data.groupby("ml_cluster")`, looks up the cluster model (falling back to `"__base__"` if the cluster is not in the dict), calls `.predict()`, and concatenates results. `ml_cluster` is automatically excluded from the feature columns passed to individual cluster models.

**Returns:** DataFrame with `_PREDICT_META_COLS` (`dfu_ck`, `dmdunit`, `dmdgroup`, `loc`, `startdate`) plus `basefcst_pref`.

---

### Modified: `run_tree_backtest()` in `common/backtest_framework.py`

**New parameter:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `recursive` | `bool` | `False` | When `True`, enables recursive multi-step inference for each timeframe's prediction window |

**Import change:** `update_grid_with_predictions` is now imported from `common.feature_engineering` inside `run_tree_backtest()` alongside the existing imports.

**Per-timeframe flow with `recursive=True`:**

```
For each timeframe (A → J):
  1. Build / mask feature matrix to train_end  (same as direct mode)
  2. Split into train_data and predict_months (sorted ascending)
  3. Run inline tuner if --tune-inline (same as direct mode)
  4. Initial training pass (one-time, on first predict month):
       predict_first_month = predict_months[0]
       preds_first, model_or_models = train_fn_*(train_data, first_month_data, ...)
  5. SHAP feature selection (if --shap-select):
       feature_selector_fn(...) → selected_features
       If features reduced: retrain on selected_features → update model_or_models + preds_first
  6. Recursive loop for months 2 … N:
       current_grid = masked_grid.copy()
       update_grid_with_predictions(current_grid, predict_months[0], preds_first)
       For each month in predict_months[1:]:
           month_data = _fill_predict_nans(current_grid[month], effective_feature_cols, ...)
           preds_month = _predict_single_month(model_or_models, month_data, effective_feature_cols, strategy)
           update_grid_with_predictions(current_grid, month, preds_month)
       preds = concat(preds_first + all preds_month)
  7. Attach model_id, timeframe, timeframe_idx to preds
  8. Append to all_predictions
```

**Metadata:** `recursive: True` is merged into `extra_metadata` so it appears in `backtest_metadata.json` for traceability.

---

## SHAP + Recursive Composability

When `--shap-select` and `--recursive` are both active, the flow is:

1. Train initial model on first predict month (same as recursive non-SHAP path).
2. `feature_selector_fn` computes SHAP on `train_data` (causally safe — same as direct mode).
3. If features are reduced, retrain on selected features. In recursive mode, this updates `model_or_models` (the dict or single model stored for the recursive loop) and `preds_first` (first-month predictions).
4. The recursive loop then uses `effective_feature_cols` (the SHAP-selected subset) and the retrained `model_or_models` for all subsequent months.

The key invariant: **SHAP is always computed on training data (causally masked), never on predict-window data.** The retrain for the SHAP-selected feature set uses `first_month_data` in recursive mode (not the full `predict_data` window), matching the single-step training call.

---

## CLI Flags

All three backtest scripts (`run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`) received an identical `--recursive` argparse flag:

| Flag | Type | Default | Effect |
|------|------|---------|--------|
| `--recursive` | `store_true` | `False` | Passes `recursive=True` to `run_tree_backtest()` |

The flag is passed via the existing `run_tree_backtest()` call in each script's `main()` function.

### Composable flag combinations

```bash
# Recursive only
make backtest-lgbm-cluster-recursive

# Recursive + SHAP feature selection
make backtest-lgbm-cluster-recursive ARGS="--shap-select"

# Recursive + per-timeframe causal inline tuning (PL-002)
make backtest-lgbm-cluster-recursive ARGS="--tune-inline"

# All three: recursive + SHAP + inline tuning
make backtest-lgbm-cluster-recursive ARGS="--shap-select --tune-inline"

# Recursive + pre-tuned params
make backtest-lgbm-cluster-recursive ARGS="--params-file data/tuning/best_params_lgbm.json"
```

---

## Makefile Targets

9 new targets added across all 3 models × 3 strategies:

| Target | Model | Strategy |
|--------|-------|----------|
| `make backtest-lgbm-recursive` | LightGBM | global |
| `make backtest-lgbm-cluster-recursive` | LightGBM | per-cluster |
| `make backtest-lgbm-transfer-recursive` | LightGBM | transfer |
| `make backtest-catboost-recursive` | CatBoost | global |
| `make backtest-catboost-cluster-recursive` | CatBoost | per-cluster |
| `make backtest-catboost-transfer-recursive` | CatBoost | transfer |
| `make backtest-xgboost-recursive` | XGBoost | global |
| `make backtest-xgboost-cluster-recursive` | XGBoost | per-cluster |
| `make backtest-xgboost-transfer-recursive` | XGBoost | transfer |

All targets accept an `ARGS` variable for passing additional flags:

```bash
make backtest-lgbm-cluster-recursive ARGS="--shap-select"
make backtest-catboost-cluster-recursive ARGS="--tune-inline"
```

---

## Output

Output format, file paths, and loading process are **identical** to direct mode:

- `data/backtest/<model_id>/backtest_predictions.csv` — execution-lag predictions
- `data/backtest/<model_id>/backtest_predictions_all_lags.csv` — all-lags archive
- `data/backtest/<model_id>/backtest_metadata.json` — includes `"recursive": true` field

Load into Postgres with the same command:
```bash
make backtest-load MODEL=lgbm_cluster
```

The `"recursive": true` flag in `backtest_metadata.json` provides traceability for distinguishing direct vs recursive runs without requiring separate model IDs.

---

## Trade-off Summary

| Dimension | Direct Mode | Recursive Mode |
|-----------|-------------|----------------|
| `qty_lag_1` for month T+2 | 0 (masked) | prediction[T+1] |
| Near-horizon realism | Lower (lag=0 signal) | Higher (model state) |
| Error compounding | None | Grows with horizon |
| Training cost | 1× per timeframe | 1× per timeframe (same) |
| Inference cost | 1 batch call | N sequential calls (one per predict month) |
| Best for | Aggregate accuracy benchmarking | Simulating real deployment |

---

## Testing

### `tests/unit/test_backtest_recursive.py` (13 tests)

Tests cover all three new/modified components:

**`TestFillPredictNans` (4 tests):**
- Fills NaN values in numeric feature columns
- Skips categorical columns (NaNs preserved)
- Skips columns not listed in `feature_cols`
- Preserves non-NaN values

**`TestPredictSingleMonthGlobal` (4 tests):**
- Returns correct shape (one row per input row)
- Clips negative predictions to zero
- Returns all required metadata columns (`dfu_ck`, `dmdunit`, `dmdgroup`, `loc`, `startdate`, `basefcst_pref`)
- Calls `model.predict()` with only the specified feature columns

**`TestPredictSingleMonthCluster` (4 tests):**
- Routes each DFU to its cluster's model by `ml_cluster` column
- Falls back to `"__base__"` model for unknown clusters (transfer learning)
- Returns empty DataFrame with correct columns when `predict_data` is empty
- Drops `ml_cluster` from the feature columns passed to individual cluster models

**`TestRecursiveLoopIntegration` (3 tests — in `test_feature_engineering.py` as `TestUpdateGridWithPredictions`, but exercising the recursive concept end-to-end, also in this file):**
- Verifies that `qty_lag_1` for month 2 is 0 in direct mode (baseline)
- Verifies that `qty_lag_1` for month 2 equals the prediction for month 1 after `update_grid_with_predictions()` (recursive mode)
- Two-month chain: lag_1 for month 3 equals the prediction written back for month 2
- Confirms that direct and recursive modes produce different lag values (the key behavioral difference)

### `tests/unit/test_feature_engineering.py` — `TestUpdateGridWithPredictions` (6 new tests)

Tests for `update_grid_with_predictions()` in `common/feature_engineering.py`:

- `test_writes_prediction_to_qty` — qty column for the target month equals `basefcst_pref`
- `test_lag1_of_next_month_updated` — `qty_lag_1` for month T+1 reflects the written-back prediction
- `test_other_months_unchanged` — months before the target are unaffected
- `test_missing_dfu_gets_zero` — DFUs not in `predictions` get qty=0 (fillna(0) behavior)
- `test_returns_new_dataframe` — function returns a copy, does not modify the input grid
- `test_rolling_features_recomputed` — rolling_mean features for subsequent months are updated after write-back

---

## Implementation Files

| File | Type | Change |
|------|------|--------|
| `mvp/demand/common/feature_engineering.py` | Modified | Added `update_grid_with_predictions()` function |
| `mvp/demand/common/backtest_framework.py` | Modified | Added `_fill_predict_nans()`, `_predict_single_month()`, `recursive: bool = False` param + recursive loop in `run_tree_backtest()` |
| `mvp/demand/scripts/run_backtest.py` | Modified | Added `--recursive` argparse flag, passes `recursive=args.recursive` to `run_tree_backtest()` |
| `mvp/demand/scripts/run_backtest_catboost.py` | Modified | Same: `--recursive` flag |
| `mvp/demand/scripts/run_backtest_xgboost.py` | Modified | Same: `--recursive` flag |
| `mvp/demand/Makefile` | Modified | 9 new targets: `backtest-{lgbm,catboost,xgboost}-{recursive,cluster-recursive,transfer-recursive}` |
| `mvp/demand/tests/unit/test_backtest_recursive.py` | New | 13 unit tests |
| `mvp/demand/tests/unit/test_feature_engineering.py` | Modified | 6 new tests in `TestUpdateGridWithPredictions` class |

**No API changes.** No frontend changes. No database schema changes. No new dependencies.

---

## Design Decisions

1. **Compute-side only:** Recursive mode is purely a backtest execution concern. Output files, loading scripts, Postgres tables, and the frontend are unchanged. A `"recursive": true` field in `backtest_metadata.json` provides traceability without requiring new DB columns or model IDs.

2. **Single training pass per timeframe:** The model is trained once per timeframe (same cost as direct mode). The recursive loop is inference-only — `_predict_single_month()` calls `model.predict()` but never fits. This keeps the recursive flag computationally cheap compared to re-training per month.

3. **`update_grid_with_predictions()` recomputes all lags on every call:** Although this is more work than a targeted update, it guarantees correctness — all lag offsets (lag_1 through lag_12) and all rolling windows (3m, 6m, 12m) are coherent after each write-back. The alternative (surgically patching only lag_1) would miss rolling features and higher-order lags.

4. **Composability with SHAP:** The SHAP retrain in recursive mode updates both `model_or_models` (the inference model) and `preds_first` (first-month predictions with selected features). This ensures that the entire recursive chain uses the SHAP-selected feature set consistently, not just months 2+.

5. **Composability with `--tune-inline`:** Inline tuning (PL-002) runs before the training call in both direct and recursive modes. The tuned params are used for the single training pass; the recursive loop uses the resulting model without re-tuning per month.

6. **No separate model_id for recursive runs:** Users who want to compare direct vs recursive accuracy in the UI can run both modes and load them with different `model_id` values by changing the output model ID via `--model-id` (if added later). Currently the same `model_id` is used (e.g., `lgbm_cluster`) and the `"recursive": true` metadata field distinguishes them at the file level.
