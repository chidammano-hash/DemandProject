# Feature 42: SHAP-Based Per-Timeframe Feature Selection for Tree-Based Backtests

## Overview

Feature 42 adds automated SHAP-based feature selection to the tree-based backtest pipeline (LGBM, CatBoost, XGBoost). For each expanding-window timeframe, an initial model is trained on the full feature set, SHAP values are computed to rank features by importance, and a reduced feature set covering 95% of cumulative SHAP mass is selected. The final model for that timeframe is then retrained on only the selected features.

This reduces model complexity, improves generalization, and provides per-timeframe interpretability. SHAP outputs are written to structured CSV files and served via a dedicated REST API, with a collapsible panel in the Accuracy tab UI.

---

## Motivation

Tree-based demand forecasting models in this platform use up to ~40 engineered features (lags, rolling stats, calendar, DFU attributes). Not all features are equally useful across all time periods, and including noise features can hurt model accuracy. Key problems addressed:

1. **Feature noise:** Low-importance features add variance without signal, particularly for smaller clusters.
2. **Per-timeframe variation:** Feature importance shifts over time (e.g., seasonality features matter more in certain periods). A single static feature list is suboptimal across 10 expanding-window timeframes.
3. **Interpretability:** Planners and data scientists need to understand which signals drive demand forecasts. SHAP provides additive, model-consistent attribution.
4. **Causality:** SHAP is computed only on training data available up to each timeframe's cutoff date — no future leakage.

---

## Architecture

### Module: `common/shap_selector.py`

Central module providing model-agnostic SHAP computation and feature selection utilities.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `compute_shap_global(model, X_sample, feature_cols, cat_cols)` | Extract absolute SHAP values via `shap.TreeExplainer` — for LGBM and XGBoost |
| `compute_shap_catboost(model, X_sample, feature_cols, cat_cols)` | Extract absolute SHAP values via CatBoost native `get_feature_importance(type="ShapValues")` — no `shap` library required |
| `compute_timeframe_shap(model_or_dict, train_data, feature_cols, ...)` | Main entry point: computes SHAP for one backtest timeframe, handles both single-model (global) and dict-of-models (per_cluster / transfer) |
| `build_shap_summary(timeframe_reports, n_timeframes)` | Aggregates per-timeframe SHAP reports into a cross-timeframe summary DataFrame |
| `save_shap_outputs(timeframe_reports, output_dir, n_timeframes)` | Writes per-timeframe CSVs and summary CSV to `output_dir/shap/` |

**Type alias:**

```python
ShapExtractorFn = Callable[[Any, pd.DataFrame, list[str], list[str]], np.ndarray]
# Signature: (model_or_dict, X_sample, feature_cols, cat_cols) → np.ndarray
# Returned shape: (n_samples, n_features), values are absolute SHAP values
```

**Feature selection logic (`_select_features_from_shap`):**

Two modes:
- **Threshold mode** (default): Keep features covering `cumulative_threshold` (default 95%) of total SHAP mass, with `min_features` (default 5) as a lower bound.
- **Top-N mode**: Keep exactly `max(top_n, min_features)` features — activated by `--shap-top-n` CLI argument.

**Cluster-pooled SHAP (`_weighted_pool_cluster_shap`):**

For per-cluster and transfer strategies, SHAP is computed independently for each cluster's model using that cluster's training rows, then pooled as a weighted average by cluster size. The `__base__` key (present in transfer-learning model dicts) is skipped. The feature `ml_cluster` is stripped from `effective_feature_cols` to match the internal convention in `train_and_predict_per_cluster()`.

**Error handling:** If SHAP computation fails for any cluster or globally, the module logs a warning and falls back to keeping all features (no feature reduction) so the backtest continues.

---

### Integration: `common/backtest_framework.py`

`run_tree_backtest()` accepts two new optional parameters:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `feature_selector_fn` | `Callable \| None` | If provided, called after the initial model is trained for each timeframe to compute SHAP and select features. When features are reduced, the final model is retrained on the selected feature subset. |

**Per-timeframe flow with SHAP enabled:**

```
For each timeframe (A → J):
  1. Build feature matrix (causally masked to cutoff_date)
  2. Train initial model on ALL features → model_or_dict
  3. feature_selector_fn(model_or_dict, train_data, feature_cols, cat_cols,
                         timeframe_idx, cutoff_date, cluster_strategy)
     → (selected_features, shap_df)
  4. If len(selected_features) < len(feature_cols):
       Retrain FINAL model on selected_features only → model_or_dict
  5. Generate predictions using final model
  6. Append shap_df to timeframe_reports list
After all timeframes:
  save_shap_outputs(timeframe_reports, output_dir, n_timeframes)
```

The `feature_selector_fn` callable is constructed in each backtest script by wrapping `compute_timeframe_shap` with the model-specific `shap_extractor_fn` and user-provided CLI parameters.

---

### Backtest Scripts: `run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`

All three tree-based backtest scripts received the same four new CLI arguments:

| Argument | Default | Purpose |
|----------|---------|---------|
| `--shap-select` | off | Enable SHAP-based feature selection |
| `--shap-top-n` | None | Select exactly N top features (overrides threshold) |
| `--shap-threshold` | `0.95` | Cumulative SHAP mass threshold for feature selection |
| `--shap-sample-size` | `500` | Max rows to sample per cluster for SHAP computation |

**LGBM/XGBoost:** Use `compute_shap_global` from `shap_selector.py` (via `shap.TreeExplainer`). Requires `shap>=0.43.0` installed.

**CatBoost:** Uses `compute_shap_catboost` from `shap_selector.py` (CatBoost native `ShapValues`). No `shap` library dependency.

The `--shap-select` flag is fully composable with existing flags:
- `--shap-select` alone: threshold-based selection (95% cumulative SHAP mass)
- `--shap-select --shap-top-n 10`: top-10 features per timeframe
- `--tune-inline --shap-select`: per-timeframe causal tuning AND SHAP selection in a single run (PL-002 + Feature 42)
- `--params-file ... --shap-select`: apply pre-tuned params AND SHAP selection

---

### API: `api/routers/shap.py`

4 REST endpoints serving SHAP data from filesystem CSVs. No database queries.

| Endpoint | Purpose |
|----------|---------|
| `GET /forecast/shap/models` | List model IDs that have SHAP outputs (`shap_summary.csv` present) |
| `GET /forecast/shap/{model_id}/summary?top_n=15` | Cross-timeframe summary sorted by `mean_abs_shap_across_timeframes` descending |
| `GET /forecast/shap/{model_id}/timeframes` | List available timeframes with labels (A–J) and cutoff dates |
| `GET /forecast/shap/{model_id}/timeframe/{idx}?top_n=15` | Per-timeframe feature detail sorted by rank ascending |

The base data directory is controlled by the `BACKTEST_DATA_DIR` environment variable (default: `data/backtest`).

The router is mounted in `api/main.py` via `app.include_router(shap_router, tags=["shap"])`.

**Vite proxy:** The `/forecast` prefix is already proxied in `frontend/vite.config.ts` — no additional proxy entry required for the SHAP endpoints.

---

### Frontend: `frontend/src/tabs/AccuracyTab.tsx`

A collapsible "Feature Importance (SHAP)" card is added to the Accuracy tab, appearing after the existing Accuracy Comparison panel.

**UI features:**
- Model selector dropdown populated from `GET /forecast/shap/models`
- Timeframe selector dropdown (cross-timeframe summary or individual timeframes A–J)
- Horizontal bar chart (Recharts `BarChart`) showing top-N features by SHAP importance
- Color coding: **indigo** bars = selected features, **gray** bars = dropped features
- `selected_count` / `n_timeframes` indicator showing feature selection consistency across timeframes
- Lazy-loaded — panel only fetches data when expanded

**TypeScript types** (`frontend/src/types/shap.ts`):

| Type | Description |
|------|-------------|
| `ShapFeatureSummary` | Cross-timeframe summary row: `feature`, `mean_abs_shap_across_timeframes`, `mean_rank`, `selected_count`, `n_timeframes` |
| `ShapFeatureDetail` | Per-timeframe row: `feature`, `mean_abs_shap`, `rank`, `selected`, `timeframe`, `cutoff_date` |
| `ShapTimeframeEntry` | Timeframe list entry: `index`, `label`, `cutoff_date` |
| `ShapModelsPayload` | Models list response |
| `ShapSummaryPayload` | Summary response envelope |
| `ShapTimeframesPayload` | Timeframes list response |
| `ShapTimeframeDetailPayload` | Per-timeframe detail response |

**TanStack Query integration** (`frontend/src/api/queries.ts`):

4 new query keys + fetch functions added:

| Query Key | Function | Endpoint |
|-----------|----------|----------|
| `["shap", "models"]` | `fetchShapModels()` | `GET /forecast/shap/models` |
| `["shap", "summary", modelId, topN]` | `fetchShapSummary(modelId, topN)` | `GET /forecast/shap/{model_id}/summary` |
| `["shap", "timeframes", modelId]` | `fetchShapTimeframes(modelId)` | `GET /forecast/shap/{model_id}/timeframes` |
| `["shap", "timeframe", modelId, idx, topN]` | `fetchShapTimeframeDetail(modelId, idx, topN)` | `GET /forecast/shap/{model_id}/timeframe/{idx}` |

---

## Output Schema

### Per-Timeframe CSV: `data/backtest/<model_id>/shap/shap_timeframe_XX.csv`

| Column | Type | Description |
|--------|------|-------------|
| `feature` | str | Feature name (e.g., `lag_1`, `rolling_mean_3`, `brand`) |
| `mean_abs_shap` | float | Mean absolute SHAP value across the training sample |
| `rank` | int | Rank 1 = highest importance |
| `selected` | bool | True = included in reduced feature set |
| `timeframe` | int | 0-based timeframe index (0=A, 1=B, ..., 9=J) |
| `cutoff_date` | str | Training cutoff date for this timeframe (YYYY-MM-DD) |

### Summary CSV: `data/backtest/<model_id>/shap/shap_summary.csv`

| Column | Type | Description |
|--------|------|-------------|
| `feature` | str | Feature name |
| `mean_abs_shap_across_timeframes` | float | Average of `mean_abs_shap` across all timeframes |
| `mean_rank` | float | Average rank across all timeframes |
| `selected_count` | int | Number of timeframes in which the feature was selected |
| `n_timeframes` | int | Total number of timeframes processed |

### Directory Layout

```
data/backtest/lgbm_cluster/
├── backtest_predictions.csv
├── backtest_archive.csv
└── shap/
    ├── shap_timeframe_00.csv   (timeframe A)
    ├── shap_timeframe_01.csv   (timeframe B)
    ├── ...
    ├── shap_timeframe_09.csv   (timeframe J)
    └── shap_summary.csv
```

---

## CLI Usage

### Basic SHAP-enabled backtest

```bash
# LGBM — all three strategies with SHAP feature selection (95% threshold)
make backtest-lgbm-shap          # global
make backtest-lgbm-cluster-shap  # per-cluster
make backtest-lgbm-transfer-shap # transfer

# CatBoost — all three strategies
make backtest-catboost-shap
make backtest-catboost-cluster-shap
make backtest-catboost-transfer-shap

# XGBoost — all three strategies
make backtest-xgboost-shap
make backtest-xgboost-cluster-shap
make backtest-xgboost-transfer-shap
```

### Advanced usage

```bash
# Select exactly top-10 features per timeframe
make backtest-lgbm-cluster-shap ARGS="--shap-top-n 10"

# Custom threshold (80% cumulative SHAP mass)
make backtest-lgbm-cluster-shap ARGS="--shap-threshold 0.80"

# Larger SHAP sample for higher-fidelity computation
make backtest-lgbm-cluster-shap ARGS="--shap-sample-size 1000"

# Combine with pre-tuned hyperparameters
make backtest-lgbm-cluster ARGS="--params-file data/tuning/best_params_lgbm.json --shap-select"

# Combine with per-timeframe causal inline tuning (PL-002 + Feature 42)
make backtest-lgbm-cluster-tuned ARGS="--shap-select"
```

### Makefile targets

| Target | Model | Strategy |
|--------|-------|----------|
| `make backtest-lgbm-shap` | LightGBM | global |
| `make backtest-lgbm-cluster-shap` | LightGBM | per-cluster |
| `make backtest-lgbm-transfer-shap` | LightGBM | transfer |
| `make backtest-catboost-shap` | CatBoost | global |
| `make backtest-catboost-cluster-shap` | CatBoost | per-cluster |
| `make backtest-catboost-transfer-shap` | CatBoost | transfer |
| `make backtest-xgboost-shap` | XGBoost | global |
| `make backtest-xgboost-cluster-shap` | XGBoost | per-cluster |
| `make backtest-xgboost-transfer-shap` | XGBoost | transfer |

---

## API Endpoint Reference

### `GET /forecast/shap/models`

List model IDs with SHAP outputs available.

**Response:**
```json
{
  "models": ["lgbm_cluster", "catboost_cluster", "xgboost_cluster"]
}
```

---

### `GET /forecast/shap/{model_id}/summary?top_n=15`

Cross-timeframe SHAP importance summary. Features sorted by `mean_abs_shap_across_timeframes` descending.

**Path params:** `model_id` — e.g., `lgbm_cluster`
**Query params:** `top_n` (1–200, default 15)

**Response:**
```json
{
  "model_id": "lgbm_cluster",
  "total_features": 38,
  "features": [
    {
      "feature": "lag_1",
      "mean_abs_shap_across_timeframes": 0.42,
      "mean_rank": 1.2,
      "selected_count": 10,
      "n_timeframes": 10
    },
    ...
  ]
}
```

**Errors:** 404 if no `shap_summary.csv` exists for the model.

---

### `GET /forecast/shap/{model_id}/timeframes`

List available timeframes for a model.

**Response:**
```json
{
  "model_id": "lgbm_cluster",
  "timeframes": [
    {"index": 0, "label": "A", "cutoff_date": "2023-06-01"},
    {"index": 1, "label": "B", "cutoff_date": "2023-09-01"},
    ...
  ]
}
```

**Errors:** 404 if no SHAP outputs exist for the model.

---

### `GET /forecast/shap/{model_id}/timeframe/{idx}?top_n=15`

Per-timeframe SHAP feature detail. Features sorted by rank ascending (rank 1 = most important).

**Path params:** `model_id`, `idx` (0-based integer)
**Query params:** `top_n` (1–200, default 15)

**Response:**
```json
{
  "model_id": "lgbm_cluster",
  "timeframe_idx": 0,
  "label": "A",
  "cutoff_date": "2023-06-01",
  "total_features": 38,
  "features": [
    {
      "feature": "lag_1",
      "mean_abs_shap": 0.45,
      "rank": 1,
      "selected": true,
      "timeframe": 0,
      "cutoff_date": "2023-06-01"
    },
    ...
  ]
}
```

**Errors:** 404 if the specific timeframe CSV does not exist.

---

## Dependency

`shap>=0.43.0` added to `mvp/demand/pyproject.toml` under `[project.dependencies]`.

Note: CatBoost uses its own native SHAP implementation and does not require the `shap` library. The import is lazy (`import shap` inside `compute_shap_global`) so CatBoost-only runs do not require `shap` to be installed.

---

## Testing

### Backend Unit Tests: `tests/unit/test_shap_selector.py` (22 tests)

Covers:
- `compute_shap_global` with mock `shap.TreeExplainer`
- `compute_shap_catboost` with mock CatBoost Pool
- `_select_features_from_shap` in threshold mode and top-N mode
- `_weighted_pool_cluster_shap` with multi-cluster dicts
- `compute_timeframe_shap` for global and per_cluster strategies
- `build_shap_summary` aggregation
- `save_shap_outputs` filesystem writes
- Error handling / fallback behavior when SHAP computation fails

### Backend API Tests: `tests/api/test_shap.py` (8 tests)

Covers:
- `GET /forecast/shap/models` — lists models with summary CSV
- `GET /forecast/shap/{model_id}/summary` — happy path and 404
- `GET /forecast/shap/{model_id}/timeframes` — lists timeframe files
- `GET /forecast/shap/{model_id}/timeframe/{idx}` — happy path and 404
- `top_n` query parameter behavior

---

## Design Decisions

1. **CSV-based storage, not DB:** SHAP outputs are large (up to 38 features × 10 timeframes) but only read for analytics. Storing as model-scoped CSVs avoids DB schema changes, keeps loading fast, and lets users inspect outputs directly.

2. **Model-agnostic extractor pattern:** The `ShapExtractorFn` callable type lets `shap_selector.py` remain framework-agnostic. Each backtest script injects its own extractor (`compute_shap_global` or `compute_shap_catboost`).

3. **Causal safety:** SHAP is always computed on `train_data` (causally masked), never on held-out validation or future test data. This is enforced by the position of the SHAP hook inside `run_tree_backtest()` — after causal masking, before prediction.

4. **Weighted cluster pooling:** For per-cluster and transfer strategies, weighting by cluster size ensures that large clusters have proportional influence on the global feature selection decision. Clusters with zero training rows are skipped.

5. **Graceful degradation:** If SHAP computation fails (e.g., memory error, incompatible model state), the module logs a warning and returns all features as selected. The backtest continues with no feature reduction.

6. **`ml_cluster` exclusion for per-cluster/transfer:** Mirrors the internal convention in `train_and_predict_per_cluster()` where `ml_cluster` is removed from the feature list before fitting individual cluster models.

---

## Implementation Files

| File | Type | Purpose |
|------|------|---------|
| `mvp/demand/common/shap_selector.py` | New | SHAP computation, feature selection, CSV output |
| `mvp/demand/api/routers/shap.py` | New | 4 REST endpoints serving SHAP CSVs |
| `mvp/demand/frontend/src/types/shap.ts` | New | TypeScript type definitions |
| `mvp/demand/tests/unit/test_shap_selector.py` | New | 22 unit tests |
| `mvp/demand/tests/api/test_shap.py` | New | 8 API tests |
| `mvp/demand/pyproject.toml` | Modified | Added `shap>=0.43.0` |
| `mvp/demand/common/backtest_framework.py` | Modified | Added `feature_selector_fn` param + SHAP hook |
| `mvp/demand/scripts/run_backtest.py` | Modified | SHAP CLI args + LGBM SHAP extractor wiring |
| `mvp/demand/scripts/run_backtest_catboost.py` | Modified | SHAP CLI args + CatBoost SHAP extractor wiring |
| `mvp/demand/scripts/run_backtest_xgboost.py` | Modified | SHAP CLI args + XGBoost SHAP extractor wiring |
| `mvp/demand/api/main.py` | Modified | Mounted shap router |
| `mvp/demand/frontend/src/api/queries.ts` | Modified | 4 SHAP query keys + fetch functions |
| `mvp/demand/frontend/src/tabs/AccuracyTab.tsx` | Modified | Collapsible SHAP panel with model/timeframe selectors + BarChart |
| `mvp/demand/Makefile` | Modified | 9 SHAP backtest targets: `backtest-{lgbm,catboost,xgboost}-{shap,cluster-shap,transfer-shap}` |
