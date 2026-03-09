<!-- SOURCE: feature41.md (Hyperparameter Tuning) -->
# Feature 41: Hyperparameter Tuning for Tree-Based Cluster Models

## Status
**Implemented** — 2026-02-28

## Objective

Introduce Bayesian hyperparameter optimisation for the three cluster-aware tree models (LGBM, CatBoost, XGBoost) using [Optuna](https://optuna.org/). Replace hardcoded default parameters with data-driven optimal parameters discovered via walk-forward cross-validation that respects demand-forecasting causality constraints.

---

## Background: No Tuning in Current Codebase

A review of `run_backtest.py`, `run_backtest_catboost.py`, and `run_backtest_xgboost.py` confirmed that all three models use **fixed hardcoded defaults**:

| Model | Fixed params |
|-------|-------------|
| LGBM | `n_estimators=500`, `lr=0.05`, `num_leaves=31`, `min_child_samples=20` |
| CatBoost | `iterations=500`, `lr=0.05`, `depth=6`, `l2_leaf_reg=3.0` |
| XGBoost | `n_estimators=500`, `lr=0.05`, `max_depth=6`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8` |

No grid search, random search, or Bayesian optimisation is performed. No regularisation parameters (`reg_alpha`, `reg_lambda`, `gamma`, `random_strength`, etc.) are searched.

---

## Scope

- **Models tuned**: LightGBM, CatBoost, XGBoost (global strategy only — best global params also used for per-cluster and transfer strategies)
- **Optimisation engine**: Optuna 3.x with TPE (Tree-structured Parzen Estimator) sampler
- **CV strategy**: Month-based expanding walk-forward folds with causal masking
- **Metric**: WAPE (Weighted Absolute Percentage Error), same formula used across all backtest scripts
- **Early stopping**: Native early stopping inside each CV fold (removes `n_estimators` from the search space)
- **Output**: `data/tuning/best_params_<model>.json` consumed by backtest scripts via `--params-file`
- **MLflow**: Each Optuna study logged under `hyperparameter_tuning` experiment

---

## Design: Walk-Forward CV for Demand Forecasting

### Why standard k-fold is wrong here

Standard k-fold randomly assigns rows to folds. In demand forecasting with lag features (1–12 months), this creates **data leakage**: training fold rows can appear after validation fold rows in time, so lag values computed from the "future" contaminate the training signal.

### Correct approach: expanding month-based folds

The CV logic mirrors the existing backtest timeframe generation:

```
Fold 1: train [M1..M13], gap 1 month, val [M15..M17]
Fold 2: train [M1..M16], gap 1 month, val [M18..M20]
Fold 3: train [M1..M19], gap 1 month, val [M21..M23]
Fold 4: train [M1..M22], gap 1 month, val [M24..M26]
Fold 5: train [M1..M25], gap 1 month, val [M27..M29]
```

Key properties:
- **Expanding window**: each fold adds ~3 months of training history
- **Gap of 1 month**: prevents lag-feature leakage from the validation boundary
- **`mask_future_sales()`** called inside each fold: ensures rolling/lag features see only the training window (reuses existing framework function from `common/backtest_framework.py`)
- **Min 13 training months**: matches existing `MIN_TRAINING_MONTHS` constant

### WAPE objective (stabilised)

```
WAPE = Σ|F - A| / max(|ΣA|, ε)    where ε = 1.0
```

The denominator floor `ε = 1.0` prevents division by near-zero actuals in short validation windows while remaining negligible for any real demand signal. Returns `float("inf")` if all actuals are missing.

---

## Expert Review & Corrections Applied

*The initial draft was reviewed against ML forecasting best practices. The following issues were identified and corrected before implementation:*

### Correction 1 — Lag feature leakage in CV (Critical)

**Initial draft**: Built the feature matrix once before the CV loop and sliced by month index.

**Problem**: `mask_future_sales()` was not called inside the loop. Rolling statistics (e.g., `rolling_mean_12m`) computed at fold-4 training cutoff would contain information from fold-1 validation months if the feature matrix was built on full history.

**Fix**: Call `mask_future_sales(full_grid, train_end)` inside every fold iteration, exactly as the backtest framework does. This zeroes out qty values after `train_end` and recomputes all lag/rolling features. Adds ~0.5s per fold per trial but is required for correctness.

### Correction 2 — WAPE denominator instability (Important)

**Initial draft**: Raw WAPE = `Σ|F - A| / |ΣA|`. If a validation fold has very low demand (e.g., all DFUs in that cluster shipped 0 units), division by zero produces `NaN` or `Inf`, and Optuna marks the trial as failed.

**Fix**: Use `max(|ΣA|, 1.0)` as the denominator floor. This value (1.0 unit) is negligible for any real demand dataset while preventing numerical instability.

### Correction 3 — n_estimators not a tunable hyperparameter (Design)

**Initial draft**: Included `n_estimators` in the Optuna search space (integer range 100–2000).

**Problem**: Mixing tree count with structural params (depth, leaves, regularisation) creates a coupled search space where Optuna wastes trials exploring bad depths at low tree counts. The correct approach is:
1. Fix `n_estimators` to a large maximum (2000)
2. Use native early stopping inside each CV fold
3. Record `best_iteration_` per fold
4. Store `best_n_estimators = ceil(mean(best_rounds) × 1.1)` in the output JSON

This gives the optimal tree count automatically and reduces effective search space by one dimension.

**Fix**: Removed `n_estimators` from all search spaces. Each backtest script uses `best_n_estimators` from the JSON when `--params-file` is provided.

### Correction 4 — Pruner configuration (Performance)

**Initial draft**: `MedianPruner()` with default settings would prune after seeing only 1 fold result in the first few trials.

**Problem**: Demand forecasting WAPE has high fold-to-fold variance. Early pruning based on 1–2 folds is unreliable and can prune actually good trials.

**Fix**: `MedianPruner(n_startup_trials=15, n_warmup_steps=3)` — wait for 15 complete trials before comparing, and require at least 3 folds before pruning within a trial. This balances exploration vs pruning quality.

### Correction 5 — Per-cluster WAPE logging (Insight)

**Initial draft**: Only global WAPE tracked per trial.

**Problem**: A global WAPE improvement may mask degraded accuracy on small or high-value clusters. The best trial's cluster-level breakdown is needed to diagnose any structural issues.

**Fix**: After the study completes, re-evaluate the best trial parameters per cluster and log per-cluster WAPE in the output JSON and MLflow. This does not affect the search objective — only the output reporting.

### Correction 6 — Learning rate range lower bound (Range)

**Initial draft**: `learning_rate` lower bound = 0.01.

**Problem**: With early stopping limiting trees to ~200–500, learning rates below 0.02 consistently need more trees than the early stopping budget allows, biasing the search toward higher learning rates.

**Fix**: Lower bound raised to `0.02` for all three models. Log scale preserved. Upper bound remains `0.3`.

### Correction 7 — CatBoost `border_count` upper bound (Numerical)

**Initial draft**: `border_count` in [32, 255].

**Problem**: CatBoost's default `border_count=254` is already near-optimal for continuous features. Searching this adds tuning runtime with minimal accuracy gain.

**Fix**: Remove `border_count` from CatBoost search space. Focus on `depth`, `learning_rate`, `l2_leaf_reg`, `random_strength`, and `bagging_temperature`.

---

## Implementation

### New Files

| File | Purpose |
|------|---------|
| `mvp/demand/scripts/tune_hyperparams.py` | Main Optuna tuning script (CLI entry point) |
| `mvp/demand/common/tuning.py` | Shared tuning utilities: CV splits, WAPE objective, early stopping, param suggestion, per-timeframe causal tuning, fold training functions |
| `mvp/demand/config/hyperparameter_tuning.yaml` | Search spaces, CV settings, trial budget, inline tuning settings |
| `mvp/demand/tests/unit/test_tuning.py` | Unit tests for CV split logic, WAPE computation, fold function registry, per-timeframe causality |

### Modified Files

| File | Change |
|------|--------|
| `mvp/demand/scripts/run_backtest.py` | Add `--params-file` argument (optional); add `--tune-inline`, `--tune-n-trials`, `--tune-config` (PL-002) |
| `mvp/demand/scripts/run_backtest_catboost.py` | Add `--params-file` argument (optional); add `--tune-inline`, `--tune-n-trials`, `--tune-config` (PL-002) |
| `mvp/demand/scripts/run_backtest_xgboost.py` | Add `--params-file` argument (optional); add `--tune-inline`, `--tune-n-trials`, `--tune-config` (PL-002) |
| `mvp/demand/common/backtest_framework.py` | Add `inline_tuner_fn` optional parameter to `run_tree_backtest()` (PL-002) |
| `mvp/demand/Makefile` | Add `tune-lgbm`, `tune-catboost`, `tune-xgboost`, `tune-all` targets; add `backtest-lgbm-cluster-tuned`, `backtest-catboost-cluster-tuned`, `backtest-xgboost-cluster-tuned` targets |

---

## Search Spaces

### LightGBM

| Hyperparameter | Range | Scale | Default |
|----------------|-------|-------|---------|
| `learning_rate` | [0.02, 0.3] | log | 0.05 |
| `num_leaves` | [15, 127] | int | 31 |
| `min_child_samples` | [10, 100] | int | 20 |
| `subsample` | [0.5, 1.0] | linear | 0.8 |
| `colsample_bytree` | [0.5, 1.0] | linear | 0.8 |
| `reg_alpha` (L1) | [1e-8, 10.0] | log | 0 |
| `reg_lambda` (L2) | [1e-8, 10.0] | log | 0 |
| `min_gain_to_split` | [0.0, 5.0] | linear | 0 |

Fixed: `n_estimators=2000` (early stopping finds actual count), `verbosity=-1`, `random_state=42`, `n_jobs=-1`

### CatBoost

| Hyperparameter | Range | Scale | Default |
|----------------|-------|-------|---------|
| `learning_rate` | [0.02, 0.3] | log | 0.05 |
| `depth` | [4, 10] | int | 6 |
| `l2_leaf_reg` | [1.0, 10.0] | log | 3.0 |
| `bagging_temperature` | [0.0, 1.0] | linear | 1.0 |
| `random_strength` | [1e-9, 10.0] | log | 1.0 |

Fixed: `iterations=2000`, `random_seed=42`, `verbose=0`

### XGBoost

| Hyperparameter | Range | Scale | Default |
|----------------|-------|-------|---------|
| `learning_rate` | [0.02, 0.3] | log | 0.05 |
| `max_depth` | [3, 10] | int | 6 |
| `min_child_weight` | [1, 20] | int | 5 |
| `subsample` | [0.5, 1.0] | linear | 0.8 |
| `colsample_bytree` | [0.5, 1.0] | linear | 0.8 |
| `gamma` | [0.0, 5.0] | linear | 0 |
| `reg_alpha` (L1) | [1e-8, 10.0] | log | 0 |
| `reg_lambda` (L2) | [1e-8, 10.0] | log | 1.0 |

Fixed: `n_estimators=2000`, `verbosity=0`, `random_state=42`, `n_jobs=-1`, `enable_categorical=True`, `tree_method="hist"`

---

## Output JSON Format

`data/tuning/best_params_<model>.json`:

```json
{
  "model": "lgbm",
  "best_wape": 11.43,
  "best_n_estimators": 387,
  "best_params": {
    "learning_rate": 0.042,
    "num_leaves": 63,
    "min_child_samples": 35,
    "subsample": 0.82,
    "colsample_bytree": 0.74,
    "reg_alpha": 0.0021,
    "reg_lambda": 1.23,
    "min_gain_to_split": 0.14
  },
  "per_cluster_wape": {
    "high_volume_steady": 8.21,
    "seasonal_medium_volume": 14.67,
    "low_volume_erratic": 22.89
  },
  "n_trials_completed": 50,
  "cv_fold_wapes": [10.1, 11.8, 12.3, 11.2, 11.8],
  "timestamp": "2026-02-28T14:30:00",
  "config": {
    "n_splits": 5,
    "gap_months": 1,
    "early_stopping_rounds": 50,
    "n_estimators_max": 2000
  }
}
```

---

## Backtest Script Integration

When `--params-file data/tuning/best_params_lgbm.json` is passed:
1. Load `best_params` from JSON
2. Set `n_estimators = best_n_estimators`
3. Merge with any CLI overrides (CLI takes precedence)
4. Log source as `"params_source": "tuning_file"` in MLflow metadata

When `--params-file` is omitted: existing default behaviour unchanged.

---

## Make Targets

```makefile
tune-lgbm:       # Tune LGBM hyperparameters (50 trials, ~20–40 min)
tune-catboost:   # Tune CatBoost hyperparameters (~30–60 min)
tune-xgboost:    # Tune XGBoost hyperparameters (~25–50 min)
tune-all:        # Run all three tuning jobs sequentially

# Honest backtesting: per-timeframe causal tuning (PL-002 fix)
backtest-lgbm-cluster-tuned:       # LGBM per-cluster with inline per-timeframe tuning
backtest-catboost-cluster-tuned:   # CatBoost per-cluster with inline per-timeframe tuning
backtest-xgboost-cluster-tuned:    # XGBoost per-cluster with inline per-timeframe tuning
```

---

## Causal Per-Timeframe Tuning (PL-002 Fix)

### Problem: Temporal Data Leakage

The global tuning workflow (`make tune-lgbm` → `--params-file`) tunes on the **full sales history** and applies those parameters to all 10 backtest timeframes. This introduces **temporal data leakage**: the tuner has already seen observations from future timeframes (e.g. timeframe J) when selecting parameters that are then applied to earlier timeframes (e.g. timeframe A). Backtest accuracy numbers are therefore optimistically biased.

### Solution: `tune_for_timeframe()`

`common/tuning.py` exposes:

```python
def tune_for_timeframe(
    model_name: str,           # "lgbm" | "catboost" | "xgboost"
    train_fold_fn: Callable,   # from TRAIN_FOLD_FNS registry
    full_grid: pd.DataFrame,   # full feature matrix (pre-filtered inside)
    feature_cols: list[str],
    cat_cols: list[str],
    cutoff_date: pd.Timestamp, # = train_end for that timeframe
    config: dict,              # loaded from hyperparameter_tuning.yaml
    n_trials: int | None,      # override inline_n_trials (default: 20)
) -> tuple[dict[str, Any], int]:
    # filters full_grid to months <= cutoff_date
    # runs in-memory Optuna study (no SQLite)
    # returns (best_params_dict, best_n_estimators) or ({}, 500) if insufficient data
```

### `TRAIN_FOLD_FNS` Registry

Fold training functions are now public in `common/tuning.py` and shared between the global tuning script and the inline tuner:

```python
TRAIN_FOLD_FNS: dict[str, Callable] = {
    "lgbm": train_lgbm_fold,
    "catboost": train_catboost_fold,
    "xgboost": train_xgboost_fold,
}
```

### `inline_tuner_fn` in `run_tree_backtest()`

`common/backtest_framework.py`'s `run_tree_backtest()` accepts a new optional parameter:

```python
inline_tuner_fn: Callable[[full_grid, feature_cols, cat_cols, train_end], dict] | None = None
```

When provided, each timeframe calls the tuner before training and uses the resulting `effective_params` instead of static `model_params`.

### CLI Flags Added to All Three Backtest Scripts

```
--tune-inline           Enable per-timeframe causal tuning (mutually exclusive with --params-file)
--tune-n-trials N       Override trial count per timeframe (default: from YAML inline_n_trials)
--tune-config PATH      Override tuning YAML path
```

### Config Additions (`config/hyperparameter_tuning.yaml`)

```yaml
tuning:
  inline_n_trials: 20    # Optuna trials per timeframe (fewer than global 50)
  inline_n_splits: 3     # CV folds per timeframe (fewer than global 5)
```

### Two-Mode Workflow

| Mode | Command | Use Case |
|------|---------|----------|
| **Production scoring** | `make tune-lgbm && make backtest-lgbm-cluster ARGS="--params-file data/tuning/best_params_lgbm.json"` | Tune once on full history, apply to future production forecasts |
| **Honest backtesting** | `make backtest-lgbm-cluster-tuned` | Per-timeframe causal tuning; no future leakage; genuine OOS accuracy |

### Performance Note

Per-timeframe inline tuning: 10 timeframes × 20 trials × 3 CV folds = **600 model fits** (vs. 250 for global one-shot tuning). Expect ~2–3× longer runtime than an untuned backtest. The trade-off is genuine out-of-sample accuracy with no future leakage.

---

## Full Tuned Backtest Workflow

### Step 1 — Tune hyperparameters

```bash
# Tune one model
make tune-catboost
# → data/tuning/best_params_catboost.json

# Or tune all three models
make tune-all
# → data/tuning/best_params_lgbm.json
# → data/tuning/best_params_catboost.json
# → data/tuning/best_params_xgboost.json
```

### Step 2 — Run backtest with tuned parameters

Pass the absolute path to the tuning JSON via `ARGS="--params-file ..."`.

```bash
# CatBoost cluster backtest with tuned params
make backtest-catboost-cluster ARGS="--params-file /Users/manoharchidambaram/projects/DemandProject/mvp/demand/data/tuning/best_params_catboost.json"

# LGBM cluster backtest with tuned params
make backtest-lgbm-cluster ARGS="--params-file /Users/manoharchidambaram/projects/DemandProject/mvp/demand/data/tuning/best_params_lgbm.json"

# XGBoost cluster backtest with tuned params
make backtest-xgboost-cluster ARGS="--params-file /Users/manoharchidambaram/projects/DemandProject/mvp/demand/data/tuning/best_params_xgboost.json"
```

Relative paths also work when running from `mvp/demand/`:
```bash
make backtest-catboost-cluster ARGS="--params-file data/tuning/best_params_catboost.json"
```

### Step 3 — Load predictions into Postgres

Each backtest writes to its own subdirectory (`data/backtest/<model_id>/`). Load individually:

```bash
make backtest-load MODEL=lgbm_cluster
make backtest-load MODEL=catboost_cluster
make backtest-load MODEL=xgboost_cluster
```

Or load all at once after all three backtests complete:

```bash
make backtest-load-all
```

### Step 4 — Run champion selection

```bash
make champion-select
```

### Full pipeline (tune → backtest → load → champion)

```bash
make tune-all
make backtest-lgbm-cluster ARGS="--params-file data/tuning/best_params_lgbm.json"
make backtest-catboost-cluster ARGS="--params-file data/tuning/best_params_catboost.json"
make backtest-xgboost-cluster ARGS="--params-file data/tuning/best_params_xgboost.json"
make backtest-load-all
make champion-select
```

### Honest backtest pipeline (per-timeframe inline tuning, no leakage)

```bash
make backtest-lgbm-cluster-tuned      # per-cluster LGBM with inline per-timeframe tuning
make backtest-catboost-cluster-tuned  # per-cluster CatBoost with inline per-timeframe tuning
make backtest-xgboost-cluster-tuned   # per-cluster XGBoost with inline per-timeframe tuning
make backtest-load-all
make champion-select
```

---

## MLflow Tracking

Experiment: `hyperparameter_tuning`

Each tuning run logs:
- **Params**: model name, n_trials, n_splits, gap_months, early_stopping_rounds
- **Metrics**: `best_wape`, `best_n_estimators`, per-cluster WAPEs
- **Artifacts**: `best_params_<model>.json`, Optuna study SQLite DB

---

## Algorithm Flow

### Global Tuning (`tune_hyperparams.py`)

```
tune_hyperparams.py
│
├── 1. Load data (load_backtest_data from backtest_framework)
├── 2. Build feature matrix once (build_feature_matrix)
├── 3. Generate CV month splits (generate_cv_month_splits in tuning.py)
│
├── 4. Create Optuna study (TPESampler, MedianPruner, SQLite storage)
│
├── 5. For each trial (50 trials):
│   ├── a. Suggest params from search space
│   ├── b. For each CV fold:
│   │   ├── mask_future_sales(full_grid, train_end)   ← causal masking
│   │   ├── Split train/val rows by month
│   │   ├── Train with early stopping on val fold
│   │   ├── Compute stabilised WAPE
│   │   └── Report intermediate value → pruner checks
│   ├── c. Return mean WAPE across folds
│   └── d. Prune if below median of completed trials
│
├── 6. Log best trial to MLflow
├── 7. Compute per-cluster WAPE on best params
└── 8. Save best_params_<model>.json
```

### Inline Per-Timeframe Tuning (`--tune-inline` in backtest scripts)

```
run_backtest.py --tune-inline
│
├── 1. Load data, build full feature matrix
│
├── For each timeframe A–J (10 timeframes):
│   ├── a. Derive train_end (cutoff date for this timeframe)
│   │
│   ├── b. Call inline_tuner_fn(full_grid, feature_cols, cat_cols, train_end)
│   │      └── tune_for_timeframe():
│   │           ├── filter full_grid to months <= cutoff_date  ← no future leakage
│   │           ├── generate_cv_month_splits (3 folds, in-memory Optuna study)
│   │           ├── For each trial (20 trials):
│   │           │   └── TRAIN_FOLD_FNS[model_name](train_fold, val_fold, params)
│   │           └── return (best_params, best_n_estimators)
│   │
│   ├── c. Merge tuned params → effective_params
│   └── d. Train model with effective_params on this timeframe's training data
│
└── Save predictions to data/backtest/<model_id>/
```

---

## Estimated Runtime

### Global tuning (`make tune-*`)

| Model | Trials | Approx time (CPU) |
|-------|--------|-------------------|
| LGBM | 50 | 20–40 min |
| CatBoost | 50 | 30–60 min |
| XGBoost | 50 | 25–50 min |

Runtime scales with dataset size. GPU acceleration (if available) reduces by ~3–5×. Use `--n-trials 20` for a fast exploratory run.

### Inline per-timeframe tuning (`--tune-inline`)

| Step | Count | Model fits |
|------|-------|-----------|
| Timeframes | 10 | — |
| Trials per timeframe | 20 | — |
| CV folds per trial | 3 | — |
| **Total model fits** | — | **600** |

Expect ~2–3× longer runtime than an untuned backtest (vs. 250 fits for global tuning). This is the cost of genuine out-of-sample accuracy with no future leakage.

---

## CV Split Example

For a dataset with 48 months and `n_splits=5`, `gap_months=1`, `min_train_months=13`, `val_months_per_fold=3`:

```
Fold 1: train M1–M13  (13 months), gap M14, val M15–M17 (3 months)
Fold 2: train M1–M21  (21 months), gap M22, val M23–M25 (3 months)
Fold 3: train M1–M29  (29 months), gap M30, val M31–M33 (3 months)
Fold 4: train M1–M37  (37 months), gap M38, val M39–M41 (3 months)
Fold 5: train M1–M45  (45 months), gap M46, val M47–M48 (2 months)
```

---

## Testing

- `tests/unit/test_tuning.py` covers (39 tests total):
  - `generate_cv_month_splits`: correct fold count, expanding windows, gap enforcement, min_train_months filtering
  - `compute_wape_stabilised`: normal case, zero actuals → inf, near-zero denominator clamping
  - `load_best_params`: JSON round-trip, missing file raises FileNotFoundError
  - `suggest_params`: valid keys returned for each model name
  - `TRAIN_FOLD_FNS`: registry has all 3 models (`lgbm`, `catboost`, `xgboost`), all are callable
  - `tune_for_timeframe`: returns `(dict, int)` tuple; best params keys match search space; **only causal months used** (core PL-002 test — verifies `full_grid` filtered to `months <= cutoff_date`); insufficient data returns `({}, 500)`; cutoff before all data returns empty; different cutoffs produce different results

---

## Dependencies

New:
- `optuna>=3.0` — Bayesian optimisation
- `optuna-integration[lightgbm]>=3.0` — optional (not required, early stopping handled manually)

Existing (already in `pyproject.toml`):
- `lightgbm`, `catboost`, `xgboost`, `scikit-learn`, `pandas`, `numpy`

---

## Design Specs Referenced

- Feature 8 — Backtesting framework (expanding window timeframes)
- Feature 9 — LGBM backtesting (shared framework integration)
- Feature 12 — CatBoost backtesting
- Feature 13 — XGBoost backtesting
- Feature 31 — Comprehensive testing strategy


---

<!-- SOURCE: feature42.md (SHAP Feature Selection) -->
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


---

<!-- SOURCE: feature43.md (Recursive Forecasting) -->
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
