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
| `mvp/demand/common/tuning.py` | Shared tuning utilities: CV splits, WAPE objective, early stopping, param suggestion |
| `mvp/demand/config/hyperparameter_tuning.yaml` | Search spaces, CV settings, trial budget |
| `mvp/demand/tests/unit/test_tuning.py` | Unit tests for CV split logic, WAPE computation |

### Modified Files

| File | Change |
|------|--------|
| `mvp/demand/scripts/run_backtest.py` | Add `--params-file` argument (optional) |
| `mvp/demand/scripts/run_backtest_catboost.py` | Add `--params-file` argument (optional) |
| `mvp/demand/scripts/run_backtest_xgboost.py` | Add `--params-file` argument (optional) |
| `mvp/demand/Makefile` | Add `tune-lgbm`, `tune-catboost`, `tune-xgboost`, `tune-all` targets |

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
```

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

---

## MLflow Tracking

Experiment: `hyperparameter_tuning`

Each tuning run logs:
- **Params**: model name, n_trials, n_splits, gap_months, early_stopping_rounds
- **Metrics**: `best_wape`, `best_n_estimators`, per-cluster WAPEs
- **Artifacts**: `best_params_<model>.json`, Optuna study SQLite DB

---

## Algorithm Flow

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

---

## Estimated Runtime

| Model | Trials | Approx time (CPU) |
|-------|--------|-------------------|
| LGBM | 50 | 20–40 min |
| CatBoost | 50 | 30–60 min |
| XGBoost | 50 | 25–50 min |

Runtime scales with dataset size. GPU acceleration (if available) reduces by ~3–5×. Use `--n-trials 20` for a fast exploratory run.

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

- `tests/unit/test_tuning.py` covers:
  - `generate_cv_month_splits`: correct fold count, expanding windows, gap enforcement, min_train_months filtering
  - `compute_wape_stabilised`: normal case, zero actuals → inf, near-zero denominator clamping
  - `load_best_params`: JSON round-trip, missing file raises FileNotFoundError
  - `suggest_params`: valid keys returned for each model name

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
