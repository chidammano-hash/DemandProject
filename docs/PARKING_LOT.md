# Parking Lot — Known Issues & Deferred Work

Issues captured here are real, confirmed problems that are not yet fixed.
Each entry includes root cause, impact, and a recommended fix when known.

---

## Fixed Issues

---

### PL-001 — Multiple Backtests Overwrite the Same Output CSVs

**Status:** Fixed — 2026-02-28
**Priority:** High
**Date captured:** 2026-02-28
**Fixed in:** `common/backtest_framework.py`, `scripts/load_backtest_forecasts.py`, `Makefile`
**Affects:** `run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`

#### Problem

Every backtest script (LGBM, CatBoost, XGBoost — all strategies) wrote output to the **same two fixed file paths** regardless of model or strategy:

```
mvp/demand/data/backtest/backtest_predictions.csv
mvp/demand/data/backtest/backtest_predictions_all_lags.csv
```

Running a second backtest before loading the first silently overwrote both CSVs, losing the first model's predictions permanently.

#### Fix Applied (Option B — Model-scoped subdirectory)

`common/backtest_framework.py` → `save_backtest_output()` now writes each model into its own subdirectory:

```
data/backtest/lgbm_cluster/backtest_predictions.csv
data/backtest/lgbm_cluster/backtest_predictions_all_lags.csv
data/backtest/catboost_cluster/backtest_predictions.csv
data/backtest/catboost_cluster/backtest_predictions_all_lags.csv
data/backtest/xgboost_cluster/backtest_predictions.csv
data/backtest/xgboost_cluster/backtest_predictions_all_lags.csv
```

`scripts/load_backtest_forecasts.py` was refactored to support:
- `--model MODEL_ID` — load a single model from `data/backtest/<MODEL_ID>/`
- `--all` — discover and load all models under `data/backtest/*/`
- `--input PATH` — backward-compatible explicit path

Makefile targets updated:
- `make backtest-load MODEL=lgbm_cluster` — load one model
- `make backtest-load-all` — load all available models

You can now batch multiple backtests safely:

```bash
make backtest-lgbm-cluster
make backtest-catboost-cluster
make backtest-xgboost-cluster
make backtest-load-all           # loads all three in sequence
```

---

---

### PL-002 — Hyperparameter Tuning Uses Full History, Causing Data Leakage Into Backtests

**Status:** Fixed — 2026-02-28
**Priority:** High
**Date captured:** 2026-02-28
**Fixed in:** `common/tuning.py`, `common/backtest_framework.py`, `scripts/run_backtest*.py`, `config/hyperparameter_tuning.yaml`, `Makefile`
**Affects:** `scripts/tune_hyperparams.py`, `common/tuning.py`, all tree-based backtest scripts (`run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`)

#### Problem

The current tuning pipeline (`make tune-lgbm/catboost/xgboost`) runs Optuna over the **full sales history** to find optimal hyperparameters. Those tuned parameters are then passed via `--params-file` to backtest scripts that evaluate model accuracy across 10 expanding timeframes (A–J).

This introduces **temporal data leakage**: the tuner has already seen observations from future timeframes (e.g. timeframe J) when selecting parameters that are then applied to earlier timeframes (e.g. timeframe A). The backtest accuracy numbers are therefore optimistically biased — the model was implicitly tuned on the data it is being evaluated against.

#### Example of the leak

```
Tuning window:  Jan-2020 → Dec-2025  (all history)
Timeframe A:    Jan-2020 → Dec-2022  (train) | Jan-2023 (test)
Timeframe J:    Jan-2020 → Nov-2025  (train) | Dec-2025 (test)

Params tuned on full history reflect signal from Dec-2025 data,
then those same params are used to "predict" Dec-2025 in Timeframe J.
```

#### Root Cause

`tune_hyperparams.py` does not receive a `cutoff_date` argument. `common/tuning.py` CV splits operate on whatever data is passed in — it has no knowledge of the backtest timeframe being evaluated.

#### Recommended Fix

Tune hyperparameters **within each backtest timeframe** using only the training data available at that point:

1. For each timeframe T (A–J), derive the training cutoff date from `TIMEFRAMES[T]`.
2. Filter sales history to `date < cutoff_T` before calling the Optuna study.
3. Use the timeframe-specific best params to train the model for that timeframe's test window.
4. This can be implemented as a new `--tune-inline` flag in each backtest script, or by pre-computing a params file per timeframe before the backtest loop.

**Result:** 10 separate param sets (one per timeframe), each tuned on strictly causal data. Backtest accuracy reflects true out-of-sample performance with no future leakage.

#### Impact

- All backtest accuracy metrics produced with a shared `--params-file` are optimistically biased (degree unknown, likely moderate for stable datasets, high for datasets with trend shifts or regime changes).
- Champion selection downstream of the backtest inherits this bias.
- The current approach is still useful for *production scoring* (tune on all history, apply to future), but **must not be used to evaluate or compare backtest accuracy**.

#### Workaround (superseded)

Run backtests without `--params-file` (default hyperparameters have no leakage). Use `--params-file` only when generating production forecasts, not for accuracy benchmarking.

#### Fix Applied

**`common/tuning.py`** — Added `tune_for_timeframe(model_name, train_fold_fn, full_grid, feature_cols, cat_cols, cutoff_date, config, n_trials)` which:
- Filters `full_grid` to months `<= cutoff_date` before building any CV splits
- Runs a lightweight Optuna study (default 20 trials, 3 folds) strictly in-sample
- Returns `(best_params_dict, best_n_estimators)` — ready to merge into the timeframe's model params

Also moved `_train_lgbm_fold`, `_train_catboost_fold`, `_train_xgboost_fold` from `tune_hyperparams.py` into `common/tuning.py` as `train_lgbm_fold`, `train_catboost_fold`, `train_xgboost_fold` and exposed them in `TRAIN_FOLD_FNS` registry.

**`common/backtest_framework.py`** — `run_tree_backtest()` accepts a new optional `inline_tuner_fn` parameter:
```python
inline_tuner_fn: Callable[[full_grid, feature_cols, cat_cols, train_end], dict] | None = None
```
When provided, each timeframe calls the tuner before training and uses `effective_params` instead of the static `model_params`.

**`scripts/run_backtest*.py`** — All three scripts accept new CLI flags:
- `--tune-inline` — enable per-timeframe causal tuning
- `--tune-n-trials N` — override trial count per timeframe
- `--tune-config PATH` — override tuning YAML path
- Mutual exclusion enforced: `--params-file` and `--tune-inline` cannot be combined

**`config/hyperparameter_tuning.yaml`** — Added:
```yaml
inline_n_trials: 20    # trials per timeframe
inline_n_splits: 3     # CV folds per timeframe
```

**Makefile** — New targets:
```bash
make backtest-lgbm-cluster-tuned       # LGBM per-cluster with inline tuning
make backtest-catboost-cluster-tuned   # CatBoost per-cluster with inline tuning
make backtest-xgboost-cluster-tuned    # XGBoost per-cluster with inline tuning
```

**Performance note:** Each of the 10 timeframes runs 20 Optuna trials × 3 CV folds = 60 model fits per timeframe (600 total vs. 250 for global one-shot tuning). Expect ~2–3× longer runtime compared to an untuned backtest. The trade-off is genuine out-of-sample accuracy with no future leakage.

**Two-mode workflow:**
- **Production scoring** (tune once on all history, apply to future): `make tune-lgbm && make backtest-lgbm-cluster ARGS="--params-file data/tuning/best_params_lgbm.json"`
- **Honest backtesting** (per-timeframe causal tuning): `make backtest-lgbm-cluster-tuned`

---

## Open Issues

*Add new issues below using the PL-NNN format.*
