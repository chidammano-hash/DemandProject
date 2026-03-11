# Feature 44: Algorithm Configuration & Simplification

## Status
Implemented

## Objective

Consolidate the backtest configuration surface from scattered CLI flags into a single declarative YAML config file (`config/algorithm_config.yaml`), and simplify the model portfolio to three per-cluster tree-based algorithms (LGBM, CatBoost, XGBoost). All previously available CLI flags (`--recursive`, `--shap-select`, `--tune-inline`, `--params-file`, etc.) are now expressed as config keys, eliminating the need to pass flags at invocation time.

---

## What Changed

### Removed Algorithms

The following backtest scripts and all their associated Makefile targets were deleted:

| Removed Script | Removed Model IDs |
|---|---|
| `scripts/run_backtest_prophet.py` | `prophet_global`, `prophet_cluster`, `prophet_pooled` |
| `scripts/run_backtest_statsforecast.py` | `statsforecast_global`, `statsforecast_cluster`, `statsforecast_pooled` |
| `scripts/run_backtest_neuralprophet.py` | `neuralprophet_global`, `neuralprophet_cluster`, `neuralprophet_pooled` |
| `scripts/run_backtest_patchtst.py` | `patchtst_global`, `patchtst_cluster`, `patchtst_transfer` |
| `scripts/run_backtest_deepar.py` | `deepar_global`, `deepar_cluster`, `deepar_transfer` |

### Supported Training Strategies

Each algorithm now supports a `cluster_strategy` config key:

| Strategy | Description | Model ID Example |
|---|---|---|
| `per_cluster` (default) | One model per `ml_cluster` partition; `ml_cluster` kept as a hard feature (constant within each partition, provides cluster identity signal) | `lgbm_cluster` |
| `global` | One model trained on ALL data; `ml_cluster` kept as a hard feature (provides cluster identity signal across the full dataset) | `lgbm_global` |

> **Key convention:** `ml_cluster` is always a **hard feature** — it is never stripped from `feature_cols` in either strategy. In `per_cluster` mode it provides a constant identity signal; in `global` mode it provides the inter-cluster discrimination signal.

### Kept Algorithms

| Script | Default Model ID | Strategies |
|---|---|---|
| `scripts/run_backtest.py` | `lgbm_cluster` | per_cluster, global |
| `scripts/run_backtest_catboost.py` | `catboost_cluster` | per_cluster, global |
| `scripts/run_backtest_xgboost.py` | `xgboost_cluster` | per_cluster, global |

Champion selection and ceiling computation are unchanged.

---

## New Config File: `config/algorithm_config.yaml`

Controls all algorithm behavior previously specified via CLI flags. Each backtest script reads its section by name.

### Structure

```yaml
lgbm:
  cluster_strategy: "per_cluster"  # "per_cluster" or "global"
  recursive: false           # Recursive multi-step inference (Feature 43)
  shap_select: false         # SHAP-based feature selection per timeframe (Feature 42)
  shap_threshold: 0.95       # Cumulative importance threshold (0.0–1.0)
  shap_top_n: null           # Exact top-N features (null = use threshold)
  shap_sample_size: 500      # Rows sampled for SHAP computation
  tune_inline: false         # Per-timeframe causal Optuna tuning (PL-002)
  params_file: null          # Path to pre-tuned params JSON (null = use defaults)
  # Default model hyperparameters (used when params_file is null and tune_inline is false)
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 31
  min_child_samples: 20

catboost:
  cluster_strategy: "per_cluster"
  recursive: false
  shap_select: false
  shap_threshold: 0.95
  shap_top_n: null
  shap_sample_size: 500
  tune_inline: false
  params_file: null
  iterations: 500
  learning_rate: 0.05
  depth: 6
  l2_leaf_reg: 3.0

xgboost:
  cluster_strategy: "per_cluster"
  recursive: false
  shap_select: false
  shap_threshold: 0.95
  shap_top_n: null
  shap_sample_size: 500
  tune_inline: false
  params_file: null
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 6
  min_child_weight: 5
  subsample: 0.8
  colsample_bytree: 0.8
```

### Config Keys Reference

| Key | Type | Default | Description |
|---|---|---|---|
| `cluster_strategy` | str | `"per_cluster"` | Training strategy: `"per_cluster"` (one model per ml_cluster partition) or `"global"` (one model on all data). `ml_cluster` is always a hard feature in both modes. |
| `recursive` | bool | `false` | Enable recursive multi-step inference (Feature 43). Each predict month is scored individually; model's prediction for month T is written back as `qty_lag_1` for month T+1. |
| `shap_select` | bool | `false` | Enable SHAP-based per-timeframe feature selection (Feature 42). Trains initial model → computes SHAP → selects features → retrains. |
| `shap_threshold` | float | `0.95` | Cumulative importance threshold. Features covering this fraction of total SHAP mass are selected. Ignored if `shap_top_n` is set. |
| `shap_top_n` | int or null | `null` | Select exactly this many top features by SHAP importance. Overrides `shap_threshold`. |
| `shap_sample_size` | int | `500` | Number of rows sampled for SHAP computation per timeframe. |
| `tune_inline` | bool | `false` | Per-timeframe causal Optuna tuning (PL-002). Each timeframe runs a mini Optuna study on only data available up to that training cutoff. Mutually exclusive with `params_file`. |
| `params_file` | str or null | `null` | Path to a pre-tuned params JSON from `make tune-lgbm` / etc. Mutually exclusive with `tune_inline`. |
| Default hyperparams | varies | see above | Algorithm-specific defaults used when `params_file` is null and `tune_inline` is false. |

---

## Updated Backtest Scripts

### Simplified CLI Interface

Each script now accepts only three arguments:

| Flag | Description |
|---|---|
| `--config PATH` | Path to algorithm_config.yaml (default: `config/algorithm_config.yaml`) |
| `--model-id STR` | Override the output model ID (default: `lgbm_cluster` / `catboost_cluster` / `xgboost_cluster`) |
| `--n-timeframes INT` | Number of expanding timeframes to run (default: 10, range 1–10) |

All feature flags (`--recursive`, `--shap-select`, `--shap-top-n`, `--shap-threshold`, `--shap-sample-size`, `--tune-inline`, `--params-file`) are now read from the config file only.

### Internal Changes

`run_tree_backtest()` in `common/backtest_framework.py` was updated:

- Removed `train_fn_global` and `train_fn_transfer` parameters — only `train_fn_per_cluster` remains.
- Removed `transfer_kwargs` parameter.
- `_predict_single_month()` signature simplified to `(models: dict, predict_data, feature_cols)` — `cluster_strategy` parameter removed (per-cluster is the only strategy, so dispatch is always dict-based).

---

## Updated Makefile

### New Targets (simplified)

```makefile
make backtest-lgbm          # Run LGBM per-cluster backtest (reads config/algorithm_config.yaml)
make backtest-catboost      # Run CatBoost per-cluster backtest
make backtest-xgboost       # Run XGBoost per-cluster backtest
make backtest-all           # Run all three sequentially (LGBM → CatBoost → XGBoost)
make backtest-all-parallel  # Run all three in parallel; per-model logs in data/backtest/logs/
```

### Removed Targets

All of the following Makefile targets were deleted:

- `backtest-lgbm-cluster`, `backtest-lgbm-transfer`, `backtest-lgbm-recursive`, `backtest-lgbm-cluster-recursive`, `backtest-lgbm-transfer-recursive`
- `backtest-lgbm-shap`, `backtest-lgbm-cluster-shap`, `backtest-lgbm-transfer-shap`
- `backtest-lgbm-cluster-tuned`
- `backtest-catboost-cluster`, `backtest-catboost-transfer`, `backtest-catboost-recursive`, `backtest-catboost-cluster-recursive`, `backtest-catboost-transfer-recursive`
- `backtest-catboost-shap`, `backtest-catboost-cluster-shap`, `backtest-catboost-transfer-shap`
- `backtest-catboost-cluster-tuned`
- `backtest-xgboost-cluster`, `backtest-xgboost-transfer`, `backtest-xgboost-recursive`, `backtest-xgboost-cluster-recursive`, `backtest-xgboost-transfer-recursive`
- `backtest-xgboost-shap`, `backtest-xgboost-cluster-shap`, `backtest-xgboost-transfer-shap`
- `backtest-xgboost-cluster-tuned`
- `backtest-prophet`, `backtest-prophet-cluster`, `backtest-prophet-pooled`
- `backtest-statsforecast`, `backtest-statsforecast-cluster`, `backtest-statsforecast-pooled`
- `backtest-neuralprophet`, `backtest-neuralprophet-cluster`, `backtest-neuralprophet-pooled`
- `backtest-patchtst`, `backtest-patchtst-cluster`, `backtest-patchtst-transfer`
- `backtest-deepar`, `backtest-deepar-cluster`, `backtest-deepar-transfer`

---

## Config-Driven Workflow

### Enabling Features

All options are controlled by editing `config/algorithm_config.yaml` before running a backtest:

**Enable SHAP feature selection for LGBM:**
```yaml
lgbm:
  shap_select: true
  shap_threshold: 0.95
```
Then run: `make backtest-lgbm`

**Enable recursive inference for CatBoost:**
```yaml
catboost:
  recursive: true
```
Then run: `make backtest-catboost`

**Enable inline tuning for XGBoost (honest backtesting, PL-002):**
```yaml
xgboost:
  tune_inline: true
```
Then run: `make backtest-xgboost`

**Apply pre-tuned params for LGBM:**
```yaml
lgbm:
  params_file: data/tuning/best_params_lgbm.json
```
Then run: `make backtest-lgbm`

**Combine SHAP + inline tuning:**
```yaml
lgbm:
  shap_select: true
  tune_inline: true
```
Then run: `make backtest-lgbm`

---

## Tests Updated

`tests/unit/test_backtest_recursive.py` was updated to reflect the simplified `_predict_single_month()` signature (no `cluster_strategy` parameter):

- Removed tests for global and transfer routing branches (those code paths no longer exist)
- Remaining tests verify per-cluster dict dispatch and NaN filling

**Backend test count after Feature 44:** 512 passed.

---

## Compatibility Notes

- **Champion selection:** Unchanged. `config/model_competition.yaml` still references `lgbm_cluster`, `catboost_cluster`, `xgboost_cluster` as the competing models — these model IDs continue to work.
- **Backtest loading:** `make backtest-load MODEL=lgbm_cluster` and `make backtest-load-all` are unchanged.
- **Cleanup:** `make backtest-clean` and `make forecast-clean` are unchanged.
- **Hyperparameter tuning:** `make tune-lgbm`, `make tune-catboost`, `make tune-xgboost`, `make tune-all` are unchanged. The output JSON files are consumed via `params_file` key in algorithm_config.yaml.
- **SHAP API:** The 4 REST endpoints under `/forecast/shap/` are unchanged — they serve CSV outputs regardless of how backtests are run.
- **Existing backtest data:** Any previously loaded model predictions for removed model IDs (e.g., `prophet_global`) remain in the database until explicitly cleaned with `make backtest-clean MODELS="prophet_global"`.

---

## Design Rationale

1. **Reduced complexity:** The previous model portfolio had 25+ model IDs across 8 frameworks. Three per-cluster tree-based models (LGBM, CatBoost, XGBoost) provide the best accuracy-to-maintenance ratio and are the only models used in champion selection.

2. **Config over flags:** CLI flags created implicit state that was easy to forget between runs. A config file makes the full algorithm behavior explicit and version-controllable.

3. **Fewer Makefile targets:** 30+ backtest targets collapsed into 4. The active configuration is the source of truth for feature options, not the make invocation.

4. **Per-cluster only:** Global models use `ml_cluster` as a categorical feature but train on all data, offering marginal accuracy benefit over per-cluster while adding framework complexity. Transfer learning was removed for the same reason — per-cluster training on sufficient data outperforms transfer fine-tuning in practice.
