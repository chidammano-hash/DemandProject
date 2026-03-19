# Algorithm Configuration

> One YAML file controls all backtest behavior -- cluster strategy, tuning, SHAP, recursive mode, hyperparameters -- so you can try different approaches without changing code.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (config file only) |
| **Key Files** | `config/algorithm_config.yaml`, `scripts/run_backtest.py`, `scripts/run_backtest_catboost.py`, `scripts/run_backtest_xgboost.py` |

---

## Problem

Before this feature, backtest options were scattered across 30+ Makefile targets and dozens of CLI flags. It was easy to forget which flags were used for the last run, and impossible to version-control the experiment configuration. Different team members running the same model with different flags produced results that couldn't be compared.

## Solution

A single declarative YAML file (`config/algorithm_config.yaml`) replaces all CLI flags. Each backtest script reads its section by algorithm name and applies the configured options. The file is checked into git, making experiments reproducible and reviewable. The three scripts now accept only `--config`, `--model-id`, and `--n-timeframes`.

## How It Works

1. Edit `config/algorithm_config.yaml` to set the desired options
2. Run `make backtest-lgbm` (or catboost/xgboost)
3. The script reads its section from the YAML file
4. All options (cluster strategy, SHAP, tuning, recursive, hyperparameters) are applied automatically
5. No CLI flags needed -- the config file is the single source of truth

## Configuration

### Full Config Structure

```yaml
lgbm:
  cluster_strategy: "per_cluster"   # "per_cluster" or "global"
  recursive: false                   # Recursive multi-step inference
  shap_select: false                 # SHAP-based feature selection
  shap_threshold: 0.95              # Cumulative SHAP mass threshold
  shap_top_n: null                  # Exact top-N features (overrides threshold)
  shap_sample_size: 500             # Rows sampled for SHAP computation
  tune_inline: false                 # Per-timeframe causal Optuna tuning
  params_file: null                  # Path to pre-tuned params JSON
  # Default hyperparameters
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 31
  min_child_samples: 20

catboost:
  cluster_strategy: "per_cluster"
  recursive: false
  shap_select: false
  # ... (same keys as lgbm)
  iterations: 500
  learning_rate: 0.05
  depth: 6
  l2_leaf_reg: 3.0

xgboost:
  cluster_strategy: "per_cluster"
  recursive: false
  shap_select: false
  # ... (same keys as lgbm)
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 6
  min_child_weight: 5
  subsample: 0.8
  colsample_bytree: 0.8
```

### Config Keys Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cluster_strategy` | string | `"per_cluster"` | `"per_cluster"` trains one model per ml_cluster; `"global"` trains one model on all data. `ml_cluster` is always a hard feature in both modes. |
| `recursive` | bool | false | Enable recursive multi-step inference. Each predict month is scored individually; model's prediction for month T becomes `qty_lag_1` for month T+1. |
| `shap_select` | bool | false | Enable SHAP-based per-timeframe feature selection. Trains initial model, computes SHAP, selects features, retrains. |
| `shap_threshold` | float | 0.95 | Cumulative importance threshold. Ignored if `shap_top_n` is set. |
| `shap_top_n` | int/null | null | Select exactly this many top features. Overrides `shap_threshold`. |
| `shap_sample_size` | int | 500 | Rows sampled for SHAP computation per timeframe. |
| `tune_inline` | bool | false | Per-timeframe causal Optuna tuning. Mutually exclusive with `params_file`. |
| `params_file` | string/null | null | Path to pre-tuned params JSON from `make tune-*`. Mutually exclusive with `tune_inline`. |
| Algorithm-specific | varies | see above | Default hyperparameters used when `params_file` is null and `tune_inline` is false. |

### Common Configurations

**Basic per-cluster backtest (default):**
```yaml
lgbm:
  cluster_strategy: "per_cluster"
```

**Apply pre-tuned parameters:**
```yaml
lgbm:
  params_file: data/tuning/best_params_lgbm.json
```

**SHAP + inline tuning (honest backtesting):**
```yaml
lgbm:
  shap_select: true
  tune_inline: true
```

**Recursive with pre-tuned params:**
```yaml
lgbm:
  recursive: true
  params_file: data/tuning/best_params_lgbm.json
```

## Pipeline

| Target | Description |
|--------|-------------|
| `make backtest-lgbm` | Run LGBM (reads config) |
| `make backtest-catboost` | Run CatBoost (reads config) |
| `make backtest-xgboost` | Run XGBoost (reads config) |
| `make backtest-all` | All three sequentially |
| `make backtest-all-parallel` | All three in parallel |

### What Was Removed

30+ granular Makefile targets were deleted (e.g., `backtest-lgbm-cluster-shap`, `backtest-catboost-cluster-tuned`, `backtest-xgboost-transfer-recursive`). The config file replaces all of them.

Five algorithm families were also removed (Prophet, StatsForecast, NeuralProphet, PatchTST, DeepAR) along with all their Makefile targets. The three tree-based models provide the best accuracy-to-maintenance ratio.

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- reads this config
- [Tree Models](./04-tree-models.md) -- the algorithms controlled by this config
- [Advanced Backtest](./05-advanced-backtest.md) -- the capabilities activated by config keys

## See Also

- [Champion Selection](./07-champion-selection.md) -- runs after backtests complete
