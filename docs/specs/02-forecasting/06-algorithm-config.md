# Algorithm Configuration

> One YAML file controls all backtest behavior -- cluster strategy, tuning, SHAP, recursive mode, hyperparameters -- so you can try different approaches without changing code.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (config file only) |
| **Key Files** | `config/forecast_pipeline_config.yaml` (algorithms section), `scripts/run_backtest.py`, `scripts/run_backtest_catboost.py`, `scripts/run_backtest_xgboost.py` |

---

## Problem

Before this feature, backtest options were scattered across 30+ Makefile targets and dozens of CLI flags. It was easy to forget which flags were used for the last run, and impossible to version-control the experiment configuration. Different team members running the same model with different flags produced results that couldn't be compared.

## Solution

Algorithm configuration is consolidated into `config/forecast_pipeline_config.yaml` under the `algorithms` section. Each algorithm entry contains lifecycle flags, behavioral options, and inline hyperparameters (under `.params`). The legacy `config/algorithm_config.yaml` has been deleted. Use `get_algorithm_params(model_id)` from `common/core/utils.py` to retrieve hyperparameters programmatically.

## How It Works

1. Edit `config/forecast_pipeline_config.yaml` under `algorithms.<model_id>` to set the desired options
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
| `cluster_strategy` | string | `"per_cluster"` | `"per_cluster"` trains one model per `ml_cluster` partition; `"global"` trains one model on all data. `ml_cluster` is used for partitioning only, not as a model feature (removed to prevent leakage). |
| `recursive` | bool | false | Enable recursive multi-step inference. Each predict month is scored individually; model's prediction for month T becomes `qty_lag_1` for month T+1. |
| `shap_select` | bool | false | Enable multi-stage per-timeframe feature selection (see [spec 23](23-feature-selection-pipeline.md)). |
| `shap_threshold` | float | 0.95 | SHAP cumulative importance threshold (Stage 3). Ignored if `shap_top_n` is set. |
| `shap_top_n` | int/null | null | Select exactly this many top features. Overrides `shap_threshold`. |
| `shap_sample_size` | int | 500 | Rows sampled for SHAP computation per timeframe. |
| `correlation_filter` | bool | true | Enable correlation pre-filter (Stage 2). Drops lower-variance member of pairs > threshold. |
| `correlation_threshold` | float | 0.95 | Absolute Pearson correlation threshold for Stage 2. |
| `variance_filter` | bool | true | Enable near-zero variance filter (Stage 1). |
| `variance_threshold` | float | 0.01 | Relative variance threshold for Stage 1 (fraction of range squared). |
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

- [Forecast Pipeline Config](./19-forecast-pipeline-config.md) -- master pipeline config with algorithm roster and lifecycle flags; governs WHICH models run at each stage
- [Champion Selection](./07-champion-selection.md) -- runs after backtests complete
