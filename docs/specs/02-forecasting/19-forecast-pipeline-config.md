# Forecast Pipeline Config Consolidation

> Single source of truth for the ML forecast pipeline -- algorithm roster, backtest settings, tuning parameters, champion selection, and production forecast configuration in one file.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (config file only) |
| **Key Files** | `config/forecasting/forecast_pipeline_config.yaml`, `common/core/utils.py` (`load_forecast_pipeline_config`, `get_algorithm_roster`, `get_competing_model_ids`, `get_forecastable_model_ids`) |

---

## Problem

Pipeline configuration was fragmented across 4 separate YAML files, each governing a different stage of the ML pipeline. When adding a new algorithm or changing pipeline behavior, developers had to update multiple files and cross-reference settings between them. Misalignment between files caused silent failures. There was no single place to see which algorithms participate in which pipeline stages.

### Legacy Config Files (REMOVED)

| File | What it governed | Status |
|---|---|---|
| `config/model_competition.yaml` | Champion selection strategy, competing models, metric, lag mode | **REMOVED** -- settings now in `forecast_pipeline_config.yaml` `champion` section. |
| `config/lgbm_tuning_config.yaml` | Tuning run tracking, backup dir, comparison thresholds | **REMOVED** -- settings now in `forecast_pipeline_config.yaml` `tracking` section. |
| `config/production_forecast_config.yaml` | Production forecast horizon, CI bands, model registry, scheduler | **REMOVED** -- settings now in `forecast_pipeline_config.yaml` `production_forecast` section. |
| `config/backtest_sampling_config.yaml` | DFU sampling for backtests | **REMOVED** -- settings now in `forecast_pipeline_config.yaml` `backtest_sampling` section. |
| `config/model_tuning_config.yaml` | Unused -- was never loaded by any script | Deleted. |
| `config/baseline_seeding.yaml` | Unused -- was never loaded by any script | Deleted. |
| `config/fva_config.yaml` | Unused -- was never loaded by any script | Deleted. |
| `config/reporting_config.yaml` | Unused -- was never loaded by any script | Deleted. |
| `config/demand_signals_external_config.yaml` | Unused -- was never loaded by any script | Deleted. |

## Solution

A single master config file (`config/forecasting/forecast_pipeline_config.yaml`) consolidates all pipeline settings. It introduces a new `algorithms` section -- a master roster of all 12 algorithms with per-algorithm lifecycle flags that control which pipeline stages each algorithm participates in. Helper functions in `common/core/utils.py` provide filtered access to the roster.

Model-specific hyperparameters (learning_rate, n_estimators, etc.) are now inline under `algorithms.<model_id>.params` in the master config. Use `get_algorithm_params(model_id)` from `common/core/utils.py` to retrieve them.

---

## Config Structure

### Algorithm Roster

The `algorithms` section is the master list of all algorithms in the pipeline. Each entry has lifecycle flags that control participation in pipeline stages.

```yaml
algorithms:
  lgbm_cluster:
    type: tree
    enabled: true
    tune: true        # Include in hyperparameter tuning
    backtest: true     # Include in expanding-window backtest
    compete: true      # Include in champion model selection
    forecast: true     # Eligible for production forecast (has .pkl artifacts)
    expert: false      # Available for expert system archetype routing
    params:            # Inline hyperparameters (formerly in algorithm_config.yaml)
    output_dir: data/backtest/lgbm_cluster
```

**Lifecycle flags:**

| Flag | Pipeline Stage | Description |
|---|---|---|
| `tune` | `make tune-all` | Include in Bayesian hyperparameter tuning (Optuna) |
| `backtest` | `make backtest-all` | Include in expanding-window backtest (10 timeframes) |
| `compete` | `make champion-select` | Include in champion model selection horse race |
| `forecast` | `make forecast-generate` | Eligible for production forecast inference (requires `.pkl` model artifacts) |
| `expert` | Expert panel | Available for expert system archetype routing |

Setting `enabled: false` disables an algorithm across ALL stages.

### Cluster Strategy

Tree and statistical algorithms include a `cluster_strategy` field that controls how backtesting partitions the data:

```yaml
algorithms:
  lgbm_cluster:
    type: tree
    cluster_strategy: per_cluster   # one model per ml_cluster
    # ...
  chronos2_enriched:
    type: foundation
    # no cluster_strategy — foundation/DL models always run globally
```

| Value | Behavior |
|---|---|
| `per_cluster` | Train one model per `ml_cluster` value. Default for tree/statistical algorithms. |
| `global` | Train a single model on all data. Used when clustering is disabled or explicitly configured. |

**Resolution order**: `forecast_pipeline_config.yaml` algorithm entry > default `"per_cluster"`.

When `clustering.enabled` is `false` (see below), backtest scripts auto-fall back to `global` regardless of the per-algorithm `cluster_strategy` setting.

Algorithms with `cluster_strategy`:


### All 10 Algorithms

| Algorithm | Type | tune | backtest | compete | forecast | expert | cluster_strategy |
|---|---|---|---|---|---|---|---|
| `lgbm_cluster` | tree | yes | yes | yes | yes | no | per_cluster |
| `chronos2_enriched` | foundation | no | yes | yes | yes | no | — |
| `mstl` | statistical | no | yes | yes | yes | yes | — |
| `nbeats` | deep_learning | no | yes | yes | yes | yes | — |
| `nhits` | deep_learning | no | yes | yes | yes | no | — |

### Per-Algorithm Backtest Config Keys

Beyond the lifecycle flags above, each tree/statistical algorithm entry accepts backtest-level
behavioral keys, read from `algorithms.<model_id>.params`. Edit
`config/forecasting/forecast_pipeline_config.yaml` under `algorithms.<model_id>.params`, then run
applies cluster strategy, SHAP, tuning, recursive mode, and hyperparameters automatically. No CLI flags
are needed; the config file is the single source of truth.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `recursive` | bool | false | Enable recursive multi-step inference. Each predict month is scored individually; the model's prediction for month T becomes `qty_lag_1` for month T+1. |
| `shap_select` | bool | false | Enable multi-stage per-timeframe feature selection (see [Feature Selection Pipeline](28-feature-selection-pipeline.md)). |
| `shap_threshold` | float | 0.90 | SHAP cumulative importance threshold (Stage 3). Ignored if `shap_top_n` is set. |
| `shap_top_n` | int/null | null | Select exactly this many top features. Overrides `shap_threshold`. |
| `shap_sample_size` | int | 500 | Rows sampled for SHAP computation per timeframe. |
| `correlation_filter` | bool | false | Enable correlation pre-filter (Stage 2). Drops the lower-variance member of pairs above the threshold. |
| `correlation_threshold` | float | 0.98 | Absolute Pearson correlation threshold for Stage 2. |
| `variance_filter` | bool | false | Enable near-zero variance filter (Stage 1). |
| `variance_threshold` | float | 0.01 | Relative variance threshold for Stage 1 (fraction of range squared). |
| `tune_inline` | bool | false | Per-timeframe causal Optuna tuning. Mutually exclusive with `params_file`. |
| `params_file` | string/null | null | Path to pre-tuned params JSON from `make tune-*`. Mutually exclusive with `tune_inline`. |
| Algorithm-specific hyperparameters | varies | see `algorithms.<model_id>.params` | Default hyperparameters used when `params_file` is null and `tune_inline` is false. |

**Common configurations:**

Basic per-cluster backtest (default -- `cluster_strategy` is a top-level key on the algorithm entry, not under `params`):
```yaml
algorithms:
  lgbm_cluster:
    cluster_strategy: "per_cluster"
```

Apply pre-tuned parameters:
```yaml
algorithms:
  lgbm_cluster:
    params:
      params_file: data/tuning/best_params_lgbm.json
```

SHAP + inline tuning (honest backtesting):
```yaml
algorithms:
  lgbm_cluster:
    params:
      shap_select: true
      tune_inline: true
```

Recursive with pre-tuned params:
```yaml
algorithms:
  lgbm_cluster:
    params:
      recursive: true
      params_file: data/tuning/best_params_lgbm.json
```

**What was removed:** 30+ granular Makefile targets (e.g., `backtest-lgbm-cluster-shap`,
file replaces all of them. Five algorithm families (Prophet, StatsForecast, NeuralProphet, PatchTST,
DeepAR) were also removed along with their Makefile targets; the algorithm roster above provides the
best accuracy-to-maintenance ratio.

### Backtest Settings

```yaml
backtest:
  n_timeframes: 10
  embargo_months: 0          # Preserve natural lag 0 (next-month prediction)
  forecast_horizon: 6
  early_stop_pct: 0.03
  shap_retrain_threshold: 0.10
  n_seeds: 1
  tweedie_variance_power: 1.5
  intermittent_threshold: 0.7
  lumpy_threshold: 0.3
  output_dir: data/backtest
```

Backtest embargo and tuning CV gap are independent. Backtests use `0` because prediction already starts after the training month and lag-0 evidence is required. Tuning may use a positive `gap_months` as a stricter model-selection leakage guard.

### Tuning Settings

```yaml
tuning:
  n_trials: 50
  n_splits: 5
  gap_months: 2              # Tuning-only CV leakage guard; independent of backtest embargo
  val_months_per_fold: 3
  min_train_months: 13
  early_stopping_rounds: 50
  n_estimators_max: 2000
  n_estimators_buffer: 1.1
  random_seed: 42
  output_dir: data/tuning
  search_space_ref: hyperparameter_tuning.yaml   # Optuna search spaces remain in separate file
```

### Champion Selection

```yaml
champion:
  strategy: hybrid_warmup
  strategy_params:
    min_prior_months: 3
    warmup_strategy: rolling
    warmup_window: 2
    warmup_min_prior: 1
    primary_strategy: adaptive_ensemble
    primary_top_k: 3
    primary_weight_method: inverse_wape
  metric: accuracy_pct
  lag: execution
  min_sku_rows: 3
  min_dfu_rows: 3
  champion_model_id: champion
  meta_learner:
    model_type: random_forest
    n_estimators: 200
    max_depth: 15
    test_months: 3
    performance_window: 6
```

The competing models list is no longer explicit -- it is derived from `algorithms[*].compete == true`.

### Production Forecast

```yaml
production_forecast:
  horizon_months: 24           # T+1 through T+24 (was 18)
  lookback_months: 36          # 3 years of sales history (was 24)
  min_history_months: 12       # Below this -> cold-start routing
  cold_start_min_months: 3     # Absolute floor -- skip DFUs below this
  fallback_model_id: lgbm_cluster
  recursive: true
  plan_version_format: "%Y-%m"
  keep_last_n_versions: 3
  confidence_interval:
    enabled: true
    residual_lag: 0
    min_residual_months: 6
    z_lower: 1.282             # P10
    z_upper: 1.282             # P90
    horizon_scaling: sqrt
    sigma_floor: 1.0
    sigma_cap_multiplier: 3.0
  model_registry:
    base_path: "data/models"
  scheduler:
    job_type: generate_production_forecast
    cron: "0 6 2 * *"
```

### Run Tracking

```yaml
tracking:
  backup_dir: data/backtest/tuning_archive
  auto_register: true
  auto_compare_to_latest: true
  improvement_threshold_bps: 5
  verdicts:
    improved_min_delta_accuracy: 0.05
    degraded_max_delta_accuracy: -0.05
  max_archive_runs: 50
```

### Clustering

The `clustering` section is the master switch for the clustering pipeline. When `enabled: false`, all backtest scripts auto-fall back to `global` strategy regardless of per-algorithm `cluster_strategy` settings.

```yaml
clustering:
  enabled: true
  config_ref: cluster_experiment table
  tuning_profiles_ref: cluster_tuning_profiles.yaml
  experiment_templates_ref: cluster_experiment_templates.yaml
  steps:
    generate_features: true
    train_model: true
    label_clusters: true
    update_db: true
  artifacts:
    features_csv: data/staged/clustering_features.csv
    output_dir: data/clustering
  db_target:
    table: dim_sku
    column: ml_cluster
```

| Field | Description |
|---|---|
| `enabled` | Master switch. `false` disables clustering and forces all backtests to global strategy. |
| `config_ref` | Reference to the detailed clustering hyperparameter config file. |
| `tuning_profiles_ref` | Reference to cluster tuning profiles config. |
| `experiment_templates_ref` | Reference to cluster experiment templates config. |
| `steps` | Toggle individual clustering pipeline steps (feature generation, model training, labeling, DB update). |
| `artifacts.features_csv` | Path to the generated clustering features CSV. |
| `artifacts.output_dir` | Directory for clustering model artifacts. |
| `db_target.table` | Target table for cluster assignments. |
| `db_target.column` | Target column for cluster assignments. |

### Backtest Sampling

The `backtest_sampling` section is the sole source for sampling configuration (the legacy `backtest_sampling_config.yaml` has been deleted). It controls how DFUs are sampled for backtesting.

```yaml
backtest_sampling:
  enabled: true
  default_target_n: 5000
  default_method: proportional
  min_per_cluster: 10
  max_target_n: 20000
  seed: 42
  representativeness_threshold: 0.05
```

| Field | Description |
|---|---|
| `enabled` | Enable/disable DFU sampling for backtests. When `false`, all DFUs are used. |
| `default_target_n` | Default number of DFUs to sample. |
| `default_method` | Sampling method (`proportional` maintains cluster distribution). |
| `min_per_cluster` | Minimum DFUs per cluster to ensure representation. |
| `max_target_n` | Upper bound on sample size. |
| `seed` | Random seed for reproducibility. |
| `representativeness_threshold` | Maximum acceptable deviation from true cluster proportions. |

**No fallback**: `common/ml/backtest_sampler.py` reads sampling config exclusively from `forecast_pipeline_config.yaml` `backtest_sampling`.

### Pipeline Stages

The `pipeline` section documents the end-to-end execution order and dependencies for the ML forecast pipeline.

```yaml
pipeline:
  stages: [clustering, backtest, load, champion, forecast]
  auxiliary: [seasonality, variability]
```

| Field | Description |
|---|---|
| `stages` | Ordered list of main pipeline stages. Each stage depends on the previous one completing. |
| `auxiliary` | Stages that run independently and feed into backtest feature engineering. |

Execution flow: `clustering` -> `backtest` -> `load` -> `champion` -> `forecast`. The `seasonality` and `variability` auxiliary stages (see [SKU Feature Engineering](../03-demand-intelligence/02-sku-feature-engineering.md)) can run in parallel with clustering and are consumed during the backtest stage.

---

## Helper Functions

Located in `common/core/utils.py`:

| Function | Description |
|---|---|
| `load_forecast_pipeline_config()` | Load and return the full master config dict |
| `get_algorithm_roster(stage=None)` | Return algorithm entries, optionally filtered by lifecycle stage (e.g., `stage="compete"` returns only algorithms with `compete: true`) |
| `get_competing_model_ids()` | Shortcut: return list of model IDs with `compete: true` |
| `get_forecastable_model_ids()` | Shortcut: return list of model IDs with `forecast: true` |
| `is_clustering_enabled()` | Check `clustering.enabled` in the pipeline config. Returns `True` if clustering is active, `False` otherwise. Used by backtest scripts to determine whether to use `per_cluster` or fall back to `global` strategy. |

### Usage

```python
from common.core.utils import load_forecast_pipeline_config, get_algorithm_roster, get_competing_model_ids, is_clustering_enabled

# Full config
cfg = load_forecast_pipeline_config()
horizon = cfg["production_forecast"]["horizon_months"]  # 24

# All algorithms that participate in champion selection
competing = get_competing_model_ids()

# All algorithms eligible for production forecasting
forecastable = get_forecastable_model_ids()

# Filtered roster for a specific stage
backtest_algos = get_algorithm_roster(stage="backtest")
# Returns dict of all algorithms with backtest: true

# Check if clustering is enabled (master switch)
if is_clustering_enabled():
    strategy = cfg["algorithms"]["lgbm_cluster"].get("cluster_strategy", "per_cluster")
else:
    strategy = "global"  # auto-fallback when clustering disabled
```

---

## Migration Guide

### For existing code

The legacy config files (`model_competition.yaml`, `lgbm_tuning_config.yaml`, `production_forecast_config.yaml`, `backtest_sampling_config.yaml`, `algorithm_config.yaml`) have been **deleted**. All code now reads from the master `forecast_pipeline_config.yaml`.

### For new code

New code should use the master config:

```python
# OLD (still works but deprecated):
from common.core.utils import load_config
cfg = load_forecast_pipeline_config()
models = cfg["competition"]["models"]

# NEW (preferred):
from common.core.utils import load_forecast_pipeline_config, get_competing_model_ids
cfg = load_forecast_pipeline_config()
models = get_competing_model_ids()
```

### Key parameter changes

| Parameter | Old Value | New Value | Location |
|---|---|---|---|
| `horizon_months` | 18 | 24 | `production_forecast.horizon_months` |
| `lookback_months` | 24 | 36 | `production_forecast.lookback_months` |
| `embargo_months` | 1 | 0 | `backtest.embargo_months` |

---

## Backward Compatibility

All legacy config files have been deleted. The master `forecast_pipeline_config.yaml` is the sole source for all pipeline settings. The `config_key` field has been removed from algorithm entries; hyperparameters are now inline under `algorithms.<model_id>.params`. Use `get_algorithm_params(model_id)` to retrieve them programmatically.

---

## Dependencies

- [Champion Selection](./07-champion-selection.md) -- champion strategy governed by `champion` section
- [Production Forecast](./08-production-forecast.md) -- inference settings governed by `production_forecast` section
- [Unified Model Tuning Studio](./11-unified-model-tuning-v2.md) -- tracking settings governed by `tracking` section
- `cluster_experiment` table (promoted row) -- detailed clustering hyperparameters, referenced by `clustering.config_ref`
- `config/forecasting/cluster_tuning_profiles.yaml` -- cluster tuning profiles, referenced by `clustering.tuning_profiles_ref`
- `config/forecasting/cluster_experiment_templates.yaml` -- experiment templates, referenced by `clustering.experiment_templates_ref`

## See Also

- [Backtest Framework](./03-backtest-framework.md) -- uses `backtest` section settings
- [Tree Models](./04-tree-models.md) -- the algorithms controlled by the per-algorithm config keys above
- [Expert Panel](./15-expert-panel-algorithm-selection.md) -- uses `expert` lifecycle flag
