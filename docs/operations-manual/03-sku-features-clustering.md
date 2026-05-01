# 03 — SKU Features & Clustering

This section covers the two pre-modelling pipelines that profile every SKU and segment the catalogue into demand-pattern clusters. Both pipelines feed downstream backtesting, tuning, and inventory planning.

> **Terminology note.** Both pipelines operate on **SKUs** (the `item_id + customer_group + loc` composite identified by `dim_sku.sku_ck`), not legacy DFUs. Clustering documentation, UI labels, and code consistently use "SKU" — keep this in mind when reading older specs.

---

## 1. SKU Features Pipeline

### 1.1 Purpose

The SKU features pipeline computes a unified set of time-series features for every SKU from `fact_sales_monthly` and persists them into the `dim_sku` dimension table. These features are the single source of truth consumed by:

- The **clustering pipeline** (`make cluster-all`) — reads pre-computed features, never raw sales.
- **Per-cluster tuning** matching (`mean_demand`, `cv_demand`, `seasonal_amplitude`, etc.).
- **Inventory planning** (variability class, intermittency, seasonality profile).
- **UI surfaces** (Cluster tab, SKU drill-downs, ABC/XYZ tagging).

### 1.2 Make targets

```bash
make features-compute      # primary entry point
make seasonality-all       # alias -> features-compute
make variability-all       # alias -> features-compute
```

All three resolve to the same script:
`/Users/manoharchidambaram/projects/DemandProject/scripts/ml/compute_sku_features.py`

The aliases exist for backward compatibility with older runbooks; new automation should call `make features-compute`.

### 1.3 What is computed

The pipeline calls `compute_all_sku_features()` in
`/Users/manoharchidambaram/projects/DemandProject/common/ml/sku_features/compute.py`,
which delegates per-SKU work to `_compute_features_for_group()` from
`/Users/manoharchidambaram/projects/DemandProject/common/ml/clustering/features.py`.

The full feature set written to `dim_sku` includes:

| Group | Features |
|---|---|
| Volume | `total_demand`, `demand_mean`, `demand_p50`, `demand_p90`, `min_demand`, `max_demand`, `demand_std`, `demand_mad`, `iqr_demand`, `demand_skewness`, `demand_kurtosis` |
| Variability | `demand_cv`, `outlier_count`, `cagr`, `acceleration` |
| Trend | `trend_slope`, `trend_slope_norm`, `trend_r2`, `trend_pct_change`, `trend_direction` |
| Seasonality | `seasonality_strength`, `seasonal_amplitude`, `seasonal_r2`, `yoy_correlation`, `seasonal_index_std`, `peak_month`, `trough_month`, `peak_trough_ratio` |
| Periodicity | `periodicity_strength`, `acf_lag12` |
| Intermittency / Lifecycle | `intermittency_ratio` (zero-demand fraction), `adi`, `recency_ratio`, `total_demand_months` |
| Classifiers | `seasonality_profile` (`highly_seasonal` / `moderate_seasonal` / `non_seasonal`), `variability_class` (`low` / `moderate` / `high` / `erratic`) |

The exact name mapping from pipeline output to `dim_sku` columns lives in
`/Users/manoharchidambaram/projects/DemandProject/common/ml/sku_features/persistence.py`
(`_FEATURE_TO_COLUMNS`). Two timestamp columns are also stamped:
`features_computed_ts` and `demand_profile_ts`.

### 1.4 Configuration

Source of truth:
`/Users/manoharchidambaram/projects/DemandProject/config/sku_features_config.yaml`

```yaml
history:
  time_window_months: 36          # months of sales history loaded
  min_months_history: 1           # minimum non-null months a SKU needs to be included
seasonality:
  amplitude_threshold: 0.3        # > -> seasonal
  r2_threshold: 0.25              # > -> confirmed seasonal
  yoy_correlation_threshold: 0.3
  peak_trough_min_ratio: 1.5
variability:
  cv_thresholds:
    smooth: 0.3                   # CV < -> smooth
    erratic: 0.8                  # CV > -> erratic
  intermittency_threshold: 0.15   # zero_demand_pct > -> intermittent
trend:
  slope_threshold: 0.01
  r2_threshold: 0.25
  cagr_growing: 5.0
  cagr_declining: -5.0
parallelism:
  max_workers: 8                  # multiprocessing pool size
```

CLI flags override the config at runtime:

```bash
uv run python scripts/ml/compute_sku_features.py --workers 4 --time-window 24
uv run python scripts/ml/compute_sku_features.py --dry-run
uv run python scripts/ml/compute_sku_features.py --output-csv data/sku_features.csv
```

### 1.5 Outputs

| Destination | Contents |
|---|---|
| `dim_sku` table | All feature columns above, plus `features_computed_ts` / `demand_profile_ts` |
| `data/clustering_features.csv` | Backward-compat CSV consumed by older clustering scripts |
| Optional `--output-csv <path>` | Full feature matrix dump for inspection |

The DB write uses `psycopg3` `COPY` into a temp staging table followed by a single `UPDATE ... FROM` join — efficient for the full SKU population.

### 1.6 Runtime expectations

- Sales loader scoped to `time_window_months` ending at `get_planning_date()`.
- Multiprocessing kicks in above 500 SKUs (`workers = min(cpu_count, 8)` by default).
- Typical wall time on a full catalogue: **3–5 minutes**. The script logs a profiled summary per stage (`load_sales_from_db`, `compute_all_sku_features`, `apply_classifiers`, `write_features_to_dim_sku`).

### 1.7 Verification

```sql
-- Row coverage
SELECT COUNT(*) AS total_skus,
       COUNT(features_computed_ts) AS skus_with_features,
       MAX(features_computed_ts) AS last_computed
FROM dim_sku;

-- Distribution sanity check
SELECT seasonality_profile, COUNT(*) FROM dim_sku GROUP BY 1 ORDER BY 2 DESC;
SELECT variability_class,    COUNT(*) FROM dim_sku GROUP BY 1 ORDER BY 2 DESC;
```

A successful run logs `Pipeline complete — <N> SKUs, <M> features, <T>s elapsed`.

---

## 2. Clustering Pipeline

### 2.1 Purpose

Segments SKUs into demand-pattern clusters (e.g. `high_volume_periodic`, `low_volume_volatile`, `sparse_intermittent`) used for per-cluster model partitioning and per-cluster hyperparameter overrides during backtest and tuning.

### 2.2 Make target

```bash
make cluster-all
```

Resolves to:
`/Users/manoharchidambaram/projects/DemandProject/scripts/ml/run_cluster_pipeline.py --label "make cluster-all"`

The script `run_unified_pipeline()` does the following in one shot:

1. Reads default `feature_params` / `model_params` / `label_params` from the **currently promoted** row in `cluster_experiment` (falls back to hardcoded defaults if no promotion exists).
2. Inserts a new `cluster_experiment` row with `status = 'running'`.
3. Calls `run_scenario()` (from `scripts/run_clustering_scenario.py`) which executes:
   - **Feature load** — reads pre-computed features from `dim_sku` (NOT from `fact_sales_monthly`). The features pipeline must have been run first.
   - **K selection + KMeans training** — `find_optimal_k()` over `model_params.k_range`, then `train_kmeans()` from `common/ml/clustering/training.py`.
   - **Labeling** — `assign_cluster_labels()` from `common/ml/clustering/labeling.py` produces human-readable cluster names from centroid characteristics.
4. Updates the experiment row with `optimal_k`, `silhouette_score`, `inertia`, `cluster_sizes`, `profiles`, `k_selection_results`, `artifacts_path`.
5. **Auto-promotes** the new experiment by default — clears `is_promoted` on the old champion and sets it on the new row, then calls `promote_scenario()` which:
   - Copies `cluster_labels.csv`, `cluster_centroids.csv`, `scenario_result.json`, `cluster_metadata.json` to `data/clustering/`.
   - `UPDATE dim_sku SET ml_cluster = u.cluster_label` via a `COPY`-loaded temp table.
   - Best-effort marks `cluster_tuning_profile_state.stale = TRUE` for each new cluster name (sql/148) so the next tuning run knows to re-train.

Skip auto-promotion with `--no-promote` for a dry segmentation evaluated only via the experiment row.

### 2.3 Library layout

| Module | Responsibility |
|---|---|
| `/Users/manoharchidambaram/projects/DemandProject/common/ml/clustering/features.py` | Per-SKU time-series feature kernel (`_compute_features_for_group`, `compute_time_series_features`) |
| `/Users/manoharchidambaram/projects/DemandProject/common/ml/clustering/training.py` | `CORE_FEATURES`, `LOG_TRANSFORM_FEATURES`, `find_optimal_k`, `merge_small_clusters` |
| `/Users/manoharchidambaram/projects/DemandProject/common/ml/clustering/labeling.py` | `assign_cluster_labels` — turns centroids into human-readable names |
| `/Users/manoharchidambaram/projects/DemandProject/common/ml/clustering/scenario.py` | `generate_scenario_id`, `promote_scenario`, `get_scenario_result` |

### 2.4 Defaults

The script's hardcoded fallbacks (used when no promoted experiment exists):

```python
DEFAULT_FEATURE_PARAMS = {"time_window_months": 36, "min_months_history": 1}
DEFAULT_MODEL_PARAMS   = {"k_range": [9, 18], "min_cluster_size_pct": 2.0,
                          "use_pca": False, "pca_components": None,
                          "all_features": False}
DEFAULT_LABEL_PARAMS   = {"volume_high": 0.75, "volume_low": 0.25,
                          "cv_steady": 0.4,    "cv_volatile": 0.8,
                          "seasonality_threshold": 0.3,
                          "zero_demand_threshold": 0.15}
```

### 2.5 Runtime expectations

A full clustering run on the production SKU population typically completes in **5–10 minutes** dominated by the K-sweep over `k_range`. The script logs `Scenario completed in <T>s` and `Promoted: <N> DFUs updated`.

---

## 3. Master Switch — Disabling Clustering

The master switch lives in
`/Users/manoharchidambaram/projects/DemandProject/config/forecast_pipeline_config.yaml`:

```yaml
clustering:
  enabled: true                # master switch
  cluster_sizing:
    samples_per_feature: 1
  steps:
    generate_features: true
    train_model: true
    label_clusters: true
    update_db: true
  artifacts:
    features_csv: data/clustering_features.csv
    output_dir: data/clustering
  db_target:
    table: dim_sku
    column: ml_cluster
```

When `clustering.enabled: false`:

- All backtest scripts auto-fall back to `cluster_strategy: global` regardless of any per-algorithm `cluster_strategy` setting in the algorithm roster.
- Per-cluster tuning profile resolution is short-circuited to base params.
- The `clustering` pipeline stage in `pipeline.stages` is skipped (it is gated by `enabled_by: clustering.enabled`).

Check the flag programmatically via `is_clustering_enabled()` in `common/utils.py`.

---

## 4. Cluster Experiment Management

There is **no clustering YAML config** — `clustering_config.yaml` was deleted intentionally. All run-to-run params live in the `cluster_experiment` DB table.

### 4.1 Source of truth: `cluster_experiment` table

Each row captures one clustering attempt:

| Column | Purpose |
|---|---|
| `experiment_id`, `scenario_id`, `label`, `status` | Identity & lifecycle (`running` / `completed` / `failed`) |
| `feature_params`, `model_params`, `label_params` | JSONB — full param set for reproducibility |
| `optimal_k`, `silhouette_score`, `inertia`, `cluster_sizes`, `profiles`, `k_selection_results` | Outcome metrics |
| `is_promoted`, `promoted_at` | Exactly one row may be `is_promoted = TRUE`; that row drives `dim_sku.ml_cluster` |
| `artifacts_path`, `runtime_seconds`, `total_dfus`, `n_clusters` | Bookkeeping |

DDL: `/Users/manoharchidambaram/projects/DemandProject/sql/101_cluster_experiments.sql`.

### 4.2 UI

Create, evaluate, compare, and promote experiments from the **Cluster** tab in the React UI. Clustering is intentionally **hidden from the Jobs tab** — it is managed exclusively through the Cluster tab and `make cluster-all`.

### 4.3 Promotion semantics

- A new promotion clears `is_promoted` on the previous champion and sets it on the new row in a single transaction.
- Promotion writes `dim_sku.ml_cluster` (the only DB column propagated from clustering).
- Per-cluster tuning profile staleness flags are best-effort updated in `cluster_tuning_profile_state` so downstream tuning knows to re-train against the new partitions.

---

## 5. Per-Cluster Tuning Profiles

Per-cluster hyperparameter overrides live in:
`/Users/manoharchidambaram/projects/DemandProject/config/cluster_tuning_profiles.yaml`

Resolution happens at backtest time via `resolve_cluster_params()` in
`/Users/manoharchidambaram/projects/DemandProject/common/ml/backtest_framework.py`.

### 5.1 Two-phase matching

For each cluster being trained, the resolver walks profiles in two phases. **First match wins.**

**Phase 1 — exact `cluster_name` match.** Iterates through `cluster_profiles`. If a profile's `match_criteria.cluster_name` equals the cluster label being resolved, its `overrides` are merged on top of `base_params` and the loop returns.

```yaml
cluster_profiles:
  high_volume_periodic:
    match_criteria:
      cluster_name: high_volume_periodic     # Phase 1 match
    overrides:
      learning_rate: 0.115
      num_leaves: 57
      ...
```

**Phase 2 — statistical fallback.** If no `cluster_name` match, the resolver walks profiles in `_PROFILE_PRIORITY` order and matches against statistical thresholds (`mean_demand_min`, `cv_demand_max`, `zero_demand_pct_min`, `seasonal_amplitude_min`, `n_rows_min` / `_max`). A profile with empty `match_criteria` matches everything (default fallback).

The stats used for matching come from `compute_cluster_demand_stats()`:
`mean_demand`, `cv_demand`, `zero_demand_pct`, `seasonal_amplitude`, `n_rows`. Suffix `_min` / `_max` is the dynamic convention.

### 5.2 When to regenerate this file

`/Users/manoharchidambaram/projects/DemandProject/config/cluster_tuning_profiles.yaml` is **auto-generated** by `scripts/tune_cluster_hyperparams.py` (invoked via `make tune-clusters` or `make tune-lgbm-clusters`). After a clustering re-promotion, profiles are flagged stale; re-run per-cluster tuning to regenerate the override block.

---

## 6. Outputs and Verification

| Artifact | Location |
|---|---|
| SKU features | `dim_sku` columns (see 1.3); CSV mirror at `data/clustering_features.csv` |
| Cluster experiment metadata | `cluster_experiment` table (one row per run) |
| Promoted cluster labels | `dim_sku.ml_cluster` (TEXT) |
| Promotion artifacts | `data/clustering/cluster_labels.csv`, `cluster_centroids.csv`, `cluster_metadata.json`, `scenario_result.json` |
| Per-cluster tuning overrides | `config/cluster_tuning_profiles.yaml` |
| Profile staleness flags | `cluster_tuning_profile_state` (sql/148) |

### 6.1 Inspect

```sql
-- Currently promoted experiment
SELECT experiment_id, scenario_id, label, optimal_k, silhouette_score,
       n_clusters, cluster_sizes, promoted_at
FROM cluster_experiment
WHERE is_promoted = TRUE;

-- Recent runs
SELECT experiment_id, label, status, optimal_k, silhouette_score,
       runtime_seconds, completed_at
FROM cluster_experiment
ORDER BY experiment_id DESC LIMIT 10;

-- Cluster distribution on dim_sku
SELECT ml_cluster, COUNT(*) AS skus
FROM dim_sku
GROUP BY ml_cluster
ORDER BY skus DESC;

-- SKUs missing a cluster assignment
SELECT COUNT(*) FROM dim_sku WHERE ml_cluster IS NULL OR ml_cluster = '';
```

---

## 7. Common Gotchas

### 7.1 `ml_cluster` is METADATA, not a model feature

`ml_cluster` lives in `METADATA_COLS` in
`/Users/manoharchidambaram/projects/DemandProject/common/core/constants.py`:

```python
METADATA_COLS = {"sku_ck", "item_id", "customer_group", "loc",
                 "startdate", "qty", "_k", "ml_cluster"}
```

`get_feature_columns()` excludes anything in `METADATA_COLS` from the feature matrix. This is **deliberate to prevent data leakage** — cluster assignments are computed from full sales history, so feeding them to the model would let it peek at future statistics.

`ml_cluster` is still merged into the feature grid by `build_feature_matrix()` for two legitimate uses:
- **Per-cluster partitioning** when `cluster_strategy: per_cluster` (one model per cluster).
- **Inventory planning + UI joins** that need the cluster label.

The only exception is the `cluster_strategy: global` path in `backtest_framework.py` (lines 1214–1219), where `ml_cluster` is intentionally promoted to a categorical feature in lieu of partitioning.

### 7.2 Intermittent cluster routing

Clusters with more than 70% zero-demand rows skip the tree model entirely and fall back to a rolling-mean baseline during backtest. Configured in `forecast_pipeline_config.yaml`:

```yaml
backtest:
  baseline_intermittent: true
  baseline_intermittent_window: 12
  intermittent_threshold: 0.7         # zero_demand_pct > this -> rolling-mean baseline
  lumpy_threshold: 0.3
```

If a cluster you expected to see tree-model predictions for is silently rolling-mean, check `compute_cluster_demand_stats()` output for that cluster — its `zero_demand_pct` is likely above 0.7.

### 7.3 Clustering reads `dim_sku`, not raw sales

If `make cluster-all` produces empty or stale clusters after a data refresh, the symptom is almost always that `make features-compute` was not re-run. Clustering does **not** recompute features from `fact_sales_monthly`.

### 7.4 Clustering hidden from Jobs tab

You will not see clustering scenarios in the Jobs tab — this is by design. Manage all clustering work through the Cluster tab UI or `make cluster-all` from the CLI.

### 7.5 Scenario IDs are validated

`promote_scenario` enforces the regex `sc_YYYYMMDD_HHMMSS_<4hex>` and refuses paths that escape `/tmp/clustering_scenarios`. Manually crafted scenario IDs will fail.

---

## 8. Re-Running — When and Why

| Trigger | Re-run features? | Re-run clustering? | Re-run per-cluster tuning? |
|---|---|---|---|
| New monthly sales loaded (`make load-sales`) | **Yes** (`make features-compute`) | Optional — only if cluster shapes drift materially | No |
| Reload of historical sales / DQ correction | **Yes** | Yes (re-promote) | Yes |
| Change `sku_features_config.yaml` thresholds | **Yes** | Optional | No |
| Promote a new clustering experiment | No | (already done) | **Yes** — profiles flagged stale by promotion |
| Change `forecast_pipeline_config.yaml` `clustering` defaults | No | **Yes** | Yes if cluster names changed |
| Onboarding a brand-new SKU population | **Yes** | **Yes** | Yes |
| `clustering.enabled` toggled to `false` | No | No | No (backtests use global strategy) |

**Rule of thumb:** `features-compute` is cheap and safe — re-run it after every data refresh. Clustering is heavier and changes downstream model partitions, so re-promote only when a metric review (silhouette, cluster sizes, stability) justifies it. Per-cluster tuning is the heaviest of the three; let the staleness flag drive it.

### 8.1 Standard refresh sequence

```bash
# Data refresh
make load-sales            # or make pipeline-refresh
make features-compute      # always

# Optional cluster re-evaluation
make cluster-all           # creates + promotes new experiment
make tune-clusters         # regenerates cluster_tuning_profiles.yaml

# Downstream
make backtest-all
make champion-all
make forecast-generate
```

The full convenience chain is `make setup-features` (data + `features-compute` + `cluster-all`).
