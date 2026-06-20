# 23 -- LGBM Accuracy Tuning

> Spec for the systematic LGBM backtest accuracy improvement from 59% to 70%.
> This documents every change, rationale, and result so the tuning state can be
> reconstructed if needed.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy, Item Analysis |
| **Key Files** | `config/forecasting/forecast_pipeline_config.yaml`, `config/forecasting/cluster_tuning_profiles.yaml`, `common/ml/shap_selector.py`, `common/core/constants.py` (now exposes `FORECAST_QTY_COL`), `common/ml/model_registry.py`, `common/ml/backtest_framework.py`, `common/ml/feature_engineering.py`, `common/ml/tuning.py` (now routes fits through `model_registry.build_tree_model` + `fit_model` -- no direct `LGBMRegressor`/`CatBoostRegressor`/`XGBRegressor` instantiation), `common/ml/champion/` (split package -- formerly `common/ml/champion_strategies.py`), `scripts/run_backtest.py`, `scripts/tune_cluster_hyperparams.py` |

---

## 1. Baseline (Before Tuning)

- **Overall accuracy:** 59.23% (WAPE 40.77%)
- **Active DFUs:** 61.31% | **Sparse:** 2.17% | **Cold-start:** 1.32%

### 1.1 Original Config Snapshot

```yaml
algorithms:
  lgbm_cluster:
    params:
      objective: regression        # default MSE
      n_estimators: 1500
      learning_rate: 0.02
      num_leaves: 127
      min_child_samples: 40
      max_depth: -1                # unlimited
      colsample_bytree: 0.7
      feature_fraction_bynode: 0.5
      path_smooth: 4
      max_bin: 127
      variance_filter: true
      variance_threshold: 0.01
      correlation_filter: true
      correlation_threshold: 0.95
      shap_threshold: 0.95
backtest:
  early_stop_pct: 0.03
  shap_retrain_threshold: 0.10
  recursive_noise_pct: 0.08
```

### 1.2 Key Issues Identified

1. **NaN masking:** `mask_future_sales` set future qty to `0` instead of `NaN`, injecting artificial zeros into rolling means and Croston features for predict rows
2. **Aggressive dropna:** Training required all 12 lag features to be non-NaN, excluding every DFU with fewer than 12 months of history from training entirely
3. **Missing derived features:** `update_grid_incremental` only recomputed 3 of 7 derived features, leaving stale values in the recursive prediction loop
4. **Variance filter destroying signal:** Within-cluster data has naturally low variance for critical features (mean_demand, lags, rolling stats). The variance filter was removing all lag and rolling features
5. **Correlation filter too aggressive:** Threshold of 0.95 dropped mean_demand, qty_lag_1, and rolling_mean_* because they are correlated with each other -- but each captures a distinct demand signal
6. **No protected features:** SHAP and pre-SHAP stages could drop any feature, including core demand signals
7. **MSE objective unstable for zero-inflated data:** Default regression (MSE) squares errors, amplifying the impact of outlier non-zero values in predominantly-zero clusters
8. **Early stopping too tight:** 3% patience (45 rounds at 1500 estimators) caused premature stopping, especially on sparse clusters where WAPE is noisy
9. **No per-cluster feature selection:** A single pooled SHAP feature set was applied to all clusters, even though different clusters have different feature importance profiles
10. **Tweedie catastrophic for intermittent demand:** Tweedie's log link produces reasonable predictions at iteration 0, causing WAPE-based early stopping to fire at iter 1 (best_iter=1, negative accuracy)
11. **Recursive noise too high:** 8% noise compounded over 6-month recursive horizon, degrading accuracy

---

## 2. Changes Applied

### 2.1 Data Layer Fixes

#### 2.1.1 NaN Masking in mask_future_sales

- **File:** `common/ml/feature_engineering.py` (line 672)
- **What:** Changed `df.loc[future_mask, "qty"] = 0` to `df.loc[future_mask, "qty"] = np.nan`
- **Why:** Artificial zeros were contaminating rolling means for predict rows. Rolling statistics computed with `min_periods=1` skip NaN values, so rolling means now only reflect real historical data. Zero-masking made the model believe demand was 0 in future months, dragging down `rolling_mean_3m`, `rolling_mean_6m`, `rolling_mean_12m` and corrupting Croston intermittency signals.
- **Impact:** Large -- rolling_mean features now reflect real demand levels instead of being diluted by fake zeros. After NaN masking, feature columns are filled with 0 for model consumption, but the `qty` column itself remains NaN (excluded from features by `get_feature_columns`).

#### 2.1.2 Training Data Dropna Relaxed

- **File:** `common/ml/backtest_framework.py` (line 1247)
- **What:** Changed `dropna(subset=[qty_lag_1..qty_lag_12])` to `dropna(subset=["qty_lag_1"])`
- **Why:** DFUs with fewer than 12 months of history had NaN for lags 9-12 in ALL rows, losing the DFU entirely from training. LightGBM, CatBoost, and XGBoost all handle NaN natively -- they create a separate "missing" bin during histogram splits. Only the first lag needs to be non-NaN to provide meaningful training signal.
- **Impact:** Sparse and cold-start DFUs now participate in training, increasing the effective training set size and improving accuracy for short-history items.

#### 2.1.3 Missing Derived Features in update_grid_incremental

- **File:** `common/ml/feature_engineering.py` (line 730+)
- **What:** `update_grid_incremental` now calls `_recompute_derived_features()` which recomputes all 7 derived features: `mom_growth`, `demand_accel`, `volatility_ratio`, `lag_ratio_yoy`, `lag_ratio_mom`, `lag_ratio_3v12`, `n_zero_last_6m`
- **Why:** The incremental updater was only recomputing lag columns and rolling stats for affected months, but the 4 ratio features (`lag_ratio_yoy`, `lag_ratio_mom`, `lag_ratio_3v12`, `n_zero_last_6m`) were not being recomputed. During recursive inference, predictions are written back as qty for the next step, changing lags, but derived features that depend on those lags were stale.
- **Impact:** Moderate -- recursive mode now uses fresh derived features at each step, preventing accumulated staleness over the 6-month recursive horizon.

### 2.2 Feature Selection Fixes

#### 2.2.1 Variance Filter Disabled

- **Config:** `forecast_pipeline_config.yaml` -> `lgbm_cluster.params.variance_filter: false`
- **Why:** Within-cluster data has naturally low variance for key features. When clustering segments SKUs into homogeneous groups, features like `mean_demand`, lag features, and rolling statistics have reduced within-cluster variance by design. The filter (threshold 0.01, measuring `var / (max - min)^2`) was removing ALL lag and rolling features in some clusters, leaving the model with only calendar and attribute features.
- **Note:** Variance filter remains enabled for catboost_cluster and xgboost_cluster (threshold 0.01). Those models have not been tuned with the same intensity.

#### 2.2.2 Correlation Threshold Relaxed

- **Config:** `forecast_pipeline_config.yaml` -> `lgbm_cluster.params.correlation_threshold: 0.98`
- **Before:** 0.95
- **Why:** At 0.95, the correlation filter was dropping `mean_demand` (correlated with `rolling_mean_12m`), `qty_lag_1` (correlated with `qty_lag_2`), and `rolling_mean_3m` (correlated with `rolling_mean_6m`). While these features are mathematically correlated, each captures a distinct aspect of demand signal:
  - `qty_lag_1` = most recent actual (recency signal)
  - `rolling_mean_3m` = short-term trend
  - `rolling_mean_6m` = medium-term trend
  - `rolling_mean_12m` = long-term level
  - `mean_demand` = overall demand magnitude
- Raising to 0.98 preserves these features while still removing true duplicates.

#### 2.2.3 Protected Features

- **File:** `common/core/constants.py` -> `PROTECTED_FEATURES` set
- **Full set after tuning:**

```python
PROTECTED_FEATURES = {
    # Calendar/seasonal (no leakage)
    "month", "quarter",
    *FOURIER_FEATURES,  # fourier_sin_12/6/4/3, fourier_cos_12/6/4/3

    # Croston decomposition (intermittent demand signals)
    "croston_demand_size", "croston_probability",

    # Customer enrichment
    "true_demand_ratio", "n_active_cust", "hhi_demand",

    # Core demand signals
    "mean_demand", "qty_lag_1", "rolling_mean_3m", "rolling_mean_6m",

    # Recursive chain features (lags 2-3 carry recency signal)
    "qty_lag_2", "qty_lag_3",
    "rolling_mean_12m",
}
```

- **Why:** These features must survive both the correlation filter (Stage 2) and SHAP selection (Stage 3). Without protection, the correlation filter drops them in favour of lower-variance proxies, and SHAP may rank them low in clusters where all DFUs have similar demand levels (low within-cluster variance = low SHAP importance, even though the features are critical for prediction). For recursive prediction, lags 1-3 are essential because the recursive chain (month N predictions become month N+1 lag inputs) loses signal if any are dropped.
- **How protection works:** In `_remove_correlated_features`, when both features in a highly-correlated pair are in `PROTECTED_FEATURES`, neither is dropped. When only one is protected, the non-protected one is dropped. In `_select_features_from_shap`, protected features are always included in the selected set regardless of their SHAP rank.

#### 2.2.4 SHAP Threshold Lowered

- **Config:** `shap_threshold: 0.95` -> `0.90`
- **Why:** At 0.95, SHAP cumulative selection was keeping too many features (often all of them), making the retrain/no-retrain decision meaningless. At 0.90, SHAP selection is more aggressive, keeping only features that contribute to 90% of cumulative SHAP importance. This allows the retrain safety check to actually evaluate whether a reduced feature set helps.

#### 2.2.5 Per-Cluster SHAP Feature Selection

- **File:** `common/ml/shap_selector.py` -> `compute_timeframe_shap_per_cluster()`
- **What:** Each cluster gets independent SHAP selection instead of a single pooled computation. The function returns `dict[str, list[str]]` (mapping cluster labels to selected feature lists) instead of `list[str]`.
- **How it works:**
  1. Pre-SHAP stages (0-2) are shared across all clusters -- same duplicate, variance, and correlation exclusions apply globally
  2. For each cluster, SHAP values are computed using only that cluster's training data
  3. Per-cluster cumulative selection picks the features most important for that specific cluster
  4. Clusters with too few non-zero rows skip SHAP entirely and keep all features
- **Integration:** The backtest framework tracks `per_cluster_feature_cols: dict[str, list[str]]` and passes it through to `_predict_single_month` and `persist_cluster_models`. During recursive inference, the union of all per-cluster feature sets is used for NaN filling so every cluster's columns are present.

#### 2.2.6 Stratified SHAP Sampling for Sparse Clusters

- **File:** `common/ml/shap_selector.py` -> `_stratified_sample_for_shap()`
- **What:** For clusters with >50% zero-demand rows, uses 50/50 stratified sampling (non-zero + zero) instead of random sampling.
- **Constants:**
  - `SPARSE_ZERO_PCT_THRESHOLD = 0.5` -- clusters above this use stratified sampling
  - `SPARSE_MIN_NONZERO_ROWS = 100` -- clusters with fewer non-zero rows skip SHAP entirely (keep all features)
- **Why:** Random sampling of sparse clusters produces a SHAP sample dominated by zeros. SHAP then attributes high importance to features that separate zero from non-zero (e.g. `zero_demand_pct`, `brand`) rather than features that predict demand *levels*. Stratified sampling ensures the SHAP computation sees a balanced mix of zero and non-zero demand, producing feature rankings that better serve actual prediction accuracy.

#### 2.2.7 SHAP Retrain Safety Check

- **File:** `common/ml/backtest_framework.py` (line 1398+)
- **What:** After per-cluster retrain with SHAP-selected features, the framework computes validation WAPE for the retrained model and compares it to the original. If retrained WAPE is worse than the original, the retrained model is reverted to the original.
- **Config:** `backtest.shap_retrain_threshold: 0.50` (in `forecast_pipeline_config.yaml`)
- **Result in practice:** Retrain was consistently worse across all clusters -- the original model trained with the full feature set always outperformed the SHAP-reduced retrain. This is likely because:
  1. Tree models already handle irrelevant features well (they simply do not split on them)
  2. Removing features reduces the model's information set, and the marginal features still contribute via interaction effects
  3. The retrain uses the same hyperparameters as the original but with fewer features, which may not be optimal
- **Current state:** The retrain threshold of 0.50 effectively disables wasteful retrains. SHAP is retained as a diagnostic/reporting tool (the `shap_outputs/` directory still contains per-cluster feature importance reports), but the retrained models are almost always reverted.

### 2.3 Model Configuration Fixes

#### 2.3.1 Hyperparameter Changes

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `objective` | `regression` (MSE) | `regression_l1` (MAE) | MAE is robust for zero-inflated data. MSE squares errors, amplifying the impact of outlier non-zero values in sparse clusters. MAE treats all errors linearly. |
| `n_estimators` | 1500 | 2000 | More iterations give the model room to learn with the lower learning rate. Combined with early stopping, unused iterations cost nothing. |
| `learning_rate` | 0.02 | 0.015 | Slower learning rate with more estimators produces a smoother fit. Prevents overfitting on small/sparse clusters. |
| `num_leaves` | 127 | 63 | Halved leaf count reduces model complexity. 127 leaves was too expressive for per-cluster models (fewer DFUs per cluster than global), causing overfitting. |
| `max_depth` | -1 (unlimited) | 8 | Bounded depth prevents pathological deep trees in small clusters. Combined with 63 leaves, enforces a balanced tree shape. |
| `min_child_samples` | 40 | 20 | Lower threshold lets the model create more splits on smaller clusters. At 40, some sparse clusters had best_iter=1 because no useful split could be found with so few samples per leaf. |
| `colsample_bytree` | 0.7 | 0.8 | More features per tree gives each tree a richer view. With protected features and variance filter disabled, the feature set is more curated. |
| `feature_fraction_bynode` | 0.5 | 0.7 | Higher per-node feature fraction ensures that key features (lags, rolling means) are available at most split points. |
| `path_smooth` | 4 | 1.0 | Reduced smoothing. Path_smooth=4 was over-regularizing leaf predictions, pushing predictions toward parent averages too aggressively. |
| `max_bin` | 127 | 255 | More histogram bins give finer split resolution for continuous features like lag values and rolling means. |

#### 2.3.2 Tweedie Disabled for Intermittent Demand

- **File:** `scripts/run_backtest.py` -> `_apply_tweedie_objective()`
- **What:** Intermittent clusters (>= `intermittent_threshold` zeros, default 70%) are forced to use MAE (`regression_l1`) instead of Tweedie. Lumpy clusters (30-70% zeros) keep the default objective.
- **Why:** Tweedie's log link function produces reasonable predictions at iteration 0 for highly sparse data (where most targets are zero). Since early stopping monitors WAPE, the WAPE at iteration 0 is already "good" (predicting near-zero for near-zero actuals), causing early stopping to fire at iteration 1 before the model learns any signal. This results in best_iter=1 and catastrophic accuracy (near-zero or negative).
- **Implementation detail:**
  - `_classify_cluster_demand()` classifies each cluster as "continuous", "lumpy", or "intermittent" based on zero-demand percentage
  - For intermittent: forces `objective=regression_l1` (LGBM), `loss_function=MAE` (CatBoost), `objective=reg:absoluteerror` (XGBoost)
  - For lumpy: keeps default objective unchanged
  - For continuous: keeps default objective unchanged
- **Config thresholds:**
  - `backtest.intermittent_threshold: 0.7` (70% zeros -> intermittent)
  - `backtest.lumpy_threshold: 0.3` (30-70% zeros -> lumpy)

#### 2.3.3 Early Stopping Tuned

- **File:** `common/ml/model_registry.py`
- **Constants:**
  - `EARLY_STOP_PCT: 0.05` (was 0.03) -- 5% of max iterations = 100 rounds at 2000 estimators
  - `EARLY_STOP_FLOOR: 10` -- minimum patience rounds (unchanged)
  - `SPARSE_EARLY_STOP_PCT: 0.10` -- 10% of max iterations for sparse/intermittent clusters = 200 rounds at 2000 estimators
  - `SPARSE_EARLY_STOP_FLOOR: 50` -- minimum patience for sparse clusters
- **How it works:** `compute_early_stop_patience(max_iterations, pct, sparse=False)` returns `max(floor, int(max_iterations * pct))`. When `sparse=True`, uses `SPARSE_EARLY_STOP_PCT` with a floor of 50.
- **Why:** For sparse clusters, WAPE is noisy due to small denominators in the validation set. A validation set with 80% zeros can have wildly fluctuating WAPE from one iteration to the next, causing premature stopping. The 10% patience (200 rounds) gives the model enough iterations to find a real minimum rather than stopping on a noise spike.
- **WAPE eval callbacks:** All three models now use custom WAPE eval functions for early stopping alignment:
  - LGBM: `_wape_lgbm(y_true, y_pred)` with a scaled denominator floor of `max(len(y_true) * 0.01, 1.0)`
  - CatBoost: `WapeMetric` class with the same floor
  - XGBoost: `_wape_xgb(y_true, y_pred)` with the same floor

#### 2.3.4 Recursive Noise Reduced

- **Config:** `backtest.recursive_noise_pct: 0.08` -> `0.03`
- **What:** Gaussian noise injected into lag features during training (to simulate recursive prediction errors) was reduced from 8% to 3% of mean absolute value.
- **Why:** Higher noise compounded over the 6-month recursive horizon. At 8%, by month 6 the accumulated noise in recursive lag features was so large that the model's predictions diverged significantly from reasonable values. At 3%, the noise is sufficient to prevent distribution shift between training (real actuals as lags) and inference (model predictions as lags) without degrading accuracy.

### 2.4 Cluster Tuning Profile Fixes

#### 2.4.1 Profile Priority Order

The `_PROFILE_PRIORITY` list in `common/ml/backtest_framework.py` defines the match order (first match wins):

```python
_PROFILE_PRIORITY = [
    "sparse_intermittent",       # Most specific: high zeros + low mean demand
    "high_volume_stable",        # High mean demand + low zeros
    "medium_volume_periodic",    # Medium demand + low zeros
    "low_volume_volatile",       # Low demand + high CV + some zeros
    "volatile_large_cluster",    # Large cluster + high CV + mostly continuous
    "seasonal_dominant",         # High seasonal amplitude
    "default",                   # Fallback (empty criteria, always matches)
]
```

**Rationale:** `sparse_intermittent` is first because intermittent demand is the most distinct pattern requiring the most different hyperparameters. `high_volume_stable` is second because it has strict criteria (high mean + low zeros) that should not be overridden by more general profiles. The `default` profile is last and has empty match criteria, acting as a catch-all.

#### 2.4.2 Profile Match Criteria and Overrides

Full profile definitions from `config/forecasting/cluster_tuning_profiles.yaml`:

**sparse_intermittent** -- Truly intermittent SKUs

| Criterion | Value |
|-----------|-------|
| `zero_demand_pct_min` | 0.60 |
| `mean_demand_max` | 50 |
| `n_rows_max` | 300,000 |

| Override | Value | Purpose |
|----------|-------|---------|
| `num_leaves` | 15 | Very simple trees for sparse data |
| `min_child_samples` | 50 | Was 200 -- too high causes best_iter=1 |
| `reg_alpha` | 1.0 | Strong L1 for feature sparsity |
| `path_smooth` | 2 | Was 10 -- extreme smoothing prevents learning |
| `learning_rate` | 0.03 | Faster learning to extract signal from sparse data |

**low_volume_volatile** -- Small, volatile clusters with intermittency

| Criterion | Value |
|-----------|-------|
| `mean_demand_max` | 20 (was 50 -- tightened) |
| `cv_demand_min` | 1.5 (was 1.0 -- require genuinely high volatility) |
| `zero_demand_pct_min` | 0.15 (NEW -- exclude continuous periodic patterns) |
| `n_rows_max` | 300,000 |

| Override | Value | Purpose |
|----------|-------|---------|
| `num_leaves` | 31 | Moderate complexity |
| `min_child_samples` | 50 | Was 100 -- lower to avoid best_iter=1 |
| `reg_alpha` | 0.5 | Moderate L1 |
| `path_smooth` | 3 | Was 5 -- moderate smoothing |

**volatile_large_cluster** -- Large, volatile, continuous demand

| Criterion | Value |
|-----------|-------|
| `cv_demand_min` | 1.0 |
| `n_rows_min` | 300,000 |
| `zero_demand_pct_max` | 0.30 |

| Override | Value | Purpose |
|----------|-------|---------|
| `num_leaves` | 127 | Full leaf budget -- data volume supports complexity |
| `max_depth` | 12 | Deep trees for large data |
| `min_child_samples` | 25 | Allow finer splits |
| `reg_alpha` | 0.3 | Moderate L1 |
| `reg_lambda` | 1.0 | Keep L2 weak -- over-smoothing increases WAPE on spike months |

**medium_volume_periodic** -- Mostly continuous, medium demand

| Criterion | Value |
|-----------|-------|
| `mean_demand_min` | 5 |
| `mean_demand_max` | 100 |
| `zero_demand_pct_max` | 0.30 |

| Override | Value | Purpose |
|----------|-------|---------|
| `num_leaves` | 47 | Moderate complexity |
| `learning_rate` | 0.015 | Slower convergence for periodic patterns |
| `min_child_samples` | 30 | Allow sufficient splits |

**high_volume_stable** -- High volume, mostly continuous

| Criterion | Value |
|-----------|-------|
| `mean_demand_min` | 50 |
| `cv_demand_max` | 3.0 |
| `zero_demand_pct_max` | 0.15 |

| Override | Value | Purpose |
|----------|-------|---------|
| `num_leaves` | 127 | Full complexity for data-rich clusters |
| `max_depth` | 10 | Deep trees supported by volume |
| `min_child_samples` | 40 | Standard for large datasets |
| `learning_rate` | 0.01 | Very slow convergence for stability |
| `n_estimators` | 2000 | Matches main config |

**seasonal_dominant** -- Strong seasonal amplitude

| Criterion | Value |
|-----------|-------|
| `seasonal_amplitude_min` | 0.50 (was 0.30 -- tightened to genuinely seasonal only) |
| `zero_demand_pct_max` | 0.40 |

| Override | Value | Purpose |
|----------|-------|---------|
| `num_leaves` | 63 | Standard complexity |
| `min_child_samples` | 50 | Prevent overfitting to seasonal peaks |
| `colsample_bytree` | 0.9 | Keep more features to capture seasonal patterns |

**default** -- Fallback (empty criteria, always matches)

No overrides -- uses base config params as-is.

#### 2.4.3 Actual Cluster Profile Assignments

From the most recent backtest run, the 9 clusters matched profiles as follows:

| Cluster Label | mean_demand | cv_demand | zero_pct | Matched Profile |
|---------------|-------------|-----------|----------|-----------------|
| low_volume_moderate | 15.1 | 17.75 | 0.87 | **sparse_intermittent** |
| very_low_volume_very_steady | 25.6 | 27.41 | 0.98 | **sparse_intermittent** |
| very_low_volume_moderate | 11.9 | 30.55 | 0.88 | **sparse_intermittent** |
| medium_volume_moderate | 39.6 | 19.53 | 0.79 | **sparse_intermittent** |
| medium_volume_periodic_seasonal | 6.6 | 4.37 | 0.23 | **medium_volume_periodic** |
| medium_volume_periodic_c3 | 22.3 | 3.95 | 0.26 | **medium_volume_periodic** |
| medium_volume_periodic_c7 | 79.8 | 5.48 | 0.26 | **medium_volume_periodic** |
| high_volume_periodic | 58.2 | 1.16 | 0.04 | **high_volume_stable** |
| very_high_volume_periodic | 1153.2 | 2.27 | 0.04 | **high_volume_stable** |

Key observations:
- Four clusters (low_volume_moderate, very_low_volume_very_steady, very_low_volume_moderate, medium_volume_moderate) have >60% zero-demand and match `sparse_intermittent`. These clusters contain the majority of SKUs but contribute the least demand volume.
- Three clusters (medium_volume_periodic_*) have 23-26% zeros and match `medium_volume_periodic`. These are the workhorses -- moderate volume with periodic demand patterns.
- Two clusters (high_volume_periodic, very_high_volume_periodic) have <5% zeros and high mean demand, matching `high_volume_stable`. These dominate the accuracy metric due to high demand volume.

### 2.5 Logging Enhancements

#### 2.5.1 Per-Cluster SHAP Labels

- Log lines from `compute_timeframe_shap_per_cluster()` now show `[cluster=<name>]` instead of `[pooled]` for each SHAP computation. Example:
  ```
  [shap] cluster=high_volume_periodic: selected 42/65 features (threshold=0.90, 23 dropped)
  [shap] cluster=very_low_volume_moderate: too few non-zero rows for SHAP; keeping all features.
  ```

#### 2.5.2 Per-Timeframe Accuracy Summary

- `_log_timeframe_accuracy()` in `common/ml/backtest_framework.py` logs overall and per-cluster accuracy after each timeframe completes. Format:
  ```
  Timeframe A accuracy: 70.1% (wape=29.9%, 162,400 matched rows)
      high_volume_periodic: accuracy=85.2% wape=14.8% (12,300 rows)
      medium_volume_periodic_c7: accuracy=72.4% wape=27.6% (28,100 rows)
      very_low_volume_moderate: accuracy=1.2% wape=98.8% (45,600 rows)
      ...
  ```

#### 2.5.3 Training Accuracy in Cluster Logs

- Added `val_accuracy` (computed as `100.0 - val_wape`) to the per-cluster training log in `scripts/run_backtest.py`:
  ```
  Cluster 5/9 'high_volume_periodic': train=892,400, pred=12,300, best_iter=1247,
  val_accuracy=86.3%, val_wape=13.7%, profile=high_volume_stable, pattern=continuous (4.2s)
  ```

### 2.6 Intermittent Cluster Baseline Routing

#### 2.6.1 Rolling Mean Baseline for Intermittent Clusters

- **File:** `scripts/run_backtest.py`
- **What:** Added `_RollingMeanModel` class and intermittent routing logic in `_train_single_cluster`. When a cluster's `demand_pattern == "intermittent"` (i.e., `zero_pct >= intermittent_threshold`, default 0.70), the cluster is routed to `_predict_rolling_mean` instead of training an LGBM model.
- **How it works:**
  - Uses a 12-month rolling window (configurable via `backtest.baseline_intermittent_window` in `forecast_pipeline_config.yaml`)
  - `_RollingMeanModel` is a lightweight placeholder stored in the models dict so recursive prediction works seamlessly via `_predict_single_month`
  - SHAP gracefully skips these models -- the exception handler keeps all features when it encounters a non-tree model
- **Why:** Tree models fundamentally cannot forecast intermittent demand (Section 4, Known Limitation #1). With 87-98% of months having zero demand, LGBM predicts near-zero for everything, producing 0-2% accuracy. A simple rolling mean baseline captures the sparse demand signal better by averaging over recent non-zero months, improving accuracy from near-zero to 5-16% depending on the cluster.

#### 2.6.2 _predict_single_month Update

- **File:** `common/ml/backtest_framework.py`
- **What:** Added `_sku_cks` passthrough parameter for baseline models that need DFU-level mapping during recursive prediction. When a cluster uses a rolling mean baseline instead of a tree model, the recursive prediction loop passes the SKU composite keys through so the baseline predictor can map predictions back to individual DFUs.

#### 2.6.3 Results

**Per-cluster accuracy (Timeframe A, 1-step):**

| Cluster | Before (LGBM) | After (rolling mean) | Change |
|---|---|---|---|
| low_volume_moderate | 2.0% | 9.9% | +7.9pp |
| medium_volume_moderate | 1.0% | 15.0% | +14.0pp |
| very_low_volume_moderate | 0.4% | 5.8% | +5.4pp |
| very_low_volume_very_steady | 0.0% | 0.0% | -- (98% zeros) |

**Cohort accuracy:**

| Cohort | Before | After | Change |
|---|---|---|---|
| Active | 71.5% | 71.5% | -- |
| Sparse | 1.9% | 16.9% | +15.0pp |
| Cold-start | 0.3% | 4.1% | +3.8pp |

**Config:**

```yaml
backtest:
  baseline_intermittent: true          # default, enable routing
  baseline_intermittent_window: 12     # months for rolling mean
  intermittent_threshold: 0.7          # zero_pct cutoff
```

---

## 3. Results

### 3.1 Final Accuracy (Timeframe A)

| Cluster | Accuracy | WAPE | Rows | Profile | Model |
|---------|----------|------|------|---------|-------|
| very_high_volume_periodic | ~88% | ~12% | High | high_volume_stable | LGBM |
| high_volume_periodic | ~85% | ~15% | High | high_volume_stable | LGBM |
| medium_volume_periodic_c7 | ~72% | ~28% | Medium | medium_volume_periodic | LGBM |
| medium_volume_periodic_c3 | ~46% | ~54% | Medium | medium_volume_periodic | LGBM |
| medium_volume_periodic_seasonal | ~20% | ~80% | Medium | medium_volume_periodic | LGBM |
| medium_volume_moderate | ~15% | ~85% | High | sparse_intermittent | rolling_mean |
| low_volume_moderate | ~10% | ~90% | High | sparse_intermittent | rolling_mean |
| very_low_volume_moderate | ~6% | ~94% | High | sparse_intermittent | rolling_mean |
| very_low_volume_very_steady | ~0% | ~100% | High | sparse_intermittent | rolling_mean |

### 3.2 Comparison

| Metric | Before | After | Best Run | Delta |
|--------|--------|-------|----------|-------|
| Overall Accuracy | 59.23% | 70.2% | 70.7% | +11.0pp |
| Overall WAPE | 40.77% | 29.8% | 29.3% | -11.0pp |
| Active DFU Accuracy | 61.31% | 71.8% | — | +10.5pp |
| Sparse DFU Accuracy | 2.17% | 16.9% | — | +14.7pp |
| Cold-start DFU Accuracy | 1.32% | 4.1% | — | +2.8pp |
| Bias | — | -9.5% | — | — |

### 3.3 Cohort Breakdown

- **Active:** 61.3% -> 71.8% (+10.5pp) -- the bulk of the improvement, driven by data layer fixes and hyperparameter tuning
- **Sparse:** 2.2% -> 16.9% (+14.7pp) -- major improvement from intermittent cluster baseline routing (Section 2.6), which routes sparse clusters to rolling mean instead of LGBM
- **Cold-start:** 1.3% -> 4.1% (+2.8pp) -- improved by rolling mean baseline routing; cold-start DFUs with sparse demand now get meaningful predictions instead of near-zero LGBM outputs
- **Bias:** -9.5% -- inherent under-forecast from MAE (median regression) on right-skewed demand

---

## 4. Known Limitations

1. **Sparse clusters (80-98% zeros) at 0-2% accuracy** -- Tree models fundamentally cannot forecast intermittent demand. With 87-98% of months having zero demand, LGBM predicts near-zero for everything, which is correct most of the time but produces terrible WAPE when demand does occur. These clusters need Croston/rolling mean baselines instead of LGBM.

2. **SHAP retrain consistently makes accuracy worse** -- Suggests that the initial training with all features is already optimal for tree models. LightGBM handles irrelevant features by not splitting on them, so removing features reduces information without benefit. The retrain threshold is set to 0.50 to effectively disable it; SHAP is retained as a diagnostic tool.

3. **Medium-volume clusters (20-46%) have room for improvement** -- The `medium_volume_periodic` profile helps but these clusters are inherently harder: moderate demand levels with periodic patterns that are not strongly seasonal. Inline tuning (per-timeframe Bayesian optimization) can provide marginal improvements here.

4. **Only Timeframe A results shown in detail** -- Timeframe A has the shortest training window. Full 10-timeframe results (A-J) would show improving accuracy in later timeframes (more training data), but Timeframe A is the hardest test and the most representative of forward-looking accuracy.

5. **Sparse/cold-start accuracy nominally declined** -- These cohorts were not previously participating in training (due to the aggressive dropna), so the "before" number was computed on a different set of DFUs. The decline is an artifact of the changed training population, not a real regression.

---

## 5. Future Tuning Directions

1. **Route sparse clusters to rolling_mean or Croston baselines** -- Champion selection already routes per-DFU, but pre-routing entire sparse clusters to statistical baselines would avoid training LGBM on data it cannot meaningfully learn from. This saves compute time and avoids the model producing misleading near-zero predictions.

2. **Two-stage models for intermittent demand** -- Train a classifier (zero/non-zero) first, then a regressor (predict qty given demand occurs). This decomposes the problem into two tasks that tree models handle well individually.

3. **Disable SHAP retrain entirely** -- Convert to diagnostic-only mode (compute and save SHAP reports, but never retrain). The safety check already reverts 100% of retrains; formalizing this saves compute time.

4. **Direct multi-output for medium clusters** -- Instead of recursive 6-step prediction (where errors compound), train 6 separate models for horizons 1-6 months. Each model directly predicts its target horizon without lag dependency.

5. **Target encoding for categorical features** -- LightGBM's native categorical handling creates one bin per category. Target encoding (mean of target per category) provides a continuous signal that tree splits can use more flexibly.

6. **More aggressive inline tuning search space** -- The current 50-trial Bayesian optimization may not explore enough of the hyperparameter space for medium-volume clusters. Increasing to 100 trials or narrowing the search space to the most impactful parameters could help.

7. **Per-cluster hyperparameter Bayesian tuning** -- ~~Rather than static profiles, run Optuna per-cluster to find optimal hyperparameters. This is expensive but could unlock significant gains for the medium-volume clusters that currently underperform.~~ **DONE** — see Section 5b below.

---

## 5b. Per-Cluster Tuning Pipeline (Implemented)

### 5b.1 Overview

A dedicated per-cluster Bayesian hyperparameter tuning pipeline that runs Optuna independently for each `ml_cluster`, producing cluster-specific overrides written to `config/forecasting/cluster_tuning_profiles.yaml`.

- **Script:** `scripts/tune_cluster_hyperparams.py`
- **Makefile targets:** `make tune-lgbm-clusters`, `make tune-clusters` (all tree models)

### 5b.2 How It Works

1. Loads the feature matrix and splits by `ml_cluster`
2. For each cluster, runs an independent Optuna study with walk-forward CV
3. Best params per cluster are written to `cluster_tuning_profiles.yaml` with `cluster_name` in `match_criteria`
4. During backtest, `resolve_cluster_params()` in `backtest_framework.py` matches profiles:
   - **Phase 1:** Exact `cluster_name` match against the cluster's `ml_cluster` label (highest priority)
   - **Phase 2:** Statistical criteria fallback (mean_demand, cv_demand, zero_demand_pct, etc.)
   - First match wins per `_PROFILE_PRIORITY` order

### 5b.4 Tuning Fit Path Matches Production

`common/ml/tuning.py` no longer instantiates `LGBMRegressor` / `CatBoostRegressor` / `XGBRegressor` directly. Every Optuna trial now constructs its estimator through `model_registry.build_tree_model(algorithm_id, params)` and trains it via `model_registry.fit_model(...)` -- the exact same path used by `scripts/run_backtest.py` and production training. This guarantees that tuned hyperparameters reproduce the same fit semantics (early-stop patience, custom WAPE eval callbacks, sparse-aware patience floors, demand-pattern routing) when promoted to production. Adding a new tree model to the registry is the only step needed to make it tunable -- no `if/elif` branches in `tuning.py`.

### 5b.3 Recursive Lag Smoothing

- **Config:** `recursive_lag_smooth: 0.15` (in `forecast_pipeline_config.yaml` under `backtest`)
- **What:** From recursive step 3 onward, lag features are exponentially smoothed: `lag_t = alpha * prediction + (1-alpha) * lag_{t-1}` where alpha = `recursive_lag_smooth`
- **Why:** Recursive errors compound over 6 months. Smoothing damps oscillations in later steps without losing the recency signal in steps 1-2

## 5c. Champion Selection Fixes (Implemented)

### 5c.1 Decimal to Float Cast

- **File:** `common/ml/champion/helpers.py` (formerly part of the now-split `common/ml/champion_strategies.py`; canonical import is `from common.ml.champion import ...`)
- **What:** Explicit `float()` cast on Decimal values from DB queries to prevent `TypeError` in numpy/pandas operations

### 5c.2 Ensemble Detection Fix

- **What:** `is_ensemble` flag now checks whether the winner `model_id` is synthetic (not in the competing model list) rather than checking for a specific prefix. This correctly identifies ensemble/blended champion rows regardless of naming convention.

### 5c.3 Per-Cluster Strategy Enhancements

- Per-cluster champion strategy now loads `dfu_features` from the feature matrix, enabling feature-aware model selection per cluster

### 5c.4 Cached Winners CSV

- Champion experiment results are cached to `data/champion/experiment_{id}_winners.csv` for fast "Load Results" in the UI
- CSV dtype fix: `item_id` is read as `str` to prevent numeric truncation

### 5c.5 Delete Promoted Experiments

- Deleting a promoted champion experiment now cleans up: forecast rows with the experiment's `model_id`, `promotion_log` FK references, then the experiment row itself

### 5c.6 Auto-Tune Ranking & Per-Model Baseline (2026-06-20)

- **File:** `scripts/ml/auto_tune.py`
- **Multi-seed ranking:** strategies are ranked and promotion is gated on `multi_seed_summary.mean_accuracy_pct` when present (falling back to the single-seed value). It previously used `accuracy_at_execution_lag.accuracy_pct`, which `run_backtest` overwrites once **per seed** — so with `n_seeds > 1` the leaderboard reflected only the **last** seed, letting a strategy win on last-seed luck and defeating the point of multi-seed variance estimation.
- **Per-model baseline:** `get_baseline_accuracy()` now filters `lgbm_tuning_run` by `AND model_id = %s`. That table is shared across lgbm/catboost/xgboost, so tuning CatBoost/XGBoost had been comparing the candidate against an **lgbm** baseline (different absolute accuracy levels) — which could promote a worse model or block a better one.

### 5c.7 Champion COPY Integrity (2026-06-20)

- **Files:** `api/routers/forecasting/competition.py`, `scripts/ml/run_champion_selection.py`
- The champion-winner COPY paths built tab-delimited `io.StringIO` buffers with raw f-strings and **no escaping**. Text-format COPY treats tab as the column delimiter, newline as the row terminator, and backslash as the escape char, so any `item_id`/`customer_group`/`loc` (free-text dimension values from external CSVs) containing one of those desynced the stream — shifting columns, routing the **wrong** model under `model_id='champion'` for a DFU, or silently dropping rows. Switched to `copy.write_row((...))` with explicit column lists (psycopg3 escapes + type-adapts each value).
- **Deferred:** `run_champion_selection.py`'s `_ensemble_winners` COPY (the `source_mix` JSONB column) has the same risk but needs a psycopg `Jsonb` wrapper + a live-DB test, so it is left for a DB-backed PR. **New rule:** never build COPY buffers by hand — always `copy.write_row(...)`.

### 5c.8 Cluster-Tuning Profile Guard (2026-06-20)

- **File:** `scripts/ml/tune_cluster_hyperparams.py` — a tuning run whose `best_wape` is non-finite (all folds errored) no longer persists its hyperparameters to `cluster_tuning_profiles.yaml`. It previously wrote `inf`-WAPE garbage profiles that then drove production training.

---

## 6. File Change Summary

| File | What Changed |
|------|-------------|
| `config/forecasting/forecast_pipeline_config.yaml` | LGBM params: objective=regression_l1, n_estimators=2000, learning_rate=0.015, num_leaves=63, max_depth=8, min_child_samples=20, colsample_bytree=0.8, feature_fraction_bynode=0.7, path_smooth=1.0, max_bin=255, variance_filter=false, correlation_threshold=0.98, shap_threshold=0.90. Backtest: early_stop_pct=0.05, shap_retrain_threshold=0.50, recursive_noise_pct=0.03, baseline_intermittent=true, baseline_intermittent_window=12, intermittent_threshold=0.7. |
| `config/forecasting/cluster_tuning_profiles.yaml` | All 7 profiles (sparse_intermittent, low_volume_volatile, volatile_large_cluster, medium_volume_periodic, high_volume_stable, seasonal_dominant, default) with tightened match criteria and tuned overrides. |
| `common/core/constants.py` | `PROTECTED_FEATURES` set expanded with mean_demand, qty_lag_1-3, rolling_mean_3m/6m/12m, croston features, customer enrichment features. |
| `common/ml/feature_engineering.py` | `mask_future_sales`: qty=NaN instead of qty=0. `update_grid_incremental`: calls `_recompute_derived_features()` to recompute all 7 derived features. `_recompute_derived_features`: added lag_ratio_yoy, lag_ratio_mom, lag_ratio_3v12, n_zero_last_6m. |
| `common/ml/backtest_framework.py` | dropna relaxed to subset=["qty_lag_1"]. `_PROFILE_PRIORITY` ordering. `_log_timeframe_accuracy()` for per-cluster accuracy reporting. `_compute_cluster_wape()` for SHAP retrain safety. Per-cluster feature selection plumbing (`per_cluster_feature_cols` dict). `_predict_single_month` accepts per_cluster_feature_cols and `_sku_cks` passthrough for baseline models needing DFU-level mapping during recursive prediction. |
| `common/ml/shap_selector.py` | `compute_timeframe_shap_per_cluster()` for independent per-cluster SHAP. `_stratified_sample_for_shap()` for sparse cluster sampling. Constants: `SPARSE_ZERO_PCT_THRESHOLD=0.5`, `SPARSE_MIN_NONZERO_ROWS=100`. |
| `common/ml/model_registry.py` | `EARLY_STOP_PCT=0.05`, `SPARSE_EARLY_STOP_PCT=0.10`, `SPARSE_EARLY_STOP_FLOOR=50`. `_wape_lgbm()`, `WapeMetric`, `_wape_xgb()` custom eval functions with scaled denominator floor. `fit_model()` accepts `demand_pattern` param for sparse-aware early stopping. |
| `scripts/run_backtest.py` | `_classify_cluster_demand()` for demand pattern classification. `_apply_tweedie_objective()` to override objective for intermittent clusters. `persist_cluster_models()` handles per-cluster feature_cols dict. Training log includes val_accuracy. `_RollingMeanModel` class and intermittent routing in `_train_single_cluster` to route sparse clusters to `_predict_rolling_mean`. `feature_selector_fn` routes to per-cluster SHAP. `train_and_predict_per_cluster` accepts `per_cluster_feature_cols`. |
| `scripts/tune_cluster_hyperparams.py` | **NEW** — Per-cluster Bayesian hyperparameter tuning pipeline. Runs Optuna independently per `ml_cluster`, writes cluster-specific overrides to `cluster_tuning_profiles.yaml` with `cluster_name` in `match_criteria`. |
| `common/ml/champion/` (split from the legacy `common/ml/champion_strategies.py` into 9 sub-modules: `registry.py`, `basic.py`, `blend.py`, `meta.py`, `bandit.py`, `segment.py`, `regime.py`, `routing.py`, `helpers.py` -- see [Champion Selection](./07-champion-selection.md#module-layout)) | Decimal -> float cast for DB values (in `helpers.py`). `is_ensemble` detection fix (checks synthetic model_id). Per-cluster strategy (in `segment.py`) loads `dfu_features`. |

---

## 7. Iterative Tuning Log

### Objective Function Experiments

| Objective | Overall Acc | Active | Bias | Notes |
|---|---|---|---|---|
| regression_l1 (MAE) | **70.2%** | 71.8% | -9.5% | Best — robust to outliers, slight under-forecast |
| huber (delta=5.0) | 9.7% | — | — | Catastrophic — delta too small for demand scale |
| regression (MSE) | 63.3% | 64.7% | +11.6% | Over-forecasts — too sensitive to outliers |
| quantile (alpha=0.55) | 47.0% | 47.9% | +41.7% | Massive over-forecast — too aggressive |

**Conclusion**: MAE is the best objective for this data. The -9.5% under-forecasting bias is a fundamental property of median regression on right-skewed demand. MSE over-corrects in the opposite direction.

### Recursive vs Direct Mode

| Mode | Overall | high_volume_periodic | medium_volume_periodic_c3 | Notes |
|---|---|---|---|---|
| Recursive | **70.2%** | 68.3% | 28.9% | Better — chained predictions provide lag signal |
| Direct | 69.5% | 64.0% | 22.6% | Worse — stale lag features for months 3+ |

**Conclusion**: Recursive is better despite error compounding. In direct mode, lag features for predict months 3+ reference pre-prediction-window data, which is stale.

### SHAP Selection Impact

| SHAP | Overall | Time | Notes |
|---|---|---|---|
| Enabled | **70.2%** | ~170s | Better accuracy, retrain always reverts but diagnostics help |
| Disabled | 69.9% | ~47s | Slightly worse, 3.6x faster |

**Conclusion**: Keep SHAP enabled — the 0.3pp accuracy gain is worth the extra time.

### Correlation Filter Impact

| Filter | Overall | Bias | Notes |
|---|---|---|---|
| Disabled | **70.2%** | -9.5% | Best — LGBM handles correlated features well |
| Enabled (0.98) | 69.9% | -10.2% | Slightly worse — removing features hurts |

**Conclusion**: Disabled correlation filter. LGBM's tree-based splitting handles correlated features natively.

### Estimators / Learning Rate Experiments

| Config | Overall | very_high_volume | medium_c7 | Notes |
|---|---|---|---|---|
| Base n=2000, lr=0.015 | 69.9% | 73.5% | 19.3% | Starting point |
| Profile n=3000, lr=0.01 | **70.7%** | **74.4%** | **24.6%** | Best — more iterations at lower LR |
| Profile n=4000, lr=0.01 | 69.5% | 73.2% | 24.4% | Overfitting |
| Base n=2500, lr=0.012 | 69.7% | 73.3% | 23.6% | Marginally worse |

**Conclusion**: 3000 estimators with lr=0.01 is optimal for medium/high-volume clusters. 4000 overfits.

### Early Stop Patience

| Pct | Overall | medium_c3 | Notes |
|---|---|---|---|
| 5% (150 rounds) | **70.7%** | 30.8% | Best overall |
| 8% (240 rounds) | 70.3% | 31.0% | c3 slightly better, others worse |

### Final Best Configuration
```yaml
# Base params (forecast_pipeline_config.yaml)
objective: regression_l1
n_estimators: 2000
learning_rate: 0.015
num_leaves: 63
min_child_samples: 20
max_depth: 8
correlation_filter: false
variance_filter: false
shap_select: true
shap_threshold: 0.9
recursive: true
recursive_noise_pct: 0.03
early_stop_pct: 0.05
shap_retrain_threshold: 0.50

# high_volume_stable profile
num_leaves: 127, max_depth: 10, min_child_samples: 40
learning_rate: 0.01, n_estimators: 3000, subsample: 0.8

# medium_volume_periodic profile
num_leaves: 95, max_depth: 10, min_child_samples: 15
learning_rate: 0.01, n_estimators: 3000
reg_alpha: 0.2, reg_lambda: 0.3, path_smooth: 0.3
colsample_bytree: 0.85, subsample: 0.8
```
