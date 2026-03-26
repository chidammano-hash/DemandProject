# Backtest Framework Defect Report

> **Reviewed**: 2026-03-25
> **Scope**: `common/ml/backtest_framework.py`, `common/ml/feature_engineering.py`, `common/ml/model_registry.py`, `common/ml/shap_selector.py`, `common/ml/tuning.py`, `common/core/constants.py`, `common/services/metrics.py`, `scripts/run_backtest.py`, `config/algorithm_config.yaml`
> **Method**: Independent review by 10 simulated time series domain experts

---

## Severity Definitions

| Level | Meaning |
|-------|---------|
| **CRITICAL** | Directly invalidates accuracy numbers or causes silent model degradation |
| **HIGH** | Materially impacts forecast quality or prevents valid model comparison |
| **MODERATE** | Degrades quality in specific scenarios or violates best practices |
| **LOW** | Minor inefficiency or cosmetic issue |

---

## Summary Matrix

| # | Defect | Severity | File(s) | Fix Complexity |
|---|--------|----------|---------|----------------|
| D-001 | TS profile features leak future data | CRITICAL | `feature_engineering.py` | Medium |
| D-002 | XGBoost early stopping not wired | CRITICAL | `model_registry.py` | Trivial |
| D-003 | Zero-prediction fallback for small clusters | HIGH | `run_backtest.py` | Medium |
| D-004 | No uncertainty quantification | HIGH | Framework-wide | High |
| D-005 | Intermittent demand treated as continuous regression | HIGH | `run_backtest.py`, `model_registry.py` | High |
| D-006 | Recursive mode train/serve distribution shift | HIGH | `backtest_framework.py` | High |
| D-007 | Single seed, no variance estimation | HIGH | `run_backtest.py` | Low |
| D-008 | Deduplication bias in accuracy reporting | HIGH | `backtest_framework.py` | Medium |
| D-009 | Training objective / evaluation metric mismatch | MODERATE | `model_registry.py`, `run_backtest.py` | Medium |
| D-010 | Feature explosion vs. small cluster sample size | MODERATE | `constants.py`, `run_backtest.py` | Medium |
| D-011 | Fourier / month_sin feature redundancy | MODERATE | `constants.py`, `feature_engineering.py` | Trivial |
| D-012 | CatBoost reg_lambda silently ignored | MODERATE | `run_backtest.py`, `model_registry.py` | Trivial |
| D-013 | Print statements bypass structured logging | MODERATE | `backtest_framework.py`, `run_backtest.py` | Medium |
| D-014 | Future masking uses zero instead of NaN | MODERATE | `feature_engineering.py` | Medium |
| D-015 | No embargo gap in main backtest loop | MODERATE | `backtest_framework.py` | Low |

---

## CRITICAL Defects

### D-001: TS Profile Features Leak Future Data

**Expert**: Data Leakage Specialist
**Location**: `common/ml/feature_engineering.py:571-578` (`_compute_ts_profile_features`), `feature_engineering.py:622-643` (`mask_future_sales`)

**Description**

`_compute_ts_profile_features()` computes 8 per-DFU static features from the **entire qty column**, including months that fall in the prediction period. These features are:

- `mean_demand` -- average demand across all months including future
- `cv_demand` -- coefficient of variation across all months
- `zero_demand_pct` -- zero-demand fraction across all months
- `trend_slope_norm` -- linear trend slope fit to all months
- `recency_ratio` -- last-6-month mean / prior mean (the "last 6" may include prediction months)
- `seasonal_amplitude` -- seasonal swing computed from all months
- `adi` -- average demand interval across all months
- `yoy_correlation` -- year-over-year correlation using all months

These 8 features are computed **once** during `build_feature_matrix()` at line 573 and joined to the grid at line 574. When `mask_future_sales()` is called per timeframe, it zeroes future qty and recomputes lags, rolling stats, derived features, Croston, and cross-DFU features -- but **never recomputes TS profile features**.

**Impact**

For timeframe A (the shortest training window with ~10 months of prediction horizon), these features carry substantial information about the future:

- `mean_demand` incorporates 10 months of future actual demand
- `trend_slope_norm` captures the demand trajectory into the prediction period
- `recency_ratio` compares a window that includes prediction-period actuals against prior history
- `yoy_correlation` uses future months in its correlation window

The model receives 8 features that encode future demand patterns. This inflates reported accuracy relative to what the model would achieve in genuine production conditions.

**Evidence**

```python
# feature_engineering.py line 571-574 (comment confirms the issue)
# Per-DFU time-series profile features (computed from full history)
t1 = time.time()
ts_profiles = _compute_ts_profile_features(grid)
grid = grid.merge(ts_profiles, on="sku_ck", how="left")
```

```python
# feature_engineering.py lines 622-643 -- mask_future_sales does NOT recompute TS profiles
def mask_future_sales(grid, cutoff):
    df = grid.copy()
    df.loc[future_mask, "qty"] = 0
    _compute_lags_and_rolling(df)       # recomputed
    _recompute_derived_features(df)     # recomputed (includes Croston, cross-DFU)
    return df
    # TS profiles are NOT recomputed -- they retain full-history values
```

**Proposed Fix**

Option A (recommended): Recompute TS profiles inside `mask_future_sales()` using only pre-cutoff data:

```python
def mask_future_sales(grid, cutoff):
    df = grid.copy()
    df.loc[future_mask, "qty"] = 0
    _compute_lags_and_rolling(df)
    _recompute_derived_features(df)

    # Recompute TS profiles using only training-period data
    train_only = df[df["startdate"] <= cutoff]
    ts_profiles = _compute_ts_profile_features(train_only)
    for col in TS_PROFILE_FEATURES:
        df.drop(columns=col, inplace=True, errors="ignore")
    df = df.merge(ts_profiles, on="sku_ck", how="left")
    for col in TS_PROFILE_FEATURES:
        df[col] = df[col].fillna(0).astype(np.float32)
    return df
```

Option B: Convert TS profiles into rolling features with explicit lookback windows so they are naturally recomputed by `_compute_lags_and_rolling()`.

---

### D-002: XGBoost Early Stopping Not Wired

**Expert**: Cross-Validation & Temporal Splitting
**Location**: `common/ml/model_registry.py:185-190` (`fit_model`)

**Description**

The `fit_model()` function computes early stopping patience at line 162 but never passes it to XGBoost. LGBM receives `callbacks=[lgb.early_stopping(patience)]` and CatBoost receives `early_stopping_rounds=patience`, but XGBoost's fit call only receives `eval_set` and `verbose`:

```python
elif model_name == "xgboost":
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
```

XGBoost always trains to its full `n_estimators=500` regardless of validation performance.

**Impact**

- XGBoost systematically overfits to training data since early stopping never fires
- Accuracy comparisons between LGBM/CatBoost (early-stopped) and XGBoost (full train) are invalid
- XGBoost runs slower than necessary (trains full 500 iterations even when optimal is ~200)

**Proposed Fix**

```python
elif model_name == "xgboost":
    model.set_params(early_stopping_rounds=patience)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
```

Or pass via the fit call directly:

```python
elif model_name == "xgboost":
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=patience,
        verbose=False,
    )
```

---

## HIGH Defects

### D-003: Zero-Prediction Fallback for Small Clusters

**Expert**: Clustering & Hierarchical Forecasting
**Location**: `scripts/run_backtest.py:233-239` (`_train_single_cluster`)

**Description**

Clusters with fewer than `MIN_CLUSTER_ROWS` (50) training samples receive zeroed predictions:

```python
if len(train_c) < MIN_CLUSTER_ROWS or len(pred_c) == 0:
    if len(pred_c) > 0:
        result = pred_c[...].copy()
        result["basefcst_pref"] = 0.0
        return cluster_label, result, None, None
```

Zeroing predictions is the worst possible fallback. It guarantees:

- 100% underforecast for every DFU in the cluster
- Phantom stockouts in inventory planning (safety stock sees zero forecast)
- WAPE of 100% for these items, dragging down aggregate accuracy

**Impact**

Any cluster with 30-49 training rows (non-trivial for clusters with new DFUs or short history) produces zero forecasts. If 5% of DFUs land in small clusters, the framework silently outputs zero for 5% of the demand base.

**Proposed Fix**

Fallback hierarchy:
1. Use the global model's predictions for that cluster's DFUs
2. Use the cluster's historical mean (seasonal naive)
3. Pool with the nearest neighboring cluster
4. As last resort, use the overall median forecast ratio

---

### D-004: No Uncertainty Quantification

**Expert**: Forecast Evaluation Metrics
**Location**: Framework-wide

**Description**

The framework produces only point forecasts (single `basefcst_pref` per DFU-month). There are no:

- Prediction intervals (e.g., 80th/95th percentile bounds)
- Bootstrapped confidence bands on accuracy metrics
- Statistical tests for model comparison (paired t-test, Diebold-Mariano)

**Impact**

- Planners cannot assess forecast risk for safety stock calculations
- Model comparisons lack statistical significance testing -- a 0.5% accuracy difference may be noise
- Inventory planning modules that consume forecasts have no way to parameterize uncertainty

**Proposed Fix**

Short-term: Compute per-timeframe accuracy and report mean +/- standard error across timeframes.
Medium-term: Add quantile regression (LGBM/CatBoost support quantile objectives natively) to produce P10/P50/P90 forecasts.
Long-term: Implement conformal prediction for distribution-free prediction intervals.

---

### D-005: Intermittent Demand Treated as Continuous Regression

**Expert**: Intermittent Demand Specialist
**Location**: `scripts/run_backtest.py:300`, `config/algorithm_config.yaml`

**Description**

All DFUs -- regardless of demand pattern -- are forecast using continuous regression (`LGBMRegressor`, `CatBoostRegressor`, `XGBRegressor`) with MAE/RMSE objectives. For intermittent demand items (high `zero_demand_pct`, ADI > 1.32):

1. The model outputs negative values that are clipped to 0: `np.maximum(preds, 0)`
2. MAE/RMSE treat prediction=5 for actual=0 identically to prediction=5 for actual=10 -- but the supply chain costs are very different (excess stock vs. stockout)
3. Croston features are computed (`croston_demand_size`, `croston_demand_interval`, `croston_probability`) but only serve as input features -- they don't change the loss function or model specification

**Impact**

For DFUs with 60-80% zero-demand months, the model learns to predict near-zero values (minimizing average error), chronically underpredicting the non-zero demand events that actually matter for stocking.

**Proposed Fix**

- Use Tweedie regression (`objective="tweedie"` in LGBM, `loss_function="Tweedie"` in CatBoost) for intermittent items. Tweedie naturally handles zero-inflated positive data.
- Alternatively, implement a two-stage hurdle model: (1) classify demand/no-demand, (2) regress demand quantity conditional on demand occurring.
- Route DFUs to model types based on their TS profile classification: continuous items -> current regression, intermittent items -> Tweedie/Croston.

---

### D-006: Recursive Mode Train/Serve Distribution Shift

**Expert**: Multi-Step Forecasting Specialist
**Location**: `common/ml/backtest_framework.py:911-930`

**Description**

In recursive mode, models are trained once on features derived from **real historical data**, then used iteratively to predict months 2+ using features derived from **model predictions**:

```python
# Month 1: features from real data
preds_first, models = train_fn_per_cluster(train_data, first_predict, ...)
# Months 2+: features from model's own predictions
for month in sorted_months[1:]:
    month_data = current_grid[current_grid["startdate"] == month].copy()
    preds_month = _predict_single_month(models, month_data, ...)
    update_grid_incremental(current_grid, month, preds_month, all_months)
```

The model was trained on features computed from actual sales. At serve time for month T+5, `qty_lag_1` is the model's prediction for T+4 (which itself used predicted T+3, etc.). The feature distribution at serve time diverges from training distribution as errors compound.

**Impact**

Accuracy degrades non-linearly with forecast horizon in recursive mode. The framework reports a single blended accuracy across all recursive steps without isolating per-step degradation. The reported accuracy is dominated by month 1 (direct prediction with real features) and masks the poor quality of months 5+.

**Proposed Fix**

- Add teacher forcing with noise injection during training: randomly replace some lag values with noisy predictions to simulate recursive conditions
- Report accuracy broken down by recursive step (month 1, 2, 3, ...) in the metadata
- Consider direct multi-output as the default and reserve recursive mode only for DFUs where it demonstrably outperforms

---

### D-007: Single Seed, No Variance Estimation

**Expert**: Statistical Rigor & Experimental Design
**Location**: `scripts/run_backtest.py:82,130,178` (all models hardcode `random_state=42`)

**Description**

Every model is trained with a single fixed random seed (`random_state=42` for LGBM/XGBoost, `random_seed=42` for CatBoost). The framework runs once and reports a single accuracy number per model.

Without multiple seed evaluation, there is no way to estimate:
- **Model variance**: How much does accuracy change with different initialization?
- **Significance of differences**: Is LGBM at 71.8% genuinely better than CatBoost at 71.2%?
- **Representativeness**: Is 71.8% a typical run, or an outlier seed?

**Impact**

Model selection decisions are based on potentially noisy single-run metrics. A "winning" model may only be winning by chance for seed 42.

**Proposed Fix**

Add a `--n-seeds` flag to the backtest runner:

```python
for seed in range(n_seeds):
    params["random_state"] = seed
    # run backtest ...
    seed_accuracies.append(accuracy)
print(f"Accuracy: {np.mean(seed_accuracies):.2f}% +/- {np.std(seed_accuracies):.2f}%")
```

Report mean, std, and confidence intervals. Use paired Wilcoxon signed-rank test for model comparison.

---

### D-008: Deduplication Bias in Accuracy Reporting

**Expert**: Statistical Rigor & Experimental Design
**Location**: `common/ml/backtest_framework.py:451-452`

**Description**

When multiple timeframes predict the same (DFU, month), the deduplication keeps the **latest** timeframe:

```python
expanded = expanded.sort_values("timeframe_idx")
expanded = expanded.drop_duplicates(subset=["forecast_ck", "model_id"], keep="last")
```

The latest timeframe has (a) the most training data and (b) the shortest prediction horizon. Both factors improve accuracy. The output table's accuracy metric is therefore biased toward best-case conditions.

**Impact**

Reported accuracy (e.g., 71.8%) overstates what the model achieves at longer horizons. The archive table preserves all lags, but the primary accuracy metric in `backtest_metadata.json` is computed from the biased deduplicated output.

**Proposed Fix**

- Report separate accuracy metrics: per-timeframe, per-lag, and overall
- In metadata, include a `accuracy_by_timeframe` dict alongside the aggregate
- Consider using only the first valid prediction per (DFU, month) for a more conservative metric

---

## MODERATE Defects

### D-009: Training Objective / Evaluation Metric Mismatch

**Expert**: Forecast Evaluation Metrics
**Location**: `common/ml/model_registry.py:168`, `scripts/run_backtest.py:131`, `common/services/metrics.py`

**Description**

| Model | Training Objective | Early Stop Metric | Evaluation Metric |
|-------|-------------------|-------------------|-------------------|
| LGBM | Default (L2) | MAE | WAPE |
| CatBoost | RMSE | RMSE | WAPE |
| XGBoost | Default (RMSE) | N/A (not wired) | WAPE |

All models are evaluated on WAPE, but none are trained or early-stopped on WAPE. WAPE = `sum|F-A| / |sum(A)|` is an **aggregate** metric that weights errors by total actual demand. MAE weights all errors equally. RMSE penalizes outliers quadratically.

A model early-stopped on MAE may stop at a different iteration than one that would minimize WAPE. For supply chain where aggregate demand accuracy matters most, the mismatch means models are not optimized for the metric that matters.

**Proposed Fix**

Implement a custom WAPE callback for LGBM early stopping:

```python
def wape_metric(y_pred, dataset):
    y_true = dataset.get_label()
    denom = max(abs(y_true.sum()), 1.0)
    wape = np.sum(np.abs(y_pred - y_true)) / denom
    return "wape", wape, False  # lower is better
```

Similarly for CatBoost and XGBoost custom eval functions.

---

### D-010: Feature Explosion vs. Small Cluster Sample Size

**Expert**: Feature Engineering Authority
**Location**: `common/core/constants.py` (`MIN_CLUSTER_ROWS=50`), `scripts/run_backtest.py:233`

**Description**

The feature set contains ~66 features: 12 lags + 6 rolling + 7 calendar + 8 Fourier + 7 derived + 3 Croston + 4 cross-DFU + 2 external + 8 TS profiles + 4 categorical + 2 SKU numeric + 3 item numeric.

For a cluster at the minimum threshold (50 rows), the 80/20 train/val split yields 40 effective training samples. Training a gradient boosted tree with 66 features on 40 samples is severely underdetermined. `colsample_bytree=0.8` (53 features per tree) provides only mild regularization.

**Impact**

Small clusters overfit to training data. Per-cluster validation WAPE may look good (the model memorized 40 rows) but generalization to the prediction period is poor.

**Proposed Fix**

Either:
1. Scale `MIN_CLUSTER_ROWS` to be proportional to feature count (e.g., `max(50, 3 * len(feature_cols))`)
2. Run SHAP feature selection *before* per-cluster training to reduce the feature count first
3. Use stronger regularization for small clusters (the cluster profile system already exists; add a `small_data` profile with aggressive regularization)

---

### D-011: Fourier / month_sin Feature Redundancy

**Expert**: Feature Engineering Authority
**Location**: `common/core/constants.py`, `common/ml/feature_engineering.py:243-256,548-551`

**Description**

`fourier_sin_12` and `fourier_cos_12` are mathematically identical to `month_sin` and `month_cos`:

```python
# feature_engineering.py line 550-551
grid["month_sin"] = np.sin(2 * np.pi * grid["month"] / 12)
grid["month_cos"] = np.cos(2 * np.pi * grid["month"] / 12)

# feature_engineering.py line 252-256 (_compute_fourier_features)
for period in [12, 6, 4, 3]:
    angle = 2.0 * np.pi * month_vals / period
    df[f"fourier_sin_{period}"] = np.sin(angle)  # fourier_sin_12 == month_sin
    df[f"fourier_cos_{period}"] = np.cos(angle)  # fourier_cos_12 == month_cos
```

Both are in `PROTECTED_FEATURES`, guaranteeing the redundancy persists through SHAP selection.

**Proposed Fix**

Remove `month_sin` and `month_cos` from the feature set (keep the more general Fourier features). Or remove period=12 from the Fourier computation since it duplicates `month_sin/month_cos`.

---

### D-012: CatBoost reg_lambda Silently Ignored

**Expert**: Gradient Boosting Practitioner
**Location**: `config/algorithm_config.yaml:117`, `common/ml/model_registry.py:33`

**Description**

The algorithm config sets `reg_lambda` for CatBoost. The canonical mapping has `l1_reg: None` for CatBoost (correct -- CatBoost doesn't support L1 regularization). However, `reg_lambda` in the YAML is passed through as a **native key** by `to_native_params()` (since it's not in the canonical mapping, it's passed unchanged).

CatBoost does not have a `reg_lambda` parameter. Its L2 regularization parameter is `l2_leaf_reg`. CatBoost silently ignores unrecognized parameters, so the intended extra regularization is not applied.

**Proposed Fix**

Remove `reg_lambda` from the CatBoost section of `algorithm_config.yaml`. L2 regularization is already configured via `l2_leaf_reg: 7.5`.

---

### D-013: Print Statements Bypass Structured Logging

**Expert**: Production ML & Reproducibility
**Location**: `common/ml/backtest_framework.py` (lines 300-304, 343, 414, 440-466, 530, 569, etc.), `scripts/run_backtest.py` (throughout `main()`)

**Description**

The backtest framework uses `print()` extensively for progress reporting, bypassing the `logging.getLogger(__name__)` logger that is defined at the module level. The project's own CLAUDE.md mandates "Scripts use `logging.getLogger(__name__)` -- no raw `print()`."

**Impact**

- Output cannot be filtered by log level in production
- Cannot be captured by centralized logging/monitoring systems
- Mixed `print()` and `logger.info()` in the same file creates inconsistent output formatting

**Proposed Fix**

Replace all `print(f"  [{_ts()}] ...")` calls with `logger.info(...)`. The `_ts()` timestamp helper is redundant when using structured logging with timestamp formatters.

---

### D-014: Future Masking Uses Zero Instead of NaN

**Expert**: Data Leakage Specialist
**Location**: `common/ml/feature_engineering.py:637`

**Description**

```python
df.loc[future_mask, "qty"] = 0
```

Future months are set to 0 rather than NaN. For **training rows**, this is harmless (their lag features only reference prior months with real data). For **prediction rows** in the direct (non-recursive) path, rolling features incorporate these artificial zeros:

- Month cutoff+3's `rolling_mean_3m` averages cutoff+2 (masked 0), cutoff+1 (masked 0), and cutoff (real)
- For continuous-demand DFUs, these artificial zeros drag the rolling mean toward zero, giving the model an unrealistically pessimistic demand signal
- For intermittent-demand DFUs, the model cannot distinguish "masked future" from "real zero-demand month"

**Impact**

In direct mode, prediction-row features for months far from the cutoff are biased toward zero. The model produces systematically lower predictions for distant months. In recursive mode, this is mitigated by the prediction write-back.

**Proposed Fix**

Use NaN instead of 0 for masking: `df.loc[future_mask, "qty"] = np.nan`. Lag/rolling computations with `min_periods=1` will naturally handle NaN by using only available real data.

---

### D-015: No Embargo Gap in Main Backtest Loop

**Expert**: Cross-Validation & Temporal Splitting
**Location**: `common/ml/backtest_framework.py:790-816`

**Description**

The main backtest loop sets `predict_start = train_end + 1 month` with no embargo gap. The tuning module correctly uses `gap_months=1` between training and validation folds, but the primary backtest framework does not.

For monthly data with lag features, `qty_lag_1` at month `train_end + 1` (first predict month) equals `qty` at `train_end` (last training month). This is valid -- the lag references training data. However, `rolling_mean_3m` at `train_end + 1` averages months `train_end`, `train_end - 1`, `train_end - 2` -- all training data, also valid.

The risk is subtle: without an embargo, the model's early stopping validation set (last 20% of training data) is immediately adjacent to the first prediction month. The validation metric optimizes for data that is maximally similar to the first predict month, potentially biasing early stopping.

**Proposed Fix**

Add a configurable `embargo_months` parameter (default 1) to `generate_timeframes()`:

```python
predict_start = train_end + pd.DateOffset(months=1 + embargo_months)
```

---

## Architectural Observations (Non-Defects)

These are not bugs but areas where the framework could be materially improved.

### A-001: No Model Ensemble or Stacking

The framework trains LGBM, CatBoost, and XGBoost independently and selects one champion. A simple weighted average of all three models would likely outperform any individual model by 1-3% WAPE due to model diversity (CatBoost uses ordered boosting, LGBM uses histogram splitting, XGBoost uses exact/approximate methods).

### A-002: Cluster Assignment Derived from Full Data

`dim_sku.ml_cluster` is computed from the entire dataset including months that fall in the prediction period. If clustering uses demand features (volume, variability, seasonality), the cluster labels encode future demand information. True temporal fidelity requires re-clustering per timeframe cutoff.

### A-003: No Per-DFU Accuracy Tracking

The framework reports aggregate accuracy (across all DFUs) but doesn't identify which specific DFUs are well-forecast vs. poorly-forecast. Per-DFU accuracy would enable targeted model improvement (e.g., different models for easy vs. hard items).

### A-004: No Automatic Model Selection Per DFU

The champion selection process picks one model for all DFUs. In practice, LGBM may outperform on smooth demand items while CatBoost excels on intermittent items. Per-DFU or per-cluster model selection would capture these complementary strengths.

### A-005: Rolling Features Not Weighted by Recency

`rolling_mean_3m` uses a simple average. An exponentially weighted moving average (EWMA) would give more weight to recent months, better capturing demand trends. Tree models can partially learn this weighting, but giving them the right input representation reduces the learning burden.

---

## Recommended Fix Priority

### Phase 1: Immediate (blocks valid accuracy reporting)

1. **D-002**: Wire XGBoost early stopping (1 line fix)
2. **D-001**: Recompute TS profiles per timeframe cutoff
3. **D-003**: Implement global-model fallback for small clusters

### Phase 2: Short-Term (improves accuracy and model comparison)

4. **D-009**: Align training objective with WAPE evaluation metric
5. **D-011**: Remove redundant Fourier/month_sin features
6. **D-012**: Remove silently-ignored CatBoost reg_lambda
7. **D-008**: Add per-timeframe accuracy breakdown to metadata
8. **D-013**: Replace print() with structured logging

### Phase 3: Medium-Term (framework maturity)

9. **D-007**: Add multi-seed evaluation
10. **D-004**: Add quantile regression for prediction intervals
11. **D-010**: Scale min cluster rows or pre-select features
12. **D-014**: Switch masking from zero to NaN
13. **D-015**: Add configurable embargo gap

### Phase 4: Long-Term (best-in-class)

14. **D-005**: Implement Tweedie/hurdle models for intermittent demand
15. **D-006**: Add teacher forcing for recursive mode
16. A-001: Implement model stacking/ensemble
17. A-002: Temporal cluster re-assignment per timeframe
