# Backtesting Review

> Reviewed: 2026-03-25
> Reviewer: Codex static review
> Scope: `common/ml/backtest_framework.py`, `common/ml/feature_engineering.py`, `common/ml/tuning.py`, `common/ml/shap_selector.py`, `scripts/run_backtest.py`, `config/algorithm_config.yaml`, `docs/specs/02-forecasting/03-backtest-framework.md`

## Findings

### 1. Critical: TS-profile features leak future information

The framework builds time-series profile features such as `mean_demand`, `recency_ratio`, `seasonal_amplitude`, and `adi` once on the full history inside `build_feature_matrix()`. Later, `mask_future_sales()` recomputes lag and rolling features but does not recompute those TS-profile columns. This means earlier backtest cutoffs still retain signal from future months.

Impact:
- Invalidates backtest accuracy as a causal estimate.
- Also contaminates SHAP-based feature selection.
- Also contaminates inline tuning, because tuning folds use `mask_future_sales(full_grid, train_end_fold)`.

References:
- `common/ml/feature_engineering.py:571`
- `common/ml/feature_engineering.py:622`
- `common/ml/tuning.py:404`

Recommendation:
- Recompute TS-profile features per cutoff or per fold.
- If that is too expensive, remove those features from backtesting until a causal implementation exists.

### 2. High: Per-cluster validation split is not truly time-aware

`_train_single_cluster()` says it uses a time-aware split, but it actually takes the last 20% of rows from the current cluster slice. That only works if the cluster slice is explicitly month-sorted. In this framework, row order can still be driven by the SKU-major grid layout, so validation can be dominated by later rows from later SKUs instead of later calendar periods.

Impact:
- Early stopping is guided by a noisy or non-causal validation set.
- Reported `val_wape` is less trustworthy than the comments imply.

Reference:
- `scripts/run_backtest.py:252`

Recommendation:
- Sort cluster training data by `startdate` before splitting.
- Better: split by calendar month, not row count.

### 3. Medium: Benchmarking discipline is weaker than the modeling stack

The framework has solid support for LightGBM, CatBoost, and XGBoost, but it does not appear to benchmark those models inside the same runner against simple baselines such as seasonal naive, last-value, rolling mean, or Croston/SBA.

Impact:
- It is possible to prove one complex tree model beats another without proving that the framework beats a disciplined low-tech baseline.
- This is especially risky for intermittent monthly demand.

References:
- `scripts/run_backtest.py:54`
- `scripts/run_backtest.py:578`
- `docs/specs/02-forecasting/03-backtest-framework.md:19`

Recommendation:
- Add first-class benchmark models to the same backtest harness.
- Report every model against those baselines by horizon and by demand segment.

### 4. Medium: Series coverage is biased toward demand-bearing DFUs

The loader filters the DFU attribute set down to only DFUs that have sales history in the loaded period.

Impact:
- Backtest metrics may look better than operational reality if production must also forecast sparse, new, or recently inactive DFUs.
- Cold-start and near-cold-start difficulty is excluded from the main score.

Reference:
- `common/ml/backtest_framework.py:288`

Recommendation:
- Keep a separate cold-start / sparse-series evaluation cohort.
- Make the headline metric explicit about whether it is “active-series only.”

### 5. Medium: The timeframe design is useful operationally, but its statistical interpretation is overstated

The 10-window setup is a reasonable rolling-origin design, but the windows overlap heavily and each origin predicts the full remaining tail. That is operationally useful, yet it should not be described too strongly as “statistically robust” without qualification.

Impact:
- Aggregated metrics can overstate effective independent evidence.
- Users may interpret the score as stronger than it is.

References:
- `common/ml/backtest_framework.py:185`
- `docs/specs/02-forecasting/03-backtest-framework.md:23`

Recommendation:
- Emphasize horizon-level and cutoff-level reporting.
- Treat aggregate portfolio accuracy as descriptive, not definitive.

## Overall Assessment

The modeling choice is reasonable: tree boosters are a practical fit for large monthly tabular SKU-location forecasting problems. The main weakness is evaluation hygiene, not model family choice. Until leakage and validation design are tightened, uplift claims from this framework should be treated as directionally useful rather than audit-grade.

## Priority Fix Order

1. Remove or causalize TS-profile leakage.
2. Replace per-cluster row-tail validation with month-based validation.
3. Add baseline benchmark models into the same backtest runner.
4. Add separate reporting for sparse and cold-start cohorts.

