# Backtest Framework

> Tests forecast models against historical data across 10 expanding time windows, so you can measure accuracy before deploying to production.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `common/ml/backtest_framework.py`, `common/ml/model_registry.py`, `common/ml/feature_engineering.py`, `common/services/metrics.py`, `common/core/constants.py`, `scripts/etl/load_backtest_forecasts.py`, `sql/010_create_backtest_lag_archive.sql` |

---

## Problem

You cannot improve what you cannot measure. Without backtesting, the only way to know if a new algorithm is better is to deploy it and wait months for actuals to come in. This is slow, risky, and provides no statistical rigor. Planners need a way to evaluate models against known history before trusting them with real purchasing decisions.

## Solution

The backtesting framework trains models on progressively larger slices of history (expanding windows) and predicts forward into periods where actuals already exist. By comparing predictions to actuals across 10 time windows and 5 lag horizons, the platform produces a statistically robust accuracy profile for every model. A dual-path storage design preserves all lag horizons in an archive while storing only the operationally relevant execution-lag prediction in the main table.

## How It Works

1. The framework generates 10 expanding-window timeframes labeled A through J
2. Timeframe A trains on the shortest history; timeframe J trains on the longest
3. For each timeframe, the model predicts all remaining months up to the latest data
4. Each prediction's **natural lag** is computed from its timeframe: `lag = months(startdate - train_end) - 1`. For a given demand month, different timeframes produce predictions at different horizons (lags 0-4), with genuinely different `basefcst_pref` values because the model was trained on different data cutoffs
5. Only the execution-lag prediction (the lag that matters operationally for each DFU) goes into the main forecast table
6. After loading, 5 materialized views are refreshed for instant accuracy queries
7. The `model_id` column distinguishes predictions from different algorithms

### Expanding Window Example

With sales data from Feb 2023 to Jan 2026 (36 months), 10 timeframes:

| Timeframe | Training Period | Prediction Period | Natural Lags (for Jan 2026) |
|-----------|----------------|-------------------|---------------------------|
| A | Feb 2023 -- Mar 2025 | Apr 2025 -- Jan 2026 (10 months) | Jan: lag 9 (filtered out, >4) |
| ... | ... | ... | ... |
| F | Feb 2023 -- Aug 2025 | Sep 2025 -- Jan 2026 (5 months) | Jan: **lag 4** (5-month-ahead) |
| G | Feb 2023 -- Sep 2025 | Oct 2025 -- Jan 2026 (4 months) | Jan: **lag 3** (4-month-ahead) |
| H | Feb 2023 -- Oct 2025 | Nov 2025 -- Jan 2026 (3 months) | Jan: **lag 2** (3-month-ahead) |
| I | Feb 2023 -- Nov 2025 | Dec 2025 -- Jan 2026 (2 months) | Jan: **lag 1** (2-month-ahead) |
| J | Feb 2023 -- Dec 2025 | Jan 2026 (1 month) | Jan: **lag 0** (1-month-ahead) |

Each lag for Jan 2026 comes from a different timeframe with a different training data cutoff, producing genuinely different predictions. Lag 0 uses the most recent data (highest accuracy), lag 4 uses 5-month-old data (lowest accuracy).

### Execution Lag

Each DFU (Demand Forecast Unit -- a unique item-location combination) has an `execution_lag` that represents how far in advance its forecast is issued. A DFU with `execution_lag = 2` means the forecast for April is issued in February. The main table stores only the prediction at this operationally relevant lag; the archive stores all 5 lags for accuracy analysis at any horizon.

**External forecast loading:** All rows in `dfu_stat_fcst.txt` are assumed to be at execution lag. The `lag` and `execution_lag` fields in the source file are ignored — the loader overwrites both from `dim_sku.execution_lag` (defaulting to 0 for unmatched DFUs). No `WHERE lag = execution_lag` filter is applied; all rows are inserted. Additionally, only the last 12 months of data (by `startdate`) are loaded, based on the current planning date.

**Backtest loading:** Backtests still produce predictions at all 5 lags (0-4). The backtest loader (`scripts/etl/load_backtest_forecasts.py`) retains the original dual-path logic: archive gets all lags, main table gets execution-lag rows only.

## Plain-Language Overview

### The Core Idea

> "If I had only known the data up to date X, how accurate would my model have been?"

We do this **10 times** with different cutoff dates, then measure accuracy across all of them.

---

### Step 1: The 10 Timeframes (Expanding Windows)

Imagine you have 3 years of sales data (Jan 2023 → Dec 2025).

We create 10 "training windows" — each one gets a bit more data than the last:

```
                    TRAIN DATA              →    PREDICT
Timeframe A  [Jan 2023 ···· Feb 2025]  →  [Mar–Dec 2025]  (10 prediction months)
Timeframe B  [Jan 2023 ···· Mar 2025]  →  [Apr–Dec 2025]  (9 prediction months)
Timeframe C  [Jan 2023 ···· Apr 2025]  →  [May–Dec 2025]
...
Timeframe J  [Jan 2023 ···· Nov 2025]  →  [Dec 2025]      (1 prediction month)
```

Each timeframe trains a **real model** and makes **real predictions** — we compare against actual sales to measure accuracy.

---

### Step 2: Lag — How Far Ahead is the Forecast?

For any given month (e.g., Dec 2025), we get a prediction from multiple timeframes:

| Timeframe | Train Cutoff | Predicting Dec 2025 | Lag |
|---|---|---|---|
| J | Nov 2025 | 1 month ahead | **lag 0** |
| I | Oct 2025 | 2 months ahead | **lag 1** |
| H | Sep 2025 | 3 months ahead | **lag 2** |
| G | Aug 2025 | 4 months ahead | **lag 3** |
| F | Jul 2025 | 5 months ahead | **lag 4** |

**lag 0** = most recent data → highest accuracy
**lag 4** = stale data → lower accuracy

All 5 lags go into the **`backtest_lag_archive`** table. This lets us answer: *"how does accuracy degrade as we forecast further out?"*

---

### Step 3: Execution Lag — Which Lag Actually Matters for This DFU?

Each DFU has an `execution_lag` — the number of months in advance the forecast must be issued for operations.

**Example:** A DFU with `execution_lag = 2` needs its December forecast ready in October → only the **lag 2** prediction matters operationally.

```
Archive table  → all 5 lags (for accuracy analysis)
Main table     → execution_lag prediction only (for planning)
```

---

### Step 4: Features the Model Trains On

For a DFU like *Item A / Customer Group East / Location NYC*:

```
qty_lag_1       = sales 1 month ago
qty_lag_2       = sales 2 months ago
...
qty_lag_12      = sales 12 months ago
rolling_3m_mean = avg of last 3 months
rolling_6m_mean = avg of last 6 months
month_of_year   = 12 (December)
region          = "WEST"                   ← categorical attribute
... ~30 features total (reduced by multi-stage selection pipeline)
```

---

### Step 5: Per-Cluster Training

Instead of one global model, we train **one model per demand cluster**:

```
Cluster "sparse_intermittent"  → tuned for lumpy, zero-heavy demand
Cluster "seasonal_dominant"    → tuned for strong seasonality
Cluster "high_volume_stable"   → tuned for smooth, predictable demand
...
```


---

### Full Example End-to-End

**DFU:** Widget-A / Customer Group East / Warehouse NYC
**Execution lag:** 2
**Cluster:** seasonal_dominant

```
1. Timeframe H trains on Jan 2023 – Sep 2025
   → predicts Widget-A NYC for Oct–Dec 2025
   → lag for Dec 2025 = 2 months ahead (lag 2)

2. Timeframe J trains on Jan 2023 – Nov 2025
   → predicts Widget-A NYC for Dec 2025 only
   → lag = 0 (most recent data)

3. All predictions land in backtest_lag_archive with lags 0–4

4. execution_lag=2 → only lag 2 prediction goes to main table
   (the October prediction of December demand)

5. Accuracy computed across lags:
   lag 0 WAPE = 8.2%   ← best (most data)
   lag 2 WAPE = 14.1%  ← what operations actually uses
   lag 4 WAPE = 22.7%  ← worst (oldest data)
```

---

### Why 10 Timeframes Instead of Just One?

Two reasons — one is operational necessity, one is statistical:

**1. Lag coverage (the primary reason)**
We support `execution_lag` values from 0 to 4, meaning some DFUs need their forecast issued up to 5 months in advance. To produce a lag 4 prediction for any given demand month, you need a timeframe whose training cutoff is 5 months earlier. Covering lags 0–4 for even a *single* demand month requires at least 5 distinct timeframes (F through J in the example above). 10 timeframes ensures every demand month near the planning date has full lag 0–4 coverage.

**2. Statistical robustness**
A single train/test split gives a noisy accuracy estimate that reflects one slice of market conditions. 10 expanding windows produce predictions across many months — different seasons, demand shocks, and trend inflections — so the accuracy metric reflects genuine model skill rather than a lucky or unlucky split.

---

### Step 6: Per-Cluster SHAP Feature Selection

When `shap_select: true`, SHAP runs independently per cluster via `compute_timeframe_shap_per_cluster()`:

1. Pre-SHAP stages (0-2: duplicate removal, variance filter, correlation filter) are shared globally
2. For each cluster, SHAP values are computed on that cluster's training data only
3. Per-cluster cumulative selection picks features covering `shap_threshold` (default 0.90) of importance
4. Sparse clusters (>50% zeros) use stratified 50/50 sampling; clusters with too few non-zero rows keep all features
5. Returns `dict[str, list[str]]` — each cluster gets its own selected feature list
6. Protected features (`PROTECTED_FEATURES` in `constants.py`) always survive selection

### Step 7: Per-Cluster Tuning Profiles

Each cluster can receive hyperparameter overrides from `config/forecasting/cluster_tuning_profiles.yaml`:

1. **Phase 1:** Exact match by `cluster_name` in `match_criteria` (e.g., `cluster_name: high_volume_periodic`)
2. **Phase 2:** Statistical criteria fallback (mean_demand, cv_demand, zero_demand_pct, seasonal_amplitude, n_rows)
3. First match wins per `_PROFILE_PRIORITY` order (sparse_intermittent first, default last)
4. Profiles can override any LGBM parameter (num_leaves, learning_rate, n_estimators, etc.)

### Step 8: Recursive Lag Smoothing

When `recursive: true` and `recursive_lag_smooth > 0` (default 0.15), lag features are exponentially smoothed from step 3 onward:

```
lag_t = alpha * prediction + (1-alpha) * lag_{t-1}
```

This damps compounding oscillations in later recursive steps without losing the recency signal in steps 1-2.

---

## Data Model

### Main Table: `fact_external_forecast_monthly`

Stores execution-lag predictions only. One row per DFU per month per model.

### Archive Table: `backtest_lag_archive`

| Column | Type | Description |
|--------|------|-------------|
| `forecast_ck` | TEXT | Composite business key |
| `item_id`, `loc` | TEXT | Item and location |
| `fcstdate` | DATE | When the forecast was issued |
| `startdate` | DATE | Month being forecast |
| `lag` | INTEGER | 0-4 (months between issue and target) |
| `basefcst_pref` | NUMERIC | Forecast quantity |
| `tothist_dmd` | NUMERIC | Actual demand |
| `model_id` | TEXT | Algorithm identifier |
| `timeframe` | TEXT | Backtest timeframe (A-J) |

**Constraint:** `UNIQUE(forecast_ck, model_id, lag)`

### Dual-Path Loading (Critical Ordering)

The loader uses phase ordering to preserve archive integrity:
1. **12-month filter** removes staging rows with `startdate` older than 12 months from planning date
2. **Archive load** inserts remaining rows into `backtest_lag_archive` FIRST from untouched staging data (original lag values preserved)
3. **Execution lag resolution** overwrites both `lag` and `execution_lag` on staging from `dim_sku`
4. **Main table INSERT** loads all rows (no lag filter — all external forecasts are assumed at execution lag)

This ordering is critical: the archive must be loaded before the staging mutation, otherwise all rows for a DFU would have the same lag value. Backtest model rows in the archive are not affected (only `model_id='external'` rows are replaced).

## API

No new endpoints. Existing multi-model endpoints handle backtest data automatically:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/domains/forecast/models` | Lists all model_ids including backtest models |
| GET | `/forecast/accuracy/slice` | Accuracy by dimension for any model |
| GET | `/forecast/accuracy/lag-curve` | Accuracy degradation across lags 0-4 |

## Pipeline

| Target | Description |
|--------|-------------|
| `make backtest-lgbm` | Run LightGBM backtest (10 timeframes) |
| `make backtest-all` | Run all three sequentially |
| `make backtest-all-parallel` | Run all three in parallel |
| `make backtest-load MODEL=lgbm_cluster` | Load one model's predictions into Postgres |
| `make backtest-load-all` | Load all models found under `data/backtest/*/` |
| `make backtest-list` | Show row counts per model in forecast + archive tables |
| `make backtest-clean MODELS="lgbm_cluster"` | Remove specific model predictions |

### Output Directory Structure

Each backtest writes to `data/backtest/<model_id>/`:
- `backtest_predictions.csv` -- execution-lag only (for main table)
- `backtest_predictions_all_lags.csv` -- all lags 0-4 (for archive)
- `backtest_metadata.json` -- run configuration and metrics
- `feature_importance.csv` -- model feature rankings

## Model Registry (`common/ml/model_registry.py`)

The model registry provides a centralized abstraction layer for all tree-based models, eliminating duplicate code across backtest scripts:

### Canonical Parameter Mapping

|-----------|------|----------|---------|
| `estimators` | `n_estimators` | `iterations` | `n_estimators` |
| `max_depth` | `max_depth` | `depth` | `max_depth` |
| `l2_reg` | `reg_lambda` | `l2_leaf_reg` | `reg_lambda` |
| `l1_reg` | `reg_alpha` | _(not supported)_ | `reg_alpha` |
| `min_leaf_samples` | `min_child_samples` | `min_data_in_leaf` | `min_child_weight` |
| `col_sample` | `colsample_bytree` | `colsample_bylevel` | `colsample_bytree` |

### Unified Functions

- **`fit_model()`** — single fit function replacing 3× duplicate if/elif/else blocks in `_train_single_cluster` and `train_and_predict_global`
- **`compute_early_stop_patience()`** — standardized 3% of max iterations (floor 10) for all models
- **`to_native_params()` / `from_native_params()`** — bidirectional canonical ↔ native translation

### Early Stopping Standardization

All models use `compute_early_stop_patience(max_iterations, pct=0.03)`:
- LGBM 1500 iterations → 45 rounds patience

## Configuration

Backtest behavior is controlled by `config/forecasting/forecast_pipeline_config.yaml`. See [Forecast Pipeline Config](./19-forecast-pipeline-config.md) for details.

Key backtest-level settings:
- `early_stop_pct: 0.05` — early stopping patience as percentage of max iterations (10% for sparse clusters)
- `shap_retrain_threshold: 0.50` — retrain safety check threshold (effectively disabled; original model consistently outperforms)
- `recursive_noise_pct: 0.03` — Gaussian noise for recursive training (reduced from 0.08)
- `recursive_lag_smooth: 0.15` — exponential smoothing for recursive lags from step 3+
- `embargo_months: 0` — preserves one-month-ahead lag 0; prediction starts in the month after training ends

### Model persistence under embargo

When the `.pkl`-persisting backtest mode runs (`model_persistence_fn` set), production-model artifacts are written for the **last timeframe that has a non-empty predict window**, resolved by `_last_persistable_timeframe()` — not blindly the last timeframe index.

> **Lag contract (2026-07-11):** operational backtests use embargo 0 because the natural next-month boundary already prevents same-month scoring. A positive embargo shifts the shortest available natural lag upward and must not be used for the standard lag 0–4 accuracy contract. `_last_persistable_timeframe()` remains a defensive guard for custom windows. Separately, `_inject_recursive_noise` is NaN-safe.

## Backtest Model Coverage

The backtest framework supports the full algorithm portfolio defined in `config/forecasting/forecast_pipeline_config.yaml`:

| Category | Models | Make Target |
|----------|--------|-------------|
| Decomposition | `mstl` (Multiple Seasonal-Trend via LOESS) | `make backtest-mstl` |
| Deep learning | `nhits` (N-HiTS), `nbeats` (N-BEATS) | `make backtest-nhits`, `backtest-nbeats` |

All models write to the same `data/backtest/<model_id>/` directory structure and are loaded through the same dual-path loader into `backtest_lag_archive` (all lags) and `fact_external_forecast_monthly` (execution lag only).

Managed runs remove prior final CSV/metadata artifacts before execution (resume checkpoints are
preserved), and database auto-load occurs only after the model process exits successfully. MSTL
runs install the `statistical` dependency extra and fail when inference or post-processing produces
no rows; an unavailable `statsforecast` installation can therefore never load stale results or be
reported as completed.

Production batch inference preserves the trained feature-column names when calling tree-model
`predict()`. This keeps sklearn/LightGBM schema validation active and prevents per-batch
feature-name warnings from flooding durable UI job logs.

## Dependencies

- [Multi-Model Support](./02-multi-model.md) -- `model_id` column in the forecast table
- Clustering (in `03-demand-intelligence/`) -- provides `ml_cluster` partition metadata for per-cluster training

## See Also

- [Advanced Backtest](./05-advanced-backtest.md) -- tuning, SHAP, and recursive extensions
- [Forecast Pipeline Config](./19-forecast-pipeline-config.md) -- config file that controls backtest behavior
- [Chronos Foundation Models](./18-chronos-foundation-models.md) -- the remaining Chronos 2 Enriched foundation model
- [Champion Selection](./07-champion-selection.md) -- picks the best model per DFU-month from backtest results
