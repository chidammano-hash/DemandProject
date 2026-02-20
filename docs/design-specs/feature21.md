# Feature 21 — Prophet Backtesting Implementation

## Overview

Facebook Prophet-based demand forecasting integrated into the existing expanding-window backtest framework. Prophet is a time series forecasting model designed for business time series with strong seasonality and trend components. Unlike the tree-based models (LGBM, CatBoost, XGBoost) which use engineered lag/rolling features, Prophet natively models trend, seasonality, and holidays — making it a fundamentally different forecasting approach in the model competition.

## Problem

The current model roster (LGBM, CatBoost, XGBoost) consists entirely of gradient-boosted tree models that share the same feature engineering pipeline. While they compete well against each other, they represent a narrow class of algorithms. Adding Prophet diversifies the model ensemble with:

1. **Native seasonality modeling** — auto-detects yearly/weekly/monthly patterns without manual lag engineering
2. **Trend decomposition** — explicitly models and projects growth/decline trends
3. **Interpretable components** — additive/multiplicative decomposition into trend + seasonality + holidays
4. **Robustness to missing data** — handles gaps in time series without zero-fill distortion
5. **Per-DFU fitting** — fits individual time series models (unlike global tree models), capturing DFU-specific patterns

## Architecture

### Why Prophet Differs from Tree Models

| Aspect | Tree Models (LGBM/CatBoost/XGBoost) | Prophet |
|--------|--------------------------------------|---------|
| Approach | Global model across all DFUs | Per-DFU individual model |
| Features | Engineered lags, rolling stats, calendar, attributes | Raw time series (ds, y) + optional regressors |
| Seasonality | Encoded via month_sin/month_cos features | Native Fourier series decomposition |
| Trend | Captured implicitly via lag features | Explicit piecewise linear/logistic growth |
| Training | One model on 100K+ rows | One model per DFU on 12-36 rows |
| Prediction | Batch predict all DFUs at once | Loop over DFUs, predict each individually |
| Speed | Fast (single model) | Slower (N models, one per DFU) |
| Cluster strategy | Global / per-cluster / transfer | Global only (per-DFU by nature) |

### Data Flow

```
fact_sales_monthly + dim_dfu
          ↓
  For each timeframe (A-J):
    ┌────────────────────────┐
    │ Mask future sales      │
    │ (same causality logic) │
    └──────────┬─────────────┘
               ↓
    For each DFU:
      ┌──────────────────────┐
      │ Extract (ds, y) pair │
      │ from sales history   │
      │       ↓              │
      │ Fit Prophet model    │
      │       ↓              │
      │ Predict future months│
      └──────────┬───────────┘
               ↓
    Collect all DFU predictions
          ↓
  Assign execution_lag, forecast_ck
          ↓
  Deduplicate across timeframes
          ↓
  Output: backtest_predictions.csv
          backtest_predictions_all_lags.csv
```

### Timeframe Structure (Same as All Models)

Uses the identical `generate_timeframes()` function from the backtest framework:

```python
def generate_timeframes(earliest, latest, n=10):
    """10 expanding windows labeled A-J."""
    timeframes = []
    for i in range(n):
        train_end = latest - pd.DateOffset(months=(n - i))
        predict_start = train_end + pd.DateOffset(months=1)
        timeframes.append({
            "label": chr(ord("A") + i),
            "train_start": earliest,
            "train_end": train_end.normalize(),
            "predict_start": predict_start,
            "predict_end": latest,
        })
    return timeframes
```

| Timeframe | Train Window | Predict Window |
|-----------|-------------|---------------|
| A | earliest → latest - 10mo | latest - 9mo → latest |
| B | earliest → latest - 9mo | latest - 8mo → latest |
| ... | ... | ... |
| J | earliest → latest - 1mo | latest |

## Implementation Design

### Script: `mvp/demand/scripts/run_backtest_prophet.py`

#### CLI Interface

```bash
uv run python scripts/run_backtest_prophet.py \
    --model-id prophet_global \
    --n-timeframes 10 \
    --min-history 12 \
    --yearly-seasonality auto \
    --weekly-seasonality false \
    --growth linear \
    --changepoint-prior-scale 0.05 \
    --seasonality-prior-scale 10.0 \
    --n-changepoints 25 \
    --seasonality-mode additive \
    --parallel 4
```

#### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--model-id` | `prophet_global` | Model ID stored in output CSV |
| `--n-timeframes` | `10` | Number of expanding windows (A-J) |
| `--min-history` | `12` | Minimum months of sales history per DFU to fit Prophet |
| `--yearly-seasonality` | `auto` | Yearly seasonality: `auto`, `true`, `false` |
| `--weekly-seasonality` | `false` | Weekly seasonality (disabled — monthly grain) |
| `--growth` | `linear` | Trend model: `linear` or `logistic` |
| `--changepoint-prior-scale` | `0.05` | Flexibility of trend changes (lower = smoother) |
| `--seasonality-prior-scale` | `10.0` | Strength of seasonality component |
| `--n-changepoints` | `25` | Number of potential trend changepoints |
| `--seasonality-mode` | `additive` | Seasonality mode: `additive` or `multiplicative` |
| `--parallel` | `4` | Number of parallel DFU fits (multiprocessing) |
| `--cap` | `None` | Capacity cap for logistic growth (required if `--growth logistic`) |
| `--cluster-strategy` | `global` | Strategy: `global`, `per_cluster`, or `pooled` |

#### Core Algorithm

```python
def fit_and_predict_dfu(
    dfu_ck: str,
    history: pd.DataFrame,   # columns: ds (datetime), y (qty)
    future_months: list[pd.Timestamp],
    prophet_params: dict,
    min_history: int = 12,
) -> pd.DataFrame | None:
    """Fit Prophet on one DFU's history, predict future months."""

    # Skip DFUs with insufficient history
    if len(history) < min_history:
        return None

    # Suppress Stan/cmdstanpy logging
    model = Prophet(
        growth=prophet_params["growth"],
        yearly_seasonality=prophet_params["yearly_seasonality"],
        weekly_seasonality=False,  # monthly grain — no weekly
        daily_seasonality=False,   # monthly grain — no daily
        changepoint_prior_scale=prophet_params["changepoint_prior_scale"],
        seasonality_prior_scale=prophet_params["seasonality_prior_scale"],
        n_changepoints=prophet_params["n_changepoints"],
        seasonality_mode=prophet_params["seasonality_mode"],
    )

    # Add monthly seasonality explicitly (12-month cycle)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(history)

    # Build future dataframe for prediction months
    future = pd.DataFrame({"ds": future_months})
    forecast = model.predict(future)

    # Extract point forecast, floor at 0
    result = forecast[["ds", "yhat"]].copy()
    result["yhat"] = result["yhat"].clip(lower=0)

    return result
```

#### Per-Timeframe Loop (Same Pattern as Tree Models)

```python
for tf in timeframes:
    # Mask: only use sales up to train_end
    train_sales = sales_df[sales_df["startdate"] <= tf["train_end"]]

    # Determine prediction months
    predict_months = pd.date_range(
        tf["predict_start"], tf["predict_end"], freq="MS"
    )

    # Fit Prophet per DFU (parallelized)
    with Pool(n_parallel) as pool:
        results = pool.starmap(fit_and_predict_dfu, [
            (dfu_ck, dfu_history, predict_months, params, min_history)
            for dfu_ck, dfu_history in grouped_sales
        ])

    # Collect predictions for this timeframe
    tf_preds = pd.concat([r for r in results if r is not None])
    tf_preds["timeframe"] = tf["label"]
    all_preds.append(tf_preds)
```

#### Parallelization Strategy

Prophet fits are embarrassingly parallel (each DFU is independent). Use Python `multiprocessing.Pool` to fit N DFUs concurrently:

- Default: 4 workers (safe for MacBook)
- Each worker fits one DFU at a time
- Memory: ~50MB per Prophet model instance
- Expected throughput: ~200-500 DFUs/minute depending on history length

#### Output Format (Identical to Tree Models)

**Main CSV (`backtest_predictions.csv`):**
```
forecast_ck,dmdunit,dmdgroup,loc,fcstdate,startdate,lag,execution_lag,basefcst_pref,tothist_dmd,model_id
100320_GRP1_1401-BULK_2025-03-01_2025-04-01,100320,GRP1,1401-BULK,2025-03-01,2025-04-01,1,1,5234.5,5100.0,prophet_global
```

**Archive CSV (`backtest_predictions_all_lags.csv`):**
Same expansion to lags 0-4 using the shared `expand_to_all_lags()` function, with `timeframe` column.

**Metadata JSON (`backtest_metadata.json`):**
```json
{
    "model_id": "prophet_global",
    "cluster_strategy": "global",
    "n_timeframes": 10,
    "prophet_params": {
        "growth": "linear",
        "yearly_seasonality": "auto",
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "n_changepoints": 25,
        "seasonality_mode": "additive",
        "min_history": 12
    },
    "n_predictions": 45000,
    "n_dfus": 1200,
    "n_dfus_skipped": 34,
    "accuracy_at_execution_lag": {
        "wape": 18.5,
        "bias": 0.02,
        "accuracy_pct": 81.5
    }
}
```

### Cluster Strategies

Prophet naturally fits one model per DFU, so the cluster strategy concept is different from tree models:

| Strategy | Model ID | Behavior |
|----------|----------|----------|
| `global` | `prophet_global` | Fit independent Prophet model per DFU (default) |
| `per_cluster` | `prophet_cluster` | Same as global but only fit DFUs within each cluster; skip unassigned |
| `pooled` | `prophet_pooled` | Pool cluster sales into one aggregated series per cluster, fit one Prophet per cluster, then disaggregate predictions back to DFU level proportionally |

**Recommended strategy:** `global` (default). Prophet already fits per-DFU, so per-cluster adds no value unless pooling. The `pooled` strategy is useful for sparse/intermittent DFUs where individual history is too short but cluster-level aggregation is smooth enough for Prophet to model.

#### Pooled Strategy Detail

```python
def train_and_predict_pooled(
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    predict_months: list,
    prophet_params: dict,
):
    """Pool DFU sales by cluster, fit Prophet per cluster, disaggregate."""

    # 1. Aggregate sales by cluster + month
    cluster_sales = (
        sales_df.merge(dfu_attrs[["dfu_ck", "ml_cluster"]], on="dfu_ck")
        .groupby(["ml_cluster", "startdate"])["qty"]
        .sum()
        .reset_index()
    )

    # 2. Fit Prophet per cluster on aggregated series
    cluster_forecasts = {}
    for cluster, group in cluster_sales.groupby("ml_cluster"):
        history = group.rename(columns={"startdate": "ds", "qty": "y"})
        forecast = fit_prophet(history, predict_months, prophet_params)
        cluster_forecasts[cluster] = forecast

    # 3. Disaggregate to DFU level using historical proportion
    #    proportion = DFU's avg monthly qty / cluster's avg monthly qty
    dfu_proportions = compute_dfu_proportions(sales_df, dfu_attrs)

    dfu_preds = []
    for dfu_ck, row in dfu_proportions.iterrows():
        cluster = row["ml_cluster"]
        proportion = row["proportion"]
        cluster_fcst = cluster_forecasts.get(cluster)
        if cluster_fcst is not None:
            dfu_fcst = cluster_fcst.copy()
            dfu_fcst["basefcst_pref"] = dfu_fcst["yhat"] * proportion
            dfu_fcst["dfu_ck"] = dfu_ck
            dfu_preds.append(dfu_fcst)

    return pd.concat(dfu_preds)
```

### Shared Infrastructure Reuse

The following components are reused without modification:

| Component | Source | Reuse |
|-----------|--------|-------|
| `generate_timeframes()` | Shared backtest utilities | Identical expanding windows |
| `assign_execution_lag()` | Shared backtest utilities | Same forecast_ck construction |
| `expand_to_all_lags()` | Shared backtest utilities | Same lag 0-4 expansion |
| `load_backtest_forecasts.py` | Shared loader | Same CSV schema, same COPY+upsert |
| Deduplication logic | Shared | Latest timeframe wins per (forecast_ck, model_id) |
| Materialized view refresh | Shared loader | Same `agg_forecast_monthly`, `agg_accuracy_by_dim` refresh |
| MLflow logging | `demand_backtest` experiment | Same experiment, different run tags |

### Avoiding Clashes with PatchTST

To ensure clean parallel development:

| Concern | Prophet (Feature 21) | PatchTST |
|---------|---------------------|----------------------|
| Script file | `run_backtest_prophet.py` | `run_backtest_patchtst.py` |
| Model IDs | `prophet_global`, `prophet_cluster`, `prophet_pooled` | `patchtst_global`, `patchtst_cluster`, `patchtst_transfer` |
| Python dependency | `prophet` | `torch` (already present), custom PatchTST module |
| Makefile targets | `backtest-prophet`, `backtest-prophet-cluster`, `backtest-prophet-pooled` | `backtest-patchtst`, `backtest-patchtst-cluster`, `backtest-patchtst-transfer` |
| Config section | `prophet_params` in metadata JSON | `patchtst_params` in metadata JSON |

No shared mutable state between the two — they share only read-only infrastructure (timeframe generation, execution lag assignment, CSV loader).

## Dependencies

### New Python Package

Add to `mvp/demand/pyproject.toml`:
```toml
"prophet>=1.1.5",
```

Prophet depends on `cmdstanpy` (Stan backend). On first install, it auto-downloads the Stan compiler (~100MB). No additional manual setup required with `uv`.

### Suppress Prophet Logging

Prophet and cmdstanpy are verbose by default. Suppress in the script:
```python
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
```

## Makefile Targets

```makefile
# Prophet backtesting
backtest-prophet:
	$(UV) python scripts/run_backtest_prophet.py --model-id prophet_global

backtest-prophet-cluster:
	$(UV) python scripts/run_backtest_prophet.py --cluster-strategy per_cluster --model-id prophet_cluster

backtest-prophet-pooled:
	$(UV) python scripts/run_backtest_prophet.py --cluster-strategy pooled --model-id prophet_pooled
```

Usage:
```bash
make backtest-prophet       # Global Prophet backtest (per-DFU fits)
make backtest-load          # Load into Postgres (same shared loader)

make backtest-prophet-cluster   # Per-cluster Prophet (skip unassigned DFUs)
make backtest-load

make backtest-prophet-pooled    # Pooled cluster Prophet (aggregate → disaggregate)
make backtest-load
```

## Model Competition Integration

Add Prophet model IDs to `config/model_competition.yaml`:

```yaml
competition:
  models:
  - lgbm_global
  - lgbm_cluster
  - lgbm_transfer
  - catboost_global
  - catboost_cluster
  - xgboost_global
  - xgboost_cluster
  - prophet_global        # ← new
  - prophet_pooled        # ← new (if run)
```

Prophet models then automatically participate in champion selection via `make champion-select`.

## Performance Considerations

### Speed

Prophet is significantly slower than tree models because it fits one model per DFU:

| Model | ~1200 DFUs × 10 timeframes | Parallelism |
|-------|---------------------------|-------------|
| LGBM global | ~2 minutes | Single model |
| CatBoost global | ~3 minutes | Single model |
| XGBoost global | ~2 minutes | Single model |
| Prophet global | ~30-60 minutes | 4 parallel workers |

**Mitigation strategies:**
1. Parallelization via `multiprocessing.Pool` (default 4 workers)
2. Skip DFUs with < `min_history` months (avoids fitting noise)
3. Disable unnecessary seasonalities (weekly, daily — not applicable for monthly data)
4. Reduce `n_changepoints` for faster fitting on short series

### Memory

- Each Prophet model: ~50MB
- With 4 parallel workers: ~200MB peak
- Total with data: ~500MB (well within MacBook limits)

### Accuracy Expectations

Prophet typically excels at:
- DFUs with strong yearly seasonality (seasonal products)
- DFUs with clear trend (growing/declining demand)
- DFUs with enough history (24+ months)

Prophet typically underperforms tree models on:
- Intermittent/sparse demand (many zero months)
- Short history DFUs (< 18 months)
- DFUs where external features (region, brand, category) matter more than time patterns

**Expected outcome:** Prophet wins champion selection for ~10-20% of DFUs (seasonal, trending items), while tree models win the majority. This is the value — ensemble diversity through champion selection.

## Key Files

| File | Purpose |
|------|---------|
| `mvp/demand/scripts/run_backtest_prophet.py` | Prophet backtest script (new) |
| `mvp/demand/scripts/load_backtest_forecasts.py` | Shared loader (unchanged) |
| `mvp/demand/config/model_competition.yaml` | Add `prophet_global` / `prophet_pooled` to models list |
| `mvp/demand/pyproject.toml` | Add `prophet>=1.1.5` dependency |
| `mvp/demand/Makefile` | Add `backtest-prophet`, `backtest-prophet-cluster`, `backtest-prophet-pooled` targets |

## Testing & Validation

### Quick Smoke Test
```bash
# Fit Prophet on a small subset (first 50 DFUs, 3 timeframes)
uv run python scripts/run_backtest_prophet.py \
    --model-id prophet_test \
    --n-timeframes 3 \
    --max-dfus 50

# Check output
ls -lh data/backtest/backtest_predictions.csv
head -5 data/backtest/backtest_predictions.csv
```

### Full Validation
```bash
# Run full Prophet backtest
make backtest-prophet
make backtest-load

# Verify in database
docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
    -c "SELECT model_id, COUNT(*) FROM fact_external_forecast_monthly WHERE model_id LIKE 'prophet%' GROUP BY 1"

# Check accuracy in UI
# Navigate to Forecast → Model selector → prophet_global
# Compare KPIs against lgbm_global, catboost_global, xgboost_global

# Run champion selection including Prophet
make champion-select
```

## Future Enhancements (Out of Scope for Feature 21)

1. **Holiday effects** — Add US federal holidays and state-specific holidays as Prophet holiday regressors
2. **External regressors** — Feed promotions, price, or weather data as additional regressors
3. **Multiplicative seasonality** — Auto-detect when to use multiplicative vs additive based on coefficient of variation
4. **Prophet hyperparameter tuning** — Per-cluster or per-DFU hyperparameter optimization via cross-validation
5. **Ensemble blending** — Weighted average of Prophet + best tree model (beyond simple champion pick)