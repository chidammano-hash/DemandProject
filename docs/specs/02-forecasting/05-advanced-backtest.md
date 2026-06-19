# Advanced Backtest Capabilities

> Three extensions to the backtest engine -- hyperparameter tuning, SHAP feature selection, and recursive forecasting -- each improving accuracy or interpretability.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy (SHAP panel) |
| **Key Files** | `scripts/tune_hyperparams.py`, `common/ml/tuning.py`, `common/ml/shap_selector.py`, `common/ml/feature_engineering.py`, `api/routers/forecasting/shap.py`, `config/forecasting/hyperparameter_tuning.yaml` |

---

## Problem

Default hyperparameters are rarely optimal. With ~40 engineered features, many add noise rather than signal. And the standard backtest predicts all future months at once using zero-valued lags, which misrepresents how a model would actually perform in production where it feeds its own predictions forward.

## Solution

Three composable capabilities address these gaps: (1) Bayesian hyperparameter tuning finds optimal model parameters, (2) multi-stage feature selection removes redundant and noise features, and (3) recursive multi-step forecasting simulates real deployment conditions. All three are activated via config keys in `forecast_pipeline_config.yaml` -- no code changes required.

---

## 1. Hyperparameter Tuning (Optuna)

### What It Does

Replaces hardcoded default parameters (like `n_estimators=500`, `learning_rate=0.05`) with data-driven optimal values found via Bayesian search.

### Why Walk-Forward CV

Standard k-fold cross-validation creates data leakage in time series: training fold rows can appear after validation rows in time, contaminating lag features. The tuner uses expanding month-based folds with a 1-month gap to prevent this.

### Two Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Global tuning | `make tune-lgbm` | Tune once on full history, apply to production forecasts |
| Per-timeframe inline | Set `tune_inline: true` in config | Honest backtesting -- each timeframe tunes on only its available data |

Global tuning is faster (50 trials, ~20-40 min). Inline tuning is slower (~600 model fits) but produces genuine out-of-sample accuracy with no future leakage.

### Key Design Decisions

- `n_estimators` is NOT in the search space -- early stopping finds the optimal tree count automatically
- WAPE denominator uses a floor of 1.0 to prevent instability on low-demand folds
- `mask_future_sales()` is called inside every CV fold to prevent lag feature leakage
- Pruner waits for 15 complete trials before comparing (demand WAPE has high fold-to-fold variance)

### Output

`data/tuning/best_params_<model>.json` containing optimal parameters, per-cluster WAPEs, and `best_n_estimators`. Applied via `params_file` key in `forecast_pipeline_config.yaml`.

---

## 2. Multi-Stage Feature Selection

### What It Does

For each backtest timeframe, runs a 4-stage pipeline to remove redundant, low-information, and noise features before retraining on the reduced set. All stages are per-timeframe (causal) — they only see training data up to the cutoff.

### Why Per-Timeframe

Feature importance and correlation structure shift over time. A single static feature list is suboptimal across 10 expanding timeframes.

### How It Works

1. **Stage 0 — Duplicate removal**: Drop exact-duplicate aliases (e.g., `sparsity_score` → keep `zero_demand_pct`)
2. **Stage 1 — Variance filter**: Drop features with near-zero relative variance (configurable threshold)
3. **Stage 2 — Correlation filter**: For pairs with |r| > 0.95, drop the lower-variance member
4. **Stage 3 — SHAP selection**: Compute SHAP values, keep features covering 95% of cumulative mass
5. Retrain final model on selected features only
6. Write per-timeframe and summary CSVs to `data/backtest/<model_id>/shap/`

See [spec 28 — Feature Selection Pipeline](28-feature-selection-pipeline.md) for full details.

### SHAP API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/forecast/shap/models` | List models with SHAP outputs |
| GET | `/forecast/shap/{model_id}/summary` | Cross-timeframe feature importance |
| GET | `/forecast/shap/{model_id}/timeframes` | Available timeframes with labels |
| GET | `/forecast/shap/{model_id}/timeframe/{idx}` | Per-timeframe feature detail |
| GET | `/forecast/shap/{model_id}/sku` | Per-DFU signed SHAP values for Item Analysis tab |

The SHAP panel in the Accuracy tab shows a horizontal bar chart with indigo bars for selected features and gray bars for dropped features.

### Algorithm-Specific SHAP

- LightGBM/XGBoost: `shap.TreeExplainer` (requires `shap>=0.43.0`)
- CatBoost: native `get_feature_importance(type="ShapValues")` -- no `shap` library needed

---

## 3. Recursive Multi-Step Forecasting

### What It Does

In the default (direct) mode, all future months are predicted at once from the same zero-valued lag baseline. In recursive mode, each month is predicted one at a time, and the model's prediction is written back as `qty_lag_1` for the next month.

### Why It Matters

`qty_lag_1` consistently ranks as the #1 feature by SHAP importance. In direct mode, months 2+ see `qty_lag_1 = 0` (masked) -- a poor signal. In recursive mode, months 2+ see `qty_lag_1 = model's own prior prediction` -- a richer signal, even though it carries prediction error.

### Trade-offs

| Aspect | Direct Mode | Recursive Mode |
|--------|-------------|----------------|
| `qty_lag_1` for month T+2 | 0 (masked) | prediction for T+1 |
| Near-horizon realism | Lower | Higher |
| Error compounding | None | Grows with horizon |
| Inference cost | 1 batch call | N sequential calls |
| Best for | Aggregate benchmarking | Simulating real deployment |

### Compute-Side Only

No API, frontend, or database changes. The `"recursive": true` flag in `backtest_metadata.json` provides traceability.

---

## Pipeline

| Target | Description |
|--------|-------------|
| `make tune-lgbm` | Tune LightGBM (50 trials, ~20-40 min) |
| `make tune-catboost` | Tune CatBoost (~30-60 min) |
| `make tune-xgboost` | Tune XGBoost (~25-50 min) |
| `make tune-all` | Tune all three sequentially |

All three capabilities are activated via `config/forecasting/forecast_pipeline_config.yaml`:

```yaml
lgbm:
  shap_select: true      # Enable SHAP feature selection
  tune_inline: true       # Enable per-timeframe tuning
  recursive: true         # Enable recursive inference
  params_file: data/tuning/best_params_lgbm.json  # Use pre-tuned params
```

## Configuration

### Tuning Config: `config/forecasting/hyperparameter_tuning.yaml`

Controls search spaces, CV settings, and trial budgets for all three algorithms.

### Algorithm Config: `config/forecasting/forecast_pipeline_config.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `shap_select` | false | Enable SHAP-based feature selection |
| `shap_threshold` | 0.95 | Cumulative SHAP mass threshold |
| `shap_top_n` | null | Exact top-N features (overrides threshold) |
| `shap_sample_size` | 500 | Rows sampled for SHAP computation |
| `tune_inline` | false | Per-timeframe causal tuning |
| `recursive` | false | Recursive multi-step inference |
| `params_file` | null | Path to pre-tuned params JSON |

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- orchestrator that these capabilities extend
- [Tree Models](./04-tree-models.md) -- the algorithms being tuned/selected/recursed
- [Algorithm Config](./06-algorithm-config.md) -- config file reference
- Python packages: `optuna>=3.0`, `shap>=0.43.0`

## See Also

- [Algorithm Config](./06-algorithm-config.md) -- how to enable these features via YAML
- [Accuracy KPIs](./01-accuracy-kpis.md) -- where SHAP panel lives in the UI
