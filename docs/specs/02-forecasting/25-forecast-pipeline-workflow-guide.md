# ML Forecast Pipeline — Workflow Guide

## Overview

The forecast pipeline turns raw sales history into production demand forecasts through 7 stages:

```
DATA → CLUSTERING → TUNING (optional) → BACKTESTING → LOAD → CHAMPION SELECTION → PRODUCTION FORECAST
```

Each stage builds on the previous. Skip stages at your own risk — the pipeline validates prerequisites.

---

## Prerequisites

Before running the pipeline, ensure data is loaded:

```bash
make normalize-all     # Clean input CSVs (~5 min)
make load-all          # Load into Postgres (~10 min)
make refresh-agg       # Refresh materialized views (~2 min)
```

Verify data exists:
```bash
make health            # DB row counts + API health check
```

---

## Stage 1: Clustering

**Purpose:** Segment items into demand-pattern clusters (high-volume steady, intermittent, seasonal, etc.) so tree models can train per-cluster for better accuracy.

**Config:** Settings → Forecasting → Forecast Pipeline → "SKU Clustering" toggle

**Run:**
```bash
make cluster-all       # ~20 min (features → train → label → update DB)
```

**What it does:**
1. Engineers 14 demand features per item (volume, trend, seasonality, intermittency)
2. Runs K-Means with k in [9, 18], picks best k via silhouette score
3. Labels clusters with business names (e.g., "high_volume_stable")
4. Writes cluster assignments to `dim_sku.ml_cluster`

**Output:** Each item gets a cluster assignment. Tree models use this for per-cluster training.

**Skip if:** You want global models only (set clustering to OFF in Settings).

---

## Stage 2: Backtesting

**Purpose:** Evaluate model accuracy on historical data using expanding-window validation. This is how we know which model is best for each item.

**Config:** Settings → Forecasting → Forecast Pipeline → "Active Models" section

**Run all enabled models:**
```bash
make backtest-all              # Sequential (~2-6 hours depending on models)
```

**Run specific models:**
```bash
# Tree models (~30 min each, run on CPU)
make backtest-lgbm
make backtest-catboost
make backtest-xgboost

# Foundation models (need GPU/MPS, ~10-60 min each)
make backtest-bolt             # Chronos Bolt (~12 min on MPS)
make backtest-chronos2         # Chronos 2 (~5.5 hours)

# Statistical/DL models
make backtest-mstl             # MSTL (~5 min)
make backtest-nbeats           # N-BEATS (~20 min)
make backtest-nhits            # N-HiTS (~20 min)
```

**What it does per model:**
1. Creates 10 expanding time windows (each adds ~1 month of training data)
2. Trains the model on each window, predicts the next 6 months
3. Compares predictions to actuals, computes WAPE/accuracy per item per month
4. Saves predictions to `data/backtest/<model_id>/backtest_predictions.csv`

**After backtesting, load results into the database:**
```bash
make backtest-load-all-bulk    # Bulk load all models (~15 min)
```

Or load specific models:
```bash
make backtest-load-bulk        # 4 core models (lgbm, catboost, xgboost, chronos)
```

---

## Stage 3: Hyperparameter Tuning (Optional)

**Purpose:** Find better model parameters using Bayesian optimization (Optuna). Improves accuracy by 1-5% typically.

**When to run:** After initial backtesting, before champion selection. Not needed for every run.

**Run:**
```bash
make tune-lgbm         # ~1-2 hours (50 Optuna trials, 5-fold CV)
make tune-catboost     # ~1-2 hours
make tune-xgboost     # ~1-2 hours
make tune-all          # All three sequentially

# Per-cluster tuning (runs Optuna independently per ml_cluster)
make tune-lgbm-clusters   # Per-cluster LGBM tuning
make tune-clusters        # Per-cluster tuning for all tree models
```

**What it does:**
1. Runs 50 Bayesian optimization trials per model
2. Each trial trains on 5 walk-forward CV folds
3. Optuna learns which parameter regions minimize WAPE
4. Best parameters saved to `data/tuning/best_params_<model>.json`

**Per-cluster tuning** (`make tune-lgbm-clusters`) runs Optuna independently per `ml_cluster`, writing cluster-specific overrides to `config/forecasting/cluster_tuning_profiles.yaml` with `cluster_name` in `match_criteria`. During backtest, profiles are matched by Phase 1 (exact cluster_name) then Phase 2 (statistical criteria fallback).

**After tuning, promote and re-backtest:**
```bash
# Option A: CLI promotion
make lgbm-auto-tune-promote    # Write best params to forecast_pipeline_config.yaml

# Option B: UI promotion (recommended)
# Go to Model Tuning → Backtest & Tune → select model → Experiments → Promote
```

Then re-run backtesting with the tuned parameters:
```bash
make backtest-lgbm && make backtest-load MODEL=lgbm_cluster
```

**UI Alternative:** Model Tuning tab → Backtest & Tune → click model → "New Experiment" button. The UI lets you create tuning experiments, compare results, and promote winners.

---

## Stage 4: Champion Selection

**Purpose:** For each item-month, pick the best-performing model from the competing set based on historical backtest accuracy.

**Config:** Settings → Forecasting → Forecast Pipeline → "Champion Selection" section

**Run:**
```bash
make champion-all      # ~10 min (meta-learner + simulate + select)
```

**What it does:**
1. **Train meta-learner:** RF classifier that predicts which model is best per item
2. **Simulate strategies:** Evaluates all champion strategies, computes oracle ceiling
3. **Select champions:** Applies the configured strategy, writes `model_id='champion'` rows

**Strategy options** (configured in Settings or via UI experiments):
| Strategy | Description | Best for |
|----------|-------------|----------|
| Expanding Window | Uses all available history | Stable portfolios |
| Rolling Window | Uses last N months only | Changing demand patterns |
| Adaptive Ensemble | Blends top-K models | General purpose |
| Per-Cluster | Best model per cluster | Heterogeneous portfolios |
| Hybrid Warmup | Rolling for new items, expanding for mature | Mixed maturity |

**UI Alternative:** Model Tuning tab → Champion → "New Experiment" button. Test different strategies, compare accuracy, promote the winner.

---

## Stage 5: Production Forecast

**Purpose:** Generate forward-looking demand predictions for the next 24 months using champion model assignments. Predictions go through a staged promotion workflow before becoming the official production forecast.

**Config:** Settings → Forecasting → Forecast Pipeline → "Forecast Settings" section

**Staged workflow:**
```
Train → Generate → Load (→ fact_candidate_forecast) → Promote (→ fact_production_forecast)
```

**Run:**
```bash
make forecast-generate    # ~30 min (all items, 24-month horizon → fact_candidate_forecast)
```

**What it does:**
1. For each item: loads the champion model's `.pkl` artifact
2. Builds feature matrix for future months (T+1 through T+24)
3. Generates recursive predictions (T+2 uses T+1's predicted value as input)
4. Optionally computes P10/P90 confidence intervals
5. Writes predictions to `fact_candidate_forecast` (staging table)

**Promotion to production:**
After reviewing candidate forecasts, promote them to `fact_production_forecast`:
- **Champion promotion:** Uses per-DFU champion assignments to pick the best model's predictions for each DFU from the candidate pool, then copies those rows to `fact_production_forecast`.
- **Single model promotion:** Copies all candidate rows for one specified model to `fact_production_forecast`.
- **Audit trail:** Every promotion event is logged in `model_promotion_log` with promotion type, model(s), row counts, and timestamp.

**Cold-start routing:**
- Items with < 12 months history → routed to Rolling Mean model
- Items with < 3 months history → skipped entirely

---

## Stage 6: Experimentation (UI-Driven)

The UI provides three experimentation studios accessible from the **Model Tuning** tab:

### Clustering Experiments
- Tab: Model Tuning → Clustering
- Test different k values, feature sets, scaling methods
- Compare silhouette scores and cluster distributions
- Promote winning config to production

### Tuning Experiments
- Tab: Model Tuning → Backtest & Tune → click a model → "New Experiment"
- Choose a template (Current Production, Conservative, High Precision, etc.)
- Compare accuracy against production baseline
- Promote best parameters

### Champion Experiments
- Tab: Model Tuning → Champion → "New Experiment"
- Test different selection strategies (Expanding, Rolling, Ensemble, etc.)
- Compare accuracy, model distribution, gap to oracle ceiling
- Promote winning strategy to production config

### Experiment Lifecycle
```
Create → Queued → Running → Completed → [Compare] → [Promote]
                     ↓
                   Failed (check logs)
```

All experiments run as background jobs. Monitor progress in the **Jobs** tab.

---

## Quick Reference: Full Pipeline Commands

### First-time setup (from scratch):
```bash
make normalize-all                  # Clean CSVs
make load-all                       # Load to Postgres
make cluster-all                    # Segment items (if clustering enabled)
make backtest-lgbm backtest-catboost backtest-xgboost  # Core tree models
make backtest-load-bulk             # Load results to DB
make champion-all                   # Select best models
make forecast-generate              # Generate candidates → fact_candidate_forecast
                                    # Then promote → fact_production_forecast
```

### Monthly refresh:
```bash
make pipeline-refresh               # Incremental data reload
make backtest-lgbm backtest-catboost backtest-xgboost
make backtest-load-bulk
make champion-select
make forecast-generate              # Generate candidates → promote to production
```

### After tuning improvements:
```bash
make tune-lgbm                      # Tune
make lgbm-auto-tune-promote         # Promote params
make backtest-lgbm                  # Re-backtest with new params
make backtest-load MODEL=lgbm_cluster
make champion-select                # Re-select with updated predictions
make forecast-generate              # Re-generate candidates → promote to production
```

---

## Configuration Quick Reference

All pipeline configuration is in **Settings → Forecasting → Forecast Pipeline**:

| Setting | Default | Description |
|---------|---------|-------------|
| Forecast Horizon | 24 months | How far ahead to forecast |
| Min History Required | 12 months | Below this → cold-start model |
| New Product Model | Rolling Mean | Model for items with < 12 months history |
| New Product Min Data | 3 months | Below this → no forecast |
| Model Selection Strategy | Expanding | How champion is chosen |
| Evaluation Metric | Accuracy % | WAPE or accuracy_pct |
| Default Model | Seasonal Naive | Fallback when no champion |
| SKU Clustering | ON/OFF | Segment items by demand pattern |
| Backtest Windows | 10 | Number of expanding evaluation windows |
| Backtest Horizon | 6 months | Months predicted per window |
| Fast Sampling | ON/OFF | Sample subset for quick iteration |
| Sample Size | 5,000 items | How many items when sampling enabled |

Individual model enable/disable toggles are in the same Settings page under **Active Models**.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Champion experiments fail immediately | No backtest data loaded | Run `make backtest-load-all-bulk` first |
| "Repository not found" for Chronos | HuggingFace auth or wrong model size | Run `huggingface-cli login`; check `model_size` in config |
| Backtest accuracy is 0% | No sales data for the period | Check `make health`; verify `fact_sales_monthly` has data |
| Production forecast all NaN | No trained `.pkl` models | Run backtest first — models are saved during backtest |
| Tuning converges in < 10 trials | Search space too narrow | Widen ranges in `config/forecasting/hyperparameter_tuning.yaml` |
| Cold-start items get no forecast | < 3 months of history | Lower `cold_start_min_months` in Settings (min 1) |
| Clustering produces 1 cluster | Insufficient data diversity | Lower `min_months_history` or increase sample size |
