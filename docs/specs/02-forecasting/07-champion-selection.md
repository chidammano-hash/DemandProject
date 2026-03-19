# Champion Model Selection

> Automatically picks the best-performing model for each product each month, creating a composite "champion" forecast that outperforms any single algorithm.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `scripts/run_champion_selection.py`, `common/champion_strategies.py`, `config/model_competition.yaml`, `api/routers/competition.py`, `frontend/src/tabs/AccuracyTab.tsx` |

---

## Problem

With three competing models (LightGBM, CatBoost, XGBoost), no single algorithm wins for every product every month. Seasonal items might favor CatBoost in peak months and LightGBM in troughs. Without automated selection, planners must either pick one model for everything (suboptimal) or manually choose per item (impractical at scale with thousands of DFUs).

## Solution

Champion selection evaluates prior model performance for each DFU (Demand Forecast Unit) each month and picks the winner. It simulates what a planner would do: look at recent accuracy, pick the best model, and use its forecast going forward. The selection is causally correct -- it only uses information that was available at the time the forecast was issued. A ceiling (oracle) model provides a theoretical upper bound using after-the-fact perfect knowledge, quantifying the gap between the rolling selection and what's theoretically possible.

## How It Works

1. Load per-model, per-DFU, per-month errors from the backtest archive
2. For each DFU-month, compute prior WAPE (Weighted Absolute Percentage Error) per model using only causally available data
3. Pick the model with the lowest prior WAPE -- this becomes the champion for that month
4. Fill warm-up months (where there is insufficient history) with a fallback model
5. Store champion predictions as `model_id = 'champion'` in the forecast table
6. Separately compute the ceiling: pick the best model per DFU-month using that month's actual error (after-the-fact oracle)
7. Store ceiling as `model_id = 'ceiling'` -- the gap between champion and ceiling measures improvement opportunity

### Execution-Lag Causality (Critical)

Each DFU has an `execution_lag` -- how many months in advance its forecast is issued. For a DFU with `execution_lag = 1`, the April forecast is issued in March. At issuance time, March actuals are NOT available yet.

**The fix:** For each DFU-model group, apply `shift(execution_lag + 1)` before any cumulative or rolling computation. This ensures month T's champion uses only months where `startdate < fcstdate` (= T minus execution_lag).

**Example: Picking April 2025's champion (execution_lag = 1)**

The April forecast is issued in March 2025. Available actuals: January and February only (March is not available yet).

| Month | Model A Error | Model B Error |
|-------|-------------|-------------|
| Jan 2025 | 10 | 5 |
| Feb 2025 | 15 | 10 |
| Mar 2025 | 5 | 15 |
| **Apr 2025** | 8 | 2 |

Prior WAPE for April (using only Jan + Feb):
- Model A: (10+15) / 200 = 12.5%
- Model B: (5+10) / 200 = 7.5%

Winner: Model B. Its April forecast (98 units) becomes the champion row.

With `execution_lag = 0`, the formula degrades to `shift(1)` -- fully backward compatible.

### Warm-Up Period

With `execution_lag = 1` and `min_dfu_rows = 3`, the first qualifying month requires 3 non-null prior observations after shifting. Earlier months get the fallback model (default: `lgbm_cluster`), ensuring every DFU-month has a champion row.

## 5 Selection Strategies

| Strategy | Key Idea | Best For |
|----------|----------|----------|
| `expanding` | Cumulative WAPE from all prior months (equal weight) | Stable demand, long history |
| `rolling` | Last N months only (default N=6) | Volatile demand, regime changes |
| `decay` | Exponential weighting (recent months count more, factor=0.9) | Rewarding recent improvement |
| `ensemble` | Blend top-K models weighted by inverse WAPE | No single model dominates |
| `meta_learner` | ML classifier predicts best model from DFU features + performance stats | Rich feature data available |

All strategies use execution-lag-aware shifting internally.

The meta-learner uses ceiling (oracle) labels as ground truth with a strict temporal train/test split. Trained via `make champion-train-meta`, saved to `data/champion/meta_learner.joblib`.

## Data Model

No new tables. Champion and ceiling predictions are stored in the existing `fact_external_forecast_monthly` with `model_id = 'champion'` and `model_id = 'ceiling'`. All accuracy views automatically include them.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/competition/config` | Current config + available model_ids from DB |
| PUT | `/competition/config` | Update config (writes YAML to disk) |
| POST | `/competition/run` | Execute champion selection, return summary |
| GET | `/competition/summary` | Last run summary |

## Pipeline

| Target | Description |
|--------|-------------|
| `make champion-select` | Run champion selection using configured strategy |
| `make champion-simulate` | Simulate all 5 strategies, compare accuracy vs. ceiling |
| `make champion-train-meta` | Train meta-learner classifier |
| `make champion-all` | train-meta + simulate + select (full pipeline) |

## Configuration

### `config/model_competition.yaml`

```yaml
competition:
  metric: "wape"                     # wape (lowest wins) or accuracy_pct
  lag: "execution"                   # "execution" (per-DFU) or fixed 0-4
  min_dfu_rows: 3                    # Minimum prior months before selection
  fallback_model_id: "lgbm_cluster"  # Used for warm-up gaps
  strategy: expanding                # expanding | rolling | decay | ensemble | meta_learner
  strategy_params:
    window_months: 6                 # For rolling strategy
    decay_factor: 0.90               # For decay strategy
    top_k: 3                         # For ensemble strategy
  models:
    - lgbm_cluster
    - catboost_cluster
    - xgboost_cluster
```

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- provides backtest predictions to compare
- [Multi-Model Support](./02-multi-model.md) -- `model_id` column stores champion/ceiling rows
- [Accuracy KPIs](./01-accuracy-kpis.md) -- WAPE formula used for selection
- Python packages: `scikit-learn>=1.3` (meta-learner), `joblib>=1.3`

## See Also

- [Production Forecast](./08-production-forecast.md) -- uses champion assignments to route inference
- [Algorithm Config](./06-algorithm-config.md) -- controls which models compete
