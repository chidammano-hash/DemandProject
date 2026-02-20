# Feature 14: Transfer Learning Backtest Strategy

## Objective
Add a transfer learning strategy to all three backtest frameworks (LGBM, CatBoost, XGBoost) that trains a global base model on all data, then fine-tunes per cluster. This improves predictions for small clusters that previously received zero predictions and provides better specialized models for larger clusters.

## Problem Statement
The existing `per_cluster` strategy trains independent models per cluster. Clusters with fewer than 50 training rows are skipped entirely and receive zero predictions. This wastes available signal from the broader dataset and produces poor forecasts for niche DFU segments.

## Solution: Global → Per-Cluster Transfer Learning

### Strategy Overview
1. **Phase 1 (Base Model):** Train a single model on ALL training data, **excluding `ml_cluster`** from features. This gives the model broad demand pattern knowledge without cluster-specific bias.
2. **Phase 2 (Fine-Tuning):** For each cluster with ≥ `transfer_min_rows` training rows, create a new model initialized from the base model (warm-start) and train for a small number of additional trees/iterations on that cluster's data only.
3. **Fallback:** Clusters with fewer than `transfer_min_rows` rows (default: 20) or unassigned DFUs use the base model's predictions directly — never zeroed out.

### Why Exclude `ml_cluster` from Base Model
- Per-cluster fine-tuning also excludes `ml_cluster` (it's the grouping key, not a feature within a cluster).
- Matching feature sets between base and fine-tuned models is **required** for warm-start (`init_model` / `xgb_model`) to work correctly across all three frameworks.
- The base model learns generalizable demand patterns; cluster-specific patterns emerge during fine-tuning.

## Implementation

### CLI Arguments (all three scripts)
| Argument | Default | Description |
|---|---|---|
| `--cluster-strategy transfer` | — | Selects transfer learning strategy |
| `--transfer-n-estimators` (LGBM/XGBoost) | 100 | Additional trees for fine-tuning |
| `--transfer-iterations` (CatBoost) | 100 | Additional iterations for fine-tuning |
| `--transfer-min-rows` | 20 | Minimum cluster training rows for fine-tuning |

### Model IDs
| Framework | Model ID |
|---|---|
| LightGBM | `lgbm_transfer` |
| CatBoost | `catboost_transfer` |
| XGBoost | `xgboost_transfer` |

### Warm-Start API by Framework
| Framework | Parameter | Value Passed |
|---|---|---|
| LightGBM | `init_model` in `.fit()` | `base_model.booster_` |
| CatBoost | `init_model` in `.fit()` | `base_model` (regressor directly) |
| XGBoost | `xgb_model` in `.fit()` | `base_model.get_booster()` |

All three APIs are **additive** — new trees/iterations are appended on top of the base model's existing ensemble.

### Make Targets
```bash
make backtest-lgbm-transfer      # LGBM transfer backtest
make backtest-catboost-transfer  # CatBoost transfer backtest
make backtest-xgboost-transfer   # XGBoost transfer backtest
make backtest-load               # Load predictions (shared, model-agnostic)
```

## Key Design Decisions

### Transfer Min Rows = 20 (vs per_cluster's 50)
The transfer strategy lowers the minimum from 50 to 20 because:
- The base model already provides a strong initialization — less cluster data is needed
- Fine-tuning 100 trees on 20 rows is more stable than training 500 trees from scratch on 50 rows
- Clusters between 20–49 rows that were previously zeroed now get meaningful predictions

### Same Feature Engineering
No changes to feature engineering. The transfer strategy uses the same lag, rolling, calendar, and attribute features. The only difference is feature *selection*: `ml_cluster` is excluded from both base and fine-tuned models.

### Output Format Unchanged
Both CSVs (`backtest_predictions.csv`, `backtest_predictions_all_lags.csv`) use the same schema. The shared `load_backtest_forecasts.py` loader works unchanged — it reads `model_id` from the CSV column.

## Files Modified
- `mvp/demand/scripts/run_backtest.py` — Contains `train_and_predict_transfer()` with LGBM-specific warm-start (`init_model=booster_`)
- `mvp/demand/scripts/run_backtest_catboost.py` — Contains `train_and_predict_transfer()` with CatBoost-specific warm-start (`init_model=regressor`)
- `mvp/demand/scripts/run_backtest_xgboost.py` — Contains `train_and_predict_transfer()` with XGBoost-specific warm-start (`xgb_model=booster`)
- `mvp/demand/common/backtest_framework.py` — Shared orchestrator dispatches to the transfer function via `train_fn_transfer` callable with `transfer_kwargs`
- `mvp/demand/Makefile` — Added 3 new targets (`backtest-lgbm-transfer`, `backtest-catboost-transfer`, `backtest-xgboost-transfer`)

## Verification
```bash
# Run and load transfer backtests
make backtest-lgbm-transfer && make backtest-load
make backtest-catboost-transfer && make backtest-load
make backtest-xgboost-transfer && make backtest-load

# Verify models appear in forecast selector
curl "http://localhost:8000/domains/forecast/models"

# Check accuracy comparison
curl "http://localhost:8000/forecast/accuracy/slice?group_by=cluster_assignment&models=lgbm_transfer,lgbm_global,lgbm_cluster"

# Verify archive data
docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive GROUP BY 1,2 ORDER BY 1,2"
```
