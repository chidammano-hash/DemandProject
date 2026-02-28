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

---

## Implementation Details

### Metadata Recording
- `extra_metadata` parameter passes transfer params (`transfer_n_estimators`, `transfer_min_rows`/`transfer_iterations`) to `backtest_metadata.json`

### MLflow Tags
- LGBM: `model_type_tag="lgbm_backtest"`
- CatBoost: `model_type_tag="catboost_backtest"`
- XGBoost: `model_type_tag="xgboost_backtest"`

### `cat_dtype` Difference
- CatBoost: `cat_dtype="str"` (categoricals as strings)
- LGBM/XGBoost: `cat_dtype="category"` (pandas category type)

### `__unknown__` Cluster Handling
- Explicitly filtered from fine-tuning loop: `clusters = [c for c in clusters if c != "__unknown__"]`
- Both `NaN` and `__unknown__` cluster DFUs use base model fallback

### Prediction Clipping
- All transfer functions use `np.maximum(preds, 0)` for non-negative predictions


---

## Examples

### Example: Transfer learning — two-phase training

```python
import lightgbm as lgb

# Phase 1: global base model (trained on ALL DFUs, ml_cluster excluded)
base_model = lgb.LGBMRegressor(n_estimators=500)
base_model.fit(X_global, y_global, categorical_feature=cat_features)

# Phase 2: per-cluster fine-tune using warm start (additive trees)
for cluster_id, (X_c, y_c) in cluster_data.items():
    if len(X_c) < transfer_min_rows:   # fallback if too few rows
        continue
    cluster_model = lgb.LGBMRegressor(n_estimators=100)
    cluster_model.fit(X_c, y_c,
                      categorical_feature=cat_features,
                      init_model=base_model.booster_)
    predictions[cluster_id] = cluster_model.predict(X_test_c)
```

### Example: Run all three LGBM strategies

```bash
make backtest-lgbm           # global: model_id=lgbm_global
make backtest-lgbm-cluster   # per-cluster: model_id=lgbm_cluster
make backtest-lgbm-transfer  # transfer: model_id=lgbm_transfer
make backtest-load           # load all three into Postgres

# Compare strategies
curl -s "http://localhost:8000/forecast/accuracy/slice?lag=2&dim=model_id" \
  | jq '[.rows[] | select(.model_id | startswith("lgbm")) | {model_id, accuracy_pct}]'
# [{"model_id": "lgbm_transfer", "accuracy_pct": 92.4},
#  {"model_id": "lgbm_cluster",  "accuracy_pct": 93.1},
#  {"model_id": "lgbm_global",   "accuracy_pct": 91.5}]
```


---

## Additional Examples

#### Example — Why ml_cluster is excluded from base model

```python
import lightgbm as lgb

# WRONG: include ml_cluster in base model
# Fine-tuning then also trains WITHOUT ml_cluster → feature mismatch
# → warm-start (init_model) raises ValueError: incompatible feature sets
base_wrong = lgb.LGBMRegressor()
base_wrong.fit(X_with_cluster, y)   # includes ml_cluster feature

fine_tuned = lgb.LGBMRegressor(n_estimators=100)
fine_tuned.fit(X_no_cluster, y_cluster,
               init_model=base_wrong.booster_)  # ERROR: feature count mismatch

# CORRECT: exclude ml_cluster from both base AND fine-tuned models
FEATURES_NO_CLUSTER = [f for f in ALL_FEATURES if f != "ml_cluster"]

base = lgb.LGBMRegressor(n_estimators=500)
base.fit(X_all[FEATURES_NO_CLUSTER], y_all)

fine_tuned = lgb.LGBMRegressor(n_estimators=100)
fine_tuned.fit(X_cluster[FEATURES_NO_CLUSTER], y_cluster,
               init_model=base.booster_)   # OK: same feature set
```

#### Example — Transfer min rows threshold in practice

```python
# Clusters with rows between 20-49 get predictions from transfer (not possible in per_cluster)
cluster_summary = X_train.groupby("ml_cluster").size().reset_index(name="rows")
# cluster       | rows | per_cluster action  | transfer action
# high_vol_st.  | 8240 | model trained (>=50)| fine-tuned (>=20)
# seasonal_med  | 3120 | model trained       | fine-tuned
# niche_export  |   31 | model trained       | fine-tuned (31 >= 20)
# micro_test    |   12 | SKIPPED → 0 preds   | base model fallback
# __unknown__   |   87 | model trained       | excluded, base fallback

# Transfer ensures micro_test DFUs get meaningful predictions (not zeros)
```

#### Example — __unknown__ cluster handling in transfer loop

```python
# From run_backtest.py train_and_predict_transfer()
clusters = X_train["ml_cluster"].dropna().unique().tolist()
clusters = [c for c in clusters if c != "__unknown__"]  # exclude unassigned DFUs

for cluster_id in clusters:
    X_c = X_train[X_train["ml_cluster"] == cluster_id]
    if len(X_c) < transfer_min_rows:      # default: 20
        # Use base model predictions for this cluster
        preds_cluster = np.maximum(base_model.predict(X_test_c), 0)
    else:
        fine_tuned = lgb.LGBMRegressor(n_estimators=transfer_n_estimators)
        fine_tuned.fit(X_c, y_c, categorical_feature=cat_features,
                       init_model=base_model.booster_)
        preds_cluster = np.maximum(fine_tuned.predict(X_test_c), 0)

# DFUs with ml_cluster=NaN or '__unknown__' use base_model predictions directly
mask_fallback = X_test["ml_cluster"].isna() | (X_test["ml_cluster"] == "__unknown__")
preds[mask_fallback] = np.maximum(base_model.predict(X_test[mask_fallback]), 0)
```

#### Example — Verify transfer coverage (no zeroed DFUs)

```sql
-- Confirm transfer model has no DFU-months with zero/null predictions
-- (per_cluster would have gaps for small clusters)
SELECT model_id,
       COUNT(*) FILTER (WHERE basefcst_pref = 0 OR basefcst_pref IS NULL) AS zero_preds,
       COUNT(*)                                                             AS total_rows
FROM fact_external_forecast_monthly
WHERE model_id IN ('lgbm_cluster', 'lgbm_transfer')
GROUP BY model_id;
-- model_id       | zero_preds | total_rows
-- lgbm_cluster   |       4320 |      91301   (small clusters zeroed out)
-- lgbm_transfer  |          0 |      95621   (full coverage — no zeros)
```
