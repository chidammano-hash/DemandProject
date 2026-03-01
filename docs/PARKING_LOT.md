# Parking Lot — Known Issues & Deferred Work

Issues captured here are real, confirmed problems that are not yet fixed.
Each entry includes root cause, impact, and a recommended fix when known.

---

## Fixed Issues

---

### PL-001 — Multiple Backtests Overwrite the Same Output CSVs

**Status:** Fixed — 2026-02-28
**Priority:** High
**Date captured:** 2026-02-28
**Fixed in:** `common/backtest_framework.py`, `scripts/load_backtest_forecasts.py`, `Makefile`
**Affects:** `run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`

#### Problem

Every backtest script (LGBM, CatBoost, XGBoost — all strategies) wrote output to the **same two fixed file paths** regardless of model or strategy:

```
mvp/demand/data/backtest/backtest_predictions.csv
mvp/demand/data/backtest/backtest_predictions_all_lags.csv
```

Running a second backtest before loading the first silently overwrote both CSVs, losing the first model's predictions permanently.

#### Fix Applied (Option B — Model-scoped subdirectory)

`common/backtest_framework.py` → `save_backtest_output()` now writes each model into its own subdirectory:

```
data/backtest/lgbm_cluster/backtest_predictions.csv
data/backtest/lgbm_cluster/backtest_predictions_all_lags.csv
data/backtest/catboost_cluster/backtest_predictions.csv
data/backtest/catboost_cluster/backtest_predictions_all_lags.csv
data/backtest/xgboost_cluster/backtest_predictions.csv
data/backtest/xgboost_cluster/backtest_predictions_all_lags.csv
```

`scripts/load_backtest_forecasts.py` was refactored to support:
- `--model MODEL_ID` — load a single model from `data/backtest/<MODEL_ID>/`
- `--all` — discover and load all models under `data/backtest/*/`
- `--input PATH` — backward-compatible explicit path

Makefile targets updated:
- `make backtest-load MODEL=lgbm_cluster` — load one model
- `make backtest-load-all` — load all available models

You can now batch multiple backtests safely:

```bash
make backtest-lgbm-cluster
make backtest-catboost-cluster
make backtest-xgboost-cluster
make backtest-load-all           # loads all three in sequence
```

---

## Open Issues

*Add new issues below using the PL-NNN format.*

