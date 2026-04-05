# Dual Promotion — Parameters + Results

> Adds a second "Promote Results" action to the Model Tuning Studio so backtest predictions from a tuning experiment are loaded into PostgreSQL, enabling accuracy screens and champion selection to compare the tuned model against others.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Model Tuning |
| **Key Files** | `sql/100_results_promotion.sql`, `common/services/job_state.py`, `api/routers/forecasting/unified_model_tuning.py`, `frontend/src/tabs/model-tuning/EnhancedPromoteModal.tsx` |

---

**Depends on:** Feature 46 (Unified Model Tuning Studio)

## Problem

The Model Tuning Studio has a single "Promote" action that writes hyperparameters to `forecast_pipeline_config.yaml`. But backtest predictions from tuning experiments are never loaded into PostgreSQL, so:

1. UI accuracy screens (Aggregate Analysis, Item Analysis, accuracy portfolio) cannot see the tuned model's accuracy — they query `fact_external_forecast_monthly`
2. Champion selection (`scripts/run_champion_selection.py`) cannot compare the tuned model vs others — it reads from `fact_external_forecast_monthly`
3. `backtest_lag_archive` doesn't have the new predictions for lag-curve analysis

## Solution

Two independent promotion actions:

### Promote Parameters (existing)
- Writes hyperparameters to `config/forecast_pipeline_config.yaml`
- Future `make forecast-generate` uses the promoted params
- Tracked: `is_promoted`, `promoted_at` on `lgbm_tuning_run`

### Promote Results (new)
- Loads `backtest_predictions.csv` into `fact_external_forecast_monthly` (replaces old rows for this model_id)
- Loads `backtest_predictions_all_lags.csv` into `backtest_lag_archive`
- Refreshes 5 materialized views: `agg_forecast_monthly`, `agg_accuracy_by_dim`, `agg_dfu_coverage`, `agg_accuracy_lag_archive`, `agg_dfu_coverage_lag_archive`
- Async job (5-15 min for ~2.7M + ~13.5M rows)
- Tracked: `is_results_promoted`, `results_promoted_at` on `lgbm_tuning_run`

## API

### POST `/{model}/experiments/{run_id}/promote-results`
Submits async job to load predictions into DB.

**Response:** `{ job_id, run_id, model, message }`

### GET `/{model}/experiments/{run_id}/promote-results/status`
Polls job progress.

**Response:** `{ status, progress_pct, progress_msg, error }`

## Schema

```sql
ALTER TABLE lgbm_tuning_run
    ADD COLUMN is_results_promoted BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN results_promoted_at TIMESTAMPTZ,
    ADD COLUMN results_promote_job_id VARCHAR(100);

ALTER TABLE tuning_promotion_log
    ADD COLUMN promotion_type VARCHAR(20) NOT NULL DEFAULT 'params';
```

## UI

The promote modal shows two cards:
1. **Promote Parameters** — existing param preview + promote button
2. **Promote Results** — progress bar during load, "Already loaded" badge when done

Experiment table shows dual badges: crown (params) + database icon (results).

## Key Files

| File | Role |
|------|------|
| `sql/100_results_promotion.sql` | Schema migration |
| `common/services/job_state.py` | `_run_load_backtest_results` job function |
| `common/services/job_registry.py` | Job type registration |
| `api/routers/forecasting/unified_model_tuning.py` | 2 new endpoints |
| `frontend/src/tabs/model-tuning/EnhancedPromoteModal.tsx` | Dual-action modal |
| `frontend/src/api/queries/unified-model-tuning.ts` | Types + fetchers |
