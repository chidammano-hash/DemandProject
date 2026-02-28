# Feature 23: Backtest Model Cleanup Utility

## Objective

Provide a CLI utility to selectively remove backtest model predictions from PostgreSQL and refresh all dependent materialized views. This enables operators to clean up obsolete or experimental model runs without manual SQL or full database rebuilds.

## Motivation

- After iterating on multiple backtest models (LGBM, CatBoost, XGBoost, Prophet, PatchTST, DeepAR × global/cluster/transfer), the database accumulates predictions that may no longer be needed.
- Stale model predictions inflate the model selector dropdown, skew accuracy comparisons, and consume disk space.
- Manual `DELETE` + `REFRESH MATERIALIZED VIEW` is error-prone and requires knowledge of all dependent views.
- A purpose-built utility provides safe, auditable cleanup with preview (dry-run) capability.

## Scope

### In Scope
- CLI script to delete model predictions by `model_id` from both `fact_external_forecast_monthly` and `backtest_lag_archive`
- Automatic refresh of all dependent materialized views after deletion
- List mode to show row counts per `model_id` across both tables
- Dry-run mode to preview deletions without executing
- Bulk cleanup of all non-external (backtest) models
- Makefile integration (`make backtest-clean`, `make backtest-list`)

### Out of Scope
- UI-based cleanup (future enhancement)
- Automatic cleanup scheduling / retention policies
- Cleanup of MLflow experiment runs (separate concern)
- Cleanup of CSV output files on disk

## Design

### Script: `scripts/clean_backtest_models.py`

**CLI Interface:**
```bash
# List row counts per model_id in both tables
uv run python scripts/clean_backtest_models.py --list

# Preview what would be deleted (no actual deletion)
uv run python scripts/clean_backtest_models.py --dry-run lgbm_global deepar_global

# Delete specific models
uv run python scripts/clean_backtest_models.py lgbm_global deepar_global

# Delete ALL non-external models (everything except model_id='external')
uv run python scripts/clean_backtest_models.py --all-backtest
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `models` | positional, `nargs=*` | Model IDs to remove (e.g., `lgbm_global deepar_global`) |
| `--all-backtest` | flag | Remove ALL non-external model predictions |
| `--list` | flag | List model_id row counts and exit |
| `--dry-run` | flag | Preview deletions without executing |

### Tables Affected

1. **`fact_external_forecast_monthly`** — main forecast table; rows deleted by `model_id`
2. **`backtest_lag_archive`** — all-lags archive table; rows deleted by `model_id`

### Materialized Views Refreshed

After deletion, the following views are refreshed to reflect the updated data:

1. `agg_forecast_monthly` — forecast aggregates for trend charts
2. `agg_accuracy_by_dim` — accuracy slicing by DFU attributes
3. `agg_dfu_coverage` — DFU coverage statistics
4. `agg_accuracy_lag_archive` — lag-curve accuracy from archive
5. `agg_dfu_coverage_lag_archive` — DFU coverage from archive

Views are refreshed sequentially with error handling — if a view doesn't exist (e.g., optional views), it is skipped with a warning.

### Safety Features

- **Dry-run mode**: `--dry-run` shows exactly what would be deleted (row counts per model per table) without executing any DELETE statements
- **List mode**: `--list` provides an inventory of all model_ids and their row counts before deciding what to clean
- **Per-model counting**: Row counts are displayed before deletion so operators can verify scope
- **Transactional commits**: Each model's deletion is committed individually, so partial failures don't roll back already-cleaned models
- **No cascade**: Only deletes from the two known tables; does not affect dimension tables, sales data, or external forecasts
- **External protection**: `--all-backtest` explicitly excludes `model_id='external'` to protect source-system forecasts

### Makefile Targets

```makefile
backtest-clean:                    ## Remove model predictions: make backtest-clean MODELS="lgbm_global deepar_global"
	$(UV) python scripts/clean_backtest_models.py $(MODELS)

backtest-list:                     ## List model_id row counts in forecast + archive tables
	$(UV) python scripts/clean_backtest_models.py --list
```

## Data Flow

```
Operator runs: make backtest-clean MODELS="lgbm_global deepar_global"
                                    │
                                    ▼
                    clean_backtest_models.py
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            Count rows        Count rows       (if --dry-run)
            in forecast       in archive       → print & exit
                    │               │
                    ▼               ▼
            DELETE WHERE      DELETE WHERE
            model_id = X      model_id = X
                    │               │
                    └───────┬───────┘
                            ▼
                  REFRESH MATERIALIZED VIEW
                    (5 views sequentially)
                            │
                            ▼
                      Done — summary
```

## Example Output

### List mode
```
── fact_external_forecast_monthly ──
  catboost_global                   1,234,567 rows
  external                          5,678,901 rows
  lgbm_global                       1,234,567 rows

── backtest_lag_archive ──
  catboost_global                   6,172,835 rows
  lgbm_global                       6,172,835 rows
```

### Dry-run mode
```
[14:30:15] Models to clean: lgbm_global, deepar_global
[14:30:15] DRY RUN — no rows will be deleted

  lgbm_global: would delete 1,234,567 forecast + 6,172,835 archive rows
  deepar_global: would delete 987,654 forecast + 4,938,270 archive rows

[14:30:16] DRY RUN total: 2,222,221 forecast + 11,111,105 archive rows
```

### Delete mode
```
[14:30:15] Models to clean: lgbm_global, deepar_global
  lgbm_global: deleted 1,234,567 forecast + 6,172,835 archive rows (2.3s)
  deepar_global: deleted 987,654 forecast + 4,938,270 archive rows (1.8s)

[14:30:19] Deleted total: 2,222,221 forecast + 11,111,105 archive rows
[14:30:19] Refreshing materialized views...
  agg_forecast_monthly (1.2s)
  agg_accuracy_by_dim (3.4s)
  agg_dfu_coverage (0.8s)
  agg_accuracy_lag_archive (2.1s)
  agg_dfu_coverage_lag_archive (0.9s)

[14:30:27] Done.
```

## Dependencies

- `psycopg` — PostgreSQL driver (already in project dependencies)
- `python-dotenv` — environment variable loading (already in project dependencies)
- No new dependencies required

## Valid Model IDs

The following model_ids may exist in the database and can be cleaned:

| Framework | Model IDs |
|-----------|-----------|
| External | `external` (protected by `--all-backtest`) |
| LGBM | `lgbm_global`, `lgbm_cluster`, `lgbm_transfer` |
| CatBoost | `catboost_global`, `catboost_cluster`, `catboost_transfer` |
| XGBoost | `xgboost_global`, `xgboost_cluster`, `xgboost_transfer` |
| Prophet | `prophet_global`, `prophet_cluster`, `prophet_pooled` |
| PatchTST | `patchtst_global`, `patchtst_cluster`, `patchtst_transfer` |
| DeepAR | `deepar_global`, `deepar_cluster`, `deepar_transfer` |
| Champion | `champion` (generated by `make champion-select`) |
| Ceiling | `ceiling` (generated by `make champion-select`) |

## Related Features

- **Feature 8** — Backtesting framework (generates the predictions this utility cleans)
- **Feature 10** — Accuracy slicing (materialized views refreshed by this utility)
- **Feature 15** — Champion selection (champion/ceiling rows can be cleaned)

---

## Implementation Details

### Additional Valid Model IDs (not in original table)
- `neuralprophet_global`, `neuralprophet_cluster`, `neuralprophet_pooled`
- `statsforecast_global`, `statsforecast_cluster`, `statsforecast_pooled`

### Status
Fully implemented.
