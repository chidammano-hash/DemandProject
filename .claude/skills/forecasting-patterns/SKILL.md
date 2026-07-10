---
name: forecasting-patterns
description: DemandProject forecasting-engine patterns and pitfalls — backtest/champion/production lifecycle, per-cluster training, leakage guards, cold-start/intermittent routing, and the accuracy formulas.
---

# Forecasting Patterns (DemandProject)

Domain knowledge for the ML forecasting subsystem (`common/ml/`, `scripts/ml/`,
`scripts/forecasting/`, `api/routers/forecasting/`). Loads on forecasting work.
Full design surface: `docs/specs/02-forecasting/`. Master config:
`config/forecasting/forecast_pipeline_config.yaml`.

## When to Activate
Editing or reviewing backtest, champion selection, tuning, feature selection,
production forecast generation, or accuracy/FVA endpoints.

## Model registry & config (hard rules)
- **All LightGBM `.fit()`/instantiation goes through `common/ml/model_registry.py`**
  (`fit_model()`, `build_model()`). Direct `LGBMRegressor()` anywhere else is a defect.
- **All hyperparameters live in YAML** `algorithms.<id>.params`. No
  `kwargs.get("n_estimators", 200)` defaults in Python.
- `build_model(algorithm_id, params=None)` reads `algorithms.<id>`, translates via
  `to_native_params()`. Foundation/DL/statistical → `_FoundationStub`; unknown id →
  `UnknownAlgorithm` (subclass of `ValueError`).
- **Forecast quantity column:** `from common.core.constants import FORECAST_QTY_COL` —
  never the literal `"basefcst_pref"`.
- Config helpers: `load_forecast_pipeline_config()`, `get_algorithm_roster(stage=…)`,
  `get_competing_model_ids()`, `get_forecastable_model_ids()`, `get_algorithm_params(id)`.

## ml_cluster is metadata, NOT a feature
- Listed in `METADATA_COLS`, excluded by `get_feature_columns()`, but merged via
  `build_feature_matrix()` for per-cluster partitioning. Using it as a model feature
  (or cross-DFU cluster aggregates) reintroduces the resolved backtest leakage.
- **Promoted labels live in `sku_cluster_assignment`; reads use
  `current_sku_cluster_assignment`.** `dim_sku.ml_cluster` does not exist. New SQL
  must not read clusters from `dim_sku`.
- **An empty `current_sku_cluster_assignment` silently collapses ALL per-cluster tree
  forecasts to near-zero** and breaks backtest + champion-select. Restore via
  `scripts/ml/restore_cluster_assignments.py` from `data/clustering/cluster_labels.csv`
  or promote a completed cluster experiment.

## Pitfalls (symptom → cause → fix)
- **SHAP dimension mismatch** → feature stripped from `feature_cols` but model trained on
  the full set → pre-SHAP stages (duplicate/variance/correlation in `shap_selector.py`)
  prune the *selection pool* only; SHAP input must match the trained set.
- **Multi-stage feature selection** (`shap_selector.py`): 4 per-timeframe (causal) stages —
  (0) duplicate alias removal, (1) near-zero variance, (2) correlation pre-filter,
  (3) SHAP cumulative. Keys `correlation_filter`, `variance_filter`. Per-cluster:
  `compute_timeframe_shap_per_cluster()` → `dict[str, list[str]]`; sparse clusters skip
  SHAP; stratified 50/50 sampling for >50% zeros.
- **Cold-start DFU has no forecast** → < `cold_start_min_months` (3) history is skipped;
  other eligible DFUs route through the configured LightGBM fallback.
  Config: `production_forecast`.
- **Intermittent routing**: sparse clusters remain in the five-model competition. LightGBM
  uses 10% early-stopping patience for sparse demand versus the standard 5%.
- **Cluster strategy**: `algorithms.<name>.cluster_strategy` (default `per_cluster`);
  tree/statistical only — foundation/DL always global. Master switch `clustering.enabled`
  (when false, all backtests fall back to `global`); check `is_clustering_enabled()`.
- **Per-cluster tuning profiles** (`config/forecasting/cluster_tuning_profiles.yaml`):
  Phase 1 by `match_criteria.cluster_name`, Phase 2 by stats (mean_demand, cv_demand,
  zero_demand_pct); first match wins per `_PROFILE_PRIORITY`; resolved by
  `resolve_cluster_params()` in `backtest_framework.py`.
- **Tuning staleness loop (closed)**: `promote_scenario()` flags
  `cluster_tuning_profile_state.stale`; re-tune via `tune_cluster_hyperparams.py
  --stale-only` (job `tune_stale_clusters`, or `POST /admin/tuning/invalidate-stale?retune=true`),
  which MERGES subset results into the YAML (never wipes untouched clusters) and clears the
  flags it covered. The YAML `metadata.cluster_experiment_id` stamps the generation;
  `warn_if_profiles_stale()` (called by `run_tree_backtest`) warns on stale flags or a
  generation mismatch with the promoted cluster experiment.
- **dim_sku joins use the full sku_ck grain** `(item_id, customer_group, loc)` —
  `customer_group` is NOT unique per `(item_id, loc)`. Any accuracy/FVA/budget join to
  `dim_sku` must match all three keys; a 2-key join fans rows across customer_groups and
  inflates WAPE/accuracy/bias. Canonical: `accuracy.py`.
- **Backtest persistence under embargo**: `embargo_months >= 1` skips the last timeframe,
  so `.pkl` persistence targets `_last_persistable_timeframe()`, NOT `len(timeframes)-1`.
- **Fail loud, don't zero-fill**: a failed recursive prediction in
  `generate_production_forecasts.py` re-raises (logged) — never substitute a zero column
  (reads downstream as "no demand" and corrupts the plan/safety stock).
- **Never write synthetic/random forecast data to fact tables.**
  `generate_quantile_forecasts.py` trains on `rng.uniform` (MVP stub) and refuses the
  `fact_demand_plan` write unless `--dry-run`/`--allow-synthetic`.
- **`POST /{model_id}/train` 400** → only `type === "tree"` supports production training;
  foundation/`deep_learning` rejected by design.

## Libraries & data grain
- Clustering lives in `common/ml/clustering/` (`features/training/labeling/scenario`);
  params in the `cluster_experiment` table (promoted row), NOT YAML. Operates on **SKUs**
  (item+location), not DFUs.
- SKU features computed once in `common/ml/sku_features/`, stored in `dim_sku`; clustering
  reads pre-computed values. Config: `config/forecasting/sku_features_config.yaml`.
- Foundation-model loaders live in `common/ml/foundation_backtest.py` — never import from
  `scripts/algorithm_testing/`.
- **Chunk fact-table reads**: `read_sql_chunked()` / `stream_query_in_chunks()` for any
  scan of sales/forecast/inventory. Bare `pd.read_sql` over a fact table OOMs at scale.

## Promotion flow
Predictions → `fact_production_forecast_staging` → promoted to `fact_production_forecast` via
`POST /backtest-management/{model_id}/promote`. Champion routes per-DFU, per-month via the
promoted experiment's `data/champion/experiment_{id}_winners.csv`, keyed on `(item_id, loc,
startdate)`; `dfu_assignments.csv` is dead code, unreferenced. `generate` accepts `horizon` +
`confidence_intervals` query params → script `--horizon` / `--confidence-intervals`|`--no-confidence-intervals`.
- **Cluster-generation lineage (sql/198)**: `champion_experiment.cluster_experiment_id`
  records the promoted clustering at experiment creation. Both
  `generate_production_forecasts.py` (preflight; `--allow-cluster-mismatch` to override)
  and the champion promote endpoint (409; `allow_cluster_mismatch=true`) refuse when it no
  longer matches the currently promoted cluster experiment — winners routing + `.pkl`
  models would silently mismatch current promoted SKU cluster assignments.
- Promote refreshes `fact_production_forecast` dependents (mv_fairness_audit) via
  `common/core/mv_refresh.py` post-commit.
- **Lifecycle as pipelines**: chain the stages via `POST /jobs/pipelines/named/{name}`
  (`data-refresh` / `model-refresh` / `forecast-publish` / `full-refresh`,
  `config/forecasting/pipelines.yaml`) instead of manually stepping jobs. Cross-stage
  staleness surfaces in `GET /dashboard/pipeline-readiness` (clustering wipe, stale tuning,
  champion↔cluster mismatch, data newer than champion).

## Formulas
- **Accuracy**: `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- **Bias**: `(SUM(Forecast) / SUM(History)) - 1`
- **WAPE**: `SUM(|F-A|) / |SUM(A)|`

## Related
- Agent: `forecasting-developer`, `forecasting-qa` · Skill: `python-patterns`, `postgres-patterns`
