# Section 4 — Forecasting: Backtests & Prediction Loading

This section explains how to run expanding-window backtests for every model
family registered in the platform and how to load the resulting predictions
into PostgreSQL for downstream champion selection and accuracy analytics.

The single source of truth for the algorithm roster, hyperparameters and
backtest settings is `config/forecast_pipeline_config.yaml`. Helpers in
`common/utils.py` (`load_forecast_pipeline_config`, `get_algorithm_roster`,
`get_competing_model_ids`, `get_forecastable_model_ids`,
`get_algorithm_params`, `is_clustering_enabled`) provide read access — do
not duplicate values in scripts.

---

## 4.1 Algorithm Roster

The `algorithms:` section of `config/forecast_pipeline_config.yaml` defines
every model that participates in the pipeline. Each entry declares a
`type` (tree / foundation / statistical / deep_learning), a set of
lifecycle `stage` flags, an `output_dir` and a `params:` block that is
consumed by `common/ml/model_registry.py::build_model()`.

### 4.1.1 Stage flags

Five boolean stage flags per algorithm control which phase of the pipeline
the model participates in:

| Stage      | Meaning                                                                          |
|------------|----------------------------------------------------------------------------------|
| `tune`     | Eligible for Bayesian hyperparameter tuning (`make tune-*`)                      |
| `backtest` | Eligible for expanding-window backtest (`make backtest-*`)                       |
| `compete`  | Included in champion selection (`make champion-all`)                             |
| `forecast` | Eligible to produce production forecasts (`make forecast-generate`)              |
| `expert`   | Included in Expert Panel algorithm-selection studies (`make expert-panel*`)      |

Use `get_algorithm_roster(stage="backtest")` to retrieve only models that
should be backtested; use `get_competing_model_ids()` for champion
selection and `get_forecastable_model_ids()` for production inference.

### 4.1.2 Model families

| Family          | Model IDs                                                                                                | Notes                                                                  |
|-----------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| Tree (core)     | `lgbm_cluster`, `catboost_cluster`, `xgboost_cluster`                                                    | Per-cluster training, full SHAP feature selection, GPU when available  |
| Tree (cust-enriched) | `lgbm_cust_enriched`, `catboost_cust_enriched`, `xgboost_cust_enriched`                             | Adds 34 customer-derived features (`make customer-features`)           |
| Foundation      | `chronos` (T5 small), `chronos_bolt` (base), `chronos2`, `chronos2_enriched`, `bolt_hierarchical`        | Zero-shot or covariate-aware; always run globally (no clusters)        |
| Statistical     | `seasonal_naive`, `rolling_mean`, `mstl`                                                                 | Lightweight baselines; `rolling_mean` is also the cold-start fallback  |
| Deep learning   | `nbeats`, `nhits`                                                                                        | NeuralForecast NHITS / NBEATS; always global                           |

The `chronos2_enriched` model is also designated as the platform's FM spine
(`fm_spine.model_id`) and produces the P10 / P50 / P90 quantile bundle.

---

## 4.2 Customer Features Pre-Compute (Required for Enriched Trees)

Before running any of the `*_cust_enriched` backtests you must materialise
the customer-derived feature pack. This produces 34 features (customer
concentration, churn, lag patterns, customer-level seasonality, etc.) keyed
on `(item_id, location_id, month)`.

```bash
make customer-features          # SQL implementation (preferred, fast)
make customer-features-python   # Python fallback (older path)
```

The cust-enriched algorithm entries in
`config/forecast_pipeline_config.yaml` set `customer_features: true` in
their `params:` block — `common/ml/backtest_framework.py` keys off this
flag to merge the customer-features table into the training grid.

If the customer-features table is stale or missing, the enriched backtests
will silently degrade to the base feature set and produce metrics that are
indistinguishable from the non-enriched variants. Always re-run
`make customer-features` after a customer-demand reload.

---

## 4.3 Running Backtests

All backtest targets live at the project root. They write per-model CSV
artefacts into `data/backtest/<model_id>/` (the `output_dir` from the
algorithm entry) and a per-run summary JSON.

### 4.3.1 Per-family commands

```bash
# Tree models (core)
make backtest-lgbm
make backtest-catboost
make backtest-xgboost

# Tree models with customer features
make backtest-lgbm-cust
make backtest-catboost-cust
make backtest-xgboost-cust
make backtest-cust-enriched-all       # all three sequentially

# Statistical baselines
make backtest-seasonal-naive
make backtest-rolling-mean
make backtest-mstl
make backtest-baselines               # seasonal_naive + rolling_mean

# Deep learning
make backtest-nbeats
make backtest-nhits

# Foundation models
make backtest-chronos                 # Chronos T5 small
make backtest-bolt                    # Chronos Bolt base
make backtest-chronos2                # Chronos 2 zero-shot
make backtest-chronos2e               # Chronos 2 Enriched (31 covariates)
make backtest-bolt-hier               # Bolt customer bottom-up + reconciliation
```

Each foundation backtest also has a `*-full` companion target that runs
the backtest and immediately loads the predictions
(e.g. `make backtest-chronos-full`, `make backtest-bolt-full`,
`make backtest-chronos2-full`, `make backtest-chronos2e-full`,
`make backtest-bolt-hier-full`, `make backtest-mstl-full`,
`make backtest-nbeats-full`, `make backtest-nhits-full`).

### 4.3.2 Run-everything targets

```bash
make backtest-all              # 7 backtests sequentially: lgbm, catboost, xgboost,
                               # chronos, chronos_bolt, chronos2, chronos2_enriched

make backtest-all-parallel     # Same models in parallel (logs in data/backtest/logs/)
```

`backtest-all-parallel` launches the six core jobs concurrently and pipes
each into a per-model log file under `data/backtest/logs/`. Use it only on
machines with sufficient RAM and CPU; on a constrained host run
`backtest-all` instead.

### 4.3.3 Approximate runtimes

Reference numbers (single workstation, GPU=auto). Real wall time depends
on DFU count, history length, GPU availability and CPU core count.

| Model                | Target                       | Typical runtime |
|----------------------|------------------------------|-----------------|
| LGBM cluster         | `make backtest-lgbm`         | 20–40 min       |
| CatBoost cluster     | `make backtest-catboost`     | 30–60 min       |
| XGBoost cluster      | `make backtest-xgboost`      | 25–50 min       |
| LGBM cust-enriched   | `make backtest-lgbm-cust`    | 25–50 min       |
| CatBoost cust-enrich | `make backtest-catboost-cust`| 35–70 min       |
| XGBoost cust-enriched| `make backtest-xgboost-cust` | 30–60 min       |
| Seasonal naive       | `make backtest-seasonal-naive` | 1–3 min       |
| Rolling mean         | `make backtest-rolling-mean` | 1–3 min         |
| MSTL                 | `make backtest-mstl`         | 5–15 min        |
| NBEATS               | `make backtest-nbeats`       | 30–90 min       |
| NHITS                | `make backtest-nhits`        | 30–90 min       |
| Chronos T5 (small)   | `make backtest-chronos`      | ~2.5 h          |
| Chronos Bolt (base)  | `make backtest-bolt`         | ~12 min         |
| Chronos 2 zero-shot  | `make backtest-chronos2`     | ~5.5 h          |
| Chronos 2 Enriched   | `make backtest-chronos2e`    | ~6 h            |
| Bolt hierarchical    | `make backtest-bolt-hier`    | 30–90 min       |

### 4.3.4 Backtest framework settings

Settings under `backtest:` in `config/forecast_pipeline_config.yaml` apply
to every backtest:

| Key                          | Meaning                                                       |
|------------------------------|---------------------------------------------------------------|
| `n_timeframes`               | Number of expanding-window folds (default 10)                 |
| `embargo_months`             | Gap between train end and validation start (default 1)        |
| `forecast_horizon`           | Months predicted per fold (default 6 → produces lags 0–4 + 5) |
| `early_stop_pct`             | Patience as fraction of `n_estimators` (5%, 10% for sparse)   |
| `shap_retrain_threshold`     | Re-train trigger when SHAP set churns > 50%                   |
| `baseline_intermittent`      | Route intermittent clusters to rolling mean                   |
| `intermittent_threshold`     | Zero-share that triggers intermittent routing (default 0.7)   |
| `recursive_noise_enabled`    | Add Gaussian noise to recursive lag features                  |

---

## 4.4 Loading Predictions

Backtest CSV artefacts are loaded into two destinations:

* `fact_candidate_forecast` — main staging table for execution-lag (lag 0)
  predictions consumed by champion selection and forecast promotion.
* `backtest_lag_archive` — long-form table holding lags 0–4 with a
  `timeframe` column, used for the lag-curve, accuracy-by-slice and
  control-tower MVs.

The execution-lag dual-path insert always loads the archive **before**
mutating the staging table — see
`docs/specs/02-forecasting/03-backtest-framework.md` for details.

### 4.4.1 Bulk loaders (preferred)

```bash
make backtest-load-all            # loads every model_id present under data/backtest/
make backtest-load-all-bulk       # same set, but drops & rebuilds indexes once → ~4× faster
make backtest-load-bulk           # 4 core models: lgbm_cluster, catboost_cluster,
                                  # xgboost_cluster, chronos
```

`--bulk` mode disables the per-model index cycle and instead drops the
target indexes once, COPYs all rows, and rebuilds the indexes a single
time at the end. For full-pipeline reloads always prefer
`make backtest-load-all-bulk`.

### 4.4.2 Selective table loads

```bash
make backtest-load-main-only MODELS="lgbm_cluster chronos2"
make backtest-load-archive-only MODELS="lgbm_cluster chronos2"
```

* `--main-only` loads `fact_candidate_forecast` and skips
  `backtest_lag_archive` (use when you only need the lag-0 staging row,
  e.g. to re-run champion selection without rebuilding the archive).
* `--archive-only` loads `backtest_lag_archive` and skips the staging
  table (use when re-populating accuracy MVs without touching candidate
  predictions).

### 4.4.3 Single-model loaders

```bash
make backtest-load MODEL=lgbm_cluster      # generic single-model loader
make backtest-load-chronos                 # convenience wrappers
make backtest-load-bolt
make backtest-load-bolt-hier
make backtest-load-chronos2
make backtest-load-chronos2e
make backtest-load-cust-enriched           # all three cust-enriched models
make backtest-load-seasonal-naive
make backtest-load-rolling-mean
make backtest-load-mstl
make backtest-load-nbeats
make backtest-load-nhits
```

All loaders accept `--replace` (the Make targets pass it by default),
which deletes the existing rows for that `model_id` before inserting.

After loading, refresh the accuracy MVs:

```bash
make accuracy-slice-refresh
```

---

## 4.5 Cluster Strategy Resolution

The `cluster_strategy` field on a tree or statistical algorithm controls
whether the backtest trains one model per `ml_cluster` or a single model
across all DFUs.

### 4.5.1 Resolution order

1. **Algorithm entry** —
   `algorithms.<model_id>.cluster_strategy` in
   `forecast_pipeline_config.yaml`. Valid values: `"per_cluster"`,
   `"global"`.
2. **Default** — `"per_cluster"` (applied by
   `common/ml/backtest_framework.py` when the field is absent).
3. **Foundation / deep-learning models** — always run globally regardless
   of any setting; the field is ignored for those types.

### 4.5.2 Master switch

`clustering.enabled` in `forecast_pipeline_config.yaml` is the master
switch for the whole clustering pipeline. When `false`:

* `is_clustering_enabled()` returns `False`.
* All tree and statistical backtests fall back to `cluster_strategy=global`
  regardless of the per-algorithm setting.
* `ml_cluster` is still preserved in the feature grid but every DFU is
  effectively in the same partition.

This switch lets you A/B test the value of clustering without re-editing
each algorithm entry.

---

## 4.6 Per-Cluster vs Global Execution

When `cluster_strategy=per_cluster`:

1. The training grid is built once via
   `build_feature_matrix()` and includes the `ml_cluster` column.
2. The grid is partitioned by `ml_cluster` and one model is fit per
   partition. Per-cluster hyperparameter overrides
   (`config/cluster_tuning_profiles.yaml`) are resolved by
   `resolve_cluster_params()` — see CLAUDE.md "Per-cluster tuning
   profiles" for the matching rules.
3. Predictions are concatenated back into a single CSV.

When `cluster_strategy=global`:

* A single model is fit on the full grid. `ml_cluster` is added to
  `cat_cols` so the model can learn cluster effects directly.

### 4.6.1 Intermittent cluster routing

Clusters whose zero-demand share exceeds `backtest.intermittent_threshold`
(default `0.7` → 70% zeros) are **not** trained with the tree model.
Instead, when `backtest.baseline_intermittent: true`, those clusters are
routed to a rolling-mean baseline using
`backtest.baseline_intermittent_window` (default 12 months).

This avoids tree models over-fitting noise in extremely sparse partitions
and is materially more accurate than a small tree fit on near-zero data.
The routing decision is logged per cluster at backtest time.

---

## 4.7 Multi-Stage Feature Selection (SHAP)

Tree backtests use `shap_selector.py` to reduce the feature set per
timeframe. The selector runs four stages **in order**:

| Stage | Name                       | Purpose                                                                    | Config key(s)                                  |
|-------|----------------------------|----------------------------------------------------------------------------|------------------------------------------------|
| 0     | Duplicate alias removal    | Drop columns that are duplicate aliases (e.g. legacy renames)              | always on                                      |
| 1     | Near-zero variance filter  | Drop columns with variance below `variance_threshold`                      | `variance_filter`, `variance_threshold`        |
| 2     | Correlation pre-filter     | Drop one of any pair with `|corr|` above `correlation_threshold`           | `correlation_filter`, `correlation_threshold`  |
| 3     | SHAP cumulative selection  | Keep top features by mean-|SHAP| up to `shap_threshold` cumulative share   | `shap_select`, `shap_threshold`, `shap_top_n`  |

Per-algorithm toggles live in the algorithm `params:` block — for example
`lgbm_cluster` ships with `correlation_filter: false, variance_filter:
false, shap_threshold: 0.9`, while the cust-enriched variants enable both
filters and a tighter `shap_threshold: 0.95`.

### 4.7.1 Per-cluster SHAP

When `cluster_strategy=per_cluster`,
`compute_timeframe_shap_per_cluster()` runs SHAP independently per
cluster and returns `dict[str, list[str]]` mapping cluster label → list
of selected features. The per-cluster feature lists are then propagated
into the per-cluster fit/predict path so each cluster trains on its own
feature subset.

* Sparse clusters with too few non-zero rows skip SHAP entirely and keep
  the full feature set (controlled by `shap_min_features`).
* For clusters with > 50% zero-demand rows the selector uses 50/50
  stratified sampling between zero and non-zero rows so SHAP attributions
  are not dominated by the zero mass.

### 4.7.2 SHAP retrain safety

If the SHAP-selected feature set churns by more than
`backtest.shap_retrain_threshold` (default 50%) between timeframes, the
fold is retrained with the new feature set. The framework also enforces a
"global retrain reverted" safety net — if the retrain produces fewer
features than `shap_min_features`, the previous feature set is restored
and a warning is logged.

---

## 4.8 GPU Acceleration

GPU usage is controlled by the `DEMAND_GPU` environment variable. Read by
`scripts/run_backtest.py` and the foundation-model scripts.

| Value  | Behaviour                                                                  |
|--------|----------------------------------------------------------------------------|
| `auto` | (default) Use GPU if `cupy`/`torch.cuda` is available, otherwise CPU       |
| `on`   | Force GPU; fail loudly if CUDA / cupy is unavailable                       |
| `off`  | Disable GPU even when available; useful for reproducible benchmark runs    |

```bash
DEMAND_GPU=on  make backtest-chronos2
DEMAND_GPU=off make backtest-lgbm
```

Optional dependencies that the platform falls back gracefully without:

* `cupy` — GPU arrays for Monte Carlo simulation steps
* `numba` — JIT-compiled seasonality kernels

Tree backtests (LGBM / CatBoost / XGBoost) use each library's native GPU
backend (`device_type='gpu'`, `task_type='GPU'`, `tree_method='hist',
device='cuda'`) when the flag is on.

---

## 4.9 Promotion to Production Forecast (Forward Reference)

Backtest predictions land in `fact_candidate_forecast` only — they do
**not** automatically appear in `fact_production_forecast`. Promotion is a
separate explicit step:

* Single-model promotion via
  `POST /backtest-management/{model_id}/promote` copies the rows for one
  `model_id` into `fact_production_forecast`.
* Champion promotion uses the per-DFU assignments from
  `data/champion/dfu_assignments.csv` and copies the chosen
  `(model_id, dfu)` rows.

Both flows write to `model_promotion_log` for audit. See **Section 6 —
Champion Selection & Forecast Promotion** for the full workflow,
promote-gate guard rails (`min_wape_improvement_pct`, `min_coverage_frac`)
and rollback procedure.

---

## 4.10 Verification & Troubleshooting

### 4.10.1 Verify rows landed

```bash
make backtest-list                 # row counts per model_id in fact_candidate_forecast
make health                        # DB row counts + API health
make accuracy-slice-check          # smoke-test accuracy slice + lag curve endpoints
```

The `psql` summary embedded in the Makefile (`make check-db`) also dumps
`SELECT model_id, count(*) FROM fact_external_forecast_monthly GROUP BY
model_id` so you can confirm every expected `model_id` is present.

### 4.10.2 Common failure modes

**Dimension mismatch in SHAP**
*Symptom:* `ValueError: shape mismatch` when SHAP values are computed.
*Cause:* A feature was stripped from `feature_cols` but the trained model
still expects it. Pre-SHAP stages (duplicate / variance / correlation)
exclude features only from the *selection pool* — the SHAP extractor must
receive the full feature set the model was trained with.
*Fix:* Never trim `feature_cols` between training and SHAP extraction.
Use `effective_feature_cols` returned by the framework, not the
post-filter list.

**Sparse cluster crashes / NaN metrics**
*Symptom:* A cluster has fewer than `shap_min_features` non-zero rows or
produces NaN WAPE.
*Cause:* Extremely intermittent cluster slipped past the 70% threshold or
has too few non-zero observations for stable SHAP.
*Fix:* The framework already routes >70% zero clusters to the rolling
mean baseline; lower `intermittent_threshold` (e.g. to 0.5) if your
dataset is unusually sparse, or raise `baseline_intermittent_window` to
smooth the baseline.

**Cust-enriched backtest produces same metrics as base**
*Symptom:* `lgbm_cust_enriched` accuracy is identical to `lgbm_cluster`.
*Cause:* Customer-features table is empty or stale, so the join produces
NULLs and the model degenerates to the base feature set.
*Fix:* Run `make customer-features` and re-run the enriched backtest.

**Loader errors on duplicate keys**
*Symptom:* `duplicate key value violates unique constraint
uq_backtest_lag_archive_ck`.
*Cause:* Loader was invoked without `--replace` after a prior partial
load.
*Fix:* All `make backtest-load-*` targets pass `--replace` automatically;
if you invoke `scripts/load_backtest_forecasts.py` directly, add the flag.

**`backtest-all-parallel` runs out of memory**
*Symptom:* OOM kill in `data/backtest/logs/<model>.log`.
*Cause:* Six processes plus optional GPU contexts exceed host RAM.
*Fix:* Run `make backtest-all` (sequential) instead, or stagger by
running the foundation models separately from the tree models.

**No predictions for cold-start DFUs**
*Symptom:* DFU is missing from the candidate forecast table.
*Cause:* DFU has < `cold_start_min_months` (3) months of sales history
and is skipped at the production-forecast stage. This is **production
forecast** behaviour, not backtest behaviour — backtests honour the
expanding-window cutoff and may emit predictions for DFUs that the
production stage skips.
*Fix:* See `production_forecast` settings in
`forecast_pipeline_config.yaml` and Section 6 of this manual.

---

## 4.11 Cross-References

* `config/forecast_pipeline_config.yaml` — algorithm roster + backtest /
  tuning / champion settings
* `config/cluster_tuning_profiles.yaml` — per-cluster hyperparameter
  overrides
* `common/ml/backtest_framework.py` — expanding-window engine, per-cluster
  partitioning, SHAP integration
* `common/ml/model_registry.py` — `build_model()`, `fit_model()`,
  `to_native_params()`, `compute_early_stop_patience()`
* `scripts/run_backtest.py`, `scripts/run_backtest_catboost.py`,
  `scripts/run_backtest_xgboost.py` — tree backtests
* `scripts/run_backtest_chronos*.py`,
  `scripts/run_backtest_chronos_bolt.py`,
  `scripts/run_backtest_bolt_hierarchical.py` — foundation model backtests
* `scripts/run_backtest_dl.py`, `scripts/run_backtest_mstl.py` — DL and
  statistical backtests
* `scripts/load_backtest_forecasts.py` — unified loader (supports
  `--bulk`, `--main-only`, `--archive-only`, `--models`, `--all`)
* `scripts/ml/generate_customer_features_sql.py` — customer features
  pre-compute
* `docs/specs/02-forecasting/03-backtest-framework.md` — full
  framework spec, including the dual-path execution-lag insert ordering
* **Section 5** — Tuning & Cluster Profiles
* **Section 6** — Champion Selection & Forecast Promotion
