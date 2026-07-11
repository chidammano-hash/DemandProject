# Section 4 — Forecasting: Backtests & Prediction Loading

This section explains how to run expanding-window backtests for every model
family registered in the platform and how to load the resulting predictions
into PostgreSQL for downstream champion selection and accuracy analytics.

The single source of truth for the algorithm roster, hyperparameters and
backtest settings is `config/forecasting/forecast_pipeline_config.yaml`. Helpers in
`common/core/utils.py` (`load_forecast_pipeline_config`, `get_algorithm_roster`,
`get_competing_model_ids`, `get_forecastable_model_ids`,
`get_algorithm_params`, `is_clustering_enabled`) provide read access — do
not duplicate values in scripts.

---

## 4.1 Algorithm Roster

The `algorithms:` section of `config/forecasting/forecast_pipeline_config.yaml` defines
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
| Foundation      | `chronos2_enriched`                                                                                       | Covariate-aware; always runs globally (no clusters)                    |
| Deep learning   | `nbeats`, `nhits`                                                                                        | NeuralForecast NHITS / NBEATS; always global                           |

> (`5ab8d593`). `chronos2_enriched` is the only remaining foundation model.

The `chronos2_enriched` model is also designated as the platform's FM spine
(`fm_spine.model_id`) and produces the P10 / P50 / P90 quantile bundle.

### 4.1.3 Per-algorithm config block

Before running any backtest, edit the algorithm entry in
`config/forecasting/forecast_pipeline_config.yaml` under
`algorithms.<model_id>`:

```yaml
algorithms:
  lgbm_cluster:
    type: tree
    cluster_strategy: per_cluster  # "per_cluster" or "global" — ml_cluster used for partitioning only, not a model feature
    recursive: false       # Set true for recursive multi-step inference (Feature 43)
    shap_select: false     # Set true for per-timeframe SHAP feature selection (Feature 42)
    shap_threshold: 0.95   # Cumulative SHAP importance threshold
    shap_top_n: null       # Exact top-N features (overrides threshold)
    shap_sample_size: 500
    tune_inline: false     # Set true for per-timeframe causal tuning (PL-002)
    params_file: null      # Set to data/tuning/best_params_lgbm.json for pre-tuned params
    params:                # Inline hyperparameters (replaces old algorithm_config.yaml)
      n_estimators: 500
      learning_rate: 0.05
      # ... (see full file for all keys)
```

`get_algorithm_params(model_id)` from `common/core/utils.py` to retrieve
hyperparameters.

> **Warning:** Using a globally tuned `params_file` in backtests introduces
> temporal leakage — the tuner sees future timeframes. Use
> `tune_inline: true` for unbiased backtest accuracy. With `tune_inline:
> true`, each of the timeframes tunes on only data available up to its
> training cutoff (~20 trials × 3 folds, ~2–3× slower).

---

## 4.2 Customer Features Pre-Compute

variants that consumed customer-derived features - were removed from
`config/forecasting/forecast_pipeline_config.yaml` in the deprecated-model cleanup
(`5ab8d593`). No algorithm entry currently sets `customer_features: true`, so no
backtest in the active roster reads the customer-features table.

The pre-compute targets and the underlying `customer_features_monthly` table still
exist for other consumers (customer analytics) and for a future re-introduction of an
enriched tree variant - `common/ml/backtest_framework.py` still supports merging it into
the training grid via `include_customer_features`:

```bash
make customer-features          # SQL implementation (preferred, fast)
make customer-features-python   # Python fallback (older path)
```

---

## 4.3 Running Backtests

All backtest targets live at the project root. They write per-model CSV
artefacts into `data/backtest/<model_id>/` (the `output_dir` from the
algorithm entry) and a per-run summary JSON. Each backtest also trains
models AND persists `.pkl` artifacts to `data/models/<model_id>/` for
downstream production forecasting.

> **Closed-month and lag contract:** Backtests use `embargo_months: 0`, so a July
> 2026 planning date scores through June and timeframe J trains through May to
> predict June at natural lag 0. Partial July actuals are excluded. Non-zero embargo
> remains supported for special studies but deliberately removes the corresponding
> shortest natural lags.

### UI run concurrency (Model Tuning → Backtest stage)

`POST /backtest-management/{model_id}/run` (the **Run** button) is concurrency-controlled
and **never blocks the user** — a submission is always accepted or reported as already
in progress, never rejected with an error:

- **Sequential by default** — backtests share the scheduler's `backtest` group, so
  one runs at a time and additional submissions queue (FIFO) and run in order. The
  response is `status: "queued"`.
- **"Run in parallel" toggle** (`?parallel=true`) — submits under a per-job-type
  group (`backtest_<family>`), so *different* model families run concurrently, bounded
  by the scheduler's 4-worker pool. Each model writes its own `data/backtest/<model_id>/`,
  so there is no output collision.
- **No-duplicate guard (informational, not blocking)** — if the *same model* already has
  a run queued or running, the endpoint does **not** start a second one. It returns
  HTTP 200 with `status: "already_running"` and the existing `job_id`; the UI shows a
  calm "already in progress" toast rather than an error. (Earlier versions returned a
  409 here — that hard block was removed.)

### 4.3.1 Per-family commands

```bash
# Tree models (core)
make backtest-lgbm

# Statistical baselines
make backtest-mstl

# Deep learning
make backtest-nbeats
make backtest-nhits

# Foundation models
make backtest-chronos2e               # Chronos 2 Enriched (31 covariates)
```

Several backtest targets also have a `*-full` companion target that runs
the backtest and immediately loads the predictions:
`make backtest-chronos2e-full`, `make backtest-mstl-full`,
`make backtest-nbeats-full`, `make backtest-nhits-full`.

### 4.3.2 Run-everything targets

```bash
                               # chronos2_enriched

make backtest-all-parallel     # Same 4 models in parallel (logs in data/backtest/logs/)
```

Enriched) concurrently and pipes each into a per-model log file under
`data/backtest/logs/`. Use it only on machines with sufficient RAM and CPU;
on a constrained host run `backtest-all` instead. Tree models use `n_jobs=-1`
(all CPU cores) and the foundation model uses the GPU, so running tree +
foundation jobs in parallel is generally safe (CPU vs GPU).

### 4.3.3 Approximate runtimes

Reference numbers (single workstation, GPU=auto). Real wall time depends
on DFU count, history length, GPU availability and CPU core count.

| Model                | Target                       | Typical runtime |
|----------------------|------------------------------|-----------------|
| LGBM cluster         | `make backtest-lgbm`         | 20–40 min       |
| MSTL                 | `make backtest-mstl`         | 5–15 min        |
| NBEATS               | `make backtest-nbeats`       | 30–90 min       |
| NHITS                | `make backtest-nhits`        | 30–90 min       |
| Chronos 2 Enriched   | `make backtest-chronos2e`    | ~6 h            |

### 4.3.4 Backtest framework settings

Settings under `backtest:` in `config/forecasting/forecast_pipeline_config.yaml` apply
to every backtest:

| Key                          | Meaning                                                       |
|------------------------------|---------------------------------------------------------------|
| `n_timeframes`               | Number of expanding-window folds (default 10)                 |
| `embargo_months`             | Extra skipped months after the natural train/predict boundary (default 0) |
| `forecast_horizon`           | Months predicted per fold (default 6 → produces lags 0–4 + 5) |
| `early_stop_pct`             | Patience as fraction of `n_estimators` (5%, 10% for sparse)   |
| `shap_retrain_threshold`     | Re-train trigger when SHAP set churns > 50%                   |
| `intermittent_threshold`     | Zero-share that triggers intermittent routing (default 0.7)   |
| `recursive_noise_enabled`    | Add Gaussian noise to recursive lag features                  |

### 4.3.5 Resuming crashed foundation runs

Foundation backtests checkpoint completed timeframes. To resume a crashed
run (skipping already-completed timeframes), invoke the script module with
`--resume` (where the script supports it):

```bash
uv run python -m scripts.ml.run_backtest_chronos2_enriched --resume
```

See `docs/specs/02-forecasting/18-chronos-foundation-models.md` for the
full architecture comparison across foundation models.

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
```

`--bulk` mode disables the per-model index cycle and instead drops the
target indexes once, COPYs all rows, and rebuilds the indexes a single
time at the end. For full-pipeline reloads always prefer
`make backtest-load-all-bulk`.

### 4.4.2 Selective table loads

```bash
make backtest-load-main-only MODELS="lgbm_cluster chronos2_enriched"
make backtest-load-archive-only MODELS="lgbm_cluster chronos2_enriched"
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
make backtest-load-chronos2e               # convenience wrapper
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

### 4.4.4 Raw loader invocations

The Make targets wrap `scripts/etl/load_backtest_forecasts.py`. For
ad-hoc multi-model loads you can call the script directly:

```bash
# Load 4 models with single index cycle (fastest):
uv run python scripts/etl/load_backtest_forecasts.py \
  --replace --bulk

# Load main table only (skip archive):
uv run python scripts/etl/load_backtest_forecasts.py \
  --models lgbm_cluster chronos2_enriched --replace --bulk --main-only

# Load archive only:
uv run python scripts/etl/load_backtest_forecasts.py \
  --models lgbm_cluster chronos2_enriched --replace --bulk --archive-only
```

> **Tip:** Use `--bulk` whenever loading 2+ models with `--replace`. It
> drops indexes before the first model and recreates them after the last,
> avoiding redundant index rebuilds. For a single model, `--bulk` has no
> benefit.

Each backtest run writes two CSVs per model so multiple models can run
back-to-back without overwriting each other:

* `data/backtest/<model_id>/backtest_predictions.csv` — execution-lag
  predictions (→ `fact_external_forecast_monthly`)
* `data/backtest/<model_id>/backtest_predictions_all_lags.csv` — lag 0–4
  archive (→ `backtest_lag_archive`)

### 4.4.5 Verify archive data

```bash
docker compose exec -T postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive GROUP BY 1,2 ORDER BY 1,2"
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
   (`config/forecasting/cluster_tuning_profiles.yaml`) are resolved by
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
`scripts/ml/run_backtest.py` and the foundation-model scripts.

| Value  | Behaviour                                                                  |
|--------|----------------------------------------------------------------------------|
| `auto` | (default) Use GPU if `cupy`/`torch.cuda` is available, otherwise CPU       |
| `on`   | Force GPU; fail loudly if CUDA / cupy is unavailable                       |
| `off`  | Disable GPU even when available; useful for reproducible benchmark runs    |

```bash
DEMAND_GPU=on  make backtest-chronos2e
DEMAND_GPU=off make backtest-lgbm
```

Optional dependencies that the platform falls back gracefully without:

* `cupy` — GPU arrays for Monte Carlo simulation steps
* `numba` — JIT-compiled seasonality kernels

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

**Loader errors on duplicate keys**
*Symptom:* `duplicate key value violates unique constraint
uq_backtest_lag_archive_ck`.
*Cause:* Loader was invoked without `--replace` after a prior partial
load.
*Fix:* All `make backtest-load-*` targets pass `--replace` automatically;
if you invoke `scripts/etl/load_backtest_forecasts.py` directly, add the flag.

**`backtest-all-parallel` runs out of memory**
*Symptom:* OOM kill in `data/backtest/logs/<model>.log`.
*Cause:* Four processes plus the foundation model's GPU context exceed host RAM.
*Fix:* Run `make backtest-all` (sequential) instead, or stagger by
running the foundation model separately from the tree models.

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

* `config/forecasting/forecast_pipeline_config.yaml` — algorithm roster + backtest /
  tuning / champion settings
* `config/forecasting/cluster_tuning_profiles.yaml` — per-cluster hyperparameter
  overrides
* `common/ml/backtest_framework.py` — expanding-window engine, per-cluster
  partitioning, SHAP integration
* `common/ml/model_registry.py` — `build_model()`, `fit_model()`,
  `to_native_params()`, `compute_early_stop_patience()`
* `scripts/ml/run_backtest_chronos2_enriched.py` - foundation model backtest
* `scripts/ml/run_backtest_dl.py`, `scripts/ml/run_backtest_mstl.py` - DL and
  statistical backtests
* `scripts/etl/load_backtest_forecasts.py` - unified loader (supports
  `--bulk`, `--main-only`, `--archive-only`, `--models`, `--all`)
* `scripts/ml/generate_customer_features_sql.py` — customer features
  pre-compute
* `docs/specs/02-forecasting/03-backtest-framework.md` — full
  framework spec, including the dual-path execution-lag insert ordering
* **Section 5** — Tuning & Cluster Profiles
* **Section 6** — Champion Selection & Forecast Promotion
