# Forecasting Subsystem — Remaining Work

**As of:** 2026-06-20 · branch `refactor/codebase-cleanup`
**Context:** Continuation of the 4-agent forecasting review/fix loop. 23 fixes already landed
(commits `33208e91..f7f772f3`, full suite green at the 2 pre-existing `consensus_plan` fails).
Everything below is **deferred / human-PR scope** — verified-but-risky, methodology, or feature work
that the autonomous loop intentionally did **not** touch.

The authoritative running backlog lives in Claude memory at
`project_forecasting_review_2026_06_20.md`. This file is the repo-local, reviewable extract,
led by a fresh review of `api/routers/forecasting/competition.py` (the open file).

---

## A. `competition.py` — review findings (the open file)

`api/routers/forecasting/competition.py` (414 LoC). The COPY-buffer escaping was already fixed
(both `_insert_pick_winners` and `_insert_ensemble_winners` now use `copy.write_row(...)`).
What remains:

### A1. [P0] `update_competition_config` strips all inline comments + no backup/atomic/lock — line ~134
```python
with _PIPELINE_CONFIG_PATH.open("w") as f:
    yaml.dump(pipeline_cfg, f, default_flow_style=False, sort_keys=False)
```
- `yaml.safe_load` → mutate → `yaml.dump` **destroys every inline comment** in
  `forecast_pipeline_config.yaml`. This violates the project rule "inline comments required on every
  key in every config YAML" — and it happens on **every UI competition-config save**.
- **Truncate-in-place with NO backup**, no atomic write (temp + `os.replace`), no serialize lock →
  a crash mid-write or two concurrent saves corrupts / loses the master config.
- This is **one of six identical writers** (see C1). Fix is shared: a centralized
  comment-preserving writer (ruamel.yaml round-trip) + atomic replace + backup + lock.
- Note: tests no longer dirty the real file — `test_competition.py` has an autouse fixture
  redirecting `_PIPELINE_CONFIG_PATH` to a temp copy (commit `0da1c2a6`). The *production* writer is
  still destructive.

### A2. [P1-scale] `_load_monthly_errors` loads the full fact-table slice into memory — line ~66
```python
cur.execute(sql, params)
rows = cur.fetchall()              # <- whole filtered fact_external_forecast_monthly in RAM
df = pd.DataFrame(rows, columns=cols)
```
- `run_competition` calls this synchronously **inside the request handler**. At 40× scale this is the
  same OOM risk we already fixed in `run_champion_selection.load_monthly_errors_df`
  (commit `f7f772f3`, switched to `read_sql_chunked`). The API sibling was not converted.
- Fix: stream via `read_sql_chunked` / `stream_query_in_chunks` from `common.core.sql_helpers`, or
  move the whole competition run to the pg-queue worker (it already enqueues an MV refresh job, so the
  precedent for "this is a long job" exists).

### A3. [P2] `_enqueue_forecast_view_refresh` bare `except Exception` — line ~296
```python
except Exception:
    ... logger.exception("Failed to enqueue forecast view refresh job")
    return None
```
- Graceful-degradation intent is fine and it **does** log, but it's a bare `except Exception` with no
  `# noqa: BLE001 — <reason>` justification (project rule). Narrow to the actual failure modes
  (scheduler-unavailable `ImportError` / `RuntimeError` + `psycopg.Error`) or add the justified noqa.

### A4. [P2-methodology] champion single-candidate "win" + no outer holdout
- `run_competition` → strategy picks a winner per DFU-month even when **only one model qualifies**
  (`min_prior_months` filter can leave a single candidate) — that "win" is not a real competition.
- Strategy selection has **no outer holdout**: the chosen strategy is scored on the same months it
  selected over (`compute_strategy_accuracy(winners_df)`), so reported champion accuracy is optimistic.
  This is a science decision, not a quick fix — flagged for the methodology PR.

---

## B. Core-engine P0s (highest value, deliberately deferred — need careful refactor + full backtest validation)

### B1. SHAP-retrain test-set leakage — `common/ml/backtest_framework.py:1425-1452` (per-cluster) & `:1521-1581` (global)
The keep-original-vs-SHAP-reduced "safety check" selects the model by WAPE on `predict_data` —
**the held-out test window** — i.e. test-set model selection. This inflates **every reported backtest
accuracy number** in the system. Fix: decide on a held-out slice of **train** (reuse the early-stop
val split, or plumb the early-stop val WAPE out of `train_fn_per_cluster`), never `predict_data`.
Touches the 1776-LoC core — validate that exec_lag=0 accuracy and champion ranking move sensibly.

### B2. Meta-learner train/serve causality skew — `train_meta_learner.py:137-170` vs `common/ml/champion/meta.py:131-193`
Training builds labels with `shift(1)`; serving builds features with `shift(exec_lag+1)`. For
`exec_lag>0` this is **train-time label leakage** (inflates `meta_learner_report.json`) **and** a
serve-time feature-distribution mismatch → wrong model routing in production. Fix: merge
`execution_lag` into the training frame before the shift and mirror the exec-lag-aware per-group
shift — ideally **extract `meta.py._build_meta_features` and share it** so train/serve can't diverge.
Validate `exec_lag=0` accuracy unchanged.

### B3. [P0] DFUs with predict rows but no train rows silently dropped — `run_backtest.py:844,950-959`
A cluster that has prediction rows but no training rows is silently skipped (not zeroed, no naive
fallback) → those DFUs vanish from the backtest with no signal. Fix: derive the cluster universe from
`predict_df` and route train-empty clusters to the naive baseline.

### B4. [P0] chronos2-enriched covariate misalignment — `foundation_models.py:746-756`
Dense-grid covariates are aligned to the sparse sales target **by position**, so gappy DFUs get the
wrong covariate month. Fix: merge by `(sku_ck, startdate)` instead of positional alignment.

---

## C. Config-writer corruption (P0 — six writers, one shared fix)

### C1. All six config-writers `yaml.dump` the master config → strip all comments
| File | Line | Backup? |
|---|---|---|
| `api/routers/platform/config_manager.py` | ~706 | — |
| `api/routers/forecasting/competition.py` | ~135 | **none** (truncate-in-place) |
| `api/routers/forecasting/.../model_tuning.py` | ~646 | **none** |
| `api/routers/forecasting/champion_experiments.py` | ~861 | — |
| `api/routers/forecasting/.../lgbm_tuning.py` | ~607 | **none** |
| `api/routers/forecasting/tuning/promote.py` | ~90 | — |

All round-trip `forecast_pipeline_config.yaml` through PyYAML, destroying inline comments (rule
violation) on every UI save. Three truncate-in-place with no backup. Fix once: a centralized
comment-preserving writer (`ruamel.yaml` round-trip) + atomic write (temp + `os.replace`) + backup +
a serialize lock (concurrent writes currently race / lost-update). **Adds a ruamel.yaml dependency —
explicitly a human decision, not an autonomous change.**

---

## D. P1 — methodology / correctness (medium risk)

- **CatBoost early-stop metric divergence** — `model_registry.py:403-441`. CatBoost early-stops on
  RMSE/MAE while LGBM/XGB stop on WAPE → champion comparison isn't apples-to-apples. Fix: pass
  CatBoost the existing `WapeMetric` (`model_registry.py:172`). Alters early-stop/outputs → needs care.
- **17 async handlers call sync `get_conn()`** (consensus_plan / blended_forecast / bias_corrections /
  fva) → block the event loop. Lower-risk fix: drop `async` so FastAPI threadpools the read handlers.
  `consensus_plan` uses `await require_api_key` in the body → handle carefully; it already has 2
  flaky-failing tests.
- **Tree-model instantiation bypasses `model_registry.build_tree_model()`** —
  `run_backtest.py:716,1007`, `train_production_models.py:382` use `model_class(**fit_params)`.
  `fit()` is compliant; only construction bypasses. Correct today (native params pass through) →
  centralization-only benefit vs hot-path risk. Low priority.
- **`champion_experiment` per-lag breakdown** re-runs the whole strategy on single-lag slices
  (degenerate priors) instead of scoring the **chosen** champion per lag. Medium-risk refactor.
- **Ensemble COPY JSONB site** — `run_champion_selection.py` ~553 (`source_mix` JSONB) still builds a
  raw tab-delimited buffer. Needs a `Jsonb` wrapper + a live-DB test before converting to `write_row`.
- **Hardcoded meta-learner hyperparams + direct `XGBClassifier()`** — `train_meta_learner.py:222-242`
  (config `champion.meta_learner` ignored on the XGB path).
- **Customer-feature latent leakage + dead flag** — customer features merged contemporaneously, never
  re-masked by `mask_future_sales`; `*_cust_enriched` tree configs silently ignore the
  `customer_features` flag (dead config). Coupled — fixing the dead flag activates the leak; **mask
  first**.
- **bias_corrections `--apply-to-plan` is a dead no-op** (writes factors, never applies) + cluster
  join `customer_group` fan-out + `print()`/bare-except.
- **Safety-stock sigma methodology** — `compute_safety_stock.py:263-302`. Demand sigma back-derived
  from CI band width (= forecast-error RMSE × √h), horizon-inflated + clamp-biased for low-volume.
  Inventory-methodology change.
- **Bolt hierarchical renorm** — `run_backtest_bolt_hierarchical.py:450-468`. Disaggregation shares not
  renormalized after the `dim_sku` inner join → children don't sum to parent.
- **Expert-panel leaked routing-gain** — `expert_panel_route_analysis.py` S2/S3/S4 "causal" strategies
  select on the same calendar months they evaluate (`predict_end=latest` for all timeframes).
- **CI bands are ±sigma normal, never empirically calibrated** — `forecast_ci.py`. Biggest CI-quality
  lever; needs residual-quantile / calibration work.
- **Unclamped `100-WAPE` accuracy** in `compute_kpis` (`api/core.py:423`),
  `helpers.compute_strategy_accuracy` (`helpers.py:187`), `accuracy_budget._accuracy`, `fva.py` SQL —
  vs canonical clamped `compute_accuracy` (`metrics.py`). Same DFU reads negative in one panel, 0 in
  another. Display/consistency decision (clamp everywhere recommended) — changes shown numbers +
  champion ranking semantics.
- **Lag-source divergence** — accuracy/budget/fva use `dim_sku.execution_lag` (COALESCE→0) while
  champion selection uses `fact.execution_lag` → can report at a different horizon than selection.
  Needs a canonical-source decision.
- **tuning `gap_months` (2) vs backtest `embargo_months` (1) disagree** — `tuning.py` docstring claims
  alignment. Single-source-of-truth fix.
- **Recursive-noise RNG unseeded** (`np.random` global) → non-reproducible backtests. Thread a seeded
  generator (kept global to preserve current semantics).
- **`production_forecast.py:253,282`** dim_sku join drops `customer_group`
  (`fact_production_forecast` has no `customer_group` col → needs DISTINCT-ON, not a join key).

---

## E. P2 — lower priority / cleanup

- Champion not gated vs a naive baseline; inverse-WAPE blend weights → inverse-variance / NNLS;
  blend-strategy in-sample eval (sweep overfitting).
- Statistical/foundation champions silently served by hand-rolled formulas in production
  (`generate_forecasts_statistical`).
- Contiguous-tail early-stop val split has no embargo gap.
- `sweep` composite in-sample optimism (per-segment argmax, no holdout); robust objective uses an
  unweighted mean of per-lag accuracies (not volume-weighted).
- `resolve_conflicts()` Step-3 duplication (**NOT dead — 8 tests cover it**; DRY refactor only).
- `foundation_backtest.py:41` module-level `parents[2]` → should be
  `common.core.paths.PROJECT_ROOT`.
- `min_months_history` default 1-vs-12 mismatch across clustering entry points; cluster-label
  non-determinism (`labeling.py` `_cN` suffix from volatile KMeans index).
- `COUNT(DISTINCT a||b)` separator collisions in remaining spots; bulk bare-except / `print()` rule
  violations across routers; `_select_features_from_shap` `list.index()` assumes unique names.
- `sensing_config.yaml` points at a non-existent `common/ml/sensing.py`.
- `.bak.37/.38/.53` config backups in `config/forecasting/` are cruft — delete.
- `apply_event_adjustments.py:325` — event `impact_value` hardwired to $0 (no cost join). Feature.

---

## Verified clean (agents found NO defect — do not re-investigate)

- `domains.py` mounted last; all forecasting write endpoints `require_api_key`-guarded; no SQL
  injection; no `$1` placeholders; vite proxy / barrel parity intact.
- Causal champion selection logic (SQL + pandas paths agree); backtest feature engineering strictly
  causal (`mask_future_sales`).
- Recursive-rolling-feature "skew" — `LAG_RANGE` is contiguous `range(1,13)`, so rolling windows
  always have all lags present.
- `resolve_conflicts()` is **not** dead code (8 unit tests).

---

## Suggested sequencing for the human PR(s)

1. **PR-1 (mechanical, low-risk):** centralized comment-preserving config writer (C1) — unblocks all
   six writers incl. `competition.py` A1, restores the "inline comments required" invariant, adds
   backups. Plus `competition.py` A2 (chunked read) and A3 (narrow the except).
2. **PR-2 (accuracy-critical, needs full backtest validation):** SHAP-retrain leakage (B1) +
   meta-learner skew (B2). These move every reported accuracy number — land them together with a
   before/after backtest comparison.
3. **PR-3 (correctness):** B3 (train-empty cluster drop), B4 (chronos2 covariate align), the customer-
   feature leak + dead flag (D), CatBoost metric (D).
4. **PR-4 (consistency/methodology):** unclamped-accuracy + lag-source canonicalization, CI
   calibration, safety-stock sigma — the display/science decisions.
