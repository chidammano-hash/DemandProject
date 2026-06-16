## scripts/ — Files Not Needed

_Read-only audit. Confidence: SAFE = zero refs anywhere (Makefile/scripts/config/docs); PROBABLE = zero refs but plausibly a manual operator tool; INVESTIGATE = needs human check._

Scope audited: `scripts/{etl,ml,forecasting,inventory,ops,ai,db,tools}` plus `scripts/ai_checks`. 88 Python scripts total. There is **no** `scripts/algorithm_testing/` directory (docs/RUNBOOK still references `algorithm_testing/run_expert_panel.py` — that's a stale doc path, the live file is `scripts/ml/run_expert_panel.py`, which IS wired).

Method: for each script, counted references by basename across `Makefile`, `scripts/`, `common/`, `api/`, `tests/`, `config/`, `docs/` (excluding `__pycache__` and the file's own self-references). Then manually verified every low-count candidate against Make targets, `subprocess`/job-registry invocation, and pipeline step lists. CLI entry-points with no import graph were kept if any Make target / job registry / subprocess invokes them.

### SAFE to delete

- `scripts/ml/expert_panel_main.py` (27 LoC) — orphaned macOS fork-safety entry-point wrapper. Its docstring claims it is the entry point for `python -m common.ml.expert_panel`, but (a) it lives in `scripts/ml/`, not `common/ml/expert_panel/`; (b) `common/ml/expert_panel/` has **no** `__main__.py`, so `python -m common.ml.expert_panel` does not resolve to it; (c) it just does `from scripts.ml.run_adv_expert_panel import main; main()`. All expert-panel Make targets (`expert-panel*`, `adv-expert-panel*`) invoke `scripts.ml.run_expert_panel` / `scripts.ml.run_adv_expert_panel` directly, never this wrapper. Evidence: `grep -rn expert_panel_main` across Makefile/scripts/common/api/docs/config/tests → 0 hits (only its own file). Leftover from the `common.ml.expert_panel` package refactor.

### PROBABLE

- `scripts/etl/trim_input_files.py` (244 LoC) — one-off dev/data-prep utility that trims `data/input/` CSV/TXT files in place to a subset of locations/sites (defaults: location `1401-BULK`, site `1`). Completely unwired: no Make target, no subprocess caller, no docs, no config, no test. Only references are the usage examples inside its own docstring. Reads like a scratch tool used to shrink the input dataset during development. PROBABLE rather than SAFE because it is a destructive in-place data tool an operator might keep around for manual dataset trimming. Evidence: `grep -rn "trim_input"` across Makefile/scripts/common/api/docs/config/tests → 0 hits outside the file itself.

- `scripts/ml/label_clusters.py` (192 LoC) — thin CLI wrapper around `common.ml.clustering.labeling.assign_cluster_labels` (its own docstring says "Core labeling logic lives in common.ml.clustering.labeling"). The actual clustering pipeline (`scripts/ml/run_clustering_scenario.py`) imports `assign_cluster_labels` **directly from the library**, not via this script. Not wired to any Make target: docs reference a `make cluster-label` target and a `label_clusters` pipeline step, but **no `cluster-label` target exists** in the Makefile, and the `label_clusters` strings in `run_clustering_scenario.py`/config are a `profiled_section()` label and a config flag — not invocations of this file. Superseded by the clustering-package refactor (per MEMORY: clustering consolidated into `common/ml/clustering/`, `run_cluster_pipeline.py`). PROBABLE rather than SAFE because it remains a usable standalone re-labeling CLI and is name-dropped in clustering docs. Evidence: no `import label_clusters` / `from scripts.ml.label_clusters` anywhere; no `cluster-label` Make target; only doc/config mentions of the word "label_clusters".

### INVESTIGATE

- `scripts/inventory/run_inventory_backtest.py` (397 LoC) — **wired but via a broken path.** Registered in the job system: `common/services/job_registry.py` → `_run_inventory_backtest` in `common/services/job_state.py`, which shells out to `["uv","run","python","scripts/run_inventory_backtest.py"]`. That path (`scripts/run_inventory_backtest.py`) does **not** exist — the file is at `scripts/inventory/run_inventory_backtest.py`. Also referenced in `docs/operations-manual/07-inventory-planning.md` as `make ip-backtest`, but **no `ip-backtest` Make target exists** either. So the script is *intended* to be live (job registry + ops docs) but is currently unreachable due to a stale path in `job_state.py`. NOT dead — but either the job-registry path needs fixing OR (if the inventory-backtest job is being retired) both the script and the job entry should be removed together. Human decision needed; do not delete in isolation (would orphan the job-registry entry).

### NOT dead (checked, keep) — looked low-ref but are wired

- `scripts/ml/expert_panel_route_analysis.py` — wired via `make route-analysis` / `route-analysis-min3` (`python -m scripts.ml.expert_panel_route_analysis`).
- `scripts/ml/run_expert_panel.py` — wired via `expert-panel`, `expert-panel-quick/mini/loc`.
- `scripts/ml/run_adv_expert_panel.py` — wired via `adv-expert-panel*`.
- `scripts/ml/run_expert_system_backtest.py` — wired via `expsys-backtest`, `expsys-backtest-dry`, `expsys-backtest-replace`.
- `scripts/tools/scaffold_router.py` — wired via `make new-router`.
- `scripts/ml/generate_customer_features.py` (Python variant) — wired via `make customer-features-python`; `_sql` variant via `make customer-features`. Both documented as alternatives; keep both.
- `scripts/inventory/compare_inventory_algorithms.py` — wired via `make algo-comparison` (the ops-manual `ip-compare` alias is stale, but the target exists under a different name).
- `scripts/tools/bench_ingestion.py` — wired via `make perf-ingestion` / RUNBOOK.
- `scripts/ml/clean_backtest_models.py`, `clean_forecasts_by_date.py` — wired via `backtest-clean` / `forecast-clean` + spec 07/06.
- `scripts/ml/simulate_champion_strategies.py` — wired via `champion-simulate`.
- `scripts/ml/run_backtest_chronos2_enriched.py` — wired via `backtest-chronos2e` (and `backtest-all`).
- `scripts/ml/run_inventory_planning_pipeline.py`, `generate_customer_features_sql.py`, `db/*`, all `run_backtest_*`, all `compute_*`, all `normalize_*`/`load_*`, `tune_*`, `auto_tune.py`, `run_champion_*`, `train_*`, `pg_queue_worker.py`, `run_perf_analysis.py`, `audit_routes.py`, `check_fstring_sql.py`, `ingest_docs.py`, `generate_ai_insights.py`, `run_ai_fva_backtest.py`, `compute_service_level_actuals.py`, `update_lead_time_actuals.py`, etc. — all reference-confirmed via Makefile targets, job registry, or subprocess.
- No leftover duplicates of the 6 deleted ML scripts (`train_clustering_model.py`, `generate_clustering_features.py`, `update_seasonality_profiles.py`, `detect_seasonality.py`, `detect_drift.py`, `compute_demand_variability.py`) exist on disk — only stale `.pyc`/doc/allowlist mentions remain (those live outside this scope). The redirect targets (`compute_sku_features.py`, `run_cluster_pipeline.py`) are present and wired.
