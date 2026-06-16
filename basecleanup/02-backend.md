## Backend (api/, common/) — Files Not Needed

_Read-only audit. Each candidate verified for zero references. Confidence: SAFE = zero static+dynamic refs; PROBABLE = zero static, minor dynamic doubt; INVESTIGATE = needs human check (dynamic loading, registry, config)._

Scope audited: `api/` (88 .py files) and `common/` (all subdirs, ~110 .py files).

Method: for each module, searched `api/ common/ scripts/ tests/` for the dotted module path and `from <pkg> import <name>` forms, plus `config/*.yaml` dotted-path references and dynamic loaders (`importlib`, `__import__`, `getattr`). A module is only SAFE with zero production importers AND zero dynamic/config wiring. Router files were cross-checked against `api/main.py` `include_router(...)` and against sub-package `__init__.py` aggregation (`tuning/`, `customer_analytics/`).

### SAFE to delete

- `common/services/query_tracker.py` (78 LoC) — `QueryTracker` / `get_tracker()` have ZERO importers anywhere; no test file; no config reference. Predates the current refactor (last touched in the `restructure`/`final` commits). evidence: only non-self hit across `api/ common/ scripts/ tests/` is `scripts/ai_checks/allowlists/rule3_bare_except.txt` (a lint allowlist line, not an importer). No `importlib`/`getattr` path constructs it.

### PROBABLE

These six modules have ZERO production importers (`api/`, `common/`, `scripts/`) — each is imported only by its own unit test. They are orphaned features (built, tested, but never wired into any runtime path). Removing the module means also removing its paired test. PROBABLE rather than SAFE because four of them have an associated config YAML, indicating an intended-but-unwired feature that a human should confirm is abandoned. If kept, they are dead weight; if removed, remove module + test together.

- `common/ai/drift.py` (207 LoC) — PSI & rolling-WAPE drift math. Only importer: `tests/unit/test_drift.py` (90 LoC). No production caller. Note: `config/operations/exception_config.yaml` mentions "model_drift" but that is the exception-engine's own drift check, not this module (no dotted-path reference). evidence: grep for `common.ai.drift` in `api/ common/ scripts/` → empty.
- `common/ai/dry_run.py` (156 LoC) — preview+confirm pipeline. Only importer: `tests/unit/test_dry_run.py` (118 LoC). evidence: grep `common.ai.dry_run` in production code → empty.
- `common/ai/envelope.py` (98 LoC) — multimodal response envelope. Only importer: `tests/unit/test_envelope.py` (96 LoC). evidence: grep `common.ai.envelope` in production code → empty.
- `common/ai/policy_engine.py` (165 LoC) — `ActionContext` / `evaluate`. Only importer: `tests/unit/test_policy_engine.py` (128 LoC). `config/ai/agent_autonomy.yaml` contains a comment "See common/ai/policy_engine.py" but no code loads it. evidence: grep `common.ai.policy_engine` in production code → empty; YAML hit is a comment only.
- `common/ml/cold_start_neighbors.py` (165 LoC) — cold-start neighbor lookup. Only importer: `tests/unit/test_cold_start_neighbors.py` (73 LoC). evidence: grep `cold_start_neighbors` in production code → empty. (Note: CLAUDE.md documents cold-start routing in `forecast_pipeline_config.yaml` `production_forecast`, but that logic lives elsewhere; this module is not the implementation referenced.)
- `common/ml/sensing.py` (102 LoC) — `blend_forecasts` horizon-weighted blend. Only importer: `tests/unit/test_sensing.py` (79 LoC). The demand-sensing feature in production (`scripts/forecasting/compute_blended_forecast.py`, `api/routers/forecasting/blended_forecast.py`) implements blending inline and does NOT import this module — it appears superseded. `config/forecasting/sensing_config.yaml` references it only in a comment. evidence: grep for the symbol `blend_forecasts` / `common.ml.sensing` in production code → empty (all "sensing" hits are the concept, not the module).

### INVESTIGATE

- `common/ml/expert_panel/__init__.py` (1 LoC) — the `expert_panel` package `__init__` is empty (does NOT aggregate the sub-modules the way `champion/`, `tuning/`, `customer_analytics/` do). The 18 `expert_panel/*.py` modules are all imported individually (mostly by `scripts/ml/run_adv_expert_panel.py` and each other), so they are NOT dead — but the empty `__init__` plus single-script consumption means the whole `expert_panel` subsystem is exercised by one script + tests only. Human should confirm `run_adv_expert_panel.py` is still a live entrypoint; if that script were ever retired, ~15 of these modules would become orphaned in one stroke. Not flagging individual files as dead now (they have live importers).

### NOT dead (checked, keep)

- All 88 router files under `api/routers/` are live. Every top-level router is imported in `api/main.py`; the apparently-unimported ones are sub-package routers aggregated through their package `__init__.py`:
  - `api/routers/inventory/inventory_main.py` — re-exported by `inventory/__init__.py`, mounted as `inventory.router`.
  - `api/routers/intelligence/customer_analytics/{geo,segments,ranking,lifecycle,kpis,_helpers}.py` — aggregated in `customer_analytics/__init__.py`, mounted once.
  - `api/routers/forecasting/tuning/{list,detail,create,compare,cluster,lag,logs,month,promote,promote_results,cancel_delete,templates,promotions,_helpers}.py` — aggregated in `tuning/__init__.py`, mounted at `/model-tuning`.
- `api/core.py` — re-exports some names from `api.pool`/`api.llm` "for backward compatibility" but is NOT a shim: it contains live definitions (`_get_pool`, `_get_async_pool`, etc.) and is the canonical import target for ~89 importers. Keep.
- No `_ShimModule` / re-export-only shim files were found in `api/` or `common/`. The "shim"/"backward-compat" string hits are all inline comments inside substantial, actively-imported modules (`common/ml/clustering/features.py` legacy aliases, `common/inventory/safety_stock.py`, `common/core/config_models.py`), not standalone shim modules.
- No dead test files: every `tests/` file's subject module under `common`/`api` resolves to an existing module (scripted check, zero unresolved imports).
- The count-1 champion sub-modules (`bandit`, `blend`, `meta`, `routing`, `segment`) and `sku_features/classifiers.py` are live — imported by their package `__init__.py` which is itself widely consumed.
- `common/services/job_scheduler.py` — single importer (`job_registry.py`) but `job_registry` is one of the most-imported modules (30 importers); live.
- `common/ml/expert_panel/dl_baselines.py` and `report.py` — single importer each, but that importer (`scripts/ml/run_adv_expert_panel.py`) is a live script entrypoint; live (see INVESTIGATE note above for the systemic risk).
