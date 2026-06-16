## sql/ · config/ · tests/ · root — Files Not Needed

_Read-only audit. SQL migrations are historical — flagged INVESTIGATE only, never SAFE. No file was deleted or edited; this is analysis only._

Audit method: every config name was grepped across `common/ api/ scripts/ frontend/ Makefile` (bare-name, because configs load dynamically via `load_config("name")`) and against `config/` for `_includes:` usage. Test imports (`scripts.*`, `common.*`) were resolved with a Python module-existence check. SQL duplicate prefixes were diffed by content/header. Stray dirs were checked for git-tracked status and last-commit recency.

---

### config/

#### SAFE
- _None._ Every one of the 41 YAML files under `config/` has at least one genuine code load. The lowest-reference files were individually verified as real `load_config(...)` / `open(...path...)` calls:
  - `config/ai/agent_autonomy.yaml` (1 ref) — loaded in `common/ai/policy_engine.py:76` `load_config("agent_autonomy")`.
  - `config/ai/ai_planner_fva_backtest_config.yaml` (1 ref) — `scripts/forecasting/run_ai_fva_backtest.py:588`.
  - `config/forecasting/champion_experiment_templates.yaml` (1 ref) — `api/routers/forecasting/champion_experiments.py:37`.
  - `config/forecasting/sensing_config.yaml` (1 ref) — `common/ml/sensing.py`.
  - `config/operations/exception_sla.yaml` (1 ref) — `common/engines/exception_engine.py:1061` `_lc("exception_sla")`.
  - `config/platform/db_maintenance_config.yaml` (1 ref) — `scripts/db/db_maintenance.py:33`.

#### PROBABLE
- _None._

#### INVESTIGATE
- _None._

> Deleted-config recheck (must NOT reappear): `clustering_config.yaml`, `model_competition.yaml`, `lgbm_tuning_config.yaml`, `production_forecast_config.yaml`, `backtest_sampling_config.yaml`, `algorithm_config.yaml` — **all confirmed absent.** None were recreated. Good.

---

### sql/

146 `.sql` files. Migrations are applied by sequence/glob by a runner, not imported — so "no code reference" is **not** grounds for deletion. Nothing is flagged SAFE.

#### INVESTIGATE only
- _None — no true duplicates, no empty/0-byte files, no migrations for fully-removed features found._

Findings that look like duplicates but are NOT (verified distinct features sharing a numeric prefix — a numbering collision, not a duplicate file):
- `039_create_ai_call_log.sql` (33 L) vs `039_create_production_forecast.sql` (44 L) — different tables (AI call log vs `fact_production_forecast`).
- `041_add_source_model_id.sql` (9 L) vs `041_create_replenishment_plan.sql` (122 L) — distinct.
- `090_create_dim_sourcing.sql` (27 L) vs `090_create_integration_job.sql` (43 L) — distinct.
- `091_add_integration_job_diff_cols.sql` (8 L) vs `091_create_fact_purchase_orders.sql` (66 L) — distinct.
- `092_create_integration_chain.sql` (50 L) vs `092_mv_supplier_po_performance.sql` (71 L) — distinct.
- `095_create_dq_corrections.sql` (31 L) vs `095_create_lgbm_tuning.sql` (111 L) — distinct.

  → Recommendation: do **not** delete any of these. The collisions are cosmetic. If the migration runner depends on a strict total ordering, the safe fix is to *renumber* (rename to a free higher prefix) — not delete — and only if the runner actually breaks on collisions. Treat as a numbering-hygiene note, not a dead-file candidate.

Other things that look superseded but are valid forward migrations (KEEP):
- `143_add_otif_drop_old_supplier_mv.sql` — drops the old `mv_supplier_performance` and adds OTIF to `mv_supplier_po_performance`. This is the migration that *performs* the supersession; it must run.
- `188_create_integration_job_unified.sql` — a unified read view spanning `integration_job` (sql/090) + `job_history` (sql/020); not a replacement of either.
- `171_drop_empty_future_partitions.sql` (39 L) / `172_drop_unused_indexes.sql` (48 L) — operational migrations, non-empty.

> Renumber recheck: the prior `88_backtest_run.sql` → `089_backtest_run.sql` rename left **no** leftover `88_*` file. Clean.

---

### tests/

#### SAFE
- _None._

#### PROBABLE
- _None._

#### INVESTIGATE
- _None._ Every test's subject resolves:
  - All `scripts.*` imports across `tests/**/*.py` resolve to an existing module/package (verified by script).
  - All `common.*` imports across `tests/**/*.py` resolve (verified by script).

Rechecks (all good):
- Previously-deleted dead tests `test_clustering_perf.py`, `test_demand_variability.py`, `test_seasonality.py` (unit) — **none reappeared in `tests/unit/`.**
- `tests/api/test_seasonality.py` **does** exist but is a *different* file — it tests seasonality columns exposed via the generic domain API (live httpx ASGITransport test), not the deleted unit-level seasonality detector. KEEP.
- `tests/api/row_builders.py` — confirmed **absent**, and no remaining test references `row_builders`. Already fully removed. Good.
- `tests/scale/` — `test_customer_analytics_scale.py`, `test_inv_planning_scale.py`, `conftest.py`, `README.md` all present and intact (excluded from default run via `-m 'not scale'`, run via `make scale-test`). KEEP.

Note (not a deletion candidate, just FYI): `tests/Automated_tests/` (24 tracked files, cycles 1–5 + `_harness/{capture.mjs,improvement-loop.mjs}` + per-cycle `screens/`) is the committed ux-loop hardening harness/output (commit 2026-06-14 "UX hardening cycles 1-5 + reset loop harness"). It is screenshot/critique artifact data, not test code. Retained intentionally per its commit; flag for the owner only if screenshot blobs are bloating the repo — not dead.

---

### root & stray

#### SAFE (disk-only scratch; already git-ignored & untracked — nothing to remove from version control)
- `catboost_info/` — 320K, **0 tracked files**, gitignored ("ML training artifacts"). Regenerated per CatBoost run. Safe to `rm -rf` locally; no repo impact.
- `lightning_logs/` — 736K, **0 tracked files**, gitignored. PyTorch-Lightning run logs. Safe to `rm -rf` locally; no repo impact.

#### PROBABLE
- _None._

#### INVESTIGATE (owner-judgment, retained-intentionally docs/backlogs — do NOT auto-delete)
- `analysis_volatility/` (2.0M, 4 tracked: `Demand_Sensing_Volatility_Review.pdf`, `build_report.py`, `scored_dfus_sample.csv`, `volatility_score.py`) — a one-off analysis committed **2026-06-14** ("analysis(volatility): demand-sensing volatility review + scoring script"). **Not referenced** by any code/Makefile/docs (grep clean). It is a standalone deliverable, not wired into the pipeline. Candidate to relocate into `archive/` or `docs/` if it is a finished artifact, but it is recent and may still be in active use — confirm with owner before moving. The 2.0M is dominated by the tracked PDF.
- `refactor/` (8 tracked `.md` files: `01-api`…`07-tests`, `README`) — the prior audit's refactor backlog, last touched **2026-06-15**. Per the task brief this may be retained intentionally. Noted, not flagged for deletion.
- `dataingestion/` (29 tracked files: `MASTER_PLAN.md`, `STATUS.md`, `README.md`, `UserStory1–21*.md`) — active data-ingestion-streamlining spec/backlog; `STATUS.md` shows all 21 stories shipped, last commit **2026-06-15**. Living planning docs, **not referenced by code** (expected for docs). Once the feature is fully merged these could move under `docs/`, but they are current — owner decision.

#### NOT stray / KEEP (verified normal)
- Root files `CLAUDE.md`, `README.md`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `.editorconfig`, `.pre-commit-config.yaml` — all standard project files.
- `.agents/` (1 tracked), `.codex/` (2 tracked), `.github/`, `.vscode/` — config dirs, normal.
- `.pytest_cache/`, `.ruff_cache/`, `.venv/` — all **0 tracked** (gitignored). No repo impact.
- `archive/` — gitignored ("Non-project materials") and CLAUDE.md says do not touch. Noted as archived; not flagged.
- No `*.log`, `*.tmp`, `*.bak`, `*.old`, `.DS_Store`, or `.ipynb` files are tracked anywhere in the repo. Clean.

---

### NOT dead (checked, keep) — looked orphaned but are used
- **All 41 `config/` YAMLs** — including `shared_constants.yaml` (8 `_includes` references) and the low-ref ones listed under config/SAFE, each verified as a real load.
- **`tests/api/test_seasonality.py`** — tests a live domain API surface, distinct from the deleted unit test of the same stem.
- **`tests/scale/*`** — gated, not orphaned.
- **SQL collision-prefix pairs (039/041/090/091/092/095)** — each pair is two genuinely different feature migrations; both are needed.
- **`143`, `188`, `171`, `172` SQL** — forward/operational migrations, must run.
- **`lgbm_tuning` SQL (095)** — its old *config* was deleted, but the table is still referenced by `api/main.py`, `cluster_experiments.py`, `production_forecast.py`. Keep.
