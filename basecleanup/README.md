# Base Cleanup — Files Not Needed After Refactoring

> **STATUS — actioned.** All SAFE + PROBABLE items below were deleted (with their
> paired tests), the broken `run_inventory_backtest` job path was **fixed** (not
> deleted), and the retired `EChartContainer` was **migrated to recharts then
> removed**. Untracked ML artifact dirs (`catboost_info/`, `lightning_logs/`) were
> removed locally. **Deferred by design:** the SQL prefix-collision *renumbering*
> (historical migrations — risky, no migration tracker) and the owner-decision
> dirs (`analysis_volatility/`, `refactor/`, `dataingestion/`). The `expert_panel`
> subsystem was confirmed live and kept. All deletions landed behind a green
> suite (backend 4313, frontend 1144 — the 1 frontend failure is the pre-existing
> DemandReferencePanel test).

A codebase-wide, **read-only** audit of files that appear dead, orphaned, or superseded — produced after the `refactor/codebase-cleanup` work (PR #2). Four parallel agents swept disjoint areas; every candidate was checked for static **and** dynamic references, then classified by confidence. **Nothing here has been deleted** — these are suggestions for a human to action.

## Confidence levels
- **SAFE** — zero static importers AND zero dynamic/config/Makefile/subprocess references. Verified. Deleting is low-risk (remove the file + any paired test).
- **PROBABLE** — zero production callers; imported only by its own test (orphaned feature). Safe to drop *with* its test, but confirm the feature is truly abandoned.
- **INVESTIGATE** — needs a human decision: dynamically loaded, registry-wired, retired-but-still-referenced, or historical (SQL migrations).

> ⚠️ **SQL migrations are historical** — "no code references it" does NOT make a migration deletable. They are flagged INVESTIGATE only, never SAFE.

## Detail files
- [02-backend.md](02-backend.md) — `api/`, `common/`
- [03-scripts.md](03-scripts.md) — `scripts/`
- [04-frontend.md](04-frontend.md) — `frontend/src/`
- [05-sql-config-tests-root.md](05-sql-config-tests-root.md) — `sql/`, `config/`, `tests/`, repo root

---

## SAFE to delete (verified zero references)

| File | LoC | Area | Note |
|---|---|---|---|
| `common/services/query_tracker.py` | 78 | backend | zero importers; only a lint-allowlist line mentions it |
| `scripts/ml/expert_panel_main.py` | 27 | scripts | orphaned fork-safety wrapper; its documented `python -m` path doesn't even resolve |
| `frontend/src/components/AlertPanel.tsx` (+ test) | ~ | frontend | zero non-test importers |
| `frontend/src/hooks/useTabVisibility.ts` (+ test) | ~ | frontend | zero non-test importers |
| `frontend/src/tabs/inventory/ItemDetailPanel.tsx` | ~ | frontend | zero importers (no test) |
| `frontend/src/tabs/model-tuning/PipelineConfigPanel.tsx` (+ test) | ~ | frontend | zero non-test importers |

All six independently re-verified by the orchestrator (grep, excluding each file's own test) — confirmed zero references.

## PROBABLE (orphaned feature + its test — confirm abandonment first)

- **backend:** `common/ai/drift.py`, `common/ai/dry_run.py`, `common/ai/envelope.py`, `common/ai/policy_engine.py`, `common/ml/cold_start_neighbors.py`, `common/ml/sensing.py` — each imported only by its own unit test, zero production callers. `sensing.py` is superseded by inline blending in `compute_blended_forecast.py`.
- **scripts:** `scripts/etl/trim_input_files.py` (244) — unwired one-off; `scripts/ml/label_clusters.py` (192) — thin CLI superseded by the clustering package (pipeline imports the library directly; its documented `cluster-label` Make target doesn't exist).
- **frontend:** `frontend/src/api/queries/expsys.ts` (62) — barrel-re-exported but zero consumers; backend `/expsys/*` routes exist but no UI calls them. Deleting also means removing its line from `api/queries/index.ts`.

## INVESTIGATE (needs a decision, not a delete)

- **`EChartContainer.tsx` + `ForecastTrendChart.tsx` (+ 2 tests)** — `EChartContainer` is RETIRED per CLAUDE.md/MEMORY, but still wired: `ForecastTrendChart` → `EChartContainer`, and `AggregateAnalysisTab` still renders `ForecastTrendChart`. **Must migrate AggregateAnalysisTab to recharts first, then delete both together.** This is the deferred **Phase 2e chart** work.
- **`scripts/inventory/run_inventory_backtest.py` (397)** — registered in the job registry but **unreachable**: `job_state.py` shells out to a stale path `scripts/run_inventory_backtest.py` (missing the `inventory/` dir). Same class of bug as the `run_inventory_planning_pipeline` stale paths fixed in PR #2 — **fix the path or retire the job**, don't just delete the file.
- **`common/ml/expert_panel/` subsystem (~18 modules)** — exercised by a single script + tests, with an empty package `__init__`. Not dead today, but a systemic risk if that script is retired. Decide whether the expert-panel feature is still in scope.
- **`sql/` shared-prefix collisions** (039/041/090/091/092/095) — distinct features sharing a numeric prefix. A numbering-**hygiene** note (renumber via a tracked migration, never delete). The earlier `88_`→`089_` rename left no leftovers.

## Local-only artifacts (untracked, gitignored — `rm` locally, not a repo change)
- `catboost_info/` (~320K) and `lightning_logs/` (~736K) — ML training artifacts. Safe to delete from your working copy; they are not in version control.

## Owner-decision directories (recent, likely intentional — left as-is)
- `analysis_volatility/` (~2.0M, one-off analysis, unreferenced), `refactor/` (the prior audit backlog), `dataingestion/`, `basecleanup/` (this dir). None auto-flagged.
- `archive/reference/` — explicitly off-limits per CLAUDE.md; not evaluated.

---

## Confirmed clean (no action)
- **config/** — all 41 YAMLs have a verified `load_config(...)`/loader; the 6 previously-deleted configs have not reappeared.
- **sql/** — no true duplicates, no empty files, no removed-feature migrations.
- **tests/** — every `scripts.*`/`common.*` test import resolves; the 3 deleted unit tests and `tests/api/row_builders.py` stayed gone; all 88 routers are live (sub-package routers aggregated via `__init__`).
- No backward-compat `_ShimModule` files remain in `common/`.

## Recommended order if actioning
1. **SAFE** (6 files + 3 tests) — quick, low-risk; run `make test-all` after.
2. **PROBABLE** — confirm each feature is abandoned with the team, then remove file + paired test together (per the "removed feature → remove its tests" rule).
3. **INVESTIGATE** — the chart removal and the `run_inventory_backtest` path are real follow-ups (the former = Phase 2e); the SQL renumbering is hygiene; the expert-panel question is a scope decision.

**Estimated removable now (SAFE + PROBABLE):** ~10 source files + ~8 tests, on the order of ~1,500 LoC, all behind a green suite once their paired tests are removed.
