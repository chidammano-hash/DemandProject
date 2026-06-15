# Refactor & Simplification Opportunities

A codebase-wide audit of the Supply Chain Command Center for refactor and simplification opportunities, organized by area. **This is analysis only — no code was changed.** Every finding is anchored to `file:line` and rated by **Impact** (High/Med/Low), **Effort** (S/M/L), and **Risk** (Low/Med/High). The quality bar throughout is CLAUDE.md's own rules.

Audited 2026-06-15 against `feat/unified-data-ingestion`. Coverage: api/ (39.8k LoC), common/ (36.7k), scripts/ (41k), frontend/src (116k), sql/ (9.1k), config/ (4.8k), tests/ (71.9k).

## Area reports

| # | Area | Headline findings |
|---|---|---|
| [01](01-api.md) | **api/** | Split `inv_planning_insights.py` (1818); shared error-handling decorator kills 47 try/except copies; 2 unauthenticated write endpoints; collapse per-algorithm tuning routers |
| [02](02-common-core-services.md) | **common/core + services** | Declarative job-spec table collapses ~25 runner duplicates; split `job_state.py` (1649) + `job_registry.py` (1410); extract DB persistence from `JobManager` |
| [03](03-common-ml-engines-ai.md) | **common/ml + engines + ai** | `exception_engine.py` has detectors duplicated pure-vs-inline (~320 dead lines); `tree_models.py` bypasses `model_registry`; split `backtest_framework.py` (1776); dedup agentic LLM loops |
| [04](04-scripts.md) | **scripts/** | Extract inventory SS math to `common/`; 396 `print()` + ~70 `parents[N]` bootstraps + bare fact reads; shared CLI/connection helper; split the 1300–1600 LoC scripts |
| [05](05-frontend.md) | **frontend/src** | 70+ raw `fetch(` bypass `fetchJson`; 72 ad-hoc formatters; 240 inline hex; 15+ tabs over the 600-line limit; dead (forbidden) `EChartContainer` path |
| [06](06-sql-config.md) | **sql/ + config/** | 6 duplicate migration prefixes + 1 unpadded break apply-order; `fact_safety_stock_targets` spread over 15 migrations; orphaned `elasticity_config.yaml` |
| [07](07-tests.md) | **tests/** | `app_client(pool)` fixture removes ~4 lines × ~1500 bodies; 716-line dead `row_builders.py`; parametrize the formula/strategy test families |

## Cross-cutting themes

These patterns recur in **every** area and are where shared effort pays off most:

1. **Oversized files (the 800/600-LoC split rules).** ~25 files exceed their limit: routers (`inv_planning_insights` 1818, `champion_experiments` 1283), services (`job_state` 1649, `job_registry` 1410), ML (`backtest_framework` 1776, `foundation_models` 1216, `ai_planner` 1149, `exception_engine` 1134, `comparison` 1117), scripts (`run_backtest` 1573, `generate_production_forecasts` 1541, `compute_safety_stock` 1412, `etl/load` 1359), 15+ frontend tabs, and 4 test files (`test_unified_model_tuning` 1621). The fix is the same everywhere: split by sub-feature into a package, re-export the public surface.

2. **Helper duplication that should be shared.** `_parse_json`/`_safe_float` (api), serialization coercers (services), 72 `fmt*` formatters + per-model color maps (frontend), `make_pool`/client boilerplate (~1500 test bodies), `make_blend_row` (ML champion ×40), CLI/path bootstrap (~80 scripts), inventory SS math imported script-to-script.

3. **Declarative-data over copy-paste.** Three god-areas re-declare the same shape N times: `JOB_TYPE_REGISTRY` + ~25 subprocess runners (services), `dq_engine` check dispatch (13 if/elif vs an unused registry), and per-algorithm tuning routers/panels (api + frontend). Each collapses to one generic path driven by a data table.

4. **Convention violations that are also refactors.** Bare `except Exception` (hundreds across api/common/scripts), 396 `print()` in scripts, ~70 `parents[N]` bootstraps, 70+ raw `fetch(` in frontend queries, 16 `: any` in queries, 240 inline hex, 5xx leaking exception text, direct model instantiation bypassing `model_registry`, hardcoded ML params, `date.today()` in `db_maintenance.py`. Each cleanup removes boilerplate *and* closes a documented rule.

5. **Dead code to delete.** `tests/api/row_builders.py` (716), `ApiTestHelper`, the forbidden `EChartContainer`/`ForecastTrendChart` path (363), `exception_engine` pure detectors (~320), `cache.cached`, `elasticity_config.yaml`, stale references to 6 deleted configs, possibly `control_tower` tests (frontend tab removed).

## Highest-leverage work (start here)

Ranked by impact-per-effort across the whole codebase:

| Rank | Opportunity | Area | Impact / Effort |
|---|---|---|---|
| 1 | Quick-win sweep: delete dead code, add `require_api_key` to the 2 unauth writes, fix `date.today()`, fix stale config refs, fix migration prefixes | all | High / S |
| 2 | `app_client(pool)` test fixture (removes ~4 lines × ~1500 test bodies) | tests | Very High / M |
| 3 | Declarative job-spec table → collapse ~25 runners + the 400-line registry | services | High / M |
| 4 | Shared `@db_endpoint` error decorator (kills 47 try/except + bare-except violations) | api | High / M |
| 5 | Route all frontend HTTP through `fetchJson` (70+ raw fetches) | frontend | High / M |
| 6 | Consolidate 72 formatters + 240 hex colors into `lib/` + `useChartColors` | frontend | High / M |
| 7 | Extract inventory SS math/loaders to `common/`; add shared script CLI/bootstrap helper | scripts | High / M |
| 8 | Fix `exception_engine` pure-vs-inline duplication + `tree_models` registry bypass | ml | High / M |
| 9 | Split the ~25 oversized files (do incrementally, behind unchanged tests) | all | Med / L |

## Suggested sequencing

- **Phase 1 — Quick wins (low risk, high signal).** Delete dead code; fix the 2 unauthenticated writes; sanitize 5xx leaks; fix `date.today()`, stale config refs, and migration prefix collisions; remove the forbidden `EChartContainer` path. Each is small and independently shippable.
- **Phase 2 — Shared helpers / dedup.** Land the shared pieces *before* splitting files so the splits inherit them: `@db_endpoint` decorator, `parse_db_json`/`to_float`/coercers in `sql_helpers`, `fetchJson` migration, `lib/formatters` + `useChartColors`, `app_client` test fixture, script CLI/bootstrap helper, `make_blend_row`.
- **Phase 3 — File splits.** With helpers in place, split the oversized routers/services/ml-modules/tabs/scripts by sub-feature into packages, re-exporting the public surface. Do one file per change, behind unchanged tests.
- **Phase 4 — Declarative refactors (highest risk).** Job-spec table + registry generation; `dq_engine` data-driven dispatch; converge per-algorithm tuning routers/panels; converge experiment-builder UIs. These touch core paths — ship each alone with strong test parity.

## How to use this

Each finding is a self-contained, TDD-able unit of work — pick by the Impact/Effort/Risk rating that fits the moment. Treat the per-area files as the working backlog; this README is the map. Re-run the audit after major refactors to track the oversized-file and rule-violation counts down.
