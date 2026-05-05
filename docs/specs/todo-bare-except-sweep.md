# Bare `except Exception` sweep — remaining work

_Created as part of Stream B Gen-4 roadmap cleanup (2026-04-23)._

## Status

- **Fixed so far (29 sites):** api/ routers (clusters, collaboration, dashboard, supply, auth_router, users, inv_planning_policy, inv_planning_projection, inventory/demand_history, intelligence/ai_planner, intelligence/intel, shap), api/llm.py, common/engines/dq_engine.py.
- **Baseline failures we also fixed in passing:** `tests/api/test_clustering_scenario.py::test_clustering_defaults` + `…_has_valid_ranges` were 500ing because `clusters.py` referenced an undefined `logger`. Narrowed the surrounding except + kept the existing module-level `log` logger. Now pass.

## Remaining hot spots (pragmatic follow-ups)

Run `grep -rn "except Exception" api/ common/ scripts/ | grep -v "# noqa"` to see the current tally (≈ 190 at time of writing).

Highest-value routers still to narrow (each has 3+ bare `except`s):

- `api/routers/forecasting/unified_model_tuning.py` — 30+ sites. Mostly DB reads behind feature-flag tables. Safe to narrow to `psycopg.Error` but high surface area.
- `api/routers/forecasting/cluster_experiments.py` — 5 sites.
- `api/routers/inventory/inv_planning_algorithm_comparison.py` — several sites.
- `api/routers/operations/events.py`, `financial_plan.py`, `service_level.py` — imperative `await require_api_key()` routers; medium priority.
- `api/routers/intelligence/storyboard.py` — 4 sites.
- `api/routers/forecasting/consensus_plan.py` — 4 sites.

## Engines / services

- `common/engines/exception_engine.py` — **owned by Stream A**, do not touch.
- `common/services/*` — several bare-excepts around rate limiter + cache fallbacks. Most are intentional (best-effort cache); mark with `# noqa: BLE001` rather than narrowing.

## Scripts

The `scripts/` tree has ~70 bare-excepts, mostly around optional feature toggles (GPU / cupy imports, optional libraries). These are typically correct "try optional dep" patterns — narrow to `ImportError` where applicable, leave the rest alone.

## Recommended follow-up commits

1. Narrow `unified_model_tuning.py` in a dedicated commit (separate PR, large diff, single router).
2. One commit per forecast router (cluster_experiments, sampled_backtest, etc).
3. Scripts: narrow `ImportError`-only sites; leave the rest.

## Policy reminder

- Only keep `except Exception` with an explicit `# noqa: BLE001 — <reason>` comment.
- Always log with `logger.exception(...)` inside a broad catch.
- New code must not reintroduce bare excepts — `ruff` rule `BLE001` is enforced.
