## API — Refactor Opportunities

_Scope: `api/` (routers, core.py, main.py, auth.py, pool.py, llm.py, domains.py). Read-only audit — no code changed._

### Quick wins
- `_parse_json` is byte-identical in 3+ routers — `routers/forecasting/champion_experiments.py:92`, `routers/forecasting/cluster_experiments.py:118`, `routers/forecasting/tuning/_helpers.py:133` (the last silently diverges: no try/except) — promote one to `common/core/sql_helpers.py`.
- `_safe_float` redefined 3× — `routers/inventory/inv_planning_insights.py:41`, `routers/forecasting/accuracy_budget.py:32`, `routers/forecasting/sku_features.py:164`; `core.py:118` already has `_f` doing the same job. Consolidate.
- `_experiment_row_to_dict` near-duplicated — `routers/forecasting/champion_experiments.py:126` and `routers/forecasting/cluster_experiments.py:130`.
- Local `_row_to_dict` defined against the rule — `routers/forecasting/backtest_management.py:102` (use `row_to_dict_from_cols`).
- 5xx interpolates exception text (rule violation) — `routers/intelligence/intel.py:184` (`detail=f"AI generation failed: {e}"`), `routers/forecasting/shap.py:799`, `routers/forecasting/model_tuning.py:659`, `routers/forecasting/lgbm_tuning.py:627`.
- Unauthenticated write endpoints — `routers/platform/data_quality.py:177,209,377` (`/run`, `/fix/apply`, `/fix`) and `routers/forecasting/lgbm_tuning.py:190,217,578` lack `Depends(require_api_key)`.
- 47 bare `except Exception:` with no `# noqa: BLE001` — `cluster_experiments.py` (18), `champion_experiments.py` (16), `backtest_management.py` (13).
- Inconsistent routing style: `champion_experiments.py` uses `prefix` + short paths; `inv_planning_insights.py` / `supply.py` / `lgbm_tuning.py` / `model_tuning.py` repeat the full prefix in every `@router.get(...)` decorator.

### Ranked opportunities

1. **Collapse per-algorithm tuning routers into the existing `tuning/` package**
   - Files: `routers/forecasting/model_tuning.py` (892), `routers/forecasting/lgbm_tuning.py` (724), `routers/forecasting/tuning/` (15 sub-routers, mounted `prefix="/model-tuning"` in `main.py:362`)
   - Problem: `model_tuning.py` hosts catboost-tuning + xgboost-tuning CRUD (mirrored halves) and `lgbm_tuning.py` hosts the lgbm equivalent. Same run/compare/clusters/promote CRUD shape per algorithm — the generalized version already exists in `tuning/`. Two 800+ LoC routers persist as legacy copies; `model_tuning.py` also exceeds the 800-LoC split rule.
   - Proposed change: Parameterize the `tuning/` package handlers by `model_id`/table, retire the per-algorithm bodies, mount one prefixed router. Move `_parse_json` to the canonical helper.
   - Impact: High · Effort: L · Risk: Med

2. **Extract the per-endpoint `try/except HTTPException/except Exception` wrapper into a shared decorator**
   - Files: `champion_experiments.py` (16×), `cluster_experiments.py` (18×), `backtest_management.py` (13×) — e.g. `champion_experiments.py:228-232`, `:591-595`
   - Problem: Every handler repeats `except HTTPException: raise` / `except Exception: logger.exception(...); raise HTTPException(500, "<msg>") from None`. 47 copies, all violating "no bare `except Exception` without `# noqa: BLE001`", pure boilerplate.
   - Proposed change: Add a `@db_endpoint("<failure verb-phrase>")` decorator (or `with db_guard(...)`) in `common/core/` that catches `psycopg.Error`/`ValueError`, logs via `logger.exception`, re-raises a sanitized 500, passing `HTTPException` through.
   - Impact: High · Effort: M · Risk: Low

3. **Split `inv_planning_insights.py` (1818 LoC) by insight sub-feature**
   - Files: `routers/inventory/inv_planning_insights.py`
   - Problem: >2× the 800-LoC limit. 11 unrelated GET endpoints + 8 module-level helpers.
   - Proposed change: `routers/inventory/insights/` with sub-modules grouped by theme (action/exceptions, dashboards/scorecards, financial/optimization), each a prefixed `APIRouter`; shared helpers to `_helpers.py`.
   - Impact: High · Effort: M · Risk: Low

4. **Add `require_api_key` to unauthenticated write endpoints**
   - Files: `routers/platform/data_quality.py:177,209,377`; `routers/forecasting/lgbm_tuning.py:190,217,578`
   - Problem: `POST /data-quality/run`, `/fix/apply`, `/fix` mutate data with no auth; `lgbm_tuning` create/update/promote writes are unguarded while sibling experiment routers guard theirs — security gap + consistency break.
   - Proposed change: Add `dependencies=[Depends(require_api_key)]`. Add an API test asserting 401 without key.
   - Impact: High · Effort: S · Risk: Low

5. **Centralize JSON/float coercion helpers in `common/core/sql_helpers.py`**
   - Files: `_parse_json` at `champion_experiments.py:92`, `cluster_experiments.py:118`, `model_tuning.py:102`, `lgbm_tuning.py:145` & `:301`, `tuning/_helpers.py:133`; `_safe_float` at `inv_planning_insights.py:41`, `accuracy_budget.py:32`, `sku_features.py:164`
   - Problem: Same parse logic copy-pasted 6+ times; the `tuning/_helpers.py` copy omits try/except (latent bug). `_safe_float` overlaps `core.py:_f`.
   - Proposed change: One `parse_db_json()` + one `to_float(v, decimals=None)` in `sql_helpers.py`; delete local copies.
   - Impact: Med · Effort: S · Risk: Low

6. **Standardize routing convention: `APIRouter(prefix=...)` + short decorator paths**
   - Files: `inv_planning_insights.py` (all 11 paths repeat `/inv-planning/`), `supply.py`, `lgbm_tuning.py`, `model_tuning.py`; vs compliant `champion_experiments.py`
   - Proposed change: Move repeated segment into `APIRouter(prefix=...)`, shorten decorators. Verify with `make audit-routers` + Vite proxy.
   - Impact: Med · Effort: M · Risk: Med (path regressions — needs test coverage)

7. **Extract the "fetch experiment or 404" boilerplate**
   - Files: `champion_experiments.py:572,620,664,707,841,951,1016,1063,1116` (10×), mirrored in `cluster_experiments.py:599,733,757`
   - Proposed change: A `_load_experiment_or_404(cur, experiment_id) -> dict` helper per table.
   - Impact: Med · Effort: S · Risk: Low

8. **Split `champion_experiments.py` (1283) and `backtest_management.py` (1061)**
   - Proposed change: Mirror the `tuning/` package layout — `champion/` with `list.py`/`detail.py`/`compare.py`/`mutations.py` + `_helpers.py`. Do after #2/#5/#7.
   - Impact: Med · Effort: M · Risk: Low

9. **Split `supply.py` (1024) into open-POs / planned-orders / purchase-orders**
   - Files: `routers/operations/supply.py` (request models at :670-693)
   - Proposed change: `routers/operations/supply/` with `open_pos.py`, `planned_orders.py`, `purchase_orders.py`; co-locate each family's Pydantic models.
   - Impact: Med · Effort: M · Risk: Low

10. **Sanitize the genuine 5xx exception-text leaks**
    - Files: `intel.py:184`, `shap.py:799`, `model_tuning.py:659`, `lgbm_tuning.py:627`
    - Proposed change: `logger.exception(...)` then `raise HTTPException(5xx, "<short verb-phrase>")`. Absorbed by #2's decorator.
    - Impact: Med · Effort: S · Risk: Low

**Not problems:** all `except Exception` in `pool.py` (8) and `main.py` (7) carry `# noqa: BLE001` with justifications. `auth_router.py` writes correctly use their own auth. Ruff reports zero unused imports in `api/`. No `Depends(_get_pool)` misuse in `inv_planning_*`.
