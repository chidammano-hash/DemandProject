## tests/ — Refactor Opportunities

_Scope: backend `tests/` (api, unit, scale). Goal: reduce maintenance burden WITHOUT weakening coverage — no proposal deletes assertions. Read-only audit — no code changed._

### Quick wins
- `tests/api/row_builders.py` (716 lines, 22 builder functions) has **zero importers** — fully dead code; delete or actually adopt.
- `ApiTestHelper` class in `tests/api/conftest.py:190-224` has **zero uses** — dead helper.
- `tests/api/test_shap.py:57` `_make_pool()` and `:216` `_make_pool_with_clusters()` re-implement the conftest factory inline — replace with `make_pool`.
- `tests/api/test_ai_planner_api.py:14`, `tests/api/test_analysis.py:22` define local `_make_pool(...)` duplicating conftest — consolidate.
- `tests/api/conftest.py:140-156` `mock_pool` fixture duplicates `make_pool()` logic verbatim — collapse one onto the other.
- `tests/unit/test_safety_stock.py:62-98` six `test_z_score_*` differ only by service level → one `@pytest.mark.parametrize`.
- `tests/unit/test_etl_helpers.py:284` and `tests/unit/test_load_dataset_postgres.py:22` define identical `_cursor_with_rowcounts` helpers → share in a unit conftest.
- `tests/unit/test_safety_stock.py` (105 tests, 0 parametrize) and `tests/unit/test_champion_strategies.py` (80 tests, 0 parametrize) — large parametrize headroom.

### Ranked opportunities

1. **Adopt an `app_client(pool)` fixture to kill per-test client boilerplate**
   - Files: ~94 files under `tests/api/` (e.g. `test_fva.py:25-28`); fixture infra at `conftest.py:170-187`
   - Problem: Nearly every API test repeats the 4-line block — `with patch("api.core._get_pool", return_value=pool): from api.main import app; transport = ASGITransport(app=app); async with httpx.AsyncClient(...) as client:`. ~1161 inline `from api.main import app`, 94 inline `ASGITransport(app=...)`. An `async_client` fixture exists but is used by ~1 file.
   - Proposed change: A parametrizable `app_client(pool)` factory fixture so tests do `pool,conn,cursor = make_pool(...); async with app_client(pool) as client:`. Migrate file-by-file.
   - Impact: Very high (removes ~4 lines × ~1500 bodies, one patch-target/transport site) · Effort: H · Risk: L-M (tests patching extra symbols must compose with added `patch` contexts)

2. **Replace the 16 hand-rolled `cursor.return_value.__enter__` chains with `make_pool`**
   - Files: `test_shap.py`, `test_dashboard.py`, `test_inventory.py`, `test_analysis.py`, `test_distinct.py`, `test_model_tuning.py`, `test_domains.py`, `test_lgbm_tuning.py`, `test_jobs.py`, `test_forecast_accuracy.py`, `test_clustering_scenario.py`, `test_tuning_chat.py`, `test_inventory_backtest.py`, `test_clusters.py`, `test_seasonality.py`, `test_ai_planner_api.py`
   - Problem: These rebuild the exact plumbing `make_pool` encapsulates — the convention CLAUDE.md forbids.
   - Proposed change: Delete local `_make_pool`/inline chains; import `make_pool`/`make_async_pool`.
   - Impact: High · Effort: M · Risk: Low

3. **Delete dead `row_builders.py` (716 lines) or genuinely adopt it**
   - Problem: 22 builders, zero importers, while tests hand-build mock tuples inline.
   - Proposed change: Either migrate the worst brittle-tuple offenders to it (see #9) or delete. Don't keep 716 unreferenced lines.
   - Impact: High (−716 dead lines, or fixes #9) · Effort: Low (delete) / M (adopt) · Risk: Low

4. **Parametrize repeated cross-strategy bodies in `test_champion_strategies.py`** (1648 lines, 80 tests, 0 parametrize)
   - Problem: Same names recur per strategy class — `test_basic_winner_selection`, `test_no_data_leak`, `test_exec_lag_1_excludes_most_recent_prior_month` — near-identical bodies.
   - Proposed change: `@pytest.mark.parametrize("strategy_id", [...])` for the invariants every strategy must satisfy; keep strategy-specific tests separate.
   - Impact: High · Effort: M · Risk: L-M (confirm each strategy shares the contract)

5. **Parametrize the formula-variant tests in `test_safety_stock.py`** (1215 lines, 105 tests)
   - Problem: Long runs of single-assertion variants (`test_z_score_85/90/95/...`, `test_ss_lt_*`, `test_rop_*`, boundary cases).
   - Proposed change: `@pytest.mark.parametrize("inputs, expected", [...])` per family; keep "known_result" anchors explicit.
   - Impact: High (105→~40 functions) · Effort: M · Risk: Low (pure functions)

6. **Collapse `mock_pool` fixture and `make_pool` factory into one** (`conftest.py:33-81` vs `:140-156`)
   - Problem: `mock_pool` duplicates `make_pool()`'s body with different defaults (`fetchone=None` vs `(0,)`, `description=[]` vs `[("col",)]`) — two divergent default-cursor definitions. 22 files use the fixture, 75 the function.
   - Proposed change: Make `mock_pool` call `make_pool()` internally; reconcile defaults deliberately.
   - Impact: Med-High · Effort: Low · Risk: M (default changes could shift behavior in 22 files — verify each)

7. **Reduce SQL-substring coupling in router tests**
   - Files: `test_distinct.py:295-349` (`"dim_sku" in executed_sql`, `"ILIKE" in ...`), plus `test_accuracy.py`, `test_data_quality.py`, `test_dashboard.py`, `test_customer_analytics.py`, `test_unified_model_tuning.py`, `test_ai_fva_backtest_*.py`
   - Problem: Asserting raw SQL substrings via `cursor.execute.call_args[0][0]` breaks on cosmetic query rewrites.
   - Proposed change: Where the goal is "table X was queried," assert on HTTP response shape/values; keep one focused assert (documented) only where SQL routing genuinely matters (e.g. the CA `mv_customer_activity_monthly` fast path).
   - Impact: Med · Effort: M · Risk: M (must not weaken load-bearing routing asserts)

8. **De-duplicate unit-test cursor/conn helpers into `tests/unit/conftest.py`**
   - Files: `test_etl_helpers.py:284` & `test_load_dataset_postgres.py:22` (identical `_cursor_with_rowcounts`); `test_load_open_pos.py:31`; `test_service_levels.py:15`; `test_lineage.py:13`; +30 files re-rolling cursor mocks inline
   - Proposed change: Add `tests/unit/conftest.py` with a `mock_db_cursor`/`mock_conn` factory; migrate duplicates.
   - Impact: Med · Effort: M · Risk: Low

9. **Replace brittle magic-column-count mock tuples with named builders**
   - Files: `test_unified_model_tuning.py` (1621 lines, `_list_row`/`_algo_config` positional tuples), `test_inv_planning_safety_stock.py` (1047), `test_inv_planning_replenishment.py`, `test_dashboard.py`
   - Problem: Positional tuples must match exact SELECT column order/count (the documented "mock tuple rejects response" pitfall); a new column silently misaligns every tuple.
   - Proposed change: Standardize per-table named-param builders (this is what `row_builders.py` was for — #3); centralize column order per table.
   - Impact: Med-High · Effort: M-H · Risk: L-M (builder must mirror real column order)

10. **Split the oversized API test files by sub-feature**
    - Files: `test_unified_model_tuning.py` (1621), `test_inv_planning_safety_stock.py` (1047), `test_dashboard.py` (895), `test_storyboard.py` (809)
    - Problem: The routers these cover are already split into sub-router packages; the tests remain monolithic.
    - Proposed change: Split along the same sub-feature boundaries, sharing fixtures via a local conftest.
    - Impact: Med · Effort: M · Risk: Low

11. **Consolidate inline `_make_pool` variants that diverge in signature** (`test_ai_planner_api.py:14` `description=`, `test_analysis.py:22` `fetchall_side_effect=`, `test_shap.py:57/216`)
    - Proposed change: Fold the missing knobs (`description=`; `fetchall_returns` already covers side_effect) into `make_pool`, delete the forks. `test_analysis.py`'s fork is likely redundant today.
    - Impact: Med · Effort: L-M · Risk: Low

12. **Audit `test_control_tower*.py` against current router surface**
    - Files: `tests/api/test_control_tower.py`, `test_control_tower_financial.py` (router still exists; frontend `ControlTowerTab.tsx` was deleted per git status)
    - Proposed change: Verify the control-tower endpoints are still mounted/consumed. If the feature is being retired, remove dead backend + tests together (per "removed feature → remove its tests"). If kept, no action.
    - Impact: Med · Effort: Low (investigation) + L/M (removal) · Risk: M (confirm truly unused before deleting)

**Highest leverage:** #1 (client-fixture adoption) is the single biggest maintenance reduction — the infra already exists in conftest and just needs broad adoption plus a pool-injection variant. #3 and #9 are linked — `row_builders.py` should either drive the brittle-tuple fix or be deleted. None of these weaken assertions.
