## scripts/ — Refactor Opportunities

_Scope: `scripts/` (etl, ml, forecasting, inventory, ops, ai, algorithm_testing, tools, db). Read-only audit — no code changed._

### Quick wins
- Stale doc references to deleted script `compute_demand_variability` in `scripts/ops/run_perf_analysis.py:62,68` — update the config-mapping table.
- `date.today()` violation: `scripts/db/db_maintenance.py:205` — replace with `get_planning_date()`.
- 396 `print()` calls across 45 production scripts; worst: `scripts/ml/run_champion_selection.py` (54), `scripts/ml/auto_tune.py` (41), `scripts/ml/train_meta_learner.py` (32), `scripts/ml/compare_backtest_runs.py` (27) — convert to `logger`.
- Bare `pd.read_sql` over fact tables: `run_champion_selection.py:203,226,638`, `run_clustering_scenario.py:420,512`, `run_inventory_backtest.py:86,106,124`, `simulate_champion_strategies.py:77,99` — switch to `read_sql_chunked`.
- `parents[2]` at module level in ~70 files (e.g. `run_champion_selection.py:34`, `etl/load.py:49`) — move into `if __name__=="__main__"` or use `common.core.paths.PROJECT_ROOT`.
- Duplicate lead-time loader: `compare_inventory_algorithms.py:186` `_load_lead_time_map` ≈ `compute_safety_stock.py:649` `_load_lead_time_data` — dedupe.
- `scripts/etl/load_open_pos.py:32` uses `Path(__file__).parent.parent.parent` (different idiom than the rest of etl/) — standardize.
- 66 `except Exception` occurrences; concentrated in `run_perf_analysis.py` (15), `run_cluster_pipeline.py` (6), `generate_replenishment_exceptions.py` (6) — narrow to specific exceptions.

### Ranked opportunities

1. **Inventory SS math + DB loaders live in a script and are imported script-to-script**
   - Files: `scripts/inventory/compute_safety_stock.py:80-364` (compute_ss_*, get_z_score, classify_xyz), imported by `compare_inventory_algorithms.py` and `run_inventory_backtest.py`; loaders at `:628-767`
   - Problem: Pure inventory math + reusable loaders defined inside a 1412-line CLI script, then imported by two other scripts — a script-as-library anti-pattern. `common/engines/` has no inventory module. Loaders duplicated in `compare_inventory_algorithms.py`.
   - Proposed change: Extract math into `common/inventory/safety_stock.py` and loaders into `common/inventory/loaders.py`; all three scripts import from `common/`.
   - Impact: High (removes coupling, dedupes 3 callers, unblocks unit-testing formulas) · Effort: M · Risk: M (3 callers + result parity)

2. **`compute_safety_stock.run()` is a ~400-line god-function** (`:965-1366`)
   - Proposed change: Decompose into `_load_inputs(cur)`, `_compute_rows(inputs, cfg)`, `_persist(cur, rows)`; keep `run()` as orchestration. Pairs with #1.
   - Impact: High · Effort: M · Risk: M

3. **`run_backtest.main()` is ~455 lines mixing CLI, timeframe loop, per-cluster training, persistence** (`run_backtest.py:1118-1573`)
   - Proposed change: Split into a `backtest_runner` module under `common/ml/` (or `scripts/ml/backtest/`): `cli.py`, `timeframes.py`, `orchestrate.py`, `persist.py`. Thin `main()`.
   - Impact: High · Effort: H · Risk: M-H (core backtest path)

4. **`generate_production_forecasts.py` (1541 LoC) bundles 3 distinct concerns**
   - Problem: loaders (141-363), inference grid/profile (385-633), three generation strategies (663-1153), writers/purge (1155-1276), `main()` (1277-1541). `load_recent_sales` (229) + `_load_customer_features_for_inference` (633) read fact tables.
   - Proposed change: `scripts/forecasting/production/` package: `loaders.py`, `inference_grid.py`, `generators.py`, `writers.py`. Verify fact reads use chunked helpers.
   - Impact: High · Effort: H · Risk: M-H (production forecast generation)

5. **Repeated sys.path/ROOT bootstrap in ~80 files violates the parents[N] rule**
   - Problem: Each script re-implements path bootstrap at module scope. Where `ROOT` is only for `sys.path` it should be inside `__main__`; where it's used for data/config (e.g. `run_champion_selection.py:824,848,880,1054`) it should use `common.core.paths`.
   - Proposed change: Standardize on `from common.core.paths import PROJECT_ROOT, CONFIG_DIR, DATA_DIR`; move genuine bootstrap into `if __name__=="__main__"`. Consider a tiny `scripts/_bootstrap.py`.
   - Impact: High (closes a FREQUENTLY-VIOLATED rule fleet-wide) · Effort: M · Risk: L-M

6. **~50 scripts repeat argparse + logging.basicConfig + DB-connect boilerplate**
   - Problem: 79 ArgumentParser sites, 49 basicConfig, 68 `get_db_params`, 62 direct `psycopg.connect`; 40 files define `--dry-run`, 46 `--item`/`--loc`.
   - Proposed change: `common/core/cli.py` with `build_base_parser()` + `configure_logging()`; a `db_cursor()` context manager in `common/core/db.py`. Migrate incrementally.
   - Impact: High · Effort: M · Risk: Low

7. **`run_champion_selection.py` — 54 print()s + bare fact reads + parallel inserters** (`:203,226,638`, insert_* `:380-633`)
   - Proposed change: prints→logging, reads→`read_sql_chunked`, collapse `insert_ceiling/champion/ensemble/fallback_forecasts` into one parameterized inserter.
   - Impact: Med-High · Effort: M · Risk: M (champion promotion correctness)

8. **Expert-panel script family (3517 LoC across 4 scripts) likely shares orchestration**
   - Files: `run_expert_panel.py` (828), `run_adv_expert_panel.py` (815), `expert_panel_route_analysis.py` (809), `run_expert_system_backtest.py` (1065)
   - Proposed change: Lift shared config-load/panel-assembly/reporting into `common/ml/expert_panel/runner.py`; each script becomes a thin CLI selecting a panel config.
   - Impact: Med-High · Effort: H · Risk: M

9. **`etl/load.py` (1359 LoC) — single file owns delta detection, partitioning, upsert, maintenance, CLI**
   - Proposed change: Split into `scripts/etl/load/` package: `delta.py`, `multifile.py`, `upsert.py`, `maintenance.py`, `cli.py`; push generic partition/index logic into `etl_helpers.py`.
   - Impact: Med-High · Effort: H · Risk: M-H (core incremental pipeline)

10. **`run_perf_analysis.py` — 15 bare `except Exception` + stale deleted-script references** (`:62,68`)
    - Proposed change: Narrow exceptions + `logger.exception()`; remove deleted-script entries from the config-mapping table.
    - Impact: Med · Effort: Low · Risk: Low

11. **`train_meta_learner.py` — documents the OOM risk but still prints 32× and reads fact tables** (`:68`)
    - Proposed change: Apply `stream_query_in_chunks`/`read_sql_chunked` per the noted comment; prints→logging.
    - Impact: Med (scale correctness) · Effort: L-M · Risk: M

12. **`tune_hyperparams.py` vs `tune_cluster_hyperparams.py` vs `auto_tune.py` — overlapping tuning orchestration**
    - Proposed change: Extract a shared `common/ml/tuning/` runner (search loop + persistence); scripts become thin config selectors; prints→logging.
    - Impact: Med · Effort: M-H · Risk: M

13. **Foundation-backtest dispatcher is the target pattern — extend it to tree backtests**
    - Problem: `run_backtest_chronos*.py` delegate cleanly to `foundation_backtest.run_foundation_backtest`; but `run_backtest_xgboost.py`/`catboost.py` delegate via `sys.argv` mutation (fragile).
    - Proposed change: After #3, expose a `run_backtest(model=...)` callable so wrappers pass `model="xgboost"` directly.
    - Impact: Low-Med · Effort: Low (depends on #3) · Risk: Low
