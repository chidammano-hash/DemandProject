## common/core + common/services — Refactor Opportunities

_Scope: `common/core/` and `common/services/`. Read-only audit — no code changed._

### Quick wins
- Plain async `cached()` decorator (`common/services/cache.py:414`) appears unused — `cached_sync`/`cached_async` are the live ones; verify and delete.
- `_run_seasonality` (`job_state.py:334`) and `_run_compute_variability` (`job_state.py:856`) are legacy passthroughs to `_run_compute_sku_features`; point the `seasonality_pipeline`/`compute_variability` callables directly at `_run_compute_sku_features` and drop both wrappers + imports (`job_registry.py:70,85`).
- `_row_to_dict` defined in `common/services/integration_runner.py:38` violates "row helpers only in `sql_helpers.py`" — it is just `row_to_dict_from_cursor` + `_to_iso`; move the coercion into a shared helper.
- 13 backtest runner aliases (`job_state.py:497-509`) exist only to rename `_run_backtest(model,...)`; the registry can call `functools.partial(_run_backtest, model)` and the alias block + 13 registry imports vanish.
- `_to_iso` (`integration_runner.py:30`), `_as_dict`/`_coerce_int` (`job_shape.py:43,48`), inline `isoformat()` coercions (`job_state.py:1495`, `job_registry.py:874-876`, `integration_chain_jobs.py:38`) — consolidate into `sql_helpers.py`.
- `except Exception:` with bare `logger.warning(...)` (no `.exception()`, no specific type) at `job_state.py:94,105,119,131,144` (and ~50 more in scope) violates the error-handling rule.
- `JobManager._db_get` and `_db_list` (`job_registry.py:608,622`) repeat the same 14-column tuple literal — hoist to a module constant `_JOB_COLS`.
- `MODEL_OUTPUT_DIRS` (`job_state.py:71`) only has 3 tree entries but is imported into `job_registry.py:35`; confirm still used or drop the cross-module import.

### Ranked opportunities

1. **Collapse ~25 copy-paste subprocess job runners into a declarative table**
   - Files: `job_state.py:654-1040` (+ simple runners from ~511), `job_registry.py:106-490`
   - Problem: ~25 `_run_*` functions are the same shape: optional `progress_cb(pct=…)`, build `cmd = [_UV, "run", "python", "scripts/X.py", …flags]`, `output = _run_subprocess(...)`, `return {"output_log": ...}`. The only variation is script path, fixed flags, and `params.get(...)`→flag mappings. The same metadata is re-declared as a `JobTypeDef` in the 400-line static registry.
   - Proposed change: A `SubprocessJobSpec(script, fixed_args, param_flags={...}, start_msg, done_msg)` + a single generic runner; drive both callable and `JobTypeDef` from one declarative dict. Bespoke runners (`_run_backtest`, `_run_tuning_*`, `_run_cluster_*`, `_run_model_tuning_experiment`) stay real functions; only the thin wrappers collapse.
   - Impact: High (−300–400 LoC, kills dual-declaration drift) · Effort: M · Risk: M (preserve exact CLI flags + `output_log` fallbacks; add a registry round-trip test)

2. **Split `job_state.py` (1649 LoC) by responsibility**
   - Problem: One module mixes DB/PID/log helpers, the subprocess engine, ~30 domain job callables, and `_row_to_dict`.
   - Proposed change: `job_state/` package: `runtime.py` (`_get_conn`, PID/log helpers, `_run_subprocess`, `_row_to_dict`), `runners_ml.py`, `runners_inventory.py`, `runners_etl.py`, `types.py` (`JobTypeDef`). Re-export public names from `__init__.py`.
   - Impact: High · Effort: M · Risk: L (mechanical move; preserve import surface)

3. **Extract DB persistence out of `JobManager` (1410 LoC)**
   - Files: `job_registry.py:569-700` (`_db_*`), `_kill_process`, `_is_pid_alive`, `recover_stale_jobs`
   - Problem: `JobManager` is a god-object: APScheduler lifecycle + concurrency groups + pipeline chaining + retry/recovery + process kill + all `job_history` SQL. The static `_db_*` methods are pure data access.
   - Proposed change: Move `_db_*` into `job_repository.py` (free functions); move PID/process control next to PID helpers in `job_state/runtime.py`.
   - Impact: High · Effort: M · Risk: M (`JobManager` is a `__new__` singleton; verify recovery/cancel paths)

4. **Standardize error handling (~55 `except Exception`)**
   - Files: `job_state.py` (`:94,105,119,131,144,276,289,300,385,436,461,469,543,601` + more), `job_registry.py`
   - Problem: Pervasive `except Exception:` + `logger.warning("Failed to …")` with no specific type and no `logger.exception()`.
   - Proposed change: Narrow to `psycopg.Error` for DB helpers, use `logger.exception(...)`; annotate genuine broad catches with `# noqa: BLE001 — <reason>`. The five PID/log helpers share one pattern → one `_safe_db_write(sql, args, warn_msg)` helper.
   - Impact: High · Effort: M · Risk: L

5. **Unify the three cache decorators**
   - Files: `cache.py:414-480`
   - Problem: `cached`, `cached_sync`, `cached_async` are near-identical; `cached` appears unused and is a subset of `cached_async`.
   - Proposed change: Delete `cached` if unused; factor shared key-build + get/set into `_make_cached(is_async)`.
   - Impact: Med · Effort: S · Risk: L (keep public names/signatures stable)

6. **Drive `JOB_TYPE_REGISTRY` from a compact data table**
   - Files: `job_registry.py:106-490`
   - Problem: 49 `JobTypeDef(...)` blocks, each repeating `type_id=` (dup of the dict key) and importing 49 callables by name.
   - Proposed change: After #1, generate most entries from subprocess-job specs; store the rest as rows and build `JobTypeDef`s in a loop, deriving `type_id` from the key.
   - Impact: High (compounds with #1) · Effort: M · Risk: M (keep every `type_id`/`group`/`params_schema`; snapshot test)

7. **De-duplicate `_db_get`/`_db_list` column lists + row→dict plumbing**
   - Files: `job_registry.py:608-655`
   - Proposed change: Hoist `_JOB_COLS`; source the dynamic-SET builder's allowed columns from one constant too.
   - Impact: Med · Effort: S · Risk: L

8. **Consolidate scattered serialization coercers (datetime/UUID/JSON) into `sql_helpers.py`**
   - Files: `integration_runner.py:30,38`, `job_shape.py:43,48,75`, `job_state.py:1469-1500`, `integration_chain_jobs.py:38`, `job_registry.py:874-876`
   - Proposed change: `coerce_json_safe(value)` + `row_to_json_dict(...)` in `sql_helpers.py`; rewrite per-module wrappers. Removes the `integration_runner` rule violation.
   - Impact: Med · Effort: M · Risk: L–M (verify each caller's JSON/None defaults — `params`→`{}`, `result`→`None`)

9. **Standardize the three DB connection helpers**
   - Files: `job_state.py:36` (autocommit ON), `pg_queue.py:80` (autocommit OFF), `etl_helpers.py:491` (inline connect)
   - Proposed change: `common/core/db.py` `connect(autocommit: bool = False)`; have all three call it (keeps lazy-import behavior centralized).
   - Impact: Med · Effort: S · Risk: L (preserve each caller's autocommit expectation)

10. **Simplify `_run_tuning_backtest` / `_run_model_tuning_experiment`**
    - Files: `job_state.py:1041-1115`, `:1116-1276` (~160 lines)
    - Proposed change: Extract `_build_tuning_config(...)` and `_register_tuning_results(...)`; hoist function-local imports to module level.
    - Impact: Med · Effort: M · Risk: M (preserve temp-config format + tracker call order)

11. **Centralize the per-runner `progress_cb` boilerplate** — folded into #1 (generic runner emits start/done from spec). · Impact: Med · Effort: S · Risk: L

12. **Move `_kill_process`/`_is_pid_alive` next to PID storage helpers** (`job_registry.py:990,1136` → `job_state/runtime.py`). · Impact: Med · Effort: S · Risk: L

**Highest leverage:** #1 + #6 + #2/#3 together eliminate the bulk of the ~3000 lines of job-engine duplication and bring both oversized files under guidance. Bundle #4 (error handling) into the same passes since it touches the same helpers.
