# User Story 12: Move magic numbers to etl_config.yaml

**Phase:** 3 — Performance
**Depends on:** —
**Complexity:** S  **Risk:** LOW

## Story
As a **platform engineer**, I want **all ETL performance constants in YAML**, so that **tuning batch size / work_mem / worker counts doesn't require code edits** (CLAUDE.md: no magic numbers).

## Background / Current State
Hardcoded constants:
- `load_backtest_forecasts.py` L35 `BATCH_SIZE = 2_000_000`
- `load_ext_ml_forecasts.py` L31 `BATCH_SIZE = 2_000_000`
- `load_customer_demand_postgres.py` L50–52 `_PG_WORK_MEM="256MB"`, `_MAX_PARALLEL_WORKERS=6`
- Unmatched-DFU threshold, index-size threshold (from US5/US10)

## Acceptance Criteria
- [ ] **AC1** — `config/etl/etl_config.yaml` gains a `performance:` section: `batch_size`, `pg_work_mem`, `max_parallel_workers`, `unmatched_warn_pct`, `index_drop_row_threshold`.
- [ ] **AC2** — Every key has an inline comment (explanation, valid range, default) per CLAUDE.md.
- [ ] **AC3** — All listed constants are read via `load_config`/`load_forecast_pipeline_config` helpers; no literals remain in the scripts.
- [ ] **AC4** — Missing config keys fall back to documented defaults (no crash on partial config).

## TDD Plan
### Write first (red)
- `tests/unit/test_etl_config.py::test_performance_section_present_with_comments` (keys exist)
- `tests/unit/test_load_engine.py::test_batch_size_read_from_config`
- `::test_work_mem_and_workers_from_config`
- `::test_defaults_when_keys_absent`
### Then implement (green) → Refactor
- Add YAML section; replace literals with config reads.

## Implementation Notes
- Reuse `load_config("etl/etl_config.yaml")`; keep the two-path fallback already in `load.py`'s config loader (or fold into a single helper).

## Definition of Done
- [ ] No ETL perf magic numbers in Python.
- [ ] `etl_config.yaml performance:` documented inline.
- [ ] `make test-all` green.
