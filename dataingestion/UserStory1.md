# User Story 1: Characterization tests for existing loaders

**Phase:** 0 — Baseline & Safety Net
**Depends on:** —
**Complexity:** M  **Risk:** LOW

## Story
As a **platform engineer**, I want **characterization (golden) tests around the current ETL loaders**, so that **the upcoming refactor cannot silently change load behavior**.

## Background / Current State
Only 2 ETL test files exist: `tests/unit/test_load_dataset_postgres.py`, `tests/unit/test_load_ext_ml_forecasts.py`. These are untested before refactor:
- `scripts/etl/load_customer_demand_postgres.py` (425 LOC)
- `scripts/etl/load_backtest_forecasts.py` (704 LOC)
- `scripts/etl/load_open_pos.py` (332 LOC)
- `scripts/etl/run_pipeline.py` (657 LOC — change detection, mode logic)
- `scripts/etl/normalize_*.py`

We need behavior pinned **before** Phases 1–4 touch these files.

## Acceptance Criteria
- [ ] **AC1** — Given the current SQL builders/decision functions, when invoked with representative inputs, then their outputs (SQL strings, WHERE clauses, domain dicts) are asserted exactly.
- [ ] **AC2** — `run_pipeline.detect_changes()` and `detect_inventory_changes()` are tested: changed vs unchanged hash → correct skip/reload decision.
- [ ] **AC3** — `build_incremental_delete()` produces the expected month-range WHERE clause for given filenames.
- [ ] **AC4** — `load_customer_demand_postgres` UPSERT vs `--replace` vs `--month` SQL paths each have an asserting test.
- [ ] **AC5** — All new tests pass against the **current, unmodified** code (`~/.local/bin/uv run pytest tests/ -q`).
- [ ] **AC6** — Tests use `make_pool`/`make_async_pool` from `tests/api/conftest.py`; no hand-rolled `MagicMock` on `psycopg.connect`.

## TDD Plan
This story *is* tests — they characterize existing behavior (must pass as-is).
### Write (red→green against current code)
- `tests/unit/test_run_pipeline.py` — `detect_changes`, `detect_inventory_changes`, `build_incremental_delete`, `get_mvs_for_domains`.
- `tests/unit/test_load_customer_demand_postgres.py` — partition SQL, upsert SQL, `--month`/`--replace` branches.
- `tests/unit/test_load_backtest_forecasts.py` — index/constraint helpers, dual-path archive insert decision.
- `tests/unit/test_load_open_pos.py` — current row-by-row behavior captured (so US11 can prove parity).
- `tests/unit/test_normalize_dataset_csv.py` — sales TYPE=1 filter, forecast lag 0–4 filter, date normalization, type casting/null coercion.

## Implementation Notes
- Refactor SQL-string builders into pure functions if needed to make them testable — but do **not** change behavior in this story.
- Mock subprocess at the boundary for `run_pipeline`.

## Definition of Done
- [ ] New test files cover the 6 previously-untested scripts' key logic.
- [ ] `make test-all` green with **zero** production-code behavior change.
- [ ] Coverage of touched modules recorded as the baseline for later parity checks.
