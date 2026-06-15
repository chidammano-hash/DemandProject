# User Story 3: Shared index/constraint management helper

**Phase:** 1 — Shared ETL Core
**Depends on:** US1
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **platform engineer**, I want **a single implementation of index/constraint drop→load→recreate**, so that **schema/perf fixes are made once, not in 3+ divergent copies**.

## Background / Current State
Three independent implementations exist:
- `load_dataset_postgres.py` L55–88 (`_get_all_indexes`, `_drop_indexes`, `_recreate_indexes`, `_get_unique_constraints`, …)
- `load_backtest_forecasts.py` L89–159 (main + archive variants, nearly identical)
- `load_ext_ml_forecasts.py` L263–336 (third copy)

## Acceptance Criteria
- [ ] **AC1** — New module `common/core/etl_helpers.py` exposes: `get_secondary_indexes(cur, table)`, `drop_indexes(cur, table)`, `recreate_indexes(cur, table, ddl)`, and the unique-constraint equivalents.
- [ ] **AC2** — All SQL uses `psycopg.sql.Identifier` for identifiers and `%s` for values (no f-string identifier interpolation).
- [ ] **AC3** — All three call sites import from `etl_helpers`; the local copies are deleted in the same change (no shims).
- [ ] **AC4** — US1 characterization tests still pass unchanged (behavior parity).
- [ ] **AC5** — `except` blocks catch specific exceptions (`psycopg.Error`) with `logger.exception()`; no bare `except Exception`.

## TDD Plan
### Write first (red)
- `tests/unit/test_etl_helpers.py::test_get_secondary_indexes_excludes_pk_and_constraints`
- `::test_drop_indexes_emits_drop_per_index`
- `::test_recreate_indexes_replays_captured_ddl`
- `::test_identifiers_quoted_values_parameterized` (assert no raw f-string identifiers)
### Then implement (green)
- `common/core/etl_helpers.py` index/constraint functions.
### Refactor
- Repoint the 3 loaders; delete duplicated helpers; re-run US1 tests.

## Implementation Notes
- Preserve current ordering (drop secondary indexes → bulk load → recreate → `ANALYZE`).
- Keep archive-table variant as the same function called with a different table name (not a separate function).

## Definition of Done
- [ ] One implementation; 3 copies deleted.
- [ ] `tests/unit/test_etl_helpers.py` + US1 tests green.
- [ ] `make test-all`, lint, type-check green.
