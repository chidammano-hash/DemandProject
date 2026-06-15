# User Story 5: Shared DFU/FK filtering + audit_load_batch writer

**Phase:** 1 — Shared ETL Core
**Depends on:** US3
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **platform engineer**, I want **one DFU-match/FK-orphan filter and one audit_load_batch writer**, so that **filtering behaves identically across domains and every load records its batch lineage**.

## Background / Current State
DFU/FK filtering implemented 3 ways:
- `load_dataset_postgres.py` L276–300 `_filter_unmatched_dfus()` (separate DELETE)
- `load_ext_ml_forecasts.py` L230–244 (inline JOIN)
- `load_backtest_forecasts.py` L504–513 (inline JOIN)

`audit_load_batch` writing is inconsistent — `customer_demand` does not record a hash row, so it is invisible to incremental change detection (`run_pipeline.detect_changes`).

## Acceptance Criteria
- [ ] **AC1** — `common/core/etl_helpers.py` adds `filter_unmatched_dfus(cur, staging, domain)` covering both sales/forecast (item+customer_group+loc) and inventory (item+loc) keying.
- [ ] **AC2** — `filter_fk_orphans(cur, staging, domain)` removes rows with missing `dim_item`/`dim_location` refs.
- [ ] **AC3** — Unmatched-row warning threshold (currently >10%) is read from `etl_config.yaml`, not hardcoded.
- [ ] **AC4** — `record_load_batch(cur, domain, source_file, source_hash, rows_in, rows_out, status, error=None)` is the single audit writer; **all** loaders call it, including `customer_demand`.
- [ ] **AC5** — After a `customer_demand` load, `detect_changes(["customer_demand"])` correctly returns "unchanged" on a re-run (closes the gap noted in the pipeline map).

## TDD Plan
### Write first (red)
- `tests/unit/test_etl_helpers.py::test_filter_unmatched_dfus_sales_keying`
- `::test_filter_unmatched_dfus_inventory_keying`
- `::test_filter_fk_orphans_removes_missing_dim_refs`
- `::test_unmatched_threshold_from_config`
- `::test_record_load_batch_writes_hash_and_counts`
- `tests/unit/test_run_pipeline.py::test_customer_demand_change_detection_roundtrip`
### Then implement (green) → Refactor
- Add helpers; repoint 3 filter sites + all audit writes; delete inline copies.

## Implementation Notes
- Use `%s` parameterized values; `psycopg.sql.Identifier` for table/column names.
- `source_hash` via `file_hash()` from `common/engines/medallion.py` (do not reimplement).

## Definition of Done
- [ ] One filter impl, one audit writer; inline copies removed.
- [ ] `customer_demand` participates in change detection.
- [ ] `tests/unit/test_etl_helpers.py`, US1 tests, `make test-all` green.
