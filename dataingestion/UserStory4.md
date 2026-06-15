# User Story 4: Shared staging + partition management

**Phase:** 1 — Shared ETL Core
**Depends on:** US3
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **platform engineer**, I want **one staging-table convention and one partition manager**, so that **adding a partitioned domain no longer means copy-pasting partition code**.

## Background / Current State
- Staging names diverge: `_stg_{domain}`, `_stg_customer_demand_bulk`, `_stg_archive`, `_stg_backtest`, `_stg_ext_ml`, `_slice_stg`, `_upsert_stg`.
- Partition strategies diverge across `load_dataset_postgres.py` (inventory), `load_customer_demand_postgres.py` L59–93, and `load.py` L160–176 slice-delete.
- Partition field map is hardcoded in `load.py` L67–72 (`_SLICE_DELETE_TABLES`).

## Acceptance Criteria
- [ ] **AC1** — `common/core/etl_helpers.py` adds `staging_table_name(domain) -> str` (one convention, e.g. `stg_<domain>`).
- [ ] **AC2** — `ensure_monthly_partition(cur, parent, month)`, `drop_monthly_partition(cur, parent, month)`, and `delete_partition_range(cur, table, date_col, start, end)` centralize partition logic.
- [ ] **AC3** — Partition field metadata (table + date column) moves into `DomainSpec` (or `etl_config.yaml`), removing the hardcoded `_SLICE_DELETE_TABLES` dict.
- [ ] **AC4** — `inventory`, `customer_demand`, `sales`, `forecast` all use the shared functions.
- [ ] **AC5** — US1 characterization tests pass (parity); partition naming output unchanged for existing partitions.

## TDD Plan
### Write first (red)
- `tests/unit/test_etl_helpers.py::test_staging_table_name_is_deterministic`
- `::test_ensure_monthly_partition_idempotent` (CREATE ... IF NOT EXISTS)
- `::test_delete_partition_range_builds_half_open_interval` (`>= start AND < end`)
- `tests/unit/test_domain_specs.py::test_partitioned_domains_declare_date_column`
### Then implement (green)
- Add helpers; add partition metadata to `DomainSpec`/config; repoint loaders.
### Refactor
- Delete `_SLICE_DELETE_TABLES` and per-loader partition snippets.

## Implementation Notes
- Keep half-open date intervals (`>= start AND < end`) — matches current `build_incremental_delete`.
- `customer_demand` `--replace` parallel-per-month insert stays, but partition create/drop goes through the helper.

## Definition of Done
- [ ] One staging convention; one partition manager.
- [ ] Partition metadata declared in a single source of truth.
- [ ] `tests/unit/test_etl_helpers.py`, US1 tests, `make test-all` green.
