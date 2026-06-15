# User Story 6: Single mode-parameterized load engine

**Phase:** 2 — Single Load Engine
**Depends on:** US3, US4, US5
**Complexity:** L  **Risk:** HIGH

## Story
As a **platform engineer**, I want **one load engine parameterized by mode (`full` | `delta` | `file`)**, so that **full load is literally "delta with an empty watermark" and there is no second loader to keep in sync**.

## Background / Current State
Two divergent engines:
- `scripts/etl/load_dataset_postgres.py` — canonical, richer (forecast archive, lag resolution, partitions).
- `scripts/etl/load.py` — used by `IntegrationRunner`; adds slice-delete + file-mode logic (L160–176, L645+).

This is the central duplication. The two must merge into one engine consuming the US3–US5 helpers.

## Acceptance Criteria
- [ ] **AC1** — `load_dataset_postgres.py` accepts an explicit `--mode {full,delta,file}` (and existing `--replace`/`--skip-archive` map onto it) with a single internal code path keyed by mode.
- [ ] **AC2** — `full` = truncate+load; `delta` = change-detected incremental (DFU/FK filter + upsert or partition-range replace); `file` = single-file/slice load. All three reuse the US3–US5 helpers.
- [ ] **AC3** — `load.py`'s slice-delete and file-mode behaviors are reproduced exactly by the merged engine (verified by US1 characterization tests retargeted at the engine).
- [ ] **AC4** — Forecast archive dual-load, execution-lag resolution from `dim_sku`, and ON-CONFLICT keys are preserved bit-for-bit.
- [ ] **AC5** — Every mode writes `audit_load_batch` via `record_load_batch` (US5).
- [ ] **AC6** — `%s` placeholders throughout; no bare `except Exception`; no `print()`.

## TDD Plan
### Write first (red)
- `tests/unit/test_load_engine.py::test_mode_full_truncates_then_inserts`
- `::test_mode_delta_upserts_changed_rows_only`
- `::test_mode_delta_partition_range_replace` (inventory/customer_demand)
- `::test_mode_file_loads_single_slice`
- `::test_forecast_archive_and_lag_resolution_preserved`
- `::test_parity_with_load_py_slice_delete` (golden vs US1-captured `load.py` output)
### Then implement (green)
- Fold `load.py` logic into `load_dataset_postgres.py` behind `--mode`.
### Refactor
- Remove now-dead branches; self-review the merged control flow for clarity.

## Implementation Notes
- Keep `IntegrationRunner` pointed at `load.py` **for now** — US7 does the swap, so US6 stays reversible.
- Mode resolution: a single dispatch table, not nested `if/elif` sprawl.

## Definition of Done
- [ ] One engine handles full/delta/file via mode flag.
- [ ] All US1 parity tests + new engine tests green.
- [ ] `make test-all`, lint, type-check green. `load.py` still present (deleted in US7).
