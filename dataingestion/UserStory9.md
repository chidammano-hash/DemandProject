# User Story 9: Conditional / streamed forecast archive load

**Phase:** 3 — Performance
**Depends on:** US6
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **platform engineer**, I want **the forecast archive load to avoid redundant work**, so that **forecast ingestion doesn't pay the dual-load cost when it isn't needed**.

## Background / Current State
`_load_forecast_archive` (`load_dataset_postgres.py` L299–368) always populates both `fact_external_forecast_monthly` and `backtest_lag_archive` (all lags). It already has a fast-path that skips ON CONFLICT when no non-external models exist (~3×). Remaining cost: the archive insert itself runs on every forecast load even when the source slice is unchanged.

## Acceptance Criteria
- [ ] **AC1** — In `delta` mode, archive rows are only (re)written for the changed slice/lags, not the full table.
- [ ] **AC2** — The existing fast-path (skip ON CONFLICT when no competing models) is preserved.
- [ ] **AC3** — Execution-lag resolution from `dim_sku` and the `(forecast_ck, model_id, lag)` conflict key are unchanged.
- [ ] **AC4** — `agg_accuracy_by_dim` / lag-archive MVs produce identical results to pre-change (parity check).
- [ ] **AC5** — `full` mode behavior is unchanged (still loads all lags).

## TDD Plan
### Write first (red)
- `tests/unit/test_load_engine.py::test_archive_delta_writes_changed_slice_only`
- `::test_archive_fast_path_preserved_no_competing_models`
- `::test_archive_full_mode_loads_all_lags`
- `::test_lag_resolution_and_conflict_key_unchanged`
### Then implement (green) → Refactor
- Gate archive writes on mode + changed slice; keep full-mode path intact.

## Implementation Notes
- Reuse `delete_partition_range`/scoped DELETE (US4) for the archive slice in delta mode.
- Do not weaken multi-lag accuracy — archive must still hold all lags for the loaded slice.

## Definition of Done
- [ ] Delta archive writes scoped; full mode unchanged.
- [ ] Accuracy MV parity verified.
- [ ] `make test-all` green; forecast load timing recorded vs baseline.
