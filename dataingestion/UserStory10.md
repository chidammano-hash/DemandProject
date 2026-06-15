# User Story 10: Size-based index drop/recreate

**Phase:** 3 — Performance
**Depends on:** US3, US6
**Complexity:** S  **Risk:** LOW

## Story
As a **platform engineer**, I want **index drop/recreate to apply only to large tables**, so that **small dimension loads aren't slowed by needless index churn**.

## Background / Current State
The loader drops ALL secondary indexes before bulk load then recreates them — beneficial for huge facts (`fact_external_forecast_monthly` ~100M+), wasteful for small dims (`dim_item`, `dim_location`, `dim_customer`).

## Acceptance Criteria
- [ ] **AC1** — `etl_helpers` decides drop/recreate based on a configurable row-count (or est. size) threshold from `etl_config.yaml`.
- [ ] **AC2** — Below threshold: indexes are kept (upsert into indexed table); above: drop→load→recreate as today.
- [ ] **AC3** — Threshold lives in `etl_config.yaml performance:` with an inline comment; no magic number in Python.
- [ ] **AC4** — Large-fact load timing is unchanged or better; small-dim load timing improves.
- [ ] **AC5** — Final index set after load is identical regardless of branch taken.

## TDD Plan
### Write first (red)
- `tests/unit/test_etl_helpers.py::test_keeps_indexes_below_threshold`
- `::test_drops_and_recreates_above_threshold`
- `::test_threshold_read_from_config`
- `::test_final_index_set_identical_both_paths`
### Then implement (green) → Refactor
- Add size check + config key; branch the drop/recreate call.

## Implementation Notes
- Estimate size via `pg_class.reltuples` (cheap) rather than `COUNT(*)`.
- Default threshold conservative (e.g. dims always keep, facts always drop) and tune later.

## Definition of Done
- [ ] Size-gated index management; config-driven threshold.
- [ ] `tests/unit/test_etl_helpers.py`, `make test-all` green.
- [ ] Small-dim speedup noted vs baseline.
