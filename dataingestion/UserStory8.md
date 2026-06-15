# User Story 8: Push DFU filtering to normalize time

**Phase:** 3 — Performance
**Depends on:** US5, US6
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **platform engineer**, I want **DFU-mismatch rows filtered during normalization instead of after COPY**, so that **the load step avoids a full-table DELETE scan on large fact loads**.

## Background / Current State
Currently sales/forecast/inventory COPY all rows into staging, then run a post-COPY DELETE of unmatched DFUs (`filter_unmatched_dfus`, US5). At 40× scale this DELETE scan is a bottleneck (noted in the pipeline map). `dim_sku` is loaded before facts in wave order, so the valid-key set is available at normalize time.

## Acceptance Criteria
- [ ] **AC1** — Normalizers for sales/forecast/inventory drop rows whose DFU key is absent from the current `dim_sku` set, before writing `data/staged/*_clean.csv`.
- [ ] **AC2** — The load-time `filter_unmatched_dfus` becomes a cheap safety net (logs 0 in the normal case) rather than the primary filter.
- [ ] **AC3** — Unmatched counts are still logged (now at normalize time) with the configured threshold warning.
- [ ] **AC4** — Output row counts and final loaded rows are **identical** to pre-change (parity vs US1 baseline).
- [ ] **AC5** — Measurable load-time reduction on the changed-domain path, recorded against the US2 baseline.

## TDD Plan
### Write first (red)
- `tests/unit/test_normalize_dataset_csv.py::test_normalize_drops_rows_absent_from_dim_sku`
- `::test_inventory_keying_item_loc_only`
- `::test_unmatched_count_logged_with_threshold`
- `tests/unit/test_load_engine.py::test_load_time_dfu_filter_is_noop_after_normalize`
### Then implement (green) → Refactor
- Add dim_sku lookup to normalize step (read keys once); keep load-time filter as guard.

## Implementation Notes
- Read the `dim_sku` key set via a single query (small) — do **not** `pd.read_sql` the fact tables.
- If `dim_sku` is empty (cold DB), skip normalize-time filtering and fall back to load-time (don't drop everything).

## Definition of Done
- [ ] Filtering happens at normalize time; load-time filter is a no-op net.
- [ ] Row-count parity vs baseline; load speedup recorded in RUNBOOK.
- [ ] `make test-all` green.
