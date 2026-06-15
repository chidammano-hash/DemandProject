# User Story 11: COPY/executemany for load_open_pos.py

**Phase:** 3 — Performance
**Depends on:** US1
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **platform engineer**, I want **load_open_pos.py to batch-insert via COPY/executemany**, so that **PO loading drops the row-by-row N+1 overhead**.

## Background / Current State
`scripts/etl/load_open_pos.py` uses `for _, row in df.iterrows():` with per-row INSERT/UPDATE in three functions: `load_suppliers()` L107, `load_pos()` L157, `load_receipts()` L217. CLAUDE.md forbids row-by-row inserts at scale.

## Acceptance Criteria
- [ ] **AC1** — All three loaders use staging + COPY (or `cursor.executemany`/`execute_values`) instead of `iterrows()`.
- [ ] **AC2** — Upserts use `ON CONFLICT DO UPDATE` on the documented keys (`fact_lead_time_actuals`(po_number,line_number), `fact_open_purchase_orders`(po_number,po_line_number), suppliers PK).
- [ ] **AC3** — Loaded rows and resulting table state are identical to the row-by-row version (parity vs US1 capture).
- [ ] **AC4** — `--dry-run` still performs no writes.
- [ ] **AC5** — Bare `except Exception` (L50) replaced with specific exceptions + `logger.exception()`.

## TDD Plan
### Write first (red)
- `tests/unit/test_load_open_pos.py::test_suppliers_batched_upsert`
- `::test_pos_batched_upsert_on_conflict`
- `::test_receipts_batched_upsert`
- `::test_dry_run_no_writes`
- `::test_result_parity_with_row_by_row` (golden from US1)
### Then implement (green) → Refactor
- Replace loops with staging+COPY; route through `etl_helpers` staging naming.

## Implementation Notes
- Use the US4 staging helper for temp table names; reuse US5 audit writer.
- Keep date-parse helper but fix its bare-except.

## Definition of Done
- [ ] No `iterrows()` inserts remain in `load_open_pos.py`.
- [ ] Parity tests green; `make test-all`, lint green.
- [ ] PO load timing improvement noted vs baseline.
