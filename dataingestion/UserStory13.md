# User Story 13: Transaction isolation for multi-step loads

**Phase:** 4 — Reliability & Observability
**Depends on:** US11
**Complexity:** M  **Risk:** MEDIUM

## Story
As a **platform engineer**, I want **multi-step loads wrapped in a single transaction with proper rollback**, so that **a mid-load failure doesn't leave the DB half-written**.

## Background / Current State
`load_open_pos.py` (L120–125, L183–196, L229–235) commits after each function over a shared `conn`; a failure in step 2 leaves step 1 committed. `main()` (L273–328) doesn't wrap the sequence.

## Acceptance Criteria
- [ ] **AC1** — A failed step rolls back the entire load unit (suppliers+pos+receipts treated atomically, or per documented unit boundaries).
- [ ] **AC2** — On success, a single commit (or well-defined commit points) finalizes the batch.
- [ ] **AC3** — `audit_load_batch` records `failed` with the error message on rollback (via US5 writer).
- [ ] **AC4** — Other loaders' transaction boundaries are reviewed; any with the same "commit-per-step over shared conn" smell are fixed.
- [ ] **AC5** — 5xx/error paths never interpolate exception text into surfaced messages (CLAUDE.md).

## TDD Plan
### Write first (red)
- `tests/unit/test_load_open_pos.py::test_failure_in_step2_rolls_back_step1`
- `::test_success_commits_once`
- `::test_failed_batch_recorded_in_audit`
### Then implement (green) → Refactor
- Introduce a transaction context around the load unit; move commit to the boundary.

## Implementation Notes
- Prefer `with conn.transaction():` (psycopg3) over manual commit/rollback where possible.
- Keep `--dry-run` writing nothing and committing nothing.

## Definition of Done
- [ ] Atomic load unit with rollback on failure.
- [ ] Audit row reflects failure cause.
- [ ] `make test-all`, lint green.
