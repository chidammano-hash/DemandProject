# User Story 15: customer_demand in change detection

**Phase:** 4 — Reliability & Observability
**Depends on:** US5
**Complexity:** S  **Risk:** LOW

## Story
As a **planner**, I want **customer_demand to participate in incremental change detection**, so that **`pipeline-refresh` skips it when unchanged and reloads it when it changes — like every other domain**.

## Background / Current State
`customer_demand` is loaded by a dedicated script and (pre-US5) does not write a comparable `audit_load_batch` hash row, so `run_pipeline.detect_changes` can't reason about it. US5 added the audit writer; this story wires `customer_demand` fully into the refresh orchestration.

## Acceptance Criteria
- [ ] **AC1** — `customer_demand` is in `run_pipeline.py`'s change-detected domain set for `--mode refresh`.
- [ ] **AC2** — Unchanged `customer_demand` CSV → domain skipped (logged "skipped"); changed → normalize+load+MV refresh.
- [ ] **AC3** — Its dependent MVs (`mv_customer_activity_monthly`, `mv_ca_*`) refresh only when it changed (via `get_mvs_for_domains`).
- [ ] **AC4** — `--month`/`--replace` manual modes still work unchanged.
- [ ] **AC5** — `etl_config.yaml` `mv_refresh` mapping includes `customer_demand → mv_customer_activity_monthly, mv_ca_*`.

## TDD Plan
### Write first (red)
- `tests/unit/test_run_pipeline.py::test_customer_demand_in_refresh_set`
- `::test_customer_demand_skipped_when_unchanged`
- `::test_customer_demand_mvs_refresh_on_change`
- `tests/unit/test_etl_config.py::test_customer_demand_mv_mapping_present`
### Then implement (green) → Refactor
- Add domain to refresh detection; extend `mv_refresh` config; route through US5 audit writer.

## Implementation Notes
- Reuse `file_hash` (medallion) and US5 `record_load_batch`.
- Keep the customer-demand loader's special partition/`--replace` logic intact.

## Definition of Done
- [ ] `customer_demand` behaves like other domains under `pipeline-refresh`.
- [ ] `make test-all` green; `make audit-routers` unaffected.

> **Scope note (implementation):** customer_demand is wired into the **refresh**
> path of run_pipeline (dedicated normalize_customer_demand + load_customer_demand
> helpers, special-cased like inventory) and the mv_refresh map (CA views). It is
> change-detected via the existing audit_load_batch hash (US5). The **full-load**
> path keeps its dedicated `make load-customer-demand --replace` target (the
> primary full-load route), so run_full was left unchanged to avoid regressing
> its generic/parallel flow.
