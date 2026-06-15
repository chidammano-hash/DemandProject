# User Story 2: Baseline timing benchmarks

**Phase:** 0 — Baseline & Safety Net
**Depends on:** —
**Complexity:** S  **Risk:** LOW

## Story
As a **platform engineer**, I want **recorded before-state timing for full and incremental loads**, so that **"streamline / performance" claims in later phases are measurable, not anecdotal**.

## Background / Current State
`docs/RUNBOOK.md` has **no** timing estimates for `make pipeline-full`, `make pipeline-refresh`, `normalize-all`, or `load-all`. Performance work in Phase 3 (US8–US11) needs a baseline.

## Acceptance Criteria
- [ ] **AC1** — A repeatable benchmark procedure captures wall-clock for: `pipeline-full --parallel`, `pipeline-refresh` (no-change run), `pipeline-refresh` (one changed domain), per-domain `load-*`.
- [ ] **AC2** — Results recorded in a new `docs/RUNBOOK.md` section "Ingestion Performance Baseline" with date, dataset size (row counts via `make health`), and hardware note.
- [ ] **AC3** — Benchmark wraps stages with `profiled_section()` from `common/services/perf_profiler.py` where a script boundary exists (per CLAUDE.md — no raw `time.time()`).
- [ ] **AC4** — A `make perf-ingestion` (or documented command) reproduces the measurement.

## TDD Plan
Measurement story — light test surface.
### Write first
- `tests/unit/test_perf_ingestion_harness.py` — assert the harness emits structured timing records (domain, stage, seconds) and respects `profiled_section` thresholds from `config/platform/perf_config.yaml`.
### Then implement
- Thin harness/script under `scripts/tools/` or a Make target invoking existing pipeline with profiling enabled.

## Implementation Notes
- Do not hardcode thresholds — read `config/platform/perf_config.yaml`.
- Use `make health` row counts to annotate dataset scale alongside timings.

## Definition of Done
- [ ] Baseline numbers committed to `docs/RUNBOOK.md` in the same change.
- [ ] Reproducible command documented.
- [ ] `make test-all` green.
