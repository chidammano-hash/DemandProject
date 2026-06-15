# User Story 14: Logging + exception + path-hack cleanup

**Phase:** 4 — Reliability & Observability
**Depends on:** —
**Complexity:** S  **Risk:** LOW

## Story
As a **platform engineer**, I want **all ETL scripts to use structured logging, specific exceptions, and canonical paths**, so that **the pipeline is debuggable and passes the CLAUDE.md rule gates**.

## Background / Current State
Violations across `scripts/etl/`:
- `print()` — `load_backtest_forecasts.py` 28 occurrences (L102–336), others.
- Bare `except Exception` — `load.py` L1207, `load_dataset_postgres.py` L849, `load_open_pos.py` L50, `trim_input_files.py` L130, `run_pipeline.py` L292/L411/L533.
- Path hack — `normalize_inventory_csv.py` L28 `sys.path.insert(0, os.path.join(...))`.
- Inconsistent root resolution across normalize scripts.

## Acceptance Criteria
- [ ] **AC1** — Zero `print()` in `scripts/etl/` (use `logging.getLogger(__name__)`; `basicConfig` only in `__main__`).
- [ ] **AC2** — Zero bare `except Exception`; each catches specific exceptions and logs via `logger.exception()`. Any unavoidable broad catch carries `# noqa: BLE001 — <reason>`.
- [ ] **AC3** — `normalize_inventory_csv.py` uses `from common.core.paths import …`; no `sys.path.insert` / `os.path.join(__file__,..)`.
- [ ] **AC4** — `scripts/ai_checks/check_unenforced_rules.sh` passes for `scripts/etl/` with no new allowlist entries.
- [ ] **AC5** — Behavior unchanged (pure cleanup) — US1 tests still green.

## TDD Plan
Mostly enforced by the rule-gate + existing tests.
### Write first (red)
- `tests/unit/test_etl_rule_compliance.py::test_no_print_in_etl_scripts`
- `::test_no_bare_except_in_etl_scripts`
- `::test_no_path_hacks_in_etl_scripts`
(grep-based assertions over `scripts/etl/`)
### Then implement (green) → Refactor
- Mechanical replacements; verify with the rule-gate script.

## Implementation Notes
- Preserve log content/levels (info for progress, exception for errors).
- Don't add allowlist entries — fix the code (per MEMORY.md note).

## Definition of Done
- [ ] Rule gate clean for `scripts/etl/`.
- [ ] `tests/unit/test_etl_rule_compliance.py`, US1 tests, `make test-all` green.
