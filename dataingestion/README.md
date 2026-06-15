# Data Ingestion Streamlining — User Stories

Epic: **Unify the data ingestion pipeline (full load + incremental refresh), consolidate code, improve performance & observability, and make every load runnable/monitorable from the UI.**

Scope: all 13 scripts in `scripts/etl/`, plus `common/core/`, `common/services/`, the API ingestion routers, and the UI tabs (`IntegrationTab`, `DataQualityTab`, `JobsTab`).

## Why (the core problem)

There are **two divergent load engines** and **three job/exec systems**:

| | CLI / Make path | UI path |
|---|---|---|
| Load engine | `load_dataset_postgres.py` | `load.py` (separate impl) |
| Orchestrator | `run_pipeline.py` (`--mode full`/`refresh`) | `IntegrationRunner` (subprocess) |
| Job backend | none (shell) | `integration_jobs` table |
| Other jobs | — | `JobManager`/APScheduler → `job_history` |

The streamline collapses these into **one load engine**, **one orchestrator**, **one job backend**, exposed through the UI.

## Delivery order

Foundation first — each story ships behind `make test-all`, TDD (tests written first), and the CLAUDE.md self-review pass.

| Story | Title | Phase | Depends on | Complexity | Risk |
|---|---|---|---|---|---|
| [US1](UserStory1.md) | Characterization tests for existing loaders | 0 Baseline | — | M | LOW |
| [US2](UserStory2.md) | Baseline timing benchmarks | 0 Baseline | — | S | LOW |
| [US3](UserStory3.md) | Shared index/constraint management helper | 1 Core | US1 | M | MED |
| [US4](UserStory4.md) | Shared staging + partition management | 1 Core | US3 | M | MED |
| [US5](UserStory5.md) | Shared DFU/FK filtering + audit_load_batch writer | 1 Core | US3 | M | MED |
| [US6](UserStory6.md) | Single mode-parameterized load engine | 2 Unify | US3,US4,US5 | L | HIGH |
| [US7](UserStory7.md) | Repoint IntegrationRunner, delete load.py | 2 Unify | US6 | M | HIGH |
| [US8](UserStory8.md) | Push DFU filtering to normalize time | 3 Perf | US5,US6 | M | MED |
| [US9](UserStory9.md) | Conditional / streamed forecast archive load | 3 Perf | US6 | M | MED |
| [US10](UserStory10.md) | Size-based index drop/recreate | 3 Perf | US3,US6 | S | LOW |
| [US11](UserStory11.md) | COPY/executemany for load_open_pos.py | 3 Perf | US1 | M | MED |
| [US12](UserStory12.md) | Move magic numbers to etl_config.yaml | 3 Perf | — | S | LOW |
| [US13](UserStory13.md) | Transaction isolation for multi-step loads | 4 Reliab | US11 | M | MED |
| [US14](UserStory14.md) | Logging + exception + path-hack cleanup | 4 Reliab | — | S | LOW |
| [US15](UserStory15.md) | customer_demand in change detection | 4 Reliab | US5 | S | LOW |
| [US16](UserStory16.md) | Register etl_pipeline job type (full+refresh) | 5 Orchestr | US6 | M | MED |
| [US17](UserStory17.md) | Converge job backends into one | 5 Orchestr | US16 | L | HIGH |
| [US18](UserStory18.md) | API endpoints for full + incremental run | 5 Orchestr | US16 | M | MED |
| [US19](UserStory19.md) | UI: full + incremental triggers w/ live status | 6 UI | US18 | M | MED |
| [US20](UserStory20.md) | UI: unified load history & lineage | 6 UI | US18 | S | LOW |
| [US21](UserStory21.md) | Docs + final verification | 7 Verify | all | S | LOW |

## Conventions every story follows

- **TDD**: tests written first (red), minimal implementation (green), then refactor + self-review.
- **CLAUDE.md hard rules**: `%s` placeholders, no bare `except Exception`, no `print()` in scripts, `get_planning_date()` not `date.today()`, `common.core.paths` not `parents[N]`, config in YAML, write endpoints guarded by `Depends(require_api_key)`, docs updated in the same commit.
- **No backward-compat shims** — when a module moves, rewrite all importers in the same change.
- **Test commands** (from MEMORY.md): backend `~/.local/bin/uv run pytest tests/ -q`; frontend (from `frontend/`) `PATH="/opt/homebrew/bin:$PATH" /opt/homebrew/bin/node node_modules/.bin/vitest run --reporter=dot`.

## Definition of Done (epic)

- One load engine, one orchestrator, one job backend.
- Full load and incremental refresh both triggerable + monitorable from the UI for any/all domains.
- `audit_load_batch` covers every domain incl. `customer_demand`.
- Before/after benchmarks recorded in `docs/RUNBOOK.md`.
- All `scripts/etl/` rule violations resolved; `make test-all`, `make audit-routers`, lint, type-check green.
