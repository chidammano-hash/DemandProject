# Developer Tools — Claude Code Skills, Agents & Commands

> A developer guide to the 7 skills, 10 agents, and 7 commands installed in `.claude/` for the DemandProject. These are tailored for the Python + FastAPI + React + PostgreSQL stack used here. Skills auto-activate based on context, agents are spawned for specialized tasks, and commands are invoked manually with slash syntax.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (developer tooling) |
| **Key Files** | `.claude/skills/`, `.claude/agents/`, `.claude/commands/` |

---

## Problem

A platform with 76 API routers, 21 frontend tabs, dozens of computation scripts, and strict architectural conventions (psycopg3 `%s` placeholders, `get_conn()` for inv_planning routers, Vite proxy entries, mandatory test coverage) needs disciplined developer workflows. Without standardized tooling, contributors miss conventions, skip tests, or commit security issues. The team needs always-on context for Claude (skills), specialized review subprocesses (agents), and explicit slash commands for common workflows (commands) — all tailored to this codebase.

---

## Solution

Three layers of developer tooling installed in `.claude/`:

- **Skills** — always-on context that auto-activates based on what the developer is working on (e.g., writing SQL triggers `postgres-patterns`).
- **Agents** — specialized subprocesses with focused roles and limited tool access (e.g., `code-reviewer` scans git diffs for issues).
- **Commands** — explicit slash commands that expand into full prompts and usually invoke an agent under the hood (e.g., `/tdd` invokes the `tdd-guide` agent).

---

## How It All Fits Together

```
You type a prompt
       │
       ├─→ Claude reads active SKILLS (auto-activated based on topic)
       │         e.g. writing SQL? → postgres-patterns activates
       │
       ├─→ Claude may spawn an AGENT (delegated subprocess)
       │         e.g. complex feature? → planner agent researches & plans
       │
       └─→ You can also invoke COMMANDS directly with /slash syntax
                 e.g. /code-review → scans git diff for issues
```

**Skills** are always-on context. They load automatically when Claude detects relevant work — no invocation needed.

**Agents** are spawned by Claude (or explicitly by you) to handle specialized tasks in a focused subprocess with limited tools.

**Commands** are explicit slash commands you type. Most of them invoke an agent under the hood.

---

## Skills

Skills live in `.claude/skills/`. Claude auto-activates them based on what you're working on. You don't invoke skills manually — they provide background context that shapes Claude's responses.

### `python-patterns`
**Activates when:** Writing or reviewing Python code (FastAPI routers, scripts, common modules)

**What it does:**
- Enforces PEP 8, type hints, and Pythonic idioms
- Prefers EAFP (try/except) over LBYL (if/check) error handling
- Pushes for small functions, readable names, modern Python 3.9+ syntax

**Relevant to DemandProject:** All files in `api/routers/`, `api/core.py`, `common/`, `scripts/`

---

### `postgres-patterns`
**Activates when:** Writing SQL, designing schemas, or troubleshooting query performance

**What it does:**
- Index cheatsheet (B-tree for equality/range, GIN for text search, partial for sparse columns)
- UPSERT patterns (`INSERT ... ON CONFLICT DO UPDATE`)
- Cursor pagination over OFFSET for large tables
- Anti-patterns to avoid: `SELECT *`, implicit casting, missing indexes on FK columns

**Relevant to DemandProject:** All files in `sql/`, queries inside `api/routers/`, materialized view refresh patterns.

---

### `tdd-workflow`
**Activates when:** Building new features, fixing bugs, or refactoring

**What it does:**
- Enforces write-tests-first discipline
- Walks through RED (failing test) → GREEN (minimal impl) → REFACTOR cycle
- Targets 80% minimum, 100% for financial/business-critical logic

**Relevant to DemandProject:** Aligns with the mandatory testing rules in CLAUDE.md. Use whenever adding new endpoints, components, or Python modules.

---

### `api-design`
**Activates when:** Adding new FastAPI endpoints or reviewing API contracts

**What it does:**
- URL naming: kebab-case, plural nouns (`/production-forecasts`, not `/getProductionForecast`)
- Correct HTTP methods (GET=read, POST=create, PUT=replace, PATCH=partial, DELETE=remove)
- Consistent status codes (200, 201, 204, 400, 401, 404, 422, 500)
- Pagination: offset/limit with `total` in response envelope

**Relevant to DemandProject:** Every new router module in `api/routers/`. Critical: remember to add new path prefixes to `frontend/vite.config.ts`.

---

### `verification-loop`
**Activates when:** Completing a feature, preparing for PR, or after refactoring

**What it does:**
Runs 6 checks in order:
1. Build — does the code compile/start?
2. Types — TypeScript errors? mypy errors?
3. Lint — ruff issues? ESLint warnings?
4. Tests — all pass? coverage threshold met?
5. Security — secrets exposed? SQL injection risk?
6. Diff review — what actually changed?

**Relevant to DemandProject:** Use before every PR. DemandProject commands: `make test-all` (backend + frontend), `uv run ruff check .`, `uv run mypy .`

---

### `security-review`
**Activates when:** Adding authentication, handling user input, creating endpoints that write data, or working with secrets

**What it does:**
- Checks for hardcoded secrets (API keys, DB passwords)
- SQL injection prevention (parameterized queries with `%s`, never f-strings)
- Input validation at system boundaries
- XSS prevention (sanitized HTML in frontend)
- CORS configuration review

**Relevant to DemandProject:** FastAPI endpoints in `api/routers/`, `api/auth.py` (API key guard), OpenAI/Anthropic API key handling in `.env`.

---

### `forecasting-patterns`
**Activates when:** Editing or reviewing backtest, champion selection, tuning, feature selection, production forecast generation, or accuracy/FVA endpoints

**What it does:**
- Documents the backtest -> champion -> production forecast lifecycle and per-cluster training
- Enforces the model-registry rule (`common/ml/model_registry.py` for all tree `.fit()`/instantiation) and the YAML-only hyperparameter rule
- Covers leakage guards, cold-start/intermittent routing, and the WAPE/bias accuracy formulas

**Relevant to DemandProject:** `common/ml/`, `scripts/ml/`, `scripts/forecasting/`, `api/routers/forecasting/`. Full design surface: `docs/specs/02-forecasting/`; master config: `config/forecasting/forecast_pipeline_config.yaml`.

---

## Agents

Agents are specialized subprocesses with a focused role and limited tool access. Claude spawns them automatically for matching tasks, or you can ask Claude to "use the planner agent" / "run the code-reviewer agent" explicitly.

Agents live in `.claude/agents/`.

---

### `planner`
**Model:** Opus | **Tools:** Read, Grep, Glob (read-only)

**Use when:** You need a structured implementation plan for a complex feature, architectural change, or multi-file refactor.

**Trigger phrases:**
- "I want to add X feature — plan it out"
- "How should I structure the Y module?"
- "Plan the migration from A to B"

**What it produces:**
A phased implementation plan with:
- Requirements + assumptions
- Architecture changes (exact file paths)
- Step-by-step implementation order with dependencies
- Testing strategy
- Risks & mitigations
- Success criteria checklist

**DemandProject example:**
```
Plan implementing the Z feature. It needs:
- A new SQL table
- A FastAPI router
- A React panel in InvPlanningTab
- Tests for backend + frontend
```

---

### `tdd-guide`
**Model:** Sonnet | **Tools:** Read, Write, Edit, Bash, Grep

**Use when:** Implementing any new feature, fixing a bug, or refactoring existing code.

**Trigger phrases:**
- `/tdd implement the X endpoint`
- "Use TDD to add the safety stock calculator"
- "Write tests first for this new router"

**Workflow it enforces:**
1. Define interface/types
2. Write failing tests (RED)
3. Run tests — verify they fail
4. Write minimal implementation (GREEN)
5. Run tests — verify they pass
6. Refactor — verify tests still pass
7. Check coverage ≥ 80%

**DemandProject example:**
```
/tdd Add a new endpoint GET /supply/overview that returns
on-hand, on-order, and coverage days per location
```

---

### `code-reviewer`
**Model:** Sonnet | **Tools:** Read, Grep, Glob, Bash

**Use when:** After writing or modifying any code, before committing.

**Trigger phrases:**
- `/code-review`
- "Review the changes I just made"
- "Check this router for issues"

**What it checks (from git diff):**
- CRITICAL: Hardcoded secrets, SQL injection, missing auth
- HIGH: Functions >50 lines, missing error handling, no tests for new code
- MEDIUM: Mutation patterns, missing type hints, TODOs left in code

**DemandProject note:** Claude will automatically use this agent after making significant code changes. You can also invoke it explicitly before committing.

---

### `database-reviewer`
**Model:** Sonnet | **Tools:** Read, Write, Edit, Bash, Grep, Glob

**Use when:** Writing new SQL files, adding indexes, designing schema changes, or troubleshooting slow queries.

**Trigger phrases:**
- "Review this SQL migration"
- "Is this query optimal?"
- "Check the indexes on this table"

**What it reviews:**
- Missing indexes on WHERE/JOIN columns
- Explain plan analysis (EXPLAIN ANALYZE)
- Materialized view refresh strategy
- Data type selection (use `NUMERIC` not `FLOAT` for money)
- N+1 patterns in ORM-style queries
- RLS policy correctness

**DemandProject note:** Especially useful when adding new tables to `sql/` or new SQL queries inside `api/routers/`. Knows about `%s` placeholder convention (psycopg3, not `$1`).

---

### `python-reviewer`
**Model:** Sonnet | **Tools:** Read, Grep, Glob, Bash

**Use when:** After writing Python code — routers, scripts, common modules, or tests.

**Trigger phrases:**
- "Review this Python file"
- "Check the FastAPI router for issues"
- "Run python-reviewer on my changes"

**What it runs:**
- `uv run ruff check` — linting
- `uv run mypy` — type checking
- Manual checks: SQL injection via f-strings, command injection in `subprocess`, path traversal, missing type hints, Pythonic patterns

**DemandProject note:** Catches common issues like using `$1` instead of `%s` in psycopg3, missing `async` on FastAPI endpoints, and `get_conn()` vs `Depends(_get_pool)` misuse.

---

### `design-developer`
**Model:** Sonnet | **Tools:** Read, Edit, Write, Bash, Grep, Glob

**Use when:** Implementing one scoped UI/UX increment (code + tests) inside a design-pod loop, against an exact token contract handed down by the orchestrator.

**Trigger phrases:**
- Spawned by the design-pod orchestrator (manager + product designer critiquing the live app via Playwright), one per developer per loop

**What it does:**
- Implements only its assigned, disjoint file slice so two design-developers can run in parallel without colliding
- Enforces token rules: no inline hex in `src/tabs/`/`src/components/`, charts read theme via `useThemeContext()`/`useChartColors()` (never a `theme` prop), all three color modes (light/soft/dark) handled
- Updates and runs tests for every component/hook it touches; self-reviews its diff before reporting

**DemandProject note:** Never commits. Keeps tab files under the 600-line limit, splitting into sub-panels when needed.

---

### `forecasting-developer`
**Model:** Opus | **Tools:** Read, Edit, Write, Bash, Grep, Glob

**Use when:** Implementing one scoped forecasting-accuracy increment (code + tests) inside the autonomous forecasting pod.

**Trigger phrases:**
- Spawned by the Forecasting Manager with exactly one increment per invocation

**What it does:**
- Scoped to `api/routers/forecasting/`, `common/ml/`, `scripts/forecasting/`, `scripts/ml/`, `config/forecasting/*.yaml`, new SQL migrations, and the matching tests
- Test-first: writes or extends the failing test that encodes the increment's success criterion before implementing
- Enforces the forecasting hard rules from CLAUDE.md (`get_planning_date()`, `model_registry.py` for tree `.fit()`, `FORECAST_QTY_COL`, `read_sql_chunked()` over fact tables)

**DemandProject note:** Never commits and never self-certifies - `forecasting-qa` and the two SME agents gate whether the increment ships.

---

### `forecasting-qa`
**Model:** Sonnet | **Tools:** Read, Bash, Grep, Glob (read-only)

**Use when:** Independently verifying a `forecasting-developer` increment before it ships.

**Trigger phrases:**
- Spawned by the Forecasting Manager after `forecasting-developer` reports done

**What it checks:**
- Reproduces the developer's claimed tests plus a broader `pytest tests/ -q` run for anything plausibly affected
- Lint delta (`ruff check` on changed files - any new E/F error is an automatic FAIL)
- CLAUDE.md rule scan on changed files (`date.today(`, bare `except`, `$1` placeholders, literal `"basefcst_pref"`, `Depends(_get_pool)` in `inv_planning_*`, missing `require_api_key` on writes, magic numbers)
- Test coverage exists for new endpoints/modules, and any accuracy claim came from a causally valid train-past/eval-future backtest

**DemandProject note:** Read-only on code - never edits, never commits. Returns a PASS/FAIL verdict with an ordered blocker list.

---

### `forecasting-sme-commercial`
**Model:** Opus | **Tools:** Read, Grep, Glob, Bash (read-only)

**Use when:** Judging whether a proposed forecasting increment is realistic and useful for the business - seasonality, promotions, long-tail SKUs, intermittent demand, cold-start, and whether planners would trust the result.

**Trigger phrases:**
- Spawned by the Forecasting Manager alongside `forecasting-sme-statistical` to review an increment

**What it judges:**
- Decision usefulness for planners, seasonality/promo-timing handling, long-tail/low-volume realism, bias direction (stockout vs. write-off cost asymmetry), planner trust, and whether the change fits the monthly planning cadence

**DemandProject note:** Advises only, never edits code. Returns a USEFUL / NOT-USEFUL / USEFUL-NICHE verdict.

---

### `forecasting-sme-statistical`
**Model:** Opus | **Tools:** Read, Grep, Glob, Bash (read-only)

**Use when:** Judging whether a proposed accuracy increment is methodologically sound - causal validity, leakage, overfitting, metric integrity, model/cluster appropriateness.

**Trigger phrases:**
- Spawned by the Forecasting Manager alongside `forecasting-sme-commercial` to review an increment

**What it judges:**
- Causal validity (train-past/eval-future, no target leakage or hindsight model selection), generalization vs. coverage/selection bias, metric integrity (WAPE/bias formulas at the right grain), model appropriateness per cluster volatility, and whether a delta survives seed/run noise

**DemandProject note:** Advises only, never edits code. Returns a SOUND / UNSOUND / SOUND-BUT-IMMATERIAL verdict.

---

## Commands

Commands are slash commands you type directly. They expand into full prompts and usually invoke an agent.

---

### `/tdd [description]`
Invokes the `tdd-guide` agent to implement something using test-driven development.

```
/tdd Add a GET /supply/scenarios endpoint that returns scenario data
/tdd Fix the bug where champion selection fails for single-DFU portfolios
/tdd Refactor the blended_forecast router to use the shared db pattern
```

**Workflow:** Define interface → Write failing tests → Implement → Refactor → Verify coverage

---

### `/plan [description]`
Invokes the `planner` agent to produce a phased implementation plan before any code is written.

```
/plan Implement the supply scenario feature (new SQL table + API router + React panel)
/plan Refactor all inv_planning_*.py routers to use a shared base class
/plan Add webhook support for PO receipt events
```

**Output:** Phased plan with file paths, dependencies, risks, and testing strategy. Use this before any feature that touches 3+ files.

---

### `/code-review`
Scans `git diff HEAD` and produces a severity-rated report.

```
/code-review
/code-review          ← reviews all uncommitted changes
```

**Severity levels:**
- CRITICAL: security vulnerabilities → blocks merge
- HIGH: code quality issues → should fix before PR
- MEDIUM: best practice violations → fix if time allows
- LOW: style suggestions

---

### `/build-fix`
Invokes the `build-error-resolver` agent to diagnose and fix build/type errors.

```
/build-fix            ← analyzes current build output
/build-fix TypeScript errors in InvPlanningTab.tsx
/build-fix mypy errors after adding type hints to production_forecast.py
```

**How it works:** Reads error output → identifies root cause → applies incremental fixes → verifies after each fix

---

### `/quality-gate`
Runs the full quality gate before creating a PR.

```
/quality-gate
/quality-gate pre-pr  ← most thorough
```

**Checks in order:** Build → Types → Lint → Tests → Security scan → Coverage → Diff review

**DemandProject equivalent:** Same as running `make test-all` + ruff + mypy + manual security check, but automated and reported in one pass.

---

### `/verify [mode]`
Runs verification checks and produces a PASS/FAIL report.

```
/verify               ← full check (default)
/verify quick         ← build + types only (fast)
/verify pre-commit    ← checks relevant before committing
/verify pre-pr        ← full checks + security scan
```

**Output format:**
```
VERIFICATION: PASS/FAIL

Build:    OK/FAIL
Types:    OK/X errors
Lint:     OK/X issues
Tests:    X/Y passed, Z% coverage
Secrets:  OK/X found
Logs:     OK/X console.logs

Ready for PR: YES/NO
```

---

### `/ux-loop [cycles]`
Runs the persona-driven critique -> fix loop against the **live** app for N cycles (default 5).

```
/ux-loop
/ux-loop 3
```

**How it works:** Playwright drives 14 planner tabs and screenshots them, a Demand-Planner agent and a Usability/Simplification agent find issues from the screenshots plus code, then a Technical-Fixer agent applies fixes under strict test-first TDD and re-validates against the live endpoints. Findings and red/green evidence land in `tests/Automated_tests/`.

**DemandProject note:** Edits the working tree but does not commit - review with `git diff` afterward. Requires a host `uvicorn --reload` on `:8000` (the Docker API image has no `--reload`) and the Vite dev server on `:5173`.

---

## Scale Test Framework (`tests/scale/`)

Performance regressions in dashboard endpoints typically only show up against
production-size data. The scale-test suite at `tests/scale/` materializes
synthetic data on demand and runs hot-path endpoints against it, gated by the
`scale` pytest marker so it never runs in `make test`.

| Component | Path |
|---|---|
| Suite root | `tests/scale/` |
| Customer Analytics | `tests/scale/test_customer_analytics_scale.py` |
| Inventory Planning | `tests/scale/test_inv_planning_scale.py` |
| Shared fixtures | `tests/scale/conftest.py` |
| Make target | `make scale-test` |

### Scale Knob

The harness accepts a `--scale=<rows>` pytest CLI flag (read by `conftest.py`)
that drives the synthetic data volume. The Makefile target wires this through
the `SCALE` env var:

```bash
make scale-test                    # default 100K rows (CI-friendly, ~minutes)
make scale-test SCALE=10000000     # nightly: 10M rows ≈ 40× production
```

40× scale is the threshold at which the customer-analytics dashboard exceeds
the 30s `statement_timeout` without the MV-routed query path (see
`03-demand-intelligence/07-customer-analytics.md` "Performance Architecture").
The nightly run guards against re-introducing fact-table scans on the hot
endpoints.

---

## Recommended Workflows

### Starting a new feature

```
1. /plan [describe the feature]
   → get phased plan with file paths

2. /tdd [implement phase 1]
   → write tests first, then implement

3. /code-review
   → catch issues before they accumulate

4. Repeat /tdd for each phase

5. /verify pre-pr
   → confirm everything passes before PR
```

### Fixing a bug

```
1. /tdd [describe the bug]
   → write a test that reproduces the bug first
   → implement the fix
   → verify test passes

2. /code-review
   → check the fix didn't introduce new issues
```

### Adding a new API router (DemandProject pattern)

```
1. /plan Add [feature] router
   → confirms: new sql/ file, new api/routers/ file,
     new tests/api/ file, new frontend query, Vite proxy entry

2. /tdd implement the router
   → uses httpx.AsyncClient + ASGITransport pattern
   → uses make_pool() from tests/api/conftest.py

3. database-reviewer agent
   → verify SQL uses %s (not $1), indexes present

4. python-reviewer agent
   → verify get_conn() used (not Depends(_get_pool))

5. /verify pre-pr
```

### Before every PR

```
/verify pre-pr
```

This runs: build → types → lint → `make test-all` → security → coverage → diff review.

---

## Quick Reference

| What you want | Use |
|---|---|
| Implement a new feature with tests | `/tdd [description]` |
| Plan a complex multi-file feature | `/plan [description]` |
| Review code before committing | `/code-review` |
| Fix build or type errors | `/build-fix` |
| Run all quality checks | `/quality-gate` or `/verify pre-pr` |
| Harden the live UI over several cycles | `/ux-loop [cycles]` |
| Writing Python code | `python-patterns` skill (auto) |
| Writing SQL or schemas | `postgres-patterns` skill (auto) |
| Building or refactoring with tests | `tdd-workflow` skill (auto) |
| Designing new endpoints | `api-design` skill (auto) |
| Preparing for a PR | `verification-loop` skill (auto) |
| Security-sensitive code | `security-review` skill (auto) |
| Backtest/champion/production forecasting work | `forecasting-patterns` skill (auto) |
| Complex feature planning | `planner` agent (auto or explicit) |
| TDD enforcement | `tdd-guide` agent (via `/tdd`) |
| Post-change review | `code-reviewer` agent (auto or `/code-review`) |
| SQL/schema review | `database-reviewer` agent (explicit) |
| Python code review | `python-reviewer` agent (explicit) |
| Scoped UI/UX design increment | `design-developer` agent (design-pod loop) |
| Scoped forecasting-accuracy increment | `forecasting-developer` agent (forecasting pod) |
| QA gate on a forecasting increment | `forecasting-qa` agent (forecasting pod) |
| Commercial-realism review of a forecasting change | `forecasting-sme-commercial` agent (forecasting pod) |
| Methodological-soundness review of a forecasting change | `forecasting-sme-statistical` agent (forecasting pod) |

---

## File Locations

```
.claude/
├── skills/
│   ├── python-patterns/SKILL.md
│   ├── postgres-patterns/SKILL.md
│   ├── tdd-workflow/SKILL.md
│   ├── api-design/SKILL.md
│   ├── verification-loop/SKILL.md
│   ├── security-review/SKILL.md
│   └── forecasting-patterns/SKILL.md
├── agents/
│   ├── planner.md
│   ├── tdd-guide.md
│   ├── code-reviewer.md
│   ├── database-reviewer.md
│   ├── python-reviewer.md
│   ├── design-developer.md
│   ├── forecasting-developer.md
│   ├── forecasting-qa.md
│   ├── forecasting-sme-commercial.md
│   └── forecasting-sme-statistical.md
└── commands/
    ├── tdd.md
    ├── plan.md
    ├── code-review.md
    ├── build-fix.md
    ├── quality-gate.md
    ├── verify.md
    └── ux-loop.md
```

---

## Dependencies

| Dependency | Reason |
|---|---|
| Claude Code CLI | Hosts the skills, agents, and commands |
| `.claude/settings.json` | Configures PostToolUse and PreToolUse hooks that auto-run ruff, anti-pattern checks, and tests |
| `uv`, `ruff`, `mypy`, `pytest` | Python tooling invoked by `python-reviewer` and `verification-loop` |
| `vitest`, `@playwright/test` | Frontend tooling invoked by `verify` and `quality-gate` |

---

## See Also

- `07-user-experience/05-testing.md` -- testing pyramid that verification commands exercise
- CLAUDE.md "Automatic Quality Workflow" section -- always-on rules driving auto-agent invocation
