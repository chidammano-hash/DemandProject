# Developer Tools — Claude Code Skills, Agents & Commands

> A developer guide to the 9 skills, 5 agents, and 6 commands installed in `.claude/` for the DemandProject. These are tailored for the Python + FastAPI + React + PostgreSQL stack used here. Skills auto-activate based on context, agents are spawned for specialized tasks, and commands are invoked manually with slash syntax.

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

### `python-testing`
**Activates when:** Writing pytest tests or designing test fixtures

**What it does:**
- Guides RED→GREEN→REFACTOR cycle with pytest
- Suggests fixture patterns, parametrize, and mock strategies
- Targets 80%+ coverage with edge cases and error paths

**Relevant to DemandProject:** All files in `tests/unit/` and `tests/api/`. Aligns with the existing `conftest.py` mock pool pattern (`make_pool()`).

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

### `backend-patterns`
**Activates when:** Designing API endpoints, background jobs, or data access layers

**What it does:**
- Repository pattern: encapsulate DB access behind a standard interface
- Service layer: business logic decoupled from HTTP layer
- N+1 query prevention
- Caching strategies for hot paths

**Relevant to DemandProject:** FastAPI router architecture, `api/core.py` connection pool patterns, APScheduler job design in `common/services/job_registry.py`.

---

### `frontend-patterns`
**Activates when:** Building React components, managing state, optimizing rendering

**What it does:**
- Composition over inheritance (compound components, render props)
- TanStack Query for server state (stale-while-revalidate, cache invalidation)
- Performance: `useMemo`, `useCallback`, `React.memo` — only where measured
- Lazy loading and code splitting

**Relevant to DemandProject:** All files in `frontend/src/tabs/`, `frontend/src/components/`, hooks in `frontend/src/hooks/`. Matches the existing TanStack Query + Recharts + shadcn/ui stack.

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
| Writing Python code | `python-patterns` skill (auto) |
| Writing pytest tests | `python-testing` skill (auto) |
| Writing SQL or schemas | `postgres-patterns` skill (auto) |
| Building React components | `frontend-patterns` skill (auto) |
| Designing new endpoints | `api-design` skill (auto) |
| Security-sensitive code | `security-review` skill (auto) |
| Complex feature planning | `planner` agent (auto or explicit) |
| TDD enforcement | `tdd-guide` agent (via `/tdd`) |
| Post-change review | `code-reviewer` agent (auto or `/code-review`) |
| SQL/schema review | `database-reviewer` agent (explicit) |
| Python code review | `python-reviewer` agent (explicit) |

---

## File Locations

```
.claude/
├── skills/
│   ├── python-patterns/SKILL.md
│   ├── python-testing/SKILL.md
│   ├── postgres-patterns/SKILL.md
│   ├── tdd-workflow/SKILL.md
│   ├── backend-patterns/SKILL.md
│   ├── frontend-patterns/SKILL.md
│   ├── api-design/SKILL.md
│   ├── verification-loop/SKILL.md
│   └── security-review/SKILL.md
├── agents/
│   ├── planner.md
│   ├── tdd-guide.md
│   ├── code-reviewer.md
│   ├── database-reviewer.md
│   └── python-reviewer.md
└── commands/
    ├── tdd.md
    ├── plan.md
    ├── code-review.md
    ├── build-fix.md
    ├── quality-gate.md
    └── verify.md
```

Source: `everything-claude-code-main/` — see that directory for 65+ additional skills and 16 agents not installed here.

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
