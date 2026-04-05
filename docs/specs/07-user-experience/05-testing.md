# Testing

> The full-stack testing strategy: pytest for backend (1636+ tests), Vitest + React Testing Library for frontend (457+ tests), and Playwright for end-to-end browser smoke tests (8 test files). All three layers run without external infrastructure -- the database is mocked in backend and frontend tests; only E2E tests require running services.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (backend only) |
| **Key Files** | `tests/unit/`, `tests/api/`, `frontend/src/**/__tests__/`, `frontend/e2e/tests/`, `vitest.config.ts`, `e2e/playwright.config.ts` |

---

## Problem

A platform with 53 API routers, 16 frontend tabs, 28 inventory planning sub-tabs, and dozens of computation scripts needs automated test coverage to prevent regressions as features accumulate. Without tests, every change risks breaking something in a distant part of the codebase. The test suite must run fast enough to execute on every change (under 5 seconds for unit/integration, under 30 seconds for E2E) and must not require a running database or external services for the unit/integration layer.

---

## Solution

A three-layer testing pyramid: unit tests at the base (fastest, most numerous), integration tests in the middle (API endpoint tests with mocked DB), and E2E smoke tests at the top (real browser, real servers, fewest tests). Mandatory test requirements ensure every new feature ships with corresponding tests.

---

## How It Works

### Testing Pyramid

| Layer | Tool | Count | Speed | Infrastructure |
|---|---|---|---|---|
| Backend unit | pytest | 1636+ | ~0.7s | None (all mocked) |
| Frontend unit/integration | Vitest + RTL | 457+ | ~1.5s | None (all mocked) |
| E2E smoke | Playwright | ~15 tests (8 files) | ~30s | API on :8000 + Vite on :5173 |

### Backend Testing (pytest)

**Unit tests** (`tests/unit/`): Test Python modules in `common/` -- feature engineering, metrics, champion strategies, exception engine, safety stock computation, etc. Pure function tests with no I/O.

**API tests** (`tests/api/`): Test FastAPI endpoints using `httpx.AsyncClient` with `ASGITransport(app)`. The database connection pool is mocked via the `make_pool` factory in `tests/api/conftest.py`.

| Pattern | When to Use |
|---|---|
| `cursor.fetchall.return_value = [...]` | Endpoint makes one `fetchall()` call |
| `cursor.fetchall.side_effect = [list1, list2]` | Endpoint makes multiple `fetchall()` calls |
| `cursor.fetchone.return_value = (value,)` | Endpoint reads a single row |
| `@patch("api.routers.X.get_conn")` | Mock DB connection for `inv_planning_*` routers |
| `@patch("api.core._get_pool")` | Mock pool for all other routers |

### Frontend Testing (Vitest + React Testing Library)

Components are wrapped with `TestQueryWrapper` from `test-utils.tsx` (provides TanStack Query client). API calls are mocked with `vi.mock("../api/queries")`.

| Pattern | When to Use |
|---|---|
| `vi.mock("echarts-for-react")` | Any component using ECharts |
| `vi.mock("@tanstack/react-virtual")` | Components with virtualized scrolling (DataTable) |
| `vi.mock("../api/queries")` | Mock all API fetch functions |
| `getByRole("button", { name })` | Preferred selector for buttons |
| `getByText()` | Fallback when role selectors match multiple elements |

### E2E Testing (Playwright)

Smoke tests verify critical user journeys in a real Chromium browser against running dev servers.

**Configuration** (`e2e/playwright.config.ts`):

| Setting | Value | Reason |
|---|---|---|
| `baseURL` | `http://localhost:5173` | Vite dev server |
| `timeout` | 30 seconds | Generous for CI environments |
| `retries` | 2 in CI, 0 locally | Flaky test resilience |
| `workers` | 1 in CI | Prevent resource contention |
| `webServer.command` | `npm run dev` | Auto-starts Vite (API must run separately) |
| `trace` | On first retry | Debug flaky tests |
| `screenshot` | Only on failure | CI artifact for diagnosis |

**Test Files:**

| File | What It Tests |
|---|---|
| `navigation.spec.ts` | Sidebar renders 16 items, click each tab loads content, keyboard shortcuts 1-9, sidebar collapse |
| `dashboard.spec.ts` | KPI cards visible, alert panel renders, top movers renders |
| `accuracy.spec.ts` | WAPE/Accuracy/Bias KPI cards, trend chart container |
| `global-filters.spec.ts` | Filter bar visibility per tab, dropdown opens, filter persists across tabs, clear all resets |
| `inv-planning.spec.ts` | Sub-tab sidebar renders 7 groups, click sub-tab loads panel, panel header shows title |
| `ai-planner.spec.ts` | Portfolio health bar renders, insight cards or healthy empty state |
| `control-tower.spec.ts` | KPI section visible, alert panel visible |
| `theme.spec.ts` | Dark mode toggle adds `dark` class, light toggle removes it, `d` shortcut works |

**Shared Fixtures** (`e2e/fixtures/base.ts`): `navigateToTab()` helper, sidebar label constants, filter bar tab lists.

**Selector Strategy:** Semantic selectors only (`getByRole`, `getByText`, `toBeVisible`, `toHaveURL`). Never CSS classes or fragile DOM paths.

---

## Mandatory Test Requirements

| Change Type | Required Test |
|---|---|
| New Python module in `common/` | Unit test in `tests/unit/test_<module>.py` |
| New API endpoint | API test in `tests/api/test_<feature>.py` |
| New React component | Component test in `__tests__/<Component>.test.tsx` |
| New React hook | Hook test in `__tests__/<hook>.test.ts` |
| New utility function | Test in `__tests__/<util>.test.ts` |
| New tab component | Smoke test in `__tests__/<Tab>.test.tsx` |
| New sidebar tab | E2E test in `navigation.spec.ts` |
| New Inv. Planning sub-tab | E2E test in `inv-planning.spec.ts` |
| Removed feature | Delete corresponding tests |

---

## Running Tests

| Command | What It Runs |
|---|---|
| `make test` | All backend pytest tests (~0.7s) |
| `make test-unit` | Backend unit tests only |
| `make test-api` | Backend API tests only |
| `make test-cov` | Backend with coverage report |
| `make ui-test` | Frontend Vitest tests (~1.5s) |
| `make test-all` | Backend + frontend (both suites) |
| `make e2e` | Playwright E2E (requires API on :8000) |
| `make e2e-ui` | Playwright interactive UI mode |
| `make e2e-headed` | Playwright with visible browser |
| `make e2e-report` | Open last HTML test report |
| `make e2e-install` | Install Playwright browsers (one-time) |

---

## Dependencies

| Dependency | Reason |
|---|---|
| `pytest`, `pytest-asyncio` | Backend test framework |
| `httpx` | Async HTTP client for API tests |
| `vitest`, `@testing-library/react` | Frontend test framework |
| `@playwright/test` (^1.52) | E2E browser automation |

---

## See Also

- `07-user-experience/02-ui-architecture.md` -- component architecture tested by frontend suite
- `07-user-experience/03-theming.md` -- theme toggle tested by E2E `theme.spec.ts`
