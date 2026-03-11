# 06-07 — End-to-End Testing (Playwright)

## Context

Demand Studio has 457+ unit/integration tests (Vitest + RTL) and 1636+ backend tests (pytest), but **zero end-to-end browser tests**. Unit tests mock the API layer and verify component rendering in jsdom — they cannot catch:

- Broken API proxy routes after deployment
- CSS/layout regressions across tabs
- Cross-tab filter state propagation failures
- Lazy-loading or code-splitting breakage
- Real browser rendering issues (scrolling, charts, responsive layout)

This spec adds **Playwright E2E smoke tests** (simulating real user actions post-deploy) — the same approach used by Shopify, Stripe, and Vercel.

---

## Architecture

### Testing Pyramid (Updated)

```
        ┌──────────────┐
        │  E2E Smoke    │  ← Playwright (10-15 tests, ~30s)
        │  (this spec)  │
        ├──────────────┤
        │  Integration  │  ← Vitest + RTL (457+ tests)
        ├──────────────┤
        │  Unit Tests   │  ← Vitest + pytest (2000+ tests)
        └──────────────┘
```

### Tool Selection

| Tool | Purpose | Why |
|---|---|---|
| **Playwright** | E2E browser automation | Best DX, auto-wait, multi-browser, trace viewer, Vite-native |

### E2E Test Scope

E2E tests are **smoke tests** — they verify critical user journeys work end-to-end against a running dev server + API. They do NOT replace unit tests.

**What E2E tests cover:**
- Page loads without errors
- Sidebar navigation works across all tabs
- Global filters render and apply
- Key data renders (KPI cards, charts, tables)
- Keyboard shortcuts function
- Light/dark mode toggle works

**What E2E tests do NOT cover:**
- Business logic validation (unit tests)
- Component prop behavior (integration tests)
- API response correctness (API tests)

---

## File Structure

```
mvp/demand/frontend/
├── e2e/
│   ├── fixtures/
│   │   └── base.ts              # Shared page fixtures (authenticated page, etc.)
│   ├── tests/
│   │   ├── navigation.spec.ts   # Sidebar nav + keyboard shortcuts
│   │   ├── dashboard.spec.ts    # Dashboard tab smoke test
│   │   ├── accuracy.spec.ts     # Accuracy tab smoke test
│   │   ├── global-filters.spec.ts  # Filter bar interaction
│   │   ├── inv-planning.spec.ts # Inv Planning sub-tab nav
│   │   ├── ai-planner.spec.ts   # AI Planner tab smoke test
│   │   ├── control-tower.spec.ts # Control Tower smoke test
│   │   └── theme.spec.ts        # Light/dark mode toggle
│   └── playwright.config.ts     # Playwright configuration
├── package.json                 # + @playwright/test
└── ...
```

---

## Playwright Configuration

```typescript
// e2e/playwright.config.ts
{
  testDir: './tests',
  baseURL: 'http://localhost:5173',
  timeout: 30_000,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  use: {
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev',
    port: 5173,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
}
```

**Key decisions:**
- Single browser (Chromium) for speed — expand to Firefox/WebKit later if needed
- `webServer` auto-starts Vite dev server (requires API running separately)
- Traces + screenshots only on failure to keep CI fast
- Retries in CI only (flaky test resilience)

---

## Smoke Test Specifications

### 1. Navigation (navigation.spec.ts)

| Test | User Action | Assertion |
|---|---|---|
| Sidebar renders all nav items | Load app | 14 nav buttons visible |
| Click each tab loads content | Click each sidebar item | No error boundary, tab content renders |
| Keyboard shortcuts 1-9 | Press number keys | Correct tab activates |
| Sidebar collapse/expand | Press `[` key | Sidebar toggles width |
| URL reflects active tab | Click "Accuracy" | URL contains `?tab=accuracy` |

### 2. Dashboard (dashboard.spec.ts)

| Test | User Action | Assertion |
|---|---|---|
| KPI cards render | Navigate to Overview | At least 4 KPI cards visible |
| Alert panel renders | Navigate to Overview | Alert panel section visible |
| Top movers render | Navigate to Overview | Top movers section visible |

### 3. Global Filters (global-filters.spec.ts)

| Test | User Action | Assertion |
|---|---|---|
| Filter bar visible on Overview | Navigate to Overview | 6 filter dropdowns visible |
| Filter bar hidden on AI Planner | Navigate to AI Planner | Filter bar not visible |
| Brand filter opens dropdown | Click Brand button | Dropdown popover opens |
| Filter persists across tabs | Select brand, switch to Accuracy | Brand filter still active |
| Clear all resets filters | Select filters, click clear | All filters reset |

### 4. Accuracy Tab (accuracy.spec.ts)

| Test | User Action | Assertion |
|---|---|---|
| KPI cards render | Navigate to Accuracy | WAPE/Accuracy/Bias cards visible |
| Trend chart renders | Navigate to Accuracy | Chart container visible |

### 5. Inventory Planning (inv-planning.spec.ts)

| Test | User Action | Assertion |
|---|---|---|
| Sub-tab sidebar renders | Navigate to Inv. Planning | Grouped sidebar with 7 groups |
| Click sub-tab loads panel | Click "EOQ" sub-tab | EOQ panel content renders |
| Panel header shows title | Click any sub-tab | Panel title + description visible |

### 6. AI Planner (ai-planner.spec.ts)

| Test | User Action | Assertion |
|---|---|---|
| Portfolio health bar renders | Navigate to AI Planner | Health KPI chips visible |
| Insight cards or empty state | Navigate to AI Planner | Either insight cards or "healthy" message |

### 7. Control Tower (control-tower.spec.ts)

| Test | User Action | Assertion |
|---|---|---|
| KPI cards render | Navigate to Control Tower | KPI section visible |
| Alert list renders | Navigate to Control Tower | Alert panel visible |

### 8. Theme Toggle (theme.spec.ts)

| Test | User Action | Assertion |
|---|---|---|
| Dark mode toggle | Click theme toggle in sidebar | `dark` class on `<html>` |
| Light mode toggle | Click theme toggle again | `dark` class removed |
| Keyboard shortcut `d` | Press `d` key | Theme toggles |

---

## Make Targets

```makefile
# E2E Testing
e2e-install:        # Install Playwright browsers (one-time)
e2e:                # Run Playwright E2E smoke tests
e2e-ui:             # Run Playwright in interactive UI mode
e2e-headed:         # Run with visible browser
e2e-report:         # Open last HTML test report
```

---

## Prerequisites

E2E tests require both servers running:
1. **API server** on `:8000` — `make api` (requires Postgres)
2. **Frontend dev server** on `:5173` — auto-started by Playwright's `webServer` config

For CI, both services must be started before the E2E test step.

---

## When to Update E2E Tests

**New E2E tests MUST be added when:**
- A new tab is added to the sidebar
- A new sub-tab is added to Inventory Planning
- The global filter bar gains/loses a filter dimension
- Navigation behavior changes (keyboard shortcuts, URL routing)
- A new critical user journey is introduced

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `@playwright/test` | ^1.52 | E2E test framework |
