# Feature 28 — World-Class Planning System UI: Architecture & Performance

## Objective

Transform the Planthium frontend from a single-file monolith (~2,700 lines in `App.tsx`) into a decomposed, high-performance UI architecture that competes with enterprise demand planning platforms like Anaplan, o9 Solutions, and Kinaxis RapidResponse.

---

## Summary of Phases Implemented

### Phase A: Foundation
- Vite build config updated with `manualChunks` for 7 named chunks: vendor (react/react-dom), charts (recharts), icons (lucide-react), radix (radix-ui), query (tanstack/react-query), table-grid (tanstack/react-table + react-virtual), echarts (echarts + echarts-for-react)
- GZip compression middleware added to FastAPI
- Cache-Control headers on read-only API endpoints
- CORS middleware configured
- Error boundary at app root level

### Phase B: Data Layer
- TanStack Query (React Query v5) replaces all 19 manual `useEffect` fetch loops
- Centralized query definitions in `src/api/queries.ts` with typed query keys
- Stale-while-revalidate caching strategy per endpoint (instant tab switching for cached data)
- Automatic request deduplication, retry with backoff, and AbortController cleanup

### Phase C: Component Decomposition
- `App.tsx` reduced from ~2,700 lines to ~230 lines (app shell only)
- 10 tab components extracted into `src/tabs/`:
  - `DashboardTab.tsx` — KPI sparkline cards, alert panel, heatmap, top movers, forecast trend chart
  - `ExplorerTab.tsx` — data grid, filters, pagination
  - `AccuracyTab.tsx` — model comparison, lag curve, champion selection
  - `DfuAnalysisTab.tsx` — sales vs forecast overlay, KPI cards
  - `ClustersTab.tsx` — cluster table, profiles, What-If scenarios
  - `MarketIntelTab.tsx` — search + narrative briefing
  - `InventoryTab.tsx` — inventory KPI cards, trend chart, position table, item detail drill-down
  - `InvBacktestTab.tsx` — inventory backtest model comparison, root cause attribution
  - `JobsTab.tsx` — job scheduler/monitor automation dashboard
  - `ChatPanel.tsx` — persistent chat drawer
- `React.lazy()` + `<Suspense>` per tab for code splitting (9 lazy-loaded tabs; ChatPanel imported directly)
- Per-tab `<ErrorBoundary>` for crash containment
- `TabErrorFallback` and `TabSuspenseFallback` helper components in App.tsx

### Phase D: Data Grid Upgrade
- TanStack Table (`@tanstack/react-table`) + TanStack Virtual (`@tanstack/react-virtual`) for virtualized data grid
- `DataTable.tsx` component with column resize, row selection, and server-side pagination
- CSV export via `papaparse`
- Skeleton loading placeholders (`Skeleton.tsx`)

### Phase E: Enterprise Features
- Keyboard shortcuts (`1-9` tab switch, `[` sidebar toggle, `t` theme cycle, `d` dark mode toggle, `/` search, `Esc` close, `?` help overlay, `Ctrl+E` column visibility, `Ctrl+M` motif settings panel)
- `KeyboardShortcutHelp.tsx` overlay component
- URL state synchronization (`useUrlState.ts`) for bookmarkable/shareable views
- Theme state hook (`useTheme.ts`) extracted from App.tsx
- Number/cell formatting utilities (`lib/formatters.ts`)
- Print-ready CSS (`@media print` rules in `index.css`)

### Phase F: Advanced
- ECharts integration for canvas-based charting (`EChartContainer.tsx` wrapper)
- Vitest testing infrastructure (`vitest.config.ts`)

---

## New Files Created

### Tab Components (`src/tabs/`)
| File | Purpose |
|------|---------|
| `DashboardTab.tsx` | Dashboard overview: KPI sparkline cards, alert panel, heatmap, top movers, forecast trend chart |
| `ExplorerTab.tsx` | Data explorer with virtualized grid, filters, pagination |
| `AccuracyTab.tsx` | Model accuracy comparison, lag curves, champion selection |
| `DfuAnalysisTab.tsx` | Sales vs multi-model forecast overlay chart, per-model KPI cards |
| `ClustersTab.tsx` | Cluster profiles, DFU assignments, What-If scenarios |
| `MarketIntelTab.tsx` | Market intelligence search + AI briefings |
| `InventoryTab.tsx` | Inventory KPI cards, trend chart, position table, item detail drill-down |
| `InvBacktestTab.tsx` | Inventory backtest model comparison, root cause attribution, DFU event table |
| `JobsTab.tsx` | Job scheduler/monitor: KPI cards, grouped job cards, schedules, history |
| `ChatPanel.tsx` | Persistent NL-to-SQL chat drawer |

### API Layer (`src/api/`)
| File | Purpose |
|------|---------|
| `queries.ts` | Centralized TanStack Query layer — all fetch functions + query keys |

### Hooks (`src/hooks/`)
| File | Purpose |
|------|---------|
| `useTheme.ts` | Product theme + color mode management (3 themes, light/dark) |
| `useUrlState.ts` | URL state synchronization (11 tabs, overview default) |
| `useKeyboardShortcuts.ts` | Keyboard shortcuts handler (1-9 tabs, sidebar, theme) |
| `useSidebar.ts` | Sidebar collapse/expand state + mobile drawer |
| `useGlobalFilters.ts` | Global filter state with debounced URL sync |
| `useMotifTheme.ts` | Motif selection, localStorage + URL persistence, CSS data-attr injection |
| `useDebounce.ts` | Generic debounce hook used by filter inputs |

### Utilities (`src/lib/`)
| File | Purpose |
|------|---------|
| `formatters.ts` | Number formatting, cell value formatting |
| `export.ts` | CSV export utility using papaparse |
| `utils.ts` | Shared utilities: `cn()` Tailwind class merger (clsx + tailwind-merge) |

### Components (`src/components/`)
| File | Purpose |
|------|---------|
| `DataTable.tsx` | Virtualized data grid (TanStack Table + Virtual) |
| `Skeleton.tsx` | Loading skeleton placeholder |
| `EChartContainer.tsx` | Theme-aware ECharts wrapper |
| `KeyboardShortcutHelp.tsx` | Keyboard shortcut help modal (triggered by `?` shortcut) |
| `AppSidebar.tsx` | Collapsible sidebar navigation (11 items, 5 sections, mobile drawer) |
| `ThemeSelector.tsx` | Theme + color mode picker (sidebar footer) |
| `GlobalFilterBar.tsx` | Cross-tab filter bar (brand, category, item, location, market, channel) |
| `WidgetGrid.tsx` | CSS Grid dashboard layout (WidgetGrid + WidgetCard) |
| `AlertPanel.tsx` | Severity-coded alert list |
| `HeatmapGrid.tsx` | CSS Grid heatmap with color scale |
| `TopMovers.tsx` | Period-over-period top movers list |
| `ForecastTrendChart.tsx` | ECharts forecast vs actual trend chart |
| `KpiCard.tsx` | Reusable KPI metric card component |
| `LoadingElement.tsx` | Chemistry-themed loading overlay with periodic table tile |
| `ElementTab.tsx` | Periodic table element tile (navigation tiles) |
| `MotifSettingsPanel.tsx` | Motif theme picker panel (opened by Ctrl+M) |

### Context Providers (`src/context/`)
| File | Purpose |
|------|---------|
| `GlobalFilterContext.tsx` | Global filter React context provider |
| `MotifContext.tsx` | React context provider for active motif theme |
| `ScenarioNotificationContext.tsx` | Cross-tab scenario notification context |
| `JobNotificationContext.tsx` | Cross-tab job notification context |

### Type Definitions (`src/types/`)
| File | Purpose |
|------|---------|
| `theme.ts` | TypeScript types for themes, sidebar, filters, dashboard |
| `motif.ts` | TypeScript types for motif system (MotifId, MotifThemeConfig, TileConfig) |
| `jobs.ts` | TypeScript types: JobStats, JobSchedule, GROUP_CONFIG |
| `index.ts` | Shared/common TypeScript type definitions |

### Constants (`src/constants/`)
| File | Purpose |
|------|---------|
| `colors.ts` | Shared color palette constants |
| `elements.ts` | Periodic table element definitions for loading overlay |
| `motifRegistry.ts` | Registry mapping motif IDs to motif config objects |
| `themes/` | Product theme configs (wineSpirits, general, obsidian) |
| `motifs/` | 5 motif theme definitions: periodic, spirits, space, f1, zen |

### Config
| File | Purpose |
|------|---------|
| `vitest.config.ts` | Vitest test configuration |

---

## NPM Packages Added

| Package | Purpose | Category |
|---------|---------|----------|
| `@tanstack/react-query` | Server state cache, fetch dedup, background refresh | Data layer |
| `@tanstack/react-table` | Headless data grid (sort, filter, resize, select) | Data grid |
| `@tanstack/react-virtual` | Row virtualization for large tables | Data grid |
| `react-error-boundary` | Per-component crash containment | Reliability |
| `papaparse` | CSV export generation | Export |
| `echarts` | Canvas-based charting (100K+ data points) | Charting |
| `echarts-for-react` | React wrapper for ECharts | Charting |
| `vitest` | Unit test runner | Testing |
| `clsx` | Conditional className helper | Styling |
| `tailwind-merge` | Tailwind class conflict resolution | Styling |

---

## Architecture Changes

### Before (Monolith)
```
App.tsx (2,700 lines)
├── 70 useState hooks
├── 20 useEffect hooks (19 fetch loops)
├── 25 fetch() calls (zero AbortController)
├── 0 React.memo / useCallback
├── All tabs rendered simultaneously
└── Single JS bundle (~800KB)
```

### After (Decomposed)
```
App.tsx (~230 lines — shell, router, theme, tab switching)
├── TanStack Query (centralized data layer with caching)
├── React.lazy() per tab (9 lazy-loaded + ChatPanel direct)
├── <ErrorBoundary> per tab (crash containment)
├── <Suspense> per tab (loading states)
├── Context providers (GlobalFilter, Motif, ScenarioNotification, JobNotification)
└── Chunked bundles (vendor, charts, icons, radix, query, table-grid, echarts, per-tab)

src/
├── api/queries.ts         — All query definitions + keys
├── tabs/                  — 10 tab components (9 lazy-loaded + ChatPanel)
├── components/            — 16 shared components (DataTable, Skeleton, EChartContainer,
│                            AppSidebar, GlobalFilterBar, KpiCard, WidgetGrid, etc.)
├── hooks/                 — 7 hooks (useTheme, useUrlState, useKeyboardShortcuts,
│                            useSidebar, useGlobalFilters, useMotifTheme, useDebounce)
├── lib/                   — formatters, export, utils (cn)
├── context/               — 4 context providers (GlobalFilter, Motif, Scenario, Job)
├── constants/             — colors, elements, motifRegistry, themes/, motifs/
└── types/                 — theme, motif, jobs, index type definitions
```

### Key Architectural Improvements
1. **Monolith to decomposed:** Each tab owns its own state; typing in chat no longer re-renders the Accuracy chart
2. **TanStack Query:** Stale-while-revalidate caching eliminates refetches on tab switch; automatic deduplication, retry, and AbortController
3. **Lazy loading:** Only the active tab's code is loaded; initial bundle reduced significantly
4. **Error boundaries:** A crash in one tab does not blank the entire application
5. **Virtualization:** Data grid renders only ~30 visible rows regardless of dataset size

---

## Testing Infrastructure

- **Vitest** configured in `frontend/vitest.config.ts` (jsdom environment, `src/__tests__/setup.ts` setup file)
- Run via `make ui-test` (218+ tests, ~1.5s)
- Component-level unit testing with React Testing Library compatibility
- Test utility: `src/tabs/__tests__/test-utils.tsx` provides `QueryClientProvider` wrapper for tab component tests
- Mocking patterns: `vi.mock("../api/queries")` for API layer; `echarts-for-react` mocked for chart components

### Test Coverage (as implemented)

| Category | Files |
|----------|-------|
| **Tab smoke tests** | `DashboardTab.test.tsx`, `ExplorerTab.test.tsx`, `AccuracyTab.test.tsx`, `DfuAnalysisTab.test.tsx`, `ClustersTab.test.tsx`, `MarketIntelTab.test.tsx`, `InventoryTab.test.tsx`, `InvBacktestTab.test.tsx`, `JobsTab.test.tsx`, `ChatPanel.test.tsx`, `WhatIfScenarios.test.tsx` |
| **Component tests** | `Skeleton.test.tsx`, `KeyboardShortcutHelp.test.tsx`, `EChartContainer.test.tsx`, `ElementTab.test.tsx`, `LoadingElement.test.tsx`, `MotifSettingsPanel.test.tsx`, `ThemeSelector.test.tsx`, `WidgetGrid.test.tsx`, `AlertPanel.test.tsx`, `TopMovers.test.tsx`, `HeatmapGrid.test.tsx`, `GlobalFilterBar.test.tsx`, `AppSidebar.test.tsx` |
| **Hook tests** | `useTheme.test.ts`, `useMotifTheme.test.ts`, `useSidebar.test.ts`, `useGlobalFilters.test.ts`, `useKeyboardShortcuts.test.ts`, `useUrlState.test.ts` |
| **Context tests** | `ScenarioNotificationContext.test.tsx`, `JobNotificationContext.test.tsx` |
| **API tests** | `queries.test.ts` |
| **Utility tests** | `formatters.test.ts`, `export.test.ts` |
| **Constants tests** | `motifRegistry.test.ts` |

---

## Performance Improvements

| Metric | Before | After |
|--------|--------|-------|
| App.tsx size | ~2,700 lines | ~230 lines |
| useState hooks in single component | 70 | 5-10 per tab |
| useEffect fetch loops | 19 manual | 0 (TanStack Query) |
| Tab switch (cached data) | ~500ms (refetch) | ~50ms (query cache) |
| Table DOM nodes (500 rows) | 10,000+ | ~30 visible (virtual) |
| JS bundle | Single chunk | Chunked (vendor, charts, per-tab) |
| Wire size (1K-row page) | ~400KB | ~40KB (gzip) |
| Error isolation | None (full app crash) | Per-tab error boundaries |

---

## Verification

1. `make api` + `make ui` -- All 10 tabs render correctly in all 3 product themes and 5 motifs
2. Tab switching -- Cached data appears instantly without loading spinners
3. Keyboard shortcuts -- `1-9` switches tabs, `[` toggles sidebar, `t` cycles theme, `d` toggles dark mode, `/` focuses search, `?` shows help, `Ctrl+M` opens motif panel
4. Data table -- Column resize, row selection, CSV export functional
5. Error resilience -- Kill API, verify error boundaries catch gracefully per tab
6. `make ui-test` -- Vitest tests pass (218+ tests)
7. `make test-all` -- All backend + frontend tests pass
