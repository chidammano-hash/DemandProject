<!-- SOURCE: feature28.md (UI Architecture & Performance) -->
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


---

## Examples

### Example: Lazy-loaded tab with error boundary

```typescript
// App.tsx — lazy loading prevents loading all tab JS upfront
import { lazy, Suspense } from 'react'
import { ErrorBoundary } from 'react-error-boundary'

const DfuAnalysisTab = lazy(() => import('./tabs/DfuAnalysisTab'))
const JobsTab        = lazy(() => import('./tabs/JobsTab'))

// Usage:
<ErrorBoundary fallback={<div>Tab failed to load</div>}>
  <Suspense fallback={<LoadingSpinner />}>
    <DfuAnalysisTab />
  </Suspense>
</ErrorBoundary>
```

### Example: TanStack Query stale-while-revalidate pattern

```typescript
import { useQuery } from '@tanstack/react-query'
import { fetchDfuAnalysis } from '@/api/queries'

const { data, isLoading, isFetching } = useQuery({
  queryKey: ['dfu-analysis', item, loc, mode, window],
  queryFn: () => fetchDfuAnalysis({ item, loc, mode, window }),
  staleTime: 5 * 60 * 1000,   // show cached data for 5 minutes
  gcTime:    10 * 60 * 1000,  // keep in cache for 10 minutes
  enabled: !!(item || loc),
})
// isFetching=true means background refresh; data still shows stale value
```

### Example: Keyboard shortcuts

```typescript
// useKeyboardShortcuts.ts
useEffect(() => {
  const handler = (e: KeyboardEvent) => {
    if (e.key >= '1' && e.key <= '9') switchTab(parseInt(e.key) - 1)
    if (e.key === '[') toggleSidebar()
    if (e.key === 'd') toggleDarkMode()
    if (e.key === '?') showHelpModal()
    if (e.key === '/') focusSearchInput()
    if (e.key === 'Escape') closeModal()
  }
  window.addEventListener('keydown', handler)
  return () => window.removeEventListener('keydown', handler)
}, [])
```

### Example: Frontend tests — run all

```bash
make ui-test
# Runs 218 Vitest tests across components, hooks, tabs, and utilities
# ~1.5 seconds total
```


---

<!-- SOURCE: feature36.md (Product-Grade UI Overhaul) -->
# Feature 36 — Product-Grade UI Overhaul with Three Industry Themes

**Status:** Implemented
**Priority:** Major Enhancement
**Dependencies:** Feature 22 (themes), Feature 28 (UI architecture), Feature 35 (motif system)

---

## 1. Overview

Transform Demand Studio from a developer-oriented analytics tool into a polished, enterprise-grade supply chain planning product. Introduce a **collapsible sidebar navigation**, **global filter bar**, **dashboard overview page**, **widget grid system**, and three complete visual themes: **Wine & Spirits** ("The Reserve"), **General** ("Studio"), and **Obsidian** ("Command"). The overhaul preserves all existing functionality while dramatically elevating the visual fidelity, information density, and professional polish of every screen.

---

## 2. Problem

1. **Horizontal tab bar limits scalability** — Adding more tabs creates horizontal overflow; no room for nested sub-pages or grouped navigation sections.
2. **No dashboard landing page** — Users land directly on the Explorer data grid with no overview of system health, alerts, or key metrics.
3. **No persistent global filters** — Users must re-apply item/location/brand filters on every tab independently.
4. **Motif system is cosmetic only** — Current motifs change tile colors and loading animations but don't transform the structural layout, navigation, or component design language.
5. **Missing enterprise UX patterns** — No alert panels, heatmap widgets, top-movers summaries, or quick-action cards that users expect from production planning tools.
6. **Visual density is low** — Large padding, single-column layouts, and sparse KPI cards waste screen real estate on wide monitors.

---

## 3. Architecture: Theme-Driven Layout System

Each theme defines not just colors but a complete **design language**: sidebar icon style, card treatment, chart palette, typography weight, border radius, shadow depth, and animation character. The theme system extends the existing motif architecture with a new `ProductTheme` layer.

```
┌──────────────────────────────────────────────────────────┐
│  ProductTheme (feature36)                                 │
│  ├── Sidebar navigation config (icon style, indicator)   │
│  ├── Global filter bar config                            │
│  ├── Dashboard widget grid                               │
│  ├── Component design tokens (radius, shadow, spacing)   │
│  ├── Typography (weight, tracking, scale)                │
│  └── Three complete palettes (light / dark per theme)    │
│                                                           │
│  Existing Motif System (preserved, optional overlay)      │
│  ├── Tile variants (periodic, card, badge, emblem)       │
│  ├── Loading animations                                  │
│  └── Per-domain tile colors                              │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Structural Layout Overhaul

### 4.1 Collapsible Sidebar Navigation

Replace the horizontal `ElementTab` bar with a vertical sidebar. The sidebar persists across all pages and supports collapsed (icon-only, 64px) and expanded (icon + label, 240px) states.

**Sidebar Sections:**

| Section | Items |
|---------|-------|
| **Overview** | Dashboard (new landing page) |
| **Demand** | Explorer, DFU Analysis, Accuracy |
| **Supply** | Inventory |
| **Intelligence** | Clusters, Market Intel |
| **System** | Chat, Settings |

**Layout (expanded):**

```
┌──────────────────────────────────────────┐
│ [LOGO]  Demand Studio           [«]      │
├──────────────────────────────────────────┤
│                                          │
│  ■  Overview                             │   ← section: overview
│                                          │
│  ─────────  Demand  ─────────            │   ← section label
│  □  Explorer                             │
│  □  DFU Analysis                         │
│  □  Accuracy                             │
│                                          │
│  ─────────  Supply  ─────────            │
│  □  Inventory                            │
│                                          │
│  ─────────  Intelligence  ───────        │
│  □  Clusters                             │
│  □  Market Intel                         │
│                                          │
│  ─────────────────────────────           │
│  □  Chat                                 │
│  □  Settings                             │
│                                          │
├──────────────────────────────────────────┤
│  [Theme Selector]  [Light|Dark]          │
└──────────────────────────────────────────┘
```

**Nav item configuration:**

| Order | Label | Icon (Lucide) | Tab Key | Section |
|-------|-------|---------------|---------|---------|
| 1 | Overview | `LayoutDashboard` | `overview` | overview |
| 2 | Explorer | `Database` | `explorer` | demand |
| 3 | DFU Analysis | `TrendingUp` | `dfuAnalysis` | demand |
| 4 | Accuracy | `Target` | `accuracy` | demand |
| 5 | Inventory | `Package` | `inventory` | supply |
| 6 | Clusters | `Network` | `clusters` | intelligence |
| 7 | Market Intel | `Globe` | `intel` | intelligence |
| 8 | Chat | `MessageSquare` | `chat` | system |
| 9 | Settings | `Settings` | `settings` | system |

**Key behaviors:**
- Collapsed by default on screens < 1440px, expanded on wider screens
- Toggle via `[<<]` button or keyboard shortcut `[`
- Tooltip on hover when collapsed (shows label)
- Active item: highlighted background + left accent bar (4px), style varies by theme
- Section dividers: thin `border-border` lines with section labels hidden in collapsed mode
- Smooth 200ms width transition (GPU-accelerated `transform` + `width`)
- Mobile (< 768px): Off-canvas drawer triggered by hamburger icon, overlay backdrop
- Stores collapsed/expanded state in `localStorage("ds-sidebar")`

**Component: `AppSidebar`**

```tsx
interface SidebarItem {
  key: string;
  label: string;
  icon: LucideIcon;
  section: "overview" | "demand" | "supply" | "intelligence" | "system";
  shortcut?: string;
}

interface AppSidebarProps {
  activeTab: string;
  onNavigate: (tab: string) => void;
  collapsed: boolean;
  onToggle: () => void;
}
```

### 4.2 Global Filter Bar

A persistent horizontal bar below the header, above content. Provides cross-tab context filters that propagate to all data-fetching queries.

```
┌───────────────────────────────────────────────────────────────────────┐
│ Brand: All ▾  │ Category: All ▾  │ Market: All ▾  │ Channel: All ▾  │  Month / Quarter  │
└───────────────────────────────────────────────────────────────────────┘
```

**Filters:**

| Filter | Source | Type | Default |
|--------|--------|------|---------|
| Brand | `dim_item.brand` distinct values | Multi-select dropdown | "All" |
| Category | `dim_item.class_` distinct values | Multi-select dropdown | "All" |
| Market | `dim_location.state` distinct values | Multi-select dropdown | "All" |
| Channel | `dim_customer.customer_group` distinct values | Multi-select dropdown | "All" |
| Time Grain | Static: Month / Quarter | Toggle group | "Month" |

**State management:**
- New hook: `useGlobalFilters()` — stores filter state, syncs to URL params
- Filters passed as context via `<GlobalFilterProvider>`
- Each tab's query hooks read from global filter context
- Filter values fetched once on mount via new API endpoint: `GET /domains/{domain}/distinct?column=brand`
- Debounced 300ms before propagating to queries
- Distinct values cached by TanStack Query (5min staleTime)

### 4.3 App Shell Layout Transformation

**Before (current):**

```
┌───────────────────────────────────────┐
│  [Logo] App Name   [Tab][Tab][Tab] ⚙  │   ← header with horizontal tabs
├───────────────────────────────────────┤
│                                       │
│          Tab Content                  │   ← full-width content
│                                       │
├───────────────────────────────────────┤
│          Chat Panel                   │
└───────────────────────────────────────┘
```

**After (new):**

```
┌────┬──────────────────────────────────────────────┐
│    │  Brand ▾ │ Category ▾ │ Market ▾ │ Mo / Qtr  │  ← global filter bar
│ S  ├──────────────────────────────────────────────┤
│ I  │                                              │
│ D  │           Page Content                       │  ← dashboard / tab content
│ E  │           (with widget grid)                 │
│ B  │                                              │
│ A  │                                              │
│ R  │                                              │
│    ├──────────────────────────────────────────────┤
│    │           Chat (collapsible)                 │
└────┴──────────────────────────────────────────────┘
```

**New App.tsx structure:**

```tsx
export default function App() {
  return (
    <ThemeProvider>
      <GlobalFilterProvider>
        <MotifProvider value={motifTheme}>
          <div className="flex h-screen overflow-hidden">
            <AppSidebar
              activeTab={activeTab}
              onNavigate={handleTabSwitch}
              collapsed={sidebarCollapsed}
              onToggle={() => setSidebarCollapsed(c => !c)}
            />
            <div className="flex flex-1 flex-col overflow-hidden">
              <GlobalFilterBar />
              <div className="flex-1 overflow-y-auto p-4 md:p-6">
                <div className="mx-auto max-w-[1600px]">
                  {activeTab === "overview" && <DashboardTab />}
                  {activeTab === "explorer" && <ExplorerTab ... />}
                  {/* ... other tabs ... */}
                </div>
              </div>
              <ChatPanel domain={domain} theme={theme} />
            </div>
          </div>
        </MotifProvider>
      </GlobalFilterProvider>
    </ThemeProvider>
  );
}
```

### 4.4 Dashboard Overview Page (New Default Landing)

A new landing page that replaces the Explorer as the default tab. Provides an at-a-glance view of the entire demand planning system.

**Layout (12-column CSS Grid):**

```
┌──────────────────────────────────────────────────────────────┐
│  KPI Cards Row (6 cards, span-2 each)                        │
│  [Accuracy %] [WAPE] [Bias] [Total Fcst] [Total Act] [WoS]  │
├──────────────┬───────────────────────┬───────────────────────┤
│ Alert Panel  │  Performance Heatmap  │  Top Movers           │
│ (span-3)     │  (span-6)             │  (span-3)             │
│              │                       │                        │
│ • OOS Risk   │  Category × Week      │  ▲ Item A  +32.4K    │
│ • Bias Drift │  color = accuracy %   │  ▼ Item B  -18.6K    │
│ • Low Acc    │                       │  ▲ Item C  +25.9K    │
├──────────────┴───────────────────────┴───────────────────────┤
│  Forecast Trend (span-12)                                     │
│  Stacked area: Baseline + Adjustment + Final + Actual overlay │
│  X-axis: time (months), Y-axis: volume                        │
├──────────────────────────────────────────────────────────────┤
│  Recent Forecast Table (span-12)                              │
│  Period │ Forecast │ Actual │ Bias │ Accuracy │ Override      │
└──────────────────────────────────────────────────────────────┘
```

**Widget Specifications:**

| Widget | Data Source | Visualization | Component |
|--------|-------------|---------------|-----------|
| KPI Cards | `agg_forecast_monthly` materialized view | `<KpiCard>` with trend sparkline | `KpiRibbon.tsx` |
| Alert Panel | Computed from accuracy/bias thresholds | Severity-sorted list with icons | `AlertPanel.tsx` |
| Performance Heatmap | Sales by category × time period | Color-coded CSS grid (green→red) | `HeatmapGrid.tsx` |
| Top Movers | Sales period-over-period delta | Ranked list with +/- indicators | `TopMovers.tsx` |
| Forecast Trend | `agg_forecast_monthly` time series | ECharts stacked area chart | `ForecastTrendChart.tsx` |
| Recent Table | Latest forecast vs actual rows | `<DataTable>` with inline bars | `RecentForecastTable.tsx` |

### 4.5 Widget Grid System

A reusable CSS Grid wrapper for composing dashboard-style layouts across any tab.

```tsx
interface WidgetGridProps {
  cols?: 6 | 12;                     // 12 default
  gap?: "sm" | "md" | "lg";         // md default (gap-4)
  children: React.ReactNode;
}

interface WidgetCardProps {
  span?: 1 | 2 | 3 | 4 | 6 | 12;   // column span
  title?: string;
  subtitle?: string;
  actions?: React.ReactNode;         // top-right slot (refresh, expand, etc.)
  children: React.ReactNode;
}
```

No external library — pure Tailwind `grid-cols-12` + `col-span-*` classes.

---

## 5. Three Production Themes

Each theme defines a complete visual identity: color palette, sidebar style, card treatment, chart colors, typography, and animation character. The theme selector replaces the current Color Mode + Motif picker with a single unified control.

### Theme Type System

```typescript
// src/types/theme.ts
type ProductThemeId = "wine-spirits" | "general" | "obsidian";

interface ProductTheme {
  id: ProductThemeId;
  displayName: string;
  tagline: string;
  description: string;
  supportedModes: ("light" | "dark")[];     // obsidian: dark only
  defaultMode: "light" | "dark";
  palette: {
    light?: ThemePalette;
    dark: ThemePalette;
  };
  sidebar: SidebarThemeConfig;
  cards: CardThemeConfig;
  charts: ChartThemeConfig;
  typography: TypographyConfig;
  logo: {
    icon: string;                            // Lucide icon name
    gradient?: string;                       // optional CSS gradient on icon
  };
}

interface SidebarThemeConfig {
  activeIndicator: "bar" | "pill" | "glow";
  iconStrokeWidth: 1 | 1.5 | 2;
  sectionLabelStyle: "uppercase" | "capitalize" | "hidden";
  hoverEffect: "bg" | "glow" | "subtle";
}

interface CardThemeConfig {
  borderRadius: string;                      // CSS value (0.25rem, 0.5rem, 0.75rem)
  shadow: string;                            // Tailwind class
  borderStyle: "solid" | "none" | "subtle";
  hoverEffect: "lift" | "glow" | "none";
}

interface ChartThemeConfig {
  seriesColors: string[];                    // 6 hex colors
  gridColor: string;
  axisColor: string;
  tooltipBg: string;
  heatmapScale: string[];                   // 5 colors: excellent → critical
}

interface TypographyConfig {
  headingWeight: 500 | 600 | 700;
  headingTracking: string;                   // letter-spacing CSS value
  bodyWeight: 400;
  kpiWeight: 700;
  kpiTracking: string;
}
```

### CSS Variable Architecture

Each theme defines its palette as CSS custom properties on `<html>`. All existing components already consume these variables via `hsl(var(--token))`, so theme switching requires zero component re-renders.

```css
/* Applied via data attribute: <html data-theme="wine-spirits" class="light"> */
[data-theme="wine-spirits"] { /* light palette */ }
[data-theme="wine-spirits"].dark { /* dark palette */ }
[data-theme="general"] { /* light palette */ }
[data-theme="general"].dark { /* dark palette */ }
[data-theme="obsidian"] { /* dark-only palette */ }
[data-theme="obsidian"].dark { /* same palette, slightly elevated */ }
```

---

### 5.1 Wine & Spirits — "The Reserve"

**Mood:** Premium wine cellar management software. Warm burgundy tones, aged oak textures, gold foil accents. Candlelit tasting room meets modern analytics.

**App Name:** "The Reserve"
**Tagline:** "Demand Intelligence, Refined"
**Logo Icon:** `Wine` (Lucide)

#### Light Palette

| Token | HSL | Hex | Usage |
|-------|-----|-----|-------|
| `--background` | `24 20% 95%` | `#F4EDE6` | Parchment cream |
| `--foreground` | `20 30% 15%` | `#2D2118` | Dark espresso |
| `--card` | `30 25% 98%` | `#FCF9F5` | Warm white |
| `--card-foreground` | `20 30% 15%` | `#2D2118` | Dark espresso |
| `--primary` | `345 55% 30%` | `#752836` | Deep burgundy |
| `--primary-foreground` | `0 0% 100%` | `#FFFFFF` | White |
| `--secondary` | `35 40% 90%` | `#F0E4CC` | Light oak |
| `--secondary-foreground` | `25 35% 22%` | `#3D2E1D` | Dark oak |
| `--muted` | `30 15% 91%` | `#EDE8E2` | Stone |
| `--muted-foreground` | `20 12% 45%` | `#7A6E62` | Warm gray |
| `--accent` | `42 75% 50%` | `#D4A020` | Gold foil |
| `--accent-foreground` | `20 35% 12%` | `#241A0E` | Near black |
| `--border` | `30 18% 82%` | `#D8CEBF` | Light oak border |
| `--destructive` | `0 65% 50%` | `#D43838` | Alert red |
| `--sidebar-bg` | `20 25% 14%` | `#2C2018` | Dark walnut |
| `--sidebar-foreground` | `30 15% 85%` | `#DDD5CB` | Light cream |
| `--sidebar-active` | `345 55% 30%` | `#752836` | Burgundy highlight |
| `--sidebar-hover` | `20 20% 20%` | `#3D3028` | Warm hover |

#### Dark Palette

| Token | HSL | Hex | Usage |
|-------|-----|-----|-------|
| `--background` | `20 20% 8%` | `#1A1410` | Dark cellar |
| `--foreground` | `30 15% 85%` | `#DDD5CB` | Warm cream |
| `--card` | `20 18% 11%` | `#221B14` | Dark oak panel |
| `--primary` | `345 50% 55%` | `#C94D6B` | Rose burgundy |
| `--secondary` | `35 30% 18%` | `#3A2D1E` | Aged oak |
| `--accent` | `42 70% 55%` | `#D9AD35` | Bright gold |
| `--muted` | `20 15% 15%` | `#2A231B` | Shadow |
| `--muted-foreground` | `25 10% 55%` | `#968A7D` | Muted warm |
| `--border` | `20 12% 20%` | `#3A3028` | Dark border |
| `--sidebar-bg` | `20 28% 7%` | `#170F09` | Near-black walnut |
| `--sidebar-foreground` | `30 12% 65%` | `#B0A494` | Muted cream |
| `--sidebar-active` | `345 50% 55%` | `#C94D6B` | Rose highlight |

#### Chart Colors

```
Light: ["#752836", "#D4A020", "#2D6A4F", "#7B4B94", "#C25B3F", "#1B6B8A"]
        burgundy   gold       forest     plum       terracotta  teal

Dark:  ["#C94D6B", "#D9AD35", "#52B788", "#A678C8", "#E07C5F", "#48B8D0"]
        rose       bright-gold mint       lavender   salmon      cyan
```

#### KPI Colors

```
Light:                              Dark:
--kpi-best:    #2D6A4F (forest)     #52B788 (mint)
--kpi-warning: #C25B3F (terracotta) #E07C5F (salmon)
--kpi-ceiling: #7B4B94 (plum)       #A678C8 (lavender)
```

#### Heatmap Scale

```
Excellent (>95%): #2D6A4F (forest green)
Good (85-95%):    #74C69D (light green)
Warning (70-85%): #F4A261 (warm amber)
Poor (50-70%):    #E07C5F (terracotta)
Critical (<50%):  #C25B3F (deep terracotta)
```

#### Background Gradient

```css
[data-theme="wine-spirits"] {
  background-image:
    radial-gradient(ellipse at 10% 5%, rgba(117, 40, 54, 0.08), transparent 40%),
    radial-gradient(ellipse at 90% 0%, rgba(212, 160, 32, 0.06), transparent 35%),
    linear-gradient(160deg, #F4EDE6 0%, #F0E4CC 50%, #EDE0D0 100%);
}
[data-theme="wine-spirits"].dark {
  background-image:
    radial-gradient(ellipse at 10% 5%, rgba(201, 77, 107, 0.06), transparent 40%),
    radial-gradient(ellipse at 90% 0%, rgba(217, 173, 53, 0.04), transparent 35%),
    linear-gradient(160deg, #1A1410 0%, #1E1712 50%, #1A1410 100%);
}
```

#### Component Config

| Property | Value |
|----------|-------|
| **Sidebar active indicator** | 4px left bar in burgundy |
| **Sidebar icon stroke** | 1.5px (refined) |
| **Sidebar section labels** | capitalize |
| **Sidebar hover** | `bg-sidebar-hover` warm glow |
| **Card border-radius** | `0.5rem` (classic, restrained) |
| **Card shadow** | `shadow-sm` (subtle, not floating) |
| **Card border** | `1px solid var(--border)` (always visible) |
| **Card hover** | `shadow-md` lift effect |
| **Heading weight** | 600 |
| **Heading tracking** | `-0.01em` (tight, refined) |
| **KPI tracking** | `0` (neutral) |
| **Loading animation** | `pour-shimmer` — gold shimmer sweep |

---

### 5.2 General — "Studio"

**Mood:** Clean SaaS analytics platform. Think Linear, Vercel Dashboard, Stripe. Neutral grays, crisp blue accents, maximum readability. Zero visual noise, zero personality — pure function.

**App Name:** "Demand Studio"
**Tagline:** "Unified Demand Intelligence"
**Logo Icon:** `BarChart3` (Lucide)

#### Light Palette

| Token | HSL | Hex | Usage |
|-------|-----|-----|-------|
| `--background` | `0 0% 98%` | `#FAFAFA` | Pure off-white |
| `--foreground` | `0 0% 9%` | `#171717` | Near black |
| `--card` | `0 0% 100%` | `#FFFFFF` | Pure white |
| `--card-foreground` | `0 0% 9%` | `#171717` | Near black |
| `--primary` | `221 83% 53%` | `#3B82F6` | Standard blue |
| `--primary-foreground` | `0 0% 100%` | `#FFFFFF` | White |
| `--secondary` | `220 14% 96%` | `#F1F5F9` | Slate-50 |
| `--secondary-foreground` | `220 9% 46%` | `#64748B` | Slate-500 |
| `--muted` | `220 14% 96%` | `#F1F5F9` | Slate-50 |
| `--muted-foreground` | `220 9% 46%` | `#64748B` | Slate-500 |
| `--accent` | `220 14% 96%` | `#F1F5F9` | Slate-50 |
| `--accent-foreground` | `221 83% 53%` | `#3B82F6` | Blue |
| `--border` | `220 13% 91%` | `#E2E8F0` | Slate-200 |
| `--destructive` | `0 84% 60%` | `#EF4444` | Red-500 |
| `--sidebar-bg` | `0 0% 100%` | `#FFFFFF` | White |
| `--sidebar-foreground` | `220 9% 46%` | `#64748B` | Slate-500 |
| `--sidebar-active` | `221 83% 53%` | `#3B82F6` | Blue |
| `--sidebar-hover` | `220 14% 96%` | `#F1F5F9` | Slate-50 |

#### Dark Palette

| Token | HSL | Hex | Usage |
|-------|-----|-----|-------|
| `--background` | `224 10% 10%` | `#18181B` | Zinc-900 |
| `--foreground` | `210 20% 98%` | `#FAFAFA` | Near white |
| `--card` | `224 10% 12%` | `#1E1E22` | Dark card |
| `--primary` | `217 91% 60%` | `#60A5FA` | Blue-400 |
| `--secondary` | `217 10% 18%` | `#27272A` | Zinc-800 |
| `--muted` | `215 10% 16%` | `#27272A` | Zinc-800 |
| `--muted-foreground` | `218 10% 55%` | `#A1A1AA` | Zinc-400 |
| `--border` | `215 10% 20%` | `#3F3F46` | Zinc-700 |
| `--sidebar-bg` | `224 10% 10%` | `#18181B` | Same as page |
| `--sidebar-foreground` | `218 10% 55%` | `#A1A1AA` | Zinc-400 |
| `--sidebar-active` | `217 91% 60%` | `#60A5FA` | Blue-400 |

#### Chart Colors

```
Light: ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4"]
        blue       emerald    amber      violet     red        cyan

Dark:  ["#60A5FA", "#34D399", "#FBBF24", "#A78BFA", "#F87171", "#22D3EE"]
        blue-400   emerald-400 amber-300 violet-400 red-400   cyan-400
```

#### KPI Colors

```
Light:                               Dark:
--kpi-best:    #10B981 (emerald)     #34D399 (emerald-400)
--kpi-warning: #EF4444 (red-500)     #F87171 (red-400)
--kpi-ceiling: #8B5CF6 (violet)      #A78BFA (violet-400)
```

#### Heatmap Scale

```
Excellent: #10B981 (emerald-500)
Good:      #6EE7B7 (emerald-300)
Warning:   #FBBF24 (amber-400)
Poor:      #F87171 (red-400)
Critical:  #EF4444 (red-500)
```

#### Background

```css
[data-theme="general"] {
  background-image: none;
  background-color: hsl(var(--background));
}
```

No gradient. Pure flat. Maximum clarity and professionalism.

#### Component Config

| Property | Value |
|----------|-------|
| **Sidebar active indicator** | pill (`rounded-md bg-primary/10 text-primary`) |
| **Sidebar icon stroke** | 1.5px (standard Lucide default) |
| **Sidebar section labels** | uppercase, `text-xs`, `font-medium`, `tracking-wider` |
| **Sidebar hover** | `bg-muted` subtle highlight |
| **Card border-radius** | `0.75rem` (modern, rounded) |
| **Card shadow** | `shadow-sm` (barely perceptible) |
| **Card border** | `1px solid var(--border)` |
| **Card hover** | none (static, clean) |
| **Heading weight** | 600 |
| **Heading tracking** | `0` (neutral, clean) |
| **KPI tracking** | `0` |
| **Loading animation** | `animate-pulse` — standard Tailwind skeleton pulse |

---

### 5.3 Obsidian — "Command"

**Mood:** Military command center / financial trading floor. Deep blacks, electric accent colors, high information density. Data-forward, zero decoration, maximum contrast. Bloomberg Terminal meets modern design.

**App Name:** "Command"
**Tagline:** "Total Demand Visibility"
**Logo Icon:** `Radar` (Lucide)

**Note:** This theme is **dark-only**. Selecting "light" mode with Obsidian shifts surfaces up by ~3% lightness (background `#111111`, cards `#181818`) but keeps the dark palette. This prevents blinding white while respecting the preference signal.

#### Dark Palette (Primary)

| Token | HSL | Hex | Usage |
|-------|-----|-----|-------|
| `--background` | `0 0% 4%` | `#0A0A0A` | True near-black |
| `--foreground` | `0 0% 90%` | `#E5E5E5` | Bright gray |
| `--card` | `0 0% 7%` | `#121212` | Dark card |
| `--card-foreground` | `0 0% 90%` | `#E5E5E5` | Bright gray |
| `--primary` | `142 71% 45%` | `#22C55E` | Electric green |
| `--primary-foreground` | `0 0% 4%` | `#0A0A0A` | Black |
| `--secondary` | `0 0% 12%` | `#1F1F1F` | Elevated surface |
| `--secondary-foreground` | `0 0% 70%` | `#B3B3B3` | Mid gray |
| `--muted` | `0 0% 10%` | `#1A1A1A` | Subtle surface |
| `--muted-foreground` | `0 0% 50%` | `#808080` | Dim gray |
| `--accent` | `47 100% 50%` | `#FFD700` | Amber signal |
| `--accent-foreground` | `0 0% 4%` | `#0A0A0A` | Black |
| `--border` | `0 0% 15%` | `#262626` | Subtle border |
| `--destructive` | `0 72% 51%` | `#DC2626` | Alert red |
| `--sidebar-bg` | `0 0% 3%` | `#080808` | Deepest black |
| `--sidebar-foreground` | `0 0% 55%` | `#8C8C8C` | Dim |
| `--sidebar-active` | `142 71% 45%` | `#22C55E` | Electric green |
| `--sidebar-hover` | `0 0% 8%` | `#141414` | Barely lighter |

#### "Light" Fallback Palette (Elevated Dark)

Same values as dark, except:

| Token | HSL | Hex | Difference |
|-------|-----|-----|------------|
| `--background` | `0 0% 7%` | `#111111` | +3% lightness |
| `--card` | `0 0% 10%` | `#181818` | +3% lightness |
| `--muted` | `0 0% 13%` | `#212121` | +3% lightness |
| `--border` | `0 0% 18%` | `#2E2E2E` | +3% lightness |

#### Chart Colors

```
["#22C55E", "#3B82F6", "#FFD700", "#EF4444", "#A78BFA", "#06B6D4"]
  green      blue       gold       red        violet     cyan
```

#### KPI Colors

```
--kpi-best:    #22C55E (electric green)
--kpi-warning: #EF4444 (red)
--kpi-ceiling: #FFD700 (gold)
```

#### Heatmap Scale

```
Excellent: #22C55E (green-500)
Good:      #86EFAC (green-300)
Warning:   #FFD700 (gold)
Poor:      #F87171 (red-400)
Critical:  #DC2626 (red-600)
```

#### Background

```css
[data-theme="obsidian"] {
  background-image:
    radial-gradient(ellipse at 50% 0%, rgba(34, 197, 94, 0.03), transparent 50%),
    linear-gradient(180deg, #0A0A0A 0%, #0D0D0D 100%);
}
```

Extremely subtle green glow at top center — like a system status indicator.

#### Component Config

| Property | Value |
|----------|-------|
| **Sidebar active indicator** | 2px green left glow bar with `shadow-[0_0_8px_rgba(34,197,94,0.4)]` |
| **Sidebar icon stroke** | 1px (thin, technical) |
| **Sidebar section labels** | uppercase, `text-[10px]`, `tracking-[0.15em]` |
| **Sidebar hover** | `bg-white/5` (subtle light bleed) |
| **Card border-radius** | `0.25rem` (sharp, technical, angular) |
| **Card shadow** | none |
| **Card border** | `1px solid var(--border)` (structural, minimal) |
| **Card hover** | `border-primary/30` green border glow |
| **Heading weight** | 500 (lighter, technical) |
| **Heading tracking** | `0.02em` (slightly spaced, like a dashboard readout) |
| **KPI tracking** | `0.05em` (monospaced feel) |
| **Loading animation** | Green `pulse-glow` variant with `--primary` shadow color |

---

## 6. Theme Comparison Matrix

| Aspect | Wine & Spirits | General | Obsidian |
|--------|---------------|---------|----------|
| **Mood** | Warm, luxurious, curated | Clean, professional, neutral | Technical, dense, command |
| **Color Modes** | Light + Dark | Light + Dark | Dark only |
| **Base tone** | Parchment / Cellar | White / Zinc | Near-black |
| **Primary** | Burgundy `#752836` | Blue `#3B82F6` | Green `#22C55E` |
| **Accent** | Gold `#D4A020` | Slate `#F1F5F9` | Amber `#FFD700` |
| **Sidebar bg** | Dark walnut | White (matches page) | Deepest black |
| **Active indicator** | Burgundy left bar | Blue pill | Green glow bar |
| **Border radius** | 0.5rem (classic) | 0.75rem (modern) | 0.25rem (sharp) |
| **Shadows** | Subtle lift | Barely there | None |
| **Card hover** | Shadow lift | None | Green border glow |
| **Background** | Warm radial gradient | Flat solid color | Subtle green glow |
| **Loading** | Pour shimmer (gold) | Skeleton pulse | Green pulse |
| **Chart palette** | Burgundy → Gold → Forest | Blue → Emerald → Amber | Green → Blue → Gold |
| **Logo** | `Wine` | `BarChart3` | `Radar` |
| **Typography** | Refined (tight tracking) | Neutral (zero tracking) | Technical (wide tracking) |
| **Personality** | High (wine cellar) | Zero (pure function) | Medium (command center) |

---

## 7. Component Redesigns

### 7.1 KPI Cards (Enhanced)

Redesigned with trend indicators, sparklines, and severity-aware coloring.

```
┌─────────────────────────┐
│  Target  Accuracy        │   ← icon + label
│                          │
│     87.2%                │   ← large value (text-3xl, font-bold)
│  ▲ +1.2% vs prior       │   ← trend delta with direction arrow
│  ▁▂▃▅▇▆▅▃▂▃▅▆           │   ← inline sparkline (optional, 12 points)
└─────────────────────────┘
```

**Extended props:**

```tsx
interface KpiCardProps {
  label: string;
  value: string | number;
  format?: "percent" | "currency" | "number" | "compact";
  trend?: { delta: number; direction: "up" | "down" | "flat" };
  sparkline?: number[];           // 8-12 data points for inline chart
  severity?: "best" | "warning" | "neutral";
  icon?: LucideIcon;
}
```

Sparklines rendered as a simple SVG polyline — no chart library overhead.

### 7.2 Alert Panel Widget

Compact severity-coded notification list.

```tsx
interface Alert {
  id: string;
  type: "oos_risk" | "bias_drift" | "low_accuracy" | "demand_spike" | "allocation_shortage";
  severity: "critical" | "high" | "medium" | "low";
  title: string;
  detail: string;
  count?: number;
}
```

**Visual treatment by severity:**

| Severity | Left border | Icon | Color token |
|----------|-------------|------|-------------|
| critical | 4px `destructive` | `AlertTriangle` | `--destructive` |
| high | 4px `--kpi-warning` | `AlertCircle` | `--kpi-warning` |
| medium | 4px `accent` | `AlertCircle` | `--accent` |
| low | 4px `muted` | `Info` | `--muted-foreground` |

### 7.3 Heatmap Grid

Pure CSS Grid-based heatmap — no canvas library for small grids (< 100 cells).

```tsx
interface HeatmapGridProps {
  rows: { label: string; values: number[] }[];
  columnLabels: string[];
  colorScale: (value: number) => string;
  valueFormat?: (value: number) => string;
  onCellClick?: (row: string, col: string) => void;
}
```

Each cell is a `<div>` with computed `background-color` from the theme's heatmap scale. Hover shows value in a tooltip. Row labels on left, column labels on top.

### 7.4 Top Movers

Ranked list with change indicators.

```tsx
interface Mover {
  label: string;
  delta: number;
  direction: "up" | "down";
  icon?: string;
}
```

Positive deltas in `--kpi-best` color with `▲` prefix, negative in `--kpi-warning` with `▼`. Values formatted as compact numbers (e.g., `+32.4K`).

### 7.5 Forecast Trend Chart

ECharts stacked area chart with 4 series:

| Series | Style | Purpose |
|--------|-------|---------|
| Baseline | Filled area, `chart-1` color, 30% opacity | Base forecast volume |
| Adjustment | Stacked area, `chart-3` color, 30% opacity | Feature/display lift |
| Final Forecast | Bold line, `chart-5` color | Combined final number |
| Actual Demand | Dashed line with dots, `chart-2` color | Ground truth overlay |

X-axis: time periods (months). Y-axis: volume. Tooltip shows all series values.

### 7.6 Theme Selector

Replaces the current Color Mode + Motif two-step picker. Lives in sidebar footer.

```
┌──────────────────────────┐
│  Theme                    │
│  ┌────┐ ┌────┐ ┌────┐   │
│  │ 🍷 │ │ □  │ │ ◆  │   │
│  │Wine │ │Gen │ │Obs │   │
│  └────┘ └────┘ └────┘   │
│                           │
│  Mode: ○ Light   ● Dark  │
│  (disabled for Obsidian)  │
└──────────────────────────┘
```

When collapsed, shows only the current theme icon in the sidebar footer. Click expands a popover with the full selector.

---

## 8. New API Endpoints

### 8.1 `GET /domains/{domain}/distinct`

Returns distinct values for a column (for filter dropdowns).

**Query params:**
- `column` (required): column name
- `limit` (optional, default 100): max values
- `search` (optional): prefix filter

**Response:**
```json
{ "column": "brand", "values": ["Absolut", "Jack Daniel's", ...], "total": 45 }
```

**SQL:** `SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY {col} LIMIT {limit}`

### 8.2 `GET /dashboard/kpis`

Aggregated KPI metrics for the overview dashboard.

**Query params:**
- `window` (optional, default 3): months of history
- Global filter params (brand, category, market, channel)

**Response:**
```json
{
  "accuracy_pct": 87.2,
  "wape_pct": 12.8,
  "bias_pct": 3.6,
  "total_forecast": 1250000,
  "total_actual": 1205000,
  "weeks_of_supply": 5.2,
  "window_months": 3,
  "deltas": {
    "accuracy_pct": 1.2,
    "wape_pct": -0.8,
    "bias_pct": 0.5
  }
}
```

**Source:** Joins `agg_forecast_monthly` + `agg_sales_monthly`. Returns `null` for metrics requiring unavailable ground truth.

### 8.3 `GET /dashboard/alerts`

Active alerts based on threshold-breaching metrics.

**Query params:**
- `limit` (optional, default 10)
- Global filter params

**Response:**
```json
{
  "alerts": [
    {
      "id": "oos-risk-001",
      "type": "oos_risk",
      "severity": "critical",
      "title": "OOS Risk Next 14 Days",
      "detail": "12 Items",
      "count": 12
    }
  ]
}
```

**Thresholds (configurable server-side):**
- OOS Risk: projected demand > on-hand + on-order (from `fact_inventory_snapshot`)
- Low Accuracy: DFUs with `accuracy_pct < 70`
- Bias Drift: categories with `|bias| > 20%`
- Demand Spike: items with period-over-period change > 30%

### 8.4 `GET /dashboard/top-movers`

Items with the largest period-over-period volume change.

**Query params:**
- `limit` (optional, default 5)
- `direction`: "up" | "down" | "both" (default "both")
- Global filter params

**Response:**
```json
{
  "movers": [
    {
      "item_description": "Sunset Vine Chardonnay",
      "delta": 32400,
      "pct_change": 18.2,
      "direction": "up"
    }
  ]
}
```

**Source:** `fact_sales_monthly` with `LAG()` window function for period-over-period comparison.

### 8.5 `GET /dashboard/heatmap`

Performance matrix by category and time period.

**Query params:**
- `grain`: "category" | "brand" | "location" (default "category")
- `periods` (optional, default 4): number of recent periods
- Global filter params

**Response:**
```json
{
  "rows": [
    { "label": "Wines", "values": [92.1, 88.3, 76.0, 91.5] }
  ],
  "period_labels": ["WK 18", "WK 19", "WK 20", "WK 21"],
  "metric": "accuracy_pct"
}
```

---

## 9. SQL: Dashboard Views

**File:** `sql/018_dashboard_views.sql`

```sql
-- Top movers: period-over-period volume change by item
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_movers AS
SELECT
    s.item,
    i.item_description,
    i.brand,
    i.class_ AS category,
    SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '1 month')
             THEN s.qty END) AS current_qty,
    SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '2 months')
              AND s.month_actual < (CURRENT_DATE - INTERVAL '1 month')
             THEN s.qty END) AS prior_qty,
    COALESCE(
      SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0
    ) - COALESCE(
      SUM(CASE WHEN s.month_actual >= (CURRENT_DATE - INTERVAL '2 months')
                AND s.month_actual < (CURRENT_DATE - INTERVAL '1 month') THEN s.qty END), 0
    ) AS delta
FROM fact_sales_monthly s
JOIN dim_item i ON s.item = i.item
GROUP BY s.item, i.item_description, i.brand, i.class_
ORDER BY ABS(delta) DESC
LIMIT 50;

CREATE INDEX IF NOT EXISTS idx_mv_top_movers_delta ON mv_top_movers (delta DESC);
```

---

## 10. New Files

### Layout Components

| File | Purpose |
|------|---------|
| `src/components/AppSidebar.tsx` | Collapsible sidebar navigation with sections, mobile drawer |
| `src/components/GlobalFilterBar.tsx` | Persistent cross-tab filter bar with dropdowns |
| `src/components/WidgetGrid.tsx` | CSS Grid dashboard layout wrapper |
| `src/components/ThemeSelector.tsx` | Unified theme + color mode picker |

### Dashboard Widgets

| File | Purpose |
|------|---------|
| `src/components/AlertPanel.tsx` | Alert/notification severity list |
| `src/components/HeatmapGrid.tsx` | CSS-based performance heatmap |
| `src/components/TopMovers.tsx` | Top movers ranked list |
| `src/components/ForecastTrendChart.tsx` | Stacked area forecast trend (ECharts) |
| `src/components/RecentForecastTable.tsx` | Recent forecast vs actual data table |

### Tab

| File | Purpose |
|------|---------|
| `src/tabs/DashboardTab.tsx` | Overview dashboard assembling all widgets |

### State Management

| File | Purpose |
|------|---------|
| `src/hooks/useGlobalFilters.ts` | Global filter state + URL sync |
| `src/hooks/useSidebar.ts` | Sidebar collapsed/expanded state with localStorage |
| `src/context/GlobalFilterContext.tsx` | Filter context provider |

### Theme Configs

| File | Purpose |
|------|---------|
| `src/constants/themes/wineSpirits.ts` | Wine & Spirits theme definition |
| `src/constants/themes/general.ts` | General theme definition |
| `src/constants/themes/obsidian.ts` | Obsidian theme definition |
| `src/constants/themes/index.ts` | Theme registry and exports |

### Backend

| File | Purpose |
|------|---------|
| `api/main.py` | New `/dashboard/*` and `/domains/*/distinct` endpoints |
| `sql/018_dashboard_views.sql` | Dashboard materialized views |
| `tests/api/test_dashboard.py` | API tests for dashboard endpoints |
| `tests/api/test_distinct.py` | API tests for distinct values endpoint |

---

## 11. Modified Files

| File | Changes |
|------|---------|
| `src/App.tsx` | Replace horizontal tabs with `<AppSidebar>` + `<GlobalFilterBar>` + content area; add `DashboardTab` |
| `src/index.css` | Add `[data-theme="wine-spirits"]`, `[data-theme="general"]`, `[data-theme="obsidian"]` CSS variable blocks; add `--sidebar-*` tokens |
| `src/hooks/useTheme.ts` | Extend to support `ProductTheme` selection alongside color mode |
| `src/hooks/useUrlState.ts` | Add `"overview"` to `VALID_TABS`; add global filter params to URL state |
| `src/hooks/useKeyboardShortcuts.ts` | Add `[` for sidebar toggle, `t` for theme cycle, `d` for mode toggle; renumber tab shortcuts |
| `src/components/KpiCard.tsx` | Add sparkline SVG, trend delta, severity icon support |
| `src/api/queries.ts` | Add dashboard fetch functions + global filter params to existing queries |
| `src/constants/elements.ts` | Add `overview` entry |
| `src/types/index.ts` | Add dashboard types (`Alert`, `Mover`, `HeatmapRow`, `DashboardKpis`, `GlobalFilters`) |
| `tailwind.config.ts` | Add sidebar utility classes, theme-specific animation variants |
| `index.html` | Dynamic `<title>` based on active theme |

---

## 12. Keyboard Shortcuts (Updated)

| Key | Action |
|-----|--------|
| `[` | Toggle sidebar collapsed/expanded |
| `1` | Navigate to Overview (Dashboard) |
| `2` | Navigate to Explorer |
| `3` | Navigate to DFU Analysis |
| `4` | Navigate to Accuracy |
| `5` | Navigate to Inventory |
| `6` | Navigate to Clusters |
| `7` | Navigate to Market Intel |
| `/` | Focus search in global filter bar |
| `?` | Show keyboard shortcut help |
| `Esc` | Close modals/panels |
| `t` | Cycle theme (Wine → General → Obsidian) |
| `d` | Toggle dark/light mode |
| `Ctrl+M` | Cycle motif (preserved from existing) |

---

## 13. Mobile Responsive Strategy

| Breakpoint | Sidebar | Filter Bar | Content Grid |
|------------|---------|------------|--------------|
| `< 640px` (sm) | Hidden; hamburger drawer | Collapsed to icon triggers, tap to expand | Single column (12→1) |
| `640-1024px` (md) | Collapsed icons only (64px) | Visible, horizontal scroll if needed | 6-column grid |
| `1024-1440px` (lg) | Collapsed by default, toggle to expand | Full visible | 12-column grid |
| `> 1440px` (xl) | Expanded (240px) | Full visible | 12-column grid |

**Mobile sidebar drawer:**
- Uses Radix `Sheet` primitive (already available via `@radix-ui/react-dialog`)
- Overlay backdrop with blur
- Swipe-to-close gesture (via touch events)
- Closes on navigation item click

---

## 14. Implementation Phases

### Phase 1: Foundation (~300 LOC)
- Create TypeScript types (`ProductTheme`, `SidebarThemeConfig`, `CardThemeConfig`, etc.)
- Create theme configs (`wineSpirits.ts`, `general.ts`, `obsidian.ts`)
- Add CSS variable blocks to `index.css`
- Create `useSidebar`, `useGlobalFilters` hooks
- Create `GlobalFilterContext`
- Update `useUrlState.ts` with `"overview"` tab
- Write tests for new hooks

### Phase 2: Sidebar Navigation (~500 LOC)
- Build `AppSidebar` component
- Refactor `App.tsx` — replace horizontal tabs with sidebar layout
- Mobile drawer implementation
- Add `ThemeSelector` to sidebar footer
- Update keyboard shortcuts
- Write component tests

### Phase 3: Global Filter Bar (~400 LOC)
- Build `GlobalFilterBar` component with dropdown selectors
- Add `GET /domains/{domain}/distinct` endpoint
- Wire filters to existing query hooks via context
- Write component + API tests

### Phase 4: Dashboard Overview (~1,200 LOC)
- Build all dashboard widgets (KPI cards, alert panel, heatmap, top movers, trend chart, table)
- Build `DashboardTab` assembling widgets in `WidgetGrid`
- Add `GET /dashboard/*` endpoints (4 endpoints)
- Add `sql/018_dashboard_views.sql`
- Write all component + API tests

### Phase 5: Polish (~200 LOC)
- Theme-specific loading animations
- Responsive breakpoint testing
- Print CSS updates for sidebar layout
- Accessibility audit (focus management, ARIA labels, color contrast)
- Documentation updates (CLAUDE.md, ARCHITECTURE.md, RUNBOOK.md)

**Total estimated:** ~22 new files, ~2,600 LOC

---

## 15. Testing Requirements

### New Component Tests

| File | Tests |
|------|-------|
| `src/components/__tests__/AppSidebar.test.tsx` | Renders all nav items; collapsed/expanded toggle; active state; keyboard nav; mobile drawer |
| `src/components/__tests__/GlobalFilterBar.test.tsx` | Dropdowns render; selection updates context; URL sync; reset to "All" |
| `src/components/__tests__/WidgetGrid.test.tsx` | Grid renders children with correct spans; responsive classes |
| `src/components/__tests__/AlertPanel.test.tsx` | Renders alerts sorted by severity; severity colors applied; empty state |
| `src/components/__tests__/HeatmapGrid.test.tsx` | Renders cells with computed colors; tooltip on hover; click handler |
| `src/components/__tests__/TopMovers.test.tsx` | Renders movers with +/- indicators; compact number format |
| `src/components/__tests__/ThemeSelector.test.tsx` | Theme switch applies `data-theme`; persists to localStorage; mode toggle disabled for Obsidian |
| `src/tabs/__tests__/DashboardTab.test.tsx` | Renders all widgets; loading states; error boundaries |

### New Hook Tests

| File | Tests |
|------|-------|
| `src/hooks/__tests__/useGlobalFilters.test.ts` | Filter state updates; URL sync; context propagation; default "All" values |
| `src/hooks/__tests__/useSidebar.test.ts` | Collapsed/expanded state; localStorage persistence; responsive auto-collapse |

### New API Tests

| File | Tests |
|------|-------|
| `tests/api/test_dashboard.py` | All 4 `/dashboard/*` endpoints with mocked DB pool |
| `tests/api/test_distinct.py` | `/domains/{domain}/distinct` with column, search, limit params |

### Updated Tests

| File | Changes |
|------|---------|
| `src/hooks/__tests__/useUrlState.test.ts` | Add `"overview"` to valid tabs; global filter URL params |
| `src/hooks/__tests__/useKeyboardShortcuts.test.ts` | Add `[` sidebar toggle, `t` theme cycle, `d` mode toggle |

---

## 16. Dependencies

**No new npm packages required.** The entire overhaul uses existing dependencies:

| Existing Package | Usage in Feature 36 |
|-----------------|---------------------|
| Tailwind CSS | All styling, grid layout, responsive utilities |
| Lucide React | Sidebar icons, KPI card icons |
| ECharts + echarts-for-react | Forecast trend chart, heatmap (optional) |
| @radix-ui/react-dialog | Mobile sidebar drawer (Sheet) |
| @radix-ui/react-tooltip | Collapsed sidebar icon tooltips |
| @tanstack/react-query | Dashboard data fetching + caching |
| class-variance-authority | Component variants (theme-specific) |

---

## 17. Performance Considerations

- **Sidebar:** Pure CSS transitions (GPU-accelerated `width` + `transform`), no JS animation library
- **Dashboard widgets:** Each widget independently wrapped in `<Suspense>` with skeleton fallback
- **Heatmap:** Pure CSS Grid for small grids (< 100 cells); ECharts canvas only for large grids
- **Global filters:** Debounced 300ms before propagating to queries; distinct values cached 5min
- **Theme switching:** CSS variable swap only — browser CSS engine handles repaint, no React re-render
- **Sparklines:** Inline SVG `<polyline>` — no chart library overhead for 12-point micro charts
- **Bundle impact:** Estimated +8KB gzipped (sidebar + widgets + theme configs). No new dependencies.

---

## 18. Accessibility

- **Sidebar:** `role="navigation"`, `aria-label="Main navigation"`, `aria-expanded` on toggle
- **Active nav item:** `aria-current="page"`
- **Filter bar:** `role="toolbar"`, each dropdown has `aria-label`
- **Heatmap:** `role="grid"` with `aria-label` per cell describing row, column, and value
- **Theme selector:** `role="radiogroup"` with `aria-checked`
- **Color contrast:** All three theme palettes tested for WCAG AA (4.5:1 body text, 3:1 large text)
- **Focus management:** Sidebar drawer traps focus when open on mobile, returns focus on close
- **Reduced motion:** `@media (prefers-reduced-motion: reduce)` disables sidebar transitions and loading animations
- **Skip link:** Preserved from existing implementation

---

## 19. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sidebar breaks existing tab functionality | High | Wrap in new layout without changing lazy-load / ErrorBoundary patterns; tabs are unchanged |
| Dashboard KPI data may have nulls | Medium | All KPI cards handle `null` gracefully with "N/A" display |
| Global filters add latency to every tab | Medium | Filters are opt-in per tab via context; queries only include filters when present |
| Mobile sidebar UX regression | Medium | Use Sheet drawer — proven pattern; test at 375px |
| Obsidian theme hard to read for some users | Low | Elevated "light" fallback still maintains >4.5:1 contrast ratio |
| Theme CSS specificity conflicts with motif palette | Medium | Theme `[data-theme]` selector has higher specificity than motif inline styles; motif overrides theme when active |

---

## Implementation Corrections

### Sidebar Navigation Items
Actual `AppSidebar.tsx` has **11 nav items** (spec lists 9):
- Missing from spec: `invBacktest` ("Inv. Backtest", icon: `Activity`, section: "supply", shortcut: "6")
- Missing from spec: `jobs` ("Jobs", icon: `PlayCircle`, section: "system", shortcut: "9")
- `AppSidebarProps` includes `appName: string` and `themeFooter?: React.ReactNode`

### Global Filters
Actual implementation has **6 filters** (spec lists 5):
- Missing from spec: `item: string[]` (searchable) and `location: string[]` (searchable)
- Column names: `brand_name`, `class_`, `item_no`, `location_id`, `state_id`, `rpt_channel_desc`
- `useGlobalFilters` exposes `hasActiveFilters` boolean

### Dashboard Widgets
- `RecentForecastTable.tsx` listed in spec does NOT exist

### KpiCard Props (actual vs spec)
- `value` is `string` type (not `string | number` — pre-formatted by caller)
- Has `sublabel?`, `colorClass?`, `borderClass?` (not in spec)
- `format?` property from spec is NOT implemented

### AlertPanel
- Additional alert types: `"scenario_complete"` (Feature 38), `"job_complete"` (Feature 39)

### URL State
- 11 valid tabs: overview, explorer, clusters, dfuAnalysis, accuracy, inventory, invBacktest, intel, jobs, chat, settings

### Keyboard Shortcuts
- `1-9` for tab switching (not `1-7`), shortcut 8 = Market Intel, 9 = Jobs

### Theme Implementation
- `applyPalette()` sets 30 CSS custom properties on `document.documentElement.style` (not just `[data-theme]` selectors)
- `ProductTheme.charts` has separate light/dark configs: `{ light?: ChartThemeConfig; dark: ChartThemeConfig }`
- `ThemePalette` has gradient fields: `bgGradientPrimary`, `bgGradientSecondary`, `bgGradientBaseStart/Mid/End`

### Additional Files (not in spec)
- `src/components/EChartContainer.tsx` — theme-aware ECharts wrapper
- `src/hooks/useDebounce.ts` — generic debounce hook


---

## Examples

### Example: Collapsible sidebar toggle

```typescript
// components/AppSidebar.tsx
const { collapsed, toggleSidebar } = useSidebar()
// Keyboard shortcut: '[' to toggle sidebar
// Mobile: drawer pattern (slides in from left on button tap)

// Collapsed state: shows icons only (no labels)
// Expanded state: shows icons + labels + section headers
```

### Example: Global filter bar — cross-tab state

```typescript
// context/GlobalFilterContext.tsx
import { useGlobalFilters } from '@/context/GlobalFilterContext'

function MyTab() {
  const { brand, category, item, location, setFilter } = useGlobalFilters()

  // Filters applied in dashboard, accuracy, explorer tabs
  // setFilter('item', '100320') → URL: ?item=100320
  // setFilter('location', '1401-BULK') → URL: ?item=100320&loc=1401-BULK
}
```

### Example: Dashboard KPI sparkline cards

```bash
curl -s "http://localhost:8000/dashboard/kpis" | jq '{accuracy_pct, total_forecast, total_actual, wape}'
# {"accuracy_pct": 91.8, "total_forecast": 18420300, "total_actual": 17950100, "wape": 8.2}
```

### Example: Top movers widget

```bash
curl -s "http://localhost:8000/dashboard/top-movers?window=3&limit=5" | jq '.rows[0]'
# {"item_no":"100320","loc":"1401-BULK","pct_change": 18.4, "direction": "up", "qty_delta": 127}
```
