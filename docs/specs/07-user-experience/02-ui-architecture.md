# UI Architecture

> The React single-page application architecture: Vite build tooling, TanStack Query for server state, lazy-loaded tabs, collapsible sidebar navigation, global filter bar, dashboard overview, and the component library powering all frontend features.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (platform-wide) |
| **Key Files** | `App.tsx`, `AppSidebar.tsx`, `GlobalFilterBar.tsx`, `vite.config.ts`, `tailwind.config.ts` |

---

## Problem

A supply chain platform with 16 lazy-loaded tabs, 28 inventory planning sub-tabs, and dozens of charts, tables, and KPI cards needs an architecture that stays fast as features accumulate. Without lazy loading, the initial bundle would be enormous. Without server-state caching, every tab switch would re-fetch data. Without a shared filter context, planners would re-enter brand/location filters on every tab. The architecture must support a single developer adding features without drowning in prop-drilling or state management complexity.

---

## Solution

A React + Vite + TypeScript stack with TanStack Query for server state, React.lazy for code splitting, a global filter context for cross-tab state, and Tailwind CSS + shadcn/ui for styling. The sidebar provides navigation across 12 tabs organized in 5 sections (operations-first layout). A dashboard landing page shows KPI cards, alerts, heatmap, top movers, and forecast trends.

---

## How It Works

### Application Shell

| Layer | Technology | Purpose |
|---|---|---|
| Build | Vite | Fast HMR, ESM-native bundling, API proxy to FastAPI |
| Routing | URL query params (`?tab=accuracy`) | No React Router -- single-page with tab state in URL |
| Server state | TanStack Query | Stale-while-revalidate caching, instant tab switching, background refetch |
| Styling | Tailwind CSS + shadcn/ui | Utility-first CSS with pre-built accessible components |
| Charts | ECharts (canvas) + Recharts (SVG) | ECharts for large datasets, Recharts for interactive charts |
| Data grid | TanStack Table + TanStack Virtual | Column resize, row selection, virtualized scrolling for large tables |

### Sidebar Navigation

12 tabs organized in 5 sections (operations-first):

| Section | Tabs |
|---|---|
| Tower | Command Center |
| Operations | S&OP, Jobs, Data Quality |
| Supply | Inv. Planning, Clusters, Inv. Backtest |
| Demand | Portfolio, Item Analysis, FVA & ROI, Customer Map |
| System | Explorer |

Keyboard shortcuts: `1`-`7` switch tabs (1=Command Center, 2=S&OP, 3=Jobs, 4=Inv. Planning, 5=Clusters, 6=Portfolio, 7=Item Analysis), `[` toggles sidebar collapse, `d` toggles dark mode, `?` opens help modal.

### Global Filter Bar

Six filter dimensions applied across Dashboard, Accuracy, and auto-populated into tab-local inputs:

| Filter | Type | Behavior |
|---|---|---|
| Brand | Multi-select dropdown | Filters by `dim_item.brand` |
| Category | Multi-select dropdown | Filters by `dim_item.category` |
| Item | Searchable multi-select | Filters by `dim_item.item_id` |
| Location | Searchable multi-select | Filters by `dim_location.loc` |
| Market | Multi-select dropdown | Filters by `dim_location.market` |
| Channel | Multi-select dropdown | Filters by `dim_customer.channel` |

Filter state is managed via `useGlobalFilters` hook with debounced URL synchronization. The filter bar is hidden on tabs where it does not apply (AI Planner, Chat, Jobs).

### Dashboard Overview

The landing page renders five zones:

| Zone | Component | Data Source |
|---|---|---|
| KPI cards | `KpiCard` (4 cards with sparklines) | `/dashboard/kpis` |
| Alert panel | Severity-coded alert list | `/dashboard/alerts` |
| Heatmap | `HeatmapGrid` (CSS Grid with color scale) | `/dashboard/heatmap` |
| Top movers | `TopMovers` (period-over-period changes) | `/dashboard/top-movers` |
| Trend chart | `ForecastTrendChart` (recharts `ComposedChart`) | `/dashboard/trend` |

### Code Splitting

Every tab is loaded via `React.lazy()` with a per-tab error boundary. This keeps the initial bundle small and isolates tab-level failures. TanStack Query's stale-while-revalidate strategy means data fetched for one tab persists in cache when switching away and back.

#### Sub-Panel Lazy Loading (`LazyPanel`)

`React.lazy()` only defers JS chunk loading; the moment the chunk resolves, every wrapped `useQuery` fires. For tabs with many below-the-fold panels (e.g. `CustomerAnalyticsTab` had 13+ initial-mount queries) this still produced a stampede of API calls on tab open.

`frontend/src/components/LazyPanel.tsx` wraps each below-the-fold panel in an `IntersectionObserver` boundary. The wrapped component (and therefore its `useQuery`) is not mounted until the panel scrolls within `rootMargin: 200px` of the viewport. Once visible the panel stays mounted (`triggerOnce` semantics) so subsequent filter changes do not retrigger the boundary.

Test environments use a polyfill installed in `src/__tests__/setup.ts` that fires `isIntersecting: true` immediately, so panels render eagerly under vitest.

Bundle wins from this session:
- `CustomerAnalyticsTab` migrated 8 ECharts panels to recharts → −728 KB raw across the tab.
- `InvPlanningTab`: −91% initial JS via `LazyPanel` wrapping of sub-panels.
- `ModelTuningTab`: −89% initial JS via the same pattern.
- `HeatmapGrid` extended to absorb panels previously rendered by ECharts heatmaps.

### Vite API Proxy

`vite.config.ts` proxies 17 path prefixes to the FastAPI backend at `http://127.0.0.1:8000`. When adding a new API path prefix, a corresponding proxy entry must be added or the frontend receives HTML instead of JSON.

| Proxy Prefix Examples |
|---|
| `/domains`, `/jobs`, `/clustering`, `/forecast`, `/inventory`, `/dashboard` |
| `/inv-planning`, `/fill-rate`, `/control-tower`, `/ai-planner`, `/storyboard` |
| `/market-intelligence`, `/sku`, `/competition`, `/health`, `/bench` |

---

## Component Library

| Component | Purpose |
|---|---|
| `DataTable` | Virtualized data grid with column resize, sort, select, CSV export |
| `KpiCard` | Metric card with label, value, delta, optional sparkline |
| `WidgetGrid` / `WidgetCard` | CSS Grid dashboard layout containers |
| `HeatmapGrid` | Color-scaled CSS Grid for cross-dimensional views (extended this session to back panels migrated off ECharts heatmaps) |
| `LazyPanel` | `IntersectionObserver`-based deferred render — wraps below-the-fold panels so their `useQuery` does not fire on tab mount |
| `TopMovers` | Period-over-period mover list with up/down indicators |
| `ForecastTrendChart` | recharts `ComposedChart` — forecast/actual lines plus an optional shaded 80% CI band |
| `EmptyState` | Placeholder when no data matches filters |
| `Skeleton` | Loading shimmer placeholder |
| `LoadingElement` | Chemistry-themed periodic table loading animation |
| `KeyboardShortcutHelp` | Modal listing all keyboard shortcuts |

---

## Context Providers

| Context | Purpose |
|---|---|
| `ThemeContext` | Light/dark mode state, eliminates theme prop-drilling |
| `GlobalFilterContext` | Cross-tab filter state with debounced URL sync |
| `ScenarioNotificationContext` | Cross-tab clustering scenario completion alerts |
| `JobNotificationContext` | Cross-tab job completion alerts, sidebar badge count |

---

## Dependencies

| Dependency | Reason |
|---|---|
| `react`, `react-dom` | UI framework |
| `@tanstack/react-query` | Server state management |
| `@tanstack/react-table`, `@tanstack/react-virtual` | Data grid + virtualization |
| `echarts`, `echarts-for-react` | Canvas-based charting |
| `recharts` | SVG charting (SHAP, scenario charts) |
| `tailwindcss`, `shadcn/ui` | Styling and component library |
| `papaparse` | CSV export |
| `lucide-react` | Icon set |

---

## See Also

- `07-user-experience/03-theming.md` -- light/dark mode implementation
- `07-user-experience/01-data-explorer.md` -- data grid and item analysis details
- `07-user-experience/04-job-scheduler.md` -- job notifications and sidebar badge
- `07-user-experience/05-testing.md` -- frontend test infrastructure
