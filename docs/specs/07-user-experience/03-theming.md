# Theming

> A single professional theme ("Supply Chain Command Center") with light and dark color modes, implemented via CSS custom properties and toggled from the sidebar footer or keyboard shortcut.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (platform-wide) |
| **Key Files** | `hooks/useTheme.ts`, `context/ThemeContext.tsx`, `components/ThemeSelector.tsx`, `constants/themes/general.ts`, `tailwind.config.ts` |

---

## Problem

A data-heavy analytics application needs two visual modes: a light mode for well-lit offices and presentations, and a dark mode for extended monitoring sessions and low-light environments. The theme system must propagate colors consistently to all components (sidebar, cards, charts, tables, alerts) without prop-drilling and without requiring each component to maintain its own color logic.

---

## Solution

A single theme definition in `constants/themes/general.ts` provides two complete color palettes (light and dark). The `useTheme` hook manages the active mode in state. `ThemeContext` makes the current theme available to any component via `useThemeContext()`. Charts access colors via the `useChartColors()` hook. The `ThemeSelector` component in the sidebar footer provides a toggle button. The `d` keyboard shortcut toggles modes globally.

---

## How It Works

### Color Mode Toggle

| Trigger | Action |
|---|---|
| Sidebar toggle button | Clicks cycle between light and dark |
| `d` keyboard shortcut | Toggles dark mode on/off |
| System preference | Respects `prefers-color-scheme` on first load |

Dark mode adds the `dark` class to the `<html>` element. Tailwind's `dark:` variant applies dark-specific styles. CSS custom properties in `:root` and `.dark` scope provide the base palette.

### Theme Structure

The `general.ts` theme config defines:

| Section | What It Contains |
|---|---|
| `brand` | Product name ("Supply Chain Command Center"), logo settings |
| `colors.light` | Light mode palette: background, foreground, card, border, primary, accent, muted, destructive |
| `colors.dark` | Dark mode palette: same keys, different values |
| `chart` | Chart-specific colors: 6 series colors, trend line colors, grid/axis colors |
| `sidebar` | Sidebar background, text, hover, active, border colors per mode |

### Chart Colors

The `useChartColors()` hook returns `{ theme, chartColors, trendColors }` derived from the active theme context. All chart components consume these colors instead of hardcoded values. recharts is the default chart engine; the `ModularReactECharts` component (`echarts-modular.tsx`) is used only for the 8 heavy customer-analytics panels and reads the same theme-derived colors. The retired `EChartContainer` wrapper has been removed.

### Design Tokens

`constants/design-tokens.ts` defines semantic tokens used across the application:

| Token Category | Examples |
|---|---|
| Severity colors | Critical (red), high (orange), medium (yellow), low (blue) |
| AI colors | AI accent (teal) for AI Planner elements |
| UX limits | `MAX_PRIMARY_KPIS = 4` to prevent dashboard clutter |
| Confidence thresholds | HIGH/MED/LOW boundaries for AI insight confidence badges |

### What Was Removed

Earlier iterations included three themes (General, Scientific, Executive) with motif-specific styling. These were consolidated into a single theme to reduce maintenance surface. The `ThemeSelector` was simplified from a theme cycler to a light/dark toggle. Old theme files in `constants/themes/` (scientific.ts, executive.ts) were deleted.

---

## Component Integration

| Component | How It Gets Colors |
|---|---|
| Sidebar, cards, tables | Tailwind `dark:` variant + CSS custom properties |
| recharts instances (default) | `useChartColors()` hook provides series colors |
| `ModularReactECharts` (8 CA panels) | `useChartColors()` colors passed into the echarts option object |
| KPI cards, alerts | Semantic CSS variables (`--destructive`, `--primary`, etc.) |
| Skeleton loading | `animate-pulse` with theme-aware background |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Tailwind CSS `dark:` variant | CSS-level dark mode support |
| React Context API | Theme propagation without prop-drilling |
| `constants/themes/general.ts` | Single source of truth for all colors |
| `constants/design-tokens.ts` | Semantic color tokens for severity, AI, and UX limits |

---

## See Also

- `07-user-experience/02-ui-architecture.md` -- application shell and component library that consume the theme
- `07-user-experience/05-testing.md` -- E2E theme toggle tests in `theme.spec.ts`
