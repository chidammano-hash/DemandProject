# Theming

> A single professional theme ("Supply Chain Command Center") with light, soft, and dark color modes, implemented through one semantic palette, CSS custom properties, and Tailwind tokens.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (platform-wide) |
| **Key Files** | `constants/palette.ts`, `hooks/useTheme.ts`, `hooks/useChartColors.ts`, `context/ThemeContext.tsx`, `components/ThemeSelector.tsx`, `constants/themes/general.ts`, `tailwind.config.ts` |

---

## Problem

A data-heavy analytics application needs a daylight mode, a lower-contrast planner-paper mode, and a dark monitoring mode. The theme system must propagate colors consistently to all components without prop-drilling, and the same business concept must retain the same semantic color across every screen.

---

## Solution

`constants/palette.ts` is the color source of truth for light, soft, and dark modes. `constants/themes/general.ts`, legacy chart exports, runtime CSS variables, and Tailwind tokens derive from it. The `useTheme` hook manages the active mode, `ThemeContext` exposes it without prop-drilling, and charts consume named semantic roles through `useChartColors()`.

---

## How It Works

### Color Mode Toggle

| Trigger | Action |
|---|---|
| Sidebar toggle button | Cycles light → soft → dark |
| `d` keyboard shortcut | Cycles the configured modes globally |
| System preference | Respects `prefers-color-scheme` on first load |

Dark mode adds `dark` to the `<html>` element. Soft mode adds `light soft`, keeping `dark:` variants disabled while enabling the custom `soft:` Tailwind variant. The root also carries `data-mode`, and fallback CSS blocks mirror the palette for first paint.

### Theme Structure

The `general.ts` theme config defines:

| Section | What It Contains |
|---|---|
| `brand` | Product name ("Supply Chain Command Center"), logo settings |
| `palette.light/soft/dark` | Mode-specific core tokens derived from `constants/palette.ts` |
| `charts` | Eight categorical colors, named semantic roles, heatmap scale, fallback series, and chart chrome |
| `sidebar` | Sidebar background, text, hover, active, border colors per mode |

### Chart Colors

The `useChartColors()` hook returns mode-aware `roles`, `series`, `heatmap`, `fallback`, and `chartColors`. Named roles cover actual, forecast/champion, good, warning, reference, error, capacity/ceiling, and AI. New charts use roles whenever the series has business meaning; positional `series` colors are reserved for unnamed categories. `TREND_COLORS_BY_THEME`, `OKABE_ITO`, and other exports in `constants/colors.ts` remain compatibility aliases while older charts migrate.

`paletteSync.test.ts` prevents the TypeScript palette and CSS fallbacks from drifting, verifies that semantic roles belong to the categorical series, and enforces WCAG text and graphical contrast gates in every mode.

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
| Sidebar, cards, tables | Semantic Tailwind tokens + CSS custom properties; `dark:`/`soft:` only for genuine mode divergences |
| recharts instances (default) | `useChartColors()` provides semantic roles and categorical series |
| `ModularReactECharts` (8 CA panels) | `useChartColors()` colors passed into the echarts option object |
| KPI cards, alerts | Semantic CSS variables (`--destructive`, `--primary`, etc.) |
| Skeleton loading | `animate-pulse` with theme-aware background |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Tailwind CSS `dark:` and custom `soft:` variants | CSS-level mode-specific differences |
| React Context API | Theme propagation without prop-drilling |
| `constants/palette.ts` | Single source of truth for all mode and chart colors |
| `constants/themes/general.ts` | Product-theme metadata derived from the palette |
| `constants/design-tokens.ts` | Semantic color tokens for severity, AI, and UX limits |

---

## See Also

- `07-user-experience/02-ui-architecture.md` -- application shell and component library that consume the theme
- `07-user-experience/05-testing.md` -- E2E theme toggle tests in `theme.spec.ts`
