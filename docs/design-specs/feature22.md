# Feature 22 — UI Theming: Light, Dark & Midnight Modes with Settings Panel

## Overview

Add a global theme system to Demand Studio with three distinct visual themes — **Light** (current), **Dark**, and **Midnight** — selectable from a new Settings panel accessible from the header. Theme preference persists in `localStorage` and applies instantly across all tabs and components.

## Problem

The current UI uses a single hardcoded light theme (warm stone/slate palette with gradient backgrounds). Users working in low-light environments or who prefer darker interfaces have no option to switch. A theming system improves:

1. **Accessibility** — reduces eye strain in different lighting conditions
2. **User preference** — professionals working long sessions benefit from dark UIs
3. **Visual polish** — a theme switcher signals a mature, production-quality application

## Three Themes

### 1. Light (Default — Current Palette)

The existing warm platinum/stone aesthetic. No visual changes — this becomes the baseline `light` theme.

| Token | HSL Value | Visual |
|-------|-----------|--------|
| `--background` | `220 15% 96%` | Warm off-white |
| `--foreground` | `230 30% 14%` | Deep navy text |
| `--card` | `0 0% 100%` | Pure white cards |
| `--primary` | `230 65% 28%` | Navy blue |
| `--secondary` | `45 80% 92%` | Light amber |
| `--accent` | `43 90% 55%` | Gold |
| `--muted` | `225 18% 93%` | Cool gray |
| `--border` | `225 14% 82%` | Light gray border |

Body gradient: radial indigo/amber blurs on warm off-white base.

### 2. Dark

A deep charcoal theme with muted accent colors. High contrast text on dark surfaces.

| Token | HSL Value | Visual |
|-------|-----------|--------|
| `--background` | `220 15% 10%` | Deep charcoal |
| `--foreground` | `220 10% 90%` | Light gray text |
| `--card` | `220 15% 13%` | Slightly lighter card |
| `--card-foreground` | `220 10% 90%` | Light gray |
| `--primary` | `230 55% 55%` | Medium blue |
| `--primary-foreground` | `0 0% 100%` | White |
| `--secondary` | `45 50% 20%` | Dark amber |
| `--secondary-foreground` | `45 60% 85%` | Light amber text |
| `--accent` | `43 80% 50%` | Warm gold |
| `--accent-foreground` | `0 0% 100%` | White |
| `--muted` | `220 12% 18%` | Dark muted |
| `--muted-foreground` | `220 10% 60%` | Medium gray |
| `--border` | `220 12% 22%` | Subtle border |
| `--input` | `220 12% 22%` | Input border |
| `--ring` | `230 55% 55%` | Focus ring |

Body gradient: radial indigo/amber blurs at low opacity on dark charcoal base.

### 3. Midnight

A rich deep-blue theme with cool-toned accents. Distinct from Dark — uses blue-shifted backgrounds and cyan/teal highlights.

| Token | HSL Value | Visual |
|-------|-----------|--------|
| `--background` | `230 25% 9%` | Deep navy |
| `--foreground` | `210 20% 88%` | Cool light gray |
| `--card` | `230 22% 12%` | Navy card |
| `--card-foreground` | `210 20% 88%` | Cool gray |
| `--primary` | `200 80% 55%` | Cyan blue |
| `--primary-foreground` | `230 25% 9%` | Dark navy |
| `--secondary` | `260 40% 20%` | Deep purple |
| `--secondary-foreground` | `260 30% 80%` | Light lavender |
| `--accent` | `175 70% 45%` | Teal |
| `--accent-foreground` | `0 0% 100%` | White |
| `--muted` | `230 20% 15%` | Muted navy |
| `--muted-foreground` | `230 15% 55%` | Blue-gray |
| `--border` | `230 18% 20%` | Navy border |
| `--input` | `230 18% 20%` | Input border |
| `--ring` | `200 80% 55%` | Cyan focus ring |

Body gradient: radial cyan/purple blurs at low opacity on deep navy base.

## Architecture

### Theme Storage & Application

```
User clicks theme button in Settings
        ↓
setState(theme) + localStorage.setItem("ds-theme", theme)
        ↓
Apply class to <html> element: "light" | "dark" | "midnight"
        ↓
CSS variables cascade automatically via :root / .dark / .midnight selectors
        ↓
All Tailwind + shadcn components update instantly (no re-render needed for CSS)
```

### CSS Variable Strategy

The existing shadcn/ui pattern uses CSS custom properties consumed via `hsl(var(--token))` in Tailwind config. This means theme switching is purely a CSS concern — change the custom property values and every component updates.

**index.css structure:**

```css
/* Light theme (default) */
:root {
  --background: 220 15% 96%;
  --foreground: 230 30% 14%;
  /* ... existing values ... */
}

/* Dark theme */
.dark {
  --background: 220 15% 10%;
  --foreground: 220 10% 90%;
  /* ... dark values ... */
}

/* Midnight theme */
.midnight {
  --background: 230 25% 9%;
  --foreground: 210 20% 88%;
  /* ... midnight values ... */
}
```

### Hardcoded Color Overrides

The current App.tsx uses hardcoded Tailwind classes that bypass CSS variables (e.g., `bg-stone-400/10`, `text-slate-800`, `border-stone-300`). These must be migrated to semantic token classes for theming to work:

| Current (Hardcoded) | Replacement (Semantic) |
|---------------------|----------------------|
| `bg-gradient-to-br from-[#E5E4E2] via-[#EDEDEB] to-[#E5E4E2]` | `bg-gradient-to-br from-background via-muted to-background` or CSS variable gradient |
| `text-slate-800` | `text-foreground` |
| `text-slate-500` | `text-muted-foreground` |
| `text-slate-600` | `text-muted-foreground` |
| `border-stone-300` | `border-border` |
| `bg-stone-400/10` | `bg-muted/40` or custom token |
| `bg-stone-400/30` | `bg-muted` |
| `ring-stone-400/50` | `ring-border` |
| `via-stone-400/30` | `via-border/30` |
| `bg-red-50`, `border-red-200`, `text-red-700` | Keep as-is (semantic error colors work in all themes) |
| `bg-green-*`, `bg-yellow-*`, `bg-blue-*` | Keep status colors but add theme-aware opacity variants |

### Body Background Gradient

The body background gradient in `index.css` must also be theme-aware:

```css
:root {
  --bg-gradient-primary: rgba(79, 70, 229, 0.12);
  --bg-gradient-secondary: rgba(217, 119, 6, 0.10);
  --bg-gradient-base-start: #eef0f6;
  --bg-gradient-base-mid: #f0eee8;
  --bg-gradient-base-end: #eae8e2;
}

.dark {
  --bg-gradient-primary: rgba(79, 70, 229, 0.08);
  --bg-gradient-secondary: rgba(217, 119, 6, 0.06);
  --bg-gradient-base-start: #161a22;
  --bg-gradient-base-mid: #181c24;
  --bg-gradient-base-end: #1a1e26;
}

.midnight {
  --bg-gradient-primary: rgba(56, 189, 248, 0.08);
  --bg-gradient-secondary: rgba(139, 92, 246, 0.06);
  --bg-gradient-base-start: #0f1525;
  --bg-gradient-base-mid: #111729;
  --bg-gradient-base-end: #10162a;
}

body {
  background-image:
    radial-gradient(circle at 12% 10%, var(--bg-gradient-primary), transparent 36%),
    radial-gradient(circle at 86% 0%, var(--bg-gradient-secondary), transparent 30%),
    linear-gradient(140deg, var(--bg-gradient-base-start) 0%, var(--bg-gradient-base-mid) 62%, var(--bg-gradient-base-end) 100%);
}
```

### Recharts Theme Integration

Recharts components use inline color props. These need to read from CSS variables or be passed theme-aware color constants:

```typescript
const CHART_COLORS = {
  light: {
    grid: "#e2e8f0",
    axis: "#64748b",
    tooltip_bg: "#ffffff",
    tooltip_border: "#e2e8f0",
  },
  dark: {
    grid: "#2d3548",
    axis: "#94a3b8",
    tooltip_bg: "#1e2433",
    tooltip_border: "#2d3548",
  },
  midnight: {
    grid: "#1e2744",
    axis: "#7b8db5",
    tooltip_bg: "#141c33",
    tooltip_border: "#1e2744",
  },
};
```

## Implementation Design

### State Management

Use React state + `localStorage` persistence. No external state library needed.

```typescript
type Theme = "light" | "dark" | "midnight";

function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem("ds-theme") as Theme | null;
    return saved && ["light", "dark", "midnight"].includes(saved) ? saved : "light";
  });

  useEffect(() => {
    const root = document.documentElement;
    // Remove all theme classes
    root.classList.remove("light", "dark", "midnight");
    // Add current theme class
    root.classList.add(theme);
    // Persist
    localStorage.setItem("ds-theme", theme);
  }, [theme]);

  return { theme, setTheme };
}
```

### Tailwind Config Update

The existing `darkMode: ["class"]` config already supports class-based switching. No changes needed to `tailwind.config.ts` — the `.dark` and `.midnight` class selectors work through the CSS variables.

However, for `.midnight` to work with Tailwind's built-in `dark:` variant, we extend with a custom variant or rely purely on CSS variable switching (recommended approach — no `dark:` prefix needed since all colors flow through CSS variables).

### Settings Panel UI

A gear icon button in the header area that opens a settings dropdown/popover. Minimal scope — theme selection only (extensible for future settings).

```
┌──────────────────────────────────────────────────────────────┐
│  ⚗ Planthium                    [Da] [Ac] [Fc] ... [⚙]    │
│  Periodic Analytics for Demand Forecasting                   │
└──────────────────────────────────────────────────────────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │    Settings        │
                                          │─────────────────── │
                                          │  Theme             │
                                          │  ┌───┐ ┌───┐ ┌───┐│
                                          │  │ ☀ │ │ 🌙│ │ 🌊││
                                          │  │Lgt│ │Drk│ │Mid││
                                          │  └───┘ └───┘ └───┘│
                                          │  [selected: ✓]     │
                                          └────────────────────┘
```

The settings button uses a periodic-table element tile style (consistent with the existing tab navigation) with the symbol "St" (Settings) and atomic number matching the next available number in the UI sequence.

### Settings Button Component

```tsx
const THEME_OPTIONS: { value: Theme; label: string; icon: string }[] = [
  { value: "light", label: "Light", icon: "☀️" },
  { value: "dark", label: "Dark", icon: "🌙" },
  { value: "midnight", label: "Midnight", icon: "🌊" },
];

// Inside the header, after the existing tab buttons:
<button
  onClick={() => setShowSettings(!showSettings)}
  className={cn(
    "relative flex h-[62px] w-[58px] flex-col items-center justify-center rounded-lg",
    "border px-2 py-1 transition-all duration-200 cursor-pointer",
    "border-border/50 bg-card/50 text-muted-foreground hover:bg-accent/20"
  )}
>
  <Settings className="h-5 w-5" />
  <span className="text-[9px] font-medium">Settings</span>
</button>

{/* Settings dropdown */}
{showSettings && (
  <div className="absolute right-0 top-full mt-2 z-50 w-64 rounded-lg border border-border bg-card p-4 shadow-xl">
    <h3 className="text-sm font-semibold text-foreground mb-3">Theme</h3>
    <div className="flex gap-2">
      {THEME_OPTIONS.map((opt) => (
        <button
          key={opt.value}
          onClick={() => { setTheme(opt.value); setShowSettings(false); }}
          className={cn(
            "flex-1 flex flex-col items-center gap-1 rounded-md border p-3 transition-all",
            theme === opt.value
              ? "border-primary bg-primary/10 text-primary"
              : "border-border text-muted-foreground hover:border-primary/50"
          )}
        >
          <span className="text-lg">{opt.icon}</span>
          <span className="text-xs font-medium">{opt.label}</span>
        </button>
      ))}
    </div>
  </div>
)}
```

### Click-Outside Dismiss

The settings dropdown should close when clicking outside:

```typescript
const settingsRef = useRef<HTMLDivElement>(null);

useEffect(() => {
  if (!showSettings) return;
  const handler = (e: MouseEvent) => {
    if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
      setShowSettings(false);
    }
  };
  document.addEventListener("mousedown", handler);
  return () => document.removeEventListener("mousedown", handler);
}, [showSettings]);
```

## Hardcoded Color Migration Checklist

All hardcoded Tailwind color classes in `App.tsx` that reference specific stone/slate/gray shades must be audited and migrated to semantic tokens. Key areas:

1. **Header section** — `bg-gradient-to-br from-[#E5E4E2]`, `text-slate-800`, `border-stone-300`, etc.
2. **Tab buttons** — `bg-stone-400/30`, `ring-stone-400/50`, active state colors
3. **Loading overlay** — `bg-indigo-*` periodic table element styles
4. **Table headers** — `bg-muted` (already semantic)
5. **KPI cards** — green/red/yellow status colors (keep but ensure readable on dark backgrounds)
6. **Chat panel** — message bubble colors
7. **Chart tooltips** — background/border colors
8. **Error cards** — `bg-red-50`, `border-red-200` (need dark variants)

### Status/Semantic Colors (Shared Across Themes)

Some colors are semantic (error, success, warning) and need theme-aware variants:

```css
:root {
  --success: 142 71% 45%;
  --success-foreground: 0 0% 100%;
  --warning: 38 92% 50%;
  --warning-foreground: 0 0% 100%;
  --error: 0 84% 60%;
  --error-foreground: 0 0% 100%;
}

.dark, .midnight {
  --success: 142 60% 40%;
  --success-foreground: 142 60% 90%;
  --warning: 38 80% 45%;
  --warning-foreground: 38 80% 90%;
  --error: 0 72% 50%;
  --error-foreground: 0 72% 90%;
}
```

## Transition & Animation

Apply a smooth CSS transition when switching themes:

```css
html {
  transition: background-color 200ms ease, color 200ms ease;
}

html * {
  transition: background-color 200ms ease, color 200ms ease, border-color 200ms ease;
}
```

**Note:** Apply transition only during theme switch (not on page load) to avoid flash. Use a `data-transitioning` attribute:

```typescript
// In useTheme hook:
root.setAttribute("data-transitioning", "true");
root.classList.remove("light", "dark", "midnight");
root.classList.add(theme);
setTimeout(() => root.removeAttribute("data-transitioning"), 300);
```

```css
html[data-transitioning] * {
  transition: background-color 200ms ease, color 200ms ease, border-color 200ms ease !important;
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `mvp/demand/frontend/src/index.css` | Modify | Add `.dark` and `.midnight` CSS variable blocks, body gradient variables, transition styles |
| `mvp/demand/frontend/src/App.tsx` | Modify | Add `useTheme` hook, settings button/dropdown, migrate hardcoded colors to semantic tokens, pass theme to chart colors |
| `mvp/demand/frontend/tailwind.config.ts` | No change | Already has `darkMode: ["class"]`, CSS variables handle the rest |
| `docs/design-specs/feature22.md` | Create | This spec |

## Dependencies

No new npm packages required. The theming system uses:
- Existing CSS custom properties (shadcn/ui pattern)
- Existing Tailwind `darkMode: ["class"]` config
- `localStorage` for persistence
- `lucide-react` `Settings` icon (already available in the project)

## Testing & Validation

### Manual Testing Checklist

1. **Theme persistence** — Select Dark, refresh page → Dark should persist
2. **All tabs** — Navigate every tab (Data, Accuracy, Forecast, DFU Analysis, Market Intel, Chat) in each theme
3. **Charts** — Verify Recharts grid lines, axis labels, tooltips are readable in all themes
4. **KPI cards** — Green/red/yellow status indicators visible in all themes
5. **Loading overlay** — Chemistry element tile visible and animated in all themes
6. **Error states** — Red error cards readable in dark themes
7. **Input fields** — Filters, search boxes, dropdowns visible with proper contrast
8. **Scrollbars** — Check scrollbar track/thumb visibility in dark themes
9. **Settings dropdown** — Opens/closes correctly, click-outside dismisses
10. **Initial load** — No flash of wrong theme on page load (read localStorage before first render)

### Contrast Requirements

All theme combinations should maintain WCAG AA minimum contrast ratios:
- Normal text: 4.5:1
- Large text: 3:1
- UI components: 3:1

## Future Enhancements (Out of Scope for Feature 22)

1. **System preference detection** — Auto-select light/dark based on `prefers-color-scheme` media query
2. **Custom theme builder** — Let users pick primary/accent colors
3. **Per-chart color palettes** — Theme-aware multi-series chart colors
4. **High contrast mode** — Accessibility-focused theme with maximum contrast
5. **Theme scheduling** — Auto-switch between light (day) and dark (night) on a schedule

---

## Implementation Corrections

### Architecture Redesign
The theming system has been completely redesigned from the spec's 3-theme approach (light/dark/midnight):

### Actual Implementation
- 3 **product themes**: `wine-spirits`, `general`, `obsidian` — each with **light AND dark** palettes
- `useTheme()` returns `themeId`, `colorMode`, `effectiveClass`, `productTheme`, plus methods (`cycleTheme`, etc.)
- Two localStorage keys: `ds-product-theme` and `ds-color-mode`
- `ThemeSelector` in sidebar footer (not header settings gear)
- Radio group with 3 theme options + separate Light/Dark toggle

### Type System (`types/theme.ts`)
- `ProductThemeId = "wine-spirits" | "general" | "obsidian"`
- `ThemePalette`: 33 CSS variable tokens including sidebar, chart, KPI, and gradient tokens
- `SidebarThemeConfig`, `CardThemeConfig`, `ChartThemeConfig`, `TypographyConfig`
- `ProductTheme`: composite interface with id, displayName, tagline, palette, sidebar, cards, charts, typography

### Additional CSS Variables
- `--sidebar-bg`, `--sidebar-foreground`, `--sidebar-active`, `--sidebar-hover`
- `--chart-1` through `--chart-6`
- `--kpi-best`, `--kpi-warning`, `--kpi-ceiling`
- `--bg-gradient-primary`, `--bg-gradient-secondary`, `--bg-gradient-base-start/mid/end`

### Behavior
- Obsidian theme always stays in dark mode
- `cycleTheme()` cycles: wine-spirits → general → obsidian
- Runtime palette via `applyPalette()` setting CSS vars on `document.documentElement` (not static CSS selectors)
- Keyboard shortcuts: `t` cycles theme, `d` toggles dark mode

### Midnight Theme — Obsolete
- `.midnight` CSS class NO LONGER EXISTS — replaced by product theme system

### File Locations
- `hooks/useTheme.ts`, `components/ThemeSelector.tsx`, `constants/themes/*.ts`, `types/theme.ts`
- Tests: `useTheme.test.ts` (8 tests), `ThemeSelector.test.tsx` (8 tests)


---

## Examples

### Example: Toggle dark mode via keyboard shortcut

```
Keyboard shortcut: Press 'd' to toggle dark/light mode
```

### Example: CSS variable palette (light vs dark)

```css
/* Light mode (default) */
:root {
  --bg-primary: #ffffff;
  --text-primary: #111827;
  --sidebar-bg: #1e293b;
  --chart-1: #3b82f6;
  --chart-2: #10b981;
  --chart-3: #f59e0b;
}

/* Dark mode override */
.dark {
  --bg-primary: #0f172a;
  --text-primary: #f1f5f9;
  --chart-1: #60a5fa;
  --chart-2: #34d399;
}
```

### Example: useThemeContext hook

```typescript
import { useThemeContext } from '@/context/ThemeContext'

function ThemeToggle() {
  const { colorMode, toggleColorMode, theme } = useThemeContext()
  return (
    <button onClick={toggleColorMode}>
      {colorMode === 'dark' ? 'Light Mode' : 'Dark Mode'}
    </button>
  )
}
// theme.name = "Demand Studio"  (single theme, light/dark only)
```

### Example: Chart colors via useChartColors

```typescript
import { useChartColors } from '@/hooks/useChartColors'
const { chartColors, trendColors } = useChartColors()
// chartColors[0] = '#3b82f6' (light) or '#60a5fa' (dark)
// Use in Recharts: <Line stroke={chartColors[0]} />
```
