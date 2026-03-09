# Multi-Theme Testing Strategy for Demand Studio

**Document Version:** 1.0
**Date:** February 24, 2026
**Status:** Design Specification

---

## Executive Summary

This document defines the **complete testing strategy for the multi-theme system** in Demand Studio. We are adding 5 configurable motif themes (Periodic Table, Hospitality/Spirits, Space, F1 Racing, Zen Garden) to complement the existing color mode system (light/dark/midnight).

The testing strategy ensures:
- All 5 themes register correctly with complete metadata
- Theme state management (selection, persistence, fallback) works reliably
- Theme changes propagate across all UI components
- Visual correctness across 15 combinations (5 themes × 3 color modes)
- Accessibility compliance (contrast, focus, labels) across all combinations
- Performance (memoization, re-render prevention)

**Total Test Count:** 52 tests across 8 test files

---

## 1. Theme Architecture Overview

### Current State
- **Color Modes:** 3 (light, dark, midnight) — managed by `useTheme()` hook
- **Element Registry:** Single periodic table theme with 12 tab configs in `ELEMENT_CONFIG`

### Proposed State
- **Motif Themes:** 5 themes (periodic-table, hospitality, space, f1-racing, zen-garden)
- **Themes Registry:** New `src/lib/themes.ts` with theme definitions
- **Theme Hook:** New `useMotifTheme()` hook for theme selection/persistence
- **Settings Component:** New `ThemeSettings` component for user selection
- **Integration:** Theme config flows to `LoadingElement` and `ElementTab` components

### Data Flow
```
localStorage
    ↓
useMotifTheme() hook
    ↓
    ├→ LoadingElement (renders tiles)
    ├→ ElementTab (renders tab buttons)
    └→ App state (controls current theme)
```

---

## 2. Test Plan by Category

### 2.1 Theme Registry Tests
**File:** `src/lib/__tests__/themes.test.ts`
**Purpose:** Validate all 5 themes are properly registered with correct metadata

#### Test Cases (8 tests)

```typescript
describe("Theme Registry", () => {

  describe("Theme Registration", () => {
    it("all 5 themes are registered", ...)
    it("each theme has exactly 7 tab configurations", ...)
    it("all themes have required fields: symbol, number, name, color, activeColor, glow", ...)
  });

  describe("Theme Data Integrity", () => {
    it("no duplicate symbols within a theme", ...)
    it("all symbol values are 2 characters", ...)
    it("all numbers are positive integers in valid ranges", ...)
    it("all Tailwind color classes are valid", ...)
    it("all animation class names are defined in tailwind config", ...)
  });
});
```

#### Assertions
- `THEMES` object has keys: `'periodic-table'`, `'hospitality'`, `'space'`, `'f1-racing'`, `'zen-garden'`
- `THEMES[id].tabs` is an array with exactly 7 items
- Each tab has: `symbol` (string), `number` (number), `name` (string), `color` (string), `activeColor` (string), `glow` (string)
- `symbol` values are unique within each theme
- All `color` and `activeColor` strings contain valid Tailwind classes (`bg-*`, `text-*`, `border-*`)
- All `glow` values are valid shadow classes or empty string

#### Mocking Strategy
- No mocks needed — pure data validation

#### Estimated Lines of Code
- ~120 lines (test code)

---

### 2.2 useMotifTheme Hook Tests
**File:** `src/hooks/__tests__/useMotifTheme.test.ts`
**Purpose:** Validate hook state management, persistence, and fallback behavior

#### Test Cases (9 tests)

```typescript
describe("useMotifTheme Hook", () => {

  describe("Initialization", () => {
    it("default theme is 'periodic-table'", ...)
    it("loads theme from localStorage on mount", ...)
    it("falls back to default if invalid theme ID in localStorage", ...)
    it("falls back to default if no theme in localStorage", ...)
  });

  describe("Theme Selection", () => {
    it("setMotifTheme updates state immediately", ...)
    it("setMotifTheme persists to localStorage", ...)
    it("changing theme returns correct theme config", ...)
  });

  describe("Configuration Retrieval", () => {
    it("returns correct tab configs for selected theme", ...)
    it("tab configs include all required fields", ...)
  });
});
```

#### Assertions
- Initial state matches localStorage, or defaults to `'periodic-table'`
- Calling `setMotifTheme('hospitality')` updates returned `motifTheme` state
- `localStorage.getItem('ds-motif-theme')` reflects the change
- Returned `config` object has shape `{ tabs: [...], name: string, description?: string }`
- Each tab in `config.tabs` has all required fields
- Caching/memoization prevents unnecessary recalculations

#### Mocking Strategy
- Mock `localStorage` using `vi.stubGlobal()`
- No query/API mocks needed

#### Estimated Lines of Code
- ~140 lines (test code)

---

### 2.3 ThemeSettings Component Tests
**File:** `src/components/__tests__/ThemeSettings.test.tsx`
**Purpose:** Validate UI for theme selection

#### Test Cases (8 tests)

```typescript
describe("ThemeSettings Component", () => {

  describe("Rendering", () => {
    it("renders all 5 theme options", ...)
    it("each theme option displays name and icon/preview", ...)
    it("current theme option is visually highlighted", ...)
  });

  describe("Interaction", () => {
    it("clicking theme option calls onSelect callback", ...)
    it("onSelect receives correct theme ID", ...)
    it("selected theme is reflected visually after click", ...)
  });

  describe("Preview", () => {
    it("preview shows first tab tile of selected theme", ...)
    it("preview tile has correct symbol/number/name", ...)
  });
});
```

#### Assertions
- Renders 5 buttons (one per theme) with aria-labels
- Button labels/descriptions match theme names
- Current/selected theme button has `aria-pressed="true"` or `aria-current="true"`
- Preview tile has correct `symbol`, `number`, `name` from first tab of selected theme
- Preview tile classes include theme's `color` and `glow`
- Clicking non-selected theme calls `onSelect` with theme ID

#### Mocking Strategy
- Mock `useMotifTheme` hook to return controlled theme state
- No API/query mocks needed

#### Estimated Lines of Code
- ~140 lines (test code)

---

### 2.4 LoadingElement Component Updates
**File:** `src/components/__tests__/LoadingElement.test.tsx` (new file)
**Purpose:** Validate LoadingElement works with all theme × color mode combinations

#### Test Cases (16 tests)

```typescript
describe("LoadingElement with Themes", () => {

  describe("Theme Integration (5 themes)", () => {
    it("renders symbol from theme config", ...)
    it("renders number from theme config", ...)
    it("applies color classes from theme", ...)
    it("applies activeColor classes from theme", ...)
    it("applies glow shadow from theme", ...)
  });

  describe("Color Modes (3 modes)", () => {
    it("renders correctly in light mode", ...)
    it("renders correctly in dark mode", ...)
    it("renders correctly in midnight mode", ...)
  });

  describe("Overlay Mode", () => {
    it("overlay mode renders with backdrop blur", ...)
    it("overlay mode positions element centered", ...)
    it("overlay mode has proper z-index", ...)
  });

  describe("Size Variants", () => {
    it("sm size applies correct padding", ...)
    it("md size applies correct padding", ...)
    it("md size applies correct font sizes", ...)
  });

  describe("Custom Message", () => {
    it("displays custom message when provided", ...)
    it("omits message element when not provided", ...)
  });
});
```

#### Assertions
- `config.symbol` is rendered in the tile
- `config.number` is rendered with correct opacity
- Class names include `config.color`, `config.activeColor`, `config.glow`
- `className` includes correct size-specific classes
- `overlay={true}` renders with `absolute inset-0 z-30` and `bg-background/70 backdrop-blur`
- `message` text is rendered when provided, omitted when not

#### Mocking Strategy
- Mock `useTheme()` to return specific color mode
- Pass different `config` objects (tabs from different themes)
- No API mocks needed

#### Estimated Lines of Code
- ~180 lines (test code)

---

### 2.5 ElementTab Component Updates
**File:** `src/components/__tests__/ElementTab.test.tsx` (new file)
**Purpose:** Validate ElementTab works with all themes

#### Test Cases (10 tests)

```typescript
describe("ElementTab with Themes", () => {

  describe("Theme Rendering", () => {
    it("renders symbol from theme config", ...)
    it("renders number from theme config", ...)
    it("renders name from theme config with uppercase", ...)
    it("applies color classes when inactive", ...)
    it("applies activeColor classes when active", ...)
    it("applies glow when active", ...)
  });

  describe("Active State", () => {
    it("aria-selected='true' when active", ...)
    it("aria-selected='false' when inactive", ...)
    it("scale-105 class only when active", ...)
    it("bottom indicator only when active", ...)
  });

  describe("Interaction", () => {
    it("onClick fires callback when clicked", ...)
    it("has proper aria-label from config.name", ...)
    it("has role='tab' attribute", ...)
  });
});
```

#### Assertions
- `config.symbol`, `config.number`, `config.name` are rendered
- Inactive tab has `config.color` classes, `hover:scale-105`
- Active tab has `config.activeColor`, `config.glow`, `scale-105`
- Active tab shows bottom indicator (pseudo-element or element)
- `aria-selected` attribute matches `isActive` prop
- `onClick` callback is called on button click

#### Mocking Strategy
- Mock `useTheme()` to return specific color mode
- Pass different `config` objects (tabs from different themes)
- No API mocks needed

#### Estimated Lines of Code
- ~160 lines (test code)

---

### 2.6 Integration Tests
**File:** `src/__tests__/theme-integration.test.tsx`
**Purpose:** Validate themes work across the entire app

#### Test Cases (6 tests)

```typescript
describe("Theme System Integration", () => {

  describe("Theme Propagation", () => {
    it("changing theme in useMotifTheme updates all children", ...)
    it("LoadingElement reflects theme change immediately", ...)
    it("ElementTab reflects theme change immediately", ...)
  });

  describe("Persistence", () => {
    it("theme persists in localStorage across re-mounts", ...)
    it("selecting new theme updates localStorage", ...)
    it("localStorage value is used on app reload (simulated)", ...)
  });

  describe("Fallback Behavior", () => {
    it("invalid theme ID falls back to periodic-table", ...)
    it("missing localStorage falls back to periodic-table", ...)
  });
});
```

#### Assertions
- When `setMotifTheme('space')` is called, both `LoadingElement` and `ElementTab` show space theme config
- `localStorage.getItem('ds-motif-theme')` returns `'space'`
- Re-mounting component with `'space'` in localStorage initializes to `'space'`
- Invalid theme ID (e.g., `'invalid-theme'`) falls back to `'periodic-table'`
- Empty localStorage falls back to `'periodic-table'`

#### Mocking Strategy
- Mock `localStorage`
- Use `renderHook` from RTL for testing hooks in isolation
- Wrap component tests with mock provider

#### Estimated Lines of Code
- ~120 lines (test code)

---

### 2.7 Snapshot Tests (Visual Regression)
**File:** `src/__tests__/theme-snapshots.test.tsx`
**Purpose:** Detect unintended visual changes across theme/color combinations

#### Test Cases (15 snapshots)

```typescript
describe("Theme Snapshots (Visual Regression)", () => {

  it("periodic-table theme in light mode matches snapshot", ...)
  it("periodic-table theme in dark mode matches snapshot", ...)
  it("periodic-table theme in midnight mode matches snapshot", ...)

  it("hospitality theme in light mode matches snapshot", ...)
  it("hospitality theme in dark mode matches snapshot", ...)
  it("hospitality theme in midnight mode matches snapshot", ...)

  // ... repeat for space, f1-racing, zen-garden
});
```

#### Assertions
- Snapshot of `LoadingElement` for each theme/color combination
- Snapshot of single `ElementTab` for each theme/color combination

#### Mocking Strategy
- Mock `useTheme()` to return each color mode
- Pass each theme's first tab config

#### Estimated Lines of Code
- ~80 lines (test code) + 30 snapshots

---

### 2.8 Accessibility Tests
**File:** `src/__tests__/theme-a11y.test.tsx`
**Purpose:** Ensure WCAG AA compliance across all theme/color combinations

#### Test Cases (8 tests)

```typescript
describe("Theme Accessibility (WCAG AA)", () => {

  describe("Color Contrast", () => {
    it("all 5 themes have AA contrast in light mode", ...)
    it("all 5 themes have AA contrast in dark mode", ...)
    it("all 5 themes have AA contrast in midnight mode", ...)
  });

  describe("Focus Management", () => {
    it("ElementTab focus outline is visible in all themes", ...)
    it("focus indicator has sufficient contrast in all themes", ...)
  });

  describe("Screen Reader", () => {
    it("LoadingElement has aria-label or aria-labelledby", ...)
    it("theme change announces updated config via aria-live", ...)
    it("ThemeSettings buttons have accessible names", ...)
  });

  describe("Keyboard Navigation", () => {
    it("ElementTab is focusable and clickable via keyboard", ...)
  });
});
```

#### Assertions
- For each theme/color combo, use [axe-core](https://github.com/dequelabs/axe-core) or similar to detect contrast violations
- Focus outline contrast is ≥ 3:1 ratio
- `LoadingElement` has `aria-label` describing what's loading
- `ThemeSettings` buttons have descriptive `aria-label`
- `ElementTab` is keyboard accessible (role="tab", tabIndex, aria-selected)

#### Mocking Strategy
- Mock `useTheme()` for each color mode
- Use axe-core for automated testing
- Manual validation for focus indicators

#### Estimated Lines of Code
- ~140 lines (test code)

---

### 2.9 Performance Tests
**File:** `src/__tests__/theme-performance.test.ts`
**Purpose:** Ensure theme system doesn't cause performance regressions

#### Test Cases (4 tests)

```typescript
describe("Theme Performance", () => {

  it("theme change doesn't cause unnecessary re-renders of unrelated components", ...)
  it("useMotifTheme hook memoizes theme config correctly", ...)
  it("theme config is not recreated on every render", ...)
  it("animated classes don't cause layout thrash", ...)
});
```

#### Assertions
- When `setMotifTheme` is called, a sibling component without theme dependencies doesn't re-render
- `useMotifTheme()` returns the same object reference if theme hasn't changed (memoization)
- No excessive DOM reads/writes during animation
- Theme switching completes in < 50ms (JS execution)

#### Mocking Strategy
- Use React's profiler API or spy on component renders
- Mock `useCallback` to track call counts

#### Estimated Lines of Code
- ~100 lines (test code)

---

## 3. Test File Structure

### Directory Layout
```
mvp/demand/frontend/src/
├── lib/
│   ├── themes.ts (NEW)
│   └── __tests__/
│       └── themes.test.ts (8 tests)
├── hooks/
│   ├── useMotifTheme.ts (NEW)
│   └── __tests__/
│       └── useMotifTheme.test.ts (9 tests)
├── components/
│   ├── ThemeSettings.tsx (NEW)
│   ├── LoadingElement.tsx (UPDATED)
│   ├── ElementTab.tsx (UPDATED)
│   └── __tests__/
│       ├── ThemeSettings.test.tsx (NEW, 8 tests)
│       ├── LoadingElement.test.tsx (NEW, 16 tests)
│       ├── ElementTab.test.tsx (NEW, 10 tests)
│       └── Skeleton.test.tsx (EXISTING, no changes)
├── __tests__/
│   ├── theme-integration.test.tsx (NEW, 6 tests)
│   ├── theme-snapshots.test.tsx (NEW, 15 snapshots)
│   ├── theme-a11y.test.tsx (NEW, 8 tests)
│   ├── theme-performance.test.ts (NEW, 4 tests)
│   └── setup.ts (EXISTING, no changes)
```

### Test Dependencies
```json
{
  "devDependencies": {
    "vitest": "latest",
    "@testing-library/react": "latest",
    "@testing-library/user-event": "latest",
    "axe-core": "latest",
    "@axe-core/react": "latest"
  }
}
```

---

## 4. Mocking Strategy

### localStorage Mocking
```typescript
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
  };
})();

vi.stubGlobal('localStorage', localStorageMock);
```

### useTheme Hook Mocking
```typescript
vi.mock('@/hooks/useTheme', () => ({
  useTheme: vi.fn(() => ({
    theme: 'light',
    setTheme: vi.fn(),
    trendColors: [],
    chartColors: {},
  })),
}));
```

### useMotifTheme Hook Mocking
```typescript
vi.mock('@/hooks/useMotifTheme', () => ({
  useMotifTheme: vi.fn(() => ({
    motifTheme: 'periodic-table',
    setMotifTheme: vi.fn(),
    config: THEMES['periodic-table'],
  })),
}));
```

---

## 5. Test Execution Plan

### Run All Theme Tests
```bash
make test-theme-all
```

### Run by Category
```bash
make test-theme-registry      # Theme registry validation
make test-theme-hooks         # Hook tests
make test-theme-components    # Component tests
make test-theme-integration   # Integration tests
make test-theme-a11y          # Accessibility tests
make test-theme-perf          # Performance tests
```

### Continuous Integration
- Run `make test-all` after every change to catch regressions
- All 52 new tests must pass before feature is considered complete
- Snapshots must be reviewed and approved (no auto-update in CI)

---

## 6. Test Data / Fixtures

### Theme Fixtures
```typescript
// src/__tests__/fixtures/themes.ts
export const MOCK_THEMES = {
  'periodic-table': {
    tabs: [
      { symbol: 'Dx', number: 1, name: 'Explorer', color: '...', activeColor: '...', glow: '...' },
      // ... 6 more tabs
    ],
  },
  'hospitality': { ... },
  'space': { ... },
  'f1-racing': { ... },
  'zen-garden': { ... },
};
```

### Component Props Fixtures
```typescript
// src/__tests__/fixtures/components.ts
export const MOCK_ELEMENT_CONFIG = {
  symbol: 'Dx',
  number: 1,
  name: 'Explorer',
  color: 'bg-pink-50/90 text-pink-800 border-pink-200/60',
  activeColor: 'bg-pink-100 text-pink-950 border-pink-300',
  glow: 'shadow-[0_0_12px_rgba(236,72,153,0.3)]',
};
```

---

## 7. Success Criteria

### Coverage Requirements
- **Statements:** ≥ 85% for new theme files
- **Branches:** ≥ 80% (especially theme fallback paths)
- **Functions:** 100% for exported functions
- **Lines:** ≥ 85%

### Quality Gates
- All 52 tests pass
- No linter warnings or type errors
- Snapshot diffs reviewed and approved
- Accessibility audit (axe-core) passes with 0 violations
- Performance benchmarks show no regressions

### Documentation
- All test files have clear describe/it names
- Complex assertions include comments
- Fixtures are documented with JSDoc
- README updated with new test structure

---

## 8. Rollout Plan

### Phase 1: Theme Library & Hook (Week 1)
- Implement `src/lib/themes.ts` with 5 themes
- Implement `src/hooks/useMotifTheme.ts` hook
- Create theme registry tests (8)
- Create hook tests (9)
- **Target:** 17 passing tests

### Phase 2: UI Components (Week 2)
- Implement `src/components/ThemeSettings.tsx`
- Update `LoadingElement.tsx` to accept theme config
- Update `ElementTab.tsx` to accept theme config
- Create component tests (34)
- **Target:** 51 passing tests

### Phase 3: Integration & Polish (Week 3)
- Create integration tests (6)
- Create snapshot tests (15)
- Create a11y tests (8)
- Create performance tests (4)
- Fix any failing tests
- **Target:** 84 passing tests (52 new + 32 existing)

### Phase 4: CI/CD & Docs (Week 4)
- Update Makefile with theme test targets
- Add theme testing to CI pipeline
- Update CLAUDE.md with theme system info
- Update mvp/demand/docs/ARCHITECTURE.md
- Final validation run

---

## 9. Dependencies & Infrastructure

### Required Packages (Already in package.json)
- `vitest` — test runner
- `@testing-library/react` — component testing
- `@testing-library/user-event` — user interaction simulation

### Additional Packages to Add
```bash
npm install --save-dev \
  @axe-core/react \
  axe-core
```

### Tailwind Config Update
Ensure `animate-pulse-glow` is defined in `tailwind.config.ts`:
```typescript
export default {
  theme: {
    extend: {
      animation: {
        'pulse-glow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite, glow 2s ease-in-out infinite',
      },
    },
  },
};
```

---

## 10. Example Test Structure

### Registry Test Example
```typescript
import { describe, it, expect } from 'vitest';
import { THEMES, THEME_IDS } from '@/lib/themes';

describe('Theme Registry', () => {
  it('all 5 themes are registered', () => {
    expect(THEME_IDS).toEqual([
      'periodic-table',
      'hospitality',
      'space',
      'f1-racing',
      'zen-garden',
    ]);
  });

  it('each theme has exactly 7 tab configurations', () => {
    THEME_IDS.forEach((id) => {
      expect(THEMES[id].tabs).toHaveLength(7);
    });
  });

  it('all required fields are present', () => {
    THEME_IDS.forEach((id) => {
      THEMES[id].tabs.forEach((tab) => {
        expect(tab).toHaveProperty('symbol');
        expect(tab).toHaveProperty('number');
        expect(tab).toHaveProperty('name');
        expect(tab).toHaveProperty('color');
        expect(tab).toHaveProperty('activeColor');
        expect(tab).toHaveProperty('glow');
      });
    });
  });
});
```

### Hook Test Example
```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useMotifTheme } from '@/hooks/useMotifTheme';

describe('useMotifTheme Hook', () => {
  beforeEach(() => {
    const localStorageMock = { /* ... */ };
    vi.stubGlobal('localStorage', localStorageMock);
  });

  it('default theme is periodic-table', () => {
    const { result } = renderHook(() => useMotifTheme());
    expect(result.current.motifTheme).toBe('periodic-table');
  });

  it('setMotifTheme updates state and persists', () => {
    const { result } = renderHook(() => useMotifTheme());

    act(() => {
      result.current.setMotifTheme('space');
    });

    expect(result.current.motifTheme).toBe('space');
    expect(localStorage.getItem('ds-motif-theme')).toBe('space');
  });
});
```

### Component Test Example
```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LoadingElement } from '@/components/LoadingElement';

vi.mock('@/hooks/useTheme', () => ({
  useTheme: () => ({ theme: 'light' }),
}));

describe('LoadingElement', () => {
  it('renders symbol from theme config', () => {
    const config = {
      symbol: 'Dx',
      number: 1,
      name: 'Explorer',
      color: '...',
      activeColor: '...',
      glow: '...',
    };

    render(<LoadingElement config={config} />);
    expect(screen.getByText('Dx')).toBeInTheDocument();
  });
});
```

---

## 11. Appendix: Theme Definition Schema

Each theme object has this structure:

```typescript
type MotifTheme = {
  id: string; // 'periodic-table' | 'hospitality' | 'space' | 'f1-racing' | 'zen-garden'
  name: string; // Display name: 'Periodic Table', 'Hospitality & Spirits', etc.
  description?: string; // Optional tagline
  animationClass: string; // Tailwind animation class, e.g., 'animate-pulse-glow'
  tabs: ElementConfig[]; // Exactly 7 tab configurations
};

type ElementConfig = {
  symbol: string; // 2-char code, e.g., 'Dx', 'Ac'
  number: number; // Atomic number equivalent
  name: string; // Tab label, e.g., 'Explorer'
  color: string; // Inactive state Tailwind classes
  activeColor: string; // Active state Tailwind classes
  glow: string; // Shadow/glow effect for animation
};
```

---

## 12. Summary: Test Statistics

| Category | File | Tests | Status |
|----------|------|-------|--------|
| Registry | `src/lib/__tests__/themes.test.ts` | 8 | New |
| Hook | `src/hooks/__tests__/useMotifTheme.test.ts` | 9 | New |
| Components | `src/components/__tests__/ThemeSettings.test.tsx` | 8 | New |
| Components | `src/components/__tests__/LoadingElement.test.tsx` | 16 | New |
| Components | `src/components/__tests__/ElementTab.test.tsx` | 10 | New |
| Integration | `src/__tests__/theme-integration.test.tsx` | 6 | New |
| Snapshots | `src/__tests__/theme-snapshots.test.tsx` | 15 | New |
| Accessibility | `src/__tests__/theme-a11y.test.tsx` | 8 | New |
| Performance | `src/__tests__/theme-performance.test.ts` | 4 | New |
| **TOTAL** | | **84** | **New** |

---

## Document Change Log

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-02-24 | Initial design specification |



---

## Examples

### Example: Vitest dark mode CSS variable test

```typescript
// src/hooks/__tests__/useTheme.test.ts
import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { useTheme } from '@/hooks/useTheme'

describe('useTheme dark mode', () => {
  beforeEach(() => {
    document.documentElement.className = ''
    localStorage.clear()
  })

  it('applies dark class to documentElement', () => {
    const { result } = renderHook(() => useTheme())
    act(() => result.current.setColorMode('dark'))
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('persists color mode in localStorage', () => {
    const { result } = renderHook(() => useTheme())
    act(() => result.current.setColorMode('dark'))
    expect(localStorage.getItem('ds-color-mode')).toBe('dark')
  })
})
```

### Example: CSS variable assertion

```typescript
it('sets correct CSS variable for dark background', () => {
  const { result } = renderHook(() => useTheme())
  act(() => result.current.setColorMode('dark'))
  const bgPrimary = document.documentElement.style.getPropertyValue('--bg-primary')
  expect(bgPrimary).toBe('#0f172a')
})
```

### Example: Run theme tests

```bash
make ui-test -- --reporter verbose src/hooks/__tests__/useTheme.test.ts
# ✓ useTheme dark mode > applies dark class (2ms)
# ✓ useTheme dark mode > persists in localStorage (1ms)
# ✓ useTheme dark mode > sets correct CSS variables (3ms)
```

### Example: Playwright visual regression (when implemented)

```typescript
// tests/visual/dark-mode.spec.ts
import { test, expect } from '@playwright/test'

test('dark mode dashboard matches baseline', async ({ page }) => {
  await page.goto('http://localhost:5173?colorMode=dark')
  await page.waitForSelector('[data-testid="dashboard"]')
  await expect(page).toHaveScreenshot('dark-mode-dashboard.png', { threshold: 0.01 })
})
```
