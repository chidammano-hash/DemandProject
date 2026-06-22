# UI/UX Design Log

A loop-by-loop record of the **global design-system** hardening run, driven by the
design pod (1 orchestrator/product-designer + 2 `design-developer` subagents, live
Playwright critique). Each loop lists the **exact** changes: files, tokens, before→after
values, and rationale. See `.claude/design-pod/README.md` for the loop protocol.

- **App:** Supply Chain Command Center (React + Vite + TS, Tailwind + shadcn/ui)
- **Token flow:** `constants/themes/general.ts` → `useTheme.applyPalette()` (CSS vars) →
  `tailwind.config.ts` (`hsl(var(--token))`). Three modes: `light` / `soft` / `dark`.
- **Constraints honored every loop:** no inline hex in `tabs/`/`components/`; charts read
  theme from context; all three modes covered; tab files < 600 LoC; tests updated with code.

---

## Baseline audit (pre-loop) — 2026-06-20

Captured the Command Center live (light mode) + read the full token layer. The bones are
clean (shadcn + a real 3-mode theme system, semantic badges, tabular-nums KPIs) but the
product reads as **generic default-SaaS**. Root causes:

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Flat, single-step elevation** | `Card` = `shadow-sm`; no layered shadow scale; everything sits on one plane. |
| 2 | **Dead background depth** | `bgGradient*` tokens are no-ops — start/mid/end identical in all 3 modes → flat fill, no premium depth. |
| 3 | **Safe, undistinctive palette** | Standard `#2563EB` blue primary + pure-slate neutrals; minimal accent personality. |
| 4 | **Dated KPI tiles** | `KpiSummaryCard` leans on a heavy colored left-border; lots of dead internal whitespace; weak number hierarchy. |
| 5 | **No semantic color tokens** | success/warning/info exist only as ad-hoc Tailwind palette classes per-component, not first-class theme tokens. |
| 6 | **Typography unrefined** | heading/KPI `tracking: "0"`; no display tracking; no global font-feature tuning. |
| 7 | **Monotone borders** | one `border` weight everywhere; no surface layering. |

Baseline screenshot: `screenshots/baseline-light-commandcenter.png`.

---
<!-- Loop entries are appended below this line -->

## Loop 1 — "Depth, Elevation & Color Refinement" — 2026-06-20

**Intent:** kill the flat default-SaaS read. Introduce a real elevation scale, activate
background depth, promote semantic colors to first-class tokens, and modernize the KPI tiles.
**Critique → contract → 2 parallel devs (A: tokens, B: components) → live validate (light+dark).**
**Result:** ✅ shipped. Before: `screenshots/baseline-light-commandcenter.png`. After: `screenshots/loop1-light-commandcenter.png`, `screenshots/loop1-dark-commandcenter.png`. Dark mode especially gains premium depth.

### Dev A — Tokens & Theme Foundation
| File | Exact change |
|---|---|
| `tailwind.config.ts` | **+ boxShadow scale:** `card` = `0 1px 2px -1px rgb(15 23 42 /.06), 0 3px 8px -2px rgb(15 23 42 /.08)`; `card-hover` = `0 6px 16px -4px …/.12, 0 12px 28px -6px …/.10`; `elevated` = `0 12px 32px -8px …/.20, 0 4px 10px -3px …/.10`. **+ semantic colors:** `success`/`warning`/`info` (+`-foreground`) → `hsl(var(--…))`. |
| `types/theme.ts` | `ThemePalette` + 6 fields: `success, successForeground, warning, warningForeground, info, infoForeground`. |
| `constants/themes/general.ts` | Semantic tokens for all 3 modes — light `success 160 84% 30%` / `warning 30 90% 42%` / `info 214 90% 52%`; soft `160 68% 32%` / `30 82% 44%` / `214 74% 50%`; dark `158 64% 52%` / `38 92% 56%` / `213 94% 68%`. **Background depth activated** (was flat): light base `#FBFCFE→#F5F8FC→#EEF2F8` + glows `hsla(224,76%,50%,.06)` / `hsla(190,85%,45%,.045)`; soft `#FCFAF7→#F7F3EC→#F1EBE1`; dark `#0C1322→#0A0F1C→#070B14` + glows `hsla(217,91%,60%,.10)` / `hsla(190,90%,52%,.06)`. Typography: `headingTracking 0→-0.01em`, `kpiTracking 0→-0.02em`. |
| `hooks/useTheme.ts` | `applyPalette()` map + 6 rows mapping the new palette fields → `--success`/`--warning`/`--info` (+`-foreground`) CSS vars. |
| `index.css` | Same 6 semantic vars added to `:root` (light) + `.dark` fallbacks; bg-gradient fallbacks synced; `body` + `text-rendering: optimizeLegibility` + `font-feature-settings: "cv05","cv11","ss01"`. |

**Verify:** `tsc --noEmit` clean for owned files; theme vitest **32/32 pass** (3 files).

### Dev B — Components & Application
| File | Exact change (before → after) |
|---|---|
| `components/ui/card.tsx` | `rounded-lg … shadow-sm` → `rounded-xl … shadow-card transition-shadow duration-200`. |
| `components/KpiCard.tsx` | `rounded-lg shadow-sm hover:shadow-md transition-shadow` → `rounded-xl shadow-card hover:shadow-card-hover hover:-translate-y-0.5 transition-all duration-200`. |
| `tabs/command-center/KpiSummaryCard.tsx` | Dropped heavy `border-l-4 border-l-<color>` + `shadow-sm`; now `rounded-xl border border-border/70 shadow-card hover:shadow-card-hover hover:-translate-y-0.5 transition-all`. Icon chip `rounded-md p-1.5` → `rounded-lg p-2`. Value `+ tabular-nums`. Semantic color now carried by the tinted icon chip + value color, not a border. |
| `components/AppSidebar.tsx` | Active indicator bar `h-6 w-1` → `h-7 w-[3px]`; section label contrast `/50` → `/60`. |
| `components/ui/button.tsx` | Base `transition-colors` → `transition-all`; `default` variant `+ shadow-sm hover:shadow-card active:translate-y-px`. |

**Verify:** component vitest **59/59 pass** (3 files); no inline hex introduced; no new tsc errors.

**Orchestrator integration check:** `shadow-card`/`card-hover`/`elevated` + `success`/`warning`/`info` resolve in `tailwind.config.ts`; 6 semantic vars present in `applyPalette` + both `index.css` blocks. Live re-capture confirms depth + elevation render in light **and** dark.

**Loop-2 candidates (deferred):** refine primary blue + neutral ramp; KPI number/label hierarchy; sidebar brand block; chart grid/axis token polish; validate on a data-dense tab; focus/hover micro-interactions.

## Loop 2 — "Palette Identity & KPI Cohesion" — 2026-06-20

**Intent:** validated on a data-dense screen (Portfolio Analysis). Two issues: the hero
KPIs (`KpiCard`) still wore a **heavy full-height `border-l-4` colored bar** (inconsistent
with Loop-1's clean Command-Center tiles), and the primary blue read generic.
**Result:** ✅ shipped. Before: `screenshots/loop2-baseline-portfolio.png`. After: `screenshots/loop2-portfolio-after.png`.

### Dev A — Palette & Chart Refinement
| File | Exact change |
|---|---|
| `constants/themes/general.ts` | **Primary enriched** (richer brand blue), `primary`/`ring`/`accentForeground` together — light `224 76% 48%`→`227 80% 55%`; soft `215 70% 48%`→`220 68% 52%`; dark `213 94% 68%`→`221 90% 70%`. **Chart grid/axis** softened — light grid `#E2E8F0`→`#EAEEF4` / axis `#64748B`→`#7C8798`; soft grid `#DDD8D0`→`#E5E0D8` / axis `#8A8078`→`#9A9088`; dark grid `#1E293B`→`#243149` / axis `#A1A1AA`→`#9AA6B8`. |
| `constants/colors.ts` | `CHART_COLORS` grid/axis (+ `tooltip_border`) synced to the new grid values across light/soft/dark. Series/Okabe-Ito/heatmap arrays untouched (signed-off). |

**Verify:** `tsc` clean for owned files; vitest **24/24 pass**.

### Dev B — KPI Cohesion & Control Polish
| File | Exact change (before → after) |
|---|---|
| `components/KpiCard.tsx` | Severity treatment: dropped `SEVERITY_BORDER` (`border-l-4 border-l-[var(--kpi-best/warning)]` on the wrapper) → new `SEVERITY_ACCENT_BG`; wrapper now `relative overflow-hidden` + `pl-4` when accented, with an absolutely-positioned **inset accent bar** `<span aria-hidden absolute left-0 top-1/2 h-[55%] w-1 -translate-y-1/2 rounded-r-full bg-[var(--kpi-best|warning)]>`. Loop-1 elevation preserved. Keeps the good/bad scan cue, modern silhouette. |
| `components/ui/input.tsx` | `+ shadow-sm` resting + `transition-colors` (focus ring already correct). |
| `components/ui/select.tsx` | `+ transition-colors` + `focus-visible:ring-2 ring-ring ring-offset-2` (keyboard focus ring matching Input). |

**Verify:** vitest **40/40 pass**; KpiCard test assertions updated honestly (assert the inset `<span>` + bg token instead of the removed `border-l-4`); no inline hex; no new tsc errors.

**Orchestrator validation:** live re-capture on Portfolio confirms the heavy borders are gone (refined inset accents in their place), richer primary, subtler grid — consistent across the elevation language from Loop 1.

## Loop 3 — "Motion & Interaction Polish" — 2026-06-20

**Intent:** the app's interactions were mostly instant / `transition-colors`. A cohesive,
tasteful motion layer is what reads as premium. Establish easing + entrance tokens and apply.
**Result:** ✅ shipped.

**Investigation note:** the "off-brand floating island" in the bottom-right corner was traced
to the **React Query Devtools toggle** (`main.tsx`, gated by `import.meta.env.DEV`). It is a
dev-only tool already stripped from production builds — **not a product element**, so it was
left untouched (removing it would only cost a dev convenience). Verified before acting.

### Dev A — Motion Foundation
| File | Exact change |
|---|---|
| `tailwind.config.ts` | `transitionTimingFunction`: `smooth` `cubic-bezier(0.4,0,0.2,1)`, `spring` `cubic-bezier(0.34,1.56,0.64,1)`, `out-expo` `cubic-bezier(0.16,1,0.3,1)`. `animation` + `scale-in` (180ms out-expo), `overlay-in` (180ms ease-out). Matching `keyframes` added. |
| `index.css` | `@keyframes scale-in`/`overlay-in` + `.animate-scale-in`/`.animate-overlay-in`; `.animate-fade-in` easing `200ms ease-out` → `220ms cubic-bezier(0.16,1,0.3,1)`; **`prefers-reduced-motion` block extended** to disable both new animations. |

**Verify:** `tsc` clean for owned files; vitest 12/12.

### Dev B — Apply Motion
| File | Exact change |
|---|---|
| `components/ui/dialog.tsx` | Radix `data-[state=open]:animate-scale-in` on content, `data-[state=open]:animate-overlay-in` on overlay (premium modal entrance). |
| `components/ui/button.tsx` | Base CVA `+ ease-smooth` (all variants share one easing). |
| `components/EmptyState.tsx` | `+ animate-fade-in` entrance; steps card `+ shadow-card`. |
| `components/AppSidebar.tsx` | Nav button `transition-colors duration-150` → `transition-all duration-150 ease-smooth` (active bg + indicator animate smoothly). |

**Verify:** vitest 21/21 (EmptyState 4 + CommandCenter 17); no new tsc errors; no inline hex; reduced-motion respected.

## Loop 4 — "Overlay Consistency Sweep" — 2026-06-20

**Intent:** Loops 1-3 built a premium elevation + motion language, but floating overlays
(filter dropdowns, select menus, searchable selects) still used generic `shadow-lg` +
`transition-colors`, breaking cohesion with the dialog/card system. Align them.
**Result:** ✅ shipped. Validated live with the Brand filter open: `screenshots/loop4-light-portfolio-dropdown.png`.

**Scope decision:** swept all `shadow-lg` overlays. Aligned the three interactive menus below.
Left intentionally: `Toaster` (transient), `ui/switch` (thumb shadow), `CommandPalette`
(already has `shadow-2xl` + `zoom-in-95` entrance), `GlobalFilterBar` (unused — filters are local).

### Dev 1 — Filter Overlays
| File | Exact change |
|---|---|
| `tabs/aggregate-analysis/FilterDropdowns.tsx` | Both dropdown panels `shadow-lg` → `shadow-elevated animate-scale-in origin-top`; trigger + option buttons + `TimeGrainToggle` (`Mo`/`Qtr`) `transition-colors` → `transition-colors ease-smooth` (6 sites). |

**Verify:** vitest 3/3; no new tsc errors.

### Dev 2 — Select & Searchable Overlays
| File | Exact change |
|---|---|
| `components/ui/select.tsx` | `SelectContent` panel `shadow-lg` → `shadow-elevated animate-scale-in origin-top`; `SelectTrigger` `+ ease-smooth`. |
| `components/SearchableSelect.tsx` | Listbox `<ul>` panel `shadow-lg` → `shadow-elevated animate-scale-in origin-top`. |

**Verify:** vitest 5/5; no new tsc errors.

---

## Summary & close-out — 2026-06-20

**4 loops shipped. Stopped at diminishing returns** (per the run's stop condition): the system
is now cohesive across surfaces and all three modes; further changes would be cosmetic churn
with regression risk.

**Final integration gate:** all design-pod-touched test files together — **10 files, 103 tests, 0 fail**.
Validated live in **light, dark, and soft** modes.

**What changed, by layer**
- **Foundation** (`tailwind.config.ts`, `index.css`, `types/theme.ts`, `constants/themes/general.ts`,
  `constants/colors.ts`, `hooks/useTheme.ts`): elevation scale (`card`/`card-hover`/`elevated`);
  semantic tokens (`success`/`warning`/`info`); **activated background depth** (3 modes); enriched
  primary; refined chart grid/axis; motion scale (`smooth`/`spring`/`out-expo`, `scale-in`/`overlay-in`)
  with reduced-motion coverage; Inter stylistic + KPI tracking.
- **Components** (`ui/card`, `ui/button`, `ui/badge`*, `ui/dialog`, `ui/input`, `ui/select`, `KpiCard`,
  `AppSidebar`, `EmptyState`, `SearchableSelect`, `command-center/KpiSummaryCard`,
  `aggregate-analysis/FilterDropdowns`): `rounded-xl` + soft elevation + hover lift; modernized KPI tiles
  (chip + inset-accent, no dated borders); premium overlay elevation + scale-in entrance; smooth-eased
  interactions everywhere.

**Pod:** orchestrator (product designer, live Playwright critique) + 2 parallel `design-developer`
subagents per loop on disjoint file sets. Apparatus: `.claude/agents/design-developer.md`,
`.claude/design-pod/README.md`.

**Constraints honored throughout:** no inline hex in `tabs/`/`components/`; charts read theme from
context; all three modes covered; tab files < 600 LoC; tests updated alongside code (never weakened).

**Status:** all changes **uncommitted** — left for human review. Screenshots in `screenshots/`.

---

## Addendum — `interface-details` skill pass (2026-06-20)

Applied the newly-installed **`interface-details`** skill ([detail.design](https://detail.design),
registered as a user skill at `~/.claude/skills/interface-details`) to the shadcn primitives. These
are micro-interaction/a11y layers on top of the design-pod close-out above — additive, token-based,
reduced-motion-aware, no API breaks.

| File | Exact change | Detail rule |
|---|---|---|
| `components/ui/button.tsx` | `active:translate-y-px` moved to base class so every variant presses consistently (was default-only); `+ select-none`; `+ motion-reduce:transition-none motion-reduce:active:translate-y-0`. | *consistent active states everywhere* + *reduce animation for frequent use* |
| `components/ui/input.tsx` | `+ focus-visible:border-ring` so the border acknowledges focus alongside the ring. | form/motion craft |
| `components/ui/card.tsx` | New opt-in `interactive?: boolean` prop → `hover:-translate-y-0.5 hover:shadow-card-hover` (reduced-motion suppressed); transition widened to `box-shadow,transform`. Static cards unchanged. | *physical metaphors* + *consistency over novelty* |
| `components/ui/select.tsx` | `+ Escape`-to-close; `SelectTrigger` `+ aria-haspopup="listbox" aria-expanded`; `SelectContent` `+ role="listbox"`. | *respect the keyboard* + accessibility |

**Tests added:** `components/__tests__/card.test.tsx` (3), `components/__tests__/select.test.tsx` (4).
**Also fixed (pre-existing, unrelated):** `DemandReferencePanel.test.tsx` asserted `"1,200"` but
`kpiFmt` uses compact notation → corrected stale expectation to `"1.2K"`.

**Verify:** full frontend suite **162 files, 1161 tests, 0 fail**.

**Status:** uncommitted — left for human review alongside the design-pod changes above.
