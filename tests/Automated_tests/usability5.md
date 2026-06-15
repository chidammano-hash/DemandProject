# Usability Review — Cycle 5

App: Supply Chain Command Center (React + Vite, branded "Vrantis"). Evidence: `tests/Automated_tests/cycle5/` (capture-digest.md, capture-dump.json, screens/*.png) + read-only code inspection of `frontend/src/tabs` and `src/components`.

Cross-checked against `LEDGER.md`, `usability1.md`–`usability4.md`. **Confirmed-fixed and NOT re-reported:**
- **U4.1** (Demand-History bare grays) — fixed; `WorkbenchPanel`/`MatrixPanel`/`DecompositionPanel`/`ComparisonPanel` now use `text-muted-foreground`.
- **U4.2** (ForecastTrendChart tooltip) — fixed; `buildForecastTrendOption` thousands-separates.
- **U4.4** (AI-FVA zero-yield run) — fixed; digest lines 2887/2890 now show "No recommendations — ran cleanly, nothing actionable for this sample."
- **U3.1/U3.2/U3.3/U3.5** (CA dark toggles, total-demand format, aria-pressed, Item breadcrumb) — verified live (digest 3465 "23.0M cases", 3164 "Item 13806 — MAKERS MARK BBN 90 SQUARE", heatmap toggle `aria-pressed` at `CustomerHeatmap.tsx:150/162`).
- **U2.6 retired-tab redirect** — `useUrlState.ts` `TAB_REDIRECTS` resolves `aiPlanner`/`controlTower`→`commandCenter` (why those digest sections show Command Center content; working as intended).
- **U4.3 rec #1 (lazy gating)** — confirmed satisfied: CA heavy panels ARE `LazyPanel`-wrapped (`CustomerAnalyticsTab.tsx:316-378`). The residual is the *unbounded* fallback only (see U5.3).

The highest-value NEW finding this cycle is that the recurring dark-theme legibility regression has a **single systemic root cause** the per-tab fixes (U3.1/U4.1) keep papering over: there is no shared themed severity/status badge, so 30+ files hand-roll `bg-{color}-100 text-{color}-700` with no `dark:` variant. Fixing the shared primitive retires the whole class.

---

## U5.1 — No shared themed severity/status badge: 30+ tabs hand-roll `bg-{color}-100 text-{color}-700` chips with no `dark:` variant (systemic root cause of the recurring dark-theme regression)

- **Category:** accessibility
- **Severity:** P1
- **Evidence:** `components/ui/badge.tsx` defines only `default`/`secondary`/`outline` variants — **no semantic `critical`/`high`/`warning`/`info`/`success` variant**. So every tab that needs a colored status pill inlines a literal tint map. `grep` for `bg-red-100|bg-green-100|bg-amber-100|bg-orange-100|bg-blue-100|bg-yellow-100` hits **30+ files** under `tabs/`, almost all without a `dark:` companion: e.g. `SopTab.tsx:34-39` (`SEVERITY_COLORS = "bg-red-100 text-red-700"…`) and `:208` (cycle-stage chip `bg-green-100 text-green-700`/`bg-gray-100 text-gray-600`/`bg-blue-100 text-blue-700`), `ai-planner/aiPlannerShared.ts`, `inv-planning/OverrideQueuePanel.tsx`, `clusters/ClusterOverviewPanel.tsx`, `data-quality/dqShared.ts`, `storyboard/storyboardShared.ts`, `model-tuning/_helpers.ts`. In Dark theme these render a light pastel tint with dark text on a near-black surface — the tint barely separates from the page and the cycle-3/cycle-4 fixes only addressed `text-gray-*` *muted labels*, never these *colored status badges*. Captures are Light, so this is code-evidenced (same basis prior dark findings used).
- **File:** `frontend/src/components/ui/badge.tsx` (missing semantic variants); representative call sites `frontend/src/tabs/SopTab.tsx:34-39,208`, `frontend/src/tabs/ai-planner/aiPlannerShared.ts`, `frontend/src/tabs/inv-planning/OverrideQueuePanel.tsx`, `frontend/src/tabs/clusters/ClusterOverviewPanel.tsx`, `frontend/src/tabs/data-quality/dqShared.ts`.
- **Recommendation:** Add semantic variants to `badgeVariants` (e.g. `critical`/`high`/`warning`/`info`/`success`) each carrying a Light **and** `dark:` tint pair (e.g. `bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-300`), and export a tiny `severityBadgeClass(sev)` helper from a shared module (mirror the existing `togglePillClass()` pattern). Migrate the duplicated `SEVERITY_COLORS`/status maps to it, starting with the highest-traffic surfaces (`SopTab`, `aiPlannerShared`, `OverrideQueuePanel`, `dqShared`). Add a source-guard test asserting no bare `bg-(red|green|amber|orange|blue|yellow)-100` without a `dark:` sibling survives in the migrated files.
- **Acceptance:** A `critical`/`warning`/`success` badge is legible (contrast ≥ AA) in Dark theme; the shared helper/variant exists and is used by `SopTab` + ≥3 other tabs; source-guard test green; visual check of S&OP cycle chips + DQ severity chips + Override-queue chips in Dark shows readable pills.

## U5.2 — Two adjacent nav items, "FVA & ROI" and "AI FVA Backtest", share the identical `BarChart3` icon and the "FVA" prefix — visually indistinguishable in the Demand section

- **Category:** information-architecture
- **Severity:** P2
- **Evidence:** `AppSidebar.tsx:46-47` — `{ key: "fva", label: "FVA & ROI", icon: BarChart3, section: "demand" }` immediately followed by `{ key: "aiPlannerFva", label: "AI FVA Backtest", icon: BarChart3, section: "demand" }`. Both render the **same** `BarChart3` glyph, both start with "FVA", and they sit on consecutive rows (`screens/sop.png` sidebar shows "FVA & ROI" / "AI FVA Backtest" stacked). When the sidebar is collapsed (`w-16`, icon-only) the two are literally identical — a planner cannot tell which FVA tool they're clicking. They are also genuinely different surfaces: "FVA & ROI" is the staged value-ladder/ROI view (`FVATab.tsx`), "AI FVA Backtest" is the walk-forward AI-vs-champion backtest run list (`AiPlannerFvaTab.tsx`).
- **File:** `frontend/src/components/AppSidebar.tsx:46-47`.
- **Recommendation:** Give the two items distinct icons (e.g. keep `BarChart3` for "FVA & ROI"; use a `FlaskConical`/`Beaker`/`Play`-style icon for the *backtest* run-runner to signal "experiment/run") and consider relabeling for scan-ability — e.g. "Forecast Value (FVA)" vs "AI Backtest". Distinct icons matter most in collapsed mode where labels are hidden.
- **Acceptance:** The two Demand nav rows render different icons (assert in an `AppSidebar` render test that `fva` and `aiPlannerFva` `NavItem.icon` references differ); collapsed-sidebar `title` tooltips disambiguate; visual check shows two recognizably different glyphs.

## U5.3 — Below-fold panels show an unbounded "Loading…" fallback with no slow-query hint and no error/retry (residual of U4.3 rec #2/#3)

- **Category:** performance
- **Severity:** P2
- **Evidence:** `screens/customerAnalytics.png` / digest lines 3586-3593 — a wall of **8** stacked "Loading…" placeholders under "Channel Mix". Code confirms the panels *are* lazily gated (`CustomerAnalyticsTab.tsx:316-378` wraps them in `LazyPanel` + `Suspense`), so U4.3's gating concern is resolved — but the fallbacks are bare unbounded spinners: `CustomerRanking.tsx:148`, `CustomerTreemap.tsx:131`, `DemandAtRisk.tsx:73`, `SegmentSparklines.tsx:121` each render `<div …>Loading...</div>` with **no timeout, no "large query" sublabel after N seconds, and no error/retry on failure**. `LazyPanel.tsx` itself only swaps `fallback`↔`children` on intersection; React-Query errors surface as a perpetual "Loading…" with no recovery affordance. A planner who scrolls down to a slow/failed panel sees an indefinitely-spinning page with no signal whether to wait or that something broke.
- **File:** `frontend/src/tabs/customer-analytics/CustomerRanking.tsx:148`, `CustomerTreemap.tsx:131`, `DemandAtRisk.tsx:73`, `SegmentSparklines.tsx:121` (and sibling lazy panels); fallback contract in `frontend/src/components/LazyPanel.tsx`.
- **Recommendation:** Replace the bare "Loading…" with a shared `PanelLoading` skeleton that (1) shows a "still loading — large query" sublabel after ~3s and (2) renders an error state with a Retry button when the panel's `useQuery` reports `isError` (wire `error`/`refetch` from the existing query hooks). Keep it one shared component so all heavy panels behave consistently.
- **Acceptance:** A panel exceeding ~3s shows a "large query" hint; a failed panel shows an inline error + working Retry (not a perpetual spinner); a render test drives a panel hook into `isError` and asserts the retry affordance renders.

## U5.4 — S&OP dual empty-state is ambiguous: left card says "Start one to kick off the monthly Demand Review" while the right pane simultaneously says "Select a cycle to view details" — but there is no cycle to select

- **Category:** usability
- **Severity:** P3
- **Evidence:** `screens/sop.png` / digest lines 2133-2138 — with **0 active cycles**, the CYCLES card reads "No active S&OP cycles yet. Start one to kick off…" + a "Start new S&OP cycle" button, yet the adjacent detail pane reads "Select a cycle to view details, gaps, and actions." The detail-pane instruction is unactionable in the zero-cycle state (nothing to select), so the two cards give contradictory guidance side-by-side. Below them, "Approved Plan → No approved plan rows found" and "Decision Log → No decisions logged yet" stack three more empty panels, making the first-run screen read as four "nothing here" boxes.
- **File:** `frontend/src/tabs/SopTab.tsx` (cycle-list vs detail-pane empty states; right-pane placeholder rendered regardless of cycle count).
- **Recommendation:** When `cycles.length === 0`, suppress or rephrase the detail-pane placeholder to point at the primary action ("Start a cycle to see its stages, gaps, and decisions here") so the two cards agree; optionally collapse the Approved-Plan/Decision-Log cards into a single muted "These populate once a cycle is approved" note on first run.
- **Acceptance:** In the zero-cycle state the detail pane no longer says "Select a cycle" (it either hides or points to "Start a cycle"); a render test with empty `cycles` asserts the contradictory "Select a cycle" copy is absent.

---

### Re-stated (still-open) prior items — NOT re-counted as new

- **F2.3 / U1.6** (P2, consistency) — DataQuality "Total Checks **166**" tile (`DataQualityTab.tsx`, `Σ domain.total`) vs "Check Catalog (**83**)" (digest 2905 vs 3042) on the same screen; per-domain rollup double-counts each physical check under its source domain and its referential-integrity domain. Open — needs denominator reconciliation. (Carried, 4th cycle.)
- **U3.6 / U2.5 / U1.7** (P2, simplification) — 7 tab files still > 600 LoC and **drifting up**: `CommandCenterTab.tsx` **941**, `DataQualityTab.tsx` 707, `InvPlanningTab.tsx` 705, `StoryboardTab.tsx` 672, `AggregateAnalysisTab.tsx` 668, `SettingsTab.tsx` 649, `AiPlannerFvaTab.tsx` **636** (was 610 last cycle — grew). CLAUDE.md: tabs MUST be < 600 lines. Mechanical splits, low correctness value. Open. (Carried.)
- **U1.3** (P2, consistency) — raw `fetch(` still bypasses `fetchJson` in 4 model-tuning panels (`EnhancedPromoteModal.tsx`, `ExperimentBuilder.tsx`, `LogViewer.tsx`, `EnhancedComparisonPanel.tsx`); pre-existing tsc errors make a clean slice risky. Open. (Carried.)
</content>
</invoke>
