# Cycle 5 — Fixes Applied

Date: 2026-06-14 · Engineer pass · strict TDD (red → green → refactor → live-verify)

The planner pass (testinput5.md) reported **0 new P0/P1** and only carried P2/P3
deferrals + honesty observations. The usability pass (usability5.md) reported one
**NEW P1** systemic root cause (U5.1) plus three NEW P2/P3 items (U5.2/U5.3/U5.4).
This cycle fixes the highest-value NEW items plus a carried denominator clarity item.

All frontend; changes go live via HMR. No backend/DB changes were required.

---

## U5.1 (P1) — No shared themed severity/status badge (systemic dark-theme root cause) — FIXED

- **What was wrong:** `components/ui/badge.tsx` had only `default`/`secondary`/`outline`
  variants — no semantic severity. 30+ tabs hand-rolled `bg-{color}-100 text-{color}-700`
  status chips with **no `dark:` companion**, so in Dark theme they rendered a pale
  pastel tint with dark text on near-black — barely separable from the page. The
  per-tab cycle-3/4 fixes only addressed muted *labels*, never these colored *status pills*.
- **Fix:**
  - `frontend/src/lib/severityBadge.ts` (NEW) — `severityBadgeClass(sev)` helper emitting a
    Light + `dark:` tint pair for `critical/high/medium/low/info/warning/success/neutral`;
    unknown → neutral themed tint (never a bare bg-*-100). Mirrors the `togglePillClass()` pattern.
  - `frontend/src/components/ui/badge.tsx` — added semantic `critical/high/warning/info/success`
    `badgeVariants`, each with a `dark:` tint pair.
  - `frontend/src/tabs/SopTab.tsx` — `SEVERITY_COLORS`, the GapCard fallback,
    mitigation-status chip, and cycle-stage chip now route through `severityBadgeClass()`.
  - `frontend/src/tabs/inv-planning/PortfolioHealthPanel.tsx` — 4 hand-rolled
    `bg-*-100 text-*-800` health-tier/score maps (13 lines) replaced by
    `healthScoreBadge()` + `HEALTH_TIER_BADGE` (themed, dark-aware).
- **Red→green evidence:**
  - `severityBadge.test.ts::returns a dark: variant for every known severity` — RED:
    module did not exist (`Cannot find module '@/lib/severityBadge'`); GREEN: 4 tests pass.
  - `SopTab.test.tsx::cycle-stage chips carry a dark: theme variant (U5.1)` — RED:
    `expected '' to match /dark:/`; GREEN: `container.innerHTML` matches `/dark:bg-green-/`.
  - `severityBadgeMigration.test.ts::PortfolioHealthPanel has no bare bg-*-100 without dark:` —
    RED: 13 offender lines listed (250/352–355/391–394/440–443); GREEN: `[]` offenders.
- **Verification:** helper used by SopTab + PortfolioHealthPanel (≥3 surfaces incl. score
  pill, heatmap cells, tier filter, detail rows); source-guard green; no NEW tsc errors.
- **Acceptance met:** YES.

## U5.2 (P2) — "FVA & ROI" and "AI FVA Backtest" share the identical BarChart3 icon — FIXED

- **What was wrong:** `AppSidebar.tsx` NAV_ITEMS gave both consecutive Demand rows the
  same `BarChart3` glyph; in collapsed (icon-only) mode the two FVA tools are indistinguishable.
- **Fix:** `frontend/src/components/AppSidebar.tsx` — `aiPlannerFva` now uses `Beaker`
  (experiment/run-runner glyph), keeping `BarChart3` for "FVA & ROI".
- **Red→green evidence:** `AppSidebar.test.tsx::FVA & ROI and AI FVA Backtest use distinct
  icons (U5.2)` — RED: `expected [Function BarChart3] not to be [Function BarChart3]`;
  GREEN: `fva.icon !== aiFva.icon`.
- **Verification:** `Beaker` confirmed exported by lucide-react; render test green.
- **Acceptance met:** YES. (Also corrected a stale `NAV_ITEMS.length` assertion 16→18 in the
  same test file — the sidebar genuinely has 18 items; honest count, no assertion weakened.)

## U5.4 (P3) — S&OP zero-cycle state gives contradictory guidance — FIXED

- **What was wrong:** With 0 cycles the CYCLES card said "Start one to kick off…" while the
  adjacent detail pane said "Select a cycle to view details" — unactionable (nothing to select).
- **Fix:** `frontend/src/tabs/SopTab.tsx` — when `cycles.length === 0` the detail-pane
  placeholder reads "Start a cycle to see its stages, gaps, and decisions here." (points at the
  primary action); otherwise keeps "Select a cycle…".
- **Red→green evidence:** `SopTab.test.tsx::does not say 'Select a cycle' in the zero-cycle
  state (U5.4)` — RED: found "Select a cycle to view details" in the empty state; GREEN: null.
- **Verification:** render test with empty `cycles` asserts the contradictory copy is absent.
- **Acceptance met:** YES.

## F5.3 / U1.6 (P2, carried 4 cycles) — Data Quality "Total Checks 166" vs "Check Catalog (83)" — FIXED (relabel)

- **What was wrong:** The summary tile read "Total Checks 166" (Σ per-domain `total`, which
  double-counts each cross-domain referential check once per domain it touches) on the same
  screen as "Check Catalog (83)" (distinct definitions). 166 ≠ the 83 clickable rows → "which
  number is real?".
- **Fix:** `frontend/src/tabs/DataQualityTab.tsx` — tile relabeled "Total Checks" → "Check Runs"
  with a sublabel "across {checkList.length} definitions", so the per-domain run total and the
  distinct-definition denominator self-explain (the acceptance's "166 check-runs across 83
  definitions" framing). Display-layer only; no aggregation change, no data fabricated.
- **Red→green evidence:** `DataQualityTab.test.tsx::F5.3 the check-count tile reconciles the
  per-domain run total with the distinct catalog definitions` — RED: `Unable to find "Check Runs"`;
  GREEN: "Check Runs" present, "Total Checks" absent, "across 3 definitions" sublabel rendered.
  (Updated the pre-existing `renders summary KPI bar` test's "Total Checks" assertion to "Check
  Runs" — legitimate label rename, not a weakened assertion.)
- **Live-verify:** `curl /data-quality/checks` → 83 distinct definitions; `/data-quality/dashboard`
  sums per-domain totals (166). Tile now says "Check Runs … across 83 definitions" — both numbers
  reconcile honestly.
- **Acceptance met:** YES (the two surfaces are now self-explaining).

---

## Deferred

- **F5.1 / U5.3 (P2)** — CA below-fold "Loading…" bounded skeleton + slow-query hint + error/retry.
  Needs a shared `PanelLoading` component wired to each lazy panel's `useQuery` `error`/`refetch`
  across 4+ panels + `LazyPanel` fallback contract; larger than one safe TDD slice this cycle.
- **F5.2 (P2)** — cold `/customer-analytics/affinity` ~11s. Perf/MV task; resolves Redis-warm.
  Backing-MV + tiered-refresh work, out of scope for a display slice.
- **F5.6 (P3)** — Customer Demand Map auto-fit to FL + State dropdown listing only populated
  states. Leaflet `fitBounds` + locating the 60-state dropdown source; moderate scope/risk, cosmetic.
- **F5.4 (P3)** — BEER `<0%*` accuracy: working-as-intended (genuine source over-forecast); UI honest.
- **F5.5 / F1.1 (P3)** — FVA AI/Planner ladder rungs "Coming Soon": needs measured-vs-actual query
  path (carried, DEFERRED).
- **U3.6 / U2.5 / U1.7 (P2)** — 7 tabs > 600 LoC: mechanical splits, low correctness value.
- **U1.3 (P2)** — raw `fetch()` in 4 model-tuning panels: pre-existing tsc errors make a clean slice risky.

## Risk / notes

- All changes are frontend display-layer; backend untouched. No CLAUDE.md rule touched
  (no raw fetch, no `: any`, themes via tokens, charts unaffected).
- The `severityBadgeClass` helper is complementary to the existing `@/constants/severity`
  `SEVERITY_CONFIG` (severity levels): the new helper adds the **status** semantics
  (success/info/warning/neutral) that `SEVERITY_CONFIG` lacks, which is exactly the
  health-tier and cycle-stage use case. No duplication conflict.
- 1 pre-existing unrelated frontend failure (`DemandReferencePanel.test.tsx::shows KPI cards`,
  a "14d" date-dependent assertion) fails with my changes stashed too — not introduced here.
- 1 pre-existing tsc error (`PortfolioHealthPanel.tsx:129` `healthKeys.summary` filter typing)
  is unrelated to the severity migration; left as-is.
