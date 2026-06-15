# Cycle 5 — Senior Demand Planner Acceptance Findings

Date: 2026-06-14 · Reviewer: automated planner-acceptance pass · Cycle 5

## Verdict

The product is in **good shape**. I scanned all 14 captured tabs (Command Center,
Portfolio/Aggregate Accuracy, Demand History, Inventory Planning, S&OP, FVA & ROI,
AI FVA Backtest, Data Quality, Item Analysis, Explorer, Customer Analytics, Clusters,
and the retired controlTower/aiPlanner keys). Every tab loaded with `ok=true`, **zero
console errors**, and every endpoint I curled returned HTTP 200 with real, non-empty data.

The two things that *look* alarming in the digest turn out to be **intended behavior**,
not regressions:

- `?tab=controlTower` and `?tab=aiPlanner` render **Command Center** content. This is the
  deliberate IA consolidation (ledger U2.6 / U5.1): `TAB_REDIRECTS` in
  `frontend/src/hooks/useUrlState.ts` maps `controlTower → commandCenter` and
  `aiPlanner → commandCenter`, and neither key appears in the sidebar. The capture harness
  (`tests/Automated_tests/_harness/capture.mjs`) still lists those retired keys, so the
  digest shows duplicated Command Center text. **This is a harness/IA artifact, not a bug.**
- Data Quality tiles show **"Failed: 0 / Warnings: 26"** even though the dashboard rollup
  reports `failed=26`. This is the **intentional F7.2 severity-aware tile rule**
  (`DataQualityTab.tsx` line 140): `Failed = failed − info_fails − warning_fails`
  (26 − 6 − 20 = 0), and warning-severity fails roll into the amber Warnings tile
  (6 + 20 = 26). There are **0 critical-severity failures**, so "Failed: 0" is honest.

No new P0/P1 defects found. The findings below are carried/deferred items and two honest
data observations, none of which block a daily workflow.

---

## Findings

### F5.1 — Customer Analytics below-fold panels show "Loading…" in capture (carried U4.3) [P2]
- **Workflow:** Customer/channel drill-down for S&OP demand review.
- **Evidence:** `cycle5/screens/customerAnalytics.png` + digest: 8 trailing "Loading…"
  rows (Channel Mix, Affinity, Lifecycle, Order Patterns, Heatmap). All five underlying
  endpoints return 200 with real data when curled directly:
  `/customer-analytics/order-patterns` (0.01s), `/channel-mix` (0.56s), `/lifecycle`
  (0.79s), `/heatmap` (0.45s). So the data is healthy — the panels are `LazyPanel`
  (IntersectionObserver) gated and simply weren't scrolled into view by the capture
  harness at snapshot time.
- **Root cause:** Below-fold `LazyPanel` gating in the Customer Analytics tab; capture
  harness does not scroll. Carried from ledger U4.3 (DEFERRED).
- **Acceptance:** Either (a) the capture harness scrolls each tab to bottom before
  snapshot, or (b) LazyPanels show a bounded skeleton with a height reservation so a
  cold viewport never reads as a stuck spinner. A planner scrolling live sees them load.
- **Planner impact:** Low — live use renders the panels; this is mostly a capture artifact.

### F5.2 — `/customer-analytics/affinity` cold latency ~11.2s (carried F1.4 / F2.4) [P2]
- **Workflow:** Cross-sell / item-affinity review inside Customer Analytics.
- **Evidence:** `curl /customer-analytics/affinity` → 200 but **11.21s** cold (all sibling
  CA endpoints are sub-second). Resolves Redis-warm on subsequent hits.
- **Root cause:** Heavy affinity aggregation query without a backing MV / cold cache.
  Same item flagged in cycles 1–2 (F1.4 / F2.4), DEFERRED as a perf/MV task.
- **Acceptance:** First-load affinity panel returns < 3s cold, or is backed by an MV
  refreshed in the tiered refresh, or shows a determinate "computing affinity…" state.
- **Planner impact:** Low-moderate — an 11s blank on first open feels broken; a planner
  may navigate away before it resolves.

### F5.3 — Data Quality "Total Checks 166" vs "Check Catalog (83)" denominator (carried F2.3 / U1.6) [P2]
- **Workflow:** Morning data-quality sanity check before trusting forecasts/exceptions.
- **Evidence:** `cycle5/screens/dataQuality.png` + digest: summary tile "Total Checks 166"
  while the catalog header reads "Check Catalog (83)". Confirmed: `/data-quality/checks`
  returns **83** rows; `/data-quality/dashboard` rolls up `total=166` across 20 domain
  rows (each check counted once per domain-pair). 116+26+6+18 = 166 reconciles the tiles
  internally, but 166 ≠ the 83 catalog rows a planner can actually click into.
- **Root cause:** Dashboard aggregates per domain/domain-pair (double-counts cross-domain
  referential checks); the catalog lists distinct check definitions.
  `api/routers/.../data-quality` dashboard vs checks endpoints. Carried, DEFERRED.
- **Acceptance:** The two surfaces use one denominator, OR the tile is relabeled
  (e.g. "166 check-runs across 83 definitions") so 166 and 83 are self-explaining.
- **Planner impact:** Low — erodes trust slightly ("which number is real?"); no workflow block.

### F5.4 — BEER accuracy reads `<0%*` across all months (honest over-forecast signal) [P3]
- **Workflow:** Portfolio accuracy triage in the Aggregate Accuracy heatmap.
- **Evidence:** `cycle5/screens/aggregateAnalysis.png`: BEER row is solid red `<0%*` for
  Nov 25–Feb 26 while every other category is 60–80%. **Verified in DB:** BEER
  forecast = 27,294 vs actual = 5,077 over that window (≈5× over-forecast), so
  WAPE > 100% → accuracy < 0. The heatmap footnote ("Cells marked <0%* have actuals near
  zero on a tiny base — review WAPE") correctly explains it.
- **Root cause:** Genuine forecast-quality issue in source/external forecast for BEER
  (massive over-forecast on a low-volume category), **not** a UI bug. The product is
  surfacing it honestly.
- **Acceptance:** N/A for the UI (working as intended). If anything, a tooltip on the cell
  could show the F/A magnitudes (27.3K vs 5.1K) inline so a planner sees *why* it's <0%.
- **Planner impact:** Low for the tool (it's doing its job); high signal *about the data* —
  BEER forecasts need source-side correction.

### F5.5 — FVA AI/Planner ladder stages permanently "Coming Soon" while AI FVA Backtest produces recs (carried F1.1) [P3]
- **Workflow:** Forecast-value-added story for S&OP / leadership.
- **Evidence:** `cycle5/screens/fva.png`: ladder shows Naive 65.3% → External 71.2% →
  Champion / AI Adjusted / Planner Adjusted all "Coming Soon"; Total Interventions 0,
  Actual Impact $0. Meanwhile `aiPlannerFva.png` shows succeeded backtests producing
  69 / 143 / 12 / 6 recommendations.
- **Root cause:** Champion/AI/Planner rungs require a measured-vs-actual query path that
  isn't wired to the ladder yet (ledger F1.1, DEFERRED). The "Reserved stage" copy is
  honest, so this is a known gap, not a regression.
- **Acceptance:** When AI FVA backtest recs have measured outcomes, the AI Adjusted rung
  promotes from "Coming Soon" to a measured % (consistent with how External already does).
- **Planner impact:** Low — labeled honestly; the FVA story is just incomplete by design.

### F5.6 — Customer Demand Map geo-concentrated to a single FL cluster [P3]
- **Workflow:** Geographic demand view in Customer Analytics.
- **Evidence:** `cycle5/screens/customerAnalytics.png`: the Leaflet map renders one tiny
  marker cluster over Florida and is otherwise empty, despite "32,469 customers / 23.0M
  cases". **Verified:** `/customer-analytics/map?level=state` returns only **2 states** —
  FL (33,159 customers, 27.6M demand) and MA (1 customer, 0 demand). The 50+ entry State
  filter dropdown therefore offers states with no data.
- **Root cause:** Genuine data shape — this single-distributor dataset geocodes essentially
  all demand to FL. The map is **honest**, but the near-empty national view + a 60-state
  filter dropdown reads as "broken map" to a first-time planner.
- **Acceptance:** Map auto-fits/zooms to the populated extent (FL) on load, and the State
  filter only lists states present in the data (FL, MA) — or shows "2 of 60 states have
  demand" context so the sparse map is self-explaining.
- **Planner impact:** Low — cosmetic/clarity; data is correct.

---

## Not re-reported (confirmed intended or already-fixed-and-live)

- controlTower / aiPlanner → Command Center redirect: **intended** (U2.6 / U5.1 fix live).
- Data Quality "Failed 0 / Warnings 26": **intended** F7.2 severity-aware tiles (0 critical
  fails; warning-severity fails roll to amber). Honest.
- AI FVA Backtest failed/0-rec runs carry honest sub-notes ("No recommendations — ran
  cleanly…"): **U4.4 fix is live.**
- Demand History panels theme-legible, MoM sparklines with color cues, honest "No series
  selected" empty state: **U4.1 fix is live.**
- Item Analysis breadcrumb "Item 13806 — MAKERS MARK BBN 90 SQUARE": **U3.5 fix is live.**
- Customer Analytics concentration / order-ratio show "— no prior period" not "→ 0.0% MoM":
  **U3.4 fix is live.**
- Clusters Overview names the populated alternative ("310,558 SKUs assigned via Source…"):
  **F4.1 fix is live.**
- Inventory Planning banner $ at Risk vs Command Center Order Value at Risk carry
  distinguishing tooltips/basis labels: **F2.1 / F1.2 fixes live.**

## newActionableCount

**0** new unresolved P0/P1/P2 findings. All P2 items above (F5.1, F5.2, F5.3) are **carried
deferrals** from prior cycles, not new. F5.4–F5.6 are P3 honesty/polish observations.
