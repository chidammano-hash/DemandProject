# Usability Review — Cycle 2

Branch: `restructure`. Method: read cycle2 capture digest + dump + screenshots, then read-only code inspection of `frontend/src/tabs`, `frontend/src/components`, query/error layers, and live endpoint probes. NEW items first. Prior-cycle resolved items (LEDGER) not re-reported.

---

## U2.1 — Raw `{"detail":"Not Found"}` JSON leaks into user-facing error toast [P1]
- **Category:** usability
- **Evidence:** `cycle2/screens/dataQuality.png` shows a red toast reading literally `{"detail":"Not Found"}`. Digest records 3× `404 (Not Found)` console errors on the Data Quality tab. Live probe: `GET /data-quality/lineage/batches` → 404 (the lineage panel's batches call).
- **Root cause:** `frontend/src/api/queries/core.ts:104-111` — `fetchJson` throws `new Error(text)` where `text` is the raw response body `{"detail":"Not Found"}`. It never attaches `res.status`. The global handler (`main.tsx:38` → `formatApiError`) is designed to map status codes to friendly copy (404 → "That record could not be found.") but `extractStatus` (`formatApiError.ts:48-61`) finds no `.status` on the error and the message-digit heuristic fails on "Not Found" (no digits) → falls through to `sanitize(err.message)`, dumping the raw JSON. The entire formatApiError sanitization layer is bypassed for the common HTTP-error case.
- **File:** `frontend/src/api/queries/core.ts:104-111`; interacts with `frontend/src/lib/formatApiError.ts:48-74`
- **Recommendation:** In `fetchJson`, attach the status to the thrown error and parse FastAPI `{detail}` so the sanitizer can map it. e.g.:
  ```ts
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let detail: unknown = text;
    try { detail = JSON.parse(text); } catch { /* keep text */ }
    const err = new Error(typeof detail === "object" && detail && "detail" in detail
      ? String((detail as { detail: unknown }).detail) : text || `HTTP ${res.status}`);
    Object.assign(err, { status: res.status, detail });
    throw err;
  }
  ```
  Then `formatApiError` maps 404→friendly copy. Keep raw body off-screen.
- **Acceptance:** Add a case to `frontend/src/lib/__tests__/formatApiError.test.ts` (or core.test) asserting an error thrown by `fetchJson` for a 404 body `{"detail":"Not Found"}` produces `"That record could not be found."`, NOT the raw JSON. No toast in the app renders a leading `{`.

## U2.2 — "Demand History" sidebar tab cannot be deep-linked or survive a page refresh [P1]
- **Category:** information-architecture
- **Evidence:** Digest shows `?tab=demandHistory` rendered Command Center (`textLen 513`, identical Command-Center copy). `demandHistory` is a first-class sidebar item (`AppSidebar.tsx:50`, "Demand History", section Demand) with a real route (`App.tsx:309-312`), but it is **missing from `VALID_TABS`** in `frontend/src/hooks/useUrlState.ts:3`.
- **Impact:** `getInitialTab` (`useUrlState.ts:22-30`) rejects the unknown tab and falls back to `commandCenter`. So: (a) a refresh on Demand History silently teleports the planner to Command Center; (b) a shared/bookmarked Demand-History URL opens the wrong screen; (c) `usePopstateSync` (`useUrlState.ts:48`) won't restore it on browser back/forward. (`controlTower`/`aiPlanner`/`storyboard` also redirect to commandCenter via `TAB_REDIRECTS`, but those are legacy and NOT in the current sidebar, so that is intentional.)
- **File:** `frontend/src/hooks/useUrlState.ts:3`
- **Recommendation:** Add `"demandHistory"` to the `VALID_TABS` array. (One-token fix; low risk — the route case already exists.)
- **Acceptance:** Add a case to `frontend/src/hooks/__tests__/useUrlState.test.ts` asserting `getInitialTab()` returns `"demandHistory"` when `?tab=demandHistory`. Manually: load `/?tab=demandHistory`, refresh — page stays on Demand History.

## U2.3 — KPI delta arrows/colors are semantically inverted for "bad-when-up" metrics [P2]
- **Category:** consistency
- **Evidence:** `cycle2/screens/customerAnalytics.png` — "Lost Sales (OOS) ↑ 42.9% MoM" rendered in **green** with an up arrow; "Fill Rate ↓ 0.3% MoM" in red. `DeltaBadge` (`KpiSummaryCards.tsx:72-81`) hardcodes `delta >= 0 → green/up`, treating any increase as good. For OOS/Lost Sales and Demand Concentration, an increase is *bad*; the green up-arrow reads as good news to a planner scanning the KPI strip.
- **File:** `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx:72-81`
- **Recommendation:** Add a per-metric `goodDirection: "up" | "down"` (or `invert: boolean`) to each `KpiCardDef`, and color by whether the delta moves in the good direction rather than by sign alone. Lost Sales / OOS / Concentration → "down is good".
- **Acceptance:** Component test: a positive delta on a `goodDirection: "down"` metric renders the `text-red-600` class; a positive delta on a `goodDirection: "up"` metric renders `text-green-600`.

## U2.4 — Zero delta shows a directional arrow ("↑ 0.0% MoM") [P3]
- **Category:** consistency
- **Evidence:** `cycle2/screens/customerAnalytics.png` — "Demand Concentration ↑ 0.0% MoM" and "Order-to-Demand Ratio ↑ 0.0% MoM": a flat (0.0%) change shows a green up-arrow. `DeltaBadge` uses `delta >= 0` so exact-zero is rendered as positive.
- **File:** `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx:72-81`
- **Recommendation:** Treat near-zero deltas (`Math.abs(delta) < 0.05`) as flat — render a neutral em-dash/`→` in muted color, no up/down arrow.
- **Acceptance:** Component test: `delta = 0` renders neither up nor down arrow and uses a muted/neutral color class.

## U2.5 — `/data-quality/lineage/batches` 404s on every Data Quality load (3 console errors) [P2]
- **Category:** usability
- **Evidence:** Digest: 3 console 404s on Data Quality. Live probe: `GET /data-quality/lineage/batches?limit=20` → 404 while sibling `/data-quality/dashboard`, `/checks`, `/history`, `/fix/preview`, `/corrections`, `/corrections/summary` all → 200. Query defined at `frontend/src/api/queries/platform.ts:123`.
- **Impact:** The Pipeline Lineage panel has a graceful empty state ("No pipeline batches yet"), so it isn't fatal — but every load throws a 404 that (with U2.1) surfaces the raw error toast and pollutes the console. The path is wrong or the route is unmounted.
- **File:** `frontend/src/api/queries/platform.ts:123` (frontend); backend route under `api/routers/platform/` for `/data-quality/lineage/*`
- **Recommendation:** Confirm the backend lineage route name; either fix the frontend path to the mounted route or mount the missing `lineage/batches` endpoint. Verify with `make audit-routers` parity and a `curl` probe returning 200.
- **Acceptance:** `GET /data-quality/lineage/batches?limit=20` returns 200 (empty list acceptable); Data Quality tab loads with 0 console 404s.

## U2.6 — Customer Concentration treemap renders empty (only legend, no cells) [P2]
- **Category:** usability
- **Evidence:** `cycle2/screens/customerAnalytics.png` — the "Customer Concentration" panel shows only the color-scale legend (0%→100%) and axis labels; no treemap cells render despite 32,469 customers / 23.0M cases loaded in the adjacent map. Digest text cuts off at "Customer Concentrati". Live probe: `GET /customer-analytics/concentration` → 404 (path likely differs).
- **File:** `frontend/src/tabs/customer-analytics/CustomerTreemap.tsx`
- **Recommendation:** Verify the treemap's data query path/params resolve (probe the real endpoint), and add an explicit empty-state ("No concentration data for the current filters") so an empty panel never looks like a broken chart. Render a loading skeleton while fetching.
- **Acceptance:** With data loaded, the treemap shows ≥1 cell; with no data, it shows a labeled empty state instead of a bare legend. No silent 404.

## U2.7 — Negative-accuracy heatmap cells (e.g. BEER −263.9%) shown without explanation/clamp [P2]
- **Category:** usability
- **Evidence:** `cycle2/screens/aggregateAnalysis.png` + digest lines 160-189 — Accuracy Heatmap shows BEER at `-186.4%`, `-263.9%`, `-92.4%`; Cluster table shows `★-128.04%` accuracy / `228.04% WAPE`. (Carried from F1.5 DEFERRED in cycle 1; still unaddressed.) A planner reading "−263.9% accuracy" has no framing for what an unbounded-negative accuracy means.
- **File:** `frontend/src/tabs/AggregateAnalysisTab.tsx` (and aggregate-analysis heatmap subpanel)
- **Recommendation:** Either (a) clamp displayed accuracy at a floor (e.g. show "<0%" or "0%" with a tooltip "actuals are tiny; WAPE > 100%"), or (b) keep the value but add a one-line legend/tooltip explaining accuracy = `100 − WAPE` can go strongly negative when forecast >> small actuals. Pair the cell with WAPE so the magnitude is interpretable. Do not silently show "−263.9%" with no context.
- **Acceptance:** Heatmap cells with accuracy < 0 render a tooltip/legend explaining the cause; or values are clamped/annotated. A planner can tell apart "model is wrong" from "denominator is tiny".

## U2.8 — Six tab files exceed the 600-line CLAUDE.md hard limit [P3]
- **Category:** simplification
- **Evidence:** `wc -l frontend/src/tabs/*.tsx`: `CommandCenterTab.tsx` 798, `InvPlanningTab.tsx` 705, `StoryboardTab.tsx` 671, `SettingsTab.tsx` 649, `AggregateAnalysisTab.tsx` 649, `DataQualityTab.tsx` 617 — all over the documented "Tab files MUST be < 600 lines" rule.
- **File:** `frontend/src/tabs/CommandCenterTab.tsx` (worst, 798) and the five above
- **Recommendation:** Split each into `frontend/src/tabs/<tab-name>/<Subpanel>.tsx` per the existing pattern (most tabs already have a sibling subpanel dir). Start with CommandCenterTab and InvPlanningTab — the two largest and most-trafficked.
- **Acceptance:** Every file in `frontend/src/tabs/*.tsx` is < 600 lines; `make ui-test` still green.

## U2.9 — `?tab=` deep-link list silently bounces 3 legacy keys with no user feedback [P3]
- **Category:** information-architecture
- **Evidence:** `useUrlState.ts:12-20` `TAB_REDIRECTS` maps `aiPlanner`/`controlTower`/`storyboard` → `commandCenter` on initial load. These keys are absent from the current sidebar (`AppSidebar.tsx`), so the redirect is intentional cleanup — but a user following an old bookmark lands on Command Center with no indication their target moved/was removed.
- **File:** `frontend/src/hooks/useUrlState.ts:12-20`
- **Recommendation:** Low priority. Either drop the redirect map once old links are unlikely, or surface a one-time toast ("Control Tower has merged into Command Center"). Mainly: keep `VALID_TABS`, `TAB_REDIRECTS`, and `NAV_ITEMS` in sync — add a unit test asserting every sidebar `NAV_ITEMS.key` is in `VALID_TABS` (this would have caught U2.2).
- **Acceptance:** Add a test asserting `NAV_ITEMS.every(i => VALID_TABS.includes(i.key))`. (Fails today on `demandHistory` until U2.2 lands; then guards future drift.)

## U2.10 — Customer Analytics filter dropdowns are unusable: dirty, duplicated, un-normalized values [P1]
- **Category:** consistency
- **Evidence:** Live `GET /customer-analytics/filter-options` returns near-duplicate channels differing only by case/trailing whitespace: `"Off Premise Chains"`, `"Off Premise Chains            "`, `"OFF PREMISE CHAINS"`; same for On/Off Premise Independents and On Premise Accounts. It also includes a literal `"null"` string option. The store-type dropdown lists hundreds of unsorted, casing-duplicated entries (capture-digest Customer Map section). A planner who wants "On Premise" sees three competing options and cannot trust the filter.
- **File:** `api/routers/intelligence/customer_analytics/segments.py:311-343` (reads `mv_customer_filter_options`, sql/173); consumed by `frontend/src/tabs/CustomerAnalyticsTab.tsx:223-256`
- **Recommendation:** Normalize the enum source in the MV: `TRIM()`, collapse case to a canonical label, `DISTINCT`, drop `NULL`/`'null'`/`''`, `ORDER BY`. The WHERE predicates in the data endpoints must match the same normalized form (apply `TRIM()`/canonical on both label and filter) so normalization doesn't drop rows. Minimum-safe first step if canonicalization is risky: TRIM + drop the `"null"` string + sort — eliminates the trailing-whitespace and stray-null duplicates without changing match semantics.
- **Acceptance:** `filter-options` returns trimmed, case-insensitively de-duplicated, sorted lists with no `"null"`/empty entry; selecting a normalized channel filters the map without losing matching rows.

## U2.11 — Raw DB enum `below_ss` leaks into Inventory action titles [P3]
- **Category:** consistency
- **Evidence:** Inventory Planning Action Feed renders "Resolve below_ss exception" and "Below Ss — 664631 @ 1401-BULK" (capture-digest Inventory Planning section). `below_ss` is a raw DB enum; "Below Ss" is a naive title-case of the snake_case value. `stockout` rows read cleanly ("Resolve stockout exception"), so the labeling is partial/inconsistent.
- **File:** `frontend/src/tabs/InvPlanningTab.tsx` and its action-feed subpanels
- **Recommendation:** Add a single exception-type label map (`below_ss → "Below Safety Stock"`, `stockout → "Stockout"`, …) and render all exception types through it instead of title-casing the enum.
- **Acceptance:** No raw `snake_case` exception enum appears in the UI; `below_ss` renders as "Below Safety Stock" in both the action title and the subtitle.

---

### Summary
The standout NEW high-leverage fixes are **U2.1** (raw JSON error leak — bypasses the whole sanitization layer; one-function fix) and **U2.2** (Demand History un-deep-linkable / refresh teleports the planner — one-token fix). **U2.3/U2.4** (KPI arrow semantics) are quick, planner-felt consistency wins. The remaining items (lineage 404, empty treemap, negative-accuracy framing, oversized tabs) are real but lower-leverage. **U2.9** proposes a guard test that prevents the U2.2 class of bug from recurring. The product is in good shape overall — the error-handling and deep-link gaps are the most impactful remaining usability defects.
