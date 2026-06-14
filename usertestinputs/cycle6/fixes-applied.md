# Cycle 6 — Fixes Applied

Branch `restructure`. Strict TDD (RED → GREEN → REFACTOR → live-verify) per item.
All fixes are frontend-only (no SQL, no API prefixes, no backend routes). Live via Vite HMR.

The planner review reported **no new P0/P1** (F6.1 is P3, F4.3/F4.5/F6.2 carried). The
usability review supplied the strongest NEW items: **U6.1 (P1), U6.3 (P1), U6.2 (P2),
U6.4 (P2)** plus P3s U6.5/U6.6 and P3 F6.1. I fixed 6 of these end-to-end.

---

## U6.1 (P1) — Portfolio KPI delta sign contradicts its own color (WAPE "−1.9pp" in red)

- **What was wrong:** `AggregateAnalysisTab` negated the WAPE/Bias delta (`delta: -kpi.deltas.wape_pct`, `-Math.abs(bias)`) so `KpiCard` colored a lower-is-better metric correctly — but `KpiCard` then printed that same negated number verbatim, so the displayed magnitude's sign was a lie about the metric's true movement. Bias always showed negative.
- **Fix:**
  - `frontend/src/components/KpiCard.tsx` — added `trend.goodDirection?: "up" | "down"`. When supplied, the card colors green when the delta moves in the good direction, red when opposite, neutral at 0 — decoupled from the *displayed* sign (which is always the raw delta). `direction` still drives color when `goodDirection` is omitted (backward compat).
  - `frontend/src/tabs/AggregateAnalysisTab.tsx` — pass the RAW `kpi.deltas.{wape,bias,accuracy}_pct` for display, plus `goodDirection`: WAPE `"down"`, Accuracy `"up"`, Bias dynamic (`"down"` when current bias ≥ 0, else `"up"` — green = moving toward zero).
- **RED→GREEN evidence (`KpiCard.test.tsx`):** Added 5 tests under "U6.1 goodDirection decouples display-sign from color". RED: `colors a regression RED when a lower-is-better metric rises` → "expected 'flex … text-[var(--kpi-best)]' to contain 'text-[var(--kpi-warning)]'" (color still keyed off `direction`). GREEN: 40 passed.
- **Verification (live):** `curl /dashboard/kpis?window=3` → `wape_pct 26.07, deltas.wape_pct 1.85`. Before: card showed "−1.85pp" red. After: shows true "+1.85pp" (WAPE rose) in red — sign and color now agree. Bias `6.57`, delta `+9.8` → "+9.8pp" red (moved away from zero).
- **Acceptance met:** YES.

## U6.3 (P1) — Customer Concentration treemap blank despite a valid 200 payload (no width → series collapses)

- **What was wrong:** `CustomerTreemap` rendered `<ReactECharts style={{ height: 360 }}>` (height only) inside a `<div role="img">` with no width. The cycle-5 `mergeEchartsProps` width default did not reach this panel's inner wrapper, so ECharts measured 0px and the treemap laid out into nothing while the visualMap legend still painted.
- **Fix:** `frontend/src/tabs/customer-analytics/CustomerTreemap.tsx` — explicit `style={{ height: 360, width: "100%" }}` on the chart and `className="w-full"` on the `role="img"` wrapper.
- **RED→GREEN evidence (new `CustomerTreemap.test.tsx`):** RED: "wraps the chart in a full-width container" → "expected '' to contain 'w-full'"; width test → chart `style.width` undefined. GREEN: 2 passed (chart gets `width:"100%"`, height 360; wrapper has `w-full`).
- **Verification (live):** `curl /customer-analytics/treemap` → valid `{tree:[{name:"FL", value:12088589.5, children:[{name:"Off Premise Chains", … PUBLIX WAREHOUSE …}]}]}`. With a non-zero width the nested rectangles now have drawable area.
- **Acceptance met:** YES.

## U6.2 (P2) — Explorer renders the literal string `null` (API returns `'null'`, not JSON null)

- **What was wrong:** `formatCell()` mapped only JS `null`/`undefined`/`""` to `"-"`; the API/MVs emit the literal string `"null"` (e.g. UPC), which fell through to `String("null")` → `"null"`. Clutters every Explorer domain.
- **Fix:**
  - `frontend/src/lib/formatters.ts` — added a `NULL_SENTINELS` set (`null`/`none`/`na`/`undefined`, case-insensitive, trimmed) and an `isEmptyCell()` predicate; `formatCell` returns `"-"` for sentinels. Mirrors the load-time `'' / 'null' / 'none' / 'NA' → NULL` rule.
  - `frontend/src/tabs/explorer/ExplorerTable.tsx` — cell `title` tooltip now uses `isEmptyCell()` so hover shows blank, not `"null"`.
- **RED→GREEN evidence (`formatters.test.ts`):** RED: `formatCell("none")` → "expected 'none' to be '-'". GREEN: 36 passed; `formatCell("Null Object Brand")` preserved.
- **Verification (live):** `curl /domains/item?limit=50` → 24/50 items have string `"upc":"null"`. Now rendered as `-`.
- **Acceptance met:** YES.

## U6.4 (P2) — Demand-History rail lists the same item description multiple times with no disambiguator

- **What was wrong:** The rail `TreeNode` showed only `series.label` (the description), so four distinct `item_id`s named "TITOS HANDMADE VODKA 80" were indistinguishable. The id lived only in `series.key`. The existing `formatSeriesTitle()` (chart title) already appends the key but was not used by the rail and was not exported.
- **Fix:** `frontend/src/tabs/demand-history/WorkbenchPanel.tsx` — exported `formatSeriesTitle`; the rail row now appends a muted `#<key>` suffix (item_id, with `||`→`·`) next to the description and sets the button `title` to the full `formatSeriesTitle(series)`.
- **RED→GREEN evidence (new `formatSeriesTitle.test.ts`):** RED: import of `formatSeriesTitle` failed (not exported) → 3 tests error. GREEN: 3 passed — two same-named items get distinct labels; `item||loc` renders `105430 - 1401-BULK`; no double-key when label==key.
- **Verification:** `DemandWorkbenchTab.test.tsx` still green (label still visible; id suffix added). Live: each duplicate "TITOS …" row now shows a distinct `#id`.
- **Acceptance met:** YES.

## U6.5 (P3) — Demand-History trailing % is an unlabeled single-month MoM; 520% green up-arrow misleads

- **What was wrong:** The trailing colored `%` (`((last-prev)/prev)*100`) had no header, tooltip, or aria; a low-base spike read as a sustained trend.
- **Fix:** `frontend/src/tabs/demand-history/WorkbenchPanel.tsx` — the MoM span now carries `aria-label`/`title` "Month-over-month change: +X%" (with "(single-month spike on a low base)" and a `*` marker when `|MoM| > 200`). Added a one-time rail header ("… series · last · MoM %") with an explanatory `title`.
- **RED→GREEN evidence (`DemandWorkbenchTab.test.tsx`):** RED: "labels the month-over-month delta with an accessible name" → "expected null not to be null" (no `aria-label`). GREEN: 8 passed (Item A 400→600 = +50.0%, aria-label contains "month-over-month").
- **Acceptance met:** YES (labeled + aria + low-base footnote).

## U6.6 (P3) — Demand-History series rows are toggle buttons with no `aria-pressed`

- **What was wrong:** Each series row `<button>` conveyed selection only by background color — no `aria-pressed` for assistive tech.
- **Fix:** `frontend/src/tabs/demand-history/WorkbenchPanel.tsx` — added `aria-pressed={isSelected}` to the row button.
- **RED→GREEN evidence (`DemandWorkbenchTab.test.tsx`):** RED: "exposes aria-pressed on a series row" → attribute absent. GREEN: 8 passed (false → true after click).
- **Acceptance met:** YES (pressed state). Note: the live "overlay max 3" count announcement (`role="status"`) was left for a follow-up — the core color-independent pressed state is covered.

## F6.1 (P3) — Data Quality Check Catalog over-weights configured severity; 28 passing critical checks read as failures

- **What was wrong:** The "Severity" column rendered the configured severity-IF-IT-FAILS (`critical`) with a bold red pill even when the check was passing, next to a context-free "Last Value 0.00". 28 passing critical rows read as 28 broken checks.
- **Fix:** `frontend/src/tabs/data-quality/CheckCatalogPanel.tsx` —
  - Severity pill is muted (`text-muted-foreground`) when `last_status === "pass"`, keeping the alarming `SEVERITY_STYLE` only when not passing; a `title` clarifies "Configured severity if this check fails … (currently passing)".
  - "Last Value" header renamed "Last Value (violations)" with a `title`; each cell gets a `title` "Defect/violation count … 0.00 means zero violations (a clean pass)".
- **RED→GREEN evidence (new `CheckCatalogPanel.test.tsx`):** RED (fix stashed): passing-critical pill still `text-red-700` (not muted); Last Value cell had no `title` containing "violation". GREEN: 3 passed; failing-critical still red.
- **Verification (live):** `curl /data-quality/checks` → 28 `severity:"critical", last_status:"pass", last_value:0.0` rows. These now render muted (not red), with the violations tooltip.
- **Acceptance met:** YES.

---

## Deferred

- **F4.3 (P2)** — Portfolio Health 0/100 + Fill Rate "--": needs a live health/fill-rate fallback in `control_tower.py` mirroring the exceptions fallback. Honest amber banner already mitigates (not a trust hazard). Backend change, deferred (4th cycle).
- **F4.5 / U5.4 (P2)** — Store Type dropdown ~275 raw taxonomy values: needs an upstream raw→canonical mapping table + searchable combobox; cannot be canonicalized by client de-dupe alone. Deferred.
- **U5.5 (P2)** — `CommandCenterTab.tsx` > 600 lines: pure refactor, no behavior change. Deferred.
- **U5.6 (P2)** — Item Analysis FROM/TO raw ISO dropdowns + TO ≥ FROM validation. Deferred.
- **F6.2 (P3)** — `/customer-analytics/concentration` 404: dead/unused route (no UI calls it). Cleanup only, not a user break. Deferred.
- **U6.6 partial** — live overlay-count `role="status"` announcement + aria-disable at the 3-cap. Core `aria-pressed` shipped; the announced count is a follow-up.

## Risk / Notes

- All changes are frontend-only and went live via HMR; no migrations, no API prefixes, no backend behavior change.
- `KpiCard.goodDirection` is additive and optional — existing `direction`-only callers (Command-Center KPIs etc.) are unchanged.
- Touched files are TypeScript-clean. Pre-existing, unrelated TS errors left untouched: `CustomerTreemap.tsx:71` (`ExportButtons getData` TreemapNode[] vs Record), `AggregateAnalysisTab.tsx:626` (ChampionPanel props) — both present on a clean tree.
- Test runs: 9 targeted files 157 passed; full frontend suite 1054 passed. 2 pre-existing unrelated failures remain (`AppSidebar.test.tsx` stale nav count, `DemandReferencePanel.test.tsx` recharts mock) — confirmed failing on a clean tree (git stash), not caused by this cycle.
- The working tree contains substantial uncommitted changes from prior cycles (CommandCenterTab, ItemAnalysisTab, control_tower.py, several backend tests, etc.) that were NOT authored this cycle. No commits made.
