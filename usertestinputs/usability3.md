# Usability Review — Cycle 3

Branch: `restructure`. Method: read cycle3 capture digest + dump + screenshots, then read-only code inspection of `frontend/src/tabs`, `frontend/src/components`, query/error layers, and live endpoint/DB probes. NEW items first. Prior-cycle resolved items (LEDGER) not re-reported. Confirmed resolved this cycle: U2.3/U2.4 (KPI delta colors — Fill Rate ↓ now red, Lost Sales ↑ now red, flat → neutral), U2.2 (Demand History deep-link), F2.1 (Command Center stale banner).

---

## U3.1 — Five tabs use raw `fetch(` instead of `fetchJson`, silently swallowing errors [P1]
- **Category:** consistency
- **Evidence:** `grep "fetch("` over `src/tabs` finds raw `fetch(` in **non-query** code: `ItemAnalysisTab.tsx:140` (`/domains/sales/sample-pair`), `:166` (`/sku/analysis`), `:216` & `:233` (`/domains/sales/suggest`); `SopTab.tsx:116` (`/sop/cycles/{id}/advance`) and `:288` (`/sop/cycles/{id}/approve`); plus `model-tuning/ExperimentBuilder.tsx:89`, `model-tuning/LogViewer.tsx:67`, `EnhancedPromoteModal.tsx` (4×), `EnhancedComparisonPanel.tsx:98`, `clusters/ClusterExperimentBuilder.tsx:126`, `lgbm-tuning/FeatureLabPanel.tsx:317`. CLAUDE.md states: "All HTTP from frontend goes through `src/api/queries/<module>.ts` using `fetchJson`. Never raw `fetch(` in tabs/components." These bypass the U2.1 error-sanitization layer (status attach + FastAPI-detail parse), so a 404/500 either silently swallows (e.g. `ItemAnalysisTab.tsx:147,180` `catch { /* non-blocking */ }`) or surfaces a raw error.
- **Impact:** (a) `ItemAnalysisTab` swallows analysis/forecast failures with empty `catch {}` — the chart just stays blank with no toast, no diagnosis. (b) `SopTab` advance/approve POSTs don't go through the API-key/error layer. (c) Bypasses the whole `formatApiError` mapping the team just built in cycle 2.
- **File:** `frontend/src/tabs/ItemAnalysisTab.tsx:140,166,216,233`; `frontend/src/tabs/SopTab.tsx:116,288`
- **Recommendation:** Add a `sku.ts` (or extend `domains.ts`) query module with `fetchSamplePair()`, `fetchSkuAnalysis()`, `fetchSalesSuggest()` and a `sop.ts` `advanceCycle()`/`approveCycle()` using `fetchJson`. Replace the raw `fetch` calls; drop the empty `catch {}` so React Query surfaces the error state. The model-tuning/clusters subpanels should follow in a sweep.
- **Acceptance:** `grep -rn "[^.]fetch(" frontend/src/tabs` returns only `fetchJson`/`prefetch`/`refetch` matches; ItemAnalysis shows an error state (not a blank chart) when `/sku/analysis` fails; `make ui-test` green.

## U3.2 — Data Quality empty-state instructs `curl -X POST .../dq/run` — a confirmed 404 — while a working "Run Checks Now" button already exists [P1]
- **Category:** usability
- **Evidence:** `DataQualityTab.tsx:243` empty-state step renders `curl -X POST http://localhost:8000/dq/run`. Live probe: `POST /dq/run` → **404**. The working button (`DataQualityTab.tsx:128` → `runDQChecks`, `platform.ts:62`) actually calls `POST /data-quality/run`. So the on-screen "HOW TO POPULATE" copy (a) names the wrong, 404-ing path, and (b) is redundant now that the button is wired (F2.5 button shipped). The digest (lines 1074-1079) shows the planner is told to run a broken curl as the primary remediation.
- **Impact:** A planner copy-pastes the documented command and gets `{"detail":"Not Found"}`; the real fix is the button two inches above. Contradictory guidance erodes trust.
- **File:** `frontend/src/tabs/DataQualityTab.tsx:242-245`
- **Recommendation:** Replace the curl/CLI steps with a one-line prompt pointing at the existing button: e.g. EmptyState `action={{ label: "Run Checks Now", onClick: () => runChecksMutation.mutate() }}`. If a CLI fallback is still wanted, fix the path to `/data-quality/run` and verify the script name `scripts/dq_run_checks.py` exists. Same stale-curl pattern likely in `dqShared.ts:104,212-213` (verify `populate_dq_checks.py` / `fix_dq_issues.py` flags).
- **Acceptance:** No on-screen command 404s; the DQ empty-state's primary action triggers the in-app run (not a curl). Any remaining CLI hint uses the live `/data-quality/run` path.

## U3.3 — Customer Map "State" filter dropdown is full of garbage codes (`.`, `00`, `0D`, `XX`, `null`, 2-char junk) [P1]
- **Category:** consistency
- **Evidence:** `cycle3/capture-digest.md` lines 1531-1665 — the State `<select>` lists `.`, `0`, `00`, `01`, `02`, `07`, `08`, `0D`, `1`, `10`, `13`, … `XX`, `99`, `null`, plus real 2-letter US states intermixed with foreign/province codes (`AB`, `BC`, `ON`, `QC`). `filterOptions?.states` (`CustomerAnalyticsTab.tsx:224`) is rendered verbatim from `mv_customer_filter_options`. U2.10 (DEFERRED) flagged the *Channel* dirtiness; this is the parallel, equally-broken **State** facet — and it visibly degrades the map: in `cycle3/screens/customerAnalytics.png` only Florida lights up because demand is smeared across ~120 junk "states".
- **Impact:** A planner cannot reliably filter by state — the list is unscannable and selecting a real state still misses rows mis-coded under junk values. The map is misleading.
- **File:** `frontend/src/tabs/CustomerAnalyticsTab.tsx:223-227`; source MV `mv_customer_filter_options` (sql/173), served by `api/routers/intelligence/customer_analytics/segments.py`.
- **Recommendation:** Same normalization as U2.10 but for `states`: in the MV, `UPPER(TRIM(state))`, keep only 2-letter `^[A-Z]{2}$` US/territory codes (whitelist or join `dim_location`/a states ref), drop `null`/`''`/numeric/`XX`, `DISTINCT`, `ORDER BY`. Make the WHERE predicate in the map/segment endpoints match the same canonical form. Minimum-safe first step (frontend-only, no MV change): filter the option list client-side to `/^[A-Z]{2}$/` and drop `null` before render — removes the worst noise immediately.
- **Acceptance:** State dropdown shows only valid 2-letter codes, sorted, no `null`/numeric/`.`/`XX`; selecting a state filters the map without dropping matching rows.

## U3.4 — S&OP tab is a dead end: empty state says "Create one via the API or CLI" with no in-app action [P2]
- **Category:** usability
- **Evidence:** `cycle3/capture-digest.md` lines 866-869 + `SopTab.tsx:164` — with 0 cycles the entire tab renders "No active S&OP cycles. Create one via the API or CLI." There is no "New Cycle" button anywhere; the tab already has `advance`/`approve` POST mutations (`SopTab.tsx:116,288`), so write affordances exist — only *create* is missing. (F1.9 DEFERRED cycle 1; still a hard dead end.) `POST /sop/cycles` → 405 confirms there's no create route mounted either.
- **Impact:** The S&OP feature is 100% unusable from the UI on a fresh install — a planner cannot start the workflow the tab's own intro paragraph describes.
- **File:** `frontend/src/tabs/SopTab.tsx:161-166`; backend `api/routers/operations/` (S&OP router — needs a `POST /sop/cycles`).
- **Recommendation:** Add a `POST /sop/cycles` create endpoint (`Depends(require_api_key)`, `%s`) + a `createCycle()` in `evolution.ts` (already the SOP query home) + a "New Cycle" button in the empty state and the Cycles header (month picker → create → select). If a create endpoint genuinely cannot exist yet, at minimum the empty-state copy should not instruct "API or CLI" with no link/command shown.
- **Acceptance:** From the empty S&OP tab, a planner can create and then select a cycle without leaving the UI; `POST /sop/cycles` returns 200/201 and the new cycle appears in the list.

## U3.5 — Demand History "% column" is unlabeled and ambiguous (e.g. PINNACLE VODKA "520.9%", CAPT MORGAN "130.4%") [P2]
- **Category:** usability
- **Evidence:** `cycle3/capture-digest.md` lines 292-492 — each series row shows three numbers: a volume (e.g. `1,361,016`), an abbreviated value (`75.8K`), and a bare percentage (`63.4%` … up to `520.9%`). The header is just "50 OF 50 SERIES" with no column labels. A 520.9% value with no legend is uninterpretable — is it CV? growth? share? recent-vs-prior?
- **Impact:** The primary list a planner scans in Demand History has three unlabeled numeric columns, one of which exceeds 500% with no framing. They cannot tell what to sort/act on.
- **File:** `frontend/src/tabs/demand-history/` series-list subpanel (the Workbench series picker rendered by `DemandHistoryTab`).
- **Recommendation:** Add column headers (Volume / Avg-or-Recent / and the % metric named — likely CV% or YoY%). For an unbounded ratio like CV, add a tooltip and consider a sane display cap or a small inline sparkline so "520.9%" reads as "highly variable" rather than a number error.
- **Acceptance:** The series list has labeled columns; the % column has a header + tooltip naming the metric; a value >100% is visibly framed (tooltip/legend), not bare.

## U3.6 — Accuracy Heatmap shows unbounded negative accuracy (BEER −263.9%) with no clamp or legend [P2]
- **Category:** usability
- **Evidence:** `cycle3/screens/aggregateAnalysis.png` + digest lines 164-193 — BEER row renders `-186.4% / -263.9% / -92.4% / -78.3%` in deep red; the cluster table (digest 246-252) shows `★-128.04%` accuracy / `228.04% WAPE`. (Carried F1.5 / U2.7, DEFERRED both prior cycles; re-confirmed live this cycle, so logging once more as the highest-leverage of the still-open deferrals.) `accuracy = 100 − WAPE` goes strongly negative when forecast ≫ tiny actuals (BEER is a sparse category here), but the cell offers no framing.
- **Impact:** "−263.9% accuracy" is meaningless to a planner with no legend — it reads as a bug, not "denominator is tiny / WAPE > 100%."
- **File:** `frontend/src/tabs/AggregateAnalysisTab.tsx` + aggregate-analysis heatmap subpanel.
- **Recommendation:** Clamp the *displayed* value (show "<0%" or "0%") with a tooltip, or keep the value but pair it with WAPE and a one-line legend: "Accuracy = 100 − WAPE; can go strongly negative when actuals are near zero." Color-scale should saturate at 0% rather than stretching the red ramp to −264%.
- **Acceptance:** Heatmap cells with accuracy < 0 show a tooltip/legend explaining the cause, or are clamped/annotated; a planner can distinguish "model is wrong" from "tiny denominator."

## U3.7 — Customer Concentration treemap still renders empty (legend only, no cells); endpoint 404 [P2]
- **Category:** usability
- **Evidence:** `cycle3/screens/customerAnalytics.png` — the "Customer Concentration" panel shows the gridlines + the 0%→100% color legend but **no treemap cells**, despite "32,469 customers | 22,986,295 cases" loaded in the adjacent map. Live probe: `GET /customer-analytics/concentration` → **404**. (U2.6 DEFERRED cycle 2; re-confirmed.)
- **Impact:** A bare chart with axes-but-no-data looks broken; the planner can't tell whether concentration is genuinely empty or the panel failed.
- **File:** `frontend/src/tabs/customer-analytics/CustomerTreemap.tsx`; concentration endpoint under `api/routers/intelligence/customer_analytics/`.
- **Recommendation:** Fix/confirm the treemap's data path (the 404 means the query path or params don't resolve), and add an explicit empty-state ("No concentration data for the current filters") + a loading skeleton so an empty panel never masquerades as a broken chart.
- **Acceptance:** Treemap shows ≥1 cell with data, or a labeled empty state with no silent 404; concentration endpoint returns 200.

## U3.8 — Five tab files still exceed the 600-line CLAUDE.md hard limit [P3]
- **Category:** simplification
- **Evidence:** `wc -l frontend/src/tabs/*.tsx`: `CommandCenterTab.tsx` **844** (grew from 798 in cycle 2), `InvPlanningTab.tsx` 705, `StoryboardTab.tsx` 671, `SettingsTab.tsx` 649, `AggregateAnalysisTab.tsx` 649, `DataQualityTab.tsx` 617. All over the documented "Tab files MUST be < 600 lines." CommandCenterTab regressed +46 lines since last cycle. (U2.8 DEFERRED; re-confirmed + worsening.)
- **File:** `frontend/src/tabs/CommandCenterTab.tsx` (worst, 844) and the five above.
- **Recommendation:** Split each into `frontend/src/tabs/<tab-name>/<Subpanel>.tsx` per the existing pattern. Prioritize CommandCenterTab (regressing) and InvPlanningTab (most-trafficked).
- **Acceptance:** Every `frontend/src/tabs/*.tsx` < 600 lines; `make ui-test` green.

---

### Summary
The standout NEW high-leverage fixes are **U3.1** (5 tabs bypass `fetchJson` and silently swallow errors — undoes the cycle-2 error-handling work and leaves ItemAnalysis failing blank) and **U3.2** (DQ empty-state instructs a 404 curl while a working button sits above it — a confirmed contradiction). **U3.3** (State filter garbage) is the State-facet twin of the still-deferred U2.10 and visibly breaks the demand map. **U3.4** makes S&OP a UI dead end. The rest (U3.5–U3.8) are real but lower-leverage, with U3.6/U3.7/U3.8 being re-confirmations of prior deferrals (U3.8 is regressing). Cycle-2 KPI-delta and deep-link fixes verified live and working. The product is solid; the frontend-discipline drift (raw fetch) and the contradictory/stale operator instructions are the most impactful remaining defects.
