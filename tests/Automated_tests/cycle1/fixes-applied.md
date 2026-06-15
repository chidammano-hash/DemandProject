# Cycle 1 — Fixes Applied

Date: 2026-06-14
Branch: main (working tree, not committed)
Engineer: senior full-stack pass (strict TDD)

Sources: `testinput1.md` (planner), `usability1.md` (usability). No prior LEDGER.

Selected the highest-value items fixable correctly with a real red→green test:
U1.1 (P1), U1.2 (P1), U1.8 (P2), F1.2 (P2). Quality over quantity — 4 complete,
root-cause fixes, each backed by a test that failed first.

---

## U1.1 — "Today's Plan" banner stamped with browser wall-clock, not the planning/data as-of date [P1]

**What was wrong:** `TodaysPlanBanner.tsx` rendered `new Date().toLocaleDateString(...)`
in the header, so "Today's Plan · Sunday, Jun 14" sat above action rows anchored
to the frozen planning date (Apr 2, 2026). A planner reads "Today's Plan" and
assumes same-day data.

**Fix (files):**
- `frontend/src/tabs/inv-planning/todaysPlanFormat.ts` — new `formatAsOfDate()`: parses the briefing ISO date as a LOCAL date (avoids UTC-midnight day-shift) and renders "Apr 2, 2026"; returns "" for missing/invalid.
- `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx` — replaced the `new Date()` stamp with `Plan as of {formatAsOfDate(briefing?.date)}` (the daily-briefing `date`, which the backend derives from `get_planning_date()`). Renders nothing until the briefing resolves — never the wall clock.

**Verification (test before→after):**
- RED: `TodaysPlanBanner.test.tsx` "stamps the banner with the planning/data as-of date" → `getByText(/Apr 2, 2026/)` not found (banner showed wall-clock).
- GREEN: passes; wall-clock string asserted absent.
- Unit RED→GREEN: `todaysPlanFormat.test.ts` "formatAsOfDate" suite (local-date parse, empty on invalid).
- Live: `curl /inv-planning/daily-briefing` → `date=2026-04-02` (confirms the source field is the planning date, not system date).

**Acceptance met:** Yes — banner date equals the action-feed as-of date, verified by a component test mocking the briefing date; wall-clock no longer used.

---

## U1.2 — Inline hex chart colors in CommandCenterTab (theme-rule violation) [P1]

**What was wrong:** `CommandCenterTab.tsx` Portfolio Trend lines used literal
`stroke="#3b82f6"` / `stroke="#10b981"`. CLAUDE.md forbids inline hex in `tabs/`;
colors must come from `useChartColors()` so they adapt to Light/Soft/Dark.

**Fix (files):**
- `frontend/src/tabs/CommandCenterTab.tsx` — imported `useChartColors`, read `trendColors` from the theme palette, and set the two `<Line stroke>` props to `trendColors[0]` / `trendColors[1]`.

**Verification (test before→after):**
- RED: new `CommandCenterTab.theme.test.ts` — reads the source; asserts no 6-digit hex literal AND that `useChartColors` is referenced. Both assertions failed (2 hex literals; no hook).
- GREEN: both pass. Existing `CommandCenterTab.test.tsx` + `CommandCenterTab.labels.test.ts` stay green (18 tests).

**Acceptance met:** Yes — `grep '#[0-9a-fA-F]\{6\}' CommandCenterTab.tsx` returns nothing; lines read theme palette colors.

---

## U1.8 — Exception/action rows were code-only ("627099 @ 1401-BULK"), no product name [P2]

**What was wrong:** The action feed listed actions by numeric `item_id` + location
only. The same SKUs are human-readable elsewhere (`dim_item.item_desc`). A planner
can't recognize "627099" without clicking through.

**Fix (files):**
- `api/routers/inventory/inv_planning_insights.py` — all three action-feed source SELECTs now `LEFT JOIN dim_item` and append `item_desc`; new `_item_desc(row)` helper reads the trailing column (tolerant of legacy/mocked short rows). The `detail` subtitle now leads with the description: `"Stockout — TITOS HANDMADE VODKA 80 (627099 @ 1401-BULK)"`.
- `frontend/src/api/queries/inv-planning-insights.ts` — added `item_desc: string | null` to `ActionFeedItem`.
- `frontend/src/tabs/inv-planning/ActionFeedPanel.tsx` — renders `item_desc` as a prominent product-name line on each row.

**Verification (test before→after):**
- Backend RED: `test_action_feed_surfaces_item_description` → `KeyError: 'item_desc'`.
- Backend GREEN: 8/8 in `test_action_feed.py` pass (existing 8-tuple mocks still pass via the length-guarded helper).
- Frontend RED: `ActionFeedPanel.test.tsx` "renders the human-readable item description" → `getByText("TITOS HANDMADE VODKA 80")` not found.
- Frontend GREEN: passes.
- Live: `curl /inv-planning/action-feed?limit=3` → `item_desc="MENAGE A TROIS A(D/R/S)3P PAD(44"`, `detail="Stockout — MENAGE A TROIS A(D/R/S)3P PAD(44 (627099 @ 1401-BULK)"`.

**Acceptance met:** Yes — each row shows a human-readable description in addition to id+location; verified by render test + live curl.

---

## F1.2 — "$ at risk" labeled as total $ but is 7-day lost margin only [P2]

**What was wrong:** `summary.financial_at_risk` ($12K over 2,500+ critical stockouts)
is 7-day lost *gross margin* (`financial_impact_total = daily_demand * margin *
min(7, days_at_risk)`), but the tile read "Financial Impact at Risk" with no scope —
an implausibly tiny, silently-scoped figure that erodes triage trust.

**Fix (files):**
- `api/routers/inventory/inv_planning_insights.py` — added `_FINANCIAL_AT_RISK_BASIS` constant and surfaced `summary.financial_at_risk_basis = "7-day lost gross margin (open exceptions) + proposed order value"`.
- `frontend/src/api/queries/inv-planning-insights.ts` — added `financial_at_risk_basis?: string` to the summary type.
- `frontend/src/tabs/inv-planning/ActionFeedPanel.tsx` — passes the basis as the KpiCard `sublabel` so the tile names its window/basis.

**Verification (test before→after):**
- Backend RED: `test_action_feed_summary_declares_financial_at_risk_basis` → `KeyError: 'financial_at_risk_basis'`.
- Backend GREEN: passes (basis names "7" and "margin").
- Frontend RED: `ActionFeedPanel.test.tsx` "labels the $ at risk tile with its 7-day lost-margin basis" → `/7-day lost gross margin/` not found.
- Frontend GREEN: passes.
- Live: `curl /inv-planning/action-feed?limit=1` → `financial_at_risk_basis="7-day lost gross margin (open exceptions) + proposed order value"`.

**Acceptance met:** Yes (option a) — the tile now names the window and basis a planner can defend; the magnitude is reconcilable to a documented formula. (The 30-day figure / metric-change is left as a future enhancement; the honest-labeling acceptance branch is satisfied.)

---

## Deferred

- **F1.1 (Champion FVA rung, P2):** The waterfall query reads only `fact_external_forecast_monthly` (external-only); champion accuracy lives in `fact_production_forecast` / the backtest lag archive. A correct fix needs a separate champion-accuracy query path (or a relabel to "Coming Soon"). Larger than one safe TDD slice this cycle; deferred to keep quality high rather than ship a shaky waterfall change.
- **F1.3 / U1.6 (Data Quality count reconciliation + CRITICAL-badge styling, P3/P2):** Frontend tile/catalog denominator mismatch + severity-vs-status badge confusion. Plausible but needs careful DataQualityTab work; deferred behind the P1/P2 triage items.
- **F1.4 (cold affinity query 11s, P3):** Performance/MV item — out of scope for a TDD UI/contract cycle; left as won't-fix-this-cycle (resolves once Redis-warm).
- **U1.3 (raw fetch in model-tuning panels, P2):** Real, but a multi-file move into a queries module + guard update; the pre-existing tsc errors in those exact files (model-tuning) make a clean slice risky this cycle. Deferred.
- **U1.5 (banner $12K vs feed $12.1K rounding, P3):** Two intentionally different formatters (compact vs full). Low value; the shared compact formatter already exists for the banner. Not worth forcing identical strings across two deliberately-different presentations this cycle.
- **U1.7 (tab files > 600 lines, P2):** Mechanical splits across 7 tabs; high churn, low correctness value. Deferred.
- **U1.8 (Command Center storyboard rows):** Item descriptions added to the Inventory Planning action feed (primary triage surface). The Command Center storyboard exception path is a separate data flow; descriptions there are a clean follow-up.

## Risk / notes

- The `_item_desc` helper is length-guarded so legacy 8-tuple test mocks and any schema drift degrade to id-only labels instead of IndexError-ing the feed — consistent with the existing per-source SAVEPOINT isolation.
- `formatAsOfDate` parses `YYYY-MM-DD` as a local date deliberately; `new Date("2026-04-02")` is UTC-midnight and would render "Apr 1" in negative-offset zones.
- No new ruff errors in the touched backend file (verified before/after stash: 6 pre-existing both ways). No new tsc errors in any touched frontend file (the remaining tsc errors are all in pre-existing untouched files: model-tuning/*, SettingsTab, SqlRunnerTab, StoryboardTab, types/index.ts).
- Test runs: backend `test_action_feed.py` 8/8 + `test_daily_briefing.py` green (193 passed across inv-planning/action/insight/briefing selection); frontend full `src/tabs` suite 611/611 green.
