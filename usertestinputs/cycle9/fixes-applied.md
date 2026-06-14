# Cycle 9 — Fixes Applied

Branch: `restructure`. Strict TDD (red → green → refactor → live-verify) per item. No commits.

---

## U9.1 (P1) — Unified Action Feed summary computed over the 20-row display page, not the full population (~70× understatement)

**What was wrong:** `get_action_feed` sliced `actions = actions[:limit]` (limit=20) and then built `summary` (total / critical / high / financial_at_risk) by counting/summing the **sliced** list. So the morning-triage headline read "20 critical · $3.6K" while Command Center, on the same exception population, showed "2,465 critical". The dev comment at the old line 302 admitted it.

**Fix (files):**
- `api/routers/inventory/inv_planning_insights.py` — added a dedicated full-population aggregate query (its own SAVEPOINT) that UNION-ALLs the three sources (open exceptions + proposed orders + high-risk targets), counts urgent/high with the SAME urgency thresholds the per-row scores use, and sums financial impact over the whole population. Summary now reads from this aggregate; on aggregate failure it degrades to the displayed-page counts (never 500s). Added `displayed` to the summary so the UI can caption the truncated list.
- `frontend/src/api/queries/inv-planning-insights.ts` — added optional `displayed` to the summary type.
- `frontend/src/tabs/inv-planning/ActionFeedPanel.tsx` — caption "Showing top N of {total} actions … KPIs above reflect the full population" when `total > rows`.

**TDD red→green evidence:**
- Backend test `test_action_feed_summary_reflects_full_population_not_display_page` — RED: `assert 2 == 4180` (summary still counted the 2 returned rows). GREEN: 5 passed. Added `test_action_feed_summary_falls_back_to_page_when_aggregate_fails` for the degrade path; updated 2 existing tests to supply the aggregate `fetchone` row.
- Frontend test `ActionFeedPanel.test.tsx` — RED: `Unable to find /showing top 20 of 6,214/i`. GREEN: 2 passed.

**Verification (curl before→after):**
- Before: `summary {total:20, critical:20, high:0, financial_at_risk:3598.89}`, 20 actions.
- After: `summary {total:6214, critical:4252, high:0, financial_at_risk:12099.96, displayed:20}`, 20 actions. Critical now reconciles with the full exception population.

**Acceptance met:** Yes.

---

## F9.1 (P2) — FVA "Forecast Value Ladder" blank at 3-month window (anchored to wall-clock `current_date`)

**What was wrong:** `fva.py` windowed the forecast horizon with `WHERE f.startdate >= current_date - (N * interval '1 month')`. The DB `current_date` (2026-06-14) is ~2.5 months ahead of the demo forecast horizon (ends 2026-02-01), so a 3-month window matched **0 rows** — every baseline showed "No data". The correct planning date (2026-04-02) keeps the window aligned with the data. Same `current_date` anchor existed in `/roi-summary` (line 254).

**Fix (files):**
- `api/routers/forecasting/fva.py` — import `get_planning_date`; replaced both `current_date` anchors with a bound `%s::date` planning-date parameter (`fva_waterfall` shared `dfu_filter` + `roi_summary`). Sibling routers (production_forecast.py, consensus_plan.py) already use this pattern.
- `tests/api/test_fva.py` — updated 3 existing SQL assertions from the `current_date` pattern to `%s::date - (%s * interval '1 month')` + assert `current_date` absent.

**TDD red→green evidence:**
- Backend test `test_fva_waterfall_windows_on_planning_date_not_wallclock` — RED: `AttributeError: module fva has no attribute 'get_planning_date'` (the import didn't exist; the SQL used `current_date`). GREEN: 9 passed.

**Verification (curl before→after):**
- Before: `/fva/waterfall?months=3` → seasonal_naive/external `state:"missing"`, `n_rows:0`.
- After: `/fva/waterfall?months=3` → seasonal_naive `actual` 65.82% (10,889 rows), external `actual` 72.57% (10,889 rows). `months=12` still healthy (no regression).

**Acceptance met:** Yes.

---

## U9.3 (P2) — Customer Map MoM delta badges had no accessible comparison-period label

**What was wrong:** `DeltaBadge` rendered `↑ 28.1% MoM` in a bare `<span>` with no `aria-label`/`title`. A screen reader announced the magnitude with no period anchor; large jumps (+28%, +43%) read ambiguously.

**Fix (files):**
- `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx` — new exported pure helper `deltaAriaLabel(delta, flat)` producing "Up/Down N% month-over-month vs prior month" (mirrors the Demand-History U6.5 pattern); `DeltaBadge` now sets `aria-label` + `title` from it.

**TDD red→green evidence:**
- Frontend test `KpiSummaryCards.test.tsx` — RED: `Unable to find a label with the text of /28.1% month-over-month vs prior month/i`. GREEN: 1 passed; `KpiDelta.test.ts` (6) still green.

**Verification:** Each delta span on the Customer Map KPI strip now carries a period-anchored `aria-label`/hover `title`; the visible "↑ 28.1% MoM" text is unchanged.

**Acceptance met:** Yes (accessible-label portion). The partial-month base correction is a backend kpis-SQL change and is deferred (see below).

---

## Deferred

- **U9.3 partial-month base** — excluding/footnoting a partial current month requires changing the `/customer-analytics/kpis` delta SQL (month-completeness logic); out of safe scope this cycle. The accessibility half (aria-label) is shipped.
- **U9.2 (P3)** — Today's Plan "Top Actions" duplicates the first 3 feed rows. IA polish; lower value than the three P0/P1/P2 items taken.
- **U9.4 (P3)** — April-dated exceptions under a June planning date need an "as of <date>" vintage line. Data-state artifact; deferred.
- **U9.5 (P3)** — DQ "0.00" passing-value cell ambiguity (muted-zero styling). Polish; deferred.
- **F4.3 (P2)** — Control Tower health/fill-rate live fallback (amber banner mitigates; 7th carry).
- **F4.5/U5.4 (P2)** — Store Type taxonomy needs upstream raw→canonical mapping + searchable combobox.
- **U5.5/U5.6 (P2)** — oversized tab files / Item-Analysis date pickers (pure refactor / validation).
- **F6.2 (P3)** — dead `/customer-analytics/concentration` 404 route (cleanup only).

## Risk / notes

- The U9.1 aggregate query reuses the exact urgency CASE expressions from the per-source detail queries, so "critical" stays semantically aligned across the feed and the headline. It runs in its own SAVEPOINT and degrades to page-level counts on failure — no new 500 surface.
- FVA change is parameterized (`%s::date`), no identifier interpolation, obeys the psycopg3 `%s` rule.
- Working tree already held prior-cycle uncommitted changes (per ledger); the pre-existing ruff RUF005/Optional nits at untouched lines (inv_planning_insights.py:15,403,704-706,763) were left alone. No commits made.
