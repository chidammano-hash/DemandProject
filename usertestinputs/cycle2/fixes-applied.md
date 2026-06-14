# Cycle 2 — Fixes Applied

Branch: `restructure`. Method: strict TDD (red → green → refactor → live-verify) per item.
Backend tests: `~/.local/bin/uv run pytest`. Frontend: `vitest`.

Pre-existing unrelated failures (NOT introduced this cycle, confirmed by stashing all
changes and re-running): `src/components/__tests__/AppSidebar.test.tsx` (stale nav count
18≠16), `src/components/__tests__/DemandReferencePanel.test.tsx` (recharts mock "1,200"),
and `tests/unit/test_backtest_chronos.py` (no torch). All untouched.

---

## F2.1 (P0) — Command Center false "Portfolio looks healthy!" while KPIs are stale  → FIXED

**What was wrong:** When the control-tower MVs are unrefreshed, `/control-tower/kpis` degrades
to an all-zero payload with a `warning`. The Command Center rendered the zeros plus a green
"Portfolio looks healthy!" empty state, indistinguishable from a genuinely healthy portfolio —
a catastrophic-trust false positive on the planner's default landing screen.

**Fix (files):**
- `frontend/src/tabs/CommandCenterTab.tsx` — derive `kpisStale` from `kpisQ.data?.warning`;
  render an amber `data-testid="mv-stale-warning"` banner ("Portfolio health data unavailable")
  when stale; replace the green "Portfolio looks healthy!" empty state with an amber
  "Exception data unavailable" state (`data-testid="empty-state-stale"`) when stale + empty.

**Red→Green evidence:** test `shows a degraded warning instead of 'healthy' when KPIs are stale (F2.1)`
in `CommandCenterTab.test.tsx`.
- RED: `Unable to find element by testid "mv-stale-warning"` (banner did not exist).
- GREEN: 10 passed (new test + the existing "Portfolio looks healthy!" test, which has no
  warning in its mock, still green).

**Live-verify:** `curl /control-tower/kpis` → `warning present: True`, `open_exceptions_total: 0`.
Before: UI showed green "healthy". After: UI shows amber "data unavailable" banner + stale empty state.

**Acceptance met:** Yes (the UI no longer shows "Portfolio looks healthy!" when a warning is present).

---

## F2.2 / U2.5 (P1) — Data Quality lineage 404s: frontend `/data-quality/lineage/*` vs router `/data-quality/*`  → FIXED

**What was wrong:** Frontend lineage fetchers prefixed `/lineage/`, a segment the medallion
router (`prefix="/data-quality"`, routes `/batches`, `/batches/{id}`) never exposes → hard 404
on every Data Quality tab load (3 console errors / raw error toast).

**Fix (files):**
- `frontend/src/api/queries/platform.ts` — drop `/lineage/` from `fetchBatches`,
  `fetchBatchDetail`, `fetchRowLineage`, and `fetchCorrections` so they hit the mounted paths
  (`/data-quality/batches`, `/data-quality/batches/{id}`, `/data-quality/corrections`).

**Red→Green evidence:** new `src/api/queries/__tests__/platform-lineage.test.ts`.
- RED: 3 failed — `expected '/data-quality/lineage/corrections…' to contain '/data-quality/corrections'`.
- GREEN: 3 passed.

**Live-verify:** `/data-quality/lineage/batches` → 404 (old); `/data-quality/batches` → 200,
`/data-quality/corrections` → 200 (new).

**Acceptance met:** Yes.

---

## U2.1 (P1) — Raw `{"detail":"Not Found"}` JSON leaked into error toasts  → FIXED

**What was wrong:** `fetchJson` threw `new Error(rawBody)` with no `.status`, so `formatApiError`'s
status mapping was bypassed and the raw FastAPI body leaked verbatim into the toast.

**Fix (files):**
- `frontend/src/api/queries/core.ts` — on `!res.ok`, parse the FastAPI `{detail}`, set the
  message to the detail string (not the raw JSON), and `Object.assign(err, { status, detail })`
  so `formatApiError` maps 404→"That record could not be found." etc.

**Red→Green evidence:** new `src/api/queries/__tests__/fetchJson-error.test.ts`.
- RED: `expected '{"detail":"Not Found"}' to be 'That record could not be found.'`.
- GREEN: 3 passed (status attached, no leading `{`, friendly copy).

**Live-verify (behavior):** A 404 from any fetchJson call now surfaces friendly copy in the toast
instead of the raw `{...}` body.

**Acceptance met:** Yes.

---

## U2.2 + U2.9 (P1 / P3) — Demand History not deep-linkable; add NAV/VALID_TABS drift guard  → FIXED

**What was wrong:** `demandHistory` is a first-class sidebar item but was missing from
`VALID_TABS`, so `getInitialTab()` bounced `?tab=demandHistory` to `commandCenter` (refresh /
bookmark teleported the planner).

**Fix (files):**
- `frontend/src/hooks/useUrlState.ts` — add `"demandHistory"` to `VALID_TABS`.
- `frontend/src/hooks/__tests__/useUrlState.test.ts` — add deep-link test + U2.9 guard test
  (`every NAV_ITEMS sidebar key is in VALID_TABS`); update the VALID_TABS length assertion
  26→29 (the assertion was already stale; corrected, not weakened).

**Red→Green evidence:**
- RED (U2.2): `getInitialTab()` returned `commandCenter` for `?tab=demandHistory`.
- GREEN: 19 passed including the deep-link test and the NAV-vs-VALID_TABS guard.

**Live-verify (behavior):** `/?tab=demandHistory` + refresh now stays on Demand History.

**Acceptance met:** Yes.

---

## U2.3 + U2.4 (P2 / P3) — KPI delta arrows/colors inverted for bad-when-up metrics; zero shows arrow  → FIXED

**What was wrong:** `DeltaBadge` hardcoded `delta >= 0 → green/up`. A rising Lost-Sales/OOS/
Concentration metric showed a green up-arrow (reads as good news); an exact-zero delta showed
a directional arrow.

**Fix (files):**
- `frontend/src/tabs/customer-analytics/KpiSummaryCards.tsx` — add `goodDirection: "up"|"down"`
  per `KpiCardDef`; new exported pure `deltaPresentation(delta, goodDirection)` that colors by
  good direction and renders near-zero (`|delta| < 0.05`) deltas flat/neutral (`→`, muted).
  Lost Sales / Demand Concentration tagged `goodDirection: "down"`.

**Red→Green evidence:** new `src/tabs/customer-analytics/__tests__/KpiDelta.test.ts`.
- RED (encoded by replicating old `delta>=0` logic): old returns green for +42.9 on a good='down'
  metric and an up-arrow + flat=false for delta=0 — both contradict the new assertions.
- GREEN: 6 passed.

**Acceptance met:** Yes.

---

## U2.11 (P3) — Raw DB enum `below_ss` leaked into Inventory action titles ("Below Ss")  → FIXED

**What was wrong:** The Action Feed built titles as `'Resolve ' || exception_type || ' exception'`
(raw enum) and the detail line title-cased the snake_case value to garbage ("Below Ss").

**Fix (files):**
- `api/routers/inventory/inv_planning_insights.py` — Source 1 SQL now selects raw `exception_type`
  as `action_label`; new `_EXCEPTION_TYPE_LABELS` map + `_humanize_action_type()`; the enrichment
  loop renders `title = "Resolve <label>"` for exception sources and uses the map for `detail`.

**Red→Green evidence:** new test `test_action_feed_humanizes_exception_enum_in_title_and_detail`
in `tests/api/test_action_feed.py`.
- RED: `assert 'below_ss' == 'Resolve Below Safety Stock'`.
- GREEN: 3 passed (existing action-feed tests' mock rows updated to emit the raw enum, matching
  the new SQL; assertions unchanged).

**Live-verify:** `curl /inv-planning/action-feed` → titles now read "Resolve Below Safety Stock" /
"Resolve Stockout"; details "Below Safety Stock — … @ …". No `below_ss` / "Below Ss".

**Acceptance met:** Yes.

---

## F2.4 (P2) — `/fva/waterfall` 3 tests IndexError on the ai_adjusted promotion path  → FIXED

**What was wrong:** The ai_fva SELECT returns 3 columns, but the guard `if ai_row and
ai_row[1] is not None` entered the block on any non-null 2nd element and then indexed
`ai_row[2]` → `IndexError` when the row had < 3 columns (the test mock supplied a 2-tuple
shared with the seasonal-naive query).

**Fix (files):**
- `api/routers/forecasting/fva.py` — add a `len(ai_row) >= 3` length guard before indexing `[2]`.
- `tests/api/test_fva.py` — add `test_fva_waterfall_ai_adjusted_promoted` proving a real 3-col
  ai_fva row still promotes (`side_effect` gives the naive row then the 3-col ai row).

**Red→Green evidence:**
- RED: `3 failed … IndexError: tuple index out of range` at `fva.py:130`.
- GREEN: 8 passed (7 originals + the new promotion test).

**Live-verify:** `curl /fva/waterfall` → 200 with valid staged ladder.

**Acceptance met:** Yes (no assertion weakened; guard is the root-cause fix and production SQL is unchanged).

---

## Deferred

- **F2.3 / U2.7 (P1/P2) — negative-accuracy heatmap framing.** Deferred: lower-leverage and needs
  a design decision (clamp vs. annotate vs. legend) that touches the aggregate-analysis heatmap
  renderer and a shared accuracy formatter; out of scope for a safe single-cycle TDD fix this round.
- **F2.6 (P3) — Item Analysis low-volume default item.** Deferred: depends on the same low-base
  presentation decision as F2.3; couples to a default-DFU selection choice.
- **F2.5 (P2) — DQ "Run Checks Now" button not wired + stale `/dq/run` instructions.** Deferred:
  requires a write endpoint wire-up (`POST /data-quality/run`) with `require_api_key` plumbing
  through the UI; larger than this cycle's safe budget.
- **U2.6 (P2) — empty Customer Concentration treemap / `/customer-analytics/concentration` 404.**
  Deferred: needs endpoint-path confirmation + an empty-state; non-trivial to verify safely without
  more probing.
- **U2.10 (P1) — dirty Customer Analytics filter options (whitespace/case dupes, "null").**
  Deferred: requires an MV (`mv_customer_filter_options`, sql/173) change PLUS matching WHERE-clause
  normalization across data endpoints to avoid dropping rows — a coordinated DDL+router change that
  warrants its own cycle.
- **U2.8 (P3) — six tab files exceed the 600-line limit.** Deferred: pure refactor; no behavior
  change, high churn, best done in isolation.
- **F2.7 (P3) — capture-harness mislabels.** Not a product defect (harness navigation).

## Risk / notes

- All fixes are additive and degrade-safe. No SQL value interpolation, no `: any` introduced,
  write-endpoint guards untouched.
- F2.1 banner relies only on the existing `warning?: string` field already on `ControlTowerKpis`.
- U2.11 changed the Source-1 SQL output shape (now raw enum); the two pre-existing action-feed
  tests' mock rows were updated to match the new query — assertions unchanged.
- The operational half of F2.1 (running `make refresh-mvs-tiered` to populate the MVs) is an env
  action, not a code change; the product-side fix ensures the UI is honest regardless of MV state.
