# Cycle 4 — Fixes Applied

Branch: `restructure`. Method: strict TDD (red → green → refactor → live-verify) per item.
3 fixes shipped — all P1. Each targets a "same screen contradicts itself" defect that the
cycle-3 KPI/data fixes exposed (a tile reads a populated table while the list/catalog beside
it reads an empty one).

---

## F4.2 / U4.1 (P1) — Data Quality Check Catalog empty + "Last Run: Never" while 166 checks ran 11m ago [FIXED]

**What was wrong:** `/data-quality/checks` selected `FROM dim_dq_check_catalog c LEFT JOIN LATERAL (… fact_dq_check_results …)`. The catalog dimension *drives* the query and is empty (0 rows), so the endpoint returned `{"checks":[]}` even though `fact_dq_check_results` held 166 rows / 83 distinct checks from the last 24h. The frontend Check Catalog panel and the "Last Run" tile (which reduces `max(last_run)` over the empty checks list) both fell back to "0 / Never", contradicting the populated dashboard + 32 recent issues on the same page.

**Fix (files):**
- `api/routers/platform/data_quality.py` — `dq_checks()` rewritten so existence is driven by `fact_dq_check_results` (latest result per `check_name` via `DISTINCT ON`), with `dim_dq_check_catalog` demoted to a `LEFT JOIN` enrichment for `check_type`/`enabled`. Catalog now populates from the table the run actually writes.
- `tests/api/test_data_quality.py` — added `test_dq_checks_derives_from_results_when_catalog_empty`.

**Red→Green evidence:**
- Test: `test_dq_checks_derives_from_results_when_catalog_empty`.
- RED: `AssertionError: ... 'dim_dq_check_catalog c left join lateral …'.startswith('fact_dq_check_results')` → the FROM driver was the empty catalog. (Final assertion form: catalog may only follow a `left join`, never be the FROM driver.)
- GREEN: `16 passed` in `tests/api/test_data_quality.py`.

**Verification (curl before→after):**
- BEFORE: `GET /data-quality/checks` → `{"checks":[]}`.
- AFTER: `GET /data-quality/checks` → 83 checks; first row `completeness_customer_customer_no` with `last_run: 2026-06-14T06:54:39…`, `last_status: pass`. The "Last Run" tile now derives a real timestamp from this list; catalog shows 83.

**Acceptance met:** YES — `/checks` returns ≥1 row when results exist (83); Last Run derives a real time; catalog non-empty.

---

## F4.1 (P1) — Command Center exception feed read the empty `exception_queue`; "6142 Open Exceptions" tile above an "Exception data unavailable" feed [FIXED]

**What was wrong:** `CommandCenterTab` builds its feed from `/storyboard/exceptions`, whose handler queries `FROM exception_queue` — a forecast-storyboard table that is empty (0 rows). The 6,142 real, open replenishment exceptions live in `fact_replenishment_exceptions` (reachable only via `/inv-planning/action-feed`). The home screen therefore showed `6142 Open Exceptions` in a KPI tile and `Exception data unavailable` in the feed directly below — self-contradictory and permanently empty (an MV refresh would not fill it).

**Fix (files):**
- `api/routers/intelligence/storyboard.py` — `list_exceptions()` now, when `exception_queue` returns 0 rows, calls a new `_replenishment_fallback(cur, …)` that queries `fact_replenishment_exceptions`, maps text severity (critical/high/medium/low) to the 0..1 numeric severity the feed sorts by, builds a headline, and returns the same `{total, limit, offset, rows}` envelope (rows tagged `source="fact_replenishment_exceptions"`). Status="all" and `severity_min` filters are honored against the replenishment table; returns `None` (normal empty state) when that table is also empty.
- `tests/api/test_storyboard.py` — added `test_list_exceptions_falls_back_to_replenishment_when_queue_empty`.

**Red→Green evidence:**
- Test: `test_list_exceptions_falls_back_to_replenishment_when_queue_empty`.
- RED: `assert data["total"] == 1` → `assert 0 == 1` (no fallback; queue empty so feed empty).
- GREEN: `29 passed` in `tests/api/test_storyboard.py`.

**Verification (curl before→after):**
- BEFORE: `GET /storyboard/exceptions?limit=3` → `{"total":0,"rows":[]}`.
- AFTER: `GET /storyboard/exceptions?limit=3` → `total: 6142`, `source: fact_replenishment_exceptions`, rows `627099 @ 1401-BULK sev 0.95 "Stockout — 627099 @ 1401-BULK"`, `664631 …`, `913305 …`. The Command Center `unified.length === 0` empty-state no longer triggers; the feed renders the same critical actions as the Inventory Planning Action Feed.

**Acceptance met:** YES — with `exception_queue` empty and ≥1 open replenishment exception, the feed is non-empty and the "Exception data unavailable" state is not shown; tile and feed agree.

**Note:** The existing `/storyboard/exceptions/summary` (all-zero) is not consumed by the Command Center feed (which uses the list endpoint), so it was left unchanged to keep the diff focused; the feed contradiction is resolved by the list-endpoint fallback.

---

## F4.5 / U4.2 (P1) — Customer Analytics Channel (33) + Store Type (293) dropdowns: raw case/whitespace dupes + `null` [FIXED]

**What was wrong:** `fetchCustomerAnalyticsFilterOptions` normalized `states` (via `normalizeStateOptions`) but passed `channels`/`store_types` through verbatim from the MV — case-variant duplicates ("Off Premise Chains" / "OFF PREMISE CHAINS"), trailing-whitespace duplicates, and literal `null`. The planner could pick `null`, and demand was smeared across duplicate buckets.

**Fix (files):**
- `frontend/src/api/queries/customer-analytics.ts` — added `normalizeLabelOptions()` (trim, drop `''`/`null`/`undefined`/`n/a`, case-insensitive de-dupe keeping the first canonical casing so the WHERE clause still matches a real value, case-insensitive sort) and applied it to `channels` + `store_types` in `fetchCustomerAnalyticsFilterOptions`.
- `frontend/src/api/queries/__tests__/customer-analytics-labels.test.ts` — new test file (5 cases).

**Red→Green evidence:**
- Test: `normalizeLabelOptions (F4.5 / U4.2)` — 5 cases.
- RED: import of `normalizeLabelOptions` failed (function did not exist) → `5 failed`.
- GREEN: `8 passed` across the labels + states test files.

**Verification (live filter-options endpoint):**
- BEFORE: raw channels 33, store_types 293 (incl. `null`, whitespace/case dupes).
- AFTER (same normalization applied in-app): channels 33→21, store_types 293→275 (nullish + case/whitespace dupes collapsed, sorted). Dropdowns render the deduped set; no `null` selectable.

**Acceptance met:** YES — Channel/Store Type options de-duplicated case-insensitively, nullish dropped, single canonical label per group, sorted — same treatment State received. (Frontend minimum-safe step; durable `UPPER(TRIM())` MV canonicalization remains a follow-up.)

---

## Deferred this cycle

- **F4.3 (P2)** — Portfolio Health 0/100 + Fill Rate "--" (health/fill-rate MVs unpopulated; no live fallback). DEFERRED: needs a new live `COUNT … GROUP BY health_tier` + fill-rate fallback in `control_tower.py` mirroring the exceptions fallback; honest amber banner already removes the trust hazard, so lower value than the three contradictions fixed.
- **F4.4 (P2)** — Cluster Accuracy Comparison table still prints raw negative accuracy. DEFERRED: frontend table component needs the `formatHeatmapAccuracy()` flooring/annotation; deferred to keep this cycle's diff focused on the higher-leverage P1 contradictions. (Acceptance is a snapshot test on a `<0` row — straightforward next cycle.)
- **U4.3 (P2)** — Demand History `%` badge is unbounded single-month MoM, unlabeled. DEFERRED: needs a metric-definition decision (windowed trend vs. capped MoM + tooltip).
- **U4.4 (P2)** — S&OP "Create one via the API or CLI" dead end. DEFERRED (3rd time): requires a new guarded `POST /sop/cycles` backend route + query module + UI button — larger than this cycle's budget.
- **U4.5 (P2)** — 8 power-user subpanels still call raw `fetch()`. DEFERRED: multi-file migration into query modules + guard-test expansion; pure consistency, no user-facing data defect.
- **U4.6 (P3)** — sidebar shortcut digits read like count badges. DEFERRED: low-severity polish.
- **F4.6 (P3)** — "Run Checks Now" button live-trigger polish + FVA Champion "No data" (genuinely-empty data state, no code defect). DEFERRED.

## Risk / notes

- All three fixes are additive/graceful: the DQ `/checks` and storyboard changes still return the catalog/queue rows when those tables ARE populated (catalog becomes a LEFT JOIN enrichment; queue path runs first and the fallback only triggers on a 0 count), so installs that populate the original tables are unaffected.
- `_replenishment_fallback` honors status/item/loc/severity_min but intentionally skips brand/category/market cross-dim filters (those join through queue-specific dims); when a cross-dim filter is active and the queue is empty, the fallback still returns the broader replenishment feed rather than an empty panel — acceptable for the home-screen triage use case.
- Pre-existing, untouched issues left alone: CA chart-component TS errors (`ChannelSunburst`/`CustomerDemandMap`/… `Record<string,unknown>[]` typings — noted in cycle-3 ledger), ruff `B905 zip(strict=)` / `I001` nits throughout `storyboard.py`/`data_quality.py` (none introduced by this cycle's new code, which builds dicts directly without `zip`).
- No commits made; changes left in the working tree.
