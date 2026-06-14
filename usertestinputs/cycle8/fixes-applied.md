# Cycle 8 — Fixes Applied

Branch: `restructure`. Method: strict TDD (RED → GREEN → REFACTOR → live-verify) per item.
Planner found **0 new defects**; usability filed 3 P2 + 3 P3. Fixed all 3 P2 trust-in-numbers items.

---

## U8.1 — "Today's Plan" At-Risk tile showed "$4K" while the Action Feed below showed "$3.6K" (same metric) [P2]

**What was wrong:** The banner formatted `financial_at_risk = 3598.89` as `$${(v/1000).toFixed(0)}K` → "$4K", rounding $3,599 *up*, contradicting the Action Feed KPI ("$3.6K") for the identical metric one card below.

**Fix:**
- `frontend/src/tabs/inv-planning/todaysPlanFormat.ts` (NEW) — `formatCompactCurrency()`: one decimal for sub-$10K values (3598.89 → "$3.6K"), no decimal ≥ $10K, "--" for null/0.
- `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx` — At-Risk tile now calls `formatCompactCurrency(summary?.financial_at_risk)`.

**Red→Green evidence:**
- Unit (`todaysPlanFormat.test.ts::formatCompactCurrency`): RED = module missing (`Failed to resolve import "../todaysPlanFormat"`). GREEN = 6 passed (`formatCompactCurrency(3598.89) === "$3.6K"`).
- Component (`TodaysPlanBanner.test.tsx`): RED (banner stashed to old logic) = found `$4K`; with old code the U8.1 assertion `queryByText("$4K")` would fail. GREEN after restore = renders "$3.6K", no "$4K".

**Verification:** Banner tile now renders the same "$3.6K" string as the feed. Acceptance met.

---

## U8.2 — "Today's Plan" stats row printed "0 SKUs · 3,152 at risk · 0 excess ($0K)" — self-contradictory [P2]

**What was wrong:** `/inv-planning/daily-briefing` returns `total_skus:0, excess_count:0, total_excess_value:0` (unpopulated) while `below_ss_count:3152` is real. The row rendered the zeros as literal data, producing "0 SKUs" alongside "3,152 at risk".

**Fix:**
- `frontend/src/tabs/inv-planning/todaysPlanFormat.ts` — `shouldRenderStat()`: a 0/null stat is no-data.
- `frontend/src/tabs/inv-planning/TodaysPlanBanner.tsx` — `total_skus` degrades to "—" when 0/null; the excess chip is omitted entirely when `excess_count` is 0/null (mirrors the existing `avg_health_score != null` guard on the same row).

**Red→Green evidence:**
- `TodaysPlanBanner.test.tsx::U8.2`: RED (banner stashed to old logic) = `expect(queryByText(/0 SKUs/)).not.toBeInTheDocument()` failed — "found `<span>0 SKUs</span>`". GREEN after restore = "3,152 at risk" present, no "0 SKUs", no "0 excess".

**Verification:** Live payload (`total_skus:0, below_ss_count:3152`) now renders "— SKUs · 3,152 at risk" with the empty excess chip dropped. Acceptance met.

---

## U8.3 — Data Quality domain score ignored severity: an INFO-only fail cratered `sku_to_item`/`sku_to_location` to 0% alarm-red [P2]

**What was wrong:** `dq_dashboard()` scored `100*passed/(passed+failed+warnings)` counting every `fail` equally regardless of severity. `sku_to_item` and `sku_to_location`, whose ONLY fails are `info`-severity referential-integrity notices, read 0% red — identical visual weight to genuinely-broken domains, while the Check Catalog correctly badged them INFO.

**Fix:**
- `api/routers/platform/data_quality.py` — dashboard SQL adds `count(*) FILTER (WHERE status='fail' AND severity='info') AS info_fails`; the score denominator now uses `scoring_fails = failed - info_fails` (info fails excluded, like skips). `info_fails` is returned; raw `failed` count stays visible. Row shape `(domain, passed, failed, warnings, skipped, info_fails, total)`.
- `frontend/src/api/queries/platform.ts` — `DQDomainScore` gains `info_fails: number`.
- `frontend/src/tabs/DataQualityTab.tsx` — domain card shows a muted blue "{n} info" chip when `info_fails > 0` (the score now reads 100% green, not red).
- `tests/api/test_data_quality.py` — existing dashboard tests' row tuples updated to the 7-column shape (added `info_fails=0`).

**Red→Green evidence:**
- Backend (`test_data_quality.py::test_dq_dashboard_info_fails_excluded_from_score`): RED = `assert 0.0 == 100.0` (info-only domain still scored 0). GREEN = 18 passed; info-only domain scores 100.0, `info_fails:2`, `failed:2` (raw count kept), and scores ≥ a domain with a genuine warning-fail.
- Frontend (`DataQualityTab.test.tsx::U8.3`): RED = `findByText("2 info")` not found. GREEN = 33 passed; "2 info" chip renders, card badge is green ("rounded-full" pill), never red.

**Verification (live):**
```
curl /data-quality/dashboard:
  before: sku_to_item score=0.0  | sku_to_location score=0.0
  after:  sku_to_item score=100.0 info_fails=2 | sku_to_location score=100.0 info_fails=2
  unchanged genuine fails: forecast_to_sku 0.0 (info_fails=0), inventory 66.7 (info_fails=0)
```
Acceptance met: info-only domains no longer alarm-red; response carries per-severity (`info_fails`) counts; a domain with a non-info fail scores below an info-only domain.

---

## Deferred

- **U8.4 (P3)** — Explorer redundant `Item Ck`/`Item Id` lead columns. Deferred: needs a generic "demote `*_ck` surrogate columns" rule in Explorer field-ordering metadata; touches shared column logic across all domains, lower value than the P2 trust items.
- **U8.5 (P3)** — Portfolio Heatmap `<0%*` cells need an inline per-cell tooltip. Deferred: caption already exists (cycle-3 F3.2 mitigation); pure polish.
- **U8.6 (P3)** — S&OP tab is a CLI-only dead end (needs a new guarded `POST /sop/cycles` + UI button). Deferred (now 5th cycle as U3.4/U4.4): net-new backend route + seeding logic; larger than this cycle's scope.
- **F4.3 (P2, carried 6th cycle)** — Control Tower Portfolio Health 0/100 + Fill Rate "--" (stale `mv_control_tower_kpis`). Honest amber banner mitigates; needs a live base-table fallback in `control_tower.py`.
- **F4.5 / U5.4 (P2, carried)** — Customer Analytics Store Type taxonomy (~275 raw values) needs an upstream canonical mapping table + searchable combobox.
- **U5.5 (P2, carried)** — CommandCenterTab >600-line split (pure refactor).
- **U5.6 (P2, carried)** — Item Analysis FROM/TO raw ISO dropdowns + range validation.
- **F6.2 (P3, carried)** — dead `/customer-analytics/concentration` 404 route (cleanup only, no UI calls it).

## Risk / notes

- All three fixes are surgical. U8.1/U8.2 are frontend-only (HMR-live). U8.3 backend change is additive (one new aggregate column + denominator tweak) and hot-reloaded; the score change only *raises* scores for info-only domains and leaves all genuine-fail domains bit-identical (verified live).
- Pre-existing, untouched: `DataQualityTab.tsx:130` TS error (header `runChecksMutation.mutate()` — documented cycle 3); router ruff nits at lines 123/174/364 (pre-existing, outside my edited block); test-file `RUF059` unused `conn`/`cursor` is the project-wide `_make_pool` convention used by every test in the file. None introduced this cycle.
- The DataQualityTab working tree already held cycle-7 uncommitted changes (Skipped tile, empty-state CTA); only the `info_fails` chip block is authored this cycle.
- No commits made. Changes left in the working tree.
