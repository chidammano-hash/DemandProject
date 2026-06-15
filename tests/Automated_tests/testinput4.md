# Cycle 4 — Acceptance Findings (Senior Demand Planner)

Capture: all 14 tabs loaded `ok=true`, **zero console errors, zero 5xx, zero empty-where-data-expected** on the workflow-critical surfaces. Command Center, Portfolio/Aggregate Analysis, Inventory Planning, FVA, Demand History, Item Analysis, Explorer, and Customer Analytics all render real, correct data. Prior-cycle fixes (banner/feed integer formatting, FVA champion-rung honest "Coming Soon", CA dark-mode theme tokens, null-MoM "— no prior period", item breadcrumb desc) all hold.

The product is in **good shape**. I found **one genuinely new, low-severity inconsistency** (Clusters tab default view) plus confirmation that several "empty" surfaces are honest empty states, not bugs. Several previously-logged P2/P3 items remain deferred (denominator, tab-line-counts, raw fetch) — not re-prioritized here.

---

## F4.1 — Clusters tab defaults to the empty "ML Pipeline" source, hiding the 310K assignments the rest of the app actually uses
**[SEV P2]**

**Workflow blocked:** Cluster review / segmentation sanity-check during S&OP prep. A planner opening the Clusters tab is told the platform has no clustering, which is false.

**Evidence:**
- Screenshot `cycle4/screens/clusters.png` + digest "DFU Clustering … **No cluster assignments yet. Run the clustering pipeline…**" with source dropdown defaulted to **ML Pipeline**.
- Same cycle, Aggregate Analysis (`cycle4/screens/aggregateAnalysis.png`, digest lines 826-841) shows **"13 CLUSTER ASSIGNMENT BUCKET(S)"** (L2_1…L2_99) with full accuracy/WAPE/bias — i.e. clustering data IS present and in active use.
- Curl proof:
  - `GET /domains/sku/clusters?source=ml` → `{"total_assigned":0,"clusters":[]}`
  - `GET /domains/sku/clusters?source=source` → `{"total_assigned":310558,"clusters":[13 buckets…]}`
- DB: `dim_sku.ml_cluster` is **100% NULL** (0 distinct), while `dim_sku.cluster_assignment` is fully populated (L2_3=109,555 … L2_99=19,691). A promoted experiment exists (`cluster_experiment` id 27, status=completed, n_clusters=16, is_promoted=t) but its labels were never written back to `ml_cluster`.

**Root cause:**
- Frontend: `frontend/src/tabs/clusters/ClusterOverviewPanel.tsx:15` — `useState<"ml"|"source">("ml")` defaults to the empty source. The "Source (sku.txt)" option (populated) is one click away but never the landing view.
- Data: the promotion path for `cluster_experiment` (id 27) did not populate `dim_sku.ml_cluster`; the `cluster_pipeline` job (`submitJob("cluster_pipeline")`) that writes `ml_cluster` has not been run. So the ML path is *genuinely* empty — but the tab presents that as "the platform has no clusters" rather than "the ML KMeans pipeline hasn't run; X assignments exist from source."

**Acceptance criterion (testable):**
- When `source=ml` returns `total_assigned=0` AND `source=source` returns `total_assigned>0`, the Clusters Overview must either (a) default the source toggle to whichever path has assignments, or (b) keep ML default but replace the bare "No cluster assignments yet" copy with an explicit two-state message that names the populated alternative (e.g. "ML KMeans pipeline has not been run. 310,558 SKUs are currently assigned via Source (sku.txt) — switch source above, or Run Clustering Pipeline to compute ML clusters."). A render test on `ClusterOverviewPanel` asserts the populated-alternative hint appears when the ML payload is empty but the source payload is non-empty.

**Planner impact:** Low-frequency tab, but the current copy actively misleads — it implies segmentation is unconfigured when in fact 310K SKUs are clustered and driving every per-cluster accuracy view. Erodes trust in the Cluster slice elsewhere.

---

## Confirmed NOT bugs (honest empty states / by-design) — do not action

- **Control Tower & AI Planner tabs render Command Center.** Intentional aliases — `frontend/src/hooks/useUrlState.ts:17-18` map `aiPlanner → commandCenter` and `controlTower → commandCenter`. The capture harness still probes the retired URL keys, so the screenshots look like dupes, but the redirect is by design (carried U2.6; cosmetic IA decision, not a defect).
- **FVA tab: 0 interventions / Champion·AI·Planner "Coming Soon".** Honest. Naive Seasonal 65.3% → External 71.2% (+5.9 pts, 92,926 rows) is real; downstream rungs are reserved stages (ledger F1.1 deferred, F2.2 fixed). `GET /fva/waterfall?window_months=12` → 200 in 0.46s.
- **S&OP: "0 active cycles".** Honest empty state — no `sop_cycle` data seeded. The 6-stage explainer + "Start new S&OP cycle" CTA are present.
- **Aggregate Analysis BEER row `<0%*` across all 4 months.** Honest. DB confirms BEER actuals are tiny (5,077 units / 83 rows over Nov-25→Feb-26) vs a large external forecast → WAPE>100%. The cell carries the correct footnote ("actuals near zero on a tiny base — review WAPE"). Not a defect.
- **Customer Analytics 8× "Loading…" at page bottom.** Below-fold `LazyPanel` (IntersectionObserver) panels not yet scrolled into view at capture. Above-fold map + concentration treemap render real data (23.0M cases, 32,469 customers). Not stuck.

## Carried / still-deferred (already logged — not re-prioritized)
- **F2.3 / U1.6 (P2/P3):** Data Quality "166 Total" vs "Check Catalog (83)" denominator, and CRITICAL-severity-vs-pass-status badge. Headline "0 Failed / 95% health" is honest (all CRITICAL checks show 0.00 violations = pass); the 166-vs-83 split and the scary CRITICAL badges on passing rows remain a clarity item. Evidence: `cycle4/screens/dataQuality.png`, digest lines 2893-3156.
- **U3.6 / U1.7 (P2):** 7 tab files > 600 LoC — mechanical splits, low correctness value.
- **U1.3 (P2):** raw `fetch()` in model-tuning panels — blocked by pre-existing tsc errors in those files.

---

### newActionableCount = 1
(F4.1, the single new unresolved P0/P1/P2 finding this cycle.)
