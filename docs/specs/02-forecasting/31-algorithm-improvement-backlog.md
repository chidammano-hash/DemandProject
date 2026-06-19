# 31 — Algorithm Improvement Backlog (Forecast Accuracy)

**Goal:** lift forecast accuracy above the ~72–75% plateau. This is a prioritized, evidence-gated
backlog, not a spec. Each item says what to change, why (with the evidence that motivated it), how to
test it cheaply, and the gate it must clear before promotion.

**Status legend:** ⬜ not started · 🔬 cheap test pending · 🏗️ in progress · ✅ done · ❌ ruled out

---

## Context (what we already know — read first)

- **Headline accuracy is volume-weighted WAPE** (`accuracy = 100·(1 − ΣABS(F−A)/ABS(ΣA))`); the
  incumbent champion (`dfu_strategy_router`) sits at **~74–75%** on the causal holdout, **86.5% oracle ceiling**.
- **Diagnostic layer is live** (use it to measure every item below): MV `agg_accuracy_by_dfu`
  (`sql/193`), endpoints `/forecast/accuracy/decomposition` + `/forecast/accuracy/error-contributors`,
  and the **Error Decomposition** panel in the Portfolio (Aggregate Analysis) tab. See
  [[project_accuracy_decomposition_2026_06_18]].
- **Error is concentrated** in two segment types with *opposite* root causes:
  - **Low-volume** (`very_low_volume_steady`, `low_volume_periodic`): signal exists but **per-cluster
    trees starve** on sparse series (lgbm ~1–36%); global/foundation models do better.
  - **High-CV / volatile** (`medium_volume_volatile`, `…_moderate`, `…_seasonal_volatile`): best model
    ~10–16%, flat lag curves → **structural / possibly intrinsic noise**.

### Ruled OUT by evidence (do NOT redo without new information)
- ❌ **Champion strategy swap** — tournament (sweep #6): incumbent beat all 12 strategies, none gated.
- ❌ **Static per-cluster model routing** — causal holdout: routed 69.78% < champion 74.21% (gate FAIL;
  the big in-sample "+83pt" was hindsight + coverage bias).
- ❌ **Customer/OOS enrichment** — high-error DFUs are *not* stockout-distorted (error-weighted OOS
  2.9% < simple 4.8%); also `customer_features_monthly` is unpopulated.
- ❌ **Horizon / recursive fixes** — lag curves are flat (error doesn't grow with horizon).

**Principle:** only three things raise the *ceiling* — give models information they lack, use an
architecture suited to the segment, or stop point-forecasting what's intrinsically random. Selection
and blending cannot (proven twice). Scope every experiment to the hard clusters, train on the causal
window, backtest the holdout, and confirm the **ceiling moves** — then gate (≥1% WAPE rel. improvement
vs incumbent, ≥80% coverage) before promotion.

---

## Backlog

### 0. 🔬 GATING — measure the irreducible-noise floor per segment
Compare realized accuracy to `1 − demand_mad/demand_mean` (best any point forecast could do) per
`ml_cluster`, from `dim_sku`. Splits the volatile segments into **model-fixable** (realized ≪ floor)
vs **intrinsic noise** (realized ≈ floor → no point model helps → item 5). This decides which volatile
work is worth doing. Read-only; ~1 query. **Do this first.**

### 1. 🏗️ Fine-tune the foundation models on our own demand history  — *highest leverage*  → **spec'd + code landed: [32-finetuned-foundation-models.md](32-finetuned-foundation-models.md)** (Chronos-Bolt FT pipeline smoke-verified; full fine-tune run + causal-holdout gate pending)
All foundation models (`chronos`, `chronos2`, `chronos2_enriched`, `bolt_hierarchical`) run **zero-shot**
today (`tune: false` in `forecast_pipeline_config.yaml`). They already beat trees on low-volume
zero-shot; fine-tuning / LoRA on the 41M-row history should raise the ceiling, esp. for low-volume.
- **Test:** fine-tune on the train window, backtest the hard clusters on the holdout, check oracle
  ceiling movement. GPU via `DEMAND_GPU`. **Retrain-scale.**
- **Where:** `common/ml/foundation_backtest.py`, foundation loaders; new `fine_tune` params per algo.

### 2. ⬜ Hierarchical / temporal aggregation for low-volume
Model where signal is denser, then disaggregate. `bolt_hierarchical` (customer bottom-up + top-down
true-demand reconciliation) already exists and *won* `medium_volume_moderate` causally. Extend it /
route low-volume clusters through it, or forecast at quarterly grain and split down.
- **Test:** backtest `bolt_hierarchical` + a temporal-aggregation baseline on low-volume clusters.

### 3. ⬜ Exogenous drivers (events/promotions) for volatile SKUs — *conditional on item 0*
Driven volatility (promo spikes, events) is reducible once the driver is a feature. Event schema exists
(`sql/057_create_event_planning.sql`); `chronos2_enriched` accepts covariates. **Only pursue if item 0
shows volatile demand is event-driven, not random.**
- **Test:** verify event coverage on volatile clusters → add event features → backtest those clusters.

### 4. ⬜ Segment-specific objectives + per-cluster tuning — *cheapest, partial retrain*
Retrain the *existing* models better: enable per-cluster tuning (`cluster_tuning_profiles.yaml`
`enabled: false` today) and use the right loss per segment — Tweedie/Poisson or quantile loss for
intermittent/volatile (not MAE), longer early-stopping for sparse. Good warm-up before item 1.
- **Test:** tune the hard clusters, backtest the holdout.

### 5. ⬜ Quantile / distribution forecasting for *irreducibly* volatile SKUs — *change the goal*
If item 0 shows these are noise-floored, no point model wins. The stack has quantile forecasting
(`quantile-train` / `quantile-all`). The win is **hitting service levels at lower inventory**, not a
higher accuracy %. Reframe the KPI/SLA for these SKUs (ties to dual-metric reporting).

### 6. ⬜ Quick win — override champion on the 2 mis-routed high-CV clusters
Causally validated: champion mis-routes `medium_volume_moderate` (static→mstl beats champion 14.3% vs
6.2%) and `medium_volume_seasonal_volatile` (keep-lgbm 27.0% vs champion 20.4%). ~1,350 low-weight
DFUs → a per-DFU/long-tail win, **not** a headline mover. Narrow champion-assignment override; gate first.

---

## Done
- ✅ Diagnostic layer (decomposition + Pareto + dual metric) — `sql/193`, accuracy endpoints, panel.
- ✅ Champion strategy tournament (sweep #6) — ruled out strategy swaps.
- ✅ Causal routing validation — ruled out static per-cluster routing.
