# PRD: Hierarchical Bolt — Customer-Level Bottom-Up with Reconciliation

**Status:** Draft
**Author:** Auto-generated
**Date:** 2026-04-03
**Spec ID:** 02-09
**Domain:** Forecasting / ML Pipeline

---

## 1. Problem Statement

Current forecasting operates on `fact_sales_monthly` — an **inventory-constrained**
signal. When a stockout occurs, sales = 0 but true demand != 0. Every model in
the pipeline trains on this biased history, systematically under-forecasting
items affected by stockouts.

`fact_customer_demand_monthly` contains `demand_qty` (what customers actually
ordered) at the individual customer level. This is the **unconstrained demand
signal** that supply chain planning actually needs.

**Chronos Bolt** is the ideal model for customer-level forecasting because:
- **Zero-shot** — no training required, handles the sparse customer-item series
- **250x faster** than Chronos T5 — fast enough to run at customer granularity
- **Handles short series** — works with as few as 3 data points
- **Quantile output** — gives uncertainty bands for free

The challenge: individual customer forecasts are noisy. The solution:
**hierarchical reconciliation** — combine bottom-up customer forecasts with a
top-down item-location forecast to get the best of both worlds.

---

## 2. Algorithm Design

### 2.1 Two-Level Hierarchy

```
Level 0 (Top):     item_id + location_id           ← top-down Bolt forecast
                          |
Level 1 (Bottom):  item_id + customer_no + location_id  ← bottom-up Bolt forecasts
                     cust_1   cust_2   cust_3  ...  cust_N
```

Each level gets its own Bolt forecast. The reconciliation engine combines them.

### 2.2 Pipeline Steps

```
Step 1: BOTTOM-UP INFERENCE
─────────────────────────────
fact_customer_demand_monthly (demand_qty)
  → group by (item_id, customer_no, location_id)
  → filter: >= 3 non-zero months
  → Bolt inference per customer series
  → output: forecast per customer-item-loc-month

Step 2: BOTTOM-UP AGGREGATION
─────────────────────────────
Sum customer forecasts to item-loc level:
  BU_forecast[item, loc, month] = SUM(forecast[item, cust_i, loc, month])

Step 3: TOP-DOWN INFERENCE
─────────────────────────────
fact_customer_demand_monthly
  → aggregate to item_id + location_id (sum demand_qty)
  → Bolt inference on aggregated series
  → output: forecast per item-loc-month
  TD_forecast[item, loc, month] = Bolt(aggregated_demand_series)

Step 4: RECONCILIATION
─────────────────────────────
Combine BU and TD using MinTrace (shrinkage) or OLS:
  Reconciled[item, loc, month] = w_BU * BU + w_TD * TD

Step 5: MAP TO DFU GRAIN
─────────────────────────────
Map item_id + location_id → item_id + customer_group + loc (DFU grain)
  using dim_sku lookup
  → output: same schema as any other backtest model

Step 6: STORE & COMPETE
─────────────────────────────
Write to backtest CSV → load to backtest_lag_archive
  → competes in champion selection alongside lgbm, catboost, etc.
```

### 2.3 Why This Architecture

| Design Choice | Rationale |
|---------------|-----------|
| Bolt (not LGBM) at customer level | Zero-shot handles sparse customer series; tree models need 12+ months |
| `demand_qty` (not `sales_qty`) | Unconstrained signal — the entire point |
| Bottom-up + top-down reconciliation | Pure bottom-up is noisy; pure top-down misses customer signals |
| MinTrace shrinkage | Industry-standard optimal reconciliation; minimizes trace of forecast error covariance |
| Output at DFU grain | Compatible with existing champion selector — no pipeline changes needed |

---

## 3. Reconciliation Methods

### 3.1 MinTrace Shrinkage (Recommended)

The Nixtla `hierarchicalforecast` library (already imported in
`adv_algorithm_testing/reconciliation.py`) provides MinTrace:

```python
from hierarchicalforecast.methods import MinTraceShrink
from hierarchicalforecast.core import HierarchicalReconciliation

reconciler = HierarchicalReconciliation(reconcilers=[MinTraceShrink()])
reconciled = reconciler.reconcile(Y_hat_df=forecasts, Y_df=actuals, S=summing_matrix)
```

**How MinTrace works:**
- Builds a summing matrix S that encodes the hierarchy (top = sum of bottoms)
- Estimates the forecast error covariance using shrinkage (handles high-dimensional case)
- Finds optimal linear combination that minimizes total forecast error variance
- Guarantees hierarchical coherence (reconciled bottom-up sums to reconciled top)

**Expected behavior:**
- Where bottom-up is more accurate (diversified customers, no stockouts) → weights favor BU
- Where top-down is more accurate (sparse customers, noisy bottoms) → weights favor TD
- Automatically adapts per item-location based on error patterns

### 3.2 Simple Weighted Average (Fallback)

If MinTrace is too slow or unstable for the first iteration:

```python
# Shrinkage parameter alpha learned from backtest residuals
alpha = 0.6  # default: 60% bottom-up, 40% top-down
reconciled = alpha * BU_forecast + (1 - alpha) * TD_forecast
```

This is faster but doesn't guarantee hierarchical coherence and uses a
global alpha rather than per-item-loc optimization.

### 3.3 Method Selection

| Method | Accuracy | Speed | Coherence | Complexity |
|--------|----------|-------|-----------|------------|
| MinTrace Shrink | Best | Medium | Guaranteed | Medium |
| OLS | Good | Fast | Guaranteed | Low |
| Simple weighted avg | OK | Fastest | No | Trivial |
| Bottom-up only (no TD) | Noisy | Fast | N/A | Trivial |

**Plan:** Start with simple weighted average (alpha=0.6) for Phase 1, upgrade to
MinTrace in Phase 2 after validating the pipeline end-to-end.

---

## 4. Backtest Framework Design

### 4.1 New Script: `scripts/run_backtest_bolt_hierarchical.py`

This is a **new backtest script** — not a modification of the existing Bolt script.
The existing `run_backtest_chronos_bolt.py` continues to work unchanged.

**Key differences from standard Bolt backtest:**

| Aspect | Standard Bolt | Hierarchical Bolt |
|--------|--------------|-------------------|
| Data source | `fact_sales_monthly` | `fact_customer_demand_monthly` |
| Grain | item + customer_group + loc | item + customer_no + location |
| Signal | `qty` (constrained sales) | `demand_qty` (true demand) |
| Inference count | ~100K DFU series | ~500K-2M customer series + ~100K aggregated |
| Post-processing | None | Aggregate + reconcile + map to DFU |
| Output grain | item + customer_group + loc | item + customer_group + loc (same!) |
| Model ID | `chronos_bolt` | `bolt_hierarchical` |

### 4.2 Data Loading SQL

```sql
-- Bottom-level series: customer × item × location
SELECT
    f.item_id || '_' || f.customer_no || '_' || f.location_id AS cust_series_key,
    f.item_id,
    f.customer_no,
    f.location_id AS loc,
    f.startdate,
    f.demand_qty AS qty
FROM fact_customer_demand_monthly f
WHERE f.demand_qty > 0
  AND f.startdate <= %(cutoff)s
ORDER BY cust_series_key, f.startdate

-- Top-level series: item × location (aggregated)
SELECT
    f.item_id || '_' || f.location_id AS agg_series_key,
    f.item_id,
    f.location_id AS loc,
    f.startdate,
    SUM(f.demand_qty) AS qty
FROM fact_customer_demand_monthly f
WHERE f.startdate <= %(cutoff)s
GROUP BY f.item_id, f.location_id, f.startdate
ORDER BY agg_series_key, f.startdate
```

### 4.3 Series Filtering

```python
MIN_NONZERO_MONTHS = 3   # customer series with < 3 non-zero months → skip
MIN_HISTORY_MONTHS = 6   # item-loc must have >= 6 months total history
MAX_CUSTOMERS_PER_ITEM_LOC = 200  # cap to prevent explosion
```

For item-locs with > 200 active customers, keep top 200 by demand volume
and aggregate the rest into an "other" bucket series.

### 4.4 Expanding Window Timeframes

Same 10-timeframe expanding window as existing backtests:

```python
timeframes = generate_timeframes(
    latest_date=planning_date,
    n_timeframes=10,
    embargo_months=0,
)
```

Each timeframe:
1. Filter customer demand data to `startdate <= train_end`
2. Run Bolt on all customer-level series (bottom-up)
3. Run Bolt on aggregated item-loc series (top-down)
4. Reconcile
5. Map to DFU grain
6. Compute accuracy against actuals

### 4.5 Output Format

Identical to existing backtests — the champion selector sees no difference:

```csv
forecast_ck, item_id, customer_group, loc, fcstdate, startdate, lag, execution_lag, basefcst_pref, tothist_dmd, model_id
item_123_cg_loc01_2026-02-01_2026-03-01, item_123, cg, loc01, 2026-02-01, 2026-03-01, 0, 0, 48.50, 50.00, bolt_hierarchical
```

### 4.6 DFU Grain Mapping

The reconciled forecast is at **item + location** level. We need to map it to
**item + customer_group + location** (the DFU grain) for champion competition.

```python
# Option A: Proportional allocation (recommended)
# Allocate reconciled item-loc forecast to customer_groups by their historical share
shares = (
    fact_sales_monthly
    .groupby(["item_id", "loc", "customer_group"])["qty"]
    .sum()
    .pipe(lambda s: s / s.groupby(level=["item_id", "loc"]).transform("sum"))
)
reconciled_dfu = reconciled_item_loc * shares

# Option B: 1:1 mapping (if customer_group is always the same per item-loc)
# Check: SELECT item_id, loc, COUNT(DISTINCT customer_group)
#         FROM dim_sku GROUP BY item_id, loc HAVING COUNT(*) > 1
# If result is empty → all item-locs have exactly one customer_group → simple rename
```

### 4.7 Actuals Alignment

For accuracy computation, actuals come from `fact_customer_demand_monthly`
(aggregated to item-loc), **not** from `fact_sales_monthly`. This is
intentional — we measure accuracy against true demand, not constrained sales.

However, for champion comparison fairness (other models are measured against
`fact_sales_monthly` actuals), we attach `tothist_dmd` from `fact_sales_monthly`
so the champion selector compares all models on the same basis.

```python
# Attach actuals from fact_sales_monthly for apples-to-apples champion comparison
actuals = fact_sales_monthly.groupby(["item_id", "customer_group", "loc", "startdate"])["qty"].sum()
preds["tothist_dmd"] = preds.set_index([...]).index.map(actuals)
```

---

## 5. Performance & Scalability

### 5.1 Series Count Estimate

| Level | Count | Series Length | Total Data Points |
|-------|-------|--------------|-------------------|
| Customer series (bottom) | ~500K-2M | 3-28 months avg | ~5-20M |
| Aggregated item-loc (top) | ~50K-100K | 12-28 months avg | ~1-2M |

### 5.2 Bolt Inference Time

| Component | Series Count | Batch Size | Batches | Time (GPU) | Time (CPU) |
|-----------|-------------|------------|---------|------------|------------|
| Bottom-up (customer) | 1M | 2048 | ~500 | ~25 min | ~90 min |
| Top-down (item-loc) | 80K | 2048 | ~40 | ~3 min | ~10 min |
| Reconciliation | 80K item-locs | — | — | ~2 min | ~5 min |
| **Total per timeframe** | | | | **~30 min** | **~105 min** |
| **Total 10 timeframes** | | | | **~5 hours** | **~17 hours** |

### 5.3 Optimization Levers

| Optimization | Impact | Effort |
|-------------|--------|--------|
| Reduce timeframes to 5 (from 10) | 2x speedup | Config change |
| Increase batch_size to 4096 (GPU) | 1.5x speedup | Config change |
| Parallel timeframes (ProcessPool) | 2-3x speedup | Already built |
| Filter to top-100 customers per item-loc | 3-5x fewer series | Script logic |
| Use `chronos-bolt-small` instead of `base` | 2x faster, ~1% accuracy loss | Config change |
| Skip item-locs with < 5 active customers | 30-50% fewer series | Script logic |

**Recommended for Phase 1:** 5 timeframes + top-100 customers + batch 4096
→ ~1.5 hours GPU, ~6 hours CPU. Acceptable for weekly/monthly backtest cadence.

---

## 6. Configuration

### 6.1 `forecast_pipeline_config.yaml` — New Model Entry

```yaml
algorithms:
  bolt_hierarchical:
    model_type: foundation
    library: chronos_bolt
    cluster_strategy: global       # no per-cluster — hierarchy handles segmentation
    stages:
      tune: false                  # zero-shot, no tuning
      backtest: true
      compete: true
      forecast: true
      expert: false
    notes: "Chronos Bolt hierarchical: customer-level bottom-up + top-down reconciliation"
```

### 6.2 `algorithm_config.yaml` — Hyperparameters

```yaml
bolt_hierarchical:
  model_size: base               # amazon/chronos-bolt-base
  device: auto
  batch_size: 2048
  prediction_length: 6
  # Hierarchy config
  data_source: customer_demand   # fact_customer_demand_monthly (not fact_sales_monthly)
  demand_column: demand_qty      # use true demand, not constrained sales
  min_nonzero_months: 3          # skip customer series with < 3 non-zero months
  max_customers_per_item_loc: 100
  # Reconciliation
  reconciliation_method: weighted_average  # Phase 1; upgrade to mint_shrink in Phase 2
  bu_weight: 0.6                           # bottom-up weight (Phase 1 only)
  td_weight: 0.4                           # top-down weight (Phase 1 only)
```

### 6.3 Makefile Targets

```makefile
backtest-bolt-hier:
	$(UV) python -m scripts.run_backtest_bolt_hierarchical

backtest-load-bolt-hier:
	$(UV) python -m scripts.load_backtest_forecasts --model bolt_hierarchical --replace

backtest-bolt-hier-full: backtest-bolt-hier backtest-load-bolt-hier
```

---

## 7. Implementation Plan

### Phase 1: Core Pipeline (1.5 weeks)

- [ ] **DDL**: No new tables needed — output goes into existing `backtest_lag_archive`
- [ ] **Script**: `scripts/run_backtest_bolt_hierarchical.py`
  - Load customer demand data (bottom-level + aggregated top-level)
  - Series filtering (min history, max customers cap)
  - Bolt inference on both levels (reuse `_run_chronos_bolt()` from foundation_models.py)
  - Simple weighted reconciliation (alpha * BU + (1-alpha) * TD)
  - Map reconciled item-loc predictions to DFU grain (item + customer_group + loc)
  - Attach actuals from `fact_sales_monthly` for fair comparison
  - Write CSV in standard backtest format
- [ ] **Config**: Add `bolt_hierarchical` to `forecast_pipeline_config.yaml` and `algorithm_config.yaml`
- [ ] **Makefile**: Add `backtest-bolt-hier`, `backtest-load-bolt-hier` targets
- [ ] **Tests**: Unit tests for aggregation, reconciliation, DFU mapping logic

### Phase 2: MinTrace Reconciliation (3 days)

- [ ] Wire up `adv_algorithm_testing/reconciliation.py` skeleton
- [ ] Build summing matrix S from customer hierarchy
- [ ] Implement MinTrace shrinkage reconciliation
- [ ] Compare accuracy: weighted-average vs. MinTrace
- [ ] Switch config to `reconciliation_method: mint_shrink` if better

### Phase 3: Competition & Analysis (3 days)

- [ ] Run full backtest (5 timeframes, top-100 customers)
- [ ] Load predictions and run champion selection
- [ ] Analyze:
  - Overall WAPE vs. standard Bolt, LGBM, CatBoost
  - WAPE on stockout-affected DFUs (where `oos_rate > 5%`)
  - WAPE on high-concentration DFUs (`hhi > 0.5`)
  - WAPE on diversified DFUs (`n_customers > 20`)
  - Reconciliation weight distribution (how much BU vs TD per item-loc?)
- [ ] SHAP-equivalent analysis: which item-locs benefit most from hierarchy?

### Phase 4: Production Forecast (2 days)

- [ ] Extend `generate_production_forecasts.py` to handle `bolt_hierarchical` champion assignments
- [ ] Customer feature refresh → Bolt inference → reconcile → write fact_production_forecast
- [ ] Validate forecast coherence (sum of customer allocations = total)

---

## 8. Accuracy Measurement

### 8.1 Metrics (Same as All Models)

```
Accuracy = 100 - WAPE
WAPE = SUM(|Forecast - Actual|) / |SUM(Actual)| × 100
Bias = (SUM(Forecast) / SUM(Actual)) - 1
```

### 8.2 Apples-to-Apples Comparison

All models are compared against `fact_sales_monthly` actuals (not customer
demand actuals). This ensures the champion selector picks the best predictor
of what actually ships, even though `bolt_hierarchical` trains on true demand.

**This creates an interesting dynamic:**
- Standard models forecast constrained sales → measured against constrained sales
- Bolt hierarchical forecasts unconstrained demand → measured against constrained sales
- If there were no stockouts, both see the same actuals
- If there were stockouts, Bolt hierarchical forecasts *higher* (true demand) → appears as positive bias
- But for future months where inventory is replenished, the higher forecast is **correct**

This is exactly the behavior we want — the model that better anticipates true
demand will win over time as stockout patterns don't repeat identically.

### 8.3 Supplementary Metric: Demand Accuracy

In addition to standard WAPE (vs sales), report a supplementary metric:

```
Demand_WAPE = SUM(|Forecast - demand_qty_actual|) / |SUM(demand_qty_actual)| × 100
```

This measures how well the model predicts true demand, not just sales. Expected:
- `bolt_hierarchical` has better Demand_WAPE than standard models
- Standard models have slightly better Sales_WAPE on non-stockout items
- Overall, `bolt_hierarchical` wins on the items that matter most (the stockout ones)

---

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Customer series too sparse (< 3 months) | Medium | Filter threshold; these customers contribute via "other" bucket |
| Bottom-up sum overshoots (noisy customer forecasts) | Medium | Reconciliation pulls toward top-down; alpha = 0.6 BU is conservative |
| Slow inference at customer level | Medium | Batch 2048, top-100 customers, 5 timeframes, GPU |
| customer_group mapping ambiguous | Low | Check dim_sku for 1:1 vs 1:N item-loc → customer_group; proportional split if N>1 |
| MinTrace requires covariance estimation | Low | Use shrinkage estimator (robust); fall back to weighted average |
| Bolt median forecast biased for intermittent demand | Medium | Floor at 0; compare with Croston-adjusted BU for intermittent DFUs |
| No improvement over standard Bolt | Low | Zero regression risk — if it doesn't win, champion selector ignores it |

---

## 10. Success Criteria

| Metric | Target |
|--------|--------|
| Overall WAPE improvement vs standard Bolt | >= 1.5 percentage points |
| Stockout-affected DFU WAPE improvement | >= 3 percentage points |
| High-concentration DFU WAPE improvement | >= 2 percentage points |
| Champion win rate (bolt_hierarchical selected) | >= 15% of DFUs |
| Demand_WAPE (vs true demand) improvement | >= 3 percentage points vs any model |
| Backtest runtime (5 timeframes, GPU) | < 2 hours |
| No WAPE regression on any cluster vs standard Bolt | 0 clusters worse |

---

## 11. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                 fact_customer_demand_monthly                      │
│            (item_id × customer_no × location_id × month)         │
└───────────────────┬──────────────────────┬──────────────────────┘
                    │                      │
          ┌─────────▼─────────┐  ┌─────────▼─────────┐
          │  Customer Series   │  │  Aggregate to      │
          │  (bottom level)    │  │  Item × Location   │
          │  ~500K-1M series   │  │  (top level)       │
          │  demand_qty        │  │  ~80K series       │
          └─────────┬─────────┘  └─────────┬──────────┘
                    │                      │
          ┌─────────▼─────────┐  ┌─────────▼──────────┐
          │  Chronos Bolt      │  │  Chronos Bolt       │
          │  inference         │  │  inference           │
          │  (batch 2048)      │  │  (batch 2048)        │
          └─────────┬─────────┘  └─────────┬──────────┘
                    │                      │
          ┌─────────▼─────────┐            │
          │  SUM to item×loc   │            │
          │  (bottom-up agg)   │            │
          └─────────┬─────────┘            │
                    │                      │
                    │    BU_forecast        │   TD_forecast
                    │                      │
          ┌─────────▼──────────────────────▼──────────┐
          │           RECONCILIATION                    │
          │  Phase 1: α·BU + (1-α)·TD  (α=0.6)        │
          │  Phase 2: MinTrace shrinkage               │
          │  Output: reconciled item × loc forecast     │
          └─────────────────────┬──────────────────────┘
                                │
                  ┌─────────────▼──────────────┐
                  │  MAP TO DFU GRAIN            │
                  │  item×loc → item×cg×loc      │
                  │  (proportional allocation     │
                  │   by historical sales share)  │
                  └─────────────┬──────────────┘
                                │
                  ┌─────────────▼──────────────┐
                  │  STANDARD BACKTEST OUTPUT    │
                  │  forecast_ck, model_id =     │
                  │  "bolt_hierarchical"          │
                  │  Same CSV schema as all       │
                  │  other models                 │
                  └─────────────┬──────────────┘
                                │
                  ┌─────────────▼──────────────┐
                  │  CHAMPION SELECTION          │
                  │  Competes head-to-head       │
                  │  with lgbm, catboost,        │
                  │  standard bolt, etc.         │
                  └────────────────────────────┘
```

---

## 12. Comparison with Other Approaches

| Approach | This PRD | Feature Enrichment (PRD 02-08) | Chronos 2 Enriched |
|----------|----------|-------------------------------|-------------------|
| Uses customer data | Yes — at inference level | Yes — as features | Yes — as covariates |
| Model change | New backtest script | Same models, new features | Extend covariate list |
| Signal type | True demand (demand_qty) | Derived features (HHI, churn) | Derived features |
| Handles stockouts | Yes — forecasts unconstrained demand | Partially — oos_rate feature | Partially |
| Reconciliation | Yes — BU + TD combined | No | No |
| Effort | Medium (new script) | Low (feature join) | Low (config change) |
| Max potential lift | High (new signal source) | Medium (feature enrichment) | Medium |

**Recommendation:** Implement all three. They are complementary:
- `bolt_hierarchical` wins on stockout-affected and high-concentration DFUs
- `lgbm_cust_enriched` wins on DFUs where customer features add signal but hierarchy is overkill
- `chronos2_enriched` (with customer covariates) wins on DFUs where foundation model + covariates is sufficient

The champion selector picks the best per DFU. More competitors = better portfolio accuracy.

---

## 13. Dependency on Feature Enrichment PRD (02-08)

This PRD is **independent** of PRD 02-08 (Customer-Enriched Features). They can be
implemented in parallel:

- PRD 02-08 creates `customer_features_monthly` (derived features) → used by tree models
- This PRD uses `fact_customer_demand_monthly` directly (raw demand series) → used by Bolt

However, if both are implemented, the feature generation script from 02-08 can
share the customer-level aggregation logic. Consider extracting shared SQL
queries into `common/core/customer_aggregations.py`.
