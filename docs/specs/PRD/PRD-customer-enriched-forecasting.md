# PRD: Customer-Enriched Demand Forecasting

**Status:** Draft
**Author:** Auto-generated
**Date:** 2026-04-03
**Domain:** Forecasting / ML Pipeline
**Spec ID:** 02-08

---

## 1. Problem Statement

Today's forecasting pipeline operates on `fact_sales_monthly` at the
**item + customer_group + location** grain. It has 50+ features — lags,
rolling statistics, calendar, Fourier, demand profiles, Croston decomposition,
cluster aggregates — but **zero features derived from customer-level data**.

`fact_customer_demand_monthly` stores individual-customer demand at
**item + customer_no + location + month** grain, with three critical signals
unavailable in the aggregated sales table:

| Signal | Why It Matters |
|--------|---------------|
| `demand_qty` (true demand) | Sales are _constrained_ by inventory; demand is what customers actually wanted |
| `oos_qty` (out-of-stock) | Reveals latent/suppressed demand invisible in sales history |
| Customer identity | Concentration risk, churn, acquisition — leading indicators of demand shifts |

An item-location where 80% of demand comes from one customer behaves
fundamentally differently from one with 50 diversified customers. Current
models cannot distinguish these — they see the same aggregated sales curve.

---

## 2. Recommended Approach: Customer-Enriched Gradient Boosted Trees

### Why Not Bottom-Up Forecasting?

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Bottom-up** (forecast per customer, aggregate) | Granular, captures each customer | Sparse series, combinatorial explosion, slow | Too noisy at customer grain |
| **Hierarchical reconciliation** (MinT/ERM) | Theoretically optimal | Complex, research-grade, needs forecasts at every level | Over-engineered for the lift available |
| **Customer embedding model** (neural) | Learns customer representations | Needs large data, opaque, doesn't fit existing pipeline | Future enhancement |
| **Customer-enriched features** (recommended) | Plugs into existing tree models, SHAP-interpretable, minimal arch change | Requires feature engineering effort | Best ROI |

### The Algorithm

**Not a new model architecture — a new feature set added to the existing tree models.**

We introduce `lgbm_cust_enriched`, `catboost_cust_enriched`, and
`xgboost_cust_enriched` as new model variants in the competition roster.
They use the same LGBM/CatBoost/XGBoost under the hood but with ~25
additional customer-derived features. This means:

- They compete head-to-head with the current `lgbm_cluster` etc. in backtests
- The champion selector picks the winner per DFU — if customer features help, they win; if not, the old model wins
- No risk of regression — it's additive to the competition pool
- SHAP analysis reveals exactly which customer features drive lift

---

## 3. Customer-Derived Feature Specification

All features are pre-computed from `fact_customer_demand_monthly` and
`dim_customer`, aggregated to the **item + location + month** grain to match
the forecasting DFU level.

### 3.1 Customer Concentration Features

These measure how diversified vs. concentrated the customer base is for
a given item-location. High concentration = high forecast risk.

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `n_active_cust` | `COUNT(DISTINCT customer_no)` over trailing 3 months | Customer breadth |
| `n_active_cust_6m` | Same, trailing 6 months | Longer-term breadth |
| `hhi_demand` | `SUM(share_i^2)` where share_i = customer_i demand / total | Herfindahl index: 0=diverse, 1=monopoly |
| `top1_cust_share` | demand of largest customer / total demand (3m window) | Single-customer dependency |
| `top3_cust_share` | demand of top 3 customers / total demand (3m window) | Top-customer dependency |
| `cust_gini` | Gini coefficient of customer demand distribution | Inequality of demand across customers |

### 3.2 Customer Dynamics Features

These capture customer acquisition, churn, and mix shifts — **leading
indicators** of demand inflection points.

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `new_cust_demand_share` | demand from customers with first order in last 3m / total | Growth from acquisition |
| `churned_cust_demand_share` | demand from customers who ordered in months [-6,-3] but not [-3,0] / total[-6,-3] | Demand at risk from churn |
| `cust_count_mom` | (n_active_cust_t - n_active_cust_{t-1}) / n_active_cust_{t-1} | Customer base growth rate |
| `cust_retention_rate` | customers ordering in both t and t-1 / customers ordering in t-1 | Loyalty signal |
| `cust_tenure_mean` | mean months since first order across active customers | Maturity of customer base |

### 3.3 True Demand Features

These exploit the `demand_qty` vs `sales_qty` vs `oos_qty` split — the
single most valuable signal unavailable in `fact_sales_monthly`.

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `true_demand_ratio` | `SUM(demand_qty) / NULLIF(SUM(sales_qty), 0)` trailing 3m | >1.0 means demand exceeds sales (stockouts) |
| `oos_rate` | `SUM(oos_qty) / NULLIF(SUM(demand_qty), 0)` trailing 3m | Stockout severity |
| `oos_cust_pct` | customers with any oos_qty > 0 / total active customers | Breadth of stockout impact |
| `demand_sales_gap_3m` | `SUM(demand_qty - sales_qty)` trailing 3m | Absolute suppressed demand |
| `oos_trend` | (oos_rate_3m - oos_rate_6m) / oos_rate_6m | Is stockout situation improving or worsening? |
| `demand_qty_lag1` | `SUM(demand_qty)` for t-1 month | True demand lag (vs constrained sales lag) |
| `demand_qty_lag3_mean` | rolling mean of `SUM(demand_qty)` over 3m | Smoothed true demand |

### 3.4 Customer Segment Mix Features

These capture how the channel/store-type mix is shifting. A change in
channel mix often precedes a volume change.

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `channel_entropy` | Shannon entropy of demand across `rpt_channel_desc` | Diversity of channels |
| `dominant_channel_share` | demand of largest channel / total | Channel concentration |
| `channel_mix_shift` | L1 distance between current 3m channel share vector and prior 3m | How much is channel mix changing? |
| `on_premise_share` | On-Premise demand / total demand | On vs Off premise balance |

### 3.5 Cross-Customer Demand Signals

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `cust_demand_cv_mean` | mean CV of individual customer demand series | Are individual customers volatile? |
| `cust_demand_sync` | average pairwise correlation of top-5 customer series | Do customers move together (macro) or independently? |
| `max_cust_share_delta` | max absolute MoM change in any single customer's share | Sudden customer-level shifts |

---

## 4. Feature Engineering Pipeline

### 4.1 SQL Pre-Aggregation (New Script)

**File:** `scripts/ml/generate_customer_features.py`

This script runs before backtesting and produces a feature table:

```sql
-- Materialized view or temp table: customer_features_monthly
-- Grain: item_id + location_id + startdate (matches fact_sales_monthly)

SELECT
    f.item_id,
    f.location_id AS loc,
    f.startdate,
    -- Concentration
    COUNT(DISTINCT f.customer_no) AS n_active_cust,
    SUM(f.demand_qty) AS total_demand,
    MAX(cust_demand) / NULLIF(SUM(f.demand_qty), 0) AS top1_cust_share,
    ...
    -- True demand
    SUM(f.demand_qty) / NULLIF(SUM(f.sales_qty), 0) AS true_demand_ratio,
    SUM(f.oos_qty) / NULLIF(SUM(f.demand_qty), 0) AS oos_rate,
    ...
FROM fact_customer_demand_monthly f
JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
GROUP BY f.item_id, f.location_id, f.startdate
```

**Output:** `customer_features_monthly` table/CSV at item + loc + month grain.

### 4.2 Feature Join in Backtest

**File:** `common/ml/feature_engineering.py` (modify existing)

After standard feature engineering, left-join customer features:

```python
df = df.merge(
    customer_features,
    on=["item_id", "loc", "startdate"],
    how="left",
)
# Fill NaN for item-locs with no customer demand data
df[CUSTOMER_FEATURE_COLS].fillna(0, inplace=True)
```

### 4.3 Lag Computation for Customer Features

Customer features need their own lags to prevent leakage:

- `n_active_cust_lag1`, `hhi_demand_lag1`, `true_demand_ratio_lag1`, etc.
- Rolling windows (3m, 6m) computed from lagged customer features
- Same 1-month embargo as other features

---

## 5. Model Configuration

### 5.1 New Model Entries in `forecast_pipeline_config.yaml`

```yaml
algorithms:
  # ... existing models ...

  lgbm_cust_enriched:
    model_type: tree
    library: lightgbm
    cluster_strategy: per_cluster
    stages:
      tune: true
      backtest: true
      compete: true
      forecast: true
      expert: false
    notes: "LGBM with customer-derived features"

  catboost_cust_enriched:
    model_type: tree
    library: catboost
    cluster_strategy: per_cluster
    stages:
      tune: true
      backtest: true
      compete: true
      forecast: true
      expert: false
    notes: "CatBoost with customer-derived features"

  xgboost_cust_enriched:
    model_type: tree
    library: xgboost
    cluster_strategy: per_cluster
    stages:
      tune: true
      backtest: true
      compete: true
      forecast: true
      expert: false
    notes: "XGBoost with customer-derived features"
```

### 5.2 Hyperparameters in `forecast_pipeline_config.yaml`

Start with the same hyperparameters as the non-enriched variants. The
tuning pipeline will optimize them for the larger feature set.

```yaml
lgbm_cust_enriched:
  n_estimators: 1500
  learning_rate: 0.02
  num_leaves: 127
  min_child_samples: 40
  reg_lambda: 1.0
  reg_alpha: 0.1
  subsample: 0.8
  colsample_bytree: 0.7    # slightly lower — more features need regularization
  shap_select: true
  shap_threshold: 0.95
```

### 5.3 Feature Column Registry

**File:** `common/core/constants.py` — add new constant:

```python
CUSTOMER_FEATURE_COLS = [
    # Concentration
    "n_active_cust", "n_active_cust_6m", "hhi_demand",
    "top1_cust_share", "top3_cust_share", "cust_gini",
    # Dynamics
    "new_cust_demand_share", "churned_cust_demand_share",
    "cust_count_mom", "cust_retention_rate", "cust_tenure_mean",
    # True demand
    "true_demand_ratio", "oos_rate", "oos_cust_pct",
    "demand_sales_gap_3m", "oos_trend",
    "demand_qty_lag1", "demand_qty_lag3_mean",
    # Segment mix
    "channel_entropy", "dominant_channel_share",
    "channel_mix_shift", "on_premise_share",
    # Cross-customer
    "cust_demand_cv_mean", "cust_demand_sync", "max_cust_share_delta",
]
```

### 5.4 Protected Features

Add to SHAP-protected list (never dropped even if low importance):

```python
PROTECTED_CUSTOMER_FEATURES = [
    "true_demand_ratio",   # most valuable — captures suppressed demand
    "n_active_cust",       # fundamental customer breadth signal
    "hhi_demand",          # concentration risk
]
```

---

## 6. Expected Lift Analysis

### 6.1 Where Customer Features Add Value

| DFU Profile | Current Weakness | Customer Feature Fix | Expected Lift |
|-------------|-----------------|---------------------|---------------|
| **High-concentration** (top1 > 60%) | Model treats it like diversified demand | `hhi_demand`, `top1_cust_share` flag risk | 3-8% WAPE reduction |
| **Stockout-affected** (oos_rate > 5%) | Sales understate true demand; lags are biased low | `true_demand_ratio`, `demand_qty_lag1` correct the signal | 5-15% WAPE reduction |
| **Customer churn inflection** | Model misses demand cliff from lost customer | `churned_cust_demand_share`, `cust_retention_rate` | 2-5% WAPE reduction |
| **Acquisition growth** | Model slow to react to demand ramp | `new_cust_demand_share`, `cust_count_mom` | 2-5% WAPE reduction |
| **Channel shift** | Model misses mix-driven volume change | `channel_mix_shift`, `channel_entropy` | 1-3% WAPE reduction |

### 6.2 Where They Won't Help

- Items with no customer-level data (missing from `fact_customer_demand_monthly`)
- Stable, diversified items with no stockouts (existing features already sufficient)
- Foundation model domains (Chronos doesn't use tabular features)

### 6.3 Aggregate Expected Impact

Based on industry benchmarks for customer-enriched demand sensing:

| Metric | Current Baseline | Expected with Customer Features |
|--------|-----------------|--------------------------------|
| Overall WAPE | ~25% (champion) | ~22-23% (2-3pp improvement) |
| Stockout-affected DFUs WAPE | ~35% | ~28-30% (5-7pp improvement) |
| High-concentration DFUs WAPE | ~30% | ~25-27% (3-5pp improvement) |
| Champion win rate for enriched models | 0% (don't exist yet) | 30-45% of DFUs |

---

## 7. Architecture: How It Fits

```
                     EXISTING PIPELINE                    NEW ADDITION
                     ─────────────────                    ────────────
fact_sales_monthly ──→ feature_engineering ──→ backtest ──→ champion
                         │                        ↑           ↑
                         │   feature_cols          │           │
                         │   + CUSTOMER_FEATURES   │           │
                         │                         │           │
fact_customer_demand ──→ generate_customer     ──→ │           │
      monthly            features.py               │           │
         │                   │                     │           │
    dim_customer             ▼                     │           │
                     customer_features_        lgbm_cust_enriched
                     monthly (table)           catboost_cust_enriched
                                               xgboost_cust_enriched
                                                   │
                                                   ▼
                                            compete alongside
                                            existing models
                                                   │
                                                   ▼
                                            champion_selector
                                            picks best per DFU
```

**Key property:** The enriched models compete in the same arena as existing
models. The champion selector handles the "is it better?" question per DFU.
No manual threshold or cutover needed.

---

## 8. Implementation Plan

### Phase 1: Feature Engineering (1 week)

- [ ] Create `scripts/ml/generate_customer_features.py`
  - SQL aggregation from `fact_customer_demand_monthly` + `dim_customer`
  - Output: `customer_features_monthly` table (item_id, loc, startdate, 25 features)
  - DDL: `sql/115_create_customer_features_monthly.sql`
- [ ] Add `CUSTOMER_FEATURE_COLS` to `common/core/constants.py`
- [ ] Modify `common/ml/feature_engineering.py` to optionally merge customer features
  - Feature flag: `use_customer_features: true/false` in model config
  - Left join on item_id + loc + startdate
  - NaN fill with 0 for item-locs without customer data
- [ ] Add lag computation for customer features (1-month lag to prevent leakage)
- [ ] Add Makefile target: `make customer-features`
- [ ] Tests: unit tests for feature computation, integration test with sample data

### Phase 2: Model Registration & Backtest (1 week)

- [ ] Add 3 new model entries to `config/forecasting/forecast_pipeline_config.yaml`
- [ ] Add hyperparameter blocks to `config/forecasting/forecast_pipeline_config.yaml`
- [ ] Add SHAP-protected customer features to constants
- [ ] Modify `model_registry.py` `_is_customer_enriched()` check
  - If model_id contains `_cust_enriched`, include customer feature columns
- [ ] Run backtests: `make backtest-lgbm-cust`, `make backtest-catboost-cust`, etc.
- [ ] Add Makefile targets for enriched model backtests
- [ ] Load backtest predictions into DB

### Phase 3: Competition & Champion (3 days)

- [ ] Add enriched models to competition roster
- [ ] Run champion selection: `make champion-all`
- [ ] Analyze results:
  - Overall WAPE vs. baseline models
  - WAPE by demand profile (high-concentration, stockout-affected, etc.)
  - SHAP importance of customer features across clusters
- [ ] Tune if needed: adjust regularization, SHAP threshold

### Phase 4: Production Forecast (2 days)

- [ ] Ensure `generate_production_forecasts.py` handles enriched model artifacts
- [ ] Add customer feature refresh to the forecast pipeline dependency chain
  - `customer_features_monthly` must be refreshed before `forecast-generate`
- [ ] Update `Makefile` pipeline targets
- [ ] Run `make forecast-generate` and validate outputs

### Phase 5: Monitoring & Iteration (Ongoing)

- [ ] Add customer feature importance to SHAP dashboard
- [ ] Monitor enriched vs. non-enriched champion win rates
- [ ] Track WAPE improvement by customer concentration cohort
- [ ] Iterate: add/remove features based on SHAP analysis

---

## 9. Data Dependencies

| Source | Freshness Requirement | Pipeline Step |
|--------|----------------------|---------------|
| `fact_customer_demand_monthly` | Must be loaded before feature generation | `make load-customer-demand` |
| `dim_customer` | Must be loaded before feature generation | `make load-all` |
| `customer_features_monthly` | Must be refreshed before backtest/forecast | `make customer-features` |
| Backtest predictions | Must include enriched models | `make backtest-all` |

### Pipeline Order

```
make load-customer-demand
  → make customer-features
    → make backtest-all       (includes enriched models)
      → make champion-all     (competes enriched vs. standard)
        → make forecast-generate  (uses champion assignments)
```

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Customer data missing for some item-locs | Medium | Left join + NaN fill with 0; enriched model competes but won't win for these DFUs |
| Feature leakage from customer features | High | Strict 1-month lag on all customer features; same embargo as existing features |
| Overfitting on 25 new features | Medium | SHAP selection (threshold 0.95) auto-drops low-value features; colsample_bytree = 0.7 |
| Computation time for feature generation | Low | SQL aggregation with partition pruning; customer features table is small (~item x loc x month) |
| Customer data quality (nulls, duplicates) | Medium | `fact_customer_demand_monthly` already has quality constraints (non-negative, first-of-month) |
| Enriched models never win | Low | No harm — they just don't get selected; zero regression risk |

---

## 11. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Overall WAPE improvement | >= 1.5 percentage points | Backtest comparison across all DFUs |
| Stockout-affected DFU WAPE improvement | >= 4 percentage points | Filter DFUs where oos_rate > 5% |
| Enriched model champion win rate | >= 25% of DFUs | Champion selection report |
| No WAPE regression on any cluster | 0 clusters worse | Per-cluster backtest comparison |
| SHAP: >= 3 customer features in top-20 | Yes | SHAP summary across clusters |
| Feature generation runtime | < 10 minutes | Profiled section timing |
| End-to-end pipeline still < 6 hours | Yes | `make setup-backtest` timing |

---

## 12. Future Extensions

Once the feature-enriched approach proves value, consider:

1. **Customer-level embedding model**: Train a small neural network on customer
   demand sequences, extract embeddings, use as features in tree models.
   More powerful than hand-crafted features but requires more infrastructure.

2. **Hierarchical reconciliation**: After enriched models are stable, add
   MinT/WLS reconciliation between customer-level and item-loc forecasts
   for items where bottom-up adds value.

3. **Customer lifetime value weighting**: Weight forecast accuracy by customer
   value — a 5% error on your largest customer matters more than 20% error
   on your smallest.

4. **Real-time demand sensing**: If customer order data becomes available
   intra-month, use it as a nowcasting signal to adjust the monthly forecast.

---

## 13. Comparison with Alternative Algorithms

### Why Not Deep Learning (TFT, DeepAR, etc.)?

| Factor | Tree + Customer Features | Temporal Fusion Transformer |
|--------|-------------------------|---------------------------|
| Training data needed | Works with 12+ months | Needs 100+ series minimum |
| Interpretability | Full SHAP analysis | Attention weights (less actionable) |
| Cold-start handling | Features work even with short history | Needs warm-up period |
| Infrastructure | Existing pipeline, same .pkl artifacts | Needs GPU, new inference pipeline |
| Iteration speed | Add/remove features in minutes | Retrain entire model (hours) |
| Integration effort | 2-3 days to wire into competition | 2-3 weeks for new architecture |
| Expected WAPE lift | 2-3pp from features alone | 3-5pp but with 10x complexity |

**Verdict:** Feature enrichment is the right first step. If it proves the
value of customer signals (which SHAP will confirm), a TFT-based approach
becomes the logical Phase 2 — but only after validating that customer data
actually moves the needle.

### Why Feature Enrichment > Bottom-Up?

Bottom-up forecasting (forecast per customer, aggregate) fails here because:

1. **Sparsity**: Most customer x item x location series have < 6 data points
2. **Noise**: Individual customer ordering is lumpy and unpredictable
3. **Combinatorial explosion**: 50K customers x 5K items x 50 locations = 12.5B potential series
4. **Signal loss**: The signal is in the _patterns across customers_ (concentration, churn, mix), not in individual customer forecasts

The enriched-feature approach extracts these cross-customer patterns as
compact, information-dense features and feeds them to models that already
know how to forecast at the item-location level. Best of both worlds.
