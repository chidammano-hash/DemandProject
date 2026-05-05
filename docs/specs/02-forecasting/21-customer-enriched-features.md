# Customer-Enriched Tree Model Forecasting

> Adds 34 customer-derived features (unconstrained demand, OOS volume, customer concentration) to LGBM, CatBoost, and XGBoost as new model variants that compete in champion selection alongside existing models trained only on inventory-constrained sales.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `scripts/compute_customer_features.py`, `scripts/run_backtest_cust_enriched.py`, `config/forecasting/forecast_pipeline_config.yaml` |

---

## 1. Problem Statement

The three tree-based forecasting models (LGBM, CatBoost, XGBoost) use 50+
features derived entirely from `fact_sales_monthly` — a signal that is
**constrained by inventory**. When stockouts occur, `qty = 0` but true demand
was non-zero. Every lag, rolling mean, and derived feature trained on this
biased signal systematically under-forecasts stockout-affected DFUs.

`fact_customer_demand_monthly` contains three signals invisible to current
models:

| Signal | Column | Why It Matters |
|--------|--------|---------------|
| True demand | `demand_qty` | What customers ordered, not what we could ship |
| Stockout volume | `oos_qty` | Suppressed demand invisible in sales |
| Customer identity | `customer_no` | Concentration risk, churn, acquisition patterns |

This spec adds **34 customer-derived features** to the existing tree models
as new model variants (`lgbm_cust_enriched`, `catboost_cust_enriched`,
`xgboost_cust_enriched`) that compete head-to-head with current models
in champion selection.

---

## 2. Architecture

### 2.1 No New Model — New Features on Existing Models

The enriched variants use the **same LGBM/CatBoost/XGBoost** under the hood.
The only change is a wider feature matrix. This means:

- Same `fit_model()` from `model_registry.py` — no code changes
- Same per-cluster training strategy
- Same multi-stage feature selection (auto-drops redundant/low-value features; see [spec 23](23-feature-selection-pipeline.md))
- Same champion selector picks the winner per DFU
- Zero regression risk — enriched models just join the competition pool

### 2.2 Data Flow

```
┌──────────────────────────┐    ┌──────────────────────────┐
│  fact_sales_monthly       │    │  fact_customer_demand     │
│  (item + cg + loc + month)│    │  (item + cust + loc + mo) │
└────────────┬─────────────┘    └────────────┬─────────────┘
             │                               │
    ┌────────▼────────┐             ┌────────▼────────┐
    │  Standard        │             │  generate_       │
    │  feature_engin.  │             │  customer_       │
    │  (50+ features)  │             │  features.py     │
    └────────┬────────┘             │  (34 features)   │
             │                       └────────┬────────┘
             │                                │
    ┌────────▼────────────────────────────────▼────────┐
    │          LEFT JOIN on (item_id, loc, startdate)   │
    │          fill NaN with 0                          │
    │          → 75+ feature matrix                     │
    └────────────────────────┬─────────────────────────┘
                             │
                ┌────────────▼────────────┐
                │  Enriched tree model     │
                │  (same fit_model() call) │
                │  SHAP auto-selects best  │
                │  features per timeframe  │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │  Champion selector sees  │
                │  "lgbm_cust_enriched"    │
                │  alongside "lgbm_cluster"│
                │  → picks best per DFU    │
                └─────────────────────────┘
```

---

## 3. Customer Feature Specification

All features are pre-computed from `fact_customer_demand_monthly` joined
with `dim_customer`, aggregated to **item_id + location_id + startdate**
grain, then mapped to DFU grain via `dim_sku`.

### 3.1 Concentration Features (6)

Measure how diversified vs. concentrated the customer base is.

| Feature | SQL | Window | Intuition |
|---------|-----|--------|-----------|
| `n_active_cust` | `COUNT(DISTINCT customer_no)` | 3m | Customer breadth |
| `n_active_cust_6m` | `COUNT(DISTINCT customer_no)` | 6m | Longer-term breadth |
| `hhi_demand` | `SUM(share_i^2)` where `share_i = cust_demand / total` | 3m | 0=diverse, 1=monopoly |
| `top1_cust_share` | `MAX(cust_demand) / total_demand` | 3m | Single-customer dependency |
| `top3_cust_share` | `SUM(top-3 cust_demand) / total_demand` | 3m | Top-customer dependency |
| `cust_gini` | Gini coefficient of customer demand distribution | 3m | Inequality index |

### 3.2 Customer Dynamics Features (5)

Leading indicators of demand inflection points.

| Feature | SQL | Window | Intuition |
|---------|-----|--------|-----------|
| `new_cust_demand_share` | demand from customers with first order in window / total | 3m | Growth from acquisition |
| `churned_cust_demand_share` | demand from custs who ordered in [-6,-3] but not [-3,0] / total[-6,-3] | 6m/3m | Demand at risk |
| `cust_count_mom` | `(n_active_t - n_active_{t-1}) / n_active_{t-1}` | 1m | Customer growth rate |
| `cust_retention_rate` | custs ordering in both t and t-1 / custs in t-1 | 1m | Loyalty signal |
| `cust_tenure_mean` | mean months since first order across active custs | 3m | Base maturity |

### 3.3 True Demand Features (7)

The single most valuable feature group — exploits `demand_qty` vs `sales_qty`.

| Feature | SQL | Window | Intuition |
|---------|-----|--------|-----------|
| `true_demand_ratio` | `SUM(demand_qty) / NULLIF(SUM(sales_qty), 0)` | 3m | >1.0 = stockouts |
| `oos_rate` | `SUM(oos_qty) / NULLIF(SUM(demand_qty), 0)` | 3m | Stockout severity |
| `oos_cust_pct` | custs with oos > 0 / total active custs | 3m | Stockout breadth |
| `demand_sales_gap_3m` | `SUM(demand_qty - sales_qty)` | 3m | Absolute suppressed demand |
| `oos_trend` | `(oos_rate_3m - oos_rate_6m) / oos_rate_6m` | 6m | Improving or worsening? |
| `demand_qty_lag1` | `SUM(demand_qty)` for t-1 month | 1m | True demand lag |
| `demand_qty_lag3_mean` | rolling mean of `SUM(demand_qty)` over 3m | 3m | Smoothed true demand |

### 3.4 Channel Mix Features (4)

Channel composition changes precede volume changes.

| Feature | SQL | Window | Intuition |
|---------|-----|--------|-----------|
| `channel_entropy` | `-SUM(p_i * ln(p_i))` across `rpt_channel_desc` | 3m | Diversity of channels |
| `dominant_channel_share` | largest channel demand / total | 3m | Channel concentration |
| `channel_mix_shift` | L1 distance: current 3m channel vector vs prior 3m | 6m | How fast is mix changing? |
| `on_premise_share` | On-Premise demand / total | 3m | On vs Off balance |

### 3.5 Cross-Customer Signals (3)

| Feature | SQL | Window | Intuition |
|---------|-----|--------|-----------|
| `cust_demand_cv_mean` | mean CV of top-10 individual customer series | 6m | Are customers volatile? |
| `cust_demand_sync` | avg pairwise corr of top-5 customer series | 6m | Macro-driven or independent? |
| `max_cust_share_delta` | max |MoM change| in any single customer's share | 1m | Sudden customer shifts |

### 3.6 Customer Attribute Mix Features (9)

Derived from `dim_customer` attributes joined to demand data, capturing
the structural composition of the customer base for each item-location.

| Feature | SQL | Window | Intuition |
|---------|-----|--------|-----------|
| `store_type_entropy` | `-SUM(p_i * ln(p_i))` across `store_type_desc` | 3m | Shannon entropy of demand across store types |
| `dominant_store_type_share` | `MAX(store_type_demand) / total_demand` | 3m | Largest store type / total demand |
| `chain_ratio` | `SUM(chain_demand) / total_demand` | 3m | % demand from chain customers (vs independents) |
| `top_chain_share` | `MAX(chain_demand) / total_demand` | 3m | Largest chain's share of demand |
| `sub_channel_entropy` | `-SUM(p_i * ln(p_i))` across `rpt_sub_channel_desc` | 3m | Shannon entropy across sub-channels |
| `active_cust_pct` | `COUNT(active) / COUNT(*)` across custs with demand | 3m | % of customers with active status |
| `avg_delivery_freq` | `MEAN(freq_score)` where D=5, W=4, M=2, Q=1 | 3m | Mean delivery frequency score |
| `on_premise_acct_share` | `SUM(on_premise_demand) / total_demand` by `premise_code` | 3m | % demand from on-premise premise_code accounts |
| `premise_diversity` | `COUNT(DISTINCT premise_code) / COUNT(DISTINCT customer_no)` | 3m | Ratio of distinct premise codes to active customers |

---

## 4. Feature Generation Pipeline

### 4.1 New Script: `scripts/ml/generate_customer_features.py`

**Purpose:** Pre-compute the 34 customer features at item × location × month
grain and store them in a table for the backtest pipeline to join.

**Input:** `fact_customer_demand_monthly` + `dim_customer`

**Output:** `customer_features_monthly` table

#### SQL Schema (New DDL: `sql/116_create_customer_features_monthly.sql`)

```sql
CREATE TABLE IF NOT EXISTS customer_features_monthly (
    item_id         TEXT NOT NULL,
    loc             TEXT NOT NULL,
    startdate       DATE NOT NULL,
    -- Concentration (6)
    n_active_cust           REAL DEFAULT 0,
    n_active_cust_6m        REAL DEFAULT 0,
    hhi_demand              REAL DEFAULT 0,
    top1_cust_share         REAL DEFAULT 0,
    top3_cust_share         REAL DEFAULT 0,
    cust_gini               REAL DEFAULT 0,
    -- Dynamics (5)
    new_cust_demand_share   REAL DEFAULT 0,
    churned_cust_demand_share REAL DEFAULT 0,
    cust_count_mom          REAL DEFAULT 0,
    cust_retention_rate     REAL DEFAULT 0,
    cust_tenure_mean        REAL DEFAULT 0,
    -- True Demand (7)
    true_demand_ratio       REAL DEFAULT 0,
    oos_rate                REAL DEFAULT 0,
    oos_cust_pct            REAL DEFAULT 0,
    demand_sales_gap_3m     REAL DEFAULT 0,
    oos_trend               REAL DEFAULT 0,
    demand_qty_lag1         REAL DEFAULT 0,
    demand_qty_lag3_mean    REAL DEFAULT 0,
    -- Channel Mix (4)
    channel_entropy         REAL DEFAULT 0,
    dominant_channel_share  REAL DEFAULT 0,
    channel_mix_shift       REAL DEFAULT 0,
    on_premise_share        REAL DEFAULT 0,
    -- Cross-Customer (3)
    cust_demand_cv_mean     REAL DEFAULT 0,
    cust_demand_sync        REAL DEFAULT 0,
    max_cust_share_delta    REAL DEFAULT 0,
    -- Customer Attribute Mix (9)
    store_type_entropy      REAL DEFAULT 0,
    dominant_store_type_share REAL DEFAULT 0,
    chain_ratio             REAL DEFAULT 0,
    top_chain_share         REAL DEFAULT 0,
    sub_channel_entropy     REAL DEFAULT 0,
    active_cust_pct         REAL DEFAULT 0,
    avg_delivery_freq       REAL DEFAULT 0,
    on_premise_acct_share   REAL DEFAULT 0,
    premise_diversity       REAL DEFAULT 0,
    -- Metadata
    load_ts TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT pk_cust_feat_monthly PRIMARY KEY (item_id, loc, startdate)
);

CREATE INDEX idx_cust_feat_item_loc ON customer_features_monthly (item_id, loc);
CREATE INDEX idx_cust_feat_startdate ON customer_features_monthly (startdate);
```

#### Core Aggregation SQL

The script executes a multi-CTE query. Simplified high-value subset:

```sql
WITH base AS (
    SELECT f.item_id, f.location_id AS loc, f.startdate,
           f.customer_no, f.demand_qty, f.sales_qty, f.oos_qty,
           c.rpt_channel_desc
    FROM fact_customer_demand_monthly f
    JOIN dim_customer c ON c.customer_no = f.customer_no AND c.site = f.site
    WHERE f.startdate >= %(cutoff)s
),
monthly_agg AS (
    SELECT item_id, loc, startdate,
           COUNT(DISTINCT customer_no) AS n_active_cust,
           SUM(demand_qty) AS total_demand,
           SUM(sales_qty) AS total_sales,
           SUM(oos_qty) AS total_oos
    FROM base GROUP BY item_id, loc, startdate
),
cust_shares AS (
    SELECT b.item_id, b.loc, b.startdate, b.customer_no,
           b.demand_qty / NULLIF(m.total_demand, 0) AS demand_share
    FROM base b
    JOIN monthly_agg m USING (item_id, loc, startdate)
)
SELECT
    m.item_id, m.loc, m.startdate,
    m.n_active_cust,
    SUM(cs.demand_share * cs.demand_share) AS hhi_demand,
    MAX(cs.demand_share) AS top1_cust_share,
    m.total_demand / NULLIF(m.total_sales, 0) AS true_demand_ratio,
    m.total_oos / NULLIF(m.total_demand, 0) AS oos_rate,
    m.total_demand - m.total_sales AS demand_sales_gap_3m
    -- ... (remaining features computed in Python for rolling windows)
FROM monthly_agg m
LEFT JOIN cust_shares cs USING (item_id, loc, startdate)
GROUP BY m.item_id, m.loc, m.startdate, m.n_active_cust,
         m.total_demand, m.total_sales, m.total_oos
```

Rolling-window features (HHI over 3 months, churn detection, retention rate,
channel entropy, mix shift) are computed in Python using pandas after loading
the per-month aggregates. This avoids complex multi-CTE window functions.

#### Grain Mapping: item × location → DFU

`customer_features_monthly` is at item × location grain. The backtest
feature matrix is at item × customer_group × location grain (DFU). The join:

```python
# In build_feature_matrix():
# customer_features has (item_id, loc, startdate, 34 features)
# grid has (sku_ck, item_id, customer_group, loc, startdate, qty, ...)
grid = grid.merge(
    customer_features,
    on=["item_id", "loc", "startdate"],
    how="left",
)
# Multiple DFUs sharing (item_id, loc) get the same customer features — correct,
# since customer features describe the item-location regardless of customer_group.
grid[CUSTOMER_FEATURE_COLS] = grid[CUSTOMER_FEATURE_COLS].fillna(0)
```

---

## 5. Integration Points (Exact Code Locations)

### 5.1 `common/core/constants.py` — Add Feature Column Constants

```python
# After line 41 (EXTERNAL_FORECAST_FEATURES):

CUSTOMER_CONCENTRATION_FEATURES = [
    "n_active_cust", "n_active_cust_6m", "hhi_demand",
    "top1_cust_share", "top3_cust_share", "cust_gini",
]
CUSTOMER_DYNAMICS_FEATURES = [
    "new_cust_demand_share", "churned_cust_demand_share",
    "cust_count_mom", "cust_retention_rate", "cust_tenure_mean",
]
CUSTOMER_TRUE_DEMAND_FEATURES = [
    "true_demand_ratio", "oos_rate", "oos_cust_pct",
    "demand_sales_gap_3m", "oos_trend",
    "demand_qty_lag1", "demand_qty_lag3_mean",
]
CUSTOMER_CHANNEL_MIX_FEATURES = [
    "channel_entropy", "dominant_channel_share",
    "channel_mix_shift", "on_premise_share",
]
CUSTOMER_CROSS_FEATURES = [
    "cust_demand_cv_mean", "cust_demand_sync", "max_cust_share_delta",
]
CUSTOMER_ATTRIBUTE_MIX_FEATURES = [
    "store_type_entropy", "dominant_store_type_share",
    "chain_ratio", "top_chain_share", "sub_channel_entropy",
    "active_cust_pct", "avg_delivery_freq",
    "on_premise_acct_share", "premise_diversity",
]
CUSTOMER_FEATURE_COLS = (
    CUSTOMER_CONCENTRATION_FEATURES
    + CUSTOMER_DYNAMICS_FEATURES
    + CUSTOMER_TRUE_DEMAND_FEATURES
    + CUSTOMER_CHANNEL_MIX_FEATURES
    + CUSTOMER_CROSS_FEATURES
    + CUSTOMER_ATTRIBUTE_MIX_FEATURES
)

# Add to PROTECTED_FEATURES (never dropped by SHAP):
PROTECTED_CUSTOMER_FEATURES = {
    "true_demand_ratio",   # most valuable signal
    "n_active_cust",       # fundamental breadth
    "hhi_demand",          # concentration risk
}
# Update PROTECTED_FEATURES set:
# PROTECTED_FEATURES = PROTECTED_FEATURES | PROTECTED_CUSTOMER_FEATURES
```

Update `ENHANCED_FEATURES` aggregation:

```python
ENHANCED_FEATURES = (
    FOURIER_FEATURES + CROSTON_FEATURES + CROSS_DFU_FEATURES
    + EXTERNAL_FORECAST_FEATURES + CUSTOMER_FEATURE_COLS
)
```

### 5.2 `common/ml/feature_engineering.py` — Customer Feature Join

Insert after Step 11 (item attribute join, ~line 604) and before Step 12
(cross-DFU cluster aggregates):

```python
# Step 11b: Customer-derived features (enriched models only)
if customer_features is not None and len(customer_features) > 0:
    grid = grid.merge(
        customer_features,
        on=["item_id", "loc", "startdate"],
        how="left",
    )
    for col in CUSTOMER_FEATURE_COLS:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0).astype("float32")
```

**Signature change to `build_feature_matrix()`:**

```python
def build_feature_matrix(
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    item_attrs: pd.DataFrame,
    all_months: list[pd.Timestamp],
    cat_dtype: str = "category",
    customer_features: pd.DataFrame | None = None,  # NEW
) -> pd.DataFrame:
```

**Signature change to `mask_future_sales()`:**

No change needed — customer features at time t are computed from data
through time t. The standard `startdate <= cutoff` filter in the join
ensures no leakage. Customer features for future months simply don't exist
in the table (they're NULL → filled with 0).

### 5.3 `common/ml/backtest_framework.py` — Load Customer Features

In `load_backtest_data()` (~line 237), add optional customer feature loading:

```python
def load_backtest_data(
    db, algo_config=None, include_item_attrs=True,
    include_customer_features=False,  # NEW
):
    # ... existing code ...

    customer_features = None
    if include_customer_features:
        cur.execute("""
            SELECT item_id, loc, startdate,
                   n_active_cust, n_active_cust_6m, hhi_demand,
                   top1_cust_share, top3_cust_share, cust_gini,
                   new_cust_demand_share, churned_cust_demand_share,
                   cust_count_mom, cust_retention_rate, cust_tenure_mean,
                   true_demand_ratio, oos_rate, oos_cust_pct,
                   demand_sales_gap_3m, oos_trend,
                   demand_qty_lag1, demand_qty_lag3_mean,
                   channel_entropy, dominant_channel_share,
                   channel_mix_shift, on_premise_share,
                   cust_demand_cv_mean, cust_demand_sync, max_cust_share_delta,
                   store_type_entropy, dominant_store_type_share,
                   chain_ratio, top_chain_share, sub_channel_entropy,
                   active_cust_pct, avg_delivery_freq,
                   on_premise_acct_share, premise_diversity
            FROM customer_features_monthly
            ORDER BY item_id, loc, startdate
        """)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        customer_features = pd.DataFrame(rows, columns=cols)
        customer_features["startdate"] = pd.to_datetime(customer_features["startdate"])
        for c in CUSTOMER_FEATURE_COLS:
            if c in customer_features.columns:
                customer_features[c] = pd.to_numeric(customer_features[c], errors="coerce").fillna(0)

    return sales_df, dfu_attrs, item_attrs, customer_features
```

### 5.4 `scripts/run_backtest.py` — Pass Customer Features

In the backtest script, determine whether to load customer features based
on model config:

```python
# After config loading (~line 990):
use_customer_features = algo.get("customer_features", False)

# In data loading (~line 1012):
sales_df, dfu_attrs, item_attrs, customer_features = load_backtest_data(
    db, algo_config=algo,
    include_customer_features=use_customer_features,
)

# In feature matrix build (~line 1050):
full_grid = build_feature_matrix(
    sales_df, dfu_attrs, item_attrs, all_months,
    customer_features=customer_features,
)
```

### 5.5 `scripts/generate_production_forecasts.py` — Inference Grid

In `build_inference_grid()` (~line 457), conditionally join customer features:

```python
# After item-level attributes join (~line 565):
if model_id.endswith("_cust_enriched"):
    cust_feat = _load_customer_features_for_inference(item_ids, locs, months)
    grid = grid.merge(cust_feat, on=["item_id", "loc", "startdate"], how="left")
    for col in CUSTOMER_FEATURE_COLS:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0)
```

The `feature_cols` stored in the artifact `.pkl` file will include customer
feature names. The generic `model.predict(X[feature_cols])` call works
unchanged — the feature columns are just wider.

### 5.6 `common/ml/shap_selector.py` — Protected Features

Add the 3 protected customer features to the existing `PROTECTED_FEATURES`
set in `constants.py`:

```python
PROTECTED_FEATURES = {
    "month", "quarter", "ml_cluster",
    *FOURIER_FEATURES,
    "croston_demand_size", "croston_probability",
    # Customer enrichment (never drop these):
    "true_demand_ratio", "n_active_cust", "hhi_demand",
}
```

The remaining 31 customer features are subject to SHAP selection —
if they don't contribute, they get dropped automatically by the 95%
cumulative threshold.

---

## 6. Configuration

### 6.1 `config/forecasting/forecast_pipeline_config.yaml` — Algorithm Roster

```yaml
  lgbm_cust_enriched:
    type: tree
    library: lightgbm
    cluster_strategy: per_cluster
    config_key: lgbm_cust_enriched
    output_dir: data/backtest/lgbm_cust_enriched
    stages:
      tune: true
      backtest: true
      compete: true
      forecast: true
      expert: false
    notes: "LGBM with 34 customer-derived features"

  catboost_cust_enriched:
    type: tree
    library: catboost
    cluster_strategy: per_cluster
    config_key: catboost_cust_enriched
    output_dir: data/backtest/catboost_cust_enriched
    stages:
      tune: true
      backtest: true
      compete: true
      forecast: true
      expert: false
    notes: "CatBoost with 34 customer-derived features"

  xgboost_cust_enriched:
    type: tree
    library: xgboost
    cluster_strategy: per_cluster
    config_key: xgboost_cust_enriched
    output_dir: data/backtest/xgboost_cust_enriched
    stages:
      tune: true
      backtest: true
      compete: true
      forecast: true
      expert: false
    notes: "XGBoost with 34 customer-derived features"
```

### 6.2 `config/forecasting/forecast_pipeline_config.yaml` — Hyperparameters

Start with the same hyperparameters as the base models, with two changes:

```yaml
lgbm_cust_enriched:
  enabled: true
  model_id: lgbm_cust_enriched
  cluster_strategy: per_cluster
  recursive: true
  shap_select: true
  shap_threshold: 0.95
  shap_sample_size: 500
  customer_features: true          # NEW: triggers customer feature loading
  # Same params as lgbm, with regularization bump for more features:
  n_estimators: 1500
  learning_rate: 0.02
  num_leaves: 127
  min_child_samples: 40
  max_depth: -1
  min_gain_to_split: 0.005
  subsample: 0.7
  bagging_freq: 1
  colsample_bytree: 0.65          # slightly lower (0.70 → 0.65) for more features
  feature_fraction_bynode: 0.45   # slightly lower (0.50 → 0.45)
  reg_lambda: 1.5                 # slightly higher (1.0 → 1.5) regularization
  reg_alpha: 0.15                 # slightly higher (0.1 → 0.15)
  path_smooth: 4
  max_bin: 127

catboost_cust_enriched:
  enabled: true
  model_id: catboost_cust_enriched
  cluster_strategy: per_cluster
  recursive: true
  shap_select: true
  shap_threshold: 0.95
  shap_sample_size: 500
  customer_features: true
  iterations: 800
  learning_rate: 0.03
  depth: 7
  l2_leaf_reg: 9.0                # bumped from 7.5
  subsample: 0.85
  grow_policy: Lossguide
  border_count: 32
  random_strength: 0.4
  min_data_in_leaf: 28
  colsample_bylevel: 0.80         # slightly lower (0.85 → 0.80)
  bagging_temperature: 0.4
  max_leaves: 80
  bootstrap_type: MVS
  model_size_reg: 0.10
  score_function: L2
  boost_from_average: true

xgboost_cust_enriched:
  enabled: true
  model_id: xgboost_cust_enriched
  cluster_strategy: per_cluster
  recursive: true
  shap_select: true
  shap_threshold: 0.95
  shap_sample_size: 500
  customer_features: true
  n_estimators: 2000
  learning_rate: 0.02
  max_depth: 7
  min_child_weight: 15
  subsample: 0.8
  colsample_bytree: 0.75          # slightly lower (0.80 → 0.75)
  grow_policy: lossguide
  max_leaves: 255
  max_bin: 256
  reg_lambda: 6.0                 # bumped from 5.0
  reg_alpha: 0.6                  # bumped from 0.5
  gamma: 0.1
  colsample_bylevel: 0.75         # slightly lower (0.80 → 0.75)
  booster: gbtree
```

**Regularization rationale:** 34 additional features increase overfitting
risk. Slightly lower `colsample_bytree` (random feature subset per tree)
and higher L1/L2 regularization counteract this. SHAP selection provides
a second line of defense by auto-dropping low-value features.

### 6.3 Makefile Targets

```makefile
# Customer feature generation
customer-features:
	$(UV) python -m scripts.ml.generate_customer_features

# Enriched tree backtests
backtest-lgbm-cust:
	$(UV) python -m scripts.run_backtest --model lgbm \
		--model-id lgbm_cust_enriched \
		--config config/forecasting/forecast_pipeline_config.yaml

backtest-catboost-cust:
	$(UV) python -m scripts.run_backtest --model catboost \
		--model-id catboost_cust_enriched \
		--config config/forecasting/forecast_pipeline_config.yaml

backtest-xgboost-cust:
	$(UV) python -m scripts.run_backtest --model xgboost \
		--model-id xgboost_cust_enriched \
		--config config/forecasting/forecast_pipeline_config.yaml

backtest-cust-enriched-all: backtest-lgbm-cust backtest-catboost-cust backtest-xgboost-cust

# Load enriched backtests
backtest-load-cust-enriched:
	$(UV) python -m scripts.load_backtest_forecasts \
		--model lgbm_cust_enriched --replace
	$(UV) python -m scripts.load_backtest_forecasts \
		--model catboost_cust_enriched --replace
	$(UV) python -m scripts.load_backtest_forecasts \
		--model xgboost_cust_enriched --replace

# Tuning
tune-lgbm-cust:
	$(UV) python -m scripts.tune_hyperparams --model lgbm_cust_enriched

tune-cust-enriched-all: tune-lgbm-cust
	$(UV) python -m scripts.tune_hyperparams --model catboost_cust_enriched
	$(UV) python -m scripts.tune_hyperparams --model xgboost_cust_enriched
```

---

## 7. Leakage Prevention

Customer features are temporal — they describe the customer base **as of
a given month**. Leakage prevention requires:

### 7.1 Feature Table is Pre-Lagged

All features in `customer_features_monthly` at `startdate = t` are computed
from data through month `t-1` only. The generation script enforces this:

```python
# For each target month t, use data from [t-window, t-1]
for target_month in all_months:
    window_end = target_month - pd.DateOffset(months=1)  # exclusive
    window_start = window_end - pd.DateOffset(months=window_months)
    window_data = demand_df[
        (demand_df["startdate"] >= window_start) &
        (demand_df["startdate"] <= window_end)
    ]
    # compute features from window_data → assign to target_month
```

### 7.2 Mask Future Sales Compatibility

`mask_future_sales(grid, cutoff)` zeros out `qty` for months after cutoff
and recomputes lag/rolling features. Customer features don't need masking
because they are already pre-lagged — the value at month t uses data only
through t-1. However, for months after cutoff, customer features simply
won't exist in the table (NULL → filled with 0), which is correct behavior.

### 7.3 No Leakage from Cross-DFU Features

Customer features are computed at item × location grain, then joined to DFU
grain. Since different DFUs sharing (item_id, loc) get the same customer
features, there is no information leakage between DFUs — they share the
same observable customer base.

---

## 8. SHAP Analysis Plan

SHAP analysis will reveal which customer features add value and which can
be pruned. Expected outcomes:

### 8.1 Expected High-Importance Features

| Feature | Expected SHAP Rank | Reason |
|---------|-------------------|--------|
| `true_demand_ratio` | Top 5 | Directly corrects stockout bias in lags |
| `demand_qty_lag1` | Top 10 | Unbiased demand signal (vs biased `qty_lag_1`) |
| `hhi_demand` | Top 15 | Concentration risk affects forecast variance |
| `oos_rate` | Top 10 | Quantifies suppressed demand magnitude |
| `n_active_cust` | Top 15 | Customer breadth is a stability signal |

### 8.2 Expected Low-Importance Features (Candidates for SHAP Pruning)

| Feature | Reason |
|---------|--------|
| `cust_demand_sync` | Pairwise correlation is noisy with small customer counts |
| `channel_mix_shift` | May be too volatile month-over-month |
| `cust_tenure_mean` | Slow-moving, low discriminative power |

### 8.3 Protected Features

`true_demand_ratio`, `n_active_cust`, and `hhi_demand` are SHAP-protected
(never dropped regardless of importance rank). This ensures the model always
has access to the stockout correction and concentration signals.

---

## 9. Testing Plan

### 9.1 Unit Tests: `tests/unit/test_customer_features.py`

```python
# Test 1: Feature table grain
def test_customer_features_grain():
    """Output has exactly one row per (item_id, loc, startdate)."""

# Test 2: No leakage
def test_features_use_only_prior_data():
    """Features at month t computed only from data through t-1."""

# Test 3: HHI computation
def test_hhi_single_customer():
    """Single customer → HHI = 1.0."""

def test_hhi_even_split():
    """N customers with equal demand → HHI = 1/N."""

# Test 4: OOS features
def test_true_demand_ratio_no_stockout():
    """When oos_qty = 0, true_demand_ratio = 1.0."""

def test_true_demand_ratio_with_stockout():
    """When demand > sales, ratio > 1.0."""

# Test 5: Churn detection
def test_churned_cust_detection():
    """Customer ordering in [-6,-3] but not [-3,0] is counted as churned."""

# Test 6: NaN handling
def test_missing_customer_data_fills_zero():
    """Item-locs with no customer demand data get 0 for all features."""
```

### 9.2 Unit Tests: `tests/unit/test_feature_engineering_enriched.py`

```python
# Test 1: Feature matrix width
def test_enriched_grid_has_customer_columns():
    """Grid with customer_features includes all 34 columns."""

# Test 2: Left join produces NaN-free output
def test_enriched_grid_no_nans():
    """After join + fillna, no NaN in customer feature columns."""

# Test 3: Backward compatibility
def test_standard_grid_unchanged():
    """Grid without customer_features has same shape as before."""
```

### 9.3 Integration Tests

```python
# Test: End-to-end backtest with enriched model
def test_enriched_backtest_produces_valid_output():
    """Run mini backtest (2 timeframes, 100 DFUs) with customer features."""

# Test: Production inference with enriched artifact
def test_production_forecast_enriched_model():
    """Load enriched .pkl, build inference grid, predict."""
```

---

## 10. Rollout Plan

### Phase 1: Feature Generation (3 days)

- [ ] DDL: `sql/116_create_customer_features_monthly.sql`
- [ ] Script: `scripts/ml/generate_customer_features.py`
- [ ] Constants: `CUSTOMER_FEATURE_COLS` in `common/core/constants.py`
- [ ] Makefile target: `make customer-features`
- [ ] Unit tests: `tests/unit/test_customer_features.py`
- [ ] Run feature generation, validate row counts and value ranges

### Phase 2: Feature Integration (2 days)

- [ ] Modify `build_feature_matrix()` signature + customer join
- [ ] Modify `load_backtest_data()` with `include_customer_features` flag
- [ ] Update `PROTECTED_FEATURES` with 3 customer features
- [ ] Update `ENHANCED_FEATURES` aggregation
- [ ] Unit tests: `tests/unit/test_feature_engineering_enriched.py`

### Phase 3: Model Configuration (1 day)

- [ ] Add 3 enriched entries to `forecast_pipeline_config.yaml`
- [ ] Add 3 enriched hyperparameter blocks to `forecast_pipeline_config.yaml`
- [ ] Add Makefile targets for backtest, load, tune

### Phase 4: Backtest & Competition (2 days)

- [ ] Run: `make customer-features && make backtest-cust-enriched-all`
- [ ] Load: `make backtest-load-cust-enriched`
- [ ] Run champion selection: `make champion-all`
- [ ] Analyze WAPE: overall, by cluster, by stockout cohort, by concentration cohort
- [ ] Analyze SHAP: which customer features rank in top-20?

### Phase 5: Production (1 day)

- [ ] Modify `build_inference_grid()` for enriched models
- [ ] Verify production forecast generates correctly
- [ ] Add `customer-features` to pipeline dependency chain

### Phase 6: Tuning (Optional, 2 days)

- [ ] Run `make tune-cust-enriched-all` (Optuna 50 trials each)
- [ ] Apply tuned params, re-run backtest
- [ ] Compare tuned vs. default hyperparameters

---

## 11. Expected Impact

| DFU Cohort | Current WAPE | Expected WAPE | Improvement | Driver |
|------------|-------------|---------------|-------------|--------|
| All DFUs | ~25% | ~22-23% | 2-3pp | Across-the-board feature enrichment |
| Stockout-affected (oos_rate > 5%) | ~35% | ~28-30% | 5-7pp | `true_demand_ratio`, `demand_qty_lag1` correct biased lags |
| High-concentration (HHI > 0.5) | ~30% | ~25-27% | 3-5pp | `hhi_demand`, `top1_cust_share` flag risk |
| Customer churn inflection | ~28% | ~24-26% | 2-4pp | `churned_cust_demand_share`, `cust_retention_rate` |
| Channel shift | ~27% | ~25-26% | 1-2pp | `channel_mix_shift`, `channel_entropy` |

Enriched models expected to win champion selection for **30-45% of DFUs**.

---

## 12. Relationship to Other Specs

| Spec | Relationship |
|------|-------------|
| **02-20 Bolt Hierarchical** | Independent. Uses raw `demand_qty` series at customer level. This spec uses derived features at item-loc level. Both compete in champion selection. |
| **02-18 Chronos Foundation Models** | Chronos 2 Enriched can also consume customer features as covariates. Extend its covariate list with the same 34 features. |
| **02-07 Champion Selection** | No changes. Enriched models participate with `model_id = "lgbm_cust_enriched"` etc. |
| **02-03 Backtest Framework** | Minor extension: `load_backtest_data()` gains `include_customer_features` param. |
| **02-05 Advanced Backtest** | SHAP selection works unchanged. 3 new protected features added. |
| **02-19 Pipeline Config** | 3 new entries in algorithm roster with lifecycle flags. |

---

## 13. File Index

| Purpose | File | Status |
|---------|------|--------|
| DDL | `sql/116_create_customer_features_monthly.sql` | New |
| Feature generation | `scripts/ml/generate_customer_features.py` | New |
| Constants | `common/core/constants.py` | Modify |
| Feature engineering | `common/ml/feature_engineering.py` | Modify |
| Backtest framework | `common/ml/backtest_framework.py` | Modify |
| Backtest script | `scripts/run_backtest.py` | Modify |
| Production forecast | `scripts/generate_production_forecasts.py` | Modify |
| Pipeline config | `config/forecasting/forecast_pipeline_config.yaml` | Modify |
| Algorithm config | `config/forecasting/forecast_pipeline_config.yaml` | Modify |
| Makefile | `Makefile` | Modify |
| Unit tests (features) | `tests/unit/test_customer_features.py` | New |
| Unit tests (integration) | `tests/unit/test_feature_engineering_enriched.py` | New |
