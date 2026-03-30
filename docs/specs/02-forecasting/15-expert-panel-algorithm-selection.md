# 02-15 Expert Panel: Intelligent Algorithm Selection Engine

> **Status:** Implemented (Advanced Panel + Per-DFU Hybrid Ensemble) | **Features:** 49 + 50 + 51
> **Priority:** Critical — Directly impacts forecast accuracy and bottom line
> **Note:** This spec supersedes and absorbs `02-16-advanced-ml-foundation-model-extension.md`

---

## Plain-Language Overview

### What Is the Expert Panel?

> **One question:** "For each type of demand pattern in our business, which of our 30+ algorithms performs best?"

It doesn't pick one winner globally. It picks the **best algorithm per demand type**, then routes each SKU to its best-fit algorithm.

### The Big Picture in 4 Steps

```
1. Sample SKUs → 2. Test 30+ algorithms → 3. Build a score matrix → 4. Assign best-per-segment
```

### Step 1: Sample & Classify SKUs

Pick a representative set of DFUs (SKU × Location combinations) from the full catalog.
Then classify each one by its **demand shape**:

| Demand Type | What it looks like | Example |
|---|---|---|
| **Intermittent** | Mostly zeros, sporadic sales | Spare parts, niche items |
| **Seasonal** | Peaks and valleys every year | Holiday items, outdoor gear |
| **Trend-driven** | Growing or declining steadily | New product launch |
| **Stable** | Flat, predictable | Everyday consumables |

Classification uses two signals:
- **ADI** (Average Demand Interval) — how often does it sell at all?
- **CV²** (Coefficient of Variation²) — when it does sell, how erratic is the quantity?

### Step 2: Test 30+ Algorithms Across Multiple Timeframes

For each algorithm, run a mini-backtest across 10 time periods — each algorithm must prove itself at multiple points in time, not just once.

**The 5 Algorithm Families:**

| Family | Count | Examples |
|---|---|---|
| Statistical | 5 | Holt-Winters, Simple ES, Croston SBA, Auto-ARIMA, Theta |
| Statistical Upgrades | 6 | AutoCES, DynamicTheta, IMAPA, TSB, ADIDA, MSTL |
| Tree Models | 3 | LightGBM, CatBoost, XGBoost |
| Deep Learning | 8 | N-BEATS, N-HiTS, TFT, DeepAR, TiDE, TCN, PatchTST, iTransformer |
| Foundation Models | 5 | Chronos (Amazon), TimesFM (Google), TimeGPT, Moirai, Lag-Llama |

> **Key difference from the main backtest:** DL models train **one global model** across all DFUs at once. Foundation models don't train at all — they predict zero-shot like GPT for time series.

### Step 3: Build the Affinity Matrix

After running everything, we get a score table:

```
                    holt_winters  nbeats  chronos  croston_sba  lgbm
intermittent            62%        71%      74%       78%        65%    ← best: croston_sba
seasonal                81%        85%      83%       61%        82%    ← best: nbeats
trend_driven            75%        83%      80%       58%        84%    ← best: lgbm
stable                  79%        78%      76%       62%        80%    ← best: lgbm
```

Each cell = accuracy % for that algorithm on that demand type, averaged across all timeframes.

### Step 4: Assign Best Algorithm Per Segment

The optimizer reads the affinity matrix and creates a routing table:

```
Intermittent demand  → Croston SBA    (78% accuracy)
Seasonal demand      → N-BEATS        (85% accuracy)
Trend-driven demand  → LightGBM       (84% accuracy)
Stable demand        → LightGBM       (80% accuracy)
```

Now for **every DFU in production**: classify its demand type → look up routing table → use that algorithm's forecast.

### Full End-to-End Example

**DFU:** Widget-A / NYC Warehouse

```
1. Classify:
   ADI = 1.8, CV² = 0.6  →  Demand type: "Intermittent"

2. Test all 30+ algorithms across 10 timeframes
   Best for intermittent: Croston SBA (WAPE = 22%)
   vs. LightGBM (WAPE = 35%)  ← trees struggle with sparse zeros
   vs. N-BEATS  (WAPE = 29%)  ← DL also worse here

3. Affinity matrix entry: [intermittent, croston_sba] = 78% accuracy

4. Portfolio assignment: Widget-A → Croston SBA

5. In production: Widget-A's forecast comes from Croston SBA
```

### The 15-Step Pipeline (Advanced Panel + Hybrid Ensemble)

```
 1. Sample golden set of DFUs
 2. Load sales + attribute data
 3. Classify demand archetypes
 4. Generate 10 backtest timeframes
 5. Build feature matrix (~62 features)

 Per timeframe:
 6.  Run 5 statistical models
 7.  Run 3 tree models + 3 baselines
 8.  Run 6 statistical upgrades
 9.  Run 2 DL baselines (DLinear, NLinear)

 Once (global):
 10. Train 8 DL models on MPS/GPU (global — all DFUs, one model)
 11. Run 5 foundation models (zero-shot)

 12. Load external forecast + existing champions for comparison
 13. Build affinity matrix → optimize portfolio assignments
 13b. Build per-DFU hybrid ensemble (DFU accuracy matrix + meta-router + blend)
 14. Compare & report (portfolio + hybrid vs all baselines)
```

---

## Problem

The current system runs three tree-based algorithms (LightGBM, CatBoost, XGBoost) trained on K-Means demand clusters. Champion selection picks the best tree per DFU-month. This leaves significant accuracy on the table:

1. **Algorithm monoculture** — All three models are gradient-boosted trees. They share the same weaknesses: poor on low-volume stable series (overfit), poor on purely seasonal items (miss simple patterns), poor on intermittent demand.

2. **Clustering blindness** — K-Means groups by demand shape features only. A high-proof spirit brand with strong seasonal patterns and a mass-market beer with identical demand shape get the same cluster, same model, same features — despite fundamentally different business drivers.

3. **No statistical baselines** — Simple Exponential Smoothing, Holt-Winters, or Croston SBA often outperform trees on 40-60% of DFUs (low-volume, stable, intermittent). The system has no way to discover this.

4. **Missing algorithm families** — Three critical families are entirely absent:
   - **Foundation models** (Chronos, Moirai, TimesFM, TimeGPT, Lag-Llama) — won VN1 2024 1st and 2nd place; eliminate cold-start problem
   - **Deep learning architectures** (N-BEATS, N-HiTS, TFT, DeepAR, etc.) — N-HiTS outperforms transformers 20% with 50x less compute; DeepAR powers Amazon's demand planning
   - **Modern statistical upgrades** (AutoCES, MSTL, IMAPA, ADIDA) — proven improvements over classical methods

5. **Champion-ceiling gap** — The gap between champion accuracy and oracle ceiling (typically 3-8%) represents real dollars. Closing even 1% across 100K+ DFUs translates directly to reduced safety stock, fewer stockouts, and better working capital.

**Without these families, the system leaves 2-5% WAPE improvement on the table.**

---

## Solution

**The Expert Panel** — a council of 31 specialized experts, each championing distinct algorithmic expertise. They collectively analyze SHAP values, demand characteristics, product attributes, and historical backtest performance to produce an optimal algorithm-to-segment mapping.

Unlike brute-force model competition, the Expert Panel uses **attribute-based intelligent segmentation** (not just K-Means), **algorithm-feature alignment** (match model strengths to demand drivers), and **sample-based rapid testing** (validate on 10% before committing).

### What Makes This Different

| Aspect | Current System | Expert Panel |
|--------|---------------|--------------|
| Algorithms | 3 tree models | 31 algorithms (statistical + ML + DL + foundation) |
| Segmentation | K-Means on demand features | Attribute-aware (brand, category, lifecycle, seasonality profile, ABC-XYZ) |
| Selection | Per-DFU-month champion (reactive) | Per-segment algorithm routing (proactive) |
| SHAP usage | Feature selection only | Algorithm routing + segment definition |
| Testing | Full 10-timeframe backtest | Stratified sample → full validation |
| Intelligence | Single meta-learner | 31-expert consensus with confidence scores |
| DL support | None | 8 NeuralForecast models + 5 foundation models |
| Cold-start | Trees fail | Foundation models excel (zero-shot) |
| Compute | CPU-only | GPU-aware (MPS on macOS, CUDA on Linux) |

---

## The 31 Experts

### Group 1: Statistical Method Experts (5)

#### Expert 1: Exponential Smoothing Specialist
- **Algorithms:** Simple ES, Holt's Linear, Holt-Winters (additive & multiplicative), Damped Trend, **AutoCES** (upgrade)
- **Strength signal:** Low CV demand, clear level/trend, moderate seasonality
- **SHAP alignment:** When `qty_lag_1`, `rolling_mean_3m`, `month` dominate → series is smooth and level-driven
- **Routing rule:** `cv_demand < 0.5 AND seasonal_amplitude < 0.3 AND zero_demand_pct < 0.1`
- **Auto-selection:** Uses AICc to pick the best ES variant per segment

#### Expert 2: ARIMA/SARIMAX Specialist
- **Algorithms:** ARIMA(p,d,q), SARIMA(p,d,q)(P,D,Q,s), Auto-ARIMA
- **Strength signal:** Stationary (after differencing) series with autocorrelation structure
- **SHAP alignment:** When lag features (lag_1 through lag_12) show high SHAP with decaying pattern
- **Routing rule:** `trend_r2 > 0.3 AND yoy_correlation > 0.4 AND n_months >= 24`

#### Expert 3: Theta Method Specialist
- **Algorithms:** Standard Theta, Optimized Theta, **DynamicOptimizedTheta** (upgrade)
- **Strength signal:** Trending series; changing trend regimes (DynamicTheta handles macro-influenced shifts)
- **SHAP alignment:** When `mom_growth`, `demand_accel`, `trend_slope_norm` are high-SHAP
- **Routing rule:** `abs(cagr) > 0.05 AND cv_demand < 0.8`

#### Expert 4: Intermittent Demand Specialist
- **Algorithms:** Croston SBA, TSB (Teunter-Syntetos-Babai), **IMAPA** (upgrade), **ADIDA** (upgrade)
- **Strength signal:** High zero-demand percentage, high ADI, lumpy patterns
- **SHAP alignment:** When `n_zero_last_6m`, `croston_adi` are high-SHAP
- **Routing rule:** `zero_demand_pct >= 0.3 AND adi >= 1.32`
- **Sub-routing:** IMAPA replaces Croston SBA as default (`adi >= 1.32`); TSB for end-of-life (`is_declining = true`); ADIDA when underlying demand is stable

#### Expert 5: Seasonal Decomposition Specialist
- **Algorithms:** STL decomposition, **MSTL** (multi-seasonal upgrade), X-13 ARIMA-SEATS
- **Strength signal:** Strong, stable seasonality that repeats predictably; multiple simultaneous seasonal patterns
- **SHAP alignment:** When `fourier_sin_12`, `fourier_cos_12`, `seasonal_amplitude` dominate SHAP
- **Routing rule:** `seasonal_amplitude > 0.4 AND is_yearly_seasonal = true`

---

### Group 2: ML Tree Experts (4)

#### Expert 6: LightGBM Specialist
- **Strength signal:** Complex nonlinear interactions, categorical features, large training sets
- **SHAP alignment:** When cross-DFU features and categorical features (`brand`, `region`) show high SHAP
- **Specialization:** Fastest tree; best for high-cardinality categoricals via histogram binning

#### Expert 7: CatBoost Specialist
- **Strength signal:** When categorical features (brand, region, abc_vol) are key demand drivers
- **SHAP alignment:** When `brand`, `region`, `abc_vol` show high SHAP → CatBoost native ordered target encoding excels
- **Specialization:** Lowest overfitting risk; best when train set is small relative to features

#### Expert 8: XGBoost Specialist
- **Strength signal:** When regularization is key (many correlated features, overfitting risk)
- **Specialization:** Elastic net regularization (L1 + L2) for automatic feature sparsity

#### Expert 9: Random Forest Specialist
- **Strength signal:** When prediction stability matters more than peak accuracy
- **SHAP alignment:** When SHAP ranks are unstable across timeframes (high rank_std) → model variance is the problem
- **Unique value:** OOB error provides free validation; no early stopping tuning needed

---

### Group 3: Advanced ML Experts (3)

#### Expert 10: Prophet Specialist
- **Strength signal:** Series with structural breaks, holiday effects, or trend changepoints
- **Routing rule:** Detects changepoints via Prophet's built-in detection; if n_changepoints > 2 in recent 24 months

#### Expert 11: Linear/Ridge Regression Specialist
- **Strength signal:** Low-complexity series where trees overfit
- **Routing rule:** `n_features_selected_by_shap <= 8 AND cv_demand < 0.4`

#### Expert 12: Gradient Boosting Ensemble Specialist
- **Strength signal:** No single algorithm dominates across timeframes for a segment
- **Approach:** Level-0 = top 3-5 base models per segment; Level-1 = Ridge meta-learner on OOF predictions
- **Constraint:** Only used when stacking measurably outperforms best base (≥0.5% WAPE improvement)

---

### Group 4: Ensemble/Hybrid Experts (3)

#### Expert 13: Weighted Blend Specialist
- **Method:** Constrained optimization: minimize WAPE subject to weights ∈ [0,1], sum = 1
- **Guard:** If one model has >85% weight → just use that model (blending adds complexity without benefit)

#### Expert 14: Regime Switching Specialist
- **Strength signal:** Structural breaks in demand (COVID, product reformulation, distribution changes)
- **Detection:** Monitors rolling WAPE per model; when champion switches frequently → regime instability

#### Expert 15: Temporal Hierarchy Specialist
- **Method:** Top-down, bottom-up, or MinT optimal reconciliation across monthly, quarterly, annual
- **Best for:** Product families with strong aggregate patterns but noisy item-level demand

---

### Group 5: Domain/Attribute Experts (3)

#### Expert 16: Demand Pattern Classifier
- **Role:** Classifies every DFU using the Syntetos-Boylan framework

| ADI \ CV² | Low (<0.49) | Medium (0.49-1.0) | High (>1.0) |
|-----------|-------------|-------------------|-------------|
| **Low (<1.32)** — Smooth | ETS / ARIMA | LGBM / CatBoost | XGBoost + Ridge |
| **Medium (1.32-2.0)** — Intermittent | Croston SBA | LGBM (Tweedie) | Ensemble |
| **High (>2.0)** — Lumpy | Croston TSB | ADIDA | Moving Average |

- **Extensions:** Lifecycle stage, seasonal dominance, promotional responsiveness

#### Expert 17: Product Attribute Segmentor
- **Segmentation dimensions:** `category × brand_name`, `abc_vol × xyz_class`, `seasonality_profile × variability_class`, `national_service_model × region`
- **Key insight:** Items in the same K-Means cluster but different ABC-XYZ segments may need different algorithms. A-X items (high volume, predictable) are best served by simple ETS; C-Z items (low volume, unpredictable) need Croston or zero-inflated models.

#### Expert 18: External Signal Specialist
- **Role:** Identifies which DFUs benefit from external forecast signals vs. pure statistical/ML methods
- **SHAP-driven:** If external forecast features have SHAP rank > 10 → external signal is noise for this segment

---

### Group 6: Meta-Strategy Experts (2)

#### Expert 19: SHAP Portfolio Analyst
- **Role:** Reads the SHAP landscape across all segments and identifies algorithm-feature alignment patterns
- **SHAP cluster types:**
  - **Lag-dominated** (lag features > 40% SHAP mass) → autoregressive models
  - **Seasonal-dominated** (fourier + month > 30%) → seasonal models
  - **Cross-DFU-dominated** (cluster features > 20%) → tree models
  - **Attribute-dominated** (categorical features > 15%) → CatBoost, Random Forest
  - **Trend-dominated** (derived features > 25%) → Theta, damped trend
  - **Sparse-signal** (top 5 features < 50% mass) → ensemble

#### Expert 20: Portfolio Optimizer
- **Objective:** Minimize total portfolio WAPE subject to: complexity budget (max N algorithms), minimum sample confidence (≥100 DFUs per algorithm), stability constraint (assignments don't change >20% month-over-month)
- **Method:** Mixed Integer Programming (MIP) via scipy.optimize or OR-Tools

---

### Group 7: Foundation Model Experts (5) — Advanced Extension

#### Expert 21: Zero-Shot Foundation Model Specialist
- **Algorithms:** Chronos (Amazon), TimesFM (Google)
- **Routing rule:** `n_history_months < 12 OR is_new_product = true OR is_cold_start = true`
- **Key value:** Eliminates cold-start problem. Chronos processes 300+ forecasts/sec on GPU. TimesFM handles 16K context.
- **When Chronos vs TimesFM:** Chronos for probabilistic outputs and fine-tuning; TimesFM for fast deterministic point forecasts on stable/trending series

#### Expert 22: Competition-Winning Foundation Model Specialist
- **Algorithms:** Moirai (Salesforce), fine-tuned variants
- **Routing rule:** `n_covariates > 3 OR segment_has_enough_data_for_finetuning = true`
- **Key value:** Won VN1 2024 (1st place, beating 250 competitors). Any-Variate Attention adapts to any number of input variables.
- **Fine-tuning protocol:** Minimum 500 DFU-months per segment; otherwise use zero-shot

#### Expert 23: API Foundation Model Specialist
- **Algorithms:** TimeGPT (Nixtla)
- **Routing rule:** `is_intermittent = true AND n_history_months >= 6`
- **Key value:** 2nd place VN1 2024 with zero training. API-driven, production-ready.
- **Trade-off:** External API dependency; commercial licensing; data leaves infrastructure
- **Fallback:** If API unavailable, route to Chronos (local, open-source)

#### Expert 24: Probabilistic Foundation Model Specialist
- **Algorithms:** Lag-Llama (ServiceNow/Mila)
- **Routing rule:** `safety_stock_sensitive = true OR requires_prediction_intervals = true`
- **Key value:** First open-source probabilistic foundation model. Calibrated prediction intervals feed safety stock calculations directly.

#### Expert 25: Foundation Model Ensemble Specialist
- **Algorithms:** AutoGluon-TimeSeries (Amazon meta-ensemble)
- **Routing rule:** `segment_oracle_gap_bps > 300 AND segment_n_dfus >= 200`
- **Key value:** Automatically ensembles ETS, ARIMA, LGBM, DeepAR, TFT, Chronos — eliminates manual model selection for difficult segments.

---

### Group 8: Deep Learning Architecture Experts (5) — Advanced Extension

#### Expert 26: N-BEATS / N-HiTS Specialist
- **Algorithms:** N-BEATS (interpretable & generic), N-HiTS (hierarchical interpolation)
- **Routing rule (N-BEATS):** `cv_demand < 0.6 AND seasonal_amplitude > 0.2 AND is_aggregate_level = true`
- **Routing rule (N-HiTS):** `forecast_horizon_months >= 6 AND n_history_months >= 24`
- **Performance:** N-HiTS achieves transformer-level accuracy with 50x less compute. N-BEATS beat M4 winner by 3%.

#### Expert 27: Temporal Fusion Transformer (TFT) Specialist
- **Strength signal:** Rich covariate environment — promotions, pricing, weather, holidays
- **Routing rule:** `n_high_shap_external_features >= 3 AND n_history_months >= 18 AND segment_n_dfus >= 100`
- **Key value:** Attention weights reveal which time steps and features drive each forecast — interpretability that builds planner trust. 1st place non-ensemble model in VN1 2024.

#### Expert 28: DeepAR Specialist
- **Strength signal:** Large catalog with related items; new product introduction; cold-start with similar products as training signal
- **Routing rule:** `(is_new_product = true AND has_similar_items = true) OR cross_dfu_shap_pct > 0.15`
- **Key value:** ~15% accuracy improvement over ARIMA/ETS at Amazon. Global model learns cross-series patterns without explicit feature engineering.

#### Expert 29: Efficient Deep Learning Specialist
- **Algorithms:** TiDE (Google MLP), TCN, DLinear/NLinear (baselines)
- **Routing rules:**
  - TiDE: `forecast_horizon_months >= 3 AND needs_covariates = true AND latency_sensitive = true`
  - TCN: `has_weekly_data = true OR needs_realtime_updates = true`
  - DLinear/NLinear: Always run as baseline — **if these beat your LightGBM, your features are not adding value**

#### Expert 30: Multivariate Transformer Specialist
- **Algorithms:** PatchTST (patch-based), iTransformer (inverted attention)
- **Routing rules:**
  - PatchTST: `n_history_months >= 36 AND can_pretrain_on_related_series = true`
  - iTransformer: `n_correlated_products >= 10 AND substitution_effect = true`
- **Key value:** iTransformer (ICLR 2024 Spotlight) generalizes to unseen variables.

---

### Group 9: Reconciliation Expert (1) — Advanced Extension

#### Expert 31: Hierarchical Reconciliation Specialist
- **Algorithms:** MinTrace, WLS, OLS, Bottom-Up, Top-Down
- **Problem solved:** Item-level forecasts do not sum to category/brand/total forecasts.
- **Method:** Define hierarchy (SKU → SubCategory → Category → Brand → Total × Location → Region → National) → reconcile via MinTrace (optimal) or WLS (faster)
- **Applied as:** Post-processing step to ALL forecasts, not a per-DFU algorithm

---

## Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      EXPERT PANEL ENGINE                              │
│                                                                      │
│  Phase 1: OBSERVE                                                    │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────────┐    │
│  │  SHAP   │  │  Demand  │  │  Product  │  │   Historical     │    │
│  │ Values  │  │ Profiles │  │  Attrs    │  │   Backtests      │    │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘  └──────┬───────────┘    │
│       └────────────┼──────────────┼────────────────┘               │
│                    ▼                                                │
│  Phase 2: SEGMENT                                                   │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Attribute-Based Intelligent Segmentation                 │      │
│  │  Layer 1: Demand archetype (Expert 16 — ADI × CV²)       │      │
│  │  Layer 2: Business attributes (Expert 17 — ABC-XYZ)      │      │
│  │  Layer 3: SHAP profile (Expert 19 — feature clustering)  │      │
│  └──────────────────┬───────────────────────────────────────┘      │
│                     ▼                                               │
│  Phase 3: PROPOSE                                                   │
│  ┌────┬────┬────┬────┬──────────────────────────────────────┐      │
│  │ E1 │ E2 │ E3 │... │ E21-E30 (DL + Foundation Models)     │      │
│  └──┬─┴──┬─┴──┬─┴──┬─┴──────────────────────────────────────┘      │
│     └────┴────┴────┘                                               │
│                    ▼                                                │
│  Phase 4: SAMPLE TEST (~15-30 min)                                  │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Stratified 10% sample per segment                        │      │
│  │  3 timeframes (A, E, J) — fast validation                │      │
│  │  Statistical: parallel per-DFU fit                       │      │
│  │  Trees: per-segment backtest framework                   │      │
│  │  DL/Foundation: global training + zero-shot              │      │
│  └──────────────────┬───────────────────────────────────────┘      │
│                     ▼                                               │
│  Phase 5: DEBATE & SCORE                                            │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Composite score = backtest WAPE (50%) +                  │      │
│  │    expert confidence (20%) +                              │      │
│  │    SHAP alignment (15%) +                                 │      │
│  │    complexity penalty (15%)                               │      │
│  └──────────────────┬───────────────────────────────────────┘      │
│                     ▼                                               │
│  Phase 6: OPTIMIZE (Expert 20 — MIP)                                │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Minimize total portfolio WAPE                            │      │
│  │  Subject to: complexity budget, stability, coverage      │      │
│  └──────────────────┬───────────────────────────────────────┘      │
│                     ▼                                               │
│  Phase 7: VALIDATE (full 10-timeframe backtest, ~45-90 min)         │
│  Phase 7b: RECONCILE (Expert 31 — MinTrace/WLS)                     │
│  Phase 8: REPORT                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Algorithm Decision Tree

```
                     ┌─ History >= 12 months?
                     │
                YES ─┤                              NO ─┐
                     │                                   │
        ┌────────────┴────────────┐           ┌─────────┴──────────┐
        │  Full Algorithm         │           │  Cold-Start Pathway │
        │  Competition            │           └─────────┬──────────┘
        └────────────┬────────────┘                     │
                     │                       Foundation Models:
          ┌──────────┼──────────┐            Chronos (zero-shot)
          │          │          │            Moirai  (zero-shot)
     Smooth/Erratic Intermittent Lumpy       TimeGPT (zero-shot)
          │          │          │            DeepAR  (cross-series)
    ┌─────┴─────┐  ┌─┴──┐  ┌───┴────┐       Ridge   (low-complexity)
    │           │  │    │  │        │
  Simple     Complex IMAPA/ TSB/  Moving
  ETS/Ridge  LGBM/  Croston ADIDA Average
  AutoCES    TFT    SBA
  DLinear    DeepAR
  N-BEATS    N-HiTS
  Theta
```

### Compute Tiers

```
Tier 1: CPU-Only
├─ Statistical models (ETS, ARIMA, Theta, Croston, AutoCES, MSTL)
├─ Tree models (LGBM, CatBoost, XGBoost, RF)
├─ Linear models (Ridge, DLinear, NLinear)
└─ Reconciliation (MinTrace, WLS)

Tier 2: GPU-Optional (DEMAND_GPU=auto)
├─ Chronos (3x faster on GPU, CPU works)
├─ TimesFM (GPU preferred, CPU fallback)
├─ N-BEATS / N-HiTS training
├─ TiDE, TCN, DeepAR training + inference

Tier 3: GPU-Required
├─ TFT training (multi-horizon attention)
├─ Moirai fine-tuning
├─ PatchTST / iTransformer
└─ Lag-Llama inference

Tier 4: API-Based
└─ TimeGPT (Nixtla API)
```

### Attribute-Based Segmentation

```
Layer 1: Demand Archetype (Expert 16)
┌──────────────────────────────────────────────────┐
│  Smooth │ Intermittent │ Lumpy │ Erratic │ New   │
│  (ADI × CV² matrix + lifecycle stage)            │
└──────────────────────────────────────────────────┘

Layer 2: Business Attributes (Expert 17)
┌──────────────────────────────────────────────────┐
│  ABC-XYZ Class │ Season Profile │ Category│Region│
│  (from dim_sku + dim_item attributes)            │
└──────────────────────────────────────────────────┘

Layer 3: SHAP Profile (Expert 19)
┌──────────────────────────────────────────────────┐
│  Lag-driven │ Seasonal │ Attribute │ Trend│Sparse│
│  (clustered by SHAP importance distribution)    │
└──────────────────────────────────────────────────┘

Combined Segment = Layer1 ∩ Layer2 ∩ Layer3
Example: "Smooth × AX × Seasonal-driven"
→ Algorithm: Holt-Winters (Expert 1 recommendation)

Target: 20-50 actionable segments
```

### Algorithm Affinity Matrix

```
                    │ ETS │ARIMA│Theta│Croston│LGBM │TFT  │N-BEATS│Chronos│
────────────────────┼─────┼─────┼─────┼───────┼─────┼─────┼───────┼───────┤
Smooth-AX-Seasonal  │0.92 │0.78 │0.65 │  ---  │0.85 │0.88 │ 0.90  │ 0.84  │
Intermit-CZ-Sparse  │ --- │ --- │ --- │ 0.88  │0.72 │ --- │  ---  │ 0.74  │
NewProduct-Cold     │0.55 │ --- │0.50 │  ---  │0.45 │0.60 │  ---  │ 0.68  │
HighVol-AX-Stable   │0.94 │0.88 │0.75 │  ---  │0.89 │0.82 │ 0.91  │ 0.86  │
...

Scores = Expected Accuracy % from sample backtest (Phase 4)
"---" = Algorithm not applicable for this segment
Winner highlighted per row; ties broken by complexity (simpler wins)
```

---

## Detailed Phase Design

### Phase 1: OBSERVE (~2 minutes)

```python
def observe(session_id: int, conn) -> ObservationBundle:
    return ObservationBundle(
        shap_summaries=load_shap_summaries(),
        backtest_archive=load_backtest_archive(conn),
        champion_results=load_champion_results(conn),
        dfu_attrs=load_dfu_attributes(conn),
        item_attrs=load_item_attributes(conn),
        demand_profiles=compute_demand_profiles(conn),  # ADI, CV², zero_pct, trend
        feature_stability=load_feature_stability(),      # SHAP rank stability
        current_accuracy=compute_current_accuracy(conn)
    )
```

### Phase 2: SEGMENT (~1 minute)

Three experts collaborate:
1. **Expert 16** classifies demand archetype via ADI × CV² matrix
2. **Expert 17** cross-tabulates with ABC-XYZ, seasonality profile, variability class; prunes segments < min_size
3. **Expert 19** clusters DFUs by SHAP importance distribution (not demand values)

```python
def create_segments(demand_archetypes, attribute_segments, shap_profiles, min_size=100):
    raw_segments = cross_join(demand_archetypes, attribute_segments, shap_profiles)
    while any(seg.n_dfus < min_size for seg in raw_segments):
        smallest = min(raw_segments, key=lambda s: s.n_dfus)
        merge(smallest, find_nearest_segment(smallest, raw_segments))
    if len(raw_segments) > 50:
        hierarchical_merge(raw_segments, target=40)
    return raw_segments
```

### Phase 3: PROPOSE (~30 seconds)

Each expert evaluates each segment and proposes an algorithm with confidence and reasoning:
- Expert 1 (ETS) on "Smooth-AX-Seasonal": *"Low CV (0.25), strong seasonality (0.52), seasonal-driven SHAP → Holt-Winters additive. Confidence: 92%."*
- Expert 4 (Croston) on "Lumpy-CZ-Sparse": *"ADI 2.8, CV² 1.4, 68% zero months → IMAPA. Trees predict non-zero when they shouldn't. Confidence: 95%."*
- Expert 21 (Chronos) on "NewProduct-Cold": *"6 months history only → zero-shot foundation model. No training needed. Confidence: 88%."*

**Self-assessment rules:** Experts only propose for segments matching their strengths. Confidence below `min_confidence_threshold` (60%) → expert abstains.

### Phase 4: SAMPLE TEST (~15-30 minutes)

For each unique (segment, proposed_algorithm) pair:
1. Stratified 10% sample per segment
2. Quick backtest: 3 timeframes (A, E, J — recent, mid, early)
3. Statistical: per-DFU fit, parallelized via ProcessPoolExecutor (>200 DFUs)
4. Trees: per-segment backtest framework
5. DL models: global training across all sample DFUs
6. Foundation models: zero-shot inference only

**Parallelization:**

| DFU count | Mode |
|-----------|------|
| ≤ 200 | Sequential (avoids process pool startup overhead) |
| > 200 | `ProcessPoolExecutor` — models and predict_months passed once per worker via `initializer` |

### Phase 5: DEBATE & SCORE (~10 seconds)

```python
def score_proposal(proposal, sample_result, segment, config):
    backtest_score    = normalize(sample_result.accuracy, segment.baseline_accuracy)
    confidence_score  = proposal.confidence / 100
    shap_score        = compute_shap_alignment(proposal.algorithm, segment.shap_profile)
    complexity_score  = 1.0 - (config['complexity_scores'][proposal.algorithm] / 10.0)

    return (
        backtest_score   * 0.50 +  # sample backtest accuracy
        confidence_score * 0.20 +  # expert self-assessment
        shap_score       * 0.15 +  # algorithm-feature alignment
        complexity_score * 0.15    # simpler wins ties
    )
```

### Phase 6: OPTIMIZE (~30 seconds)

Expert 20 (Portfolio Optimizer) solves the global assignment via MIP:
- **Decision variables:** binary assignment — segment_i → algorithm_j
- **Objective:** minimize Σ expected_wape(DFU_i, assigned_algo) weighted by DFU count
- **Constraints:** each segment gets exactly one algorithm; complexity budget (≤ max_algorithms); minimum DFU coverage per algorithm
- **Fallback:** greedy assignment if MIP infeasible

### Phase 7: VALIDATE (~45-90 minutes)

Full 10-timeframe backtest using the winning algorithm map. Compare against current champion baseline.

### Phase 7b: RECONCILE

Expert 31 applies cross-sectional reconciliation via MinTrace (optimal) or WLS (fast) across the product-location hierarchy.

---

## Data Model

### Tables

#### `expert_panel_session`
Tracks each Expert Panel run — configuration, status, results, promotion status.

```sql
CREATE TABLE expert_panel_session (
    session_id      SERIAL PRIMARY KEY,
    label           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'initializing'
                    CHECK (status IN ('initializing','observing','segmenting',
                           'proposing','testing','debating','optimizing',
                           'validating','completed','failed','cancelled')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    runtime_seconds INTEGER,
    config          JSONB NOT NULL DEFAULT '{}',
    sample_pct      NUMERIC(5,2) NOT NULL DEFAULT 10.0,
    max_algorithms  INTEGER NOT NULL DEFAULT 6,
    min_segment_size INTEGER NOT NULL DEFAULT 100,
    n_sample_timeframes INTEGER NOT NULL DEFAULT 3,
    n_segments         INTEGER,
    n_algorithms_used  INTEGER,
    baseline_accuracy  NUMERIC(8,4),
    panel_accuracy     NUMERIC(8,4),
    improvement_bps    INTEGER,
    algorithm_distribution JSONB,
    compute_tier_used  TEXT,
    foundation_models_used JSONB,
    dl_models_used     JSONB,
    reconciliation_method TEXT,
    reconciliation_improvement_bps INTEGER,
    is_promoted     BOOLEAN NOT NULL DEFAULT FALSE,
    promoted_at     TIMESTAMPTZ
);
```

#### `expert_panel_segment`
Defines each segment — demand archetype, business attributes, SHAP profile, winning algorithm.

```sql
CREATE TABLE expert_panel_segment (
    segment_id      SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES expert_panel_session(session_id),
    segment_label   TEXT NOT NULL,
    demand_archetype    TEXT,
    abc_xyz_class       TEXT,
    seasonality_profile TEXT,
    variability_class   TEXT,
    shap_profile        TEXT,
    n_dfus              INTEGER NOT NULL,
    cv_demand           NUMERIC(8,4),
    zero_demand_pct     NUMERIC(5,2),
    assigned_algorithm  TEXT NOT NULL,
    algorithm_accuracy  NUMERIC(8,4),
    runner_up_algorithm TEXT,
    accuracy_margin_bps INTEGER,
    baseline_accuracy   NUMERIC(8,4),
    improvement_bps     INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

#### `expert_panel_affinity`
The algorithm affinity matrix — expected accuracy per (segment, algorithm) pair.

```sql
CREATE TABLE expert_panel_affinity (
    affinity_id     SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES expert_panel_session(session_id),
    segment_id      INTEGER NOT NULL REFERENCES expert_panel_segment(segment_id),
    algorithm       TEXT NOT NULL,
    expected_accuracy NUMERIC(8,4),
    confidence_low  NUMERIC(8,4),
    confidence_high NUMERIC(8,4),
    n_sample_dfus   INTEGER,
    is_winner       BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (session_id, segment_id, algorithm)
);
```

#### `expert_panel_assignment`
Final DFU-level algorithm assignment.

```sql
CREATE TABLE expert_panel_assignment (
    assignment_id   SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES expert_panel_session(session_id),
    sku_ck          TEXT NOT NULL,
    segment_id      INTEGER NOT NULL REFERENCES expert_panel_segment(segment_id),
    assigned_algorithm TEXT NOT NULL,
    expected_accuracy  NUMERIC(8,4),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (session_id, sku_ck)
);
CREATE INDEX idx_epa_session_algo ON expert_panel_assignment(session_id, assigned_algorithm);
CREATE INDEX idx_epa_sku ON expert_panel_assignment(sku_ck);
```

#### `expert_panel_proposal`
Each expert's proposal per segment with scoring breakdown.

```sql
CREATE TABLE expert_panel_proposal (
    proposal_id     SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES expert_panel_session(session_id),
    segment_id      INTEGER NOT NULL REFERENCES expert_panel_segment(segment_id),
    expert_id       INTEGER NOT NULL,
    expert_name     TEXT NOT NULL,
    proposed_algorithm TEXT NOT NULL,
    confidence_score   NUMERIC(5,2),
    reasoning          TEXT,
    sample_wape        NUMERIC(8,4),
    sample_accuracy    NUMERIC(8,4),
    backtest_score     NUMERIC(5,2),
    confidence_weight  NUMERIC(5,2),
    shap_alignment     NUMERIC(5,2),
    complexity_penalty  NUMERIC(5,2),
    total_score        NUMERIC(5,2),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

#### `expert_panel_dl_prediction`
Predictions from deep learning and foundation models (with prediction intervals).

```sql
CREATE TABLE expert_panel_dl_prediction (
    prediction_id   BIGSERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES expert_panel_session(session_id),
    sku_ck          TEXT NOT NULL,
    startdate       DATE NOT NULL,
    basefcst_pref   NUMERIC(18,4) NOT NULL,
    forecast_lower  NUMERIC(18,4),
    forecast_upper  NUMERIC(18,4),
    confidence_level NUMERIC(5,2),
    algorithm_id    TEXT NOT NULL,
    model_variant   TEXT,
    is_zero_shot    BOOLEAN NOT NULL DEFAULT FALSE,
    is_fine_tuned   BOOLEAN NOT NULL DEFAULT FALSE,
    training_time_seconds NUMERIC(10,2),
    compute_tier    TEXT CHECK (compute_tier IN ('cpu', 'gpu', 'api')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, sku_ck, startdate, algorithm_id)
);
CREATE INDEX idx_epdl_session_algo ON expert_panel_dl_prediction(session_id, algorithm_id);
CREATE INDEX idx_epdl_sku_date ON expert_panel_dl_prediction(sku_ck, startdate);
```

#### `expert_panel_reconciliation`
Reconciled forecasts and adjustments from Expert 31.

```sql
CREATE TABLE expert_panel_reconciliation (
    reconciliation_id BIGSERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES expert_panel_session(session_id),
    sku_ck          TEXT NOT NULL,
    startdate       DATE NOT NULL,
    base_algorithm  TEXT NOT NULL,
    base_forecast   NUMERIC(18,4) NOT NULL,
    reconciled_forecast NUMERIC(18,4) NOT NULL,
    adjustment_pct  NUMERIC(8,4),
    hierarchy_level TEXT NOT NULL,
    hierarchy_path  TEXT NOT NULL,
    reconciliation_method TEXT NOT NULL CHECK (reconciliation_method IN ('mintrace','wls','ols','bottom_up','top_down')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, sku_ck, startdate, hierarchy_level)
);
```

---

## Algorithm Catalog (Config Format)

### Complexity Scores

Lower = simpler = bonus in the scoring formula.

```yaml
complexity_scores:
  # Baselines
  seasonal_naive: 1
  dlinear: 2
  nlinear: 2
  # Statistical
  simple_es: 2
  holt_winters: 3
  croston_sba: 3
  croston_tsb: 3
  autoces: 3
  tsb: 3
  adida: 3
  ridge: 3
  arima: 4
  theta: 4
  dynamic_theta: 4
  imapa: 4
  mstl: 4
  prophet: 5
  # Foundation models
  timegpt: 4
  chronos: 5
  timesfm: 5
  lag_llama: 6
  moirai: 6
  # Deep learning
  random_forest: 6
  nhits: 6
  nbeats: 6
  tcn: 6
  tide: 6
  deepar: 7
  lgbm: 7
  catboost: 7
  xgboost: 7
  tft: 8
  weighted_blend: 8
  patchtst: 8
  itransformer: 8
  stacked_ensemble: 9
  autogluon_ts: 9
```

### Algorithm Strengths & Compute Tiers

```yaml
algorithms:
  # --- Statistical (cpu) ---
  holt_winters:   {tier: cpu,  expert: 1, best_for: [smooth, seasonal, moderate_trend]}
  auto_arima:     {tier: cpu,  expert: 2, best_for: [autocorrelated, trending, long_history]}
  theta:          {tier: cpu,  expert: 3, best_for: [trending, medium_horizon]}
  dynamic_theta:  {tier: cpu,  expert: 3, best_for: [changing_trends, macro_influenced]}
  croston_sba:    {tier: cpu,  expert: 4, best_for: [intermittent, sparse]}
  imapa:          {tier: cpu,  expert: 4, best_for: [intermittent, multi_level]}
  tsb:            {tier: cpu,  expert: 4, best_for: [obsolescence, end_of_life]}
  adida:          {tier: cpu,  expert: 4, best_for: [intermittent, stable_underlying]}
  autoces:        {tier: cpu,  expert: 1, best_for: [complex_seasonal, ets_upgrade]}
  mstl:           {tier: cpu,  expert: 5, best_for: [multiple_seasonalities]}
  # --- DL baselines (cpu) ---
  dlinear:        {tier: cpu,  expert: 29, best_for: [sanity_check, stable_trending]}
  nlinear:        {tier: cpu,  expert: 29, best_for: [sanity_check, distribution_shift]}
  # --- Trees (cpu/gpu) ---
  lgbm_cluster:   {tier: cpu,  expert: 6,  best_for: [complex_interactions, large_data]}
  catboost_cluster:{tier: cpu, expert: 7,  best_for: [categorical_heavy, small_clusters]}
  xgboost_cluster:{tier: cpu,  expert: 8,  best_for: [regularization, mixed_signal]}
  # --- Deep learning (gpu_optional) ---
  nhits:          {tier: gpu_optional, expert: 26, best_for: [long_horizon, multi_seasonal]}
  nbeats:         {tier: gpu_optional, expert: 26, best_for: [stable, seasonal, aggregate]}
  deepar:         {tier: gpu_optional, expert: 28, best_for: [large_catalog, cold_start_via_related]}
  tcn:            {tier: gpu_optional, expert: 29, best_for: [high_frequency, realtime]}
  tide:           {tier: gpu_optional, expert: 29, best_for: [long_horizon_covariates, latency_sensitive]}
  # --- Deep learning (gpu_required) ---
  tft:            {tier: gpu_required, expert: 27, best_for: [rich_covariates, interpretable, multi_horizon]}
  patchtst:       {tier: gpu_required, expert: 30, best_for: [transfer_learning, long_horizon]}
  itransformer:   {tier: gpu_required, expert: 30, best_for: [cross_product_correlation, substitution]}
  # --- Foundation models ---
  chronos:        {tier: gpu_optional, expert: 21, best_for: [cold_start, zero_shot, probabilistic]}
  timesfm:        {tier: gpu_optional, expert: 21, best_for: [long_context, stable_trending, fast]}
  moirai:         {tier: gpu_optional, expert: 22, best_for: [any_variate, competition_winner, fine_tunable]}
  timegpt:        {tier: api,          expert: 23, best_for: [zero_shot, intermittent, production_ready]}
  lag_llama:      {tier: gpu_required, expert: 24, best_for: [probabilistic, prediction_intervals, safety_stock]}
  autogluon_ts:   {tier: gpu_required, expert: 25, best_for: [meta_ensemble, difficult_segments]}
  # --- Reconciliation ---
  mintrace:       {tier: cpu,  expert: 31, best_for: [full_hierarchy, provably_optimal]}
  wls_reconciliation: {tier: cpu, expert: 31, best_for: [faster_reconciliation, scalable]}
```

---

## Implementation Plan

### Phase 1: Statistical Upgrades (Low Risk, Immediate Value)
1. Add `statsforecast` dependency
2. Implement AutoCES, DynamicTheta, IMAPA, TSB, ADIDA, MSTL
3. Add DLinear/NLinear baselines
4. Run affinity matrix comparison
- **Expected lift:** 0.5-1.0% WAPE on intermittent/seasonal segments

### Phase 2: Deep Learning Core (N-BEATS, N-HiTS, DeepAR)
1. Add `neuralforecast` and `gluonts` dependencies
2. Create `adv_algorithm_testing/dl_models.py` — unified wrapper ✅ **Done**
3. Implement N-BEATS, N-HiTS, DeepAR, TiDE, TCN, PatchTST, TFT, iTransformer ✅ **Done**
4. Integrate with golden set and affinity matrix pipeline
5. Add GPU detection and fallback logic ✅ **Done**
- **Expected lift:** 1-2% WAPE on stable/seasonal segments

### Phase 3: Foundation Models (Chronos, Moirai)
1. Add `chronos-forecasting` and `uni2ts` dependencies
2. Create `adv_algorithm_testing/foundation_models.py` ✅ **Done**
3. Implement Chronos zero-shot + TimesFM zero-shot + TimeGPT API
- **Expected lift:** 1-3% WAPE on cold-start/sparse, 0.5-1% on full portfolio

### Phase 4: TFT + Advanced Transformers
1. Integrate TFT with full covariate pipeline
2. Add PatchTST and iTransformer for multivariate scenarios
- **Expected lift:** 0.5-1.5% on covariate-rich segments

### Phase 5: Cross-Sectional Reconciliation
1. Add `hierarchicalforecast` dependency
2. Define product/location hierarchies from `dim_item` and `dim_sku`
3. Implement MinTrace + WLS as post-processing step
- **Expected lift:** 0.3-0.8% WAPE + coherent plans

### Phase 6: Integration & Portfolio Re-Optimization
1. Extend affinity matrix to include all new algorithms
2. Re-run portfolio optimizer with expanded catalog
3. Add TimeGPT as optional external enrichment
- **Expected total lift: 3-5% WAPE improvement over current system**

---

## Expected Accuracy Impact

| Segment | Current Best | Extended Best | Expected Lift | Key New Algorithm |
|---------|-------------|---------------|---------------|-------------------|
| Smooth-High-Volume | LGBM ~88% | N-HiTS/ETS ~90% | +150-250 bps | N-HiTS, AutoCES |
| Smooth-Low-Volume | ETS ~82% | Chronos ~85% | +200-350 bps | Chronos, DLinear |
| Seasonal-Strong | Holt-Winters ~85% | TFT ~88% | +200-350 bps | TFT, MSTL, N-BEATS |
| Intermittent | Croston SBA ~72% | IMAPA ~76% | +300-450 bps | IMAPA, TSB |
| Lumpy | Croston ~65% | ADIDA+LGBM ~69% | +300-450 bps | ADIDA, DeepAR |
| New Products | Ridge ~55% | Chronos ~68% | +1000-1500 bps | Chronos, Moirai |
| Erratic-Complex | LGBM ~78% | TFT/DeepAR ~82% | +300-450 bps | TFT, DeepAR |
| Aggregate/Category | LGBM ~90% | N-BEATS+Recon ~93% | +200-350 bps | N-BEATS, MinTrace |

**Portfolio-level expected improvement: 200-400 basis points (2-4% accuracy lift)**

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GPU not available | All foundation models have CPU fallback; N-HiTS/DLinear run on CPU; `DEMAND_GPU=off` disables GPU models |
| Foundation model API unavailability (TimeGPT) | Automatic fallback to Chronos (local, open-source) |
| DL overfitting on small segments | Minimum 200 DFU-months for DL; below threshold → statistical/tree only |
| Training time explosion | Max 30 min per model; timeout → fall back to simpler variant; heartbeat logging every 30s |
| Dependency bloat | Optional dependency groups (`foundation`, `deeplearning`, `statsupgrade`); core works without any |
| Reconciliation worsening forecasts | Monitor per-level accuracy pre/post; disable for levels where it degrades |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Portfolio WAPE | Baseline | -200 to -400 bps |
| Cold-start accuracy (<12 months) | ~55% (Ridge) | ~68% (foundation models) |
| Intermittent accuracy | ~72% (Croston SBA) | ~76% (IMAPA/ADIDA) |
| Forecast hierarchy coherence | Not measured | <1% total deviation |
| Oracle ceiling gap | ~5-8% | <3% |
| Algorithm diversity in production | 6-8 | 8-12 |

---

## Output Files (Advanced Panel)

```
adv_algorithm_testing/results/
├── golden_set_skus.csv          ← sampled DFUs
├── classification.csv            ← demand archetype per DFU
├── all_predictions.parquet       ← every algorithm × every DFU × every timeframe
├── affinity_matrix.csv           ← accuracy % per (segment, algorithm)
├── affinity_detail.csv           ← detailed breakdown with confidence intervals
├── assignments.csv               ← final routing: segment → best algorithm
├── portfolio_stats.json          ← overall accuracy, n_algorithms, improvement
├── comparison.json               ← portfolio vs. seasonal naive vs. external vs. existing
├── metadata.json                 ← runtime, n_dfus, n_predictions, algorithm_counts
└── experiment_report.txt         ← human-readable summary
```

---

## References

### Competition Results
- **M4 (2018):** N-BEATS (pure DL) beat the statistical ensemble winner by 3%. [Oreshkin et al., ICLR 2020]
- **M5 (Walmart, 2020):** LightGBM ensembles dominated. Top-2 used LGBM + N-BEATS hybrid with reconciliation.
- **VN1 (Supply Chain, 2024):** 1st = fine-tuned Moirai; 2nd = TimeGPT zero-shot; LightGBM remained dominant among traditional ML.

### Key Papers
- Oreshkin et al. "N-BEATS." ICLR 2020.
- Challu et al. "N-HiTS." AAAI 2023.
- Lim et al. "Temporal Fusion Transformers." IJF 2021.
- Salinas et al. "DeepAR." IJF 2020.
- Ansari et al. "Chronos: Learning the Language of Time Series." Amazon Science, 2024.
- Woo et al. "Moirai: A Time Series Foundation Model." Salesforce Research, 2024.
- Rasul et al. "Lag-Llama." NeurIPS 2023 Workshop.
- Das et al. "TimesFM." Google Research, 2024.
- Liu et al. "iTransformer." ICLR 2024 Spotlight.
- Das et al. "TiDE: Time-series Dense Encoder." Google, 2023.
- Zeng et al. "DLinear/NLinear." AAAI 2023.
- Hyndman et al. "Forecast reconciliation: A review." IJF 2024.
- Bandara et al. "MSTL." 2024.

### Expert Advisory Panel (Informed This Spec)
| # | Expert | Affiliation | Key Recommendation |
|---|--------|-------------|-------------------|
| 1 | Rob Hyndman | Monash University | Foundation models + reconciliation; MSTL; AutoCES |
| 2 | Spyros Makridakis | U. of Nicosia | LGBM ensembles (M5 winner); N-BEATS; DeepAR for quantiles |
| 3 | Sean Taylor | Motif Analytics | Prophet as benchmark; Bayesian structural models |
| 4 | Tim Januschowski | Amazon AWS | Chronos for zero-shot; DeepAR as production workhorse; TFT as the only worthwhile transformer |
| 5 | Nicolas Vandeput | VN1 Organizer | Foundation models competition-ready; LGBM king for traditional ML |
| 6 | Boris Oreshkin | ServiceNow | N-BEATS for interpretable deep forecasting; N-HiTS for long-horizon |
| 7 | Bryan Lim | Google DeepMind | TFT for interpretable multi-horizon with rich covariates |
| 8 | Kashif Rasul | Morgan Stanley | Lag-Llama for probabilistic zero-shot |
| 9 | Valentin Flunkert | Amazon AWS | DeepAR for global cross-series learning; cold-start |
| 10 | Azul Garza | Nixtla | NeuralForecast as unified implementation; N-HiTS best efficiency/accuracy tradeoff |

### Libraries
- NeuralForecast: N-BEATS, N-HiTS, TFT, DeepAR, PatchTST, iTransformer, TiDE, TCN, DLinear — `pip install neuralforecast`
- StatsForecast: AutoCES, MSTL, DynamicTheta, IMAPA, TSB, ADIDA — `pip install statsforecast`
- HierarchicalForecast: MinTrace, WLS — `pip install hierarchicalforecast`
- Chronos: `pip install chronos-forecasting`
- AutoGluon: `pip install autogluon.timeseries`

---

## Feature 51: Per-DFU Hybrid Ensemble

> **Status:** Implemented | **Target:** +8–12 percentage points accuracy vs segment-level portfolio

### Motivation

Empirical results from the expert panel show the segment-level affinity matrix leaves ~3–8% WAPE improvement on the table relative to the oracle ceiling (`compute_ceiling_accuracy`). The gap exists because:

1. All DFUs in a segment (e.g., `erratic_low`) share one algorithm, even though within that segment some are better served by CatBoost, others by Croston SBA, others by Ridge.
2. Segment-level WAPE averages heterogeneous DFUs — one catastrophically bad DFU can drag the whole segment to the wrong algorithm assignment.
3. No blending: a single wrong assignment vs. a weighted ensemble of top-3 is often 5–10pp different for individual DFUs.

### Architecture

```
All predictions (12–31 algorithms × N timeframes)
          ↓
build_dfu_accuracy_matrix()      → DFU × Algorithm WAPE table
          ↓                               ↓
train_meta_router()          compute_inverse_wape_blend()
(LightGBM classifier)         (top-K inverse-WAPE weights)
          ↓                               ↓
predict_meta_router()        ─── blend for low-confidence DFUs
          ↓
  confidence >= 0.6?
    YES → single best algorithm
    NO  → inverse-WAPE blend (top-3)
          ↓
compute_hybrid_predictions()     → algorithm_id = "hybrid"
          ↓
compare_all() + hybrid metrics injected into comparison dict
```

### Component 1: `algorithm_testing/dfu_accuracy_matrix.py`

**`build_dfu_accuracy_matrix(predictions_df, actuals_df, min_n_months=2)`**

Computes WAPE per `(sku_ck, algorithm_id)` across all backtest timeframes.  Predictions are first averaged across overlapping timeframe windows (same deduplication as `build_affinity_matrix`) to avoid double-counting months.

Returns: `sku_ck | algorithm_id | wape | accuracy_pct | n_months`

**`compute_inverse_wape_blend(predictions_df, dfu_accuracy_matrix, top_k=3)`**

For each DFU, selects the top-K algorithms by lowest WAPE and blends their predictions using inverse-WAPE weights:

```
weight_k = 1 / max(WAPE_k, 1e-6)   (normalised to sum to 1)
prediction = Σ(weight_k × forecast_k)
```

### Component 2: `algorithm_testing/meta_router.py`

**`train_meta_router(dfu_accuracy_matrix, dfu_attrs, classification_df)`**

Trains a `LGBMClassifier` to predict which algorithm wins per DFU.

- **Target:** `argmin(WAPE)` per `sku_ck` from `dfu_accuracy_matrix`
- **Features (from `dfu_attrs`):** `ml_cluster`, `variability_class`, `seasonality_profile`, `abc_xyz_segment`
- **Features (from `classification_df`):** `adi`, `cv2`, `mean_demand`, `std_demand`, `n_periods`, `n_nonzero`, `segment`, `volume_tier`
- **Encoding:** Categorical columns stored as integer codes; `cat_categories` dict saved in `MetaRouterModel` for consistent prediction-time encoding

Returns: `MetaRouterModel` dataclass (model + feature_cols + cat_categories + label_to_algorithm)

**`predict_meta_router(meta_model, dfu_attrs, classification_df)`**

Returns: `sku_ck | predicted_algorithm | confidence` (confidence = max softmax probability)

### Component 3: `algorithm_testing/hybrid_ensemble.py`

**`compute_hybrid_predictions(..., confidence_threshold=0.6, blend_top_k=3)`**

Routing decision per DFU:

| Condition | Action |
|---|---|
| confidence >= threshold | Use single predicted-best algorithm |
| confidence < threshold | Inverse-WAPE blend of top-K algorithms |
| DFU missing from meta-router | Blend (treated as low-confidence) |
| No accuracy history | Fall back to `seasonal_naive` |
| Predicted algorithm has no data for a DFU-month | Fall back to `seasonal_naive` |

Output: `sku_ck | startdate | basefcst_pref | algorithm_id="hybrid"`

### Integration in Run Scripts

Both `run_expert_panel.py` and `run_adv_expert_panel.py` execute step `9b` after the affinity matrix + portfolio optimization and before `compare_all`:

```python
dfu_accuracy_matrix = build_dfu_accuracy_matrix(all_predictions_df, actuals_df)
meta_model = train_meta_router(dfu_accuracy_matrix, dfu_attrs, classification_df)
hybrid_preds = compute_hybrid_predictions(all_predictions_df, dfu_accuracy_matrix,
                                           dfu_attrs, classification_df, meta_model)
# Injected into comparison dict post compare_all():
comparison["baselines"]["hybrid"] = hybrid_metrics
comparison["lift"]["hybrid_vs_portfolio_bps"] = ...
comparison["lift"]["hybrid_vs_naive_bps"] = ...
```

Saved artifacts: `dfu_accuracy_matrix.csv`, `hybrid_assignments.csv`

### Configuration (`hybrid_ensemble` key in both config.yaml files)

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable/disable the hybrid step |
| `min_n_months` | `2` | Min matched months for reliable per-DFU WAPE |
| `meta_n_estimators` | `300` | LightGBM boosting rounds |
| `meta_learning_rate` | `0.05` | LightGBM learning rate |
| `meta_num_leaves` | `31` | LightGBM max leaves |
| `confidence_threshold` | `0.6` | Below this → blend |
| `blend_top_k` | `3` | Algorithms in the blend |

### Expected Accuracy Impact

| Mechanism | Expected Gain |
|---|---|
| Per-DFU top-3 inverse-WAPE blend alone | ~4–6 pp |
| Meta-router routing vs segment-level | ~3–5 pp |
| Combined (hybrid) vs segment portfolio | **~8–12 pp** |

The oracle ceiling from `compute_ceiling_accuracy` is the theoretical upper bound. Once implemented, `hybrid` accuracy should be within 3–5pp of the ceiling, compared to 8–15pp for segment-level routing.

---

## Known Issues and Mitigations (from empirical run 2026-03-27)

### Erratic Segment Fallback Catastrophe

**Observation:** `erratic_high` and `erratic_low` archetypes show negative accuracy (−55.7% and −14.2%) in the portfolio, far worse than naive (29.0% and 4.3%).

**Root cause:** The affinity matrix assigned `tft` as the winner for erratic segments based on a biased sample (~109 of 2,739 DFUs had TFT coverage). At compare time, 3,584 DFU-months fell back to `seasonal_naive` because TFT predictions were not available for the full golden set. Seasonal naive performs catastrophically on erratic demand.

**Fix:** The hybrid ensemble's fallback chain avoids this by routing erratic DFUs to the inverse-WAPE blend of algorithms that actually have full coverage. Additionally, the comparison module's fallback for missing predictions should prefer `croston_sba` or `tsb` over `seasonal_naive` for archetypes where `adi >= 1.32`.

### Portfolio Bias Inflation

**Observation:** Portfolio bias = +0.048 vs champion +0.012 and external +0.007.

**Root cause:** Segment-level algorithm assignment ignores bias as a selection criterion — it optimises WAPE only. Algorithms that win on WAPE but have systematic positive bias inflate the portfolio.

**Mitigation:** The hybrid ensemble's per-DFU accuracy matrix includes `accuracy_pct` which already penalises WAPE (symmetric), but bias is not directly penalised. A future improvement is to add a bias penalty to the WAPE metric: `adjusted_wape = wape × (1 + abs(bias))`.

### TFT Coverage vs Accuracy Tradeoff

**Observation:** TFT routes 84% of DFUs but achieves only 58% overall accuracy. MSTL achieves comparable accuracy with 100% coverage.

**Fix:** The hybrid ensemble naturally handles this — if TFT has incomplete coverage for a DFU, that DFU's per-DFU WAPE for TFT will be missing from `dfu_accuracy_matrix`, so TFT will not be included in that DFU's blend or routing. Full-coverage algorithms dominate for DFUs they haven't been tested on.
