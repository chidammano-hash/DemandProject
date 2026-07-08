# Unified Model Tuning Studio

> A production-grade, UI-driven hyperparameter tuning platform for LightGBM, CatBoost, and XGBoost. Users configure experiment parameters directly in the browser, launch backtest runs that survive API restarts, monitor real-time logs in the Jobs tab, compare results with execution-lag filtering (lags 0-4), and promote winners to the champion pipeline — all without touching the command line.

| | |
|---|---|
| **Status** | Implemented |
| **Replaces** | Feature 45 (Model Tuning) — full rewrite of UI, API, and job integration |
| **UI Tab** | Model Tuning (sidebar: Demand section) |
| **API Prefix** | `/model-tuning/{model}` (model = lgbm, catboost, xgboost) |
| **Router** | `api/routers/forecasting/tuning/` (15-module package; `__init__.py` re-exports the unified router for `api/main.py` to mount) |
| **Frontend** | `frontend/src/api/queries/unified-model-tuning.ts`, `frontend/src/tabs/LgbmTuningTab.tsx` |
| **Tests** | `tests/api/test_unified_model_tuning.py` (40 tests), `frontend/src/tabs/__tests__/ModelTuningTab.test.tsx` |
| **Key Files** | `api/routers/forecasting/tuning/` (split from the legacy 1,798-LoC `unified_model_tuning.py` -- see [Router Layout](#router-layout)), `frontend/src/tabs/LgbmTuningTab.tsx`, `frontend/src/api/queries/unified-model-tuning.ts` |
| **Depends On** | Feature 3 (Backtest Framework), Feature 44 (Resilient Jobs), Feature 45 (Tuning Registry) |

---

## Table of Contents

1. [Problem](#problem)
2. [Solution Overview](#solution-overview)
3. [Expert-Recommended Experiment Plans](#expert-recommended-experiment-plans)
4. [UI Design](#ui-design)
5. [API Design](#api-design)
6. [Database Schema Changes](#database-schema-changes)
7. [Job Integration](#job-integration)
8. [Promotion Workflow](#promotion-workflow)
9. [Execution-Lag Filtering](#execution-lag-filtering)
10. [Resilience & Error Handling](#resilience--error-handling)
11. [Data Flow](#data-flow)
12. [Configuration](#configuration)
13. [Testing Spec](#testing-spec)
14. [Migration Plan](#migration-plan)
15. [Acceptance Criteria](#acceptance-criteria)
16. [Pre-Implementation Blockers](#pre-implementation-blockers)
17. [Gap Report & Fixes (Expert Review)](#gap-report--fixes-expert-review)
18. [Carried Forward from Feature 45 (10b-lgbm-tuning)](#carried-forward-from-feature-45-10b-lgbm-tuning)

---

## Problem

The current tuning system (Feature 45) has critical gaps:

1. **No UI-driven experiment launch** — Users must SSH into the server and run CLI commands (`scripts/ml/auto_tune.py`) to start experiments. There is no way to configure hyperparameters, select features, or pick a cluster strategy from the browser.

2. **No execution-lag filtering** — The comparison panel shows portfolio-level accuracy but cannot break down results by execution lag (0, 1, 2, 3, 4). Demand planners need to evaluate model quality at each lag horizon to understand how accuracy degrades over time.

3. **Confusing Jobs integration** — The existing `tuning_backtest` job type is only triggered by the AI chat advisor. There is no dedicated "Launch Experiment" flow. When a tuning job appears in the Jobs tab, it is labeled generically ("AI Tuning Backtest") with no model type indicator, making it hard to distinguish LGBM vs CatBoost vs XGBoost runs.

4. **Promotion is opaque** — The promote modal writes to `forecast_pipeline_config.yaml` but does not record which champion pipeline version will use the promoted params. Users cannot see promoted parameters alongside champion selection results.

5. **No experiment templates** — Each experiment requires manually typing every hyperparameter. There are no "start from current production" or "start from recommended strategy" shortcuts.

6. **Fragmented model UX** — LGBM has its own router (`lgbm_tuning.py`), CatBoost/XGBoost share a different router (`model_tuning.py`), and the UI dispatches via a `model-tuning.ts` abstraction layer. This creates inconsistency in API paths, response shapes, and error handling.

---

## Solution Overview

Build a **Unified Model Tuning Studio** with these capabilities:

| Capability | Description |
|-----------|-------------|
| **Experiment Builder** | Form-based UI to configure hyperparameters per model, with templates (production baseline, expert recommendations, custom) |
| **One-Click Launch** | Submit button creates a resilient job (subprocess isolation, PID tracking, log streaming) |
| **Real-Time Monitoring** | Live log streaming in Jobs tab with model-specific labels and progress indicators |
| **Execution-Lag Analysis** | Filter all metrics (accuracy, WAPE, bias) by execution lag 0-4 across runs, comparisons, and charts |
| **Side-by-Side Comparison** | Enhanced comparison panel with lag-level breakdown, cluster heatmaps, and parameter diffs |
| **Promote & Deploy** | Promote winning params to `forecast_pipeline_config.yaml` + record in `fact_production_forecast` lineage |
| **Leaderboard** | Sortable run table with filters by model, status, lag, and date range |

---

## Expert-Recommended Experiment Plans

### LightGBM — 5 Runs (Run 16 as Baseline)

The current production LGBM parameters (from `forecast_pipeline_config.yaml`) are designated as **Run 1** (baseline). These represent the culmination of 16 prior experiments.

#### Run 1 — Production Baseline (Run 16 Params)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 1500 | High tree count for slow-LR convergence |
| learning_rate | 0.02 | Conservative LR — residual pattern capture |
| num_leaves | 127 | Full leaf complexity for demand heterogeneity |
| min_child_samples | 40 | Moderate min-leaf — prevents noisy leaf splits |
| max_depth | -1 | Unlimited depth (leaf-limited by num_leaves) |
| min_gain_to_split | 0.01 | Low split threshold — allow fine-grained splits |
| subsample | 0.8 | 80% row sampling per iteration |
| bagging_freq | 1 | Bagging every iteration |
| colsample_bytree | 0.8 | 80% feature sampling per tree |
| feature_fraction_bynode | 0.7 | 70% feature sampling per split node |
| reg_lambda | 1.0 | L2 regularization |
| reg_alpha | 0.1 | L1 regularization (light sparsity) |
| path_smooth | 4.0 | Leaf output smoothing (reduces overfit on small leaves) |
| max_bin | 127 | Histogram bin count |
| cluster_strategy | per_cluster | Train separate model per ML cluster |
| recursive | true | Multi-step recursive forecasting |
| shap_select | true | SHAP-based feature selection |
| shap_threshold | 0.95 | Keep features covering 95% of total SHAP mass |

#### Run 2 — Aggressive Depth + Heavy Regularization

**Hypothesis:** The unlimited depth (-1) with 127 leaves may overfit on sparse clusters. Capping depth at 10 while increasing regularization should improve generalization on intermittent demand DFUs without losing accuracy on high-volume items.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| max_depth | 10 | -1 → 10 (capped) |
| num_leaves | 63 | 127 → 63 (halved) |
| reg_lambda | 3.5 | 1.0 → 3.5 (+250%) |
| reg_alpha | 0.5 | 0.1 → 0.5 (+400%) |
| path_smooth | 8.0 | 4.0 → 8.0 (+100%) |
| min_child_samples | 60 | 40 → 60 (+50%) |

#### Run 3 — Ultra-Slow Learning + Maximum Trees

**Hypothesis:** Reducing LR to 0.008 with 3000 trees allows the ensemble to capture subtle residual patterns that 1500 trees at 0.02 LR miss. The slower convergence reduces variance at the cost of training time.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| learning_rate | 0.008 | 0.02 → 0.008 (-60%) |
| n_estimators | 3000 | 1500 → 3000 (+100%) |
| subsample | 0.85 | 0.8 → 0.85 (+6%) |
| colsample_bytree | 0.85 | 0.8 → 0.85 (+6%) |

#### Run 4 — Feature Fraction Boost + Sparse Demand Campaign

**Hypothesis:** Increasing feature retention at the node level (0.7 → 0.9) while adding aggressive sparse-demand regularization should improve performance on the long-tail SKUs (>50% zero demand months) that dominate the DFU population.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| feature_fraction_bynode | 0.9 | 0.7 → 0.9 (+29%) |
| colsample_bytree | 0.9 | 0.8 → 0.9 (+13%) |
| min_child_samples | 100 | 40 → 100 (+150%) |
| min_gain_to_split | 0.05 | 0.01 → 0.05 (+400%) |
| reg_alpha | 1.0 | 0.1 → 1.0 (+900%) |
| path_smooth | 12.0 | 4.0 → 12.0 (+200%) |

#### Run 5 — Balanced Champion Candidate

**Hypothesis:** Combine the winning elements from Runs 2-4 based on cluster-level analysis. Moderate depth cap, slightly slower LR, increased node-level feature fraction, and balanced regularization should yield the best overall portfolio accuracy.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| learning_rate | 0.015 | 0.02 → 0.015 (-25%) |
| n_estimators | 2000 | 1500 → 2000 (+33%) |
| max_depth | 12 | -1 → 12 (capped) |
| num_leaves | 95 | 127 → 95 (-25%) |
| reg_lambda | 2.5 | 1.0 → 2.5 (+150%) |
| reg_alpha | 0.3 | 0.1 → 0.3 (+200%) |
| feature_fraction_bynode | 0.8 | 0.7 → 0.8 (+14%) |
| path_smooth | 6.0 | 4.0 → 6.0 (+50%) |
| min_child_samples | 50 | 40 → 50 (+25%) |

---

### CatBoost — 5 Runs (Expert Recommendations)

#### Run 1 — Current Production Baseline

Uses current `forecast_pipeline_config.yaml` CatBoost section as-is (champion_v2 params from Phase 3).

| Parameter | Value |
|-----------|-------|
| iterations | 3000 |
| learning_rate | 0.008 |
| depth | 10 |
| l2_leaf_reg | 7.5 |
| subsample | 0.85 |
| reg_lambda | 3.5 |
| grow_policy | Lossguide |
| border_count | 64 |
| random_strength | 0.5 |
| min_data_in_leaf | 28 |
| colsample_bylevel | 0.85 |
| bagging_temperature | 0.4 |
| max_leaves | 127 |
| bootstrap_type | MVS |
| model_size_reg | 0.08 |
| score_function | L2 |
| boost_from_average | true |
| leaf_estimation_method | Newton |
| leaf_estimation_iterations | 10 |
| max_ctr_complexity | 1 |

#### Run 2 — Ordered Boosting + Symmetric Trees

**Hypothesis:** CatBoost's default ordered boosting with symmetric trees (depth-wise growth) reduces prediction shift on time-series data. Removing Lossguide and using depth=8 symmetric trees with ordered boosting may better handle the temporal autocorrelation in demand data.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| grow_policy | SymmetricTree | Lossguide → SymmetricTree |
| depth | 8 | 10 → 8 |
| max_leaves | (removed) | 127 → N/A (depth-controlled) |
| bootstrap_type | Ordered | MVS → Ordered |
| iterations | 4000 | 3000 → 4000 |
| learning_rate | 0.006 | 0.008 → 0.006 |
| random_strength | 1.0 | 0.5 → 1.0 |

#### Run 3 — High Border Count + Reduced Leaf Reg

**Hypothesis:** Increasing border_count from 64 to 128 provides finer split resolution on continuous demand features (lag values, rolling averages). Combined with reduced l2_leaf_reg, the model can capture sharper demand transitions.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| border_count | 128 | 64 → 128 (+100%) |
| l2_leaf_reg | 3.0 | 7.5 → 3.0 (-60%) |
| min_data_in_leaf | 40 | 28 → 40 (+43%) |
| bagging_temperature | 0.6 | 0.4 → 0.6 (+50%) |
| model_size_reg | 0.02 | 0.08 → 0.02 (-75%) |

#### Run 4 — Langevin Gradient Boosting

**Hypothesis:** CatBoost's Langevin boosting adds diffusion noise to gradients, acting as implicit regularization. This can improve generalization on volatile demand clusters without explicit L2/L1 penalties.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| langevin | true | false → true (new) |
| diffusion_temperature | 10000 | N/A → 10000 (new) |
| learning_rate | 0.005 | 0.008 → 0.005 |
| iterations | 5000 | 3000 → 5000 |
| l2_leaf_reg | 5.0 | 7.5 → 5.0 |
| bootstrap_type | Bayesian | MVS → Bayesian |
| bagging_temperature | 1.0 | 0.4 → 1.0 |

#### Run 5 — Ensemble-Optimized Blend

**Hypothesis:** Designed to complement LGBM champion in the meta-learner ensemble. Prioritizes low-bias predictions (slightly aggressive) since LGBM tends toward conservative forecasts. The ensemble benefits from model diversity.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| iterations | 3500 | 3000 → 3500 |
| learning_rate | 0.01 | 0.008 → 0.01 |
| depth | 12 | 10 → 12 |
| l2_leaf_reg | 4.0 | 7.5 → 4.0 |
| max_leaves | 191 | 127 → 191 |
| subsample | 0.9 | 0.85 → 0.9 |
| colsample_bylevel | 0.9 | 0.85 → 0.9 |
| reg_lambda | 2.0 | 3.5 → 2.0 |
| model_size_reg | 0.04 | 0.08 → 0.04 |

---

### XGBoost — 5 Runs (Expert Recommendations)

#### Run 1 — Current Production Baseline

Uses current `forecast_pipeline_config.yaml` XGBoost section as-is.

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| learning_rate | 0.05 |
| max_depth | 6 |
| min_child_weight | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

#### Run 2 — Lossguide + Heavy Regularization

**Hypothesis:** XGBoost's hist tree method with lossguide growth and heavy L1/L2 regularization brings it closer to LightGBM's leaf-wise splitting while controlling overfitting. The current baseline uses depth-wise (max_depth=6) which limits expressiveness.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| grow_policy | lossguide | depthwise → lossguide |
| max_leaves | 127 | N/A → 127 (new) |
| max_depth | 10 | 6 → 10 |
| n_estimators | 2000 | 500 → 2000 (+300%) |
| learning_rate | 0.015 | 0.05 → 0.015 (-70%) |
| reg_lambda | 5.0 | (default) → 5.0 |
| reg_alpha | 0.5 | (default) → 0.5 |
| gamma | 0.2 | (default) → 0.2 |
| min_child_weight | 15 | 5 → 15 (+200%) |
| max_bin | 256 | (default) → 256 |
| colsample_bylevel | 0.8 | N/A → 0.8 (new) |

#### Run 3 — DART Booster + Conservative Dropout

**Hypothesis:** DART (Dropouts meet Multiple Additive Regression Trees) addresses the over-specialization problem where later trees only fix residuals of earlier trees. By randomly dropping trees during boosting, DART produces a more balanced ensemble that generalizes better on unseen demand patterns.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| booster | dart | gbtree → dart |
| rate_drop | 0.08 | N/A → 0.08 (new) |
| skip_drop | 0.5 | N/A → 0.5 (new) |
| n_estimators | 2500 | 500 → 2500 |
| learning_rate | 0.012 | 0.05 → 0.012 |
| max_depth | 8 | 6 → 8 |
| subsample | 0.85 | 0.8 → 0.85 |
| colsample_bytree | 0.85 | 0.8 → 0.85 |
| reg_lambda | 3.0 | (default) → 3.0 |

#### Run 4 — Ultra-High Tree Count + Micro Learning Rate

**Hypothesis:** XGBoost's current baseline uses only 500 trees at LR=0.05, which is extremely aggressive compared to the LGBM champion (1500 trees at 0.02). Increasing to 3000 trees at 0.008 LR with histogram binning should close the accuracy gap with LGBM.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| n_estimators | 3000 | 500 → 3000 (+500%) |
| learning_rate | 0.008 | 0.05 → 0.008 (-84%) |
| max_depth | 10 | 6 → 10 |
| max_leaves | 95 | N/A → 95 |
| grow_policy | lossguide | depthwise → lossguide |
| max_bin | 256 | (default) → 256 |
| subsample | 0.82 | 0.8 → 0.82 |
| colsample_bytree | 0.8 | unchanged |
| colsample_bylevel | 0.85 | N/A → 0.85 |
| reg_lambda | 4.0 | (default) → 4.0 |
| gamma | 0.15 | (default) → 0.15 |

#### Run 5 — Champion Candidate Blend

**Hypothesis:** Merge best findings from Runs 2-4. Use lossguide with DART dropout, 256-bin histograms, and balanced regularization to maximize accuracy while maintaining ensemble diversity for champion selection.

| Parameter | Value | Delta from Run 1 |
|-----------|-------|-------------------|
| booster | dart | gbtree → dart |
| rate_drop | 0.05 | N/A → 0.05 |
| skip_drop | 0.6 | N/A → 0.6 |
| n_estimators | 2800 | 500 → 2800 |
| learning_rate | 0.01 | 0.05 → 0.01 |
| grow_policy | lossguide | depthwise → lossguide |
| max_leaves | 127 | N/A → 127 |
| max_depth | 10 | 6 → 10 |
| max_bin | 256 | (default) → 256 |
| min_child_weight | 12 | 5 → 12 |
| subsample | 0.85 | 0.8 → 0.85 |
| colsample_bytree | 0.8 | unchanged |
| colsample_bylevel | 0.85 | N/A → 0.85 |
| reg_lambda | 5.0 | (default) → 5.0 |
| reg_alpha | 0.3 | (default) → 0.3 |
| gamma | 0.12 | (default) → 0.12 |

---

## UI Design

### 4.1 Tab Structure

The Model Tuning tab is a single sidebar entry under the **Demand** section. It contains:

```
Model Tuning Tab
├── Model Selector Bar (LGBM | CatBoost | XGBoost pills)
├── Execution Lag Filter (All | Lag 0 | Lag 1 | Lag 2 | Lag 3 | Lag 4)
├── Sub-Tab Navigation
│   ├── Experiments (default)
│   ├── Comparison
│   ├── Cluster EDA
│   └── Feature Lab
└── Content Area (renders selected sub-tab)
```

### 4.2 Model Selector Bar

- Three pill buttons: **LGBM**, **CatBoost**, **XGBoost**
- Active pill has solid background + ring indicator
- Each pill shows a small status badge:
  - Green dot: has a promoted champion
  - Gray dot: no champion promoted yet
- Switching model resets the sub-tab to "Experiments" and reloads data

### 4.3 Execution Lag Filter

A horizontal segmented control immediately below the model selector:

```
[ All ] [ Lag 0 (1mo) ] [ Lag 1 (2mo) ] [ Lag 2 (3mo) ] [ Lag 3 (4mo) ] [ Lag 4 (5mo) ]
```

- **All** (default): Shows portfolio-level metrics (aggregated across all lags)
- **Lag 0 (1mo)**: Filters to 1-month-ahead predictions only (most accurate)
- **Lag 1 (2mo)**: 2-month-ahead predictions
- **Lag 2 (3mo)**: 3-month-ahead predictions
- **Lag 3 (4mo)**: 4-month-ahead predictions
- **Lag 4 (5mo)**: 5-month-ahead predictions (least accurate, longest horizon)

**Help Tooltip** (info icon next to the filter): "Execution lag is the number of months between when the forecast was generated and the target month. Lag 0 = 1-month ahead (most accurate), Lag 4 = 5 months ahead (least accurate)."

**First-Visit Callout**: A dismissible blue banner above the filter on first visit: "New: Filter accuracy by forecast horizon. Click a lag to see how model performance degrades over time."

When a specific lag is selected:
- KPI cards update to show accuracy/WAPE/bias for that lag only
- Run history table shows lag-specific metrics
- Comparison panel filters delta calculations to the selected lag
- Charts re-render with lag-filtered data
- The lag filter is passed as `?exec_lag=0` (or 1,2,3,4) query parameter to all API calls

### 4.4 Experiments Sub-Tab

This is the primary view. It has three sections:

#### 4.4.1 KPI Summary Strip

Four cards across the top:

| Card | Content |
|------|---------|
| **Best Accuracy** | Highest accuracy across all completed runs (for selected model + lag) |
| **Production Accuracy** | Accuracy of the currently promoted run (for selected lag) |
| **Total Runs** | Count of all runs for this model |
| **Active Runs** | Count of running/queued experiments |

Each card shows a trend arrow comparing to the previous best.

#### 4.4.2 Run History Table

A paginated, sortable table of all tuning runs for the selected model:

| Column | Description | Sortable |
|--------|-------------|----------|
| # | Run ID | Yes |
| Label | Human-readable name | Yes |
| Status | Badge: `running` (blue pulse), `completed` (green), `failed` (red), `queued` (yellow) | Yes |
| Accuracy % | Portfolio or lag-specific accuracy | Yes (default sort DESC) |
| WAPE % | Weighted Absolute Percentage Error | Yes |
| Bias % | Forecast bias | Yes |
| Predictions | Total prediction count | No |
| DFUs | Unique DFU count | No |
| Duration | Wall-clock time (formatted: "45m 23s") | Yes |
| Started | Relative time ("2h ago") with tooltip showing absolute time | Yes |
| Promoted | Crown icon if this run is the current champion | No |

**Row Interactions:**
- **Single click**: Selects row for comparison (first click = baseline in blue, second click = candidate in green)
- **Click selected row again**: Deselects it
- **Click "..." menu**: Shows actions: View Logs, View in Jobs, Promote, Delete
- **Click "View Logs"**: Opens a slide-over panel with the full execution log (streamed from `job_history.log`)

**Empty States:**
- **Zero runs**: Illustration + "No experiments yet for {model}. Click 'New Experiment' to start your first tuning run." + CTA button
- **Zero completed runs**: "All experiments are still running or have failed. Completed runs will appear here."
- **No promoted champion**: KPI card shows "Not yet promoted" instead of a number, no trend arrow
- **Legacy runs without lag data**: Show "--" in lag-filtered columns, tooltip: "This run predates lag tracking. Re-run to generate lag metrics."

**Table Toolbar:**
- **"New Experiment" button** (primary, prominent): Opens the Experiment Builder modal
- **Status filter** dropdown: All, Running, Completed, Failed
- **Date range** filter: Last 24h, Last 7d, Last 30d, All
- **Pagination**: 25/50/100 rows per page

**AI Tuning Advisor Integration:**
- The existing floating AI chat FAB is retained in the Tuning tab
- When the AI Advisor recommends parameters, recommendation cards include a "Use in Experiment Builder" button
- Clicking this button opens the Experiment Builder pre-filled with the AI's suggested values
- The AI Advisor can reference run history and suggest the next experiment based on prior results

#### 4.4.3 Experiment Builder Modal

A full-screen modal (or slide-over panel) for configuring and launching a new experiment. This is the core innovation of the spec.

**Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│  New Experiment — LightGBM                              [Close] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Experiment Label: [________________________]                   │
│  Notes:            [________________________]                   │
│                                                                 │
│  ── Template ──────────────────────────────────────────────────  │
│  ( ) Start from Production Baseline                             │
│  ( ) Expert: Aggressive Depth + Heavy Reg                       │
│  ( ) Expert: Ultra-Slow LR + Max Trees                          │
│  ( ) Expert: Feature Fraction Boost + Sparse Campaign           │
│  ( ) Expert: Balanced Champion Candidate                        │
│  ( ) Custom (blank slate)                                       │
│                                                                 │
│  ── Hyperparameters ──────────────────────────────────────────  │
│  ┌───────────────────────┬────────────┬──────────┬────────────┐ │
│  │ Parameter             │ Value      │ Default  │ Delta      │ │
│  ├───────────────────────┼────────────┼──────────┼────────────┤ │
│  │ n_estimators          │ [1500    ] │ 1500     │ —          │ │
│  │ learning_rate         │ [0.02   ]  │ 0.02     │ —          │ │
│  │ num_leaves            │ [127    ]  │ 127      │ —          │ │
│  │ max_depth             │ [-1     ]  │ -1       │ —          │ │
│  │ min_child_samples     │ [40     ]  │ 40       │ —          │ │
│  │ ...                   │ ...        │ ...      │ ...        │ │
│  └───────────────────────┴────────────┴──────────┴────────────┘ │
│                                                                 │
│  ── Training Config ──────────────────────────────────────────  │
│  Cluster Strategy:  [per_cluster ▼]  (per_cluster | global)    │
│  Recursive:         [✓]                                         │
│  SHAP Selection:    [✓]                                         │
│  SHAP Threshold:    [0.95]                                      │
│  SHAP Sample Size:  [500 ]                                      │
│                                                                 │
│  ── Summary ──────────────────────────────────────────────────  │
│  Model: lgbm_cluster | Strategy: per_cluster | Recursive: Yes   │
│  14 hyperparameters configured | 3 changed from production      │
│                                                                 │
│  [Cancel]                                    [Launch Experiment] │
└─────────────────────────────────────────────────────────────────┘
```

**Template Selection:**
- Selecting a template pre-fills all hyperparameters and training config
- "Start from Production Baseline" loads current `forecast_pipeline_config.yaml` values
- Expert templates load the recommended Run 2-5 configurations from this spec
- "Custom (from Production)" pre-fills with current production baseline values but marks all fields as "unchanged" (gray text, no delta). When the user edits a field, it becomes "changed" (bold text, delta shown). This avoids an empty form with validation errors.
- After selecting a template, users can modify any individual parameter
- The "Delta" column shows `+25%`, `-60%`, etc. relative to production baseline
- Each parameter has an info icon tooltip explaining what it does, its valid range, and the direction of effect (e.g., "Higher values = more regularization, reduces overfitting")
- Parameters are grouped into collapsible sections: **Tree Structure** (depth, leaves, estimators), **Regularization** (lambda, alpha, gamma, path_smooth), **Sampling** (subsample, colsample), **Advanced** (model-specific: DART, Langevin, etc.)

**Parameter Validation:**
- `n_estimators`: integer, min 100, max 10000
- `learning_rate`: float, min 0.001, max 0.5
- `num_leaves`: integer, min 2, max 512
- `max_depth`: integer, min -1 (unlimited), max 20
- `subsample`: float, min 0.1, max 1.0
- `reg_lambda`: float, min 0.0, max 100.0
- Invalid values show inline red validation messages
- Launch button disabled until all validations pass
- Launch button shows spinner immediately on click, re-enables only on error (prevents double-submit)

**Cross-Parameter Validation Rules:**
- CatBoost: `langevin=true` requires `bootstrap_type=Bayesian` (show inline error if MVS/Ordered selected)
- CatBoost: `subsample` is ignored when `bootstrap_type=Ordered` (show yellow warning: "subsample has no effect with Ordered bootstrap")
- CatBoost: `max_leaves` is disabled when `grow_policy=SymmetricTree` (grayed out with tooltip: "SymmetricTree uses depth to control tree size")
- XGBoost: `rate_drop` and `skip_drop` are hidden until `booster=dart` is selected (show info on booster field: "DART unlocks dropout parameters")
- XGBoost: `max_leaves` is disabled when `grow_policy=depthwise` (only active with `lossguide`)

**Launch Behavior:**
1. Validate all parameters
2. POST to `/model-tuning/{model}/experiments` with full config
3. API creates `lgbm_tuning_run` record (status=queued)
4. API submits resilient job via `JobManager.submit_job("model_tuning_run", {...})`
5. Modal closes, table refreshes, new row appears with status=queued
6. Toast notification: "Experiment 'Aggressive Depth + Heavy Reg' queued for LightGBM"
7. If API is unreachable: Modal shows error banner, does NOT close, user can retry

**Model-Specific Parameter Sets:**

The hyperparameter form adapts to the selected model:

| LGBM | CatBoost | XGBoost |
|------|----------|---------|
| n_estimators | iterations | n_estimators |
| learning_rate | learning_rate | learning_rate |
| num_leaves | max_leaves | max_leaves |
| max_depth | depth | max_depth |
| min_child_samples | min_data_in_leaf | min_child_weight |
| subsample | subsample | subsample |
| colsample_bytree | colsample_bylevel | colsample_bytree |
| reg_lambda | l2_leaf_reg | reg_lambda |
| reg_alpha | — | reg_alpha |
| path_smooth | — | gamma |
| max_bin | border_count | max_bin |
| bagging_freq | bagging_temperature | — |
| feature_fraction_bynode | random_strength | colsample_bylevel |
| min_gain_to_split | model_size_reg | — |
| — | grow_policy | grow_policy |
| — | bootstrap_type | booster |
| — | leaf_estimation_method | rate_drop (DART) |
| — | leaf_estimation_iterations | skip_drop (DART) |
| — | score_function | — |
| — | boost_from_average | — |
| — | max_ctr_complexity | — |
| — | langevin | — |
| — | diffusion_temperature | — |

### 4.5 Comparison Sub-Tab

Activated when 2 runs are selected from the Experiments table (or navigated to directly with `?baseline=X&candidate=Y` query params).

**Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Comparison: Run #12 (baseline) vs Run #15 (candidate)          │
│  Lag Filter: [All ▼]                                            │
├──────────────────────────┬──────────────────────────────────────┤
│                          │                                      │
│  VERDICT: IMPROVED       │  Accuracy Delta: +1.23 pp            │
│  (large badge)           │  WAPE Delta: -1.23 pp                │
│                          │  Bias Delta: +0.05 pp                │
├──────────────────────────┴──────────────────────────────────────┤
│                                                                 │
│  ── Accuracy by Execution Lag ─────────────────────────────────│
│  ┌──────┬──────────┬──────────┬──────────┐                     │
│  │ Lag  │ Baseline │ Candidate│ Delta    │                     │
│  ├──────┼──────────┼──────────┼──────────┤                     │
│  │ 0    │ 78.2%    │ 79.5%    │ +1.3 pp  │                     │
│  │ 1    │ 72.4%    │ 73.8%    │ +1.4 pp  │                     │
│  │ 2    │ 68.1%    │ 69.0%    │ +0.9 pp  │                     │
│  │ 3    │ 64.5%    │ 65.8%    │ +1.3 pp  │                     │
│  │ 4    │ 61.2%    │ 62.1%    │ +0.9 pp  │                     │
│  └──────┴──────────┴──────────┴──────────┘                     │
│                                                                 │
│  ── Lag Accuracy Curve (Chart) ────────────────────────────────│
│  [Line chart: X=lag(0-4), Y=accuracy%, two lines]              │
│                                                                 │
│  ── Per-Timeframe Accuracy ────────────────────────────────────│
│  [Grouped bar chart: A-J timeframes, baseline vs candidate]    │
│                                                                 │
│  ── Per-Cluster Accuracy ──────────────────────────────────────│
│  [ML Clusters table + Business Clusters table]                  │
│                                                                 │
│  ── Per-Month Accuracy ────────────────────────────────────────│
│  [Bar chart + table with month-by-month comparison]             │
│                                                                 │
│  ── Parameter Changes ─────────────────────────────────────────│
│  [Diff table: parameter | baseline | candidate | delta]         │
│                                                                 │
│  ── Feature Changes ───────────────────────────────────────────│
│  [Added: N | Removed: M | Common: K]                           │
│                                                                 │
│  ── Config Changes ────────────────────────────────────────────│
│  [cluster_strategy, recursive, shap_select diffs]               │
│                                                                 │
│                              [Promote Candidate] [Promote Base] │
└─────────────────────────────────────────────────────────────────┘
```

**Key Addition: Accuracy by Execution Lag**

This is a NEW section not present in the current UI. It shows:

1. **Table**: 5 rows (lag 0-4), columns: Lag, Baseline Accuracy, Candidate Accuracy, Delta, Verdict per Lag
2. **Line Chart**: Two lines (baseline=blue, candidate=green) plotting accuracy degradation across lags 0-4
3. **Insight Text**: Auto-generated observation, e.g., "Candidate improves most at Lag 1 (+1.4 pp) suggesting better short-term pattern capture"

### 4.6 Promote Modal (Enhanced)

When user clicks "Promote" on a run:

```
┌─────────────────────────────────────────────────────────────────┐
│  👑 Promote to Production                                       │
│                                                                 │
│  Run: #15 — "Balanced Champion Candidate"                       │
│  Model: LightGBM (lgbm_cluster)                                 │
│                                                                 │
│  ── Performance Summary ───────────────────────────────────────│
│  Accuracy: 73.45% | WAPE: 26.55% | Bias: +0.32%               │
│                                                                 │
│  ── Accuracy by Lag ───────────────────────────────────────────│
│  Lag 0: 79.5% | Lag 1: 73.8% | Lag 2: 69.0%                   │
│  Lag 3: 65.8% | Lag 4: 62.1%                                   │
│                                                                 │
│  ── What This Does ────────────────────────────────────────────│
│  1. Writes 14 hyperparameters to forecast_pipeline_config.yaml          │
│  2. Marks this run as the promoted champion for lgbm_cluster    │
│  3. Clears previous champion (Run #12)                          │
│  4. Records promotion in tuning_promotion_log                   │
│  5. Next champion selection will use these parameters            │
│                                                                 │
│  ── Parameters to Write ───────────────────────────────────────│
│  ┌──────────────────────┬──────────┬──────────┐                │
│  │ Parameter            │ Current  │ New      │                │
│  │ learning_rate        │ 0.020    │ 0.015    │                │
│  │ n_estimators         │ 1500     │ 2000     │                │
│  │ ...                  │ ...      │ ...      │                │
│  └──────────────────────┴──────────┴──────────┘                │
│                                                                 │
│  [Cancel]                                     [Confirm Promote] │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 Log Viewer Slide-Over

When user clicks "View Logs" on any run:

```
┌──────────────────────────────────────────────┐
│  Experiment Logs — Run #15                [X] │
│  Status: running ● | Duration: 12m 45s       │
│  Job ID: abc-123 | PID: 54321                 │
├──────────────────────────────────────────────┤
│  [2026-03-24 10:30:01] Starting LightGBM     │
│  backtest with per_cluster strategy           │
│  [2026-03-24 10:30:02] Loading config from    │
│  /tmp/tuning_run_15/forecast_pipeline_config.yaml     │
│  [2026-03-24 10:30:05] Generating 10          │
│  expanding-window timeframes (A-J)            │
│  [2026-03-24 10:30:08] Timeframe A:           │
│  train_end=2024-06, predict=2024-07..2025-04  │
│  [2026-03-24 10:30:08] Loading backtest data  │
│  from PostgreSQL (lag 0-4)...                 │
│  [2026-03-24 10:30:12] 580,000 rows loaded    │
│  [2026-03-24 10:30:12] Computing cluster      │
│  demand stats for 8 ML clusters...            │
│  [2026-03-24 10:31:45] Cluster 0: training    │
│  on 42,000 rows, 17 features                  │
│  [2026-03-24 10:31:45] LGBMRegressor.fit()    │
│  started — 1500 estimators, LR=0.015          │
│  [2026-03-24 10:33:22] Cluster 0 complete:    │
│  accuracy=74.2%, 5,200 predictions             │
│  [2026-03-24 10:33:23] Cluster 1: training... │
│  ...                                          │
│  [Auto-scrolls to bottom]                     │
├──────────────────────────────────────────────┤
│  [Copy Logs]  [Open in Jobs Tab]  [Download]  │
└──────────────────────────────────────────────┘
```

**Log Streaming:**
- Polls `GET /model-tuning/{model}/experiments/{run_id}/logs?offset=<N>` every 2 seconds
- Auto-scrolls to bottom (with "scroll lock" toggle to pause auto-scroll)
- Stops polling when status changes to completed/failed
- Shows real-time duration counter while running

### 4.8 Jobs Tab Integration

Tuning experiments appear in the Jobs tab with clear differentiation:

**Job Card in Active Jobs Panel:**

```
┌────────────────────────────────────────────────────┐
│ 🧪 LightGBM Tuning — "Aggressive Depth"           │
│ Job Type: model_tuning_run                          │
│ Status: running (35% — Timeframe D/J)               │
│ Started: 2 min ago | PID: 54321                     │
│ [View Logs]  [Cancel]                               │
└────────────────────────────────────────────────────┘
```

**Key Differentiators from Other Jobs:**
- **Icon**: Beaker/flask icon (not the generic gear icon)
- **Label Format**: `"{Model} Tuning — {run_label}"` (e.g., "LightGBM Tuning — Aggressive Depth")
- **Job Type**: `model_tuning_run` (distinct from legacy `tuning_backtest`)
- **Group**: `tuning` (queuing: only 1 tuning job per model runs at a time; different models can run in parallel)
- **Progress**: Shows timeframe progress (e.g., "Timeframe D/J — 40%")
- **Color**: Tuning jobs use a distinct accent color in the active jobs panel

**Job History in History Panel:**
- Filterable by `job_type = model_tuning_run`
- Shows model type badge (LGBM/CB/XGB) in the type column
- "View in Tuning" link navigates to Model Tuning tab with run selected

---

## API Design

### 5.1 Unified Router: `/model-tuning/{model}/`

Replace the current split between `lgbm_tuning.py` and `model_tuning.py` with a single parametrized router.

**Path prefix:** `/model-tuning/{model}` where `model` is one of `lgbm`, `catboost`, `xgboost`.

#### Router Layout

The router has been split from the legacy 1,798-LoC `api/routers/forecasting/unified_model_tuning.py` into a 15-module package at `api/routers/forecasting/tuning/`. All 15 endpoints below are preserved at the same `/model-tuning/{model}/*` paths -- the package's `__init__.py` re-exports a single `router` for `api/main.py` to mount.

| Sub-module | Endpoints owned |
|------------|------------------|
| `__init__.py` | Aggregates sub-routers, exposes the unified `router` symbol |
| `_helpers.py` | Shared helpers (model whitelist, run lookup, exec-lag filter SQL, JSON shaping) |
| `list.py` | `GET /experiments` |
| `detail.py` | `GET /experiments/{run_id}` |
| `create.py` | `POST /experiments` (create + launch) |
| `compare.py` | `GET /compare` |
| `cluster.py` | `GET /experiments/{run_id}/clusters` |
| `lag.py` | `GET /experiments/{run_id}/lags` |
| `logs.py` | `GET /experiments/{run_id}/logs` |
| `month.py` | `GET /experiments/{run_id}/months` |
| `promote.py` | `POST /experiments/{run_id}/promote` |
| `promote_results.py` | `GET /promoted` |
| `cancel_delete.py` | `POST /experiments/{run_id}/cancel`, `DELETE /experiments/{run_id}` |
| `templates.py` | `GET /templates` |
| `promotions.py` | `GET /promotions`, `POST /promotions/rollback` |

Importers should reference the package, not the module: `from api.routers.forecasting.tuning import router`. The legacy single-file path is no longer importable.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/model-tuning/{model}/experiments` | List runs with pagination, status/date filters |
| GET | `/model-tuning/{model}/experiments/{run_id}` | Full run detail + timeframe breakdown |
| POST | `/model-tuning/{model}/experiments` | **Create + launch** a new experiment |
| GET | `/model-tuning/{model}/experiments/{run_id}/clusters` | Per-cluster accuracy (filterable by exec_lag) |
| GET | `/model-tuning/{model}/experiments/{run_id}/months` | Per-month accuracy (filterable by exec_lag) |
| GET | `/model-tuning/{model}/experiments/{run_id}/lags` | Per-execution-lag accuracy breakdown |
| GET | `/model-tuning/{model}/experiments/{run_id}/logs` | Incremental log streaming (offset-based) |
| GET | `/model-tuning/{model}/compare` | Compare two runs with lag-level deltas |
| POST | `/model-tuning/{model}/experiments/{run_id}/promote` | Promote to production |
| GET | `/model-tuning/{model}/promoted` | Get currently promoted run |
| POST | `/model-tuning/{model}/experiments/{run_id}/cancel` | Cancel a running/queued experiment |
| DELETE | `/model-tuning/{model}/experiments/{run_id}` | Delete a completed/failed/cancelled experiment |
| GET | `/model-tuning/{model}/promotions` | List promotion audit trail |
| POST | `/model-tuning/{model}/promotions/rollback` | Revert to previous champion |
| GET | `/model-tuning/{model}/templates` | List available experiment templates |

### 5.2 Request/Response Models

#### POST `/model-tuning/{model}/experiments` — Create + Launch

**Request:**
```json
{
  "run_label": "Aggressive Depth + Heavy Reg",
  "notes": "Testing depth cap with increased L2 regularization",
  "template": "expert_aggressive_depth",
  "params": {
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": 10,
    "min_child_samples": 60,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "feature_fraction_bynode": 0.7,
    "reg_lambda": 3.5,
    "reg_alpha": 0.5,
    "path_smooth": 8.0,
    "max_bin": 127,
    "bagging_freq": 1,
    "min_gain_to_split": 0.01
  },
  "config": {
    "cluster_strategy": "per_cluster",
    "recursive": true,
    "shap_select": true,
    "shap_threshold": 0.95,
    "shap_sample_size": 500
  }
}
```

**Response (201 Created):**
```json
{
  "run_id": 15,
  "job_id": "abc-123-def-456",
  "status": "queued",
  "model": "lgbm",
  "run_label": "Aggressive Depth + Heavy Reg",
  "started_at": null,
  "message": "Experiment queued. Track progress in Jobs tab or via GET /model-tuning/lgbm/experiments/15/logs"
}
```

**Server-Side Behavior:**
1. Validate `model` path parameter
2. Validate all params against model-specific schema (ranges, types)
3. INSERT into `lgbm_tuning_run` with status='queued', params=JSONB, model_id='{model}_cluster'
4. Build temp `forecast_pipeline_config.yaml` with overrides
5. Submit job: `JobManager.submit_job("model_tuning_run", {run_id, model, config_path, run_label})`
6. Return 201 with run_id and job_id

#### GET `/model-tuning/{model}/experiments/{run_id}/lags` — NEW Endpoint

**Response:**
```json
{
  "run_id": 15,
  "model": "lgbm",
  "lags": [
    {"exec_lag": 0, "n_predictions": 116000, "accuracy_pct": 79.5, "wape": 20.5, "bias": 0.28},
    {"exec_lag": 1, "n_predictions": 116000, "accuracy_pct": 73.8, "wape": 26.2, "bias": 0.35},
    {"exec_lag": 2, "n_predictions": 116000, "accuracy_pct": 69.0, "wape": 31.0, "bias": 0.42},
    {"exec_lag": 3, "n_predictions": 116000, "accuracy_pct": 65.8, "wape": 34.2, "bias": 0.51},
    {"exec_lag": 4, "n_predictions": 116000, "accuracy_pct": 62.1, "wape": 37.9, "bias": 0.60}
  ]
}
```

#### GET `/model-tuning/{model}/compare` — Enhanced with Lag Deltas

**Query Params:** `baseline_id`, `candidate_id`, `exec_lag` (optional, 0-4 or omit for all)

**Response (additional fields vs current):**
```json
{
  "baseline": { "run_id": 12, "accuracy_pct": 72.22, "...": "..." },
  "candidate": { "run_id": 15, "accuracy_pct": 73.45, "...": "..." },
  "delta_accuracy": 1.23,
  "delta_wape": -1.23,
  "delta_bias": 0.05,
  "verdict": "improved",
  "per_lag": [
    {"exec_lag": 0, "baseline_acc": 78.2, "candidate_acc": 79.5, "delta_acc": 1.3, "baseline_wape": 21.8, "candidate_wape": 20.5, "delta_wape": -1.3, "baseline_bias": 0.25, "candidate_bias": 0.22, "delta_bias": -0.03},
    {"exec_lag": 1, "baseline_acc": 72.4, "candidate_acc": 73.8, "delta_acc": 1.4, "baseline_wape": 27.6, "candidate_wape": 26.2, "delta_wape": -1.4, "baseline_bias": 0.30, "candidate_bias": 0.28, "delta_bias": -0.02},
    {"exec_lag": 2, "baseline_acc": 68.1, "candidate_acc": 69.0, "delta_acc": 0.9, "baseline_wape": 31.9, "candidate_wape": 31.0, "delta_wape": -0.9, "baseline_bias": 0.38, "candidate_bias": 0.36, "delta_bias": -0.02},
    {"exec_lag": 3, "baseline_acc": 64.5, "candidate_acc": 65.8, "delta_acc": 1.3, "baseline_wape": 35.5, "candidate_wape": 34.2, "delta_wape": -1.3, "baseline_bias": 0.45, "candidate_bias": 0.42, "delta_bias": -0.03},
    {"exec_lag": 4, "baseline_acc": 61.2, "candidate_acc": 62.1, "delta_acc": 0.9, "baseline_wape": 38.8, "candidate_wape": 37.9, "delta_wape": -0.9, "baseline_bias": 0.55, "candidate_bias": 0.52, "delta_bias": -0.03}
  ],
  "per_cluster": ["...existing structure..."],
  "per_month": ["...existing structure..."],
  "per_timeframe": ["...existing structure..."],
  "param_diffs": ["...existing structure..."],
  "feature_diffs": {"added": 0, "removed": 0, "common": 17},
  "config_diffs": ["...existing structure..."]
}
```

#### GET `/model-tuning/{model}/templates`

**Response:**
```json
{
  "model": "lgbm",
  "templates": [
    {
      "id": "production_baseline",
      "label": "Production Baseline (Run 16)",
      "description": "Current production parameters — use as reference",
      "params": { "n_estimators": 1500, "learning_rate": 0.02, "..." : "..." },
      "config": { "cluster_strategy": "per_cluster", "recursive": true, "..." : "..." },
      "source": "algorithm_config"
    },
    {
      "id": "expert_aggressive_depth",
      "label": "Expert: Aggressive Depth + Heavy Reg",
      "description": "Cap depth at 10, halve leaves to 63, increase L2 to 3.5",
      "params": { "n_estimators": 1500, "max_depth": 10, "num_leaves": 63, "reg_lambda": 3.5, "..." : "..." },
      "config": { "cluster_strategy": "per_cluster", "recursive": true, "..." : "..." },
      "source": "expert"
    }
  ]
}
```

**Template Source:**
- `"algorithm_config"`: Reads live from `config/forecasting/forecast_pipeline_config.yaml`
- `"expert"`: Reads from `config/forecasting/tuning_templates.yaml` (new file, contains the 5 runs per model from this spec)
- `"custom"`: Empty — user fills everything manually

### 5.3 Backward Compatibility

The existing endpoints (`/lgbm-tuning/*`, `/catboost-tuning/*`, `/xgboost-tuning/*`) continue to work as aliases that redirect to the new unified router. This ensures existing frontend code and scripts continue to function during the migration.

```python
# In lgbm_tuning.py — add redirect aliases
@router.get("/lgbm-tuning/runs")
async def legacy_list_runs(...):
    """Redirect to unified router."""
    return await unified_list_experiments("lgbm", ...)
```

---

## Database Schema Changes

### 6.1 New Table: `lgbm_tuning_lag`

Per-execution-lag accuracy breakdown for each run.

```sql
CREATE TABLE IF NOT EXISTS lgbm_tuning_lag (
    id          SERIAL PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id) ON DELETE CASCADE,
    exec_lag    SMALLINT NOT NULL CHECK (exec_lag BETWEEN 0 AND 4),
    n_predictions INTEGER NOT NULL DEFAULT 0,
    n_dfus      INTEGER NOT NULL DEFAULT 0,
    accuracy_pct NUMERIC(6, 2),
    wape        NUMERIC(6, 2),
    bias        NUMERIC(8, 4),
    UNIQUE (run_id, exec_lag)
);

CREATE INDEX idx_tuning_lag_run ON lgbm_tuning_lag(run_id);
```

### 6.2 New Table: `tuning_promotion_log`

Audit trail of all promotions across models.

```sql
CREATE TABLE IF NOT EXISTS tuning_promotion_log (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id),
    model_id        VARCHAR(50) NOT NULL,
    promoted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_by     VARCHAR(100),  -- user or 'system'
    previous_run_id INTEGER REFERENCES lgbm_tuning_run(run_id),
    params_written  JSONB NOT NULL,
    accuracy_pct    NUMERIC(6, 2),
    wape            NUMERIC(6, 2),
    bias            NUMERIC(8, 4),
    notes           TEXT
);

CREATE INDEX idx_promotion_log_model ON tuning_promotion_log(model_id, promoted_at DESC);
```

### 6.3 Alter `lgbm_tuning_run` — Add columns and fix status CHECK

```sql
-- Add new columns
ALTER TABLE lgbm_tuning_run
    ADD COLUMN IF NOT EXISTS job_id VARCHAR(100),
    ADD COLUMN IF NOT EXISTS template_id VARCHAR(100);

-- Extend status CHECK to include 'queued' and 'cancelled'
ALTER TABLE lgbm_tuning_run DROP CONSTRAINT IF EXISTS lgbm_tuning_run_status_check;
ALTER TABLE lgbm_tuning_run ADD CONSTRAINT lgbm_tuning_run_status_check
    CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'));

CREATE INDEX IF NOT EXISTS idx_tuning_run_job ON lgbm_tuning_run(job_id) WHERE job_id IS NOT NULL;
```

### 6.4 New Table: `lgbm_tuning_lag_cluster`

Per-lag-per-cluster accuracy for deep drill-down.

```sql
CREATE TABLE IF NOT EXISTS lgbm_tuning_lag_cluster (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES lgbm_tuning_run(run_id) ON DELETE CASCADE,
    exec_lag        SMALLINT NOT NULL CHECK (exec_lag BETWEEN 0 AND 4),
    cluster_type    TEXT NOT NULL,
    cluster_value   TEXT NOT NULL,
    n_predictions   INTEGER NOT NULL DEFAULT 0,
    accuracy_pct    NUMERIC(6, 2),
    wape            NUMERIC(6, 2),
    bias            NUMERIC(8, 4),
    UNIQUE (run_id, exec_lag, cluster_type, cluster_value)
);

CREATE INDEX idx_tuning_lag_cluster_run ON lgbm_tuning_lag_cluster(run_id);
```

---

## Job Integration

### 7.1 New Job Type: `model_tuning_run`

Register in `JOB_TYPE_REGISTRY`:

```python
"model_tuning_run": JobTypeDef(
    type_id="model_tuning_run",
    label="Model Tuning Experiment",
    description="Run backtest with custom hyperparameters and register results",
    group="tuning",
    callable=_run_model_tuning_experiment,
    params_schema={
        "run_id": 0,
        "model": "",          # "lgbm", "catboost", "xgboost"
        "config_path": "",    # path to temp forecast_pipeline_config.yaml
        "run_label": "",
    },
)
```

### 7.2 Job Callable: `_run_model_tuning_experiment`

```python
def _run_model_tuning_experiment(params, progress_cb, cancel_event, job_id):
    """
    1. Update lgbm_tuning_run.job_id = job_id, status = 'running'
    2. Build subprocess command:
       uv run python scripts/run_backtest.py --model {model} --config {config_path}
    3. Run via _run_subprocess() with PID tracking + log streaming
    4. On success:
       a. Read backtest_metadata.json
       b. Update lgbm_tuning_run with accuracy, WAPE, bias, status='completed'
       c. Insert timeframe breakdowns into lgbm_tuning_timeframe
       d. Insert cluster breakdowns into lgbm_tuning_cluster
       e. Insert month breakdowns into lgbm_tuning_month
       f. Compute per-lag accuracy from backtest_predictions_all_lags.csv
       g. Insert lag breakdowns into lgbm_tuning_lag
       h. Insert lag-cluster breakdowns into lgbm_tuning_lag_cluster
    5. On failure:
       a. Update lgbm_tuning_run with status='failed', error in notes
    6. Clean up temp config file
    """
```

### 7.3 Per-Group Concurrency

The `tuning` job group allows **one job per model type**. This is enforced by extending the group to be model-specific:

- LGBM tuning jobs use group `tuning_lgbm`
- CatBoost tuning jobs use group `tuning_catboost`
- XGBoost tuning jobs use group `tuning_xgboost`

This allows parallel execution across models (LGBM + CatBoost can run simultaneously) while preventing two LGBM experiments from colliding on the same output directory.

### 7.4 Detailed Logging

The backtest subprocess must produce detailed, structured logs. The job runner captures stdout and streams to both `job_history.log` (DB) and the log viewer.

**Required Log Points:**

| Stage | Log Message | Progress % |
|-------|-------------|-----------|
| Start | `[TUNING] Starting {model} experiment: "{run_label}" (run_id={N})` | 0% |
| Config | `[TUNING] Config written to {config_path}` | 2% |
| Config | `[TUNING] Hyperparameters: {json_summary}` | 2% |
| Config | `[TUNING] Strategy: {cluster_strategy}, Recursive: {bool}, SHAP: {bool}` | 2% |
| Data Load | `[TUNING] Loading backtest data from PostgreSQL...` | 5% |
| Data Load | `[TUNING] {N} rows loaded across {M} DFUs, lags 0-4` | 8% |
| Timeframe | `[TUNING] Timeframe {label} ({i}/{total}): train_end={date}, predict={start}..{end}` | 10-90% (proportional) |
| Cluster | `[TUNING] Cluster {id}: {N} rows, {M} features, training...` | (within timeframe %) |
| Cluster | `[TUNING] Cluster {id}: accuracy={X}%, {N} predictions, {duration}s` | (within timeframe %) |
| Feature | `[TUNING] SHAP selection: {N} → {M} features (threshold={T})` | (within timeframe %) |
| Post | `[TUNING] Post-processing: dedup by forecast_ck, expand to all lags` | 92% |
| Save | `[TUNING] Saving predictions: {N} exec-lag rows, {M} all-lag rows` | 95% |
| Metrics | `[TUNING] Portfolio accuracy: {X}% | WAPE: {Y}% | Bias: {Z}%` | 97% |
| Lag | `[TUNING] Per-lag accuracy: Lag0={a}%, Lag1={b}%, Lag2={c}%, Lag3={d}%, Lag4={e}%` | 98% |
| Complete | `[TUNING] Experiment complete. Duration: {Xm Ys}. Results registered as run_id={N}` | 100% |

### 7.5 Jobs Tab Display Rules

To prevent confusion between tuning jobs and other jobs:

1. **Label Format**: `"{Model} Tuning — {run_label}"` (always includes model name)
2. **Icon**: Beaker icon (`Flask` from lucide-react)
3. **Type Badge**: Shows model pill (blue for LGBM, green for CatBoost, orange for XGBoost)
4. **Progress Format**: `"Timeframe {X}/{N} — {pct}%"` (not generic "Running...")
5. **Quick Link**: "Open in Tuning Tab" button navigates to Model Tuning tab with run auto-selected
6. **Group Label**: Shows as "Tuning (LightGBM)" not just "Tuning"

---

## Promotion Workflow

### 8.1 Who Can Promote

Any authenticated user (or unauthenticated if API_KEY is unset). No role-based restrictions in v1.

### 8.2 Promotion Steps

1. User clicks "Promote" on a completed run (from Experiments table or Comparison panel)
2. **Pre-flight check**: API verifies run is `status='completed'` and `accuracy_pct IS NOT NULL`
3. **Promote Modal** opens showing:
   - Run metrics (accuracy, WAPE, bias) at portfolio and per-lag level
   - Parameter diff table (current production vs new)
   - Clear description of what promotion does
4. User clicks "Confirm Promote"
5. **API performs atomic transaction:**
   a. Load current `forecast_pipeline_config.yaml`
   b. Record current promoted run_id as `previous_run_id` in promotion log
   c. Update `forecast_pipeline_config.yaml` with new params (model-specific section only)
   d. Clear `is_promoted` on previous champion (same model_id)
   e. Set `is_promoted=true, promoted_at=NOW()` on new champion
   f. INSERT into `tuning_promotion_log` with full audit trail
   g. COMMIT
6. **UI updates:**
   - Crown icon appears on promoted run row
   - Previous champion loses crown icon
   - Toast: "Run #15 promoted as LightGBM champion"
   - KPI card "Production Accuracy" updates

### 8.3 Promotion Lineage

The `tuning_promotion_log` table creates a full audit trail:

```
Run #1 (baseline) → Run #8 (enhanced_reg_v1) → Run #12 (champion_blend) → Run #15 (balanced)
```

This is queryable via `GET /model-tuning/{model}/promotions` (returns ordered list).

### 8.4 Champion Pipeline Integration

After promotion, the next `make champion-all` run will:
1. Read `forecast_pipeline_config.yaml` (which now has promoted params)
2. Run backtests with the new params
3. Meta-learner selects per-DFU winners using the updated model
4. Production forecasts reflect the promoted champion

The promotion does NOT automatically trigger a champion pipeline run. Users must explicitly run `make champion-all` or submit a `champion_select` job.

---

## Execution-Lag Filtering

### 9.1 Data Source

Execution lag is computed during backtest by `assign_execution_lag()` in `backtest_framework.py`. Each prediction row has an `execution_lag` value (0-4) representing the number of months between the forecast issuance date and the target month.

The all-lags CSV (`backtest_predictions_all_lags.csv`) contains rows for lags 0-4, each with its own accuracy metrics.

### 9.2 Storage

After a backtest completes, the job callable computes per-lag accuracy by grouping the all-lags predictions:

```python
all_lags_df = pd.read_csv(f"{output_dir}/backtest_predictions_all_lags.csv")
for lag in range(5):
    lag_df = all_lags_df[all_lags_df["lag"] == lag]
    abs_error = (lag_df["basefcst_pref"] - lag_df["tothist_dmd"]).abs().sum()
    sum_actual = abs(lag_df["tothist_dmd"].sum())
    accuracy = 100 - (100 * abs_error / sum_actual) if sum_actual > 0 else None
    wape = (abs_error / sum_actual * 100) if sum_actual > 0 else None
    bias = (lag_df["basefcst_pref"].sum() / lag_df["tothist_dmd"].sum()) - 1 if sum_actual > 0 else None
    # INSERT INTO lgbm_tuning_lag (run_id, exec_lag, n_predictions, n_dfus, accuracy_pct, wape, bias)
```

**Note:** The archive CSV does NOT have an `abs_error` column. Compute it inline from `basefcst_pref` and `tothist_dmd`. Guard against zero-actual division.

Similarly, per-lag-per-cluster accuracy is computed and stored in `lgbm_tuning_lag_cluster`.

### 9.3 API Filtering

All list/compare endpoints accept an optional `exec_lag` query parameter:

- `GET /model-tuning/lgbm/experiments?exec_lag=0` — Returns runs with lag-0 metrics
- `GET /model-tuning/lgbm/compare?baseline_id=12&candidate_id=15&exec_lag=2` — Compares at lag 2

When `exec_lag` is specified:
- `accuracy_pct`, `wape`, `bias` in the response come from `lgbm_tuning_lag` (not the portfolio-level `lgbm_tuning_run`)
- Cluster breakdowns come from `lgbm_tuning_lag_cluster`

When `exec_lag` is omitted:
- Portfolio-level metrics from `lgbm_tuning_run` are returned (same as current behavior)

### 9.4 UI Interaction

The execution lag filter is a **global filter** at the tab level. Changing it affects:
- KPI summary cards
- Run history table metric columns
- Comparison panel (all sections)
- Cluster EDA panel
- Feature Lab panel

The selected lag is stored in component state and passed to all queries as `exec_lag` parameter. React Query automatically invalidates and refetches when the lag filter changes.

---

## Resilience & Error Handling

### 10.1 API Drop Resilience

The experiment system is designed to survive API restarts at any point:

| Failure Scenario | Behavior |
|-----------------|----------|
| API drops during experiment submission | Modal shows error banner, user retries. No orphan records (DB insert + job submit are sequential, not atomic — if insert succeeds but job fails, a cleanup endpoint marks the run as failed). |
| API drops during backtest execution | Subprocess continues (start_new_session=True). On API restart, `recover_stale_jobs()` finds the PID, checks if alive, re-adopts the monitoring thread. |
| API drops after backtest, before results registration | On restart, recovery finds PID is dead. Marks job as completed (subprocess exited cleanly) or failed. A background task checks for unregistered results by scanning `data/backtest/` for metadata files with `run_id` in their path. |
| Frontend loses connection during experiment | React Query retries with exponential backoff. Polling stops, but state is preserved. On reconnect, data automatically refetches. |
| Backtest subprocess crashes | Job runner detects non-zero exit code. Updates `lgbm_tuning_run.status='failed'` with stderr as notes. Updates `job_history` with error details. |
| Disk full during backtest | Subprocess writes to temp dir first. On disk-full error, subprocess exits with code 1. Job runner captures the error and updates status. |

### 10.2 Idempotency

- Each experiment has a unique `run_id` (serial). Re-submitting the same parameters creates a new run.
- Promotion is idempotent: promoting the same run twice is a no-op (already promoted).
- Log streaming is offset-based: re-reading from offset 0 replays the full log.

### 10.3 Concurrent Experiment Safety

- Per-model group concurrency prevents two LGBM experiments from running simultaneously
- Different models can run in parallel (LGBM + CatBoost)
- Each experiment writes to a unique temp config path: `/tmp/tuning_run_{run_id}/forecast_pipeline_config.yaml`
- Backtest output goes to model-specific dirs: `data/backtest/{model_id}/`
- File locking on `forecast_pipeline_config.yaml` during promotion prevents race conditions

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        UI: Experiment Builder                         │
│  User selects template → adjusts params → clicks Launch               │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ POST /model-tuning/{model}/experiments
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        API: Unified Router                            │
│  1. Validate params against model schema                              │
│  2. INSERT lgbm_tuning_run (status='queued', params=JSONB)           │
│  3. Write temp forecast_pipeline_config.yaml with overrides                   │
│  4. JobManager.submit_job("model_tuning_run", {...})                 │
│  5. Return 201 {run_id, job_id}                                      │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ APScheduler dispatches
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Job Runner (Subprocess Isolation)                   │
│  1. Popen(run_backtest.py --model {m} --config {path})               │
│  2. start_new_session=True (survives API restart)                     │
│  3. PID stored in job_history.pid                                     │
│  4. stdout → DB log (every 2s or 20 lines)                           │
│  5. Progress callbacks: Timeframe X/N — Y%                           │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ Subprocess completes
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Results Registration                                │
│  1. Read backtest_metadata.json → portfolio accuracy, WAPE, bias      │
│  2. Read backtest_predictions.csv → per-cluster, per-month metrics    │
│  3. Read backtest_predictions_all_lags.csv → per-lag metrics          │
│  4. UPDATE lgbm_tuning_run (status='completed', metrics)             │
│  5. INSERT lgbm_tuning_timeframe (10 rows, A-J)                     │
│  6. INSERT lgbm_tuning_cluster (ML + business clusters)              │
│  7. INSERT lgbm_tuning_month (per-month rows)                        │
│  8. INSERT lgbm_tuning_lag (5 rows, lag 0-4)                         │
│  9. INSERT lgbm_tuning_lag_cluster (lag × cluster rows)              │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ User reviews in UI
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    UI: Comparison & Promotion                         │
│  1. Select baseline + candidate runs                                  │
│  2. View per-lag, per-cluster, per-month deltas                      │
│  3. Click Promote → modal → confirm                                  │
│  4. POST /model-tuning/{model}/experiments/{id}/promote              │
│  5. API writes to forecast_pipeline_config.yaml + DB flags                   │
│  6. Promotion logged in tuning_promotion_log                         │
│  7. Next champion pipeline uses promoted params                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### 12.1 New Config File: `config/forecasting/tuning_templates.yaml`

Contains expert-recommended experiment templates per model.

```yaml
templates:
  lgbm:
    - id: production_baseline
      label: "Production Baseline (Run 16)"
      description: "Current production parameters"
      source: algorithm_config  # read live from forecast_pipeline_config.yaml
    - id: expert_aggressive_depth
      label: "Expert: Aggressive Depth + Heavy Reg"
      description: "Cap depth at 10, halve leaves, increase L2 to 3.5"
      params:
        max_depth: 10
        num_leaves: 63
        reg_lambda: 3.5
        reg_alpha: 0.5
        path_smooth: 8.0
        min_child_samples: 60
    - id: expert_ultra_slow_lr
      label: "Expert: Ultra-Slow LR + Max Trees"
      description: "LR=0.008 with 3000 trees for residual capture"
      params:
        learning_rate: 0.008
        n_estimators: 3000
        subsample: 0.85
        colsample_bytree: 0.85
    - id: expert_sparse_demand
      label: "Expert: Feature Fraction + Sparse Campaign"
      description: "High node-level features with sparse demand regularization"
      params:
        feature_fraction_bynode: 0.9
        colsample_bytree: 0.9
        min_child_samples: 100
        min_gain_to_split: 0.05
        reg_alpha: 1.0
        path_smooth: 12.0
    - id: expert_balanced_champion
      label: "Expert: Balanced Champion Candidate"
      description: "Moderate depth cap, slower LR, balanced regularization"
      params:
        learning_rate: 0.015
        n_estimators: 2000
        max_depth: 12
        num_leaves: 95
        reg_lambda: 2.5
        reg_alpha: 0.3
        feature_fraction_bynode: 0.8
        path_smooth: 6.0
        min_child_samples: 50

  catboost:
    - id: production_baseline
      label: "Production Baseline"
      source: algorithm_config
    - id: expert_ordered_symmetric
      label: "Expert: Ordered Boosting + Symmetric Trees"
      description: "Symmetric trees with ordered boosting for temporal data"
      params:
        grow_policy: SymmetricTree
        depth: 8
        bootstrap_type: Ordered
        iterations: 4000
        learning_rate: 0.006
        random_strength: 1.0
    - id: expert_high_border
      label: "Expert: High Border Count + Reduced Leaf Reg"
      description: "Finer split resolution with 128 borders"
      params:
        border_count: 128
        l2_leaf_reg: 3.0
        min_data_in_leaf: 40
        bagging_temperature: 0.6
        model_size_reg: 0.02
    - id: expert_langevin
      label: "Expert: Langevin Gradient Boosting"
      description: "Diffusion noise for implicit regularization"
      params:
        langevin: true
        diffusion_temperature: 10000
        learning_rate: 0.005
        iterations: 5000
        l2_leaf_reg: 5.0
        bootstrap_type: Bayesian
        bagging_temperature: 1.0
    - id: expert_ensemble_optimized
      label: "Expert: Ensemble-Optimized Blend"
      description: "Low-bias params to complement LGBM in meta-learner"
      params:
        iterations: 3500
        learning_rate: 0.01
        depth: 12
        l2_leaf_reg: 4.0
        max_leaves: 191
        subsample: 0.9
        colsample_bylevel: 0.9
        reg_lambda: 2.0
        model_size_reg: 0.04

  xgboost:
    - id: production_baseline
      label: "Production Baseline"
      source: algorithm_config
    - id: expert_lossguide_heavy_reg
      label: "Expert: Lossguide + Heavy Regularization"
      description: "Leaf-wise splitting with L1/L2 regularization"
      params:
        grow_policy: lossguide
        max_leaves: 127
        max_depth: 10
        n_estimators: 2000
        learning_rate: 0.015
        reg_lambda: 5.0
        reg_alpha: 0.5
        gamma: 0.2
        min_child_weight: 15
        max_bin: 256
        colsample_bylevel: 0.8
    - id: expert_dart
      label: "Expert: DART Booster + Conservative Dropout"
      description: "Tree dropout for balanced ensemble generalization"
      params:
        booster: dart
        rate_drop: 0.08
        skip_drop: 0.5
        n_estimators: 2500
        learning_rate: 0.012
        max_depth: 8
        subsample: 0.85
        colsample_bytree: 0.85
        reg_lambda: 3.0
    - id: expert_ultra_high_trees
      label: "Expert: Ultra-High Trees + Micro LR"
      description: "3000 trees at 0.008 LR — close the gap with LGBM"
      params:
        n_estimators: 3000
        learning_rate: 0.008
        max_depth: 10
        max_leaves: 95
        grow_policy: lossguide
        max_bin: 256
        subsample: 0.82
        colsample_bylevel: 0.85
        reg_lambda: 4.0
        gamma: 0.15
    - id: expert_champion_blend
      label: "Expert: Champion Candidate Blend"
      description: "DART + lossguide + 256 bins + balanced reg"
      params:
        booster: dart
        rate_drop: 0.05
        skip_drop: 0.6
        n_estimators: 2800
        learning_rate: 0.01
        grow_policy: lossguide
        max_leaves: 127
        max_depth: 10
        max_bin: 256
        min_child_weight: 12
        subsample: 0.85
        colsample_bylevel: 0.85
        reg_lambda: 5.0
        reg_alpha: 0.3
        gamma: 0.12
```

### 12.2 Updated Config File: `config/model_tuning_config.yaml`

```yaml
model_tuning:
  # Verdict thresholds
  improved_min_delta_accuracy: 0.05    # pp
  degraded_max_delta_accuracy: -0.05   # pp

  # Job concurrency
  max_concurrent_per_model: 1
  job_timeout_seconds: 7200  # 2 hours

  # Log streaming
  log_flush_interval_seconds: 5
  log_flush_line_count: 20

  # Temp config directory
  temp_config_dir: /tmp/tuning_runs

  # Backup
  backup_dir: data/tuning/backups
  backup_enabled: true

  # Parameter validation ranges (per model)
  param_ranges:
    lgbm:
      n_estimators: { min: 100, max: 10000, type: int }
      learning_rate: { min: 0.001, max: 0.5, type: float }
      num_leaves: { min: 2, max: 512, type: int }
      max_depth: { min: -1, max: 20, type: int }
      min_child_samples: { min: 1, max: 500, type: int }
      subsample: { min: 0.1, max: 1.0, type: float }
      colsample_bytree: { min: 0.1, max: 1.0, type: float }
      feature_fraction_bynode: { min: 0.1, max: 1.0, type: float }
      reg_lambda: { min: 0.0, max: 100.0, type: float }
      reg_alpha: { min: 0.0, max: 100.0, type: float }
      path_smooth: { min: 0.0, max: 100.0, type: float }
      max_bin: { min: 15, max: 512, type: int }
      bagging_freq: { min: 0, max: 100, type: int }
      min_gain_to_split: { min: 0.0, max: 10.0, type: float }
    catboost:
      iterations: { min: 100, max: 10000, type: int }
      learning_rate: { min: 0.001, max: 0.5, type: float }
      depth: { min: 1, max: 16, type: int }
      l2_leaf_reg: { min: 0.0, max: 100.0, type: float }
      subsample: { min: 0.1, max: 1.0, type: float }
      border_count: { min: 1, max: 512, type: int }
      random_strength: { min: 0.0, max: 100.0, type: float }
      min_data_in_leaf: { min: 1, max: 500, type: int }
      colsample_bylevel: { min: 0.1, max: 1.0, type: float }
      bagging_temperature: { min: 0.0, max: 100.0, type: float }
      max_leaves: { min: 2, max: 512, type: int }
      model_size_reg: { min: 0.0, max: 10.0, type: float }
      reg_lambda: { min: 0.0, max: 100.0, type: float }
      max_ctr_complexity: { min: 1, max: 10, type: int }
      diffusion_temperature: { min: 1, max: 100000, type: int }
    xgboost:
      n_estimators: { min: 100, max: 10000, type: int }
      learning_rate: { min: 0.001, max: 0.5, type: float }
      max_depth: { min: 0, max: 20, type: int }
      min_child_weight: { min: 0, max: 500, type: float }
      subsample: { min: 0.1, max: 1.0, type: float }
      colsample_bytree: { min: 0.1, max: 1.0, type: float }
      colsample_bylevel: { min: 0.1, max: 1.0, type: float }
      reg_lambda: { min: 0.0, max: 100.0, type: float }
      reg_alpha: { min: 0.0, max: 100.0, type: float }
      gamma: { min: 0.0, max: 100.0, type: float }
      max_bin: { min: 15, max: 1024, type: int }
      max_leaves: { min: 0, max: 512, type: int }
      rate_drop: { min: 0.0, max: 1.0, type: float }
      skip_drop: { min: 0.0, max: 1.0, type: float }
```

---

## Testing Spec

### 13.1 Backend Tests — API Router

**File:** `tests/api/test_unified_model_tuning.py`

| Test | Description | Key Assertions |
|------|-------------|----------------|
| `test_list_experiments_lgbm` | GET /model-tuning/lgbm/experiments returns paginated runs | Status 200, response has `runs` array, pagination metadata |
| `test_list_experiments_catboost` | Same for CatBoost | Status 200, model_id filter applies |
| `test_list_experiments_xgboost` | Same for XGBoost | Status 200, model_id filter applies |
| `test_list_experiments_invalid_model` | GET /model-tuning/invalid/experiments | Status 422 or 400 |
| `test_list_experiments_status_filter` | GET with `?status=completed` | Only completed runs returned |
| `test_list_experiments_lag_filter` | GET with `?exec_lag=0` | Accuracy comes from lgbm_tuning_lag not lgbm_tuning_run |
| `test_get_experiment_detail` | GET /model-tuning/lgbm/experiments/1 | Full run + timeframes |
| `test_get_experiment_not_found` | GET /model-tuning/lgbm/experiments/999 | Status 404 |
| `test_create_experiment_lgbm` | POST with valid LGBM params | Status 201, run_id + job_id returned |
| `test_create_experiment_catboost` | POST with valid CatBoost params | Status 201, correct model_id |
| `test_create_experiment_xgboost` | POST with valid XGBoost params | Status 201, correct model_id |
| `test_create_experiment_invalid_params` | POST with out-of-range learning_rate | Status 422, validation error |
| `test_create_experiment_missing_label` | POST without run_label | Status 422 |
| `test_create_experiment_unknown_param` | POST with unknown param key | Ignored (only known keys stored) |
| `test_get_experiment_lags` | GET /model-tuning/lgbm/experiments/1/lags | 5 rows (lag 0-4), each has accuracy/WAPE/bias |
| `test_get_experiment_lags_empty` | GET lags for run with no lag data | Status 200, empty array |
| `test_get_experiment_clusters_with_lag` | GET /clusters?exec_lag=2 | Cluster data filtered to lag 2 |
| `test_get_experiment_months` | GET /months | Per-month accuracy |
| `test_get_experiment_logs` | GET /logs?offset=0 | Returns log text, next_offset |
| `test_get_experiment_logs_incremental` | GET /logs?offset=500 | Returns only new log text after offset |
| `test_compare_runs` | GET /compare?baseline_id=1&candidate_id=2 | Delta metrics, verdict, per_lag array |
| `test_compare_runs_with_lag` | GET /compare?baseline_id=1&candidate_id=2&exec_lag=0 | Deltas computed at lag 0 only |
| `test_compare_runs_not_found` | GET /compare?baseline_id=1&candidate_id=999 | Status 404 |
| `test_compare_runs_same_id` | GET /compare?baseline_id=1&candidate_id=1 | Status 400 |
| `test_promote_run` | POST /experiments/1/promote | Status 200, is_promoted=true, algorithm_config updated |
| `test_promote_run_not_completed` | POST /experiments/{running_run}/promote | Status 400, "only completed runs" |
| `test_promote_run_clears_previous` | Promote run B after run A is promoted | Run A.is_promoted=false, Run B.is_promoted=true |
| `test_promote_creates_audit_log` | After promotion | tuning_promotion_log has 1 row |
| `test_get_promoted_run` | GET /model-tuning/lgbm/promoted | Returns promoted run or 404 |
| `test_get_templates` | GET /model-tuning/lgbm/templates | Returns 5 templates |
| `test_get_templates_catboost` | GET /model-tuning/catboost/templates | CatBoost-specific templates |
| `test_legacy_endpoint_redirect` | GET /lgbm-tuning/runs | Same data as GET /model-tuning/lgbm/experiments |
| `test_concurrent_model_experiments` | Submit LGBM + CatBoost simultaneously | Both accepted (different groups) |
| `test_concurrent_same_model` | Submit two LGBM experiments | First runs, second queues |

### 13.2 Backend Tests — Job Integration

**File:** `tests/unit/test_model_tuning_job.py`

| Test | Description |
|------|-------------|
| `test_job_type_registered` | `model_tuning_run` exists in JOB_TYPE_REGISTRY |
| `test_job_creates_temp_config` | Temp config file created with correct overrides |
| `test_job_runs_backtest_subprocess` | Subprocess invoked with correct args |
| `test_job_registers_results_on_success` | lgbm_tuning_run updated with metrics |
| `test_job_inserts_lag_breakdowns` | lgbm_tuning_lag has 5 rows after completion |
| `test_job_inserts_lag_cluster_breakdowns` | lgbm_tuning_lag_cluster populated |
| `test_job_handles_subprocess_failure` | Status set to 'failed', error captured |
| `test_job_cleans_up_temp_config` | Temp file deleted after completion |
| `test_job_survives_api_restart` | PID recovery re-adopts running subprocess |
| `test_job_logs_streamed_to_db` | job_history.log contains subprocess output |
| `test_job_progress_updates` | progress_pct updates during timeframe processing |
| `test_job_cancellation_kills_process` | Cancel sends SIGTERM to process group |

### 13.3 Backend Tests — SQL Schema

**File:** `tests/unit/test_tuning_schema.py`

| Test | Description |
|------|-------------|
| `test_lgbm_tuning_lag_unique_constraint` | Duplicate (run_id, exec_lag) rejected |
| `test_lgbm_tuning_lag_check_constraint` | exec_lag outside 0-4 rejected |
| `test_promotion_log_insert` | Full audit row inserted |
| `test_promotion_log_previous_run_fk` | previous_run_id references valid run |
| `test_run_job_id_column` | job_id stored and queryable |
| `test_run_template_id_column` | template_id stored |

### 13.4 Frontend Tests — Experiments Tab

**File:** `frontend/src/tabs/__tests__/ModelTuningTab.test.tsx`

| Test | Description |
|------|-------------|
| `renders model selector pills` | 3 pills visible: LGBM, CatBoost, XGBoost |
| `switches model on pill click` | Clicking CatBoost reloads data with model=catboost |
| `renders lag filter bar` | 6 segments: All, Lag 0-4 |
| `filters by lag on click` | Clicking "Lag 2" passes exec_lag=2 to API |
| `renders KPI summary cards` | Best Accuracy, Production Accuracy, Total Runs, Active Runs |
| `renders run history table` | Columns: #, Label, Status, Accuracy, WAPE, Bias, Duration, Started |
| `sorts by accuracy descending` | Default sort order correct |
| `selects baseline on first row click` | Blue highlight, baseline_id set |
| `selects candidate on second row click` | Green highlight, candidate_id set |
| `deselects on re-click` | Highlight removed, selection cleared |
| `opens experiment builder on button click` | Modal visible with form fields |
| `renders status filter dropdown` | All, Running, Completed, Failed options |
| `shows promoted crown icon` | Promoted run has crown badge |
| `shows running pulse badge` | Running runs have animated blue dot |
| `paginates with 25/50/100 options` | Pagination controls work |

### 13.5 Frontend Tests — Experiment Builder Modal

**File:** `frontend/src/tabs/__tests__/ExperimentBuilder.test.tsx`

| Test | Description |
|------|-------------|
| `renders template radio buttons` | 6 options visible for LGBM |
| `pre-fills params on template select` | Selecting "Expert: Aggressive Depth" fills max_depth=10, num_leaves=63 |
| `shows delta column values` | Delta shows "+250%" for reg_lambda when changed from 1.0 to 3.5 |
| `validates learning_rate range` | Entering 0.0001 shows "min 0.001" error |
| `validates n_estimators integer` | Entering 1500.5 shows "must be integer" error |
| `disables launch button on validation error` | Button grayed out with invalid params |
| `submits experiment on launch click` | POST called with correct payload |
| `shows error banner on API failure` | Network error displays red alert, modal stays open |
| `closes modal on successful submit` | Modal hidden after 201 response |
| `shows toast notification on success` | Toast with experiment name appears |
| `adapts form to CatBoost params` | Shows iterations, depth, l2_leaf_reg (not n_estimators, max_depth) |
| `adapts form to XGBoost params` | Shows booster, rate_drop, skip_drop fields for DART |
| `renders training config section` | Cluster strategy, recursive, SHAP toggles visible |

### 13.6 Frontend Tests — Comparison Panel

**File:** `frontend/src/tabs/__tests__/ComparisonPanel.test.tsx`

| Test | Description |
|------|-------------|
| `renders verdict badge` | IMPROVED badge for positive delta |
| `renders per-lag accuracy table` | 5 rows with lag 0-4 |
| `renders lag accuracy curve chart` | Line chart with 2 lines |
| `applies lag filter to comparison` | Selecting lag 1 updates all metric cards |
| `renders parameter diff table` | Changed params highlighted |
| `renders promote buttons` | "Promote Candidate" and "Promote Baseline" visible |
| `shows loading state while fetching` | Skeleton placeholders during fetch |
| `handles missing lag data gracefully` | Shows "No lag data" message if legacy run |

### 13.7 Frontend Tests — Promote Modal

**File:** `frontend/src/tabs/__tests__/PromoteModal.test.tsx`

| Test | Description |
|------|-------------|
| `renders run metrics` | Accuracy, WAPE, Bias displayed |
| `renders per-lag summary` | Lag 0-4 accuracy shown |
| `renders parameter diff table` | Current vs New columns |
| `renders promotion checklist` | 5 steps listed |
| `calls promote API on confirm` | POST /promote called |
| `shows spinner during promotion` | Loading state visible |
| `closes modal on success` | Modal dismissed after 200 |
| `shows error on failure` | Red alert with error message |

### 13.8 Frontend Tests — Log Viewer

**File:** `frontend/src/tabs/__tests__/LogViewer.test.tsx`

| Test | Description |
|------|-------------|
| `renders log lines` | Log text displayed in monospace |
| `auto-scrolls to bottom` | Scroll position at bottom |
| `polls for new logs` | Interval timer active every 2s |
| `stops polling when completed` | Timer cleared on status=completed |
| `shows duration counter` | Real-time elapsed time while running |
| `copy button copies to clipboard` | navigator.clipboard called |
| `scroll lock pauses auto-scroll` | Toggling lock prevents scroll |

### 13.9 Frontend Tests — Jobs Tab Integration

**File:** `frontend/src/tabs/__tests__/JobsTab.tuning.test.tsx`

| Test | Description |
|------|-------------|
| `shows tuning jobs with model label` | "LightGBM Tuning — Aggressive Depth" visible |
| `shows beaker icon for tuning jobs` | Flask icon rendered (not gear) |
| `shows model type badge` | Blue "LGBM" pill on job card |
| `shows timeframe progress` | "Timeframe D/J — 40%" visible |
| `open in tuning tab link works` | Click navigates to Model Tuning tab |
| `can filter by model_tuning_run type` | Filter dropdown includes model_tuning_run |
| `distinguishes tuning from other jobs` | Tuning and backtest jobs visually distinct |

### 13.10 E2E Tests

**File:** `frontend/e2e/tests/model-tuning.spec.ts`

| Test | Description |
|------|-------------|
| `navigate to Model Tuning tab` | Sidebar click opens tuning tab |
| `switch between model pills` | LGBM → CatBoost → XGBoost switches content |
| `open experiment builder` | "New Experiment" button opens modal |
| `select template and launch` | Full flow: template → launch → toast |
| `view running experiment in Jobs` | Navigate to Jobs tab, see tuning job |
| `compare two completed runs` | Select 2 rows, comparison panel appears |
| `filter by execution lag` | Click Lag 0, KPIs update |
| `promote a run` | Click Promote, confirm, crown appears |
| `view experiment logs` | Click View Logs, log viewer opens |

### 13.11 Test Mock Patterns

**Backend (pytest):**
```python
# Standard pattern for unified router tests
@pytest.mark.asyncio
async def test_list_experiments_lgbm():
    pool = _make_pool()
    cursor = pool.getconn().cursor().__enter__()
    cursor.fetchall.return_value = [
        (1, "baseline", "lgbm_cluster", "completed", 72.5, 27.5, 0.3,
         100000, 5000, "{}", "[]", None, "2026-03-24T10:00:00Z",
         "2026-03-24T11:00:00Z", False, None, None, None)
    ]
    cursor.description = [
        ("run_id",), ("run_label",), ("model_id",), ("status",),
        ("accuracy_pct",), ("wape",), ("bias",), ("n_predictions",),
        ("n_dfus",), ("params",), ("features",), ("notes",),
        ("started_at",), ("completed_at",), ("is_promoted",),
        ("promoted_at",), ("job_id",), ("template_id",)
    ]
    with patch("api.core._get_pool", return_value=pool):
        from api.main import app
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/model-tuning/lgbm/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 1
    assert data["runs"][0]["model_id"] == "lgbm_cluster"
```

**Frontend (vitest):**
```typescript
// Mock the unified query module
vi.mock("@/api/queries", () => ({
  ...existingMocks,
  modelTuningKeys: {
    experiments: (model: string) => ["model-tuning", model, "experiments"],
    experiment: (model: string, id: number) => ["model-tuning", model, "experiment", id],
    compare: (model: string, b: number, c: number) => ["model-tuning", model, "compare", b, c],
    lags: (model: string, id: number) => ["model-tuning", model, "lags", id],
    templates: (model: string) => ["model-tuning", model, "templates"],
    promoted: (model: string) => ["model-tuning", model, "promoted"],
  },
  fetchModelExperiments: vi.fn().mockResolvedValue({ runs: mockRuns }),
  fetchModelExperimentLags: vi.fn().mockResolvedValue({ lags: mockLags }),
  fetchModelComparison: vi.fn().mockResolvedValue(mockComparison),
  fetchModelTemplates: vi.fn().mockResolvedValue({ templates: mockTemplates }),
  submitModelExperiment: vi.fn().mockResolvedValue({ run_id: 1, job_id: "abc" }),
  promoteModelRun: vi.fn().mockResolvedValue({ success: true }),
}));
```

---

## Migration Plan

### Phase 1: Schema + API (Backend)
1. Create SQL migration `sql/099_unified_model_tuning.sql` with new tables
2. Create unified router package `api/routers/forecasting/tuning/` (15 modules; see [Router Layout](#router-layout))
3. Register new job type `model_tuning_run` in `job_registry.py`
4. Implement job callable `_run_model_tuning_experiment` in `job_state.py`
5. Create `config/forecasting/tuning_templates.yaml`
6. Create `config/model_tuning_config.yaml`
7. Add backward-compatible aliases in existing routers
8. Write all backend tests
9. Add `/model-tuning` prefix to `vite.config.ts` proxy

### Phase 2: Frontend
1. Create new query module `frontend/src/api/queries/unified-model-tuning.ts`
2. Build `ExperimentBuilderModal` component
3. Build `LagFilterBar` component
4. Build enhanced `ComparisonPanel` with lag sections
5. Build `LogViewer` slide-over component
6. Build enhanced `PromoteModal` with lag summary
7. Refactor `LgbmTuningTab.tsx` → `ModelTuningTab.tsx` with unified state
8. Update `AppSidebar.tsx` to point to new tab
9. Update Jobs tab to recognize `model_tuning_run` type
10. Write all frontend tests
11. Write E2E tests

### Phase 3: Cleanup
1. Deprecate `lgbm_tuning.py` and `model_tuning.py` (keep as aliases)
2. Update `CLAUDE.md`, `ARCHITECTURE.md`
3. Update this spec with implementation results

---

## Acceptance Criteria

### Must Have (P0)
- [ ] User can configure hyperparameters in the UI and launch an experiment for any of the 3 models
- [ ] Experiment runs as a resilient subprocess that survives API restarts
- [ ] Real-time log streaming viewable in both Model Tuning tab and Jobs tab
- [ ] All metrics (accuracy, WAPE, bias) filterable by execution lag (0-4)
- [ ] Side-by-side comparison with per-lag, per-cluster, per-month, and per-timeframe breakdowns
- [ ] Promote winning run to production (writes to forecast_pipeline_config.yaml + DB audit log)
- [ ] Jobs tab clearly distinguishes tuning experiments from other job types
- [ ] Expert-recommended templates pre-loaded for all 3 models (5 per model)
- [ ] Parameter validation with inline error messages
- [ ] All existing tests continue to pass
- [ ] 100+ new backend tests, 50+ new frontend tests

### Should Have (P1)
- [ ] "Open in Jobs" / "Open in Tuning" cross-navigation links
- [ ] Experiment builder shows delta% from production baseline
- [ ] Promotion audit trail (tuning_promotion_log) queryable via API
- [ ] Lag accuracy curve chart in comparison panel
- [ ] Auto-generated insight text for lag analysis

### Nice to Have (P2)
- [ ] Bayesian optimization suggestions based on run history
- [ ] Batch experiment launch (run all 5 templates sequentially)
- [ ] Export comparison report as PDF/HTML
- [ ] Notification (toast/email) when long-running experiment completes
- [ ] Dark mode support for log viewer

---

## File Placement

| New File | Location | Purpose |
|----------|----------|---------|
| `sql/099_unified_model_tuning.sql` | `sql/` | New tables: lgbm_tuning_lag, tuning_promotion_log, alter lgbm_tuning_run |
| `api/routers/forecasting/tuning/` | `api/routers/forecasting/` | Unified /model-tuning/{model}/* router package (15 modules: list, detail, create, compare, cluster, lag, logs, month, promote, promote_results, cancel_delete, templates, promotions, _helpers, __init__) |
| `config/forecasting/tuning_templates.yaml` | `config/` | Expert-recommended experiment templates |
| `config/model_tuning_config.yaml` | `config/` | Tuning system configuration |
| `frontend/src/tabs/ModelTuningTab.tsx` | `frontend/src/tabs/` | Main tab component (replaces LgbmTuningTab.tsx) |
| `frontend/src/tabs/model-tuning/ExperimentBuilder.tsx` | `frontend/src/tabs/model-tuning/` | Experiment builder modal |
| `frontend/src/tabs/model-tuning/LagFilterBar.tsx` | `frontend/src/tabs/model-tuning/` | Execution lag filter component |
| `frontend/src/tabs/model-tuning/EnhancedComparisonPanel.tsx` | `frontend/src/tabs/model-tuning/` | Comparison with lag analysis |
| `frontend/src/tabs/model-tuning/LogViewer.tsx` | `frontend/src/tabs/model-tuning/` | Slide-over log viewer |
| `frontend/src/tabs/model-tuning/EnhancedPromoteModal.tsx` | `frontend/src/tabs/model-tuning/` | Promotion with lag summary |
| `frontend/src/api/queries/unified-model-tuning.ts` | `frontend/src/api/queries/` | Unified query module |
| `tests/api/test_unified_model_tuning.py` | `tests/api/` | Backend API tests |
| `tests/unit/test_model_tuning_job.py` | `tests/unit/` | Job callable tests |
| `tests/unit/test_tuning_schema.py` | `tests/unit/` | Schema constraint tests |
| `frontend/src/tabs/__tests__/ModelTuningTab.test.tsx` | `frontend/src/tabs/__tests__/` | Tab component tests |
| `frontend/src/tabs/__tests__/ExperimentBuilder.test.tsx` | `frontend/src/tabs/__tests__/` | Builder modal tests |
| `frontend/src/tabs/__tests__/ComparisonPanel.test.tsx` | `frontend/src/tabs/__tests__/` | Comparison tests |
| `frontend/src/tabs/__tests__/PromoteModal.test.tsx` | `frontend/src/tabs/__tests__/` | Promote modal tests |
| `frontend/src/tabs/__tests__/LogViewer.test.tsx` | `frontend/src/tabs/__tests__/` | Log viewer tests |
| `frontend/src/tabs/__tests__/JobsTab.tuning.test.tsx` | `frontend/src/tabs/__tests__/` | Jobs tab tuning integration |
| `frontend/e2e/tests/model-tuning.spec.ts` | `frontend/e2e/tests/` | E2E tests |

---

## Pre-Implementation Blockers

These must be resolved **before** any spec implementation begins.

> **Status update:** Blockers 1 & 2 are resolved by the tuning fit-path refactor.
> `common/ml/tuning.py` now constructs estimators via `model_registry.build_tree_model(algorithm_id, params)` and trains via `model_registry.fit_model(...)`. `to_native_params()` translates the canonical YAML keys to the appropriate native constructor arguments for `LGBMRegressor`/`CatBoostRegressor`/`XGBRegressor`, so all keys in `forecast_pipeline_config.yaml` `algorithms.<model_id>.params` reach the model. The legacy 6-parameter `default_params` lambda in `scripts/run_backtest.py`'s MODEL_REGISTRY no longer applies -- backtest, tuning, and production all share the same registry-driven fit path.

### Blocker 1: CatBoost `default_params` Drops Most Parameters (CRITICAL)

**File:** `scripts/run_backtest.py` (CatBoost model registry entry)

The CatBoost `default_params` lambda in the MODEL_REGISTRY only extracts 6 parameters (`iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `border_count`, `max_ctr_complexity`). All other CatBoost parameters in `forecast_pipeline_config.yaml` are **silently discarded** before reaching `CatBoostRegressor`. This means:

- `grow_policy`, `max_leaves`, `subsample`, `reg_lambda`, `random_strength`, `min_data_in_leaf`, `colsample_bylevel`, `bagging_temperature`, `bootstrap_type`, `model_size_reg`, `score_function`, `boost_from_average`, `leaf_estimation_method`, `leaf_estimation_iterations` are all **ignored**
- All prior CatBoost tuning experiments may have tested parameters that never reached the model
- All 5 proposed CatBoost experiments would produce identical results to the baseline

**Fix:** Extend `default_params` to pass through all keys from the algo config section that are valid `CatBoostRegressor` constructor arguments.

### Blocker 2: XGBoost `default_params` Drops Most Parameters (CRITICAL)

**File:** `scripts/run_backtest.py` (XGBoost model registry entry)

Same issue. XGBoost's `default_params` only extracts 6 parameters (`n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`). All proposed tuning parameters are dropped:

- `grow_policy`, `max_leaves`, `max_bin`, `reg_lambda`, `reg_alpha`, `gamma`, `colsample_bylevel`, `booster`, `rate_drop`, `skip_drop` are all **ignored**
- XGBoost Runs 2-5 (lossguide, DART, high-tree-count, champion blend) would silently degrade to the 6-parameter baseline

**Fix:** Extend `default_params` to include all valid `XGBRegressor` constructor arguments.

### Blocker 3: Verify Prior CatBoost/XGBoost Results

Given Blockers 1 and 2, the production accuracy numbers for CatBoost (72.15%) and XGBoost (71.23%) may have been achieved with only 6 parameters despite the config showing 15+ parameters. Before building the tuning UI, verify whether the existing champion results actually used the full parameter set or the truncated one.

---

## Gap Report & Fixes (Expert Review)

This spec was reviewed by 4 AI expert agents. Below is the consolidated gap report with fixes applied.

### Review Panel

| Agent | Focus Area | Gaps Found |
|-------|-----------|------------|
| **AI Product Manager** | User journeys, edge cases, feature completeness | 17 gaps (2 Critical, 6 Major, 9 Minor) |
| **ML Engineer** | Hyperparameter validity, experiment design, lag computation | 12 gaps (2 Critical, 6 Major, 4 Minor) |
| **Frontend UX Engineer** | Information architecture, accessibility, empty states | 20 gaps (4 Critical, 8 Major, 8 Minor) |
| **QA Architect** | Test coverage, integration tests, security | 20 gaps (5 Critical, 8 Major, 7 Minor) |

### Critical Gaps — Fixed in Spec

| # | Gap | Reviewer | Fix Applied |
|---|-----|----------|-------------|
| 1 | CatBoost/XGBoost `default_params` silently drop most parameters | ML Engineer | Added [Pre-Implementation Blockers](#pre-implementation-blockers) section |
| 2 | `lgbm_tuning_run.status` CHECK constraint missing 'queued' | Product Manager | Fixed in Section 6.3 — ALTER constraint to include 'queued' and 'cancelled' |
| 3 | Execution lag pseudocode references non-existent `abs_error` column | ML Engineer | Fixed in Section 9.2 — compute abs_error inline from basefcst_pref - tothist_dmd |
| 4 | Comparison sub-tab not explicitly specified vs inline panel confusion | UX Engineer | Clarified: Comparison is inline side-panel (not a separate sub-tab), sub-tabs are Experiments/Cluster EDA/Feature Lab/Accuracy Budget |
| 5 | No cancel/delete endpoints in API | Product Manager | Added POST .../cancel, DELETE .../experiments/{id}, POST .../rollback in Section 5.1 |

### Major Gaps — Fixed in Spec

| # | Gap | Reviewer | Fix Applied |
|---|-----|----------|-------------|
| 6 | No parameter tooltips/help text | UX Engineer | Added info_text specification, collapsible sections in Section 4.4.3 |
| 7 | Execution lag filter has no contextual explanation | UX Engineer | Added help tooltip, first-visit callout, horizon labels "(1mo)" in Section 4.3 |
| 8 | Empty states not specified | Product/UX | Added empty state specifications for zero runs, no champion, legacy runs in Section 4.4.2 |
| 9 | Custom template starts blank | Product/UX | Changed to "Custom (from Production)" with pre-filled baseline values |
| 10 | Cross-parameter validation missing | ML Engineer | Added CatBoost Langevin, bootstrap_type, SymmetricTree rules; XGBoost DART conditional rendering |
| 11 | Per-lag comparison response missing WAPE/bias | Product Manager | Extended per_lag array to include baseline/candidate WAPE and bias with deltas |
| 12 | AI Tuning Advisor integration unaddressed | UX Engineer | Added AI Advisor section with "Use in Experiment Builder" button |
| 13 | Promotion rollback not specified | Product/UX | Added POST .../promotions/rollback endpoint |

### Major Gaps — Noted for Implementation (Not Changed in Spec Text)

| # | Gap | Reviewer | Action Required |
|---|-----|----------|-----------------|
| 14 | Feature set confounding (SHAP varies between runs) | ML Engineer | Add "Lock feature set to baseline" checkbox in Experiment Builder during implementation |
| 15 | Concurrent promotion race condition — need advisory lock | Product Manager | Use `pg_advisory_xact_lock` on model-specific key during promotion transaction |
| 16 | Log viewer needs size limit and virtual scrolling | Product/UX | Add `max_log_size_bytes: 5MB` config, use `@tanstack/react-virtual` for log lines |
| 17 | Cluster EDA and Feature Lab sub-tabs not fully specified | Product Manager | These are carried forward from Feature 45 — defer to existing implementation |
| 18 | Keyboard accessibility not specified | UX Engineer | Add ARIA roles during implementation: tablist for model selector, radiogroup for lag filter |
| 19 | No responsive/mobile layout | UX Engineer | Experiment Builder becomes full-screen sheet on mobile; comparison stacks vertically on <1280px |
| 20 | Pagination API contract underspecified | Product Manager | Use `page`/`page_size` params, response: `{runs: [...], total_count: N, page: N, page_size: N}` |

### Minor Gaps — Noted for Implementation

| # | Gap | Reviewer | Note |
|---|-----|----------|------|
| 21 | "..." row menu interactions not fully specified (keyboard, delete confirmation) | UX Engineer | Delete requires confirmation dialog, disabled for running/promoted runs |
| 22 | Model_id mapping (lgbm → lgbm_cluster) not documented as constant | Product Manager | Add `MODEL_ID_MAP` constant in router code |
| 23 | No rate limiting on experiment submission | Product Manager | Disable Launch button immediately on click (already added to spec) |
| 24 | Verdict inconsistency (mixed vs neutral) | Product Manager | Standardize on "neutral" across all tables and endpoints |
| 25 | DART prediction-time cost not mentioned | ML Engineer | Add note to XGBoost Run 3: "DART is slower at inference due to tree renormalization" |
| 26 | No run tagging system (W&B pattern) | UX Engineer | P2 feature — add optional `tags: string[]` to run schema |
| 27 | No parallel coordinates or hyperparameter importance charts | UX Engineer | P2 feature — after 10+ runs accumulated |
| 28 | No artifact tracking (model files, SHAP plots) | UX Engineer | P2 feature — add artifact links to run detail view |
| 29 | Toast notifications underspecified | UX Engineer | Duration=5s, auto-dismiss, action button "View in Jobs" |
| 30 | No log search/filter | UX Engineer | Add Ctrl+F search and log-level filter chips |

### Testing Gaps — Additional Tests Required

The following test cases were identified as missing from Section 13 and must be added during implementation:

**Backend (12 additional tests):**
- `test_cancel_running_experiment` — Status transitions to 'cancelled', subprocess killed
- `test_cancel_queued_experiment` — Removed from queue, status='cancelled'
- `test_delete_completed_experiment` — Returns 204, cascade deletes lag/cluster/month rows
- `test_delete_promoted_experiment_blocked` — Returns 400 "Demote first"
- `test_delete_running_experiment_blocked` — Returns 400 "Cancel first"
- `test_promote_idempotent` — Promoting already-promoted run is no-op
- `test_compare_same_run_id_rejected` — Returns 400
- `test_promotion_rollback` — Restores previous champion params to forecast_pipeline_config.yaml
- `test_promotions_audit_trail` — GET /promotions returns ordered list
- `test_legacy_redirect_catboost` — /catboost-tuning/runs returns same as /model-tuning/catboost/experiments
- `test_legacy_redirect_xgboost` — Same for XGBoost
- `test_sql_injection_in_run_label` — Malicious label stored as literal text

**Frontend (8 additional tests):**
- `test_empty_state_zero_runs` — Shows empty illustration + CTA
- `test_empty_state_no_champion` — KPI shows "Not yet promoted"
- `test_lag_filter_legacy_run_dash` — Shows "--" for runs without lag data
- `test_conditional_dart_params` — rate_drop/skip_drop hidden until booster=dart
- `test_conditional_langevin_params` — diffusion_temperature hidden until langevin=true
- `test_ai_advisor_use_in_builder` — "Use in Experiment Builder" opens pre-filled modal
- `test_cancel_button_on_running_row` — Shows cancel confirmation
- `test_log_viewer_queued_state` — Shows "Logs will appear when execution starts"

### ML Experiment Design Feedback

**LGBM gap:** No run explores `bagging_freq` changes or disabling SHAP selection to test full feature set. Consider adding a P1 Run 6 that sets `shap_select: false`.

**CatBoost gap:** No run explores `score_function=Cosine` or `leaf_estimation_method=Gradient`. These could be future experiment candidates.

**XGBoost gap:** The production baseline is severely under-parameterized (500 trees, no regularization). Consider adding a "minimal upgrade" run that only increases trees to 2000 + basic reg_lambda=1.0 to establish how much improvement comes from "more trees" alone vs architectural changes.

### Strengths Acknowledged by All Reviewers

1. **Resilience matrix** (Section 10.1) — 6 failure scenarios with specific recovery behaviors
2. **Expert-recommended experiment plans** — 15 runs with testable hypotheses and rationale
3. **Complete test specification** — 100+ test cases with mock patterns
4. **Backward compatibility strategy** — Legacy endpoints preserved as aliases
5. **Execution-lag as first-class dimension** — Global filter affecting all views
6. **Promotion audit trail** — Full lineage with previous_run_id tracking

---

## Carried Forward from Feature 45 (10b-lgbm-tuning)

This spec (Feature 11) replaced the pre-rewrite tuning spec (`10b-lgbm-tuning.md`, Feature 45),
which has since been deleted. The sections below fold forward the parts of that spec that are
still accurate and not otherwise covered above - most directly, Gap #17 above notes that the
Cluster EDA and Feature Lab sub-tabs are "carried forward from Feature 45 - defer to existing
implementation"; this section **is** that existing-implementation reference.

### Running Multiple Tuning Experiments

There are four ways to run tuning experiments, from simplest to most sophisticated:

1. **Auto-Tune Campaign** (CLI, batch) - runs predefined strategies from
   `config/forecasting/tune_strategies.yaml` (organized by model key: `lgbm`, `catboost`, `xgboost`).
   Each strategy overrides specific params, runs a full backtest, registers the result, and prints a
   leaderboard. `uv run python scripts/ml/auto_tune.py --model <lgbm|catboost|xgboost> --runs N`, or
   `make lgbm-auto-tune RUNS=N` / `make lgbm-auto-tune-promote RUNS=N` for LGBM. Also seedable via
   `uv run python scripts/ml/seed_model_tuning.py`.
2. **Manual Single Run** (CLI) - edit `algorithms.<model_id>.params` in
   `forecast_pipeline_config.yaml`, run the model's backtest, then register and compare via
   `uv run python scripts/ml/compare_backtest_runs.py --register-latest --label "..."` and
   `--compare --baseline <id> --candidate <id>`.
3. **AI Tuning Advisor** (interactive, UI) - see below.
4. **Sampled Fast Iteration** (API) - stratified DFU sampling runs a backtest in ~3 min instead of
   ~25 min (±1-2pp deviation from a full backtest). `POST /lgbm-tuning/sampled/preview` previews the
   sample allocation; `POST /lgbm-tuning/sampled/run` triggers a sampled run with optional
   `param_overrides`; `GET /lgbm-tuning/sampled/strata` returns cluster-level DFU counts and demand
   stats. Implemented in `api/routers/forecasting/sampled_backtest.py` (backed by
   `common/ml/backtest_sampler.py`). Sampling methods: `proportional` (by cluster size), `equal`
   (same per cluster), `sqrt` (square-root allocation).

### AI Tuning Chat

An interactive AI-powered chat panel embedded in the Model Tuning tab (the "AI Advisor Integration"
referenced in Section 4.4.2). The advisor reviews previous runs, identifies patterns across clusters
and timeframes, recommends parameter changes, and - with user confirmation - kicks off new backtest
runs, all within a conversational interface.

- **Agent:** `common/ai/tuning_advisor.py`, provider-agnostic (OpenAI/Anthropic via config), agentic
  tool-use loop with `MAX_TURNS=20`, `TOKEN_BUDGET=50,000`, a 40-message sliding context window (first
  3 + last 37), and per-turn logging to `ai_call_log`.
- **Tools (7):** `list_tuning_runs`, `get_run_detail`, `compare_runs`, `analyze_cluster_patterns`,
  `get_current_config`, `recommend_params`, `check_run_status`.
- **Database:** `sql/096_create_tuning_chat.sql` - `tuning_chat_session` (session metadata, cached
  run-summary context) and `tuning_chat_message` (role: user/assistant/system; message_type:
  text/recommendation/run_started/run_completed/run_failed/analysis/error).
- **API** (`api/routers/forecasting/tuning_chat.py`, still on the legacy `/lgbm-tuning/chat/*` prefix
  - not migrated to `/model-tuning/{model}/*`):

  | Method | Endpoint | Purpose |
  |---|---|---|
  | POST | `/lgbm-tuning/chat/sessions` | Create a new session, seeded with run-summary context |
  | GET | `/lgbm-tuning/chat/sessions` | List sessions (active/archived) with message count |
  | GET | `/lgbm-tuning/chat/sessions/{id}` | Get session + full message history |
  | POST | `/lgbm-tuning/chat/sessions/{id}/messages` | Send a user message → synchronous AI response |
  | POST | `/lgbm-tuning/chat/sessions/{id}/confirm-run` | Confirm a recommendation → trigger async backtest |
  | GET | `/lgbm-tuning/chat/sessions/{id}/run-status/{run_id}` | Poll run completion status |

  Safety guards: `max_concurrent_runs=1` (409 on conflict), `min_seconds_between_runs=300`,
  `require_confirmation=true`.
- **Frontend:** `TuningChatPanel` (session management, message list, input area), `RecommendationCard`
  (parameter overrides + expected impact + risk, with Confirm & Run / Reject), `RunStatusCard` (polls
  run status), `SessionList`. Integrated as a collapsible section in `LgbmTuningTab.tsx`, toggled by an
  "AI Tuning Advisor" button.

### Analysis Sub-Tabs (Cluster EDA, Feature Lab, Accuracy Budget, Sampled Backtest)

Beyond the Experiments and Comparison sub-tabs specified above, the tab includes:

- **Cluster EDA:** cluster profiles (mean demand, CV, zero %, seasonal amplitude, accuracy), error
  concentration by cluster/month, demand-value histograms per cluster, a month-by-cluster accuracy
  seasonality heatmap.
- **Feature Lab:** SHAP-based feature-importance ranking (top 30), cross-fold feature-rank stability
  (stable/moderate/unstable), a feature-correlation matrix (flags `|r| > 0.9`), per-cluster feature
  importance.
- **Accuracy Budget:** accuracy waterfall (naive baseline → ML model → oracle ceiling), gap
  decomposition (intermittent demand, seasonality, new products), ABC-class accuracy vs. targets,
  monthly accuracy trend, side-by-side model comparison.
- **Sampled Backtest:** strata preview (cluster-level DFU counts/stats), sample-allocation preview,
  and the "Quick Runs" trigger for the ~3-minute sampled backtests described above.

### Verdict Logic

The `improved` / `degraded` / `mixed` verdict returned by the compare endpoints:

```
IF delta_accuracy > 0.0 AND delta_wape < 0.0:
    verdict = "improved"
ELIF delta_accuracy < 0.0 AND delta_wape > 0.0:
    verdict = "degraded"
ELSE:
    verdict = "mixed"
```

A "mixed" verdict occurs when, for example, accuracy improves but bias worsens significantly -
review the per-timeframe/per-lag breakdown to make a judgment call.
