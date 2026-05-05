# Champion Model Selection

> Automatically picks the best-performing model for each product each month, creating a composite "champion" forecast that outperforms any single algorithm.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Accuracy |
| **Key Files** | `scripts/run_champion_selection.py`, `common/ml/champion_strategies.py`, `config/forecasting/forecast_pipeline_config.yaml` (champion section), `api/routers/forecasting/competition.py`, `frontend/src/tabs/AccuracyTab.tsx` |

---

## Problem

With many competing models (tree models, foundation models, statistical baselines), no single algorithm wins for every product every month. Seasonal items might favor CatBoost in peak months and LightGBM in troughs, while intermittent items may be best served by Chronos 2 or seasonal naive. Without automated selection, planners must either pick one model for everything (suboptimal) or manually choose per item (impractical at scale with thousands of DFUs).

## Solution

Champion selection evaluates prior model performance for each DFU (Demand Forecast Unit) each month and picks the winner. It simulates what a planner would do: look at recent accuracy, pick the best model, and use its forecast going forward. The selection is causally correct -- it only uses information that was available at the time the forecast was issued. A ceiling (oracle) model provides a theoretical upper bound using after-the-fact perfect knowledge, quantifying the gap between the rolling selection and what's theoretically possible.

## How It Works

1. Load per-model, per-DFU, per-month errors from the backtest archive
2. For each DFU-month, compute prior WAPE (Weighted Absolute Percentage Error) per model using only causally available data
3. Pick the model with the lowest prior WAPE -- this becomes the champion for that month
4. Fill warm-up months (where there is insufficient history) with a fallback model
5. Store champion predictions as `model_id = 'champion'` in the forecast table
6. Separately compute the ceiling: pick the best model per DFU-month using that month's actual error (after-the-fact oracle)
7. Store ceiling as `model_id = 'ceiling'` -- the gap between champion and ceiling measures improvement opportunity

### Execution-Lag Causality (Critical)

Each DFU has an `execution_lag` -- how many months in advance its forecast is issued. For a DFU with `execution_lag = 1`, the April forecast is issued in March. At issuance time, March actuals are NOT available yet.

**The fix:** For each DFU-model group, apply `shift(execution_lag + 1)` before any cumulative or rolling computation. This ensures month T's champion uses only months where `startdate < fcstdate` (= T minus execution_lag).

**Example: Picking April 2025's champion (execution_lag = 1)**

The April forecast is issued in March 2025. Available actuals: January and February only (March is not available yet).

| Month | Model A Error | Model B Error |
|-------|-------------|-------------|
| Jan 2025 | 10 | 5 |
| Feb 2025 | 15 | 10 |
| Mar 2025 | 5 | 15 |
| **Apr 2025** | 8 | 2 |

Prior WAPE for April (using only Jan + Feb):
- Model A: (10+15) / 200 = 12.5%
- Model B: (5+10) / 200 = 7.5%

Winner: Model B. Its April forecast (98 units) becomes the champion row.

With `execution_lag = 0`, the formula degrades to `shift(1)` -- fully backward compatible.

### Warm-Up Period

With `execution_lag = 1` and `min_dfu_rows = 3`, the first qualifying month requires 3 non-null prior observations after shifting. Earlier months get the fallback model (default: `seasonal_naive`), ensuring every DFU-month has a champion row.

## 31 Selection Strategies

### Tier 1: Core Strategies (5)

| Strategy | Key Idea | Best For |
|----------|----------|----------|
| `expanding` | Cumulative WAPE from all prior months (equal weight) | Stable demand, long history |
| `rolling` | Last N months only (default N=6) | Volatile demand, regime changes |
| `decay` | Exponential weighting (recent months count more, factor=0.9) | Rewarding recent improvement |
| `ensemble` | Blend top-K models weighted by inverse WAPE | No single model dominates |
| `meta_learner` | ML classifier predicts best model from DFU features + performance stats | Rich feature data available |

### Tier 2: Coverage & Adaptiveness (4)

| Strategy | Key Idea | Best For |
|----------|----------|----------|
| `hybrid_warmup` | Fast-adapting strategy (rolling) for warm-up months, then switches to a stronger primary strategy (any registered strategy) once enough history accumulates | Maximizing DFU-month coverage; addresses the coverage gap where expanding/ensemble discard early months |
| `adaptive_ensemble` | Varies top-K per DFU-month based on model WAPE spread; low spread uses fewer models (min_k), high spread uses more (max_k) | Mixed portfolios where some DFUs have a clear winner and others don't |
| `ensemble_rolling` | Blend top-K models using rolling-window WAPE instead of expanding | Combines rolling's adaptiveness to regime changes with ensemble's hedging against wrong picks |
| `optimized_decay` | Walk-forward validation to auto-select the best decay factor from candidates [0.75, 0.80, 0.85, 0.90, 0.95]; splits timeline into train/validation, picks factor with lowest validation WAPE | When the optimal decay rate is unknown; eliminates manual tuning |

### Tier 3: ML-Driven Selection (1)

| Strategy | Key Idea | Best For |
|----------|----------|----------|
| `hybrid_meta_router` | Uses trained ML classifier to predict best model per DFU with confidence score. High confidence (>=60%) trusts the single predicted model; low confidence hedges by blending top-K via inverse-WAPE weights | Large DFU populations with diverse demand patterns; ported from algorithm_testing hybrid ensemble |

### Tier 4: Demand-Aware Routing (3)

| Strategy | Key Idea | Best For |
|----------|----------|----------|
| `seasonal` | Evaluates model performance using only same-calendar-quarter prior months; falls back to expanding when insufficient seasonal history | Demand with strong seasonal patterns where Q4 performance matters more than Q2 for predicting next Q4 |
| `per_segment` | Classifies each DFU into Syntetos-Boylan demand archetypes (smooth, erratic, intermittent, lumpy) based on ADI and CV² metrics, then routes each segment to a different sub-strategy | Heterogeneous portfolios mixing steady sellers with intermittent/lumpy SKUs |
| `per_cluster` | Groups DFUs by ML cluster label from dim_sku, computes best model per cluster via aggregate expanding WAPE, assigns cluster champion to all DFUs in that cluster | When ML clustering captures demand structure that aligns with model strengths |

### Tier 5: Advanced Blending & Risk (4)

| Strategy | Key Idea | Best For |
|----------|----------|----------|
| `learned_blend` | Fits Ridge regression per DFU on causal prior data to learn optimal model blending weights (actual ≈ w1×model1 + w2×model2 + ...); clips negative weights to 0 and normalizes | When models capture complementary demand signals; learns non-uniform weights |
| `ridge_blend` | Similar to learned_blend with additional safeguards: drops constant columns, requires min 2 non-constant models, and uses a separate min_train_months threshold | Production-grade Ridge blending with better numerical stability |
| `uncertainty_aware` | Adds a penalty for models with volatile errors (high std-dev of absolute errors). Score = prior_wape + weight × normalized_std_err. Picks the model with lowest risk-adjusted score | When model consistency matters as much as average accuracy; penalizes unreliable models |
| `diverse_ensemble` | Blends top-K models with a diversity penalty: models from the same family (e.g., chronos2 + chronos2_enriched) get penalized during selection. Blending weights use unpenalized inverse-WAPE | Portfolios where structurally similar models would otherwise dominate the ensemble |

### Tier 6: Intelligent Ensembles (10)

| Strategy | Key Idea | Best For |
|----------|----------|----------|
| `cascade_ensemble` | Adapts ensemble breadth to model confidence: WAPE < 10% → trust best model solo; 10-25% → blend top-2; >25% → blend top-5 | Avoiding dilution when one model clearly dominates |
| `adversarial_filter` | Detects outlier models (z-score > threshold) and excludes them before ensembling; removes the model most likely to blow up the forecast | Portfolios with occasional model failures |
| `dynamic_window` | Cross-validates lookback window [2,3,4,6,9,12] per DFU; picks the window with lowest recent CV error | When optimal memory length varies across DFUs |
| `regime_adaptive` | Detects demand regime changes via rolling variance ratio; uses expanding during stable periods, rolling during shifts | Demand with periodic structural breaks |
| `bayesian_model_avg` | Gaussian likelihood-based Bayesian updating of model probabilities; starts with uniform prior, updates with each causal observation | Principled probabilistic model selection |
| `error_correcting` | Picks best model by WAPE, then adds a correction for recent systematic bias: corrected = forecast - strength × recent_bias | When the best model has consistent over/under-forecasting |
| `shrinkage_blend` | Bates-Granger optimal forecast combination: weight = shrinkage × (1/N) + (1-shrinkage) × inverse_wape_weight | Theoretically optimal blending under parameter uncertainty |
| `dfu_strategy_router` | Per-DFU, evaluates multiple candidate strategies on recent months (walk-forward), routes each DFU to its best-performing strategy | When different DFUs respond to different selection approaches |
| `stacked_strategies` | Runs multiple base strategies, evaluates each on recent months, blends their outputs weighted by walk-forward accuracy | Meta-ensemble of strategies rather than models |
| `cluster_regime_hybrid` | Per-cluster strategy selection combined with regime-aware switching within each cluster | Combines structural demand grouping with temporal adaptation |

### Tier 7: Reinforcement Learning (4)

These strategies treat model selection as an online learning problem with exploration/exploitation tradeoffs. Unlike greedy strategies (Tiers 1-6) that always pick what *was* best, RL strategies occasionally explore models that *might* have become better.

| Strategy | Algorithm | Key Mechanism | Best For |
|----------|-----------|---------------|----------|
| `thompson_sampling` | Thompson Sampling | Maintains Beta(α,β) posterior per (DFU, model). Samples from posteriors to decide — naturally balances explore/exploit. Discounted updates forget old observations. | General purpose; adapts to regime changes via discount factor |
| `thompson_ensemble` | Thompson + Ensemble | Same posteriors but blends top-K models by sampled probability instead of picking one. Combines exploration with hedging. | When single-model picks are too risky |
| `linucb` | LinUCB Contextual Bandit | Builds context vector from demand features (mean, CV, trend, zeros, seasonality). Learns *which features predict which model wins*. UCB exploration bonus for undersampled contexts. | When demand context drives model selection (not just past WAPE) |
| `exp3` | EXP3 Adversarial | Makes ZERO assumptions about reward distributions. Multiplicative weight updates with importance-weighted rewards. Provably optimal against worst-case. | Non-stationary demand; adversarial environments; when other strategies overfit |

#### Why RL Strategies May Close the Gap

Current strategies (expanding, ensemble, etc.) are **purely greedy** — they pick the model that performed best historically. This fails when:

1. **Model performance shifts** — CatBoost was best for 6 months but Chronos improved after a retraining. Greedy strategies stick with CatBoost; Thompson Sampling would explore Chronos.
2. **Context matters** — A model might win during trend-up periods but lose during trend-down. LinUCB learns this mapping; expanding WAPE averages it away.
3. **Adversarial dynamics** — Demand might shift in ways that systematically fool historical-WAPE strategies. EXP3 is robust to this by design.

#### RL Strategy Parameters

| Parameter | Strategy | Default | Description |
|-----------|----------|---------|-------------|
| `discount` | thompson_sampling, thompson_ensemble | 0.95 | Forgetting factor for posterior updates. Lower = faster adaptation. |
| `n_samples` | thompson_sampling, thompson_ensemble | 100 | Thompson samples averaged for stability. |
| `top_k` | thompson_ensemble | 3 | Models to blend in ensemble variant. |
| `alpha_ucb` | linucb | 1.0 | Exploration strength. Higher = more exploration. |
| `gamma` | exp3 | 0.1 | Exploration rate. 0 = pure exploit, 1 = uniform random. |

### Strategy Design Principles

All 31 strategies share these properties:

- **Strictly causal**: Selection for month T uses only data from months with `startdate < T - execution_lag`
- **Exec-lag-aware**: Uses `shift(execution_lag + 1)` to exclude months whose actuals weren't available at forecast issuance time
- **Standard output schema**: All return `[item_id, customer_group, loc, startdate, model_id, prior_wape, basefcst_pref, tothist_dmd]`
- **Empty-safe**: All return an empty DataFrame with correct columns when given empty input
- **Registry-based**: All registered via `@register_strategy("name")` in `STRATEGY_REGISTRY`

### Segment Routing (per_segment)

The `per_segment` strategy uses Syntetos-Boylan demand classification with these default sub-strategy assignments:

| Segment | ADI | CV² | Sub-Strategy | Rationale |
|---------|-----|-----|-------------|-----------|
| Smooth | < 1.32 | < 0.49 | `expanding` | Stable demand benefits from long history |
| Erratic | < 1.32 | ≥ 0.49 | `ensemble` (top_k=5) | High variance → hedge with more models |
| Intermittent | ≥ 1.32 | < 0.49 | `rolling` (window=6) | Sporadic demand → recent patterns matter |
| Lumpy | ≥ 1.32 | ≥ 0.49 | `rolling` (window=3, min_prior=2) | Most volatile → very short memory |

### Model Family Groups (diverse_ensemble)

| Family | Models |
|--------|--------|
| chronos | chronos2, chronos2_enriched, chronos_bolt |
| tree | catboost_cluster, xgboost_cluster, lgbm_cluster |
| baseline | seasonal_naive, rolling_mean |
| statistical | mstl |
| dl | nhits, nbeats |

The meta-learner uses ceiling (oracle) labels as ground truth with a strict temporal train/test split. Trained via `make champion-train-meta`, saved to `data/champion/meta_learner.joblib`.

## Data Model

No new tables. Champion and ceiling predictions are stored in the existing `fact_external_forecast_monthly` with `model_id = 'champion'` and `model_id = 'ceiling'`. All accuracy views automatically include them.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/competition/config` | Current config + available model_ids from DB |
| PUT | `/competition/config` | Update config (writes YAML to disk) |
| POST | `/competition/run` | Execute champion selection, return summary |
| GET | `/competition/summary` | Last run summary |

## Pipeline

| Target | Description |
|--------|-------------|
| `make champion-select` | Run champion selection using configured strategy |
| `make champion-simulate` | Simulate all 31 strategies, compare accuracy vs. ceiling (supports `--parallel N`) |
| `make champion-train-meta` | Train meta-learner classifier |
| `make champion-all` | train-meta + simulate + select (full pipeline) |

## Configuration

### `config/forecasting/forecast_pipeline_config.yaml` (champion section)

> The legacy `config/model_competition.yaml` has been deleted. All settings now live in the master config.

```yaml
champion:
  strategy: hybrid_warmup            # Any registered strategy name (17 available)
  strategy_params:
    min_prior_months: 3              # For all strategies
    warmup_strategy: rolling         # Phase 1: fast-adapting for new DFUs
    warmup_window: 2                 # Rolling window for warm-up phase
    warmup_min_prior: 1              # Minimum history for warm-up selection
    primary_strategy: adaptive_ensemble  # Phase 2: once enough history
    primary_top_k: 3                 # Top-K models for ensemble blending
    primary_weight_method: inverse_wape  # Weighting: inverse_wape | equal
  fallback_model_id: seasonal_naive  # Used for warm-up gaps
  metric: accuracy_pct               # wape (lowest wins) or accuracy_pct
  lag: execution                     # "execution" (per-DFU) or fixed 0-4
  min_sku_rows: 3
  min_dfu_rows: 3                    # Minimum prior months before selection
  champion_model_id: champion        # model_id for stored champion rows
```

The competing models list is derived from `algorithms[*].compete == true` in the master config rather than an explicit list.

### Strategy-Specific Parameters

| Parameter | Strategies | Default | Description |
|-----------|-----------|---------|-------------|
| `min_prior_months` | All | 3 | Minimum causal prior months to qualify |
| `window_months` | rolling, ensemble_rolling | 6 | Rolling window size |
| `decay_factor` | decay | 0.90 | Exponential decay weight per month |
| `top_k` | ensemble, ensemble_rolling, adaptive_ensemble, diverse_ensemble | 3 | Number of models to blend |
| `weight_method` | ensemble, ensemble_rolling, adaptive_ensemble | inverse_wape | Blending weight method |
| `warmup_strategy` | hybrid_warmup | rolling | Phase 1 strategy name |
| `warmup_window` | hybrid_warmup | 2 | Rolling window for warm-up |
| `warmup_min_prior` | hybrid_warmup | 1 | Min months for warm-up phase |
| `primary_strategy` | hybrid_warmup | adaptive_ensemble | Phase 2 strategy name (any registered) |
| `decay_candidates` | optimized_decay | [0.75..0.95] | Candidate decay factors to evaluate |
| `validation_months` | optimized_decay | 3 | Walk-forward validation window |
| `confidence_threshold` | hybrid_meta_router | 0.6 | Min classifier confidence for single-model pick |
| `blend_top_k` | hybrid_meta_router | 3 | Models to blend when confidence is low |
| `adi_threshold` | per_segment | 1.32 | ADI boundary for demand classification |
| `cv2_threshold` | per_segment | 0.49 | CV² boundary for demand classification |
| `uncertainty_weight` | uncertainty_aware | 0.3 | Penalty weight for error volatility |
| `correlation_penalty` | diverse_ensemble | 0.5 | Penalty for same-family model pairs |
| `alpha` | learned_blend | 100.0 | Ridge regularization strength |
| `ridge_alpha` | ridge_blend | 100.0 | Ridge regularization strength |
| `cluster_col` | per_cluster | ml_cluster | Column name for cluster labels |
| `solo_threshold` | cascade_ensemble | 0.10 | WAPE below which single best model is trusted |
| `mid_threshold` | cascade_ensemble | 0.25 | WAPE above which wide blend (top-5) is used |
| `outlier_z_threshold` | adversarial_filter | 1.5 | Z-score threshold for excluding outlier models |
| `correction_strength` | error_correcting | 0.5 | How much to correct for recent bias (0-1) |
| `shrinkage_intensity` | shrinkage_blend | 0.5 | Blend of equal weights vs inverse-WAPE (0-1) |
| `variance_threshold` | regime_adaptive | 2.0 | Variance ratio threshold for regime detection |
| `discount` | thompson_sampling, thompson_ensemble | 0.95 | Posterior forgetting factor |
| `alpha_ucb` | linucb | 1.0 | Exploration strength for UCB bonus |
| `gamma` | exp3 | 0.1 | Exploration rate for EXP3 |

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- provides backtest predictions to compare
- [Multi-Model Support](./02-multi-model.md) -- `model_id` column stores champion/ceiling rows
- [Accuracy KPIs](./01-accuracy-kpis.md) -- WAPE formula used for selection
- Python packages: `scikit-learn>=1.3` (meta-learner), `joblib>=1.3`

## See Also

- [Production Forecast](./08-production-forecast.md) -- uses champion assignments to route inference
- [Algorithm Config](./06-algorithm-config.md) -- controls which models compete
- [Chronos Foundation Models](./18-chronos-foundation-models.md) -- Chronos 2, Chronos Bolt, and other foundation model variants that participate in champion selection
- [Expert Panel](./15-expert-panel-algorithm-selection.md) -- 31-expert panel that routes DFUs to best-fit algorithms, complementary to champion selection
