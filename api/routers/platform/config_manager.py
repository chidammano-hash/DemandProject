"""Configuration management API — list, read, and update YAML config files.

Provides a unified interface for the Settings UI to view and modify all
system configuration with rich metadata (descriptions, types, constraints)
for every parameter.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_api_key
from common.core.utils import load_config, reset_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/config", tags=["config"])

from common.core.paths import CONFIG_DIR as _CONFIG_DIR  # noqa: E402


def _resolve_yaml_path(yaml_name: str) -> Path:
    """Locate a config yaml at flat root or any domain subdir.

    Mirrors the lookup logic in :func:`common.core.utils.load_config`.
    Returns the flat-root path when the file does not exist anywhere
    (so callers writing a new file get a predictable location).
    """
    flat = _CONFIG_DIR / yaml_name
    if flat.exists():
        return flat
    matches = list(_CONFIG_DIR.rglob(yaml_name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HTTPException(
            status_code=500,
            detail=f"ambiguous config name {yaml_name!r}: {[str(m) for m in matches]}",
        )
    return flat

# ---------------------------------------------------------------------------
# Field metadata types
# ---------------------------------------------------------------------------
# Each field entry: { label, description, type, unit?, min?, max?, options?, step? }
# Types: "number", "integer", "text", "boolean", "select", "array", "object"

FieldMeta = dict[str, Any]

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------
CATEGORIES = [
    {"key": "forecasting", "label": "Forecasting & Models", "description": "Model training, backtesting, hyperparameter tuning, and forecast generation settings."},
    {"key": "inventory", "label": "Inventory Planning", "description": "Safety stock, EOQ, replenishment policies, service levels, and variability analysis."},
    {"key": "operations", "label": "Supply Chain Operations", "description": "Multi-echelon planning, rebalancing, projections, procurement, and financial planning."},
    {"key": "pipeline", "label": "Data Pipeline", "description": "Medallion ETL, data quality checks, clustering, exception detection, and caching."},
    {"key": "planning", "label": "Planning & Collaboration", "description": "Planning date, S&OP cycles, consensus planning, and event calendar."},
    {"key": "system", "label": "System & Integration", "description": "Authentication, API governance, notifications, reporting, and AI planner."},
]

# ---------------------------------------------------------------------------
# Comprehensive config registry with field-level metadata
# ---------------------------------------------------------------------------
CONFIG_REGISTRY: dict[str, dict[str, Any]] = {
    # ═══════════════════════════════════════════════════════════════════════
    # FORECASTING & MODELS
    # ═══════════════════════════════════════════════════════════════════════
    "hyperparameter_tuning": {
        "label": "Hyperparameter Tuning",
        "category": "forecasting",
        "description": "Optuna-based Bayesian hyperparameter optimization settings. Controls search space boundaries, trial counts, cross-validation strategy, and early stopping.",
        "fields": {
            "tuning.n_trials": {"label": "Number of Trials", "description": "How many hyperparameter combinations Optuna will evaluate. More trials find better parameters but take longer.", "type": "integer", "min": 10, "max": 500, "step": 10},
            "tuning.n_splits": {"label": "CV Folds", "description": "Number of time-series cross-validation folds.", "type": "integer", "min": 2, "max": 10, "step": 1},
            "tuning.gap_months": {"label": "CV Gap Months", "description": "Gap between training and validation sets in each CV fold to prevent data leakage.", "type": "integer", "min": 0, "max": 6, "step": 1},
            "tuning.val_months_per_fold": {"label": "Validation Months per Fold", "description": "Number of months in each validation window.", "type": "integer", "min": 1, "max": 12},
            "tuning.min_train_months": {"label": "Min Training Months", "description": "Minimum months of data required in the training set for the first fold.", "type": "integer", "min": 6, "max": 36},
            "tuning.early_stopping_rounds": {"label": "Early Stopping Rounds", "description": "Stop training if validation metric hasn't improved for this many rounds.", "type": "integer", "min": 10, "max": 200, "step": 10},
            "tuning.n_estimators_max": {"label": "Max Trees Cap", "description": "Hard upper limit on tree count during tuning.", "type": "integer", "min": 500, "max": 10000, "step": 100},
            "tuning.random_seed": {"label": "Random Seed", "description": "Fixed seed for reproducible results across tuning runs.", "type": "integer", "min": 0, "max": 99999},
            "lgbm.search_space.learning_rate.low": {"label": "LGBM LR Lower Bound", "description": "Minimum learning rate Optuna will try for LGBM.", "type": "number", "min": 0.001, "max": 0.1, "step": 0.001},
            "lgbm.search_space.learning_rate.high": {"label": "LGBM LR Upper Bound", "description": "Maximum learning rate for LGBM tuning.", "type": "number", "min": 0.01, "max": 1.0, "step": 0.01},
            "lgbm.search_space.num_leaves.low": {"label": "LGBM Leaves Lower", "description": "Minimum num_leaves in LGBM search space.", "type": "integer", "min": 4, "max": 64},
            "lgbm.search_space.num_leaves.high": {"label": "LGBM Leaves Upper", "description": "Maximum num_leaves in LGBM search space.", "type": "integer", "min": 32, "max": 512},
            "catboost.search_space.depth.low": {"label": "CatBoost Depth Lower", "description": "Minimum tree depth in CatBoost search space.", "type": "integer", "min": 1, "max": 8},
            "catboost.search_space.depth.high": {"label": "CatBoost Depth Upper", "description": "Maximum tree depth in CatBoost search space.", "type": "integer", "min": 4, "max": 16},
            "xgboost.search_space.max_depth.low": {"label": "XGBoost Depth Lower", "description": "Minimum tree depth in XGBoost search space.", "type": "integer", "min": 1, "max": 8},
            "xgboost.search_space.max_depth.high": {"label": "XGBoost Depth Upper", "description": "Maximum tree depth in XGBoost search space.", "type": "integer", "min": 4, "max": 20},
            "lgbm.fixed_params.random_state": {"label": "LGBM Random Seed", "description": "Fixed seed for LGBM reproducibility.", "type": "integer", "min": 0, "max": 99999},
        },
    },
    "forecast_pipeline_config": {
        "label": "Forecast Pipeline (Master)",
        "category": "forecasting",
        "description": "Master pipeline configuration controlling the full ML forecast lifecycle: algorithm roster with lifecycle flags, clustering, backtest, tuning, champion selection, production forecast, and run tracking.",
        "fields": {
            # ── Forecast Settings (always visible) ───────────────────────────
            "production_forecast.horizon_months": {"label": "Forecast Horizon", "description": "How many months ahead to generate forecasts.", "type": "integer", "min": 6, "max": 36, "unit": "months", "group": "Forecast Settings"},
            "production_forecast.min_history_months": {"label": "Min History Required", "description": "Items with fewer months of sales get routed to the cold-start model.", "type": "integer", "min": 3, "max": 24, "unit": "months", "group": "Forecast Settings"},
            "production_forecast.cold_start_model_id": {"label": "New Product Model", "description": "Model used for items with limited sales history.", "type": "text", "group": "Forecast Settings"},
            "production_forecast.cold_start_min_months": {"label": "New Product Min Data", "description": "Items with fewer months than this get no forecast at all.", "type": "integer", "min": 1, "max": 12, "unit": "months", "group": "Forecast Settings"},
            # ── Champion Selection ────────────────────────────────────────────
            "champion.strategy": {"label": "Model Selection Strategy", "description": "How the best model is chosen per item.", "type": "select", "options": ["expanding", "rolling", "adaptive_ensemble", "hybrid_warmup", "per_cluster", "decay", "ensemble", "meta_learner"], "group": "Champion Selection"},
            "champion.metric": {"label": "Evaluation Metric", "description": "Metric used to compare model accuracy.", "type": "select", "options": ["accuracy_pct", "wape"], "group": "Champion Selection"},
            "champion.fallback_model_id": {"label": "Default Model", "description": "Model used when no champion can be determined.", "type": "text", "group": "Champion Selection"},
            # ── Active Models ────────────────────────────────────────────────
            # These use a special "model_toggle" type rendered as a compact table
            "algorithms.lgbm_cluster.enabled": {"label": "LightGBM", "description": "Tree-based gradient boosting", "type": "model_toggle", "group": "Active Models", "model_type": "tree"},
            "algorithms.chronos2_enriched.enabled": {"label": "Chronos 2E", "description": "Chronos 2 with covariates", "type": "model_toggle", "group": "Active Models", "model_type": "foundation"},
            "algorithms.mstl.enabled": {"label": "MSTL", "description": "Statistical decomposition", "type": "model_toggle", "group": "Active Models", "model_type": "statistical"},
            "algorithms.nbeats.enabled": {"label": "N-BEATS", "description": "Deep learning", "type": "model_toggle", "group": "Active Models", "model_type": "deep_learning"},
            "algorithms.nhits.enabled": {"label": "N-HiTS", "description": "Deep learning", "type": "model_toggle", "group": "Active Models", "model_type": "deep_learning"},
            # ── Clustering ────────────────────────────────────────────────────
            "clustering.enabled": {"label": "DFU Clustering", "description": "Segment items by demand pattern before training.", "type": "boolean", "group": "Clustering & Backtest"},
            "backtest.n_timeframes": {"label": "Backtest Windows", "description": "Number of historical evaluation windows.", "type": "integer", "min": 1, "max": 30, "group": "Clustering & Backtest"},
            "backtest.forecast_horizon": {"label": "Backtest Horizon", "description": "Months predicted per evaluation window.", "type": "integer", "min": 1, "max": 24, "unit": "months", "group": "Clustering & Backtest"},
            "backtest_sampling.enabled": {"label": "Fast Sampling", "description": "Sample a subset of items for quick iteration.", "type": "boolean", "group": "Clustering & Backtest"},
            "backtest_sampling.default_target_n": {"label": "Sample Size", "description": "Number of items to include when sampling.", "type": "integer", "min": 100, "max": 50000, "step": 100, "group": "Clustering & Backtest"},
        },
    },
    "forecast_domain_config": {
        "label": "Forecast Domain",
        "category": "forecasting",
        "description": "Consolidated forecast domain settings: seasonality detection, demand variability, quantile forecasts, and bias correction.",
        "fields": {
            # --- Seasonality ---
            "seasonality.min_months_history": {"label": "Seasonality Min History", "description": "Minimum months of sales history required before seasonality analysis is attempted.", "type": "integer", "min": 6, "max": 48, "unit": "months"},
            "seasonality.thresholds.low": {"label": "Low Seasonality Threshold", "description": "Seasonal strength below this value is classified as 'none'.", "type": "number", "min": 0.0, "max": 0.5, "step": 0.01},
            "seasonality.thresholds.medium": {"label": "Medium Seasonality Threshold", "description": "Seasonal strength between low and this value is 'moderate'. Above this is 'strong'.", "type": "number", "min": 0.1, "max": 0.8, "step": 0.01},
            "seasonality.thresholds.high": {"label": "High Seasonality Threshold", "description": "Seasonal strength above this value is classified as 'strong' or 'very strong'.", "type": "number", "min": 0.3, "max": 1.0, "step": 0.01},
            "seasonality.confirmation.yoy_correlation": {"label": "YoY Correlation Threshold", "description": "Minimum year-over-year correlation to confirm a seasonal pattern is real.", "type": "number", "min": 0.1, "max": 0.9, "step": 0.05},
            "seasonality.confirmation.acf_lag12": {"label": "ACF Lag-12 Threshold", "description": "Minimum autocorrelation at lag 12 to confirm annual seasonality.", "type": "number", "min": 0.1, "max": 0.9, "step": 0.05},
            "seasonality.peak_trough_min_ratio": {"label": "Peak-Trough Ratio", "description": "Minimum ratio of peak-month to trough-month demand to qualify as seasonal.", "type": "number", "min": 1.0, "max": 5.0, "step": 0.1},
            "seasonality.output_path": {"label": "Output File", "description": "Path for seasonality detection results CSV output.", "type": "text"},
            # --- Variability ---
            "variability.history.history_months": {"label": "Variability History Months", "description": "Number of months of sales history to analyze for variability.", "type": "integer", "min": 6, "max": 60, "unit": "months"},
            "variability.history.min_months_history": {"label": "Variability Min History Months", "description": "Minimum months required before computing variability.", "type": "integer", "min": 3, "max": 24, "unit": "months"},
            "variability.outlier.sigma_threshold": {"label": "Outlier Sigma", "description": "Winsorization threshold in standard deviations. Values beyond this are capped before CV computation.", "type": "number", "min": 1.0, "max": 5.0, "step": 0.5},
            "variability.cv_thresholds.low": {"label": "CV — Low Threshold", "description": "CV below this is classified as 'stable' (X-class).", "type": "number", "min": 0.05, "max": 0.5, "step": 0.05},
            "variability.cv_thresholds.medium": {"label": "CV — Medium Threshold", "description": "CV between low and this is 'moderate' (Y-class).", "type": "number", "min": 0.3, "max": 1.5, "step": 0.05},
            "variability.cv_thresholds.high": {"label": "CV — High Threshold", "description": "CV above this is 'volatile' (Z-class).", "type": "number", "min": 0.5, "max": 3.0, "step": 0.1},
            "variability.intermittency_threshold.ratio": {"label": "Intermittency Threshold", "description": "Fraction of months with zero demand above which the item is classified as intermittent.", "type": "number", "min": 0.1, "max": 0.8, "step": 0.05},
            # --- Quantile Forecast ---
            "quantile_forecast.quantiles": {"label": "Quantile Levels", "description": "List of quantile levels to forecast. [0.10, 0.50, 0.90] gives pessimistic, median, and optimistic scenarios.", "type": "array"},
            "quantile_forecast.default_horizon_months": {"label": "Default Forecast Horizon", "description": "Default number of months to forecast at each quantile level.", "type": "integer", "min": 1, "max": 36, "unit": "months"},
            "quantile_forecast.max_horizon_months": {"label": "Max Forecast Horizon", "description": "Maximum allowed forecast horizon.", "type": "integer", "min": 1, "max": 36, "unit": "months"},
            "quantile_forecast.weekly_disaggregation": {"label": "Weekly Disaggregation", "description": "Disaggregate monthly quantile forecasts into weekly buckets for operational planning.", "type": "boolean"},
            "quantile_forecast.model.n_estimators": {"label": "LGBM Trees", "description": "Number of boosting rounds for the quantile regression model.", "type": "integer", "min": 50, "max": 2000},
            "quantile_forecast.model.learning_rate": {"label": "LGBM Learning Rate", "description": "Learning rate for quantile regression training.", "type": "number", "min": 0.001, "max": 0.5, "step": 0.005},
            "quantile_forecast.sigma_guard_rails.max_sigma_multiplier": {"label": "Max Sigma Multiplier", "description": "Cap the P90-P10 spread at this multiple of the median forecast.", "type": "number", "min": 1.0, "max": 10.0},
            "quantile_forecast.plan_versioning.max_active_versions": {"label": "Max Active Versions", "description": "Number of quantile forecast versions to keep active.", "type": "integer", "min": 1, "max": 20},
            "quantile_forecast.plan_versioning.retention_days": {"label": "Retention Days", "description": "Days to retain archived quantile forecast data.", "type": "integer", "min": 30, "max": 730, "unit": "days"},
            # --- Bias Correction ---
            "bias_correction.rolling_weights": {"label": "Rolling Window Weights", "description": "Weights for recent months in bias calculation. [0.50, 0.30, 0.20] means most recent month gets 50%. Must sum to 1.0.", "type": "array"},
            "bias_correction.min_months_dfu": {"label": "Min Months (DFU)", "description": "Minimum months of history before applying DFU-level bias correction.", "type": "integer", "min": 1, "max": 12, "unit": "months"},
            "bias_correction.min_months_segment": {"label": "Min Months (Segment)", "description": "Minimum months for segment-level fallback when DFU history is insufficient.", "type": "integer", "min": 1, "max": 12, "unit": "months"},
            "bias_correction.min_bias_threshold": {"label": "Min Bias Threshold", "description": "Minimum absolute bias before correction is applied. Below this is treated as noise.", "type": "number", "min": 0, "max": 0.5, "step": 0.01},
            "bias_correction.correction_factor_min": {"label": "Correction Factor Min", "description": "Minimum correction multiplier. 0.70 means correction can reduce forecast by at most 30%.", "type": "number", "min": 0.1, "max": 1.0, "step": 0.05},
            "bias_correction.correction_factor_max": {"label": "Correction Factor Max", "description": "Maximum correction multiplier. 1.30 means correction can increase forecast by at most 30%.", "type": "number", "min": 1.0, "max": 3.0, "step": 0.05},
            "bias_correction.review_threshold": {"label": "Review Threshold", "description": "Bias above which a DFU is flagged for manual review rather than auto-corrected.", "type": "number", "min": 0.05, "max": 1.0, "step": 0.01},
            "bias_correction.segment_priority": {"label": "Segment Priority", "description": "Priority order for segment-level fallback when DFU has insufficient history.", "type": "array"},
            "bias_correction.lookback_months": {"label": "Lookback Months", "description": "Number of recent months to analyze for bias detection.", "type": "integer", "min": 1, "max": 24, "unit": "months"},
        },
    },

    "tune_strategies": {
        "label": "Tuning Strategies",
        "category": "forecasting",
        "description": "Per-model auto-tune strategy definitions for Bayesian hyperparameter optimization. Each model type has a list of named strategies with parameter overrides.",
        "fields": {},
    },

    # ═══════════════════════════════════════════════════════════════════════
    # INVENTORY PLANNING
    # ═══════════════════════════════════════════════════════════════════════
    "safety_stock_config": {
        "label": "Safety Stock",
        "category": "inventory",
        "description": "Safety stock calculation methodology. Controls service-level targets by ABC class, calculation method, and guard rails.",
        "fields": {
            "safety_stock.default_method": {"label": "Calculation Method", "description": "'combined' uses both demand and lead time variability (most accurate), 'demand_only' ignores LT variability, 'lt_only' ignores demand variability.", "type": "select", "options": ["combined", "demand_only", "lt_only"]},
            "safety_stock.policy_version": {"label": "Policy Version", "description": "Version label for the safety stock parameters. Increment when making significant changes.", "type": "text"},
            "safety_stock.service_levels.A": {"label": "Service Level — A Items", "description": "Target cycle service level for A-class (high-volume) items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "safety_stock.service_levels.B": {"label": "Service Level — B Items", "description": "Target service level for B-class (medium-volume) items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "safety_stock.service_levels.C": {"label": "Service Level — C Items", "description": "Target service level for C-class (low-volume) items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "safety_stock.service_levels.default": {"label": "Service Level — Default", "description": "Fallback service level for items without ABC classification.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "safety_stock.min_ss_days": {"label": "Min Safety Stock Days", "description": "Floor for safety stock in days of supply.", "type": "integer", "min": 0, "max": 30, "unit": "days"},
            "safety_stock.max_ss_days": {"label": "Max Safety Stock Days", "description": "Ceiling for safety stock in days of supply.", "type": "integer", "min": 30, "max": 365, "unit": "days"},
            "safety_stock.min_demand_months": {"label": "Min Demand Months", "description": "Minimum months of demand history required before computing safety stock.", "type": "integer", "min": 1, "max": 12, "unit": "months"},
            "safety_stock.lt_std_fallback_pct": {"label": "LT Std Fallback %", "description": "When lead time std dev is unknown, estimate as this percentage of mean lead time.", "type": "number", "min": 0.05, "max": 0.50, "step": 0.05, "unit": "%"},
            "safety_stock.use_demand_variability": {"label": "Use Demand Variability", "description": "Include demand standard deviation in safety stock calculation.", "type": "boolean"},
            "safety_stock.use_lt_variability": {"label": "Use LT Variability", "description": "Include lead time standard deviation in safety stock calculation.", "type": "boolean"},
            "safety_stock.batch_size": {"label": "Batch Size", "description": "Number of DFUs to process in each database batch.", "type": "integer", "min": 100, "max": 5000, "step": 100},
        },
    },
    "eoq_config": {
        "label": "Economic Order Quantity (EOQ)",
        "category": "inventory",
        "description": "EOQ calculation parameters — the classic trade-off between ordering costs and holding costs to find the optimal order quantity.",
        "fields": {
            "costs.default_ordering_cost": {"label": "Default Ordering Cost", "description": "Fixed cost per purchase order (admin, receiving, inspection).", "type": "number", "min": 0, "max": 1000, "unit": "$"},
            "costs.default_holding_cost_pct": {"label": "Holding Cost %", "description": "Annual holding cost as a percentage of unit cost. Typically 20–30%.", "type": "number", "min": 0.01, "max": 1.0, "step": 0.01, "unit": "%"},
            "costs.default_unit_cost": {"label": "Default Unit Cost", "description": "Fallback unit cost when item-specific cost data is unavailable.", "type": "number", "min": 0.01, "max": 10000, "unit": "$"},
            "costs.default_moq": {"label": "Default MOQ", "description": "Minimum order quantity when supplier MOQ data is not available.", "type": "integer", "min": 1, "max": 10000},
            "constraints.max_eoq_months_supply": {"label": "Max EOQ (Months Supply)", "description": "Cap EOQ at this many months of average demand.", "type": "integer", "min": 1, "max": 24, "unit": "months"},
            "constraints.min_annual_demand": {"label": "Min Annual Demand", "description": "Minimum annualized demand to compute EOQ.", "type": "number", "min": 0, "max": 100},
            "sensitivity.ordering_cost_steps": {"label": "Sensitivity Steps", "description": "Number of steps in EOQ sensitivity analysis.", "type": "integer", "min": 5, "max": 50},
            "batch.batch_size": {"label": "Batch Size", "description": "DFUs processed per database batch.", "type": "integer", "min": 100, "max": 5000},
        },
    },
    "replenishment_policy_config": {
        "label": "Replenishment Policies",
        "category": "inventory",
        "description": "Defines available replenishment policies and their auto-assignment rules.",
        "fields": {
            "auto_assign.enabled": {"label": "Auto-Assign Enabled", "description": "Automatically assign replenishment policies to new DFUs based on their ABC-XYZ classification.", "type": "boolean"},
            "auto_assign.variability_override.lumpy": {"label": "Lumpy Override Policy", "description": "Policy ID to assign to lumpy/intermittent demand items.", "type": "text"},
        },
    },
    "replenishment_plan_config": {
        "label": "Replenishment Plan",
        "category": "inventory",
        "description": "Detailed replenishment plan computation settings. Controls sigma estimation, confidence intervals, safety stock bounds, and order optimization.",
        "fields": {
            "replenishment_plan.sigma_method": {"label": "Sigma Method", "description": "'ci_spread' derives demand sigma from confidence interval spread (P90-P10). 'historical' uses raw demand std dev.", "type": "select", "options": ["ci_spread", "historical"]},
            "replenishment_plan.ci_confidence": {"label": "CI Confidence", "description": "Confidence level for the prediction interval used to derive sigma.", "type": "number", "min": 0.5, "max": 0.99, "step": 0.01},
            "replenishment_plan.ci_z_score": {"label": "CI Z-Score", "description": "Z-score corresponding to the CI confidence level.", "type": "number", "min": 0.5, "max": 3.0, "step": 0.001},
            "replenishment_plan.service_levels.A": {"label": "Service Level — A", "description": "Service level for A-class items in replenishment planning.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "replenishment_plan.service_levels.B": {"label": "Service Level — B", "description": "Service level for B-class items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "replenishment_plan.service_levels.C": {"label": "Service Level — C", "description": "Service level for C-class items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "replenishment_plan.service_levels.default": {"label": "Default Service Level", "description": "Fallback service level.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "replenishment_plan.min_ss_days": {"label": "SS Min Days", "description": "Minimum safety stock in days of supply.", "type": "integer", "min": 0, "max": 30, "unit": "days"},
            "replenishment_plan.max_ss_days": {"label": "SS Max Days", "description": "Maximum safety stock in days of supply.", "type": "integer", "min": 30, "max": 365, "unit": "days"},
            "replenishment_plan.lt_default_days": {"label": "Default Lead Time", "description": "Default lead time when supplier data is unavailable.", "type": "integer", "min": 1, "max": 180, "unit": "days"},
            "replenishment_plan.lt_std_fallback_pct": {"label": "LT Std Fallback %", "description": "Estimate LT std dev as this % of mean LT when unknown.", "type": "number", "min": 0.05, "max": 0.5, "step": 0.05},
            "replenishment_plan.eoq_annualization_months": {"label": "EOQ Annualization", "description": "Months of demand used to annualize for EOQ calculation.", "type": "integer", "min": 6, "max": 24, "unit": "months"},
            "replenishment_plan.costs.default_ordering_cost": {"label": "EOQ Ordering Cost", "description": "Cost per order for EOQ calculation.", "type": "number", "min": 0, "max": 1000, "unit": "$"},
            "replenishment_plan.costs.default_holding_cost_pct": {"label": "EOQ Holding Cost %", "description": "Annual holding cost percentage for EOQ.", "type": "number", "min": 0.01, "max": 1.0, "step": 0.01},
            "replenishment_plan.horizon_months": {"label": "Planning Horizon", "description": "Number of months to plan replenishment orders ahead.", "type": "integer", "min": 1, "max": 36, "unit": "months"},
            "replenishment_plan.default_policy_type": {"label": "Default Policy", "description": "Replenishment policy for items without an assigned policy.", "type": "select", "options": ["continuous_rop", "periodic_review", "min_max", "manual"]},
            "replenishment_plan.batch_size": {"label": "Batch Size", "description": "DFUs per database batch.", "type": "integer", "min": 100, "max": 5000},
        },
    },
    "service_level_config": {
        "label": "Service Level Monitoring",
        "category": "inventory",
        "description": "Service level target monitoring and chronic miss detection.",
        "fields": {
            "service_level.targets_by_abc.A": {"label": "Target — A Items", "description": "Service level target for A-class items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "service_level.targets_by_abc.B": {"label": "Target — B Items", "description": "Service level target for B-class items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "service_level.targets_by_abc.C": {"label": "Target — C Items", "description": "Service level target for C-class items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "service_level.targets_by_abc.X": {"label": "Target — X Items", "description": "Service level target for X-class (unclassified) items.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005},
            "service_level.min_months_history": {"label": "Min History", "description": "Minimum months of fill-rate history required before evaluating service levels.", "type": "integer", "min": 1, "max": 12, "unit": "months"},
            "service_level.chronic_miss_count": {"label": "Chronic Miss Count", "description": "Number of months within the window that must miss target to be flagged as chronic.", "type": "integer", "min": 1, "max": 12},
            "service_level.chronic_miss_window_months": {"label": "Chronic Miss Window", "description": "Rolling window (months) for evaluating chronic misses.", "type": "integer", "min": 2, "max": 24, "unit": "months"},
            "service_level.miss_reason_thresholds.stockout_days_for_stockout_miss": {"label": "Stockout Days Threshold", "description": "Days of zero on-hand inventory to classify a miss as stockout-driven.", "type": "integer", "min": 1, "max": 30, "unit": "days"},
            "service_level.miss_reason_thresholds.lt_variance_for_lt_miss": {"label": "LT Variance Days", "description": "Lead time deviation (days above planned) to classify as LT-driven miss.", "type": "integer", "min": 1, "max": 30, "unit": "days"},
            "service_level.miss_reason_thresholds.demand_spike_ratio_for_demand_miss": {"label": "Demand Surge Ratio", "description": "Actual/forecast ratio above which a miss is classified as demand-surge driven.", "type": "number", "min": 1.0, "max": 5.0, "step": 0.1},
            "service_level.target_dos": {"label": "Target Days of Supply", "description": "Target days-of-supply for inventory health scoring.", "type": "integer", "min": 7, "max": 180, "unit": "days"},
        },
    },
    # variability_config — merged into forecast_domain_config (variability section)
    "inventory_planning_config": {
        "label": "Inventory Planning",
        "category": "inventory",
        "description": "Unified inventory planning: lead time analysis, Monte Carlo simulation, and forward projection.",
        "fields": {
            "lead_time.history.history_months": {"label": "LT History Window", "description": "Months of receipt history to analyze for lead time patterns.", "type": "integer", "min": 3, "max": 36, "unit": "months"},
            "lead_time.change_point.min_observations": {"label": "LT Min Change Points", "description": "Minimum number of data points required for change-point detection algorithm.", "type": "integer", "min": 2, "max": 10},
            "lead_time.cv_thresholds.stable": {"label": "Stable LT Threshold", "description": "LT CV below this is 'stable'.", "type": "number", "min": 0.0, "max": 0.3, "step": 0.01},
            "lead_time.cv_thresholds.moderate": {"label": "Moderate LT Threshold", "description": "LT CV between stable and this is 'moderate'.", "type": "number", "min": 0.1, "max": 0.6, "step": 0.01},
            "lead_time.batch.batch_size": {"label": "LT Batch Size", "description": "Rows per database batch upsert.", "type": "integer", "min": 100, "max": 5000},
            "simulation.n_simulations": {"label": "Number of Simulations", "description": "Monte Carlo iterations per DFU. More iterations give more precise service level estimates.", "type": "integer", "min": 1000, "max": 100000, "step": 1000},
            "simulation.demand_distribution": {"label": "Demand Distribution", "description": "'empirical' samples from actual historical demand; 'normal' assumes normally distributed demand.", "type": "select", "options": ["empirical", "normal", "lognormal"]},
            "simulation.lt_distribution": {"label": "Lead Time Distribution", "description": "Distribution for lead time sampling.", "type": "select", "options": ["empirical", "normal", "lognormal"]},
            "simulation.ss_levels_to_test": {"label": "SS Levels Tested", "description": "Number of safety stock levels to evaluate between 0 and 2x analytical SS.", "type": "integer", "min": 5, "max": 50},
            "simulation.min_demand_days": {"label": "Min Demand Days", "description": "Minimum days of demand data required before running simulation.", "type": "integer", "min": 30, "max": 365, "unit": "days"},
            "simulation.max_skus_per_batch": {"label": "Max SKUs per Batch", "description": "SKUs processed in each batch to manage memory.", "type": "integer", "min": 50, "max": 2000},
            "projection.projection.horizon_days": {"label": "Projection Horizon", "description": "Days ahead to project inventory levels.", "type": "integer", "min": 14, "max": 365, "unit": "days"},
            "projection.projection.scenarios": {"label": "Projection Scenarios", "description": "List of scenarios to model: 'no_order', 'with_open_po', 'with_planned_orders'.", "type": "array"},
            "projection.thresholds.reorder_point_source": {"label": "Reorder Point Source", "description": "Source for reorder points: 'safety_stock' uses computed SS, 'manual' uses planner-set ROP.", "type": "select", "options": ["safety_stock", "manual"]},
            "projection.thresholds.excess_coverage_months": {"label": "Excess Coverage", "description": "Months of forward demand coverage above which inventory is excess.", "type": "integer", "min": 1, "max": 18, "unit": "months"},
            "demand_history.default_months": {"label": "Demand History Default Months", "description": "Default number of months of demand history to display.", "type": "integer", "min": 1, "max": 60, "unit": "months"},
            "demand_history.max_months": {"label": "Demand History Max Months", "description": "Maximum months of demand history a user can request.", "type": "integer", "min": 12, "max": 120, "unit": "months"},
            "demand_history.pareto_top_n": {"label": "Pareto Top N", "description": "Number of top customers to show in demand Pareto analysis.", "type": "integer", "min": 1, "max": 20},
            "demand_history.matrix_max_rows": {"label": "Matrix Max Rows", "description": "Maximum rows in the demand cross-reference matrix.", "type": "integer", "min": 10, "max": 500},
            "demand_history.matrix_max_cols": {"label": "Matrix Max Cols", "description": "Maximum columns in the demand cross-reference matrix.", "type": "integer", "min": 10, "max": 200},
            "demand_history.cache_ttl_seconds": {"label": "Cache TTL", "description": "Seconds to cache demand history responses.", "type": "integer", "min": 0, "max": 3600, "unit": "sec"},
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SUPPLY CHAIN OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════
    "echelon_config": {
        "label": "Multi-Echelon Planning",
        "category": "operations",
        "description": "Multi-echelon inventory optimization for distribution networks.",
        "fields": {
            "echelon.z_score_default": {"label": "Z-Score", "description": "Z-score for echelon safety stock calculation. 1.645 corresponds to 95% service level.", "type": "number", "min": 0.5, "max": 3.0, "step": 0.001},
            "echelon.min_downstream_nodes": {"label": "Min Downstream Nodes", "description": "Minimum downstream locations to qualify as a DC/hub in the echelon hierarchy.", "type": "integer", "min": 1, "max": 20},
            "echelon.cascade_risk_multiplier": {"label": "Cascade Risk Multiplier", "description": "Multiplier for downstream stockout risk propagation.", "type": "number", "min": 0.5, "max": 3.0, "step": 0.1},
            "echelon.coverage_alert_threshold_days": {"label": "DC Coverage Alert", "description": "Alert when DC days-of-coverage drops below this threshold.", "type": "integer", "min": 1, "max": 30, "unit": "days"},
            "echelon.default_lt_days": {"label": "Default Lead Time", "description": "Default inter-echelon lead time when actual data is unavailable.", "type": "integer", "min": 1, "max": 60, "unit": "days"},
            "echelon.default_lt_std_days": {"label": "Default LT Std Dev", "description": "Default lead time standard deviation for echelon calculations.", "type": "number", "min": 0, "max": 30, "unit": "days"},
        },
    },
    "rebalancing_config": {
        "label": "Inventory Rebalancing",
        "category": "operations",
        "description": "Network inventory rebalancing — identifies transfer opportunities to reduce total network imbalance.",
        "fields": {
            "network.default_transfer_lt_days": {"label": "Transfer Lead Time", "description": "Default inter-location transfer lead time.", "type": "integer", "min": 1, "max": 30, "unit": "days"},
            "network.default_cost_per_unit": {"label": "Transfer Cost/Unit", "description": "Cost per unit to transfer between locations.", "type": "number", "min": 0, "max": 100, "unit": "$", "step": 0.1},
            "network.default_min_transfer_qty": {"label": "Min Transfer Qty", "description": "Minimum units for a transfer to be worthwhile.", "type": "integer", "min": 1, "max": 1000},
            "optimization.solver": {"label": "Solver Algorithm", "description": "'greedy' uses a fast heuristic; 'milp' uses mixed-integer linear programming (slower, optimal).", "type": "select", "options": ["greedy", "milp"]},
            "optimization.horizon_weeks": {"label": "Planning Horizon", "description": "Weeks ahead for rebalancing evaluation.", "type": "integer", "min": 1, "max": 12, "unit": "weeks"},
            "triggers.excess_threshold_pct": {"label": "Excess Trigger", "description": "DOS above this percentage of target triggers a location as 'excess'.", "type": "number", "min": 1.0, "max": 5.0, "step": 0.1, "unit": "%"},
            "triggers.shortage_threshold_pct": {"label": "Shortage Trigger", "description": "DOS below this percentage of target flags as 'shortage'.", "type": "number", "min": 0.1, "max": 1.0, "step": 0.05, "unit": "%"},
            "triggers.min_benefit_per_transfer": {"label": "Min Benefit", "description": "Minimum dollar benefit for a rebalancing recommendation.", "type": "number", "min": 0, "max": 10000, "unit": "$"},
            "costs.default_handling_cost": {"label": "Handling Cost/Unit", "description": "Per-unit handling cost at origin.", "type": "number", "min": 0, "max": 50, "unit": "$", "step": 0.05},
            "costs.default_freight_cost": {"label": "Freight Cost/Unit", "description": "Per-unit freight cost.", "type": "number", "min": 0, "max": 50, "unit": "$", "step": 0.05},
            "costs.stockout_cost_multiplier": {"label": "Stockout Cost Multiplier", "description": "Stockout cost as a multiple of unit cost.", "type": "number", "min": 1.0, "max": 20.0, "step": 0.5},
            "constraints.frozen_period_days": {"label": "Frozen Period", "description": "Don't suggest transfers within this many days of a previous transfer.", "type": "integer", "min": 0, "max": 60, "unit": "days"},
            "constraints.max_source_drawdown_pct": {"label": "Max Drawdown %", "description": "Maximum percentage of excess stock that can be transferred from a donor.", "type": "number", "min": 0.05, "max": 1.0, "step": 0.05, "unit": "%"},
        },
    },
    # projection_config merged into inventory_planning_config (projection section)
    "order_recommendation_config": {
        "label": "Order Recommendations",
        "category": "operations",
        "description": "Automated purchase order recommendation engine.",
        "fields": {
            "recommendation.horizon_days": {"label": "Order Horizon", "description": "Days ahead to evaluate for potential order recommendations.", "type": "integer", "min": 14, "max": 365, "unit": "days"},
            "recommendation.max_orders_per_sku": {"label": "Max Orders per DFU", "description": "Maximum recommended orders per DFU in the planning horizon.", "type": "integer", "min": 1, "max": 10},
            "recommendation.confidence.high_threshold": {"label": "High Confidence Threshold", "description": "Minimum confidence score for auto-approval.", "type": "number", "min": 0.5, "max": 1.0, "step": 0.05},
            "recommendation.confidence.low_threshold": {"label": "Low Confidence Threshold", "description": "Below this confidence, recommendations require manual review.", "type": "number", "min": 0.1, "max": 0.9, "step": 0.05},
            "recommendation.confidence.penalty_no_open_po_data": {"label": "No PO Data Penalty", "description": "Confidence penalty when no historical PO data exists.", "type": "number", "min": 0, "max": 0.5, "step": 0.05},
            "recommendation.confidence.penalty_fallback_forecast": {"label": "Fallback Forecast Penalty", "description": "Confidence penalty when using segment-level forecast.", "type": "number", "min": 0, "max": 0.5, "step": 0.05},
            "recommendation.confidence.penalty_past_due_order": {"label": "Past-Due Penalty", "description": "Confidence penalty for items with past-due open POs.", "type": "number", "min": 0, "max": 0.5, "step": 0.05},
            "moq_handling.rounding_strategy": {"label": "MOQ Rounding", "description": "How to round order quantities to meet MOQ.", "type": "select", "options": ["ceil_to_moq", "nearest_moq", "none"]},
        },
    },
    "procurement_config": {
        "label": "Procurement",
        "category": "operations",
        "description": "Purchase order generation and ERP integration settings.",
        "fields": {
            "procurement.po_number_format": {"label": "PO Number Format", "description": "Template for generated PO numbers. {YEAR}, {MONTH:02d}, {SEQ:03d} are substituted.", "type": "text"},
            "procurement.default_currency": {"label": "Currency", "description": "Default currency code for purchase orders.", "type": "text"},
            "procurement.default_unit_of_measure": {"label": "Unit of Measure", "description": "Default UoM for order quantities.", "type": "text"},
            "procurement.approval_thresholds.requires_buyer_release_above_value": {"label": "Approval Threshold", "description": "Order value above which buyer approval is required.", "type": "number", "min": 0, "max": 100000, "unit": "$"},
            "procurement.erp_integration.max_retries": {"label": "ERP Max Retries", "description": "Retries for failed ERP API calls.", "type": "integer", "min": 0, "max": 10},
            "procurement.erp_integration.retry_delay_seconds": {"label": "ERP Retry Delay", "description": "Seconds between ERP retry attempts.", "type": "integer", "min": 1, "max": 120, "unit": "sec"},
            "procurement.erp_integration.timeout_seconds": {"label": "ERP Timeout", "description": "Timeout for individual ERP API calls.", "type": "integer", "min": 5, "max": 120, "unit": "sec"},
            "procurement.lead_time_fallback_days": {"label": "LT Fallback", "description": "Default lead time when supplier data is unavailable.", "type": "integer", "min": 1, "max": 90, "unit": "days"},
        },
    },
    "po_integration_config": {
        "label": "PO Data Integration",
        "category": "operations",
        "description": "Purchase order and receipt data ingestion from external systems.",
        "fields": {
            "ingest.strategy": {"label": "Ingestion Strategy", "description": "How PO data is loaded: 'csv_export' reads from flat files, 'api' pulls from ERP API.", "type": "select", "options": ["csv_export", "api"]},
            "validation.reject_pos_past_due_days": {"label": "Max Past Due Days", "description": "Reject POs this many days past their delivery due date.", "type": "integer", "min": 30, "max": 365, "unit": "days"},
            "data_quality.past_due_threshold_days": {"label": "Past Due Threshold", "description": "Days past due before a PO is flagged as problematic.", "type": "integer", "min": 1, "max": 30, "unit": "days"},
        },
    },
    "supply_scenario_config": {
        "label": "Supply Scenarios",
        "category": "operations",
        "description": "Supply chain disruption scenario modeling.",
        "fields": {
            "supply_scenario.simulation_horizon_weeks": {"label": "Simulation Horizon", "description": "Weeks to simulate forward for disruption impact.", "type": "integer", "min": 4, "max": 52, "unit": "weeks"},
            "supply_scenario.service_level_target": {"label": "Service Level Target", "description": "Target service level for disruption impact evaluation.", "type": "number", "min": 0.80, "max": 0.999, "step": 0.005, "unit": "%"},
            "supply_scenario.disruption_defaults.supplier_delay.typical_impact_pct": {"label": "Supplier Delay Impact %", "description": "Default impact percentage of a supplier delay.", "type": "integer", "min": 0, "max": 100},
            "supply_scenario.disruption_defaults.supplier_delay.typical_duration_weeks": {"label": "Supplier Delay Duration", "description": "Default duration of a supplier delay.", "type": "integer", "min": 1, "max": 26, "unit": "weeks"},
            "supply_scenario.disruption_defaults.capacity_constraint.typical_impact_pct": {"label": "Capacity Constraint Impact %", "description": "Default impact percentage of a capacity constraint.", "type": "integer", "min": 0, "max": 100},
            "supply_scenario.disruption_defaults.capacity_constraint.typical_duration_weeks": {"label": "Capacity Duration", "description": "Default duration of a capacity constraint.", "type": "integer", "min": 1, "max": 26, "unit": "weeks"},
            "supply_scenario.disruption_defaults.demand_shock.typical_impact_pct": {"label": "Demand Shock Impact %", "description": "Default impact percentage of a demand shock.", "type": "integer", "min": 0, "max": 100},
            "supply_scenario.disruption_defaults.demand_shock.typical_duration_weeks": {"label": "Demand Shock Duration", "description": "Default duration of a demand shock.", "type": "integer", "min": 1, "max": 12, "unit": "weeks"},
            "supply_scenario.alert_threshold_usd": {"label": "Alert Threshold", "description": "Financial impact above which a scenario triggers an alert.", "type": "number", "min": 0, "max": 10000000, "unit": "$"},
        },
    },
    "financial_plan_config": {
        "label": "Financial Planning",
        "category": "operations",
        "description": "Inventory financial planning and working capital management.",
        "fields": {
            "financial_plan.carrying_cost_pct": {"label": "Carrying Cost %", "description": "Annual carrying cost as a percentage of inventory value.", "type": "number", "min": 0.05, "max": 0.60, "step": 0.01, "unit": "%"},
            "financial_plan.months_ahead": {"label": "Projection Horizon", "description": "Months ahead for financial projections.", "type": "integer", "min": 1, "max": 24, "unit": "months"},
            "financial_plan.excess_dos_threshold": {"label": "Excess DOS Threshold", "description": "Days of supply above which inventory is classified as excess.", "type": "integer", "min": 30, "max": 365, "unit": "days"},
            "financial_plan.budget_breach_alert_pct": {"label": "Budget Breach Alert %", "description": "Alert when inventory investment reaches this percentage of budget.", "type": "number", "min": 0.5, "max": 1.0, "step": 0.05, "unit": "%"},
            "financial_plan.target_inventory_turns": {"label": "Inventory Turns Target", "description": "Target annual inventory turns (COGS / Average Inventory).", "type": "number", "min": 1.0, "max": 52.0, "step": 0.5, "unit": "turns"},
            "financial_plan.target_dos": {"label": "Target Days of Supply", "description": "Target portfolio-level days of supply for budgeting.", "type": "integer", "min": 7, "max": 180, "unit": "days"},
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # DATA PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    "data_quality_config": {
        "label": "Data Quality",
        "category": "pipeline",
        "description": "Comprehensive data quality monitoring rules.",
        "fields": {
            "schedule.cron": {"label": "Check Schedule", "description": "Cron expression for automated DQ checks.", "type": "text"},
            "schedule.on_load": {"label": "Check on Load", "description": "Run DQ checks automatically after each data load.", "type": "boolean"},
            "global_defaults.severity": {"label": "Default Severity", "description": "Default severity level for unspecified checks.", "type": "select", "options": ["info", "warning", "critical"]},
            "global_defaults.enabled": {"label": "DQ Checks Enabled", "description": "Master switch to enable/disable all data quality checks.", "type": "boolean"},
        },
    },
    # clustering_config removed — params now managed via Cluster Experimentation Studio UI
    "exception_config": {
        "label": "Exception Detection",
        "category": "pipeline",
        "description": "Storyboard exception engine configuration.",
        "fields": {
            "exception_engine.max_exceptions_per_run": {"label": "Max Exceptions", "description": "Maximum number of exceptions generated per engine run.", "type": "integer", "min": 50, "max": 5000},
            "exception_engine.exception_ttl_days": {"label": "Exception TTL", "description": "Days before unresolved exceptions automatically expire.", "type": "integer", "min": 7, "max": 90, "unit": "days"},
            "exception_engine.dedupe_window_days": {"label": "Dedupe Window", "description": "Days within which duplicate exceptions are suppressed.", "type": "integer", "min": 1, "max": 30, "unit": "days"},
            "exception_engine.severity_weights.financial_impact": {"label": "Severity: Financial Weight", "description": "Weight of financial impact in severity scoring (0–1).", "type": "number", "min": 0, "max": 1.0, "step": 0.05},
            "exception_engine.severity_weights.rule_score": {"label": "Severity: Rule Weight", "description": "Weight of rule-based score.", "type": "number", "min": 0, "max": 1.0, "step": 0.05},
            "exception_engine.severity_weights.urgency": {"label": "Severity: Urgency Weight", "description": "Weight of time urgency.", "type": "number", "min": 0, "max": 1.0, "step": 0.05},
        },
    },
    "cache_config": {
        "label": "API Cache",
        "category": "pipeline",
        "description": "In-memory API response caching configuration.",
        "fields": {
            "backend": {"label": "Cache Backend", "description": "Cache storage backend: 'memory' for in-process, 'redis' for distributed.", "type": "select", "options": ["memory", "redis"]},
            "max_memory_mb": {"label": "Max Size (MB)", "description": "Maximum memory allocated for the cache.", "type": "integer", "min": 32, "max": 2048, "unit": "MB"},
            "default_ttl_seconds": {"label": "Default TTL", "description": "Default time-to-live for cached responses.", "type": "integer", "min": 10, "max": 3600, "unit": "sec"},
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PLANNING & COLLABORATION
    # ═══════════════════════════════════════════════════════════════════════
    "planning_config": {
        "label": "Planning Date",
        "category": "planning",
        "description": "Controls the system planning date. Can be frozen to a specific date for testing or demos.",
        "fields": {
            "planning.planning_date": {"label": "Planning Date", "description": "Fixed planning date (YYYY-MM-DD). When 'use_system_date' is false, all date calculations reference this date.", "type": "text"},
            "planning.use_system_date": {"label": "Use System Date", "description": "When true, ignores the fixed planning_date and uses the actual system clock.", "type": "boolean"},
        },
    },
    "sop_config": {
        "label": "S&OP Cycle",
        "category": "planning",
        "description": "Sales & Operations Planning cycle configuration.",
        "fields": {
            "sop.planning_horizon_months": {"label": "Planning Horizon", "description": "Months ahead covered by the S&OP plan.", "type": "integer", "min": 3, "max": 24, "unit": "months"},
            "sop.demand_review_day": {"label": "Demand Review Day", "description": "Day of month for demand review stage.", "type": "integer", "min": 1, "max": 28},
            "sop.supply_review_day": {"label": "Supply Review Day", "description": "Day of month for supply review stage.", "type": "integer", "min": 1, "max": 28},
            "sop.pre_sop_day": {"label": "Pre-S&OP Day", "description": "Day of month for pre-S&OP alignment meeting.", "type": "integer", "min": 1, "max": 28},
            "sop.executive_sop_day": {"label": "Executive S&OP Day", "description": "Day of month for executive S&OP decision meeting.", "type": "integer", "min": 1, "max": 28},
            "sop.demand_baseline_model": {"label": "Demand Baseline", "description": "Source for the demand baseline in S&OP.", "type": "select", "options": ["champion", "consensus", "external", "latest"]},
            "sop.supply_gap_alert_pct": {"label": "Supply Gap Alert %", "description": "Supply-demand gap percentage that triggers an alert.", "type": "number", "min": 0.01, "max": 0.50, "step": 0.01, "unit": "%"},
        },
    },
    "consensus_config": {
        "label": "Consensus Planning",
        "category": "planning",
        "description": "Demand consensus process settings.",
        "fields": {
            "consensus_plan.approval_required_threshold_units": {"label": "Override Min Qty", "description": "Minimum quantity change that requires manager approval.", "type": "integer", "min": 1, "max": 10000, "unit": "units"},
            "consensus_plan.approval_required_threshold_pct": {"label": "Override Min %", "description": "Minimum percentage change that requires approval.", "type": "number", "min": 0.01, "max": 1.0, "step": 0.01, "unit": "%"},
            "consensus_plan.approval_required_threshold_value": {"label": "Override Min Impact", "description": "Minimum dollar impact that requires approval.", "type": "number", "min": 0, "max": 100000, "unit": "$"},
            "consensus_plan.auto_expiry.enabled": {"label": "Auto-Expiry", "description": "Automatically expire unconfirmed overrides after their effective period.", "type": "boolean"},
        },
    },
    "event_planning_config": {
        "label": "Event Planning",
        "category": "planning",
        "description": "Promotional and event calendar management.",
        "fields": {
            "event_planning.min_uplift_multiplier": {"label": "Min Uplift", "description": "Minimum allowed demand uplift multiplier. 0.0 allows events that reduce demand.", "type": "number", "min": 0, "max": 1.0, "step": 0.1},
            "event_planning.max_uplift_multiplier": {"label": "Max Uplift", "description": "Maximum allowed demand uplift multiplier.", "type": "number", "min": 1.0, "max": 10.0, "step": 0.5},
            "event_planning.require_approval_above_impact_value": {"label": "Approval Min Impact", "description": "Dollar impact above which event uplift requires approval.", "type": "number", "min": 0, "max": 100000, "unit": "$"},
            "event_planning.require_approval_above_uplift_pct": {"label": "Approval Min Uplift %", "description": "Uplift percentage above which approval is required.", "type": "number", "min": 0, "max": 100, "step": 1, "unit": "%"},
            "event_planning.post_event_lag_weeks": {"label": "Post-Event Lag", "description": "Weeks after event end to measure actual impact vs predicted uplift.", "type": "integer", "min": 1, "max": 8, "unit": "weeks"},
            "event_planning.min_advance_days": {"label": "Min Advance Days", "description": "Minimum days in advance an event must be created.", "type": "integer", "min": 1, "max": 30, "unit": "days"},
            "event_planning.conflict_window_days": {"label": "Conflict Window", "description": "Days within which overlapping events are flagged as conflicts.", "type": "integer", "min": 1, "max": 30, "unit": "days"},
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SYSTEM & INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════
    "auth_config": {
        "label": "Authentication & RBAC",
        "category": "system",
        "description": "JWT authentication and role-based access control settings.",
        "fields": {
            "jwt.algorithm": {"label": "JWT Algorithm", "description": "Cryptographic algorithm for JWT signing.", "type": "select", "options": ["HS256", "RS256", "ES256"]},
            "jwt.access_token_expire_minutes": {"label": "Access Token Lifetime", "description": "Minutes before an access token expires.", "type": "integer", "min": 5, "max": 1440, "unit": "min"},
            "jwt.refresh_token_expire_days": {"label": "Refresh Token Lifetime", "description": "Days before a refresh token expires.", "type": "integer", "min": 1, "max": 90, "unit": "days"},
            "governance.rate_limit_tiers.free.requests_per_minute": {"label": "Free Tier RPM", "description": "Maximum API requests per minute for free-tier users.", "type": "integer", "min": 10, "max": 1000},
            "governance.rate_limit_tiers.standard.requests_per_minute": {"label": "Standard Tier RPM", "description": "Maximum requests per minute for standard-tier users.", "type": "integer", "min": 50, "max": 5000},
            "governance.rate_limit_tiers.premium.requests_per_minute": {"label": "Premium Tier RPM", "description": "Maximum requests per minute for premium-tier users.", "type": "integer", "min": 100, "max": 10000},
            "governance.deprecation.sunset_header": {"label": "Sunset Header", "description": "Include HTTP Sunset header for deprecated API versions.", "type": "boolean"},
            "governance.versioning.current_version": {"label": "Current API Version", "description": "Active API version identifier.", "type": "text"},
        },
    },
    # api_governance_config merged into auth_config (governance section)
    "notification_config": {
        "label": "Notifications",
        "category": "system",
        "description": "Multi-channel notification configuration (Slack, Teams, Email, PagerDuty).",
        "fields": {
            "channels.slack.enabled": {"label": "Slack Enabled", "description": "Enable Slack notifications.", "type": "boolean"},
            "channels.teams.enabled": {"label": "Teams Enabled", "description": "Enable Microsoft Teams notifications.", "type": "boolean"},
            "channels.email.enabled": {"label": "Email Enabled", "description": "Enable email notifications.", "type": "boolean"},
            "channels.pagerduty.enabled": {"label": "PagerDuty Enabled", "description": "Enable PagerDuty alerting for critical incidents.", "type": "boolean"},
            "rate_limits.max_per_minute_per_channel": {"label": "Rate Limit RPM", "description": "Maximum notifications per minute per channel.", "type": "integer", "min": 1, "max": 100},
            "rate_limits.cooldown_seconds": {"label": "Cooldown Period", "description": "Minimum seconds between duplicate notifications.", "type": "integer", "min": 30, "max": 3600, "unit": "sec"},
            "rate_limits.burst_limit": {"label": "Burst Limit", "description": "Maximum notifications in a single burst before throttling.", "type": "integer", "min": 1, "max": 50},
        },
    },
    "ai_planner_config": {
        "label": "AI Planner",
        "category": "system",
        "description": "AI-powered planning agent that scans the portfolio for anomalies and generates actionable insights using LLMs.",
        "fields": {
            "provider": {"label": "LLM Provider", "description": "LLM provider for insight generation.", "type": "select", "options": ["openai", "anthropic", "local"]},
            "model": {"label": "Model Name", "description": "Specific LLM model to use (e.g., gpt-4o-mini).", "type": "text"},
            "max_tokens": {"label": "Max Tokens", "description": "Maximum tokens in LLM response.", "type": "integer", "min": 256, "max": 16384},
            "temperature": {"label": "Temperature", "description": "LLM sampling temperature. Lower = more deterministic.", "type": "number", "min": 0.0, "max": 2.0, "step": 0.1},
            "portfolio_scan_limit": {"label": "Max DFUs to Scan", "description": "Maximum DFUs analyzed per scheduled scan.", "type": "integer", "min": 5, "max": 200},
            "forecast_lookback_months": {"label": "Forecast Lookback", "description": "Months of forecast history to provide to the LLM.", "type": "integer", "min": 1, "max": 24, "unit": "months"},
            "insight_thresholds.stockout_dos_multiplier": {"label": "Stockout Risk Multiplier", "description": "DOS multiplier below which a DFU triggers stockout risk insight.", "type": "number", "min": 0.5, "max": 5.0, "step": 0.1},
            "insight_thresholds.excess_dos_days": {"label": "Excess DOS Threshold", "description": "Days of supply above which excess inventory insight is generated.", "type": "integer", "min": 30, "max": 365, "unit": "days"},
            "insight_thresholds.bias_threshold_pct": {"label": "Forecast Bias Threshold", "description": "Absolute bias percentage above which a bias insight is generated.", "type": "number", "min": 1, "max": 50, "step": 1, "unit": "%"},
            "insight_thresholds.champion_wape_critical": {"label": "Champion WAPE Critical", "description": "WAPE percentage above which a critical model degradation insight is generated.", "type": "number", "min": 1, "max": 100, "step": 1, "unit": "%"},
            "insight_thresholds.champion_wape_high": {"label": "Champion WAPE High", "description": "WAPE percentage above which a high-severity model degradation insight is generated.", "type": "number", "min": 1, "max": 100, "step": 1, "unit": "%"},
            "default_unit_cost": {"label": "Default Unit Cost", "description": "Default unit cost for financial impact estimation.", "type": "number", "min": 0.01, "max": 10000, "unit": "$"},
            "carrying_cost_rate": {"label": "Carrying Cost Rate", "description": "Annual carrying cost rate for financial impact calculations.", "type": "number", "min": 0.01, "max": 1.0, "step": 0.01},
            "stockout_cost_multiplier": {"label": "Stockout Cost Multiplier", "description": "Stockout cost as a multiple of unit cost.", "type": "number", "min": 1.0, "max": 20.0, "step": 0.5},
        },
    },
    "shared_constants": {
        "label": "Shared Constants",
        "category": "inventory",
        "description": "Canonical values shared across multiple config files via _includes. Service levels by ABC class, Z-table, financial defaults, and safety stock guard rails. Changes here propagate to all consuming configs.",
        "fields": {
            "service_levels_by_abc.A": {"label": "Service Level — A Class", "description": "Target service level for A-class (top 20% by volume) items.", "type": "number", "min": 0.80, "max": 1.0, "step": 0.01},
            "service_levels_by_abc.B": {"label": "Service Level — B Class", "description": "Target service level for B-class (middle 30% by volume) items.", "type": "number", "min": 0.80, "max": 1.0, "step": 0.01},
            "service_levels_by_abc.C": {"label": "Service Level — C Class", "description": "Target service level for C-class (bottom 50% by volume) items.", "type": "number", "min": 0.80, "max": 1.0, "step": 0.01},
            "service_levels_by_abc.default": {"label": "Service Level — Default", "description": "Fallback service level when ABC class is NULL or unrecognized.", "type": "number", "min": 0.80, "max": 1.0, "step": 0.01},
            "financial_defaults.carrying_cost_pct": {"label": "Carrying Cost %", "description": "Annual carrying cost as a fraction of inventory value.", "type": "number", "min": 0.01, "max": 1.0, "step": 0.01},
            "financial_defaults.default_ordering_cost": {"label": "Default Ordering Cost", "description": "Default cost per order (setup / procurement cost).", "type": "number", "min": 1.0, "max": 500.0, "step": 1.0, "unit": "$"},
            "financial_defaults.default_unit_cost": {"label": "Default Unit Cost", "description": "Default cost per unit (placeholder; override per item when available).", "type": "number", "min": 0.01, "max": 10000, "step": 0.01, "unit": "$"},
            "financial_defaults.default_moq": {"label": "Default MOQ", "description": "Minimum order quantity (units).", "type": "integer", "min": 1, "max": 10000},
            "financial_defaults.stockout_cost_per_unit": {"label": "Stockout Cost per Unit", "description": "Cost per unit of unmet demand.", "type": "number", "min": 0.0, "max": 1000.0, "step": 1.0, "unit": "$"},
            "safety_stock_guard_rails.min_ss_days": {"label": "Min Safety Stock Days", "description": "Minimum safety stock expressed as days of supply.", "type": "integer", "min": 0, "max": 30, "unit": "days"},
            "safety_stock_guard_rails.max_ss_days": {"label": "Max Safety Stock Days", "description": "Maximum safety stock cap expressed as days of supply.", "type": "integer", "min": 30, "max": 365, "unit": "days"},
            "safety_stock_guard_rails.lt_std_fallback_pct": {"label": "LT Std Fallback %", "description": "When lead time std is unknown, approximate as this fraction of mean lead time.", "type": "number", "min": 0.05, "max": 0.50, "step": 0.01},
        },
    },
}


# ---------------------------------------------------------------------------
# Configuration metadata completion
# ---------------------------------------------------------------------------
def _iter_config_leaves(data: Any, prefix: str = ""):
    """Yield dot paths for every value editable as one Settings control."""
    if isinstance(data, dict) and data:
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_config_leaves(value, path)
        return
    yield prefix, data


def _humanize_config_key(key: str) -> str:
    """Turn a YAML key into a compact UI label."""
    acronyms = {
        "ci": "CI",
        "cv": "CV",
        "dfu": "DFU",
        "id": "ID",
        "lgbm": "LightGBM",
        "shap": "SHAP",
        "sku": "SKU",
        "wape": "WAPE",
    }
    return " ".join(acronyms.get(part, part.capitalize()) for part in key.split("_"))


def _infer_field_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "text"


def _generated_field_meta(path: str, value: Any) -> FieldMeta:
    parts = path.split(".")
    if parts[0] == "algorithms" and len(parts) > 2:
        group = f"Model: {_humanize_config_key(parts[1])}"
    else:
        group = _humanize_config_key(parts[0])
    return {
        "label": _humanize_config_key(parts[-1]),
        "description": f"Configuration setting: {path}.",
        "type": _infer_field_type(value),
        "group": group,
    }


def _complete_registry_fields() -> None:
    """Expose every YAML leaf while preserving hand-authored field metadata."""
    for name, config_meta in CONFIG_REGISTRY.items():
        config_path = _resolve_yaml_path(f"{name}.yaml")
        if not config_path.exists():
            continue
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            logger.exception("Failed to load config metadata for %s", name)
            continue

        fields = config_meta["fields"]
        for path, value in _iter_config_leaves(raw):
            fields.setdefault(path, _generated_field_meta(path, value))

    # Operationally important controls get explicit guidance instead of a
    # generated description.
    fields = CONFIG_REGISTRY["forecast_pipeline_config"]["fields"]
    fields["algorithms.lgbm_cluster.params.tune_inline"].update(
        {
            "label": "Tune During Backtest",
            "description": (
                "Run Optuna inside each LightGBM backtest. Keep off for fast "
                "backtests; use tuning experiments to optimize parameters separately."
            ),
            "group": "Model: LightGBM",
        }
    )
    fields["backtest.embargo_months"].update(
        {
            "label": "Embargo Months",
            "description": (
                "Closed months withheld between the planning cutoff and the latest "
                "backtest prediction window to prevent leakage."
            ),
            "min": 0,
            "max": 24,
            "unit": "months",
            "group": "Clustering & Backtest",
        }
    )


_complete_registry_fields()


# ---------------------------------------------------------------------------
# Helper — resolve a dot-path in a nested dict
# ---------------------------------------------------------------------------
def _get_nested(data: dict, path: str, default: Any = None) -> Any:
    """Get value from nested dict using dot-path (e.g. 'lgbm.hyperparameters.n_estimators')."""
    keys = path.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def _set_nested(data: dict, path: str, value: Any) -> None:
    """Set value in nested dict using dot-path."""
    keys = path.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ConfigUpdate(BaseModel):
    values: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("")
def list_configs():
    """List all available configuration files with categories and descriptions."""
    configs = []
    for name, meta in CONFIG_REGISTRY.items():
        yaml_name = f"{name}.yaml"
        exists = _resolve_yaml_path(yaml_name).exists()
        configs.append({
            "name": name,
            "label": meta["label"],
            "category": meta["category"],
            "description": meta["description"],
            "exists": exists,
        })
    return {"categories": CATEGORIES, "configs": configs}


@router.get("/{name}")
def get_config(name: str):
    """Get configuration values with field metadata."""
    if name not in CONFIG_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown config: {name}")

    meta = CONFIG_REGISTRY[name]
    yaml_name = f"{name}.yaml"
    values = load_config(yaml_name)

    # Build field list with current values
    fields = []
    for path, field_meta in meta["fields"].items():
        current_value = _get_nested(values, path)
        fields.append({
            "path": path,
            "value": current_value,
            **field_meta,
        })

    return {
        "name": name,
        "label": meta["label"],
        "category": meta["category"],
        "description": meta["description"],
        "fields": fields,
        "raw": values,
    }


@router.put("/{name}", dependencies=[Depends(require_api_key)])
def update_config(name: str, body: ConfigUpdate):
    """Update configuration values and write back to YAML.

    Accepts a dict of dot-path -> value pairs, e.g.:
    { "algorithms.lgbm.n_estimators": 800, "algorithms.lgbm.recursive": false }
    """
    if name not in CONFIG_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown config: {name}")

    yaml_name = f"{name}.yaml"
    yaml_path = _resolve_yaml_path(yaml_name)

    # Load current values
    current = load_config(yaml_name)
    updated = copy.deepcopy(current)

    meta = CONFIG_REGISTRY[name]
    valid_paths = set(meta["fields"].keys())

    changes: list[str] = []
    for path, value in body.values.items():
        if path not in valid_paths:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown field path: {path}. Valid paths: {sorted(valid_paths)}",
            )
        old_val = _get_nested(updated, path)
        if old_val != value:
            _set_nested(updated, path, value)
            changes.append(path)

    if not changes:
        return {"name": name, "changed": [], "message": "No changes detected"}

    # Write updated config with backup
    backup_path = yaml_path.with_suffix(".yaml.bak")
    if yaml_path.exists():
        backup_path.write_text(yaml_path.read_text())

    with open(yaml_path, "w") as f:
        yaml.dump(updated, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Invalidate cached config
    reset_config(yaml_name)

    logger.info("Config %s updated: %s", name, changes)
    return {"name": name, "changed": changes, "message": f"Updated {len(changes)} field(s)"}


@router.post("/{name}/reset", dependencies=[Depends(require_api_key)])
def reset_to_backup(name: str):
    """Reset configuration to the last backup (.yaml.bak)."""
    if name not in CONFIG_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown config: {name}")

    yaml_name = f"{name}.yaml"
    yaml_path = _resolve_yaml_path(yaml_name)
    backup_path = yaml_path.with_suffix(".yaml.bak")

    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="No backup found")

    yaml_path.write_text(backup_path.read_text())
    reset_config(yaml_name)

    logger.info("Config %s reset from backup", name)
    return {"name": name, "message": "Reset to backup successful"}
