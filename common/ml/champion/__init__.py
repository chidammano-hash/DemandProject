"""Champion model selection strategies.

All strategies take a DataFrame of per-DFU per-month per-model errors
and return per-DFU per-month winner selections.

Input DataFrame schema (monthly_errors):
    item_id, customer_group, loc, startdate, model_id,
    basefcst_pref, tothist_dmd, abs_err
    [optional: execution_lag, fcstdate]

Output DataFrame schema:
    item_id, customer_group, loc, startdate, model_id,
    basefcst_pref, tothist_dmd
    (+ strategy-specific columns like prior_wape)

CRITICAL: Every strategy must be strictly causal — selection for month T
uses ONLY data from months < T - execution_lag, i.e. only months whose
actuals were available at the time the forecast was issued (fcstdate = T - L).

The previous monolithic ``common/ml/champion_strategies.py`` was split into
domain-focused modules; this package re-exports the full public API so that
``from common.ml.champion import STRATEGY_REGISTRY, strategy_expanding`` keeps
working.
"""

from __future__ import annotations

# Importing each module triggers its @register_strategy decorators, populating
# STRATEGY_REGISTRY. Order matters only insofar as dependencies (basic -> blend
# -> regime -> segment, etc.) need to resolve, but every module imports its
# helpers explicitly.
from common.ml.champion.registry import STRATEGY_REGISTRY, register_strategy
from common.ml.champion.helpers import (
    compute_ceiling,
    compute_strategy_accuracy,
    make_blend_row,
)
from common.ml.champion.basic import (
    strategy_decay,
    strategy_ensemble,
    strategy_ensemble_rolling,
    strategy_expanding,
    strategy_rolling,
)
from common.ml.champion.blend import (
    strategy_adaptive_ensemble,
    strategy_adversarial_filter,
    strategy_bayesian_model_avg,
    strategy_cascade_ensemble,
    strategy_diverse_ensemble,
    strategy_error_correcting,
    strategy_learned_blend,
    strategy_ridge_blend,
    strategy_shrinkage_blend,
    strategy_uncertainty_aware,
)
from common.ml.champion.meta import (
    strategy_hybrid_meta_router,
    strategy_meta_learner,
)
from common.ml.champion.regime import (
    strategy_dynamic_window,
    strategy_regime_adaptive,
)
from common.ml.champion.routing import (
    strategy_dfu_strategy_router,
    strategy_hybrid_warmup,
    strategy_optimized_decay,
    strategy_seasonal,
    strategy_stacked_strategies,
)
from common.ml.champion.segment import (
    strategy_cluster_regime_hybrid,
    strategy_per_cluster,
    strategy_per_segment,
)
from common.ml.champion.bandit import (
    strategy_exp3,
    strategy_linucb,
    strategy_thompson_ensemble,
    strategy_thompson_sampling,
)

__all__ = [
    # Registry
    "STRATEGY_REGISTRY",
    "register_strategy",
    # Metrics / helpers
    "compute_ceiling",
    "compute_strategy_accuracy",
    "make_blend_row",
    # Basic
    "strategy_decay",
    "strategy_ensemble",
    "strategy_ensemble_rolling",
    "strategy_expanding",
    "strategy_rolling",
    # Blending / ensembles
    "strategy_adaptive_ensemble",
    "strategy_adversarial_filter",
    "strategy_bayesian_model_avg",
    "strategy_cascade_ensemble",
    "strategy_diverse_ensemble",
    "strategy_error_correcting",
    "strategy_learned_blend",
    "strategy_ridge_blend",
    "strategy_shrinkage_blend",
    "strategy_uncertainty_aware",
    # Meta-learner
    "strategy_hybrid_meta_router",
    "strategy_meta_learner",
    # Regime detection
    "strategy_dynamic_window",
    "strategy_regime_adaptive",
    # Meta-routing
    "strategy_dfu_strategy_router",
    "strategy_hybrid_warmup",
    "strategy_optimized_decay",
    "strategy_seasonal",
    "strategy_stacked_strategies",
    # Segment / cluster
    "strategy_cluster_regime_hybrid",
    "strategy_per_cluster",
    "strategy_per_segment",
    # Bandits
    "strategy_exp3",
    "strategy_linucb",
    "strategy_thompson_ensemble",
    "strategy_thompson_sampling",
]
