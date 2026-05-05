"""Promoted model params <-> backtest params consistency tests.

Ensures the parameters shown as "promoted" in the UI are exactly the
parameters used when running a backtest.  The data-flow is:

    Promote endpoint -> writes to forecast_pipeline_config.yaml algorithms.<model_id>.params
    Backtest script  -> reads  from forecast_pipeline_config.yaml -> MODEL_REGISTRY default_params

These tests catch drift between the promote-side allowed keys, the
config file contents, and the backtest-side keys for LGBM, CatBoost,
and XGBoost.
"""
from __future__ import annotations

import pytest

from common.core.utils import load_forecast_pipeline_config

# ---------------------------------------------------------------------------
# Promote-side key sets  (authoritative source: API routers)
# ---------------------------------------------------------------------------
from api.routers.forecasting.lgbm_tuning import _LGBM_PARAM_KEYS  # noqa: E402
from api.routers.forecasting.model_tuning import _MODEL_CONFIGS    # noqa: E402

PROMOTE_KEYS: dict[str, set[str]] = {
    "lgbm": set(_LGBM_PARAM_KEYS),
    "catboost": set(_MODEL_CONFIGS["catboost"]["param_keys"]),
    "xgboost": set(_MODEL_CONFIGS["xgboost"]["param_keys"]),
}

# ---------------------------------------------------------------------------
# Backtest-side key sets  (extracted from MODEL_REGISTRY default_params lambdas
# in scripts/run_backtest.py -- these are the keys each lambda reads via
# algo.get("key", default)).  Fixed constants (verbosity, random_state, etc.)
# are excluded because they are not read from config.
# ---------------------------------------------------------------------------
BACKTEST_TUNABLE_KEYS: dict[str, set[str]] = {
    "lgbm": {
        "n_estimators", "learning_rate", "num_leaves", "min_child_samples",
        "max_depth", "min_gain_to_split", "subsample", "bagging_freq",
        "colsample_bytree", "feature_fraction_bynode", "reg_lambda",
        "reg_alpha", "path_smooth", "max_bin",
    },
    "catboost": {
        "iterations", "learning_rate", "depth", "l2_leaf_reg",
        "border_count", "max_ctr_complexity",
    },
    "xgboost": {
        "n_estimators", "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree",
    },
}

# Pipeline config algorithm IDs for each model family
_PIPELINE_ALGO_IDS = {"lgbm": "lgbm_cluster", "catboost": "catboost_cluster", "xgboost": "xgboost_cluster"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def algo_config() -> dict:
    """Load forecast_pipeline_config.yaml once for the module."""
    return load_forecast_pipeline_config()


def _get_params(cfg: dict, model: str) -> dict:
    """Extract the params sub-dict for a model from pipeline config."""
    algo_id = _PIPELINE_ALGO_IDS[model]
    entry = cfg.get("algorithms", {}).get(algo_id, {})
    return dict(entry.get("params", {}))


# ===========================================================================
# 1. Promote keys cover all backtest-tunable keys
#    If a backtest lambda reads algo.get("foo"), then "foo" must be promotable.
# ===========================================================================
@pytest.mark.parametrize("model", ["lgbm", "catboost", "xgboost"])
def test_promote_keys_cover_backtest_keys(model: str):
    """Every key the backtest reads from config must be in the promote allow-list."""
    missing = BACKTEST_TUNABLE_KEYS[model] - PROMOTE_KEYS[model]
    assert not missing, (
        f"{model}: backtest reads {sorted(missing)} from config but "
        f"promote endpoint does not allow writing them"
    )


# ===========================================================================
# 2. forecast_pipeline_config.yaml has values for every backtest-tunable key
#    If the config is missing a key the backtest reads, the backtest would
#    silently fall back to a hardcoded default -- not the promoted value.
# ===========================================================================
@pytest.mark.parametrize("model", ["lgbm", "catboost", "xgboost"])
def test_config_has_all_backtest_keys(algo_config, model: str):
    """forecast_pipeline_config.yaml must contain all keys the backtest reads."""
    params = _get_params(algo_config, model)
    missing = BACKTEST_TUNABLE_KEYS[model] - set(params.keys())
    assert not missing, (
        f"{model}: forecast_pipeline_config.yaml is missing keys {sorted(missing)} "
        f"that the backtest reads -- backtest would use hardcoded defaults instead"
    )


# ===========================================================================
# 3. Every hyperparam in the config params section is promotable
#    If someone adds a new key to the YAML but not to the promote endpoint,
#    the UI can never update that key.
# ===========================================================================
# Non-hyperparameter keys inside params that are NOT model-tunable params
_PARAMS_META_KEYS = {
    "recursive", "shap_select", "shap_threshold", "shap_top_n",
    "shap_sample_size", "tune_inline", "params_file", "customer_features",
    # Feature-selection thresholds — applied before training, not model hparams.
    "correlation_filter", "correlation_threshold",
    "variance_filter", "variance_threshold",
    # Config-only knobs the tuning UI does not expose.
    "objective", "quantile_heads",
}


@pytest.mark.parametrize("model", ["lgbm", "catboost", "xgboost"])
def test_config_params_are_all_promotable(algo_config, model: str):
    """Every hyperparameter in config params must be in the promote allow-list."""
    params = _get_params(algo_config, model)
    param_keys = {k for k in params if k not in _PARAMS_META_KEYS}
    not_promotable = param_keys - PROMOTE_KEYS[model]
    assert not not_promotable, (
        f"{model}: forecast_pipeline_config.yaml has keys {sorted(not_promotable)} "
        f"that the promote endpoint will NOT write -- UI can never set them"
    )


# ===========================================================================
# 4. Promoted run params -> config -> backtest round-trip
#    Simulate a promote: given a run's params, filter through the promote
#    allow-list, merge into config, then verify the backtest lambda would
#    read exactly those values.
# ===========================================================================
_SAMPLE_PROMOTED_PARAMS = {
    "lgbm": {
        "n_estimators": 2500, "learning_rate": 0.015, "num_leaves": 128,
        "min_child_samples": 35, "max_depth": -1, "min_gain_to_split": 0.005,
        "subsample": 0.85, "bagging_freq": 1, "colsample_bytree": 0.75,
        "feature_fraction_bynode": 0.65, "reg_lambda": 2.0, "reg_alpha": 0.15,
        "path_smooth": 5.0,
    },
    "catboost": {
        "iterations": 3000, "learning_rate": 0.008, "depth": 10,
        "l2_leaf_reg": 7.5, "border_count": 64,
        "bootstrap_type": "MVS", "model_size_reg": 0.08,
        "boost_from_average": True,
    },
    "xgboost": {
        "n_estimators": 2800, "learning_rate": 0.009, "max_depth": 10,
        "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.75,
    },
}


@pytest.mark.parametrize("model", ["lgbm", "catboost", "xgboost"])
def test_promote_roundtrip_matches_backtest(algo_config, model: str):
    """After promoting, the backtest must read the promoted values, not defaults."""
    promoted_params = _SAMPLE_PROMOTED_PARAMS[model]
    params = _get_params(algo_config, model)

    # Step 1: Filter through promote allow-list (same as endpoint does)
    allowed = PROMOTE_KEYS[model]
    written = {k: v for k, v in promoted_params.items() if k in allowed}
    assert written, f"No params survived filtering for {model}"

    # Step 2: Merge into a copy of the params (same as endpoint does)
    import copy
    section = copy.deepcopy(params)
    for k, v in written.items():
        section[k] = v

    # Step 3: Simulate what the backtest default_params lambda does --
    # read each tunable key from the config section via .get()
    for key in BACKTEST_TUNABLE_KEYS[model]:
        config_value = section.get(key)
        if key in written:
            assert config_value == written[key], (
                f"{model}: backtest would read {key}={config_value!r} from config "
                f"but promote wrote {written[key]!r}"
            )
        else:
            # Key not in promoted params -- config should still have a value
            # (from the original YAML) so backtest doesn't fall back to hardcoded default
            assert config_value is not None, (
                f"{model}: key {key!r} not in promoted params and missing from "
                f"config -- backtest would use hardcoded default"
            )


# ===========================================================================
# 5. Promote key sets are in sync with each other
#    The LGBM router and model_tuning router must not accidentally define
#    overlapping or inconsistent key sets.
# ===========================================================================
def test_no_cross_model_key_leakage():
    """Model-specific param names should not appear in other models' key sets."""
    # LGBM-only params
    lgbm_specific = {"num_leaves", "min_child_samples", "bagging_freq",
                     "feature_fraction_bynode", "path_smooth", "min_gain_to_split"}
    # CatBoost-only params
    catboost_specific = {"iterations", "depth", "l2_leaf_reg", "border_count",
                         "bagging_temperature", "random_strength", "min_data_in_leaf",
                         "bootstrap_type", "model_size_reg", "leaf_estimation_method",
                         "boost_from_average", "leaf_estimation_iterations",
                         "score_function", "colsample_bylevel"}
    # XGBoost-only params
    xgboost_specific = {"min_child_weight", "booster", "rate_drop", "skip_drop",
                        "colsample_bynode"}

    leaked_to_cb = lgbm_specific & PROMOTE_KEYS["catboost"]
    leaked_to_xgb = lgbm_specific & PROMOTE_KEYS["xgboost"]
    leaked_to_lgbm_from_cb = catboost_specific & PROMOTE_KEYS["lgbm"]
    leaked_to_lgbm_from_xgb = xgboost_specific & PROMOTE_KEYS["lgbm"]

    assert not leaked_to_cb, f"LGBM-specific keys leaked to CatBoost promote: {leaked_to_cb}"
    assert not leaked_to_xgb, f"LGBM-specific keys leaked to XGBoost promote: {leaked_to_xgb}"
    assert not leaked_to_lgbm_from_cb, f"CatBoost-specific keys leaked to LGBM promote: {leaked_to_lgbm_from_cb}"
    assert not leaked_to_lgbm_from_xgb, f"XGBoost-specific keys leaked to LGBM promote: {leaked_to_lgbm_from_xgb}"
