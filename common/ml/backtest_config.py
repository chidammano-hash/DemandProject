"""Validated, checksummed configuration contract for forecast model execution."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from common.core.utils import load_config, load_forecast_pipeline_config

BACKTEST_CONFIG_CONTRACT_VERSION = 1
BACKTEST_CONFIG_METADATA_KEY = "backtest_config"
CANONICAL_FORECAST_MODEL_IDS = frozenset(
    {"lgbm_cluster", "chronos2_enriched", "mstl", "nbeats", "nhits"}
)

_MODEL_TYPES = {
    "lgbm_cluster": "tree",
    "chronos2_enriched": "foundation",
    "mstl": "statistical",
    "nbeats": "deep_learning",
    "nhits": "deep_learning",
}
_COMMON_BACKTEST_KEYS = frozenset(
    {
        "n_timeframes",
        "embargo_months",
        "forecast_horizon",
        "early_stop_pct",
        "shap_retrain_threshold",
        "shap_min_features",
        "n_seeds",
        "tweedie_variance_power",
        "baseline_intermittent",
        "intermittent_threshold",
        "lumpy_threshold",
        "output_dir",
        "recursive_noise_enabled",
        "recursive_noise_pct",
        "recursive_lag_smooth",
    }
)
_MODEL_PARAM_KEYS = {
    "lgbm_cluster": frozenset(
        {
            "recursive",
            "shap_select",
            "shap_threshold",
            "shap_top_n",
            "shap_sample_size",
            "correlation_filter",
            "correlation_threshold",
            "variance_filter",
            "variance_threshold",
            "tune_inline",
            "params_file",
            "objective",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "max_depth",
            "min_gain_to_split",
            "subsample",
            "bagging_freq",
            "colsample_bytree",
            "feature_fraction_bynode",
            "reg_lambda",
            "reg_alpha",
            "path_smooth",
            "max_bin",
            "quantile_heads",
        }
    ),
    "chronos2_enriched": frozenset(
        {
            "device",
            "batch_size",
            "prediction_length",
            "min_history",
            "model_name",
            "model_revision",
            "num_workers",
        }
    ),
    "mstl": frozenset({"season_length", "min_history", "num_workers"}),
    "nbeats": frozenset(
        {
            "h",
            "input_size",
            "max_steps",
            "batch_size",
            "learning_rate",
            "scaler_type",
            "early_stop_patience_steps",
            "min_history",
            "random_seed",
            "start_padding_enabled",
            "val_size",
            "deterministic",
        }
    ),
    "nhits": frozenset(
        {
            "h",
            "input_size",
            "max_steps",
            "batch_size",
            "learning_rate",
            "scaler_type",
            "early_stop_patience_steps",
            "min_history",
            "random_seed",
            "start_padding_enabled",
            "val_size",
            "deterministic",
        }
    ),
}


class ForecastConfigContractError(ValueError):
    """A required forecast execution setting is missing or invalid."""


@dataclass(frozen=True, slots=True)
class BacktestConfigSnapshot:
    """Immutable canonical JSON plus its SHA-256 identity."""

    model_id: str
    checksum: str
    _config_json: str

    @property
    def config(self) -> dict[str, Any]:
        """Return an isolated copy of the validated canonical payload."""
        return json.loads(self._config_json)

    def as_metadata(self) -> dict[str, Any]:
        """Serialize the snapshot for backtest, artifact, or generation evidence."""
        return {
            "model_id": self.model_id,
            "config_checksum": self.checksum,
            "config": self.config,
        }


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForecastConfigContractError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ForecastConfigContractError(f"{path} keys must be strings")
    return value


def _required(mapping: Mapping[str, Any], key: str, *, path: str) -> Any:
    if key not in mapping:
        raise ForecastConfigContractError(f"{path}.{key} is required")
    return mapping[key]


def _positive_int(value: object, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ForecastConfigContractError(f"{path} must be a positive integer")


def _nonnegative_int(value: object, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ForecastConfigContractError(f"{path} must be a non-negative integer")


def _finite_number(value: object, *, path: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ForecastConfigContractError(f"{path} must be a finite number")


def _boolean(value: object, *, path: str) -> None:
    if not isinstance(value, bool):
        raise ForecastConfigContractError(f"{path} must be true or false")


def _nonempty_string(value: object, *, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ForecastConfigContractError(f"{path} must be a non-empty string")


def _validate_backtest(backtest: Mapping[str, Any]) -> None:
    for key in sorted(_COMMON_BACKTEST_KEYS):
        _required(backtest, key, path="backtest")
    for key in ("n_timeframes", "forecast_horizon", "shap_min_features", "n_seeds"):
        _positive_int(backtest[key], path=f"backtest.{key}")
    _nonnegative_int(backtest["embargo_months"], path="backtest.embargo_months")
    _nonempty_string(backtest["output_dir"], path="backtest.output_dir")
    for key in ("baseline_intermittent", "recursive_noise_enabled"):
        _boolean(backtest[key], path=f"backtest.{key}")
    for key in (
        "early_stop_pct",
        "shap_retrain_threshold",
        "tweedie_variance_power",
        "intermittent_threshold",
        "lumpy_threshold",
        "recursive_noise_pct",
        "recursive_lag_smooth",
    ):
        _finite_number(backtest[key], path=f"backtest.{key}")


def _validate_algorithm(model_id: str, algorithm: Mapping[str, Any]) -> None:
    path = f"algorithms.{model_id}"
    model_type = _required(algorithm, "type", path=path)
    if model_type != _MODEL_TYPES[model_id]:
        raise ForecastConfigContractError(f"{path}.type must be {_MODEL_TYPES[model_id]!r}")
    for key in ("enabled", "tune", "backtest", "compete", "forecast"):
        _boolean(_required(algorithm, key, path=path), path=f"{path}.{key}")
    params = _mapping(_required(algorithm, "params", path=path), path=f"{path}.params")
    for key in sorted(_MODEL_PARAM_KEYS[model_id]):
        _required(params, key, path=f"{path}.params")

    if model_id == "mstl":
        for key in ("season_length", "min_history", "num_workers"):
            _positive_int(params[key], path=f"{path}.params.{key}")
    elif model_id == "chronos2_enriched":
        for key in ("batch_size", "prediction_length", "min_history", "num_workers"):
            _positive_int(params[key], path=f"{path}.params.{key}")
        for key in ("device", "model_name", "model_revision"):
            _nonempty_string(params[key], path=f"{path}.params.{key}")
    elif model_id in {"nbeats", "nhits"}:
        for key in (
            "h",
            "input_size",
            "max_steps",
            "batch_size",
            "min_history",
        ):
            _positive_int(params[key], path=f"{path}.params.{key}")
        _nonnegative_int(params["val_size"], path=f"{path}.params.val_size")
        patience = params["early_stop_patience_steps"]
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < -1:
            raise ForecastConfigContractError(
                f"{path}.params.early_stop_patience_steps must be an integer at least -1"
            )
        if isinstance(params["random_seed"], bool) or not isinstance(params["random_seed"], int):
            raise ForecastConfigContractError(f"{path}.params.random_seed must be an integer")
        _finite_number(params["learning_rate"], path=f"{path}.params.learning_rate")
        _nonempty_string(params["scaler_type"], path=f"{path}.params.scaler_type")
        for key in ("start_padding_enabled", "deterministic"):
            _boolean(params[key], path=f"{path}.params.{key}")
    else:
        for key in (
            "n_estimators",
            "num_leaves",
            "min_child_samples",
            "bagging_freq",
            "max_bin",
            "shap_sample_size",
        ):
            _positive_int(params[key], path=f"{path}.params.{key}")
        for key in (
            "learning_rate",
            "shap_threshold",
            "correlation_threshold",
            "variance_threshold",
            "min_gain_to_split",
            "subsample",
            "colsample_bytree",
            "feature_fraction_bynode",
            "reg_lambda",
            "reg_alpha",
            "path_smooth",
        ):
            _finite_number(params[key], path=f"{path}.params.{key}")
        for key in (
            "recursive",
            "shap_select",
            "correlation_filter",
            "variance_filter",
            "tune_inline",
        ):
            _boolean(params[key], path=f"{path}.params.{key}")
        _nonempty_string(params["objective"], path=f"{path}.params.objective")
        if params["params_file"] is not None:
            _nonempty_string(params["params_file"], path=f"{path}.params.params_file")
        if params["shap_top_n"] is not None:
            _positive_int(params["shap_top_n"], path=f"{path}.params.shap_top_n")
        if not isinstance(params["quantile_heads"], list):
            raise ForecastConfigContractError(f"{path}.params.quantile_heads must be a list")


def build_backtest_config_snapshot(
    pipeline_config: Mapping[str, object],
    model_id: str,
    *,
    cluster_tuning_profiles: Mapping[str, object] | None = None,
) -> BacktestConfigSnapshot:
    """Validate and checksum one model's exact lifecycle execution config."""
    if model_id not in CANONICAL_FORECAST_MODEL_IDS:
        raise ForecastConfigContractError(
            f"Forecast model {model_id!r} is outside the canonical five-model roster"
        )
    root = _mapping(pipeline_config, path="forecast configuration")
    algorithms = _mapping(
        _required(root, "algorithms", path="forecast configuration"),
        path="algorithms",
    )
    algorithm = _mapping(
        _required(algorithms, model_id, path="algorithms"),
        path=f"algorithms.{model_id}",
    )
    backtest = _mapping(
        _required(root, "backtest", path="forecast configuration"),
        path="backtest",
    )
    clustering = _mapping(
        _required(root, "clustering", path="forecast configuration"),
        path="clustering",
    )
    tuning = _mapping(
        _required(root, "tuning", path="forecast configuration"),
        path="tuning",
    )
    production = _mapping(
        _required(root, "production_forecast", path="forecast configuration"),
        path="production_forecast",
    )
    _validate_algorithm(model_id, algorithm)
    _validate_backtest(backtest)
    _boolean(
        _required(clustering, "enabled", path="clustering"),
        path="clustering.enabled",
    )
    _positive_int(
        _required(production, "lookback_months", path="production_forecast"),
        path="production_forecast.lookback_months",
    )

    payload: dict[str, Any] = {
        "contract_version": BACKTEST_CONFIG_CONTRACT_VERSION,
        "model_id": model_id,
        "algorithm": dict(algorithm),
        "backtest": dict(backtest),
        "clustering": dict(clustering),
        "tuning": dict(tuning),
        "production_forecast": dict(production),
    }
    if model_id == "lgbm_cluster":
        if cluster_tuning_profiles is None:
            raise ForecastConfigContractError("LightGBM cluster tuning profiles are required")
        payload["cluster_tuning_profiles"] = dict(
            _mapping(cluster_tuning_profiles, path="cluster_tuning_profiles")
        )
    try:
        config_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ForecastConfigContractError(
            "Forecast execution configuration must be JSON-serializable"
        ) from exc
    return BacktestConfigSnapshot(
        model_id=model_id,
        checksum=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        _config_json=config_json,
    )


def load_backtest_config_snapshot(model_id: str) -> BacktestConfigSnapshot:
    """Load the canonical YAML documents and return one validated snapshot."""
    profiles = load_config("cluster_tuning_profiles.yaml") if model_id == "lgbm_cluster" else None
    return build_backtest_config_snapshot(
        load_forecast_pipeline_config(),
        model_id,
        cluster_tuning_profiles=profiles,
    )
