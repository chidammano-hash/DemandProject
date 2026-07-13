"""Fail-closed configuration lineage shared by the five forecast models."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest

from common.core.utils import load_config, load_forecast_pipeline_config

CANONICAL_MODELS = (
    "lgbm_cluster",
    "chronos2_enriched",
    "mstl",
    "nbeats",
    "nhits",
)


def _config() -> dict:
    return copy.deepcopy(load_forecast_pipeline_config())


@pytest.mark.parametrize("model_id", CANONICAL_MODELS)
def test_backtest_config_snapshot_is_deterministic_for_every_model(model_id: str) -> None:
    from common.ml.backtest_config import build_backtest_config_snapshot

    config = _config()
    profiles = load_config("cluster_tuning_profiles.yaml") if model_id == "lgbm_cluster" else None

    first = build_backtest_config_snapshot(
        config,
        model_id,
        cluster_tuning_profiles=profiles,
    )
    second = build_backtest_config_snapshot(
        copy.deepcopy(config),
        model_id,
        cluster_tuning_profiles=copy.deepcopy(profiles),
    )

    assert first == second
    assert first.model_id == model_id
    assert len(first.checksum) == 64
    assert first.config["algorithm"] == config["algorithms"][model_id]
    assert first.config["backtest"] == config["backtest"]
    assert first.config["clustering"] == config["clustering"]


def test_lightgbm_snapshot_captures_exact_cluster_tuning_profiles() -> None:
    from common.ml.backtest_config import build_backtest_config_snapshot

    config = _config()
    profiles = copy.deepcopy(load_config("cluster_tuning_profiles.yaml"))
    before = build_backtest_config_snapshot(
        config,
        "lgbm_cluster",
        cluster_tuning_profiles=profiles,
    )
    profiles["metadata"]["cluster_experiment_id"] += 1
    after = build_backtest_config_snapshot(
        config,
        "lgbm_cluster",
        cluster_tuning_profiles=profiles,
    )

    assert before.checksum != after.checksum
    assert before.config["cluster_tuning_profiles"] != after.config["cluster_tuning_profiles"]


@pytest.mark.parametrize(
    ("model_id", "mutate", "message"),
    [
        (
            "mstl",
            lambda cfg: cfg["backtest"].pop("embargo_months"),
            "backtest.embargo_months is required",
        ),
        (
            "chronos2_enriched",
            lambda cfg: cfg["algorithms"]["chronos2_enriched"]["params"].pop("model_revision"),
            "algorithms.chronos2_enriched.params.model_revision is required",
        ),
        (
            "nbeats",
            lambda cfg: cfg["algorithms"]["nbeats"]["params"].pop("deterministic"),
            "algorithms.nbeats.params.deterministic is required",
        ),
        (
            "lgbm_cluster",
            lambda cfg: cfg["backtest"].pop("recursive_lag_smooth"),
            "backtest.recursive_lag_smooth is required",
        ),
        (
            "nhits",
            lambda cfg: cfg["production_forecast"].pop("lookback_months"),
            "production_forecast.lookback_months is required",
        ),
    ],
)
def test_backtest_config_snapshot_rejects_missing_required_values(
    model_id: str,
    mutate: Callable[[dict], object],
    message: str,
) -> None:
    from common.ml.backtest_config import (
        ForecastConfigContractError,
        build_backtest_config_snapshot,
    )

    config = _config()
    mutate(config)

    with pytest.raises(ForecastConfigContractError, match=message):
        build_backtest_config_snapshot(
            config,
            model_id,
            cluster_tuning_profiles=(
                load_config("cluster_tuning_profiles.yaml") if model_id == "lgbm_cluster" else None
            ),
        )


@pytest.mark.parametrize(
    ("model_id", "mutate", "message"),
    [
        (
            "mstl",
            lambda cfg: cfg["algorithms"]["mstl"]["params"].__setitem__("num_workers", True),
            "algorithms.mstl.params.num_workers must be a positive integer",
        ),
        (
            "chronos2_enriched",
            lambda cfg: cfg["algorithms"]["chronos2_enriched"]["params"].__setitem__(
                "model_revision", 7
            ),
            "algorithms.chronos2_enriched.params.model_revision must be a non-empty string",
        ),
        (
            "nbeats",
            lambda cfg: cfg["algorithms"]["nbeats"]["params"].__setitem__("deterministic", "yes"),
            "algorithms.nbeats.params.deterministic must be true or false",
        ),
        (
            "lgbm_cluster",
            lambda cfg: cfg["backtest"].__setitem__("recursive_noise_pct", "five"),
            "backtest.recursive_noise_pct must be a finite number",
        ),
        (
            "nhits",
            lambda cfg: cfg["clustering"].__setitem__("enabled", 1),
            "clustering.enabled must be true or false",
        ),
    ],
)
def test_backtest_config_snapshot_rejects_type_invalid_values(
    model_id: str,
    mutate: Callable[[dict], object],
    message: str,
) -> None:
    from common.ml.backtest_config import (
        ForecastConfigContractError,
        build_backtest_config_snapshot,
    )

    config = _config()
    mutate(config)

    with pytest.raises(ForecastConfigContractError, match=message):
        build_backtest_config_snapshot(
            config,
            model_id,
            cluster_tuning_profiles=(
                load_config("cluster_tuning_profiles.yaml") if model_id == "lgbm_cluster" else None
            ),
        )


def test_lightgbm_snapshot_requires_profile_document() -> None:
    from common.ml.backtest_config import (
        ForecastConfigContractError,
        build_backtest_config_snapshot,
    )

    with pytest.raises(ForecastConfigContractError, match="cluster tuning profiles are required"):
        build_backtest_config_snapshot(_config(), "lgbm_cluster")
