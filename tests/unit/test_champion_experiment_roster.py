"""Champion experiment runner model-roster boundary tests."""

import pytest

from scripts.ml import run_champion_experiment as experiment


def test_experiment_runner_accepts_canonical_models(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "get_competing_model_ids",
        lambda: ["lgbm_cluster", "mstl"],
    )

    experiment._validate_experiment_models(["lgbm_cluster", "mstl"])


def test_experiment_runner_rejects_retired_models(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "get_competing_model_ids",
        lambda: ["lgbm_cluster", "mstl"],
    )

    with pytest.raises(ValueError, match="xgboost_cluster"):
        experiment._validate_experiment_models(["lgbm_cluster", "xgboost_cluster"])
