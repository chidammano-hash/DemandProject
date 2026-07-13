"""Current-config lineage for immutable forecast generation runs."""

import pytest


def _pipeline(*, lookback_months: int = 36) -> dict[str, object]:
    return {
        "algorithms": {
            "lgbm_cluster": {"type": "tree", "params": {"recursive": True}},
            "mstl": {
                "type": "statistical",
                "params": {"season_length": 12, "min_history": 25},
            },
        },
        "production_forecast": {
            "lookback_months": lookback_months,
            "horizon_months": 24,
            "confidence_interval": {"enabled": True, "z_lower": 1.282},
        },
        "forecast_snapshot": {"active_window_months": 12},
        "backtest": {"recursive_lag_smooth": 0.15},
        "clustering": {"enabled": True},
    }


def test_generation_config_lineage_changes_with_global_inference_policy() -> None:
    from common.ml.generation_config_lineage import build_generation_config_lineage

    before = build_generation_config_lineage(
        _pipeline(lookback_months=36),
        {"lgbm_cluster", "mstl"},
    )
    after = build_generation_config_lineage(
        _pipeline(lookback_months=48),
        {"lgbm_cluster", "mstl"},
    )

    assert before["config_checksum"] != after["config_checksum"]
    assert before["config"]["production_forecast"]["lookback_months"] == 36


def test_generation_config_lineage_rejects_current_config_drift() -> None:
    from common.ml.generation_config_lineage import (
        GenerationConfigLineageError,
        build_generation_config_lineage,
        validate_generation_config_lineage,
    )

    generated = build_generation_config_lineage(
        _pipeline(lookback_months=36),
        {"lgbm_cluster", "mstl"},
    )

    with pytest.raises(GenerationConfigLineageError, match="changed"):
        validate_generation_config_lineage(
            generated,
            pipeline_config=_pipeline(lookback_months=48),
            source_model_ids={"lgbm_cluster", "mstl"},
        )
