"""Immutable config lineage for direct MSTL and Chronos production adapters."""

import pytest


def _algorithms(*, revision: str = "rev-a") -> dict[str, object]:
    return {
        "mstl": {
            "type": "statistical",
            "params": {"season_length": 12, "min_history": 25, "num_workers": 8},
        },
        "chronos2_enriched": {
            "type": "foundation",
            "params": {
                "model_name": "amazon/chronos-2",
                "model_revision": revision,
                "prediction_length": 6,
                "device": "auto",
                "batch_size": 1024,
                "num_workers": 1,
            },
        },
        "nhits": {"type": "deep_learning", "params": {"h": 6}},
    }


def test_direct_lineage_records_exact_config_and_pinned_chronos_identity() -> None:
    from common.ml.direct_model_lineage import build_direct_model_config_lineage

    lineage = build_direct_model_config_lineage(
        _algorithms(),
        {"mstl", "chronos2_enriched", "nhits"},
    )

    assert set(lineage) == {"mstl", "chronos2_enriched"}
    assert len(lineage["mstl"]["config_checksum"]) == 64
    chronos = lineage["chronos2_enriched"]
    assert chronos["config"]["params"]["model_name"] == "amazon/chronos-2"
    assert chronos["config"]["params"]["model_revision"] == "rev-a"


def test_direct_lineage_rejects_a_config_change_after_generation() -> None:
    from common.ml.direct_model_lineage import (
        DirectModelLineageError,
        build_direct_model_config_lineage,
        validate_direct_model_config_lineage,
    )

    generated = build_direct_model_config_lineage(
        _algorithms(revision="rev-a"),
        {"chronos2_enriched"},
    )

    with pytest.raises(DirectModelLineageError, match=r"chronos2_enriched.*changed"):
        validate_direct_model_config_lineage(
            generated,
            algorithms=_algorithms(revision="rev-b"),
            required_model_ids={"chronos2_enriched"},
        )


def test_direct_lineage_requires_exact_direct_model_set() -> None:
    from common.ml.direct_model_lineage import (
        DirectModelLineageError,
        build_direct_model_config_lineage,
        validate_direct_model_config_lineage,
    )

    generated = build_direct_model_config_lineage(_algorithms(), {"mstl"})

    with pytest.raises(DirectModelLineageError, match=r"missing.*chronos2_enriched"):
        validate_direct_model_config_lineage(
            generated,
            algorithms=_algorithms(),
            required_model_ids={"mstl", "chronos2_enriched"},
        )


def test_generator_metadata_records_routed_source_roster_and_direct_configs() -> None:
    from common.ml.direct_model_lineage import (
        DIRECT_MODEL_CONFIG_METADATA_KEY,
        SOURCE_MODEL_ROSTER_METADATA_KEY,
    )
    from scripts.forecasting.generate_production_forecasts import (
        _direct_generation_metadata,
    )

    metadata = _direct_generation_metadata(
        {"algorithms": _algorithms()},
        {"lgbm_cluster", "mstl", "chronos2_enriched"},
    )

    assert metadata[SOURCE_MODEL_ROSTER_METADATA_KEY] == [
        "chronos2_enriched",
        "lgbm_cluster",
        "mstl",
    ]
    assert set(metadata[DIRECT_MODEL_CONFIG_METADATA_KEY]) == {
        "chronos2_enriched",
        "mstl",
    }
