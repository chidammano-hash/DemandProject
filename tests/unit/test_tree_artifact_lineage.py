"""Production LightGBM artifact-lineage contract regressions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from common.ml.tree_artifact_lineage import (
    TREE_ARTIFACT_LINEAGE_KEY,
    ProductionTreeArtifactLineage,
    TreeArtifactLineageError,
    validate_tree_artifact_map,
)
from common.services.forecast_generation import GENERATOR_CONTRACT_VERSION

DATA_CHECKSUM = "a" * 64
ASSIGNMENT_CHECKSUM = "b" * 64


def _lineage(**overrides: object) -> ProductionTreeArtifactLineage:
    values: dict[str, object] = {
        "source_sales_batch_id": 91,
        "data_checksum": DATA_CHECKSUM,
        "history_end": date(2026, 6, 1),
        "cluster_experiment_id": 17,
        "cluster_assignment_count": 2,
        "cluster_assignment_checksum": ASSIGNMENT_CHECKSUM,
        "generator_contract_version": GENERATOR_CONTRACT_VERSION,
    }
    values.update(overrides)
    return ProductionTreeArtifactLineage(**values)  # type: ignore[arg-type]


def _artifact(
    cluster_label: object,
    *,
    lineage: ProductionTreeArtifactLineage | None = None,
) -> dict[str, object]:
    return {
        "model": object(),
        "feature_cols": ["qty_lag_1"],
        "model_id": "lgbm_cluster",
        "cluster_label": str(cluster_label),
        "training_mode": "production",
        "cluster_strategy": "per_cluster",
        TREE_ARTIFACT_LINEAGE_KEY: (lineage or _lineage()).to_metadata(),
    }


def test_lineage_is_frozen_and_serializes_canonical_metadata() -> None:
    lineage = _lineage()

    assert lineage.to_metadata() == {
        "source_sales_batch_id": 91,
        "data_checksum": DATA_CHECKSUM,
        "history_end": "2026-06-01",
        "cluster_experiment_id": 17,
        "cluster_assignment_count": 2,
        "cluster_assignment_checksum": ASSIGNMENT_CHECKSUM,
        "generator_contract_version": GENERATOR_CONTRACT_VERSION,
    }
    assert ProductionTreeArtifactLineage.from_metadata(lineage.to_metadata()) == lineage
    with pytest.raises(FrozenInstanceError):
        lineage.source_sales_batch_id = 92  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_sales_batch_id", 0, "positive integer"),
        ("source_sales_batch_id", True, "positive integer"),
        ("data_checksum", "not-a-checksum", "SHA-256"),
        ("data_checksum", "A" * 64, "lowercase"),
        ("history_end", date(2026, 6, 2), "first day"),
        ("cluster_experiment_id", 0, "positive integer"),
        ("cluster_experiment_id", True, "positive integer"),
        ("cluster_assignment_count", 0, "positive integer"),
        ("cluster_assignment_checksum", "not-a-checksum", "SHA-256"),
        (
            "generator_contract_version",
            "retired-generator-v0",
            "generator contract",
        ),
    ],
)
def test_lineage_rejects_invalid_values(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(TreeArtifactLineageError, match=message):
        _lineage(**{field: value})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.pop("history_end"), "missing.*history_end"),
        (lambda metadata: metadata.update({"unexpected": "value"}), "unexpected"),
        (lambda metadata: metadata.update({"history_end": "2026-06-02"}), "first day"),
        (lambda metadata: metadata.update({"source_sales_batch_id": "91"}), "positive integer"),
    ],
)
def test_metadata_parser_is_strict(
    mutation,
    message: str,
) -> None:
    metadata = _lineage().to_metadata()
    mutation(metadata)

    with pytest.raises(TreeArtifactLineageError, match=message):
        ProductionTreeArtifactLineage.from_metadata(metadata)


def test_metadata_parser_rejects_non_mapping() -> None:
    with pytest.raises(TreeArtifactLineageError, match="must be a mapping"):
        ProductionTreeArtifactLineage.from_metadata([])  # type: ignore[arg-type]


def test_metadata_parser_rejects_non_string_keys_cleanly() -> None:
    metadata: dict[object, object] = _lineage().to_metadata()
    metadata[1] = "unexpected"

    with pytest.raises(TreeArtifactLineageError, match="keys must be strings"):
        ProductionTreeArtifactLineage.from_metadata(metadata)  # type: ignore[arg-type]


def test_artifact_map_validation_returns_canonical_exact_cluster_map() -> None:
    lineage = _lineage()

    validated = validate_tree_artifact_map(
        {0: _artifact(0, lineage=lineage), "high_volume": _artifact("high_volume")},
        expected_cluster_labels={"0", "high_volume"},
        expected_lineage=lineage,
    )

    assert list(validated) == ["0", "high_volume"]
    assert validated["0"]["cluster_label"] == "0"


@pytest.mark.parametrize(
    ("lineage", "mismatched_field"),
    [
        (_lineage(source_sales_batch_id=92), "source_sales_batch_id"),
        (_lineage(data_checksum="b" * 64), "data_checksum"),
        (_lineage(history_end=date(2026, 5, 1)), "history_end"),
        (_lineage(cluster_experiment_id=18), "cluster_experiment_id"),
        (_lineage(cluster_assignment_count=3), "cluster_assignment_count"),
        (
            _lineage(cluster_assignment_checksum="c" * 64),
            "cluster_assignment_checksum",
        ),
    ],
)
def test_artifact_map_rejects_every_lineage_mismatch(
    lineage: ProductionTreeArtifactLineage,
    mismatched_field: str,
) -> None:
    with pytest.raises(
        TreeArtifactLineageError,
        match=rf"cluster '0'.*{mismatched_field}.*retrain",
    ):
        validate_tree_artifact_map(
            {0: _artifact(0, lineage=lineage)},
            expected_cluster_labels={0},
            expected_lineage=_lineage(),
        )


def test_artifact_map_rejects_retired_generator_contract() -> None:
    artifact = _artifact(0)
    artifact[TREE_ARTIFACT_LINEAGE_KEY] = {
        **_lineage().to_metadata(),
        "generator_contract_version": "retired-generator-v0",
    }

    with pytest.raises(
        TreeArtifactLineageError,
        match=r"cluster '0'.*generator contract.*retrain",
    ):
        validate_tree_artifact_map(
            {0: artifact},
            expected_cluster_labels={0},
            expected_lineage=_lineage(),
        )


@pytest.mark.parametrize(
    ("artifacts", "expected_labels", "message"),
    [
        ({0: _artifact(0)}, {0, 1}, r"missing=\['1'\]"),
        (
            {0: _artifact(0), 1: _artifact(1)},
            {0},
            r"unexpected=\['1'\]",
        ),
        (
            {"global": _artifact("global")},
            {0},
            r"missing=\['0'\].*unexpected=\['global'\].*fallback",
        ),
    ],
)
def test_artifact_map_requires_exact_cluster_coverage_without_fallback(
    artifacts: dict[object, dict[str, object]],
    expected_labels: set[object],
    message: str,
) -> None:
    with pytest.raises(TreeArtifactLineageError, match=message):
        validate_tree_artifact_map(
            artifacts,
            expected_cluster_labels=expected_labels,
            expected_lineage=_lineage(),
        )


def test_artifact_map_rejects_cluster_keys_that_canonicalize_to_duplicates() -> None:
    with pytest.raises(TreeArtifactLineageError, match="duplicate cluster label"):
        validate_tree_artifact_map(
            {0: _artifact(0), "0": _artifact(0)},
            expected_cluster_labels={0},
            expected_lineage=_lineage(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (TREE_ARTIFACT_LINEAGE_KEY, None, "lineage metadata"),
        ("model_id", "xgboost", "model_id"),
        ("cluster_label", "1", "cluster_label"),
        ("training_mode", "backtest", "production final fit"),
        ("cluster_strategy", "global", "per_cluster"),
    ],
)
def test_artifact_map_rejects_invalid_artifact_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = _artifact(0)
    artifact[field] = value

    with pytest.raises(
        TreeArtifactLineageError,
        match=rf"cluster '0'.*{message}.*retrain",
    ):
        validate_tree_artifact_map(
            {0: artifact},
            expected_cluster_labels={0},
            expected_lineage=_lineage(),
        )


@pytest.mark.parametrize("expected_labels", [set(), {None}, {""}])
def test_artifact_map_rejects_invalid_expected_cluster_population(
    expected_labels: set[object],
) -> None:
    with pytest.raises(TreeArtifactLineageError, match="expected cluster"):
        validate_tree_artifact_map(
            {},
            expected_cluster_labels=expected_labels,
            expected_lineage=_lineage(),
        )
