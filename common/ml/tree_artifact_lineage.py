"""Strict lineage contract for production per-cluster LightGBM artifacts.

Training stamps every cluster artifact with the same immutable source and
clustering identity. Generation validates the complete loaded artifact map in
one pass before resolving any individual cluster, preventing mixed-generation
models or an implicit global fallback from reaching production forecasts.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral

from common.services.forecast_generation import GENERATOR_CONTRACT_VERSION

TREE_ARTIFACT_LINEAGE_KEY = "tree_artifact_lineage"
LIGHTGBM_CLUSTER_MODEL_ID = "lgbm_cluster"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LINEAGE_FIELDS = (
    "source_sales_batch_id",
    "data_checksum",
    "history_end",
    "cluster_experiment_id",
    "cluster_assignment_count",
    "cluster_assignment_checksum",
    "generator_contract_version",
)
_LINEAGE_FIELD_SET = frozenset(_LINEAGE_FIELDS)
_RETRAIN_ACTION = (
    "retrain lgbm_cluster production models against the current completed sales "
    "batch and promoted clustering"
)

LineageMetadata = dict[str, int | str | None]


class TreeArtifactLineageError(RuntimeError):
    """A production tree artifact cannot prove the required lineage."""


def _require_positive_integer(value: object, *, field_name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise TreeArtifactLineageError(f"{field_name} must be a positive integer")
    return int(value)


def _require_sha256(value: object, *, field_name: str = "data_checksum") -> str:
    if not isinstance(value, str):
        raise TreeArtifactLineageError(f"{field_name} must be a 64-character SHA-256 hex digest")
    if value != value.lower():
        raise TreeArtifactLineageError(f"{field_name} must use lowercase SHA-256 hex")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise TreeArtifactLineageError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return value


def _require_history_month(value: object) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TreeArtifactLineageError("history_end must be a date")
    if value.day != 1:
        raise TreeArtifactLineageError(
            "history_end must be the first day of the latest closed history month"
        )
    return value


@dataclass(frozen=True, slots=True)
class ProductionTreeArtifactLineage:
    """Immutable identity shared by every artifact in one LightGBM final fit."""

    source_sales_batch_id: int
    data_checksum: str
    history_end: date
    cluster_experiment_id: int | None
    cluster_assignment_count: int | None
    cluster_assignment_checksum: str | None
    generator_contract_version: str = GENERATOR_CONTRACT_VERSION

    def __post_init__(self) -> None:
        source_sales_batch_id = _require_positive_integer(
            self.source_sales_batch_id,
            field_name="source_sales_batch_id",
        )
        data_checksum = _require_sha256(self.data_checksum)
        history_end = _require_history_month(self.history_end)
        cluster_experiment_id = (
            None
            if self.cluster_experiment_id is None
            else _require_positive_integer(
                self.cluster_experiment_id,
                field_name="cluster_experiment_id",
            )
        )
        cluster_assignment_count = (
            None
            if self.cluster_assignment_count is None
            else _require_positive_integer(
                self.cluster_assignment_count,
                field_name="cluster_assignment_count",
            )
        )
        cluster_assignment_checksum = (
            None
            if self.cluster_assignment_checksum is None
            else _require_sha256(
                self.cluster_assignment_checksum,
                field_name="cluster_assignment_checksum",
            )
        )
        if cluster_experiment_id is None:
            if cluster_assignment_count is not None or cluster_assignment_checksum is not None:
                raise TreeArtifactLineageError(
                    "global tree lineage cannot declare promoted cluster assignments"
                )
        elif cluster_assignment_count is None or cluster_assignment_checksum is None:
            raise TreeArtifactLineageError(
                "per-cluster tree lineage requires assignment count and checksum"
            )
        if self.generator_contract_version != GENERATOR_CONTRACT_VERSION:
            raise TreeArtifactLineageError(
                "generator contract version does not match the current production "
                f"contract {GENERATOR_CONTRACT_VERSION!r}"
            )

        # Integral subclasses (for example values read from numpy/pandas) are
        # normalized once so equality and serialized metadata remain canonical.
        object.__setattr__(self, "source_sales_batch_id", source_sales_batch_id)
        object.__setattr__(self, "data_checksum", data_checksum)
        object.__setattr__(self, "history_end", history_end)
        object.__setattr__(self, "cluster_experiment_id", cluster_experiment_id)
        object.__setattr__(self, "cluster_assignment_count", cluster_assignment_count)
        object.__setattr__(
            self,
            "cluster_assignment_checksum",
            cluster_assignment_checksum,
        )

    def to_metadata(self) -> LineageMetadata:
        """Serialize to the exact JSON-safe artifact metadata schema."""
        return {
            "source_sales_batch_id": self.source_sales_batch_id,
            "data_checksum": self.data_checksum,
            "history_end": self.history_end.isoformat(),
            "cluster_experiment_id": self.cluster_experiment_id,
            "cluster_assignment_count": self.cluster_assignment_count,
            "cluster_assignment_checksum": self.cluster_assignment_checksum,
            "generator_contract_version": self.generator_contract_version,
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> ProductionTreeArtifactLineage:
        """Parse exact metadata without accepting missing or unknown fields."""
        if not isinstance(metadata, Mapping):
            raise TreeArtifactLineageError("tree artifact lineage metadata must be a mapping")
        if any(not isinstance(key, str) for key in metadata):
            raise TreeArtifactLineageError("tree artifact lineage metadata keys must be strings")

        keys = frozenset(metadata)
        missing = sorted(_LINEAGE_FIELD_SET - keys)
        unexpected = sorted(keys - _LINEAGE_FIELD_SET)
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected fields: {', '.join(unexpected)}")
            raise TreeArtifactLineageError(
                "tree artifact lineage metadata has an invalid schema (" + "; ".join(details) + ")"
            )

        raw_history_end = metadata["history_end"]
        if not isinstance(raw_history_end, str):
            raise TreeArtifactLineageError("history_end must be an ISO month-start date")
        try:
            history_end = date.fromisoformat(raw_history_end)
        except ValueError as exc:
            raise TreeArtifactLineageError("history_end must be an ISO month-start date") from exc
        if history_end.isoformat() != raw_history_end:
            raise TreeArtifactLineageError(
                "history_end must use canonical ISO date format YYYY-MM-DD"
            )

        generator_contract_version = metadata["generator_contract_version"]
        if not isinstance(generator_contract_version, str):
            raise TreeArtifactLineageError("generator contract version must be a string")

        return cls(
            source_sales_batch_id=_require_positive_integer(
                metadata["source_sales_batch_id"],
                field_name="source_sales_batch_id",
            ),
            data_checksum=_require_sha256(metadata["data_checksum"]),
            history_end=history_end,
            cluster_experiment_id=(
                None
                if metadata["cluster_experiment_id"] is None
                else _require_positive_integer(
                    metadata["cluster_experiment_id"],
                    field_name="cluster_experiment_id",
                )
            ),
            cluster_assignment_count=(
                None
                if metadata["cluster_assignment_count"] is None
                else _require_positive_integer(
                    metadata["cluster_assignment_count"],
                    field_name="cluster_assignment_count",
                )
            ),
            cluster_assignment_checksum=(
                None
                if metadata["cluster_assignment_checksum"] is None
                else _require_sha256(
                    metadata["cluster_assignment_checksum"],
                    field_name="cluster_assignment_checksum",
                )
            ),
            generator_contract_version=generator_contract_version,
        )


def _normalize_cluster_label(value: object, *, context: str) -> str:
    if isinstance(value, bool) or value is None:
        raise TreeArtifactLineageError(f"{context} cluster label must be a non-empty string")
    if isinstance(value, Integral):
        return str(int(value))
    if not isinstance(value, str) or not value or value != value.strip():
        raise TreeArtifactLineageError(f"{context} cluster label must be a non-empty string")
    return value


def _normalize_expected_cluster_labels(labels: Collection[object]) -> set[str]:
    if isinstance(labels, (str, bytes)) or not labels:
        raise TreeArtifactLineageError("expected cluster labels must be a non-empty collection")
    normalized: set[str] = set()
    for raw_label in labels:
        label = _normalize_cluster_label(raw_label, context="expected")
        if label in normalized:
            raise TreeArtifactLineageError(
                f"expected cluster labels contain duplicate cluster label {label!r}"
            )
        normalized.add(label)
    return normalized


def _normalize_artifact_map(
    artifacts: Mapping[object, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if not isinstance(artifacts, Mapping):
        raise TreeArtifactLineageError("loaded tree artifacts must be a mapping")
    normalized: dict[str, Mapping[str, object]] = {}
    for raw_label, artifact in artifacts.items():
        label = _normalize_cluster_label(raw_label, context="loaded artifact")
        if label in normalized:
            raise TreeArtifactLineageError(
                f"loaded artifacts contain duplicate cluster label {label!r}"
            )
        if not isinstance(artifact, Mapping):
            raise TreeArtifactLineageError(
                f"loaded artifact for cluster {label!r} must be a mapping; {_RETRAIN_ACTION}"
            )
        normalized[label] = artifact
    return normalized


def _artifact_error(cluster_label: str, detail: str) -> TreeArtifactLineageError:
    return TreeArtifactLineageError(
        f"lgbm_cluster production artifact for cluster {cluster_label!r} {detail}; "
        f"{_RETRAIN_ACTION}"
    )


def _lineage_mismatches(
    actual: ProductionTreeArtifactLineage,
    expected: ProductionTreeArtifactLineage,
) -> list[str]:
    return [
        field_name
        for field_name in _LINEAGE_FIELDS
        if getattr(actual, field_name) != getattr(expected, field_name)
    ]


def validate_tree_artifact_map(
    artifacts: Mapping[object, Mapping[str, object]],
    *,
    expected_cluster_labels: Collection[object],
    expected_lineage: ProductionTreeArtifactLineage,
    expected_cluster_strategy: str = "per_cluster",
    expected_config_checksum: str | None = None,
) -> dict[str, Mapping[str, object]]:
    """Validate exact per-cluster coverage and lineage before any inference.

    The returned map has canonical string labels. Coverage is exact: absent and
    unexpected artifacts are both fatal, and a global/default artifact is never
    treated as a fallback for the promoted cluster population.
    """
    if not isinstance(expected_lineage, ProductionTreeArtifactLineage):
        raise TreeArtifactLineageError("expected_lineage must be a ProductionTreeArtifactLineage")
    if expected_cluster_strategy not in {"per_cluster", "global"}:
        raise TreeArtifactLineageError("expected cluster strategy must be per_cluster or global")
    expected_labels = _normalize_expected_cluster_labels(expected_cluster_labels)
    if expected_cluster_strategy == "global" and expected_labels != {"global"}:
        raise TreeArtifactLineageError(
            "global tree artifact strategy requires exactly the global cluster label"
        )
    if expected_cluster_strategy == "per_cluster" and "global" in expected_labels:
        raise TreeArtifactLineageError(
            "per_cluster tree artifact strategy may not use the global cluster label"
        )
    if expected_config_checksum is not None:
        expected_config_checksum = _require_sha256(
            expected_config_checksum,
            field_name="config_checksum",
        )
    normalized = _normalize_artifact_map(artifacts)

    actual_labels = set(normalized)
    missing = sorted(expected_labels - actual_labels)
    unexpected = sorted(actual_labels - expected_labels)
    if missing or unexpected:
        raise TreeArtifactLineageError(
            "lgbm_cluster production artifact coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}; global/default fallback is "
            f"forbidden; {_RETRAIN_ACTION}"
        )

    for cluster_label in sorted(expected_labels):
        artifact = normalized[cluster_label]
        if artifact.get("model_id") != LIGHTGBM_CLUSTER_MODEL_ID:
            raise _artifact_error(
                cluster_label,
                f"has mismatched model_id {artifact.get('model_id')!r}",
            )
        if str(artifact.get("cluster_label")) != cluster_label:
            raise _artifact_error(cluster_label, "has mismatched cluster_label")
        if artifact.get("training_mode") != "production":
            raise _artifact_error(cluster_label, "is not a production final fit")
        if artifact.get("cluster_strategy") != expected_cluster_strategy:
            raise _artifact_error(
                cluster_label,
                f"must declare {expected_cluster_strategy} training",
            )
        if (
            expected_config_checksum is not None
            and artifact.get("config_checksum") != expected_config_checksum
        ):
            raise _artifact_error(cluster_label, "has a mismatched config_checksum")

        raw_lineage = artifact.get(TREE_ARTIFACT_LINEAGE_KEY)
        if not isinstance(raw_lineage, Mapping):
            raise _artifact_error(cluster_label, "is missing lineage metadata")
        try:
            lineage = ProductionTreeArtifactLineage.from_metadata(raw_lineage)
        except TreeArtifactLineageError as exc:
            raise _artifact_error(
                cluster_label,
                f"has invalid lineage metadata: {exc}",
            ) from exc

        mismatches = _lineage_mismatches(lineage, expected_lineage)
        if mismatches:
            raise _artifact_error(
                cluster_label,
                "has lineage that differs from the generation snapshot for "
                + ", ".join(mismatches),
            )

    return {label: normalized[label] for label in sorted(expected_labels)}
