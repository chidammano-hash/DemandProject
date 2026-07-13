"""Immutable, atomic artifact sets for production LightGBM final fits.

One published version owns the complete configured cluster roster.  The active
pointer is switched only after every pickle, its metadata, and the complete file
manifest have been verified.  Generation validates the manifest before it
deserializes any model, so partial or stale loose files cannot enter inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from common.core.paths import PROJECT_ROOT
from common.core.utils import load_config
from common.ml.backtest_config import (
    BACKTEST_CONFIG_METADATA_KEY,
    build_backtest_config_snapshot,
)
from common.ml.tree_artifact_lineage import (
    ProductionTreeArtifactLineage,
    TreeArtifactLineageError,
    validate_tree_artifact_map,
)

TREE_ARTIFACT_SCHEMA_VERSION = 1
TREE_ARTIFACT_NAMESPACE = "production_tree"
TREE_ARTIFACT_MODEL_ID = "lgbm_cluster"

_VERSIONS_DIRNAME = "versions"
_METADATA_FILENAME = "metadata.json"
_TRAINING_METADATA_FILENAME = "training_metadata.json"
_CHECKSUMS_FILENAME = "checksums.json"
_ACTIVE_FILENAME = "active.json"
_SHA256_HEX_LENGTH = 64

ClusterStrategy = Literal["per_cluster", "global"]


@dataclass(frozen=True, slots=True)
class TreeArtifactSpec:
    """Expected immutable identity for one complete LightGBM final fit."""

    model_id: str
    lineage: ProductionTreeArtifactLineage
    cluster_strategy: ClusterStrategy
    cluster_labels: tuple[str, ...]
    config_checksum: str
    _model_config_json: str

    @property
    def model_config(self) -> dict[str, Any]:
        """Return a detached copy of the canonical training configuration."""
        payload = json.loads(self._model_config_json)
        if not isinstance(payload, dict):  # pragma: no cover - constructor guarantees it
            raise RuntimeError("Tree artifact model configuration is not a mapping")
        return payload


@dataclass(frozen=True, slots=True)
class TreeArtifactSetRef:
    """Filesystem identity and metadata for one immutable cluster set."""

    artifact_set_id: str
    model_id: str
    version_dir: Path
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedTreeArtifactSet:
    """A fully validated all-cluster artifact set ready for inference."""

    artifacts: dict[str, dict[str, Any]]
    ref: TreeArtifactSetRef


class TreeArtifactLineageMismatchError(RuntimeError):
    """The active tree artifact set is valid but stale for the requested spec."""


def get_production_validation_fraction(
    pipeline_config: Mapping[str, Any],
) -> float:
    """Return the required final-fit validation fraction from its canonical path."""
    production = pipeline_config.get("production_forecast")
    if not isinstance(production, Mapping):
        raise ValueError("Forecast configuration is missing production_forecast settings")
    training = production.get("production_training")
    if not isinstance(training, Mapping):
        raise ValueError(
            "Forecast configuration is missing production_forecast.production_training"
        )
    try:
        raw_fraction = training["val_fraction"]
    except KeyError as exc:
        raise ValueError(
            "Forecast configuration requires production_forecast.production_training.val_fraction"
        ) from exc
    if (
        not isinstance(raw_fraction, Real)
        or isinstance(raw_fraction, bool)
        or not math.isfinite(float(raw_fraction))
    ):
        raise ValueError("production_training.val_fraction must be a finite number")
    fraction = float(raw_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("production_training.val_fraction must be between 0 and 1")
    return fraction


def build_tree_model_config_payload(
    pipeline_config: Mapping[str, Any],
    *,
    model_id: str = TREE_ARTIFACT_MODEL_ID,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Capture every configurable input used by the LightGBM final fit.

    The payload deliberately includes the tuning profile document and optional
    tuned-parameter artifact contents, not merely their paths.  A same-name
    file edit therefore invalidates the active artifact at generation time.
    """
    algorithms = pipeline_config.get("algorithms")
    if not isinstance(algorithms, Mapping):
        raise ValueError("Forecast configuration is missing algorithms")
    algorithm = algorithms.get(model_id)
    if not isinstance(algorithm, Mapping):
        raise ValueError(f"Forecast configuration is missing algorithm {model_id}")
    params = algorithm.get("params")
    if not isinstance(params, Mapping):
        raise ValueError(f"Forecast configuration is missing parameters for {model_id}")

    clustering = pipeline_config.get("clustering")
    backtest = pipeline_config.get("backtest")
    production = pipeline_config.get("production_forecast")
    if not isinstance(clustering, Mapping):
        raise ValueError("Forecast configuration is missing clustering settings")
    if not isinstance(backtest, Mapping):
        raise ValueError("Forecast configuration is missing backtest settings")
    if not isinstance(production, Mapping):
        raise ValueError("Forecast configuration is missing production_forecast settings")

    params_file = params.get("params_file")
    tuned_params_artifact: dict[str, Any] | None = None
    if params_file:
        if not isinstance(params_file, str):
            raise ValueError("LightGBM params_file must be a path string")
        params_path = Path(params_file)
        if not params_path.is_absolute():
            params_path = Path(project_root) / params_path
        try:
            tuned_payload = json.loads(params_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"LightGBM tuned params artifact is missing: {params_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"LightGBM tuned params artifact is unreadable: {params_path}"
            ) from exc
        if not isinstance(tuned_payload, dict):
            raise ValueError("LightGBM tuned params artifact must be a JSON object")
        tuned_params_artifact = tuned_payload

    clustering_enabled = clustering.get("enabled")
    if not isinstance(clustering_enabled, bool):
        raise ValueError("clustering.enabled must be explicitly true or false")
    validation_fraction = get_production_validation_fraction(pipeline_config)
    return {
        "algorithm": dict(algorithm),
        "effective_cluster_strategy": ("per_cluster" if clustering_enabled else "global"),
        "clustering": {"enabled": clustering_enabled},
        "backtest_demand_routing": {
            "intermittent_threshold": backtest.get("intermittent_threshold"),
            "lumpy_threshold": backtest.get("lumpy_threshold"),
        },
        "recursive_training": {
            "enabled": params.get("recursive"),
            "noise_enabled": backtest.get("recursive_noise_enabled"),
            "noise_pct": backtest.get("recursive_noise_pct"),
            "lag_smooth": backtest.get("recursive_lag_smooth"),
        },
        "feature_selection": {
            "enabled": params.get("shap_select"),
            "cumulative_threshold": params.get("shap_threshold"),
            "top_n": params.get("shap_top_n"),
            "sample_size": params.get("shap_sample_size"),
            "min_features": backtest.get("shap_min_features"),
            "retrain_threshold": backtest.get("shap_retrain_threshold"),
            "correlation_filter": params.get("correlation_filter"),
            "correlation_threshold": params.get("correlation_threshold"),
            "variance_filter": params.get("variance_filter"),
            "variance_threshold": params.get("variance_threshold"),
        },
        "feature_history": {"lookback_months": production.get("lookback_months")},
        "production_training": {"val_fraction": validation_fraction},
        "cluster_tuning_profiles": load_config("cluster_tuning_profiles.yaml"),
        "tuned_params_artifact": tuned_params_artifact,
    }


def build_production_tree_model_config_payload(
    pipeline_config: Mapping[str, Any],
    *,
    model_id: str = TREE_ARTIFACT_MODEL_ID,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build the one tree lineage contract shared by training and inference."""
    payload = build_tree_model_config_payload(
        pipeline_config,
        model_id=model_id,
        project_root=project_root,
    )
    snapshot = build_backtest_config_snapshot(
        pipeline_config,
        model_id,
        cluster_tuning_profiles=payload["cluster_tuning_profiles"],
    )
    return {
        **payload,
        BACKTEST_CONFIG_METADATA_KEY: {model_id: snapshot.as_metadata()},
    }


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Tree artifact configuration cannot contain non-finite floats")
        return value
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("Tree artifact configuration cannot contain non-finite floats")
        return normalized
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Tree artifact configuration keys must be strings")
            result[key] = _canonicalize(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise ValueError(
        f"Tree artifact configuration contains unsupported value type: {type(value).__name__}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")
    checksum = value.strip().lower()
    if len(checksum) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return checksum


def _normalize_cluster_labels(labels: Collection[object]) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)) or not labels:
        raise ValueError("Tree artifact cluster labels must be a non-empty collection")
    normalized: set[str] = set()
    for raw_label in labels:
        if isinstance(raw_label, bool) or raw_label is None:
            raise ValueError("Tree artifact cluster labels must be non-empty strings")
        label = str(raw_label)
        if not label or label != label.strip():
            raise ValueError("Tree artifact cluster labels must be non-empty strings")
        if label in normalized:
            raise ValueError(f"Duplicate tree artifact cluster label: {label!r}")
        normalized.add(label)
    return tuple(sorted(normalized))


def build_tree_artifact_spec(
    *,
    model_id: str,
    model_config: Mapping[str, Any],
    lineage: ProductionTreeArtifactLineage,
    cluster_strategy: str,
    cluster_labels: Collection[object],
) -> TreeArtifactSpec:
    """Build the strict expected identity for an all-cluster artifact set."""
    if model_id != TREE_ARTIFACT_MODEL_ID:
        raise ValueError(f"Unsupported production tree artifact model: {model_id}")
    if not isinstance(lineage, ProductionTreeArtifactLineage):
        raise ValueError("Tree artifact lineage must be ProductionTreeArtifactLineage")
    if cluster_strategy not in {"per_cluster", "global"}:
        raise ValueError("Tree artifact cluster strategy must be per_cluster or global")

    labels = _normalize_cluster_labels(cluster_labels)
    if cluster_strategy == "global":
        if labels != ("global",):
            raise ValueError(
                "Tree artifact cluster strategy global requires exactly label 'global'"
            )
    elif "global" in labels or lineage.cluster_experiment_id is None:
        raise ValueError(
            "Tree artifact cluster strategy per_cluster requires a promoted cluster "
            "experiment and may not use label 'global'"
        )

    normalized_config = _canonicalize(model_config)
    if not isinstance(normalized_config, dict) or not normalized_config:
        raise ValueError("Tree artifact model configuration must be a non-empty mapping")
    config_json = json.dumps(
        normalized_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return TreeArtifactSpec(
        model_id=model_id,
        lineage=lineage,
        cluster_strategy=cluster_strategy,
        cluster_labels=labels,
        config_checksum=_sha256_bytes(config_json.encode("utf-8")),
        _model_config_json=config_json,
    )


def _artifact_root(base_dir: Path, model_id: str) -> Path:
    return Path(base_dir) / model_id / TREE_ARTIFACT_NAMESPACE


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = _canonical_json_bytes(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _write_pickle(path: Path, payload: object) -> None:
    with path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid4().hex}.tmp"
    try:
        _write_json(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"Tree artifact {label} may not be a symbolic link")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Tree artifact {label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Tree artifact {label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Tree artifact {label} must be a JSON object")
    return payload


def _write_checksum_manifest(version_dir: Path) -> str:
    manifest_path = version_dir / _CHECKSUMS_FILENAME
    files = {
        path.relative_to(version_dir).as_posix(): _sha256_file(path)
        for path in sorted(version_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    if not files:
        raise RuntimeError("Tree artifact set contains no files to checksum")
    _write_json(
        manifest_path,
        {
            "artifact_schema_version": TREE_ARTIFACT_SCHEMA_VERSION,
            "files": files,
        },
    )
    return _sha256_file(manifest_path)


def _validate_file_manifest(
    version_dir: Path,
    *,
    expected_manifest_checksum: str,
) -> dict[str, str]:
    manifest_path = version_dir / _CHECKSUMS_FILENAME
    try:
        actual_manifest_checksum = _sha256_file(manifest_path)
    except FileNotFoundError as exc:
        raise RuntimeError("Tree artifact checksum manifest is missing") from exc
    expected_checksum = _normalize_sha256(
        expected_manifest_checksum,
        field_name="checksums_sha256",
    )
    if actual_manifest_checksum != expected_checksum:
        raise RuntimeError("Tree artifact manifest checksum does not match active pointer")

    manifest = _read_json(manifest_path, label="checksum manifest")
    if manifest.get("artifact_schema_version") != TREE_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Tree artifact checksum manifest has an unsupported schema")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise RuntimeError("Tree artifact checksum manifest has no file entries")

    actual_files: set[str] = set()
    for path in version_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Tree artifact set may not contain symbolic links")
        if path.is_file() and path != manifest_path:
            actual_files.add(path.relative_to(version_dir).as_posix())
    if set(raw_files) != actual_files:
        raise RuntimeError("Tree artifact file roster does not match its checksum manifest")

    files: dict[str, str] = {}
    for relative, raw_checksum in raw_files.items():
        if not isinstance(relative, str):
            raise RuntimeError("Tree artifact checksum paths must be strings")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("Tree artifact checksum manifest contains an unsafe path")
        try:
            checksum = _normalize_sha256(
                raw_checksum,
                field_name=f"checksum for {relative}",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        file_path = version_dir.joinpath(*relative_path.parts)
        if not file_path.is_file() or file_path.is_symlink():
            raise RuntimeError(f"Tree artifact file is missing or unsafe: {relative}")
        if _sha256_file(file_path) != checksum:
            raise RuntimeError(f"Tree artifact file checksum mismatch: {relative}")
        files[relative] = checksum
    return files


def _validate_artifact_set_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError("Tree artifact set ID is invalid")
    return value


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    expected_spec: TreeArtifactSpec,
    artifact_set_id: str,
    manifest_files: Mapping[str, str],
) -> dict[str, str]:
    expected_fields: dict[str, Any] = {
        "artifact_schema_version": TREE_ARTIFACT_SCHEMA_VERSION,
        "artifact_set_id": artifact_set_id,
        "model_id": expected_spec.model_id,
        "model_format": "pickle",
        "cluster_strategy": expected_spec.cluster_strategy,
        "cluster_labels": list(expected_spec.cluster_labels),
        "lineage": expected_spec.lineage.to_metadata(),
        "model_config": expected_spec.model_config,
        "config_checksum": expected_spec.config_checksum,
    }
    for field, expected_value in expected_fields.items():
        if metadata.get(field) != expected_value:
            raise RuntimeError(
                f"Tree artifact metadata field {field!r} does not match expected lineage"
            )

    trained_at = metadata.get("trained_at")
    if not isinstance(trained_at, str):
        raise RuntimeError("Tree artifact trained_at metadata is missing")
    try:
        parsed_trained_at = datetime.fromisoformat(trained_at)
    except ValueError as exc:
        raise RuntimeError("Tree artifact trained_at metadata is invalid") from exc
    if parsed_trained_at.tzinfo is None:
        raise RuntimeError("Tree artifact trained_at metadata must be timezone-aware")

    raw_model_files = metadata.get("model_files")
    raw_model_checksums = metadata.get("model_checksums")
    expected_labels = set(expected_spec.cluster_labels)
    if not isinstance(raw_model_files, dict) or set(raw_model_files) != expected_labels:
        raise RuntimeError("Tree artifact model file roster does not match cluster labels")
    if not isinstance(raw_model_checksums, dict) or set(raw_model_checksums) != expected_labels:
        raise RuntimeError("Tree artifact model checksum roster does not match cluster labels")

    model_files: dict[str, str] = {}
    seen_paths: set[str] = set()
    for label in expected_spec.cluster_labels:
        relative = raw_model_files[label]
        if not isinstance(relative, str):
            raise RuntimeError("Tree artifact model paths must be strings")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix != ".pkl"
            or relative in seen_paths
        ):
            raise RuntimeError("Tree artifact metadata contains an unsafe model path")
        seen_paths.add(relative)
        expected_checksum = raw_model_checksums[label]
        if not isinstance(expected_checksum, str):
            raise RuntimeError("Tree artifact model checksums must be strings")
        if manifest_files.get(relative) != expected_checksum:
            raise RuntimeError("Tree artifact model checksum differs from file manifest")
        model_files[label] = relative
    manifest_model_files = {
        relative for relative in manifest_files if PurePosixPath(relative).suffix == ".pkl"
    }
    if manifest_model_files != seen_paths:
        raise RuntimeError("Tree artifact pickle roster contains an unregistered cluster model")
    return model_files


def _load_version(
    version_dir: Path,
    *,
    expected_spec: TreeArtifactSpec,
    artifact_set_id: str,
    expected_manifest_checksum: str,
) -> LoadedTreeArtifactSet:
    if not version_dir.is_dir() or version_dir.is_symlink():
        raise RuntimeError(f"Tree artifact version is missing or unsafe: {version_dir}")
    manifest_files = _validate_file_manifest(
        version_dir,
        expected_manifest_checksum=expected_manifest_checksum,
    )
    metadata = _read_json(version_dir / _METADATA_FILENAME, label="metadata")
    model_files = _validate_metadata(
        metadata,
        expected_spec=expected_spec,
        artifact_set_id=artifact_set_id,
        manifest_files=manifest_files,
    )
    training_metadata = _read_json(
        version_dir / _TRAINING_METADATA_FILENAME,
        label="training metadata",
    )

    artifacts: dict[str, dict[str, Any]] = {}
    for label in expected_spec.cluster_labels:
        model_path = version_dir.joinpath(*PurePosixPath(model_files[label]).parts)
        try:
            with model_path.open("rb") as handle:
                artifact = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError, ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"Tree artifact for cluster {label!r} could not be deserialized"
            ) from exc
        if not isinstance(artifact, dict):
            raise RuntimeError(f"Tree artifact for cluster {label!r} must be a mapping")
        artifacts[label] = artifact

    try:
        validated = validate_tree_artifact_map(
            artifacts,
            expected_cluster_labels=expected_spec.cluster_labels,
            expected_lineage=expected_spec.lineage,
            expected_cluster_strategy=expected_spec.cluster_strategy,
            expected_config_checksum=expected_spec.config_checksum,
        )
    except TreeArtifactLineageError as exc:
        raise RuntimeError(str(exc)) from exc
    loaded: dict[str, dict[str, Any]] = {}
    for label, raw_artifact in validated.items():
        artifact = dict(raw_artifact)
        if not callable(getattr(artifact.get("model"), "predict", None)):
            raise RuntimeError(
                f"Tree artifact for cluster {label!r} has no callable predict method"
            )
        loaded[label] = artifact

    ref = TreeArtifactSetRef(
        artifact_set_id=artifact_set_id,
        model_id=expected_spec.model_id,
        version_dir=version_dir,
        metadata={**metadata, "training_metadata": training_metadata},
    )
    return LoadedTreeArtifactSet(artifacts=loaded, ref=ref)


def publish_tree_artifact_set(
    *,
    artifacts: Mapping[object, Mapping[str, Any]],
    training_metadata: Mapping[str, Any],
    spec: TreeArtifactSpec,
    base_dir: Path,
    trained_at: datetime | None = None,
) -> LoadedTreeArtifactSet:
    """Publish one complete version, then atomically make it active."""
    if not isinstance(spec, TreeArtifactSpec):
        raise ValueError("spec must be a TreeArtifactSpec")
    try:
        validated = validate_tree_artifact_map(
            artifacts,
            expected_cluster_labels=spec.cluster_labels,
            expected_lineage=spec.lineage,
            expected_cluster_strategy=spec.cluster_strategy,
            expected_config_checksum=spec.config_checksum,
        )
    except TreeArtifactLineageError as exc:
        raise RuntimeError(str(exc)) from exc
    normalized_training_metadata = _canonicalize(training_metadata)
    if not isinstance(normalized_training_metadata, dict):
        raise ValueError("Tree artifact training metadata must be a mapping")
    timestamp = trained_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("trained_at must be timezone-aware")

    root = _artifact_root(Path(base_dir), spec.model_id)
    versions_dir = root / _VERSIONS_DIRNAME
    versions_dir.mkdir(parents=True, exist_ok=True)
    artifact_set_id = uuid4().hex
    version_dir = versions_dir / artifact_set_id
    temporary_dir = versions_dir / f".{artifact_set_id}.{uuid4().hex}.building"
    temporary_dir.mkdir()
    try:
        models_dir = temporary_dir / "models"
        models_dir.mkdir()
        model_files: dict[str, str] = {}
        model_checksums: dict[str, str] = {}
        for index, label in enumerate(spec.cluster_labels):
            relative = f"models/{index:04d}.pkl"
            model_path = temporary_dir / relative
            _write_pickle(model_path, dict(validated[label]))
            model_files[label] = relative
            model_checksums[label] = _sha256_file(model_path)

        metadata = {
            "artifact_schema_version": TREE_ARTIFACT_SCHEMA_VERSION,
            "artifact_set_id": artifact_set_id,
            "model_id": spec.model_id,
            "model_format": "pickle",
            "cluster_strategy": spec.cluster_strategy,
            "cluster_labels": list(spec.cluster_labels),
            "lineage": spec.lineage.to_metadata(),
            "model_config": spec.model_config,
            "config_checksum": spec.config_checksum,
            "model_files": model_files,
            "model_checksums": model_checksums,
            "trained_at": timestamp.astimezone(UTC).isoformat(),
        }
        _write_json(temporary_dir / _METADATA_FILENAME, metadata)
        _write_json(
            temporary_dir / _TRAINING_METADATA_FILENAME,
            normalized_training_metadata,
        )
        manifest_checksum = _write_checksum_manifest(temporary_dir)
        built = _load_version(
            temporary_dir,
            expected_spec=spec,
            artifact_set_id=artifact_set_id,
            expected_manifest_checksum=manifest_checksum,
        )

        os.rename(temporary_dir, version_dir)
        _fsync_directory(versions_dir)
        pointer = {
            "artifact_schema_version": TREE_ARTIFACT_SCHEMA_VERSION,
            "artifact_set_id": artifact_set_id,
            "checksums_sha256": manifest_checksum,
        }
        _atomic_write_json(root / _ACTIVE_FILENAME, pointer)
        return LoadedTreeArtifactSet(
            artifacts=built.artifacts,
            ref=TreeArtifactSetRef(
                artifact_set_id=artifact_set_id,
                model_id=spec.model_id,
                version_dir=version_dir,
                metadata=built.ref.metadata,
            ),
        )
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)


def read_active_tree_artifact_ref(
    *,
    model_id: str,
    base_dir: Path,
    expected_spec: TreeArtifactSpec | None = None,
    expected_model_config: Mapping[str, Any] | None = None,
) -> TreeArtifactSetRef:
    """Validate the active set without deserializing LightGBM models."""
    if expected_spec is not None and expected_model_config is not None:
        raise ValueError("Pass expected_spec or expected_model_config, not both")
    root = _artifact_root(Path(base_dir), model_id)
    active_path = root / _ACTIVE_FILENAME
    if not active_path.exists():
        raise FileNotFoundError(active_path)
    pointer = _read_json(active_path, label="active pointer")
    if pointer.get("artifact_schema_version") != TREE_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Tree artifact active pointer has an unsupported schema")
    artifact_set_id = _validate_artifact_set_id(pointer.get("artifact_set_id"))
    manifest_checksum = pointer.get("checksums_sha256")
    if not isinstance(manifest_checksum, str):
        raise RuntimeError("Tree artifact active pointer lacks a manifest checksum")

    version_dir = root / _VERSIONS_DIRNAME / artifact_set_id
    if not version_dir.is_dir() or version_dir.is_symlink():
        raise RuntimeError(f"Tree artifact version is missing or unsafe: {version_dir}")
    manifest_files = _validate_file_manifest(
        version_dir,
        expected_manifest_checksum=manifest_checksum,
    )
    metadata = _read_json(version_dir / _METADATA_FILENAME, label="metadata")
    try:
        raw_lineage = metadata["lineage"]
        if not isinstance(raw_lineage, Mapping):
            raise ValueError("lineage must be a mapping")
        lineage = ProductionTreeArtifactLineage.from_metadata(raw_lineage)
        raw_strategy = metadata["cluster_strategy"]
        if not isinstance(raw_strategy, str):
            raise ValueError("cluster_strategy must be a string")
        raw_labels = metadata["cluster_labels"]
        if not isinstance(raw_labels, list):
            raise ValueError("cluster_labels must be a list")
        raw_config = metadata["model_config"]
        if not isinstance(raw_config, Mapping):
            raise ValueError("model_config must be a mapping")
        recorded_spec = build_tree_artifact_spec(
            model_id=model_id,
            model_config=raw_config,
            lineage=lineage,
            cluster_strategy=raw_strategy,
            cluster_labels=raw_labels,
        )
    except (KeyError, TypeError, ValueError, TreeArtifactLineageError) as exc:
        raise RuntimeError("Tree artifact metadata has invalid lineage fields") from exc

    _validate_metadata(
        metadata,
        expected_spec=recorded_spec,
        artifact_set_id=artifact_set_id,
        manifest_files=manifest_files,
    )
    current_spec = expected_spec
    if current_spec is None and expected_model_config is not None:
        current_spec = build_tree_artifact_spec(
            model_id=model_id,
            model_config=expected_model_config,
            lineage=recorded_spec.lineage,
            cluster_strategy=recorded_spec.cluster_strategy,
            cluster_labels=recorded_spec.cluster_labels,
        )
    if current_spec is not None and current_spec != recorded_spec:
        raise TreeArtifactLineageMismatchError(
            "Active tree artifact metadata lineage is stale for the current "
            "configuration, completed sales batch, latest closed month, or promoted clustering"
        )
    training_metadata = _read_json(
        version_dir / _TRAINING_METADATA_FILENAME,
        label="training metadata",
    )
    return TreeArtifactSetRef(
        artifact_set_id=artifact_set_id,
        model_id=model_id,
        version_dir=version_dir,
        metadata={**metadata, "training_metadata": training_metadata},
    )


def load_active_tree_artifact_set(
    *,
    model_id: str,
    expected_spec: TreeArtifactSpec,
    base_dir: Path,
) -> LoadedTreeArtifactSet:
    """Load the active all-cluster set only after full lineage validation."""
    if model_id != expected_spec.model_id:
        raise ValueError("Tree artifact model_id does not match expected spec")
    root = _artifact_root(Path(base_dir), model_id)
    active_path = root / _ACTIVE_FILENAME
    if not active_path.exists():
        raise FileNotFoundError(active_path)
    pointer = _read_json(active_path, label="active pointer")
    if pointer.get("artifact_schema_version") != TREE_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Tree artifact active pointer has an unsupported schema")
    artifact_set_id = _validate_artifact_set_id(pointer.get("artifact_set_id"))
    manifest_checksum = pointer.get("checksums_sha256")
    if not isinstance(manifest_checksum, str):
        raise RuntimeError("Tree artifact active pointer lacks a manifest checksum")
    return _load_version(
        root / _VERSIONS_DIRNAME / artifact_set_id,
        expected_spec=expected_spec,
        artifact_set_id=artifact_set_id,
        expected_manifest_checksum=manifest_checksum,
    )
