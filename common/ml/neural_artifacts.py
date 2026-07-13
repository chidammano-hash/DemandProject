"""Immutable, lineage-validated artifacts for global neural forecast models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from psycopg import sql

from common.ml.neural_forecast import (
    NEURAL_TRAINING_CONTRACT_VERSION,
    SUPPORTED_NEURAL_MODELS,
    FittedNeuralModel,
    NeuralCohortIdentity,
    compute_neural_cohort_identity,
    current_neural_runtime_contract,
)
from common.services.forecast_generation import GENERATOR_CONTRACT_VERSION

ARTIFACT_SCHEMA_VERSION = 2
_ARTIFACT_NAMESPACE = "neuralforecast"
_VERSIONS_DIRNAME = "versions"
_MODEL_DIRNAME = "model"
_METADATA_FILENAME = "metadata.json"
_CHECKSUMS_FILENAME = "checksums.json"
_ACTIVE_FILENAME = "active.json"
_SHA256_HEX_LENGTH = 64

ModelLoader = Callable[[Path], Any]


@dataclass(frozen=True)
class NeuralArtifactRef:
    """Stable filesystem reference and immutable metadata for one model version."""

    artifact_id: str
    model_id: str
    version_dir: Path
    model_dir: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedNeuralArtifact:
    """A validated runtime model paired with its immutable artifact reference."""

    fitted_model: FittedNeuralModel
    ref: NeuralArtifactRef


class NeuralArtifactLineageMismatchError(RuntimeError):
    """The active neural artifact is valid but stale for the requested lineage."""


@dataclass(frozen=True)
class _ExpectedArtifact:
    artifact_id: str
    model_id: str
    params: dict[str, Any]
    config_checksum: str
    fitted_horizon: int
    min_history: int
    source_sales_batch_id: int
    data_checksum: str
    history_end: str
    generator_contract_version: str
    training_dfu_count: int
    training_row_count: int
    training_cohort_checksum: str
    training_data_checksum: str
    training_contract_version: str
    runtime_contract: dict[str, str]
    runtime_contract_checksum: str


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Artifact lineage cannot contain non-finite floats")
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Artifact lineage mapping keys must be strings")
            result[key] = _canonicalize(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise ValueError(f"Artifact lineage contains unsupported value type: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _canonicalize(value)
    return json.dumps(
        normalized,
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


def _normalize_checksum(value: str, *, field_name: str) -> str:
    checksum = str(value).strip().lower()
    if len(checksum) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return checksum


def _normalize_history_end(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        normalized = value.date()
    elif isinstance(value, date):
        normalized = value
    elif isinstance(value, str):
        try:
            normalized = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("history_end must be an ISO date") from exc
    else:
        raise ValueError("history_end must be a date, datetime, or ISO date string")
    if normalized.day != 1:
        raise ValueError("history_end must be normalized to the first day of its month")
    return normalized.isoformat()


def _normalize_positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _normalize_runtime_contract(value: Mapping[str, Any]) -> dict[str, str]:
    normalized = _canonicalize(value)
    required = {"neuralforecast", "numpy", "pandas", "python"}
    if not isinstance(normalized, dict) or set(normalized) != required:
        raise ValueError(f"runtime_contract must contain exactly {sorted(required)}")
    for key, item in normalized.items():
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"runtime_contract {key!r} must be a non-empty string")
        normalized[key] = item.strip()
    return normalized


def _subtract_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    return date(month_index // 12, month_index % 12 + 1, 1)


def load_neural_training_cohort_identity(
    conn: Any,
    *,
    sales_table: str,
    history_end: date | datetime | str,
    min_history: int,
    fetch_size: int = 10_000,
) -> NeuralCohortIdentity:
    """Hash the current DB cohort matching calendar-complete neural eligibility.

    Only the sorted ``sku_ck`` roster is streamed. This makes training,
    generation, and readiness able to prove cohort drift without loading sales
    frames or deserializing model weights.
    """
    normalized_end = date.fromisoformat(_normalize_history_end(history_end))
    normalized_min_history = _normalize_positive_int(min_history, field_name="min_history")
    normalized_fetch_size = _normalize_positive_int(fetch_size, field_name="fetch_size")
    first_month_cutoff = _subtract_months(normalized_end, normalized_min_history - 1)
    query = sql.SQL(
        """SELECT d.sku_ck
           FROM {} sales
           INNER JOIN dim_sku d
                   ON d.item_id = sales.item_id
                  AND d.customer_group = sales.customer_group
                  AND d.loc = sales.loc
           WHERE sales.qty IS NOT NULL
             AND sales.type = 1
             AND sales.startdate <= %s
           GROUP BY d.sku_ck
           HAVING MIN(sales.startdate) <= %s
           ORDER BY d.sku_ck COLLATE "C"
        """
    ).format(sql.Identifier(sales_table))

    with conn.cursor(name="neural_training_cohort") as cursor:
        cursor.execute(query, (normalized_end, first_month_cutoff))

        def _sku_cks() -> Iterator[object]:
            while True:
                rows = cursor.fetchmany(normalized_fetch_size)
                if not rows:
                    return
                for row in rows:
                    yield row[0]

        return compute_neural_cohort_identity(_sku_cks(), presorted=True)


def _build_expected_artifact(
    *,
    model_id: str,
    params: Mapping[str, Any],
    source_sales_batch_id: int,
    data_checksum: str,
    history_end: date | datetime | str,
    generator_contract_version: str,
    training_dfu_count: int,
    training_row_count: int,
    training_cohort_checksum: str,
    training_data_checksum: str,
    training_contract_version: str,
    runtime_contract: Mapping[str, Any],
) -> _ExpectedArtifact:
    if model_id not in SUPPORTED_NEURAL_MODELS:
        raise ValueError(f"Unsupported neural artifact model: {model_id}")
    if (
        not isinstance(source_sales_batch_id, int)
        or isinstance(source_sales_batch_id, bool)
        or source_sales_batch_id <= 0
    ):
        raise ValueError("source_sales_batch_id must be a positive integer")
    normalized_params = _canonicalize(params)
    if not isinstance(normalized_params, dict):
        raise ValueError("Neural artifact params must be a mapping")
    try:
        fitted_horizon = normalized_params["h"]
        min_history = normalized_params["min_history"]
    except KeyError as exc:
        raise ValueError("Neural artifact params require integer h and min_history") from exc
    if (
        not isinstance(fitted_horizon, int)
        or isinstance(fitted_horizon, bool)
        or not isinstance(min_history, int)
        or isinstance(min_history, bool)
    ):
        raise ValueError("Neural artifact params require integer h and min_history")
    if fitted_horizon <= 0 or min_history <= 0:
        raise ValueError("Neural artifact h and min_history must be positive")

    normalized_data_checksum = _normalize_checksum(
        data_checksum,
        field_name="data_checksum",
    )
    normalized_history_end = _normalize_history_end(history_end)
    contract = str(generator_contract_version).strip()
    if not contract:
        raise ValueError("generator_contract_version must not be blank")
    config_checksum = _sha256_bytes(_canonical_json_bytes(normalized_params))
    normalized_training_dfu_count = _normalize_positive_int(
        training_dfu_count,
        field_name="training_dfu_count",
    )
    normalized_training_row_count = _normalize_positive_int(
        training_row_count,
        field_name="training_row_count",
    )
    if normalized_training_row_count < normalized_training_dfu_count:
        raise ValueError("training_row_count cannot be less than training_dfu_count")
    normalized_cohort_checksum = _normalize_checksum(
        training_cohort_checksum,
        field_name="training_cohort_checksum",
    )
    normalized_training_data_checksum = _normalize_checksum(
        training_data_checksum,
        field_name="training_data_checksum",
    )
    normalized_training_contract = str(training_contract_version).strip()
    if normalized_training_contract != NEURAL_TRAINING_CONTRACT_VERSION:
        raise ValueError(
            "training_contract_version does not match the current neural training contract"
        )
    normalized_runtime_contract = _normalize_runtime_contract(runtime_contract)
    runtime_contract_checksum = _sha256_bytes(
        _canonical_json_bytes(normalized_runtime_contract)
    )
    lineage = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_id": model_id,
        "params": normalized_params,
        "config_checksum": config_checksum,
        "source_sales_batch_id": source_sales_batch_id,
        "data_checksum": normalized_data_checksum,
        "history_end": normalized_history_end,
        "generator_contract_version": contract,
        "training_dfu_count": normalized_training_dfu_count,
        "training_row_count": normalized_training_row_count,
        "training_cohort_checksum": normalized_cohort_checksum,
        "training_data_checksum": normalized_training_data_checksum,
        "training_contract_version": normalized_training_contract,
        "runtime_contract": normalized_runtime_contract,
        "runtime_contract_checksum": runtime_contract_checksum,
    }
    artifact_id = _sha256_bytes(_canonical_json_bytes(lineage))
    return _ExpectedArtifact(
        artifact_id=artifact_id,
        model_id=model_id,
        params=normalized_params,
        config_checksum=config_checksum,
        fitted_horizon=fitted_horizon,
        min_history=min_history,
        source_sales_batch_id=source_sales_batch_id,
        data_checksum=normalized_data_checksum,
        history_end=normalized_history_end,
        generator_contract_version=contract,
        training_dfu_count=normalized_training_dfu_count,
        training_row_count=normalized_training_row_count,
        training_cohort_checksum=normalized_cohort_checksum,
        training_data_checksum=normalized_training_data_checksum,
        training_contract_version=normalized_training_contract,
        runtime_contract=normalized_runtime_contract,
        runtime_contract_checksum=runtime_contract_checksum,
    )


def build_neural_artifact_id(
    *,
    model_id: str,
    params: Mapping[str, Any],
    source_sales_batch_id: int,
    data_checksum: str,
    history_end: date | datetime | str,
    generator_contract_version: str = GENERATOR_CONTRACT_VERSION,
    training_dfu_count: int,
    training_row_count: int,
    training_cohort_checksum: str,
    training_data_checksum: str,
    training_contract_version: str = NEURAL_TRAINING_CONTRACT_VERSION,
    runtime_contract: Mapping[str, Any],
) -> str:
    """Build the deterministic ID for one exact model/config/data lineage."""
    return _build_expected_artifact(
        model_id=model_id,
        params=params,
        source_sales_batch_id=source_sales_batch_id,
        data_checksum=data_checksum,
        history_end=history_end,
        generator_contract_version=generator_contract_version,
        training_dfu_count=training_dfu_count,
        training_row_count=training_row_count,
        training_cohort_checksum=training_cohort_checksum,
        training_data_checksum=training_data_checksum,
        training_contract_version=training_contract_version,
        runtime_contract=runtime_contract,
    ).artifact_id


def _artifact_root(base_dir: Path, model_id: str) -> Path:
    return Path(base_dir) / model_id / _ARTIFACT_NAMESPACE


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = _canonical_json_bytes(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(serialized)
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
        raise RuntimeError(f"Neural artifact {label} may not be a symbolic link")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Neural artifact {label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Neural artifact {label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Neural artifact {label} must be a JSON object")
    return payload


def _default_loader(model_dir: Path) -> Any:
    from neuralforecast import NeuralForecast

    return NeuralForecast.load(str(model_dir))


def _runtime_model_identity(runtime: Any) -> str:
    explicit = getattr(runtime, "model_id", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()

    models = getattr(runtime, "models", None)
    if not isinstance(models, (list, tuple)) or len(models) != 1:
        raise RuntimeError("Loaded neural artifact must contain exactly one model")
    model = models[0]
    class_identity = model.__class__.__name__.lower()
    if class_identity in SUPPORTED_NEURAL_MODELS:
        return class_identity
    alias = getattr(model, "alias", None)
    if isinstance(alias, str) and alias.strip().lower() in SUPPORTED_NEURAL_MODELS:
        return alias.strip().lower()
    raise RuntimeError("Loaded neural artifact has an unsupported model identity")


def _validate_runtime_model(runtime: Any, expected: _ExpectedArtifact) -> None:
    identity = _runtime_model_identity(runtime)
    if identity != expected.model_id:
        raise RuntimeError(
            f"Loaded neural artifact model identity {identity!r} does not match "
            f"{expected.model_id!r}"
        )
    runtime_horizon = getattr(runtime, "h", None)
    if (
        not isinstance(runtime_horizon, int)
        or isinstance(runtime_horizon, bool)
        or runtime_horizon != expected.fitted_horizon
    ):
        raise RuntimeError(
            "Loaded neural artifact horizon does not match the configured fitted horizon"
        )


def _metadata_for_publish(
    *,
    expected: _ExpectedArtifact,
    fitted: FittedNeuralModel,
    trained_at: datetime,
) -> dict[str, Any]:
    if trained_at.tzinfo is None:
        raise ValueError("trained_at must be timezone-aware")
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": expected.artifact_id,
        "model_id": expected.model_id,
        "model_format": "neuralforecast",
        "dataset_embedded": False,
        "fitted_horizon": fitted.fitted_horizon,
        "min_history": fitted.min_history,
        "training_dfu_count": fitted.training_dfu_count,
        "training_row_count": fitted.training_row_count,
        "training_cohort_checksum": expected.training_cohort_checksum,
        "training_data_checksum": expected.training_data_checksum,
        "training_contract_version": expected.training_contract_version,
        "runtime_contract": expected.runtime_contract,
        "runtime_contract_checksum": expected.runtime_contract_checksum,
        "params": expected.params,
        "config_checksum": expected.config_checksum,
        "source_sales_batch_id": expected.source_sales_batch_id,
        "data_checksum": expected.data_checksum,
        "history_end": expected.history_end,
        "generator_contract_version": expected.generator_contract_version,
        "trained_at": trained_at.astimezone(UTC).isoformat(),
    }


def _write_checksum_manifest(version_dir: Path) -> str:
    manifest_path = version_dir / _CHECKSUMS_FILENAME
    files = {
        path.relative_to(version_dir).as_posix(): _sha256_file(path)
        for path in sorted(version_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    if not files:
        raise RuntimeError("Neural artifact contains no files to checksum")
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "files": files,
    }
    _write_json(manifest_path, manifest)
    return _sha256_file(manifest_path)


def _validate_file_manifest(
    version_dir: Path,
    *,
    expected_manifest_checksum: str | None,
) -> str:
    manifest_path = version_dir / _CHECKSUMS_FILENAME
    manifest_checksum = _sha256_file(manifest_path) if manifest_path.exists() else ""
    if expected_manifest_checksum is not None:
        expected_checksum = _normalize_checksum(
            expected_manifest_checksum,
            field_name="checksums_sha256",
        )
        if manifest_checksum != expected_checksum:
            raise RuntimeError("Neural artifact manifest checksum does not match active pointer")

    manifest = _read_json(manifest_path, label="checksum manifest")
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Neural artifact checksum manifest has an unsupported schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Neural artifact checksum manifest has no file entries")

    actual_files: set[str] = set()
    for path in version_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Neural artifact may not contain symbolic links")
        if path.is_file() and path != manifest_path:
            actual_files.add(path.relative_to(version_dir).as_posix())
    if set(files) != actual_files:
        raise RuntimeError("Neural artifact file roster does not match its checksum manifest")

    for relative, expected_checksum in files.items():
        if not isinstance(relative, str):
            raise RuntimeError("Neural artifact checksum paths must be strings")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("Neural artifact checksum manifest contains an unsafe path")
        checksum = _normalize_checksum(
            expected_checksum,
            field_name=f"checksum for {relative}",
        )
        file_path = version_dir.joinpath(*relative_path.parts)
        if not file_path.is_file() or file_path.is_symlink():
            raise RuntimeError(f"Neural artifact file is missing or unsafe: {relative}")
        if _sha256_file(file_path) != checksum:
            raise RuntimeError(f"Neural artifact file checksum mismatch: {relative}")
    return manifest_checksum


def _validate_metadata(
    metadata: dict[str, Any],
    expected: _ExpectedArtifact,
) -> None:
    expected_fields: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": expected.artifact_id,
        "model_id": expected.model_id,
        "model_format": "neuralforecast",
        "dataset_embedded": False,
        "fitted_horizon": expected.fitted_horizon,
        "min_history": expected.min_history,
        "params": expected.params,
        "config_checksum": expected.config_checksum,
        "source_sales_batch_id": expected.source_sales_batch_id,
        "data_checksum": expected.data_checksum,
        "history_end": expected.history_end,
        "generator_contract_version": expected.generator_contract_version,
        "training_dfu_count": expected.training_dfu_count,
        "training_row_count": expected.training_row_count,
        "training_cohort_checksum": expected.training_cohort_checksum,
        "training_data_checksum": expected.training_data_checksum,
        "training_contract_version": expected.training_contract_version,
        "runtime_contract": expected.runtime_contract,
        "runtime_contract_checksum": expected.runtime_contract_checksum,
    }
    for field, expected_value in expected_fields.items():
        if metadata.get(field) != expected_value:
            raise RuntimeError(
                f"Neural artifact metadata field {field!r} does not match expected lineage"
            )
    trained_at = metadata.get("trained_at")
    if not isinstance(trained_at, str):
        raise RuntimeError("Neural artifact trained_at metadata is missing")
    try:
        parsed_trained_at = datetime.fromisoformat(trained_at)
    except ValueError as exc:
        raise RuntimeError("Neural artifact trained_at metadata is invalid") from exc
    if parsed_trained_at.tzinfo is None:
        raise RuntimeError("Neural artifact trained_at metadata must be timezone-aware")


def _loaded_from_runtime(
    *,
    runtime: Any,
    metadata: dict[str, Any],
    version_dir: Path,
) -> LoadedNeuralArtifact:
    fitted = FittedNeuralModel(
        neural_forecast=runtime,
        model_id=str(metadata["model_id"]),
        fitted_horizon=int(metadata["fitted_horizon"]),
        min_history=int(metadata["min_history"]),
        training_dfu_count=int(metadata["training_dfu_count"]),
        training_row_count=int(metadata["training_row_count"]),
        training_cohort_checksum=str(metadata["training_cohort_checksum"]),
        training_data_checksum=str(metadata["training_data_checksum"]),
        training_contract_version=str(metadata["training_contract_version"]),
        runtime_contract=dict(metadata["runtime_contract"]),
    )
    ref = NeuralArtifactRef(
        artifact_id=str(metadata["artifact_id"]),
        model_id=str(metadata["model_id"]),
        version_dir=version_dir,
        model_dir=version_dir / _MODEL_DIRNAME,
        metadata=dict(metadata),
    )
    return LoadedNeuralArtifact(fitted_model=fitted, ref=ref)


def _load_version(
    version_dir: Path,
    *,
    expected: _ExpectedArtifact,
    loader: ModelLoader,
    expected_manifest_checksum: str | None,
) -> LoadedNeuralArtifact:
    if not version_dir.is_dir() or version_dir.is_symlink():
        raise RuntimeError(f"Neural artifact version is missing: {version_dir}")
    _validate_file_manifest(
        version_dir,
        expected_manifest_checksum=expected_manifest_checksum,
    )
    metadata = _read_json(version_dir / _METADATA_FILENAME, label="metadata")
    _validate_metadata(metadata, expected)
    model_dir = version_dir / _MODEL_DIRNAME
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise RuntimeError("Neural artifact model directory is missing or unsafe")
    runtime = loader(model_dir)
    _validate_runtime_model(runtime, expected)
    return _loaded_from_runtime(
        runtime=runtime,
        metadata=metadata,
        version_dir=version_dir,
    )


def _activate_version(root: Path, version_dir: Path, artifact_id: str) -> None:
    manifest_checksum = _sha256_file(version_dir / _CHECKSUMS_FILENAME)
    pointer = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "checksums_sha256": manifest_checksum,
    }
    _atomic_write_json(root / _ACTIVE_FILENAME, pointer)


def publish_neural_artifact(
    *,
    fitted: FittedNeuralModel,
    params: Mapping[str, Any],
    source_sales_batch_id: int,
    data_checksum: str,
    history_end: date | datetime | str,
    base_dir: Path,
    generator_contract_version: str = GENERATOR_CONTRACT_VERSION,
    loader: ModelLoader | None = None,
    trained_at: datetime | None = None,
) -> LoadedNeuralArtifact:
    """Publish or reuse one immutable, deterministic neural model version."""
    expected = _build_expected_artifact(
        model_id=fitted.model_id,
        params=params,
        source_sales_batch_id=source_sales_batch_id,
        data_checksum=data_checksum,
        history_end=history_end,
        generator_contract_version=generator_contract_version,
        training_dfu_count=fitted.training_dfu_count,
        training_row_count=fitted.training_row_count,
        training_cohort_checksum=fitted.training_cohort_checksum,
        training_data_checksum=fitted.training_data_checksum,
        training_contract_version=fitted.training_contract_version,
        runtime_contract=fitted.runtime_contract,
    )
    if fitted.fitted_horizon != expected.fitted_horizon:
        raise ValueError("Fitted model horizon does not match artifact params")
    if fitted.min_history != expected.min_history:
        raise ValueError("Fitted model minimum history does not match artifact params")
    if fitted.training_dfu_count != expected.training_dfu_count:
        raise ValueError("Fitted model training_dfu_count does not match artifact lineage")
    if fitted.training_row_count != expected.training_row_count:
        raise ValueError("Fitted model training_row_count does not match artifact lineage")
    _validate_runtime_model(fitted.neural_forecast, expected)

    selected_loader = loader or _default_loader
    root = _artifact_root(Path(base_dir), expected.model_id)
    versions_dir = root / _VERSIONS_DIRNAME
    versions_dir.mkdir(parents=True, exist_ok=True)
    version_dir = versions_dir / expected.artifact_id
    if version_dir.exists():
        loaded = _load_version(
            version_dir,
            expected=expected,
            loader=selected_loader,
            expected_manifest_checksum=None,
        )
        _activate_version(root, version_dir, expected.artifact_id)
        return loaded

    temporary_dir = versions_dir / f".{expected.artifact_id}.{uuid4().hex}.building"
    temporary_dir.mkdir()
    try:
        model_dir = temporary_dir / _MODEL_DIRNAME
        save = getattr(fitted.neural_forecast, "save", None)
        if not callable(save):
            raise RuntimeError("Fitted neural runtime does not support artifact saving")
        save(
            str(model_dir),
            save_dataset=False,
            overwrite=False,
        )
        model_files = [path for path in model_dir.rglob("*") if path.is_file()]
        if not model_dir.is_dir() or not model_files:
            raise RuntimeError("NeuralForecast.save produced no model artifact files")

        metadata = _metadata_for_publish(
            expected=expected,
            fitted=fitted,
            trained_at=trained_at or datetime.now(UTC),
        )
        _write_json(temporary_dir / _METADATA_FILENAME, metadata)
        manifest_checksum = _write_checksum_manifest(temporary_dir)
        built = _load_version(
            temporary_dir,
            expected=expected,
            loader=selected_loader,
            expected_manifest_checksum=manifest_checksum,
        )

        try:
            os.rename(temporary_dir, version_dir)
        except OSError:
            if not version_dir.exists():
                raise
            shutil.rmtree(temporary_dir, ignore_errors=True)
            built = _load_version(
                version_dir,
                expected=expected,
                loader=selected_loader,
                expected_manifest_checksum=None,
            )
        else:
            built = _loaded_from_runtime(
                runtime=built.fitted_model.neural_forecast,
                metadata=built.ref.metadata,
                version_dir=version_dir,
            )
            _fsync_directory(versions_dir)

        _activate_version(root, version_dir, expected.artifact_id)
        return built
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)


def read_active_neural_artifact_ref(
    *,
    model_id: str,
    base_dir: Path,
    expected_params: Mapping[str, Any] | None = None,
    expected_source_sales_batch_id: int | None = None,
    expected_data_checksum: str | None = None,
    expected_history_end: date | datetime | str | None = None,
    expected_training_cohort_checksum: str | None = None,
    expected_training_dfu_count: int | None = None,
    expected_runtime_contract: Mapping[str, Any] | None = None,
    generator_contract_version: str = GENERATOR_CONTRACT_VERSION,
) -> NeuralArtifactRef:
    """Validate the active artifact and return metadata without loading its model.

    This is the lightweight readiness path for APIs. It verifies the active
    pointer, immutable lineage ID, checksum manifest, metadata, and model
    directory without deserializing NeuralForecast or allocating model weights.
    """
    if model_id not in SUPPORTED_NEURAL_MODELS:
        raise ValueError(f"Unsupported neural artifact model: {model_id}")

    root = _artifact_root(Path(base_dir), model_id)
    active_path = root / _ACTIVE_FILENAME
    if not active_path.exists():
        raise FileNotFoundError(active_path)
    pointer = _read_json(active_path, label="active pointer")
    if pointer.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Neural artifact active pointer has an unsupported schema")
    artifact_id = pointer.get("artifact_id")
    if not isinstance(artifact_id, str):
        raise RuntimeError("Neural artifact active pointer lacks an artifact ID")
    normalized_artifact_id = _normalize_checksum(artifact_id, field_name="artifact_id")
    manifest_checksum = pointer.get("checksums_sha256")
    if not isinstance(manifest_checksum, str):
        raise RuntimeError("Neural artifact active pointer lacks a manifest checksum")

    version_dir = root / _VERSIONS_DIRNAME / normalized_artifact_id
    if not version_dir.is_dir() or version_dir.is_symlink():
        raise RuntimeError(f"Neural artifact version is missing: {version_dir}")
    _validate_file_manifest(
        version_dir,
        expected_manifest_checksum=manifest_checksum,
    )
    metadata = _read_json(version_dir / _METADATA_FILENAME, label="metadata")
    try:
        recorded = _build_expected_artifact(
            model_id=model_id,
            params=metadata["params"],
            source_sales_batch_id=metadata["source_sales_batch_id"],
            data_checksum=metadata["data_checksum"],
            history_end=metadata["history_end"],
            generator_contract_version=metadata["generator_contract_version"],
            training_dfu_count=metadata["training_dfu_count"],
            training_row_count=metadata["training_row_count"],
            training_cohort_checksum=metadata["training_cohort_checksum"],
            training_data_checksum=metadata["training_data_checksum"],
            training_contract_version=metadata["training_contract_version"],
            runtime_contract=metadata["runtime_contract"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Neural artifact metadata has invalid lineage fields") from exc
    if recorded.artifact_id != normalized_artifact_id:
        raise RuntimeError("Neural artifact active pointer does not match metadata lineage")
    _validate_metadata(metadata, recorded)

    current_lineage = (
        expected_source_sales_batch_id,
        expected_data_checksum,
        expected_history_end,
    )
    if any(value is not None for value in current_lineage) and not all(
        value is not None for value in current_lineage
    ):
        raise ValueError(
            "Current neural artifact readiness requires sales batch, checksum, and history end"
        )
    try:
        expected = _build_expected_artifact(
            model_id=model_id,
            params=expected_params if expected_params is not None else recorded.params,
            source_sales_batch_id=(
                expected_source_sales_batch_id
                if expected_source_sales_batch_id is not None
                else recorded.source_sales_batch_id
            ),
            data_checksum=(
                expected_data_checksum
                if expected_data_checksum is not None
                else recorded.data_checksum
            ),
            history_end=(
                expected_history_end
                if expected_history_end is not None
                else recorded.history_end
            ),
            generator_contract_version=generator_contract_version,
            training_dfu_count=(
                expected_training_dfu_count
                if expected_training_dfu_count is not None
                else recorded.training_dfu_count
            ),
            training_row_count=recorded.training_row_count,
            training_cohort_checksum=(
                expected_training_cohort_checksum
                if expected_training_cohort_checksum is not None
                else recorded.training_cohort_checksum
            ),
            training_data_checksum=recorded.training_data_checksum,
            training_contract_version=NEURAL_TRAINING_CONTRACT_VERSION,
            runtime_contract=(
                expected_runtime_contract
                if expected_runtime_contract is not None
                else current_neural_runtime_contract()
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Current neural artifact lineage is invalid") from exc
    if expected.artifact_id != normalized_artifact_id:
        if (
            expected.training_cohort_checksum != recorded.training_cohort_checksum
            or expected.training_dfu_count != recorded.training_dfu_count
        ):
            raise NeuralArtifactLineageMismatchError(
                "Active neural artifact training cohort is stale for the current eligible "
                "DFU roster"
            )
        if expected.runtime_contract != recorded.runtime_contract:
            raise NeuralArtifactLineageMismatchError(
                "Active neural artifact runtime contract is stale for the current environment"
            )
        raise NeuralArtifactLineageMismatchError(
            "Active neural artifact metadata lineage is stale for the current "
            "configuration, completed sales batch, or latest closed month"
        )

    model_dir = version_dir / _MODEL_DIRNAME
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise RuntimeError("Neural artifact model directory is missing or unsafe")
    return NeuralArtifactRef(
        artifact_id=normalized_artifact_id,
        model_id=model_id,
        version_dir=version_dir,
        model_dir=model_dir,
        metadata=dict(metadata),
    )


def load_active_neural_artifact(
    *,
    model_id: str,
    params: Mapping[str, Any],
    source_sales_batch_id: int,
    data_checksum: str,
    history_end: date | datetime | str,
    base_dir: Path,
    expected_training_cohort_checksum: str | None = None,
    expected_training_dfu_count: int | None = None,
    expected_runtime_contract: Mapping[str, Any] | None = None,
    generator_contract_version: str = GENERATOR_CONTRACT_VERSION,
    loader: ModelLoader | None = None,
) -> LoadedNeuralArtifact:
    """Load the active version only when it matches the requested full lineage."""
    try:
        ref = read_active_neural_artifact_ref(
            model_id=model_id,
            base_dir=base_dir,
            expected_params=params,
            expected_source_sales_batch_id=source_sales_batch_id,
            expected_data_checksum=data_checksum,
            expected_history_end=history_end,
            expected_training_cohort_checksum=expected_training_cohort_checksum,
            expected_training_dfu_count=expected_training_dfu_count,
            expected_runtime_contract=expected_runtime_contract,
            generator_contract_version=generator_contract_version,
        )
    except NeuralArtifactLineageMismatchError as exc:
        raise RuntimeError("Active neural artifact does not match expected lineage") from exc
    metadata = ref.metadata
    expected = _build_expected_artifact(
        model_id=model_id,
        params=params,
        source_sales_batch_id=source_sales_batch_id,
        data_checksum=data_checksum,
        history_end=history_end,
        generator_contract_version=generator_contract_version,
        training_dfu_count=metadata["training_dfu_count"],
        training_row_count=metadata["training_row_count"],
        training_cohort_checksum=metadata["training_cohort_checksum"],
        training_data_checksum=metadata["training_data_checksum"],
        training_contract_version=metadata["training_contract_version"],
        runtime_contract=metadata["runtime_contract"],
    )
    return _load_version(
        ref.version_dir,
        expected=expected,
        loader=loader or _default_loader,
        expected_manifest_checksum=None,
    )
