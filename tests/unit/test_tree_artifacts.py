"""Immutable all-cluster artifact registry contracts for production LightGBM."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from common.ml.tree_artifact_lineage import (
    TREE_ARTIFACT_LINEAGE_KEY,
    ProductionTreeArtifactLineage,
)
from common.services.forecast_generation import GENERATOR_CONTRACT_VERSION

DATA_CHECKSUM = "a" * 64
TRAINED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
MODEL_CONFIG = {
    "algorithm": {
        "type": "tree",
        "cluster_strategy": "per_cluster",
        "params": {"n_estimators": 20, "learning_rate": 0.05},
    },
    "clustering": {"enabled": True},
    "cluster_tuning_profiles": {"enabled": False},
}


class PicklableTree:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame):
        return [self.value] * len(frame)


def _lineage(
    *,
    source_sales_batch_id: int = 91,
    data_checksum: str = DATA_CHECKSUM,
    history_end: date = date(2026, 6, 1),
    cluster_experiment_id: int | None = 17,
) -> ProductionTreeArtifactLineage:
    return ProductionTreeArtifactLineage(
        source_sales_batch_id=source_sales_batch_id,
        data_checksum=data_checksum,
        history_end=history_end,
        cluster_experiment_id=cluster_experiment_id,
        cluster_assignment_count=(2 if cluster_experiment_id is not None else None),
        cluster_assignment_checksum=("b" * 64 if cluster_experiment_id is not None else None),
        generator_contract_version=GENERATOR_CONTRACT_VERSION,
    )


def _spec(
    *,
    labels: tuple[str, ...] = ("0", "1"),
    lineage: ProductionTreeArtifactLineage | None = None,
    model_config: dict | None = None,
    strategy: str = "per_cluster",
):
    from common.ml.tree_artifacts import build_tree_artifact_spec

    return build_tree_artifact_spec(
        model_id="lgbm_cluster",
        model_config=model_config or MODEL_CONFIG,
        lineage=lineage or _lineage(),
        cluster_strategy=strategy,
        cluster_labels=labels,
    )


def _artifact(label: str, spec, value: float = 1.0) -> dict[str, object]:
    return {
        "model": PicklableTree(value),
        "feature_cols": ["qty_lag_1"],
        "model_id": "lgbm_cluster",
        "cluster_label": label,
        "training_mode": "production",
        "cluster_strategy": spec.cluster_strategy,
        "categorical_encoders": {},
        "feature_importance": {"qty_lag_1": 1.0},
        "config_checksum": spec.config_checksum,
        TREE_ARTIFACT_LINEAGE_KEY: spec.lineage.to_metadata(),
    }


def _publish(tmp_path: Path, *, spec=None, artifacts=None):
    from common.ml.tree_artifacts import publish_tree_artifact_set

    resolved_spec = spec or _spec()
    resolved_artifacts = artifacts or {
        label: _artifact(label, resolved_spec, float(index + 1))
        for index, label in enumerate(resolved_spec.cluster_labels)
    }
    return publish_tree_artifact_set(
        artifacts=resolved_artifacts,
        training_metadata={"n_rows": 100, "n_dfus": 20},
        spec=resolved_spec,
        base_dir=tmp_path,
        trained_at=TRAINED_AT,
    )


def _load(tmp_path: Path, *, spec=None):
    from common.ml.tree_artifacts import load_active_tree_artifact_set

    resolved_spec = spec or _spec()
    return load_active_tree_artifact_set(
        model_id="lgbm_cluster",
        expected_spec=resolved_spec,
        base_dir=tmp_path,
    )


def test_publish_writes_one_immutable_cluster_set_and_atomic_pointer(tmp_path: Path) -> None:
    published = _publish(tmp_path)

    root = tmp_path / "lgbm_cluster" / "production_tree"
    version_dir = root / "versions" / published.ref.artifact_set_id
    pointer = json.loads((root / "active.json").read_text())
    metadata = json.loads((version_dir / "metadata.json").read_text())
    checksums = json.loads((version_dir / "checksums.json").read_text())

    assert published.ref.version_dir == version_dir
    assert set(published.artifacts) == {"0", "1"}
    assert pointer["artifact_set_id"] == published.ref.artifact_set_id
    assert len(pointer["checksums_sha256"]) == 64
    assert metadata["cluster_labels"] == ["0", "1"]
    assert metadata["cluster_strategy"] == "per_cluster"
    assert metadata["config_checksum"] == published.ref.metadata["config_checksum"]
    assert metadata["model_checksums"] == {
        label: checksums["files"][relative] for label, relative in metadata["model_files"].items()
    }
    assert not list((root / "versions").glob(".*.building"))
    assert not list(root.glob(".active.*.tmp"))


def test_partial_training_cannot_replace_an_existing_active_set(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    pointer_path = tmp_path / "lgbm_cluster" / "production_tree" / "active.json"
    before = pointer_path.read_bytes()
    spec = _spec()

    with pytest.raises(RuntimeError, match=r"coverage mismatch.*missing=\['1'\]"):
        _publish(tmp_path, spec=spec, artifacts={"0": _artifact("0", spec)})

    assert pointer_path.read_bytes() == before
    assert _load(tmp_path).ref.artifact_set_id == first.ref.artifact_set_id


def test_pointer_write_failure_leaves_previous_active_set_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _publish(tmp_path)
    pointer_path = tmp_path / "lgbm_cluster" / "production_tree" / "active.json"
    before = pointer_path.read_bytes()

    def fail_pointer(*_args, **_kwargs) -> None:
        raise OSError("simulated pointer failure")

    monkeypatch.setattr("common.ml.tree_artifacts._atomic_write_json", fail_pointer)
    with pytest.raises(OSError, match="pointer failure"):
        _publish(tmp_path)

    assert pointer_path.read_bytes() == before
    monkeypatch.undo()
    assert _load(tmp_path).ref.artifact_set_id == first.ref.artifact_set_id


def test_loader_rejects_tampered_pickle_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = _publish(tmp_path)
    (published.ref.version_dir / "cluster_0.pkl").write_bytes(b"tampered")

    def forbidden_load(*_args, **_kwargs):
        pytest.fail("checksum validation must happen before pickle.load")

    monkeypatch.setattr(pickle, "load", forbidden_load)
    with pytest.raises(RuntimeError, match="checksum"):
        _load(tmp_path)


def test_loader_ignores_legacy_loose_pickles_and_loads_only_manifest_roster(
    tmp_path: Path,
) -> None:
    published = _publish(tmp_path)
    legacy_path = tmp_path / "lgbm_cluster" / "cluster_stale.pkl"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(pickle.dumps({"model": PicklableTree(999.0)}))

    loaded = _load(tmp_path)

    assert loaded.ref.artifact_set_id == published.ref.artifact_set_id
    assert set(loaded.artifacts) == {"0", "1"}
    assert "stale" not in loaded.artifacts


def test_generator_loads_registry_bundle_and_stamps_exact_set_id(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    from scripts.forecasting.generate_production_forecasts import (
        _LoadedTreeModelMap,
        _tree_generation_metadata,
        load_active_models,
    )

    loaded = load_active_models(
        "lgbm_cluster",
        {"model_registry": {"base_path": str(tmp_path)}},
        expected_spec=_spec(),
    )
    runtime = _LoadedTreeModelMap()
    runtime["lgbm_cluster"] = loaded.artifacts
    runtime.artifact_sets["lgbm_cluster"] = loaded

    metadata = _tree_generation_metadata(runtime)

    assert loaded.ref.artifact_set_id == published.ref.artifact_set_id
    assert metadata["tree_artifacts"]["lgbm_cluster"]["artifact_set_id"] == (
        published.ref.artifact_set_id
    )
    assert metadata["tree_artifacts"]["lgbm_cluster"]["lineage"] == (_lineage().to_metadata())


def test_read_active_ref_validates_bundle_without_unpickling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = _publish(tmp_path)
    from common.ml.tree_artifacts import read_active_tree_artifact_ref

    monkeypatch.setattr(
        pickle,
        "load",
        lambda *_args, **_kwargs: pytest.fail("readiness must not unpickle models"),
    )
    ref = read_active_tree_artifact_ref(
        model_id="lgbm_cluster",
        base_dir=tmp_path,
        expected_model_config=MODEL_CONFIG,
    )

    assert ref.artifact_set_id == published.ref.artifact_set_id
    assert ref.metadata["training_metadata"] == {"n_dfus": 20, "n_rows": 100}


def test_read_active_ref_rejects_current_config_drift(tmp_path: Path) -> None:
    _publish(tmp_path)
    from common.ml.tree_artifacts import (
        TreeArtifactLineageMismatchError,
        read_active_tree_artifact_ref,
    )

    with pytest.raises(TreeArtifactLineageMismatchError, match="metadata lineage is stale"):
        read_active_tree_artifact_ref(
            model_id="lgbm_cluster",
            base_dir=tmp_path,
            expected_model_config={**MODEL_CONFIG, "revision": 2},
        )


def test_read_active_ref_rejects_current_exact_spec_drift(tmp_path: Path) -> None:
    _publish(tmp_path)
    from common.ml.tree_artifacts import (
        TreeArtifactLineageMismatchError,
        read_active_tree_artifact_ref,
    )

    with pytest.raises(TreeArtifactLineageMismatchError, match="metadata lineage is stale"):
        read_active_tree_artifact_ref(
            model_id="lgbm_cluster",
            base_dir=tmp_path,
            expected_spec=_spec(lineage=_lineage(source_sales_batch_id=92)),
        )


def test_snapshot_metadata_always_carries_sales_source_without_model_artifact() -> None:
    from scripts.forecasting.generate_production_forecasts import (
        _source_generation_metadata,
    )

    metadata = _source_generation_metadata(
        source_sales_batch_id=91,
        data_checksum=DATA_CHECKSUM,
        history_end=date(2026, 6, 1),
    )

    assert metadata == {
        "source_sales": {
            "source_sales_batch_id": 91,
            "data_checksum": DATA_CHECKSUM,
            "history_end": "2026-06-01",
        }
    }


@pytest.mark.parametrize(
    "spec",
    [
        _spec(model_config={**MODEL_CONFIG, "revision": 2}),
        _spec(lineage=_lineage(source_sales_batch_id=92)),
        _spec(lineage=_lineage(data_checksum="b" * 64)),
        _spec(lineage=_lineage(history_end=date(2026, 5, 1))),
        _spec(lineage=_lineage(cluster_experiment_id=18)),
    ],
)
def test_loader_rejects_every_stale_lineage_or_config(spec, tmp_path: Path) -> None:
    _publish(tmp_path)

    with pytest.raises(RuntimeError, match="expected lineage"):
        _load(tmp_path, spec=spec)


def test_explicit_global_strategy_supports_disabled_clustering(tmp_path: Path) -> None:
    global_config = {
        **MODEL_CONFIG,
        "algorithm": {
            **MODEL_CONFIG["algorithm"],
            "cluster_strategy": "global",
        },
        "clustering": {"enabled": False},
    }
    spec = _spec(
        labels=("global",),
        lineage=_lineage(cluster_experiment_id=None),
        model_config=global_config,
        strategy="global",
    )

    _publish(tmp_path, spec=spec)
    loaded = _load(tmp_path, spec=spec)

    assert set(loaded.artifacts) == {"global"}
    assert loaded.artifacts["global"]["cluster_strategy"] == "global"


@pytest.mark.parametrize(
    ("strategy", "labels", "cluster_experiment_id"),
    [
        ("global", ("0",), None),
        ("global", ("global", "0"), None),
        ("per_cluster", ("global",), 17),
        ("per_cluster", ("0",), None),
    ],
)
def test_spec_rejects_implicit_or_incoherent_global_fallback(
    strategy: str,
    labels: tuple[str, ...],
    cluster_experiment_id: int | None,
) -> None:
    with pytest.raises(ValueError, match="cluster strategy"):
        _spec(
            labels=labels,
            lineage=_lineage(cluster_experiment_id=cluster_experiment_id),
            strategy=strategy,
        )
