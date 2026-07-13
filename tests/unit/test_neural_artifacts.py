"""Immutable artifact-registry contracts for canonical neural models."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from common.ml.neural_forecast import (
    NEURAL_TRAINING_CONTRACT_VERSION,
    FittedNeuralModel,
    current_neural_runtime_contract,
)
from common.services.forecast_generation import GENERATOR_CONTRACT_VERSION

PARAMS = {
    "h": 6,
    "input_size": 24,
    "max_steps": 10,
    "batch_size": 4,
    "learning_rate": 0.001,
    "scaler_type": "standard",
    "early_stop_patience_steps": -1,
    "min_history": 12,
    "random_seed": 73,
    "start_padding_enabled": True,
    "val_size": 0,
    "deterministic": True,
}
DATA_CHECKSUM = "a" * 64
TRAINING_COHORT_CHECKSUM = "b" * 64
TRAINING_DATA_CHECKSUM = "c" * 64
TRAINING_CONTRACT_VERSION = NEURAL_TRAINING_CONTRACT_VERSION
RUNTIME_CONTRACT = current_neural_runtime_contract()
HISTORY_END = date(2026, 6, 1)
SOURCE_BATCH_ID = 91
TRAINED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


class FakeSavableNeuralForecast:
    def __init__(self, *, model_id: str = "nhits", horizon: int = 6) -> None:
        self.model_id = model_id
        self.h = horizon
        self.save_calls: list[dict[str, object]] = []

    def save(
        self,
        path: str,
        *,
        save_dataset: bool,
        overwrite: bool,
    ) -> None:
        self.save_calls.append(
            {
                "path": path,
                "save_dataset": save_dataset,
                "overwrite": overwrite,
            }
        )
        model_dir = Path(path)
        model_dir.mkdir(parents=True)
        (model_dir / "NHITS_0.ckpt").write_bytes(b"weights-v1")
        (model_dir / "configuration.pkl").write_bytes(b"configuration-v1")
        (model_dir / "alias_to_model.pkl").write_bytes(b"aliases-v1")


class FakeLoadedNeuralForecast:
    def __init__(self, *, model_id: str = "nhits", horizon: int = 6) -> None:
        self.model_id = model_id
        self.h = horizon


def _fitted(model: FakeSavableNeuralForecast | None = None) -> FittedNeuralModel:
    runtime = model or FakeSavableNeuralForecast()
    return FittedNeuralModel(
        neural_forecast=runtime,
        model_id="nhits",
        fitted_horizon=6,
        min_history=12,
        training_dfu_count=2_500,
        training_row_count=150_000,
        training_cohort_checksum=TRAINING_COHORT_CHECKSUM,
        training_data_checksum=TRAINING_DATA_CHECKSUM,
        training_contract_version=TRAINING_CONTRACT_VERSION,
        runtime_contract=RUNTIME_CONTRACT,
    )


def _loader(path: Path) -> FakeLoadedNeuralForecast:
    assert path.name == "model"
    return FakeLoadedNeuralForecast()


def _publish(tmp_path: Path, **overrides: object):
    from common.ml.neural_artifacts import publish_neural_artifact

    kwargs = {
        "fitted": _fitted(),
        "params": PARAMS,
        "source_sales_batch_id": SOURCE_BATCH_ID,
        "data_checksum": DATA_CHECKSUM,
        "history_end": HISTORY_END,
        "base_dir": tmp_path,
        "generator_contract_version": GENERATOR_CONTRACT_VERSION,
        "loader": _loader,
        "trained_at": TRAINED_AT,
    }
    kwargs.update(overrides)
    return publish_neural_artifact(**kwargs)


def _load(tmp_path: Path, **overrides: object):
    from common.ml.neural_artifacts import load_active_neural_artifact

    kwargs = {
        "model_id": "nhits",
        "params": PARAMS,
        "source_sales_batch_id": SOURCE_BATCH_ID,
        "data_checksum": DATA_CHECKSUM,
        "history_end": HISTORY_END,
        "base_dir": tmp_path,
        "generator_contract_version": GENERATOR_CONTRACT_VERSION,
        "loader": _loader,
    }
    kwargs.update(overrides)
    return load_active_neural_artifact(**kwargs)


def test_artifact_id_is_deterministic_and_covers_all_lineage_inputs() -> None:
    from common.ml.neural_artifacts import build_neural_artifact_id

    common = {
        "model_id": "nhits",
        "params": PARAMS,
        "source_sales_batch_id": SOURCE_BATCH_ID,
        "data_checksum": DATA_CHECKSUM,
        "history_end": HISTORY_END,
        "generator_contract_version": GENERATOR_CONTRACT_VERSION,
        "training_cohort_checksum": TRAINING_COHORT_CHECKSUM,
        "training_data_checksum": TRAINING_DATA_CHECKSUM,
        "training_dfu_count": 2_500,
        "training_row_count": 150_000,
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "runtime_contract": RUNTIME_CONTRACT,
    }
    artifact_id = build_neural_artifact_id(**common)

    assert artifact_id == build_neural_artifact_id(
        **{**common, "params": dict(reversed(list(PARAMS.items())))}
    )
    assert len(artifact_id) == 64
    assert artifact_id != build_neural_artifact_id(**{**common, "model_id": "nbeats"})
    assert artifact_id != build_neural_artifact_id(
        **{**common, "params": {**PARAMS, "random_seed": 74}}
    )
    assert artifact_id != build_neural_artifact_id(
        **{**common, "source_sales_batch_id": SOURCE_BATCH_ID + 1}
    )
    assert artifact_id != build_neural_artifact_id(**{**common, "data_checksum": "b" * 64})
    assert artifact_id != build_neural_artifact_id(**{**common, "history_end": date(2026, 5, 1)})
    assert artifact_id != build_neural_artifact_id(
        **{**common, "generator_contract_version": "next-contract"}
    )
    assert artifact_id != build_neural_artifact_id(
        **{**common, "training_cohort_checksum": "d" * 64}
    )
    assert artifact_id != build_neural_artifact_id(
        **{**common, "training_data_checksum": "e" * 64}
    )
    with pytest.raises(ValueError, match="current neural training contract"):
        build_neural_artifact_id(
            **{**common, "training_contract_version": "calendar-complete-neural-training-v2"}
        )
    assert artifact_id != build_neural_artifact_id(
        **{
            **common,
            "runtime_contract": {**RUNTIME_CONTRACT, "neuralforecast": "different-version"},
        }
    )


def test_current_training_cohort_identity_streams_the_exact_sorted_database_roster() -> None:
    from common.ml.neural_artifacts import load_neural_training_cohort_identity
    from common.ml.neural_forecast import compute_neural_cohort_identity

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.fetchmany.side_effect = [[("sku-1",), ("sku-2",)], []]
    conn = MagicMock()
    conn.cursor.return_value = cursor

    identity = load_neural_training_cohort_identity(
        conn,
        sales_table="fact_sales_monthly_original",
        history_end=date(2026, 6, 1),
        min_history=12,
        fetch_size=2,
    )

    assert identity == compute_neural_cohort_identity(["sku-1", "sku-2"])
    assert cursor.execute.call_args.args[1] == (
        date(2026, 6, 1),
        date(2025, 7, 1),
    )
    assert cursor.fetchmany.call_args_list[0].args == (2,)
    conn.cursor.assert_called_once_with(name="neural_training_cohort")
    assert "sales.type = 1" in str(cursor.execute.call_args.args[0])


def test_publish_writes_immutable_version_checksums_and_atomic_pointer(
    tmp_path: Path,
) -> None:
    model = FakeSavableNeuralForecast()
    loaded = _publish(tmp_path, fitted=_fitted(model))

    version_dir = tmp_path / "nhits" / "neuralforecast" / "versions" / loaded.ref.artifact_id
    assert loaded.ref.version_dir == version_dir
    assert loaded.ref.model_dir == version_dir / "model"
    assert loaded.fitted_model.fitted_horizon == 6
    assert loaded.fitted_model.min_history == 12
    assert len(model.save_calls) == 1
    save_call = model.save_calls[0]
    temporary_model_dir = Path(str(save_call["path"]))
    assert temporary_model_dir.name == "model"
    assert temporary_model_dir.parent.parent == version_dir.parent
    assert temporary_model_dir.parent.name.startswith(f".{loaded.ref.artifact_id}.")
    assert temporary_model_dir.parent.name.endswith(".building")
    assert save_call["save_dataset"] is False
    assert save_call["overwrite"] is False

    metadata = json.loads((version_dir / "metadata.json").read_text())
    checksums = json.loads((version_dir / "checksums.json").read_text())
    active = json.loads((tmp_path / "nhits" / "neuralforecast" / "active.json").read_text())
    assert metadata["artifact_id"] == loaded.ref.artifact_id
    assert metadata["model_id"] == "nhits"
    assert metadata["fitted_horizon"] == 6
    assert metadata["history_end"] == "2026-06-01"
    assert metadata["source_sales_batch_id"] == SOURCE_BATCH_ID
    assert metadata["data_checksum"] == DATA_CHECKSUM
    assert metadata["generator_contract_version"] == GENERATOR_CONTRACT_VERSION
    assert metadata["training_cohort_checksum"] == TRAINING_COHORT_CHECKSUM
    assert metadata["training_data_checksum"] == TRAINING_DATA_CHECKSUM
    assert metadata["training_row_count"] == 150_000
    assert metadata["training_contract_version"] == TRAINING_CONTRACT_VERSION
    assert metadata["runtime_contract"] == RUNTIME_CONTRACT
    assert len(metadata["runtime_contract_checksum"]) == 64
    assert metadata["dataset_embedded"] is False
    assert set(checksums["files"]) == {
        "metadata.json",
        "model/NHITS_0.ckpt",
        "model/alias_to_model.pkl",
        "model/configuration.pkl",
    }
    assert active["artifact_id"] == loaded.ref.artifact_id
    assert len(active["checksums_sha256"]) == 64
    assert not list((version_dir.parent.parent).glob(".active.*.tmp"))
    assert not list(version_dir.parent.glob(".*.building"))


def test_retry_reuses_valid_deterministic_version_without_resaving(tmp_path: Path) -> None:
    first = _publish(tmp_path)

    class MustNotSave(FakeSavableNeuralForecast):
        def save(self, *args, **kwargs) -> None:
            pytest.fail("a valid deterministic version must be reused")

    second = _publish(tmp_path, fitted=_fitted(MustNotSave()))

    assert second.ref.artifact_id == first.ref.artifact_id
    versions = tmp_path / "nhits" / "neuralforecast" / "versions"
    assert [path.name for path in versions.iterdir()] == [first.ref.artifact_id]


def test_changed_training_cohort_publishes_a_new_version_instead_of_reusing_old_fit(
    tmp_path: Path,
) -> None:
    first = _publish(tmp_path)
    fresh_runtime = FakeSavableNeuralForecast()
    changed = _fitted(fresh_runtime)
    changed = FittedNeuralModel(
        **{
            **changed.__dict__,
            "training_cohort_checksum": "d" * 64,
            "training_data_checksum": "e" * 64,
            "training_dfu_count": changed.training_dfu_count + 1,
            "training_row_count": changed.training_row_count + 12,
        }
    )

    second = _publish(tmp_path, fitted=changed)

    assert second.ref.artifact_id != first.ref.artifact_id
    assert len(fresh_runtime.save_calls) == 1
    versions = tmp_path / "nhits" / "neuralforecast" / "versions"
    assert {path.name for path in versions.iterdir()} == {
        first.ref.artifact_id,
        second.ref.artifact_id,
    }


def test_retry_refuses_to_overwrite_tampered_existing_version(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    (first.ref.model_dir / "NHITS_0.ckpt").write_bytes(b"tampered")

    class MustNotSave(FakeSavableNeuralForecast):
        def save(self, *args, **kwargs) -> None:
            pytest.fail("an immutable existing version must never be overwritten")

    with pytest.raises(RuntimeError, match="checksum"):
        _publish(tmp_path, fitted=_fitted(MustNotSave()))


def test_failed_load_validation_never_publishes_partial_version(tmp_path: Path) -> None:
    def wrong_loader(path: Path) -> FakeLoadedNeuralForecast:
        return FakeLoadedNeuralForecast(horizon=24)

    with pytest.raises(RuntimeError, match="horizon"):
        _publish(tmp_path, loader=wrong_loader)

    root = tmp_path / "nhits" / "neuralforecast"
    assert not (root / "active.json").exists()
    assert not list((root / "versions").glob("*"))
    assert not list((root / "versions").glob(".*.building"))


def test_load_active_returns_validated_fitted_model_and_reference(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    loaded = _load(tmp_path)

    assert loaded.ref.artifact_id == published.ref.artifact_id
    assert loaded.fitted_model.model_id == "nhits"
    assert loaded.fitted_model.fitted_horizon == 6
    assert loaded.fitted_model.min_history == 12
    assert loaded.fitted_model.training_dfu_count == 2_500
    assert loaded.fitted_model.training_row_count == 150_000
    assert loaded.fitted_model.training_cohort_checksum == TRAINING_COHORT_CHECKSUM
    assert loaded.fitted_model.training_data_checksum == TRAINING_DATA_CHECKSUM


def test_read_active_ref_validates_artifact_without_loading_model(tmp_path: Path) -> None:
    published = _publish(tmp_path)

    from common.ml.neural_artifacts import read_active_neural_artifact_ref

    ref = read_active_neural_artifact_ref(
        model_id="nhits",
        base_dir=tmp_path,
        expected_params=PARAMS,
    )

    assert ref.artifact_id == published.ref.artifact_id
    assert ref.metadata["trained_at"] == TRAINED_AT.isoformat()
    assert ref.metadata["training_dfu_count"] == 2_500


def test_read_active_ref_rejects_a_proven_current_cohort_mismatch_without_loading_model(
    tmp_path: Path,
) -> None:
    _publish(tmp_path)

    from common.ml.neural_artifacts import read_active_neural_artifact_ref

    with pytest.raises(RuntimeError, match="training cohort"):
        read_active_neural_artifact_ref(
            model_id="nhits",
            base_dir=tmp_path,
            expected_params=PARAMS,
            expected_training_cohort_checksum="f" * 64,
            expected_runtime_contract=RUNTIME_CONTRACT,
        )


def test_load_rejects_a_proven_cohort_mismatch_before_deserializing_weights(
    tmp_path: Path,
) -> None:
    _publish(tmp_path)

    def must_not_load(_path: Path) -> FakeLoadedNeuralForecast:
        pytest.fail("cohort validation must run before heavyweight model loading")

    with pytest.raises(RuntimeError, match="expected lineage"):
        _load(
            tmp_path,
            expected_training_cohort_checksum="f" * 64,
            expected_training_dfu_count=2_501,
            loader=must_not_load,
        )


def test_read_active_ref_validates_exact_current_sales_lineage(tmp_path: Path) -> None:
    published = _publish(tmp_path)

    from common.ml.neural_artifacts import read_active_neural_artifact_ref

    ref = read_active_neural_artifact_ref(
        model_id="nhits",
        base_dir=tmp_path,
        expected_params=PARAMS,
        expected_source_sales_batch_id=SOURCE_BATCH_ID,
        expected_data_checksum=DATA_CHECKSUM,
        expected_history_end=HISTORY_END,
    )

    assert ref.artifact_id == published.ref.artifact_id


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("expected_source_sales_batch_id", SOURCE_BATCH_ID + 1),
        ("expected_data_checksum", "b" * 64),
        ("expected_history_end", date(2026, 5, 1)),
    ],
)
def test_read_active_ref_distinguishes_stale_current_sales_lineage(
    tmp_path: Path,
    override: str,
    value: object,
) -> None:
    _publish(tmp_path)

    from common.ml.neural_artifacts import (
        NeuralArtifactLineageMismatchError,
        read_active_neural_artifact_ref,
    )

    kwargs = {
        "model_id": "nhits",
        "base_dir": tmp_path,
        "expected_params": PARAMS,
        "expected_source_sales_batch_id": SOURCE_BATCH_ID,
        "expected_data_checksum": DATA_CHECKSUM,
        "expected_history_end": HISTORY_END,
    }
    kwargs[override] = value
    with pytest.raises(NeuralArtifactLineageMismatchError, match="metadata lineage is stale"):
        read_active_neural_artifact_ref(**kwargs)


def test_read_active_ref_rejects_tampered_model_file(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    (published.ref.model_dir / "NHITS_0.ckpt").write_bytes(b"tampered")

    from common.ml.neural_artifacts import read_active_neural_artifact_ref

    with pytest.raises(RuntimeError, match="checksum"):
        read_active_neural_artifact_ref(model_id="nhits", base_dir=tmp_path)


def test_read_active_ref_reports_missing_artifact_without_model_loading(tmp_path: Path) -> None:
    from common.ml.neural_artifacts import read_active_neural_artifact_ref

    with pytest.raises(FileNotFoundError):
        read_active_neural_artifact_ref(model_id="nhits", base_dir=tmp_path)


def test_read_active_ref_rejects_artifact_for_stale_model_config(tmp_path: Path) -> None:
    _publish(tmp_path)

    from common.ml.neural_artifacts import read_active_neural_artifact_ref

    with pytest.raises(RuntimeError, match="metadata lineage"):
        read_active_neural_artifact_ref(
            model_id="nhits",
            base_dir=tmp_path,
            expected_params={**PARAMS, "random_seed": 74},
        )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("source_sales_batch_id", SOURCE_BATCH_ID + 1),
        ("data_checksum", "b" * 64),
        ("history_end", date(2026, 5, 1)),
        ("generator_contract_version", "next-contract"),
    ],
)
def test_load_active_rejects_stale_or_mismatched_lineage(
    tmp_path: Path,
    override: str,
    value: object,
) -> None:
    _publish(tmp_path)

    with pytest.raises(RuntimeError, match="expected lineage"):
        _load(tmp_path, **{override: value})


def test_load_active_rejects_config_mismatch(tmp_path: Path) -> None:
    _publish(tmp_path)

    with pytest.raises(RuntimeError, match="expected lineage"):
        _load(tmp_path, params={**PARAMS, "random_seed": 74})


def test_load_active_rejects_tampered_model_file(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    (published.ref.model_dir / "NHITS_0.ckpt").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="checksum"):
        _load(tmp_path)


def test_load_active_rejects_tampered_checksum_manifest(tmp_path: Path) -> None:
    published = _publish(tmp_path)
    checksums_path = published.ref.version_dir / "checksums.json"
    manifest = json.loads(checksums_path.read_text())
    manifest["files"]["model/NHITS_0.ckpt"] = "0" * 64
    checksums_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="manifest checksum"):
        _load(tmp_path)


def test_load_active_rejects_tampered_active_manifest_reference(tmp_path: Path) -> None:
    _publish(tmp_path)
    active_path = tmp_path / "nhits" / "neuralforecast" / "active.json"
    active = json.loads(active_path.read_text())
    active["checksums_sha256"] = "0" * 64
    active_path.write_text(json.dumps(active))

    with pytest.raises(RuntimeError, match="manifest checksum"):
        _load(tmp_path)


def test_load_active_rejects_wrong_runtime_model_identity(tmp_path: Path) -> None:
    _publish(tmp_path)

    def wrong_loader(path: Path) -> FakeLoadedNeuralForecast:
        return FakeLoadedNeuralForecast(model_id="nbeats")

    with pytest.raises(RuntimeError, match="model identity"):
        _load(tmp_path, loader=wrong_loader)


@pytest.mark.parametrize("filename", ["unexpected.bin", "checksums.json"])
def test_load_active_rejects_untracked_files(tmp_path: Path, filename: str) -> None:
    published = _publish(tmp_path)
    (published.ref.model_dir / filename).write_bytes(b"not checksummed")

    with pytest.raises(RuntimeError, match="file roster"):
        _load(tmp_path)
