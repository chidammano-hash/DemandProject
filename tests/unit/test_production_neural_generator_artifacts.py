"""Generator wiring for immutable NHITS/NBEATS final-refit artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import psycopg
import pytest

from common.ml.neural_forecast import NeuralCohortIdentity
from common.services.sales_lineage import SalesSourceLineage


def _config(tmp_path: Path) -> dict[str, object]:
    params = {
        "h": 6,
        "input_size": 24,
        "max_steps": 10,
        "batch_size": 4,
        "learning_rate": 0.001,
        "scaler_type": "standard",
        "early_stop_patience_steps": -1,
        "min_history": 12,
        "random_seed": 42,
        "start_padding_enabled": True,
        "val_size": 0,
        "deterministic": True,
    }
    return {
        "algorithms": {
            "nhits": {"params": params},
            "nbeats": {"params": {**params, "random_seed": 43}},
            "mstl": {"params": {"min_history": 25}},
        },
        "model_registry": {"base_path": str(tmp_path)},
    }


def test_generator_loads_only_required_neural_artifacts_with_exact_lineage(tmp_path) -> None:
    from scripts.forecasting.generate_production_forecasts import (
        _load_neural_models,
    )

    loaded = SimpleNamespace(
        fitted_model=MagicMock(),
        ref=SimpleNamespace(artifact_id="c" * 64, metadata={"config_checksum": "d" * 64}),
    )
    lineage = SalesSourceLineage(batch_id=91, source_hash="a" * 64)
    history_end = pd.Timestamp("2026-06-01")
    cohort = NeuralCohortIdentity(checksum="e" * 64, dfu_count=2_500)

    with patch(
        "scripts.forecasting.generate_production_forecasts.load_active_neural_artifact",
        return_value=loaded,
    ) as load:
        result = _load_neural_models(
            {"mstl", "lgbm_cluster", "nhits"},
            _config(tmp_path),
            source_sales_batch_id=lineage.batch_id,
            data_checksum=lineage.source_hash,
            history_end=history_end,
            expected_cohorts={"nhits": cohort},
        )

    assert result == {"nhits": loaded}
    assert load.call_args.kwargs["model_id"] == "nhits"
    assert load.call_args.kwargs["source_sales_batch_id"] == 91
    assert load.call_args.kwargs["data_checksum"] == "a" * 64
    assert load.call_args.kwargs["history_end"] == history_end
    assert load.call_args.kwargs["base_dir"] == tmp_path
    assert load.call_args.kwargs["expected_training_cohort_checksum"] == "e" * 64
    assert load.call_args.kwargs["expected_training_dfu_count"] == 2_500


def test_generator_refuses_neural_loading_without_a_current_cohort_identity(tmp_path) -> None:
    from scripts.forecasting.generate_production_forecasts import _load_neural_models

    with (
        patch(
            "scripts.forecasting.generate_production_forecasts.load_active_neural_artifact"
        ) as load,
        pytest.raises(RuntimeError, match="current training-cohort identity"),
    ):
        _load_neural_models(
            {"nhits"},
            _config(tmp_path),
            source_sales_batch_id=91,
            data_checksum="a" * 64,
            history_end=pd.Timestamp("2026-06-01"),
            expected_cohorts={},
        )

    load.assert_not_called()


def test_canonical_generator_passes_loaded_neural_runtime_to_adapter(tmp_path) -> None:
    from scripts.forecasting.generate_production_forecasts import (
        _generate_canonical_non_tree_rows,
    )

    fitted = MagicMock()
    common = {
        "config": _config(tmp_path),
        "model_id": "nhits",
        "sales_df": pd.DataFrame(),
        "dfu_attrs": pd.DataFrame(),
        "item_attrs": pd.DataFrame(),
        "target_dfus": pd.DataFrame(),
        "predict_months": [pd.Timestamp("2026-07-01")],
        "forecast_month_generated": pd.Timestamp("2026-07-01").date(),
        "run_id": "artifact-wiring",
        "fitted_neural_model": fitted,
    }
    with patch(
        "scripts.forecasting.generate_production_forecasts.run_canonical_non_tree_forecast",
        return_value=[],
    ) as run:
        _generate_canonical_non_tree_rows(**common)

    assert run.call_args.kwargs["fitted_neural_model"] is fitted


def test_generation_metadata_records_every_loaded_neural_artifact() -> None:
    from scripts.forecasting.generate_production_forecasts import (
        _neural_generation_metadata,
    )

    artifacts = {
        "nhits": SimpleNamespace(
            ref=SimpleNamespace(
                artifact_id="a" * 64,
                metadata={
                    "config_checksum": "b" * 64,
                    "data_checksum": "c" * 64,
                    "source_sales_batch_id": 91,
                    "history_end": "2026-06-01",
                    "training_cohort_checksum": "d" * 64,
                    "training_data_checksum": "e" * 64,
                    "training_contract_version": "calendar-complete-neural-training-v1",
                    "runtime_contract_checksum": "f" * 64,
                },
            )
        )
    }

    assert _neural_generation_metadata(artifacts) == {
        "neural_artifacts": {
            "nhits": {
                "artifact_id": "a" * 64,
                "config_checksum": "b" * 64,
                "data_checksum": "c" * 64,
                "source_sales_batch_id": 91,
                "history_end": "2026-06-01",
                "training_cohort_checksum": "d" * 64,
                "training_data_checksum": "e" * 64,
                "training_contract_version": "calendar-complete-neural-training-v1",
                "runtime_contract_checksum": "f" * 64,
            }
        }
    }


def test_generation_inputs_begin_in_one_repeatable_read_write_snapshot() -> None:
    from scripts.forecasting.generate_production_forecasts import (
        _begin_generation_snapshot,
    )

    conn = MagicMock()

    _begin_generation_snapshot(conn)

    conn.execute.assert_called_once_with(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"
    )
    assert "READ ONLY" not in conn.execute.call_args.args[0]


def test_cluster_lineage_database_error_fails_without_resetting_snapshot() -> None:
    from scripts.forecasting.generate_production_forecasts import (
        check_champion_cluster_lineage,
    )

    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = psycopg.OperationalError("database unavailable")

    with pytest.raises(RuntimeError, match="could not be verified"):
        check_champion_cluster_lineage(conn)

    conn.rollback.assert_not_called()
