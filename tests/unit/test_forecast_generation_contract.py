"""Tests for immutable production-forecast generator provenance."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from scripts.forecasting.generate_production_forecasts import (
    _collect_generation_evidence,
    _direct_generation_metadata,
    _generation_config_metadata,
    _invalidate_generation_reservation_on_failure,
    write_forecast_staging,
)


def _governed_champion_lineage(*, sales_batch_id: int = 91) -> dict:
    return {
        "experiment_id": 82,
        "models": [
            "lgbm_cluster",
            "nhits",
            "nbeats",
            "mstl",
            "chronos2_enriched",
        ],
        "backtest_run_ids": {
            "lgbm_cluster": 101,
            "nhits": 102,
            "nbeats": 103,
            "mstl": 104,
            "chronos2_enriched": 105,
        },
        "source_sales_batch_id": sales_batch_id,
        "data_checksum": "a" * 64,
        "cluster_experiment_id": 35,
        "cluster_assignment_count": 13_968,
        "cluster_assignment_checksum": "b" * 64,
    }


def test_generation_metadata_preserves_pre_aggregation_direct_model_roster():
    from common.ml.direct_model_lineage import (
        DIRECT_MODEL_CONFIG_METADATA_KEY,
        SOURCE_MODEL_ROSTER_METADATA_KEY,
    )

    config = {
        "algorithms": {
            "lgbm_cluster": {"type": "tree", "params": {"n_estimators": 20}},
            "mstl": {
                "type": "statistical",
                "params": {"season_length": 12, "min_history": 25},
            },
        }
    }

    metadata = _direct_generation_metadata(
        config,
        {"mstl", "lgbm_cluster"},
    )

    assert metadata[SOURCE_MODEL_ROSTER_METADATA_KEY] == ["lgbm_cluster", "mstl"]
    assert set(metadata[DIRECT_MODEL_CONFIG_METADATA_KEY]) == {"mstl"}


def test_generation_metadata_stamps_global_output_config():
    from common.ml.generation_config_lineage import GENERATION_CONFIG_METADATA_KEY

    pipeline = {
        "algorithms": {
            "mstl": {
                "type": "statistical",
                "params": {"season_length": 12, "min_history": 25},
            }
        },
        "production_forecast": {"lookback_months": 36},
        "forecast_snapshot": {"active_window_months": 12},
        "backtest": {"recursive_lag_smooth": 0.15},
        "clustering": {"enabled": True},
    }

    metadata = _generation_config_metadata(
        {"_full_pipeline": pipeline},
        {"mstl"},
    )

    assert len(metadata[GENERATION_CONFIG_METADATA_KEY]["config_checksum"]) == 64


def test_staging_manifest_records_current_generator_contract():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 1
    cur.fetchone.side_effect = [
        (
            "snapshot_contender",
            "mstl",
            date(2026, 7, 1),
            1,
            "generating",
            0,
        ),
        (False,),
        ("c" * 64, 1, 1, 1),
    ]
    run_id = "00000000-0000-0000-0000-000000000123"
    row = {
        "model_id": "mstl",
        "item_id": "sku-1",
        "loc": "1401-BULK",
        "forecast_month": date(2026, 7, 1),
        "forecast_month_generated": date(2026, 7, 1),
        "forecast_qty": 10.0,
        "forecast_qty_lower": 8.0,
        "forecast_qty_upper": 12.0,
        "cluster_id": "stable",
        "horizon_months": 1,
        "is_recursive": False,
        "lag_source": "actual",
        "run_id": run_id,
        "generated_at": pd.Timestamp("2026-07-01T00:00:00Z").to_pydatetime(),
    }

    write_forecast_staging(
        [row],
        conn,
        "mstl",
        generation_purpose="snapshot_contender",
        generation_evidence={
            "governed_champion_lineage": _governed_champion_lineage(),
            "source_backtest_run_id": 104,
        },
    )

    manifest_sql, manifest_params = cur.execute.call_args_list[0].args
    assert "metadata" in manifest_sql
    assert manifest_params[-1].obj == {
        GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
        "governed_champion_lineage": _governed_champion_lineage(),
        "source_backtest_run_id": 104,
    }


def test_snapshot_manifest_evidence_records_the_current_sales_batch():
    conn = MagicMock()
    with patch(
        "scripts.forecasting.generate_production_forecasts.load_completed_sales_lineage",
        return_value=SimpleNamespace(batch_id=91, source_hash="a" * 64),
    ), patch(
        "scripts.forecasting.generate_production_forecasts.load_active_governed_champion_lineage",
        return_value=_governed_champion_lineage(),
    ), patch(
        "scripts.forecasting.generate_production_forecasts.load_promoted_cluster_population",
        return_value=SimpleNamespace(
            experiment_id=35,
            assignment_count=13_968,
            assignment_checksum="b" * 64,
        ),
    ):
        evidence = _collect_generation_evidence(
            conn,
            candidate_model_id="mstl",
            generation_purpose="snapshot_contender",
        )

    assert evidence["source_sales_batch_id"] == 91
    assert evidence["source_backtest_run_id"] == 104
    assert evidence["governed_champion_lineage"] == _governed_champion_lineage()


def test_champion_generation_binds_exact_governed_refresh_lineage():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [
        (35,),
        (82, 35, True, "c" * 64, "d" * 64),
    ]
    clusters = SimpleNamespace(
        experiment_id=35,
        assignment_count=13_968,
        assignment_checksum="b" * 64,
    )
    with (
        patch(
            "scripts.forecasting.generate_production_forecasts.load_completed_sales_lineage",
            return_value=SimpleNamespace(batch_id=91, source_hash="a" * 64),
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.load_governed_champion_lineage",
            return_value=_governed_champion_lineage(),
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.load_promoted_cluster_population",
            return_value=clusters,
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "scripts.forecasting.generate_production_forecasts.sha256_file",
            return_value="c" * 64,
        ),
    ):
        evidence = _collect_generation_evidence(
            conn,
            candidate_model_id="champion",
            generation_purpose="release_candidate",
        )

    assert evidence["governed_champion_lineage"] == _governed_champion_lineage()


def test_champion_generation_rejects_refresh_from_older_sales_batch():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [
        (35,),
        (82, 35, True, "c" * 64, "d" * 64),
    ]
    with (
        patch(
            "scripts.forecasting.generate_production_forecasts.load_completed_sales_lineage",
            return_value=SimpleNamespace(batch_id=91, source_hash="a" * 64),
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.load_governed_champion_lineage",
            return_value=_governed_champion_lineage(sales_batch_id=90),
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.load_promoted_cluster_population",
            return_value=SimpleNamespace(
                experiment_id=35,
                assignment_count=13_968,
                assignment_checksum="b" * 64,
            ),
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "scripts.forecasting.generate_production_forecasts.sha256_file",
            return_value="c" * 64,
        ),
        pytest.raises(ValueError, match="run model-refresh"),
    ):
        _collect_generation_evidence(
            conn,
            candidate_model_id="champion",
            generation_purpose="release_candidate",
        )


def test_generator_failure_invalidates_pre_reserved_run_after_rollback():
    run_id = "00000000-0000-0000-0000-000000000123"
    connect = MagicMock()
    invalidate = MagicMock(return_value=True)

    with (
        patch(
            "scripts.forecasting.generate_production_forecasts.psycopg.connect",
            connect,
        ),
        patch(
            "scripts.forecasting.generate_production_forecasts.invalidate_generation_run",
            invalidate,
        ),
        pytest.raises(RuntimeError, match="inference failed"),
    ):
        with _invalidate_generation_reservation_on_failure(
            {"dbname": "demand"},
            run_id=run_id,
            dry_run=False,
        ):
            raise RuntimeError("inference failed")

    invalidate.assert_called_once()
    assert invalidate.call_args.args[1] == run_id
