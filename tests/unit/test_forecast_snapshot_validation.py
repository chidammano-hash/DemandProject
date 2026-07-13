"""Tests for reusable, current-lineage snapshot contender validation."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from common.core.paths import PROJECT_ROOT
from common.ml.direct_model_lineage import (
    DIRECT_MODEL_CONFIG_METADATA_KEY,
    SOURCE_MODEL_ROSTER_METADATA_KEY,
    build_direct_model_config_lineage,
)
from common.ml.generation_config_lineage import (
    GENERATION_CONFIG_METADATA_KEY,
    build_generation_config_lineage,
)
from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from common.services.forecast_lineage import ForecastPayloadStats
from common.services.forecast_snapshot_validation import (
    CurrentSalesSource,
    SnapshotContenderIntegrityError,
    SnapshotContenderStaleError,
    _load_current_sales_source,
    _validate_current_model_lineage,
    validate_ready_snapshot_contender,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000123")
RECORD_MONTH = date(2026, 7, 1)
CURRENT_SOURCE = CurrentSalesSource(
    batch_id=91,
    source_hash="a" * 64,
    history_end=date(2026, 6, 1),
    sales_table="fact_sales_monthly_original",
)
CURRENT_GOVERNED = {
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
    "source_sales_batch_id": 91,
    "data_checksum": "a" * 64,
    "cluster_experiment_id": 35,
    "cluster_assignment_count": 13_968,
    "cluster_assignment_checksum": "b" * 64,
}


def _metadata(model_id: str = "mstl") -> dict:
    algorithms = {
        "mstl": {
            "type": "statistical",
            "params": {"season_length": 12, "min_history": 25},
        }
    }
    config = {
        "algorithms": algorithms,
        "production_forecast": {
            "lookback_months": 36,
            "model_registry": {"base_path": "data/models"},
        },
        "forecast_snapshot": {"active_window_months": 12},
        "backtest": {"recursive_lag_smooth": 0.15},
        "clustering": {"enabled": True},
    }
    return {
        GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
        "source_sales": {
            "source_sales_batch_id": CURRENT_SOURCE.batch_id,
            "data_checksum": CURRENT_SOURCE.source_hash,
            "history_end": CURRENT_SOURCE.history_end.isoformat(),
        },
        SOURCE_MODEL_ROSTER_METADATA_KEY: [model_id],
        DIRECT_MODEL_CONFIG_METADATA_KEY: build_direct_model_config_lineage(
            algorithms,
            {model_id},
        ),
        GENERATION_CONFIG_METADATA_KEY: build_generation_config_lineage(
            config,
            {model_id},
        ),
        "governed_champion_lineage": CURRENT_GOVERNED,
        "source_backtest_run_id": CURRENT_GOVERNED["backtest_run_ids"][model_id],
    }


def _manifest(metadata: dict | None = None) -> tuple:
    return (
        "snapshot_contender",
        "mstl",
        RECORD_MONTH,
        6,
        "ready",
        False,
        600,
        100,
        1,
        CURRENT_SOURCE.batch_id,
        "c" * 64,
        metadata if metadata is not None else _metadata(),
    )


def _ready_cursor(metadata: dict | None = None) -> MagicMock:
    cur = MagicMock()
    cur.fetchone.return_value = _manifest(metadata)
    cur.fetchall.return_value = [(lag, 100) for lag in range(6)]
    return cur


def test_current_source_history_is_capped_before_the_record_month():
    cur = MagicMock()
    cur.fetchone.side_effect = [
        (91, "a" * 64, "sku_lvl2_hist_clean.csv"),
        (date(2026, 6, 1),),
    ]

    with patch(
        "common.services.forecast_snapshot_validation.resolve_forecast_sales_table",
        return_value="fact_sales_monthly_original",
    ):
        current = _load_current_sales_source(cur, RECORD_MONTH)

    assert current == CURRENT_SOURCE
    history_sql, history_params = cur.execute.call_args_list[-1].args
    assert "startdate < %s" in history_sql.as_string(None)
    assert history_params == (RECORD_MONTH,)


def test_ready_contender_requires_exact_payload_and_current_lineage():
    cur = _ready_cursor()
    stats = ForecastPayloadStats(
        checksum="c" * 64,
        row_count=600,
        dfu_count=100,
        source_model_count=1,
    )

    with (
        patch(
            "common.services.forecast_snapshot_validation.compute_staging_payload_stats",
            return_value=stats,
        ),
        patch(
            "common.services.forecast_snapshot_validation._load_current_sales_source",
            return_value=CURRENT_SOURCE,
        ),
        patch(
            "common.services.forecast_snapshot_validation.load_active_governed_champion_lineage",
            return_value=CURRENT_GOVERNED,
        ),
        patch(
            "common.services.forecast_snapshot_validation.load_promoted_cluster_population",
            return_value=SimpleNamespace(
                experiment_id=35,
                assignment_count=13_968,
                assignment_checksum="b" * 64,
            ),
        ),
        patch(
            "common.services.forecast_snapshot_validation._validate_current_model_lineage"
        ) as validate_model,
    ):
        result = validate_ready_snapshot_contender(
            cur,
            run_id=RUN_ID,
            model_id="mstl",
            record_month=RECORD_MONTH,
        )

    assert result == stats
    validate_model.assert_called_once_with(
        cur,
        model_id="mstl",
        metadata=_metadata(),
        current_source=CURRENT_SOURCE,
        project_root=PROJECT_ROOT,
    )


def test_ready_contender_rejects_manifest_payload_checksum_mismatch():
    cur = _ready_cursor()
    changed = ForecastPayloadStats(
        checksum="d" * 64,
        row_count=600,
        dfu_count=100,
        source_model_count=1,
    )

    with (
        patch(
            "common.services.forecast_snapshot_validation.compute_staging_payload_stats",
            return_value=changed,
        ),
        pytest.raises(SnapshotContenderIntegrityError, match="payload"),
    ):
        validate_ready_snapshot_contender(
            cur,
            run_id=RUN_ID,
            model_id="mstl",
            record_month=RECORD_MONTH,
        )


def test_ready_contender_rejects_same_month_run_after_sales_reload():
    cur = _ready_cursor()
    stats = ForecastPayloadStats(
        checksum="c" * 64,
        row_count=600,
        dfu_count=100,
        source_model_count=1,
    )
    reloaded = CurrentSalesSource(
        batch_id=92,
        source_hash="b" * 64,
        history_end=date(2026, 6, 1),
        sales_table="fact_sales_monthly_original",
    )

    with (
        patch(
            "common.services.forecast_snapshot_validation.compute_staging_payload_stats",
            return_value=stats,
        ),
        patch(
            "common.services.forecast_snapshot_validation._load_current_sales_source",
            return_value=reloaded,
        ),
        patch(
            "common.services.forecast_snapshot_validation.load_active_governed_champion_lineage",
            return_value=CURRENT_GOVERNED,
        ),
        pytest.raises(SnapshotContenderStaleError, match="sales"),
    ):
        validate_ready_snapshot_contender(
            cur,
            run_id=RUN_ID,
            model_id="mstl",
            record_month=RECORD_MONTH,
        )


def test_neural_contender_requires_the_current_active_artifact(tmp_path):
    metadata = {
        SOURCE_MODEL_ROSTER_METADATA_KEY: ["nhits"],
        DIRECT_MODEL_CONFIG_METADATA_KEY: {},
        "neural_artifacts": {
            "nhits": {
                "artifact_id": "old-id",
                "config_checksum": "1" * 64,
                "data_checksum": CURRENT_SOURCE.source_hash,
                "source_sales_batch_id": CURRENT_SOURCE.batch_id,
                "history_end": CURRENT_SOURCE.history_end.isoformat(),
                "training_cohort_checksum": "3" * 64,
                "training_data_checksum": "4" * 64,
                "training_contract_version": "neural-v1",
                "runtime_contract_checksum": "7" * 64,
            }
        },
    }
    config = {
        "algorithms": {
            "nhits": {
                "type": "deep_learning",
                "params": {"h": 6, "min_history": 12},
            }
        },
        "production_forecast": {"model_registry": {"base_path": "data/models"}},
        "forecast_snapshot": {"active_window_months": 12},
        "backtest": {"recursive_lag_smooth": 0.15},
        "clustering": {"enabled": True},
    }
    metadata[GENERATION_CONFIG_METADATA_KEY] = build_generation_config_lineage(
        config,
        {"nhits"},
    )
    ref = SimpleNamespace(
        artifact_id="new-id",
        metadata={
            "config_checksum": "2" * 64,
            "data_checksum": CURRENT_SOURCE.source_hash,
            "source_sales_batch_id": CURRENT_SOURCE.batch_id,
            "history_end": CURRENT_SOURCE.history_end.isoformat(),
            "training_cohort_checksum": "5" * 64,
            "training_data_checksum": "6" * 64,
            "training_contract_version": "neural-v1",
            "runtime_contract_checksum": "8" * 64,
        },
    )

    with (
        patch(
            "common.services.forecast_snapshot_validation.load_forecast_pipeline_config",
            return_value=config,
        ),
        patch(
            "common.services.forecast_snapshot_validation.read_active_neural_artifact_ref",
            return_value=ref,
        ),
        patch(
            "common.services.forecast_snapshot_validation.load_neural_training_cohort_identity",
            return_value=SimpleNamespace(checksum="3" * 64, dfu_count=100),
        ),
        pytest.raises(SnapshotContenderStaleError, match="neural artifact"),
    ):
        _validate_current_model_lineage(
            MagicMock(),
            model_id="nhits",
            metadata=metadata,
            current_source=CURRENT_SOURCE,
            project_root=tmp_path,
        )


def test_tree_contender_requires_the_current_active_artifact_set(tmp_path):
    config = {
        "algorithms": {
            "lgbm_cluster": {
                "type": "tree",
                "params": {"n_estimators": 100},
            }
        },
        "production_forecast": {
            "lookback_months": 36,
            "model_registry": {"base_path": "data/models"},
        },
        "forecast_snapshot": {"active_window_months": 12},
        "backtest": {"recursive_lag_smooth": 0.15},
        "clustering": {"enabled": False},
    }
    metadata = {
        SOURCE_MODEL_ROSTER_METADATA_KEY: ["lgbm_cluster"],
        DIRECT_MODEL_CONFIG_METADATA_KEY: {},
        GENERATION_CONFIG_METADATA_KEY: build_generation_config_lineage(
            config,
            {"lgbm_cluster"},
        ),
        "tree_artifacts": {
            "lgbm_cluster": {
                "artifact_set_id": "old-set",
                "config_checksum": "1" * 64,
                "cluster_strategy": "global",
                "cluster_labels": ["global"],
                "lineage": {"source_sales_batch_id": CURRENT_SOURCE.batch_id},
            }
        },
    }
    ref = SimpleNamespace(
        artifact_set_id="new-set",
        metadata={
            "config_checksum": "2" * 64,
            "cluster_strategy": "global",
            "cluster_labels": ["global"],
            "lineage": {"source_sales_batch_id": CURRENT_SOURCE.batch_id},
        },
    )

    with (
        patch(
            "common.services.forecast_snapshot_validation.load_forecast_pipeline_config",
            return_value=config,
        ),
        patch(
            "common.services.forecast_snapshot_validation.build_production_tree_model_config_payload",
            return_value={"model": "current"},
        ),
        patch(
            "common.services.forecast_snapshot_validation.build_tree_artifact_spec",
            return_value=MagicMock(),
        ),
        patch(
            "common.services.forecast_snapshot_validation.read_active_tree_artifact_ref",
            return_value=ref,
        ),
        pytest.raises(SnapshotContenderStaleError, match="LightGBM artifact set"),
    ):
        _validate_current_model_lineage(
            MagicMock(),
            model_id="lgbm_cluster",
            metadata=metadata,
            current_source=CURRENT_SOURCE,
            project_root=tmp_path,
        )


def test_snapshot_payload_must_contain_only_lags_zero_through_five():
    cur = _ready_cursor()
    cur.fetchall.return_value = [(lag, 100) for lag in range(5)] + [(6, 100)]
    stats = ForecastPayloadStats(
        checksum="c" * 64,
        row_count=600,
        dfu_count=100,
        source_model_count=1,
    )

    with (
        patch(
            "common.services.forecast_snapshot_validation.compute_staging_payload_stats",
            return_value=stats,
        ),
        patch(
            "common.services.forecast_snapshot_validation._load_current_sales_source",
            return_value=CURRENT_SOURCE,
        ),
        pytest.raises(SnapshotContenderIntegrityError, match=r"lags 0\.\.5"),
    ):
        validate_ready_snapshot_contender(
            cur,
            run_id=RUN_ID,
            model_id="mstl",
            record_month=RECORD_MONTH,
        )
