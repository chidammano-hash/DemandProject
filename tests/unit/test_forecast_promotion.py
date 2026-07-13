"""Tests for immutable, transaction-safe forecast promotion."""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from common.ml.direct_model_lineage import (
    DIRECT_MODEL_CONFIG_METADATA_KEY,
    SOURCE_MODEL_ROSTER_METADATA_KEY,
    build_direct_model_config_lineage,
)
from common.ml.generation_config_lineage import (
    GENERATION_CONFIG_METADATA_KEY,
    build_generation_config_lineage,
)
from common.ml.tree_artifact_lineage import ProductionTreeArtifactLineage
from common.services.champion_lineage import (
    CANONICAL_CHAMPION_MODELS,
    GOVERNED_CHAMPION_LINEAGE_METADATA_KEY,
)
from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from common.services.forecast_promotion import (
    ForecastGenerationManifest,
    PromotionConflictError,
    _archive_outgoing_release,
    _candidate_coverage,
    _candidate_quality_report,
    _current_release_evidence,
    _validate_active_model_artifacts,
    _validate_direct_model_lineage,
    _validate_governed_champion_source,
    _validate_incoming_snapshot_roster,
    _validate_tree_model_lineage,
    promote_forecast_run,
    validate_generation_manifest,
)
from common.services.forecast_snapshot_validation import SnapshotContenderStaleError

RUN_ID = UUID("00000000-0000-0000-0000-000000000111")
PRODUCTION_RUN_ID = UUID("00000000-0000-0000-0000-000000000222")


def _pipeline_config(algorithms: dict) -> dict:
    return {
        "algorithms": algorithms,
        "production_forecast": {"lookback_months": 36},
        "forecast_snapshot": {"active_window_months": 12},
        "backtest": {"recursive_lag_smooth": 0.15},
        "clustering": {"enabled": True},
    }


def _quality_policy() -> dict:
    return {
        "require_external_benchmark": True,
        "quality_lookback_months": 6,
        "min_relative_wape_lift_vs_naive_pct": 10.0,
        "min_accuracy_delta_vs_external_pct_points": 0.0,
        "max_abs_bias_pct": 5.0,
        "min_coverage_frac": 0.95,
        "min_common_cohort_coverage_frac": 0.95,
        "min_common_cohort_closed_months": 6,
        "min_common_cohort_dfus": 1000,
        "min_common_cohort_actual_volume": 1.0,
    }


def test_release_evidence_counts_stale_profiles_for_current_clusters_only():
    cur = MagicMock()
    cur.fetchone.return_value = tuple(range(11))

    _current_release_evidence(cur)

    sql = cur.execute.call_args.args[0]
    assert "current_sku_cluster_assignment" in sql
    assert "assignment.ml_cluster = tuning.cluster_name" in sql


def test_candidate_coverage_uses_shared_all_active_group_eligibility_contract():
    cur = MagicMock()
    completed_at = datetime(2026, 7, 1, tzinfo=UTC)
    cur.fetchone.side_effect = [
        ("fact_sales_monthly_original",),
        (100, completed_at, 100, completed_at, "sku_lvl2_hist_clean.csv"),
        (10, 10, 0, 60, 60, 0),
    ]

    result = _candidate_coverage(
        cur,
        source_run_id=RUN_ID,
        model_id="champion",
        planning_month=date(2026, 7, 1),
        required_months=6,
        min_history_months=3,
        active_window_months=12,
    )

    sql, params = cur.execute.call_args.args
    normalized_sql = " ".join(sql.split())
    assert "group_history AS" in normalized_sql
    assert "active_customer_groups AS" in normalized_sql
    assert "BOOL_AND(" in normalized_sql
    assert "active.history_months >= %s" in normalized_sql
    assert "FROM fact_sales_monthly_original sales" in normalized_sql
    assert "JOIN eligible_item_locations USING (item_id, loc)" in normalized_sql
    assert params[:4] == (date(2026, 7, 1), date(2026, 7, 1), 12, 3)
    assert result == (10, 10, 0, 60, 60, 0)


def _manifest(**overrides) -> ForecastGenerationManifest:
    governed_lineage = {
        "experiment_id": 33,
        "models": list(CANONICAL_CHAMPION_MODELS),
        "backtest_run_ids": dict(
            zip(
                CANONICAL_CHAMPION_MODELS,
                range(201, 206),
                strict=True,
            )
        ),
        "source_sales_batch_id": 101,
        "data_checksum": "f" * 64,
        "cluster_experiment_id": 7,
        "cluster_assignment_count": 13_968,
        "cluster_assignment_checksum": "b" * 64,
    }
    values = {
        "run_id": RUN_ID,
        "generation_purpose": "release_candidate",
        "run_status": "ready",
        "promotion_eligible": True,
        "requested_model_id": "champion",
        "forecast_month_generated": date(2026, 7, 1),
        "horizon_months": 12,
        "row_count": 120,
        "dfu_count": 10,
        "source_model_count": 3,
        "champion_experiment_id": 33,
        "cluster_experiment_id": 7,
        "source_sales_batch_id": 101,
        "routing_artifact_checksum": "a" * 64,
        "champion_results_checksum": "e" * 64,
        "artifact_checksum": "c" * 64,
        "metadata": {
            GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
            SOURCE_MODEL_ROSTER_METADATA_KEY: ["lgbm_cluster"],
            DIRECT_MODEL_CONFIG_METADATA_KEY: {},
            "source_sales": {
                "source_sales_batch_id": 101,
                "data_checksum": "f" * 64,
                "history_end": "2026-06-01",
            },
            GOVERNED_CHAMPION_LINEAGE_METADATA_KEY: governed_lineage,
        },
    }
    values.update(overrides)
    return ForecastGenerationManifest(**values)


def test_champion_promotion_requires_same_governed_refresh_lineage():
    manifest = _manifest()
    current = manifest.metadata[GOVERNED_CHAMPION_LINEAGE_METADATA_KEY]

    with patch(
        "common.services.forecast_promotion.load_governed_champion_lineage",
        return_value=current,
    ):
        _validate_governed_champion_source(MagicMock(), manifest=manifest)


def test_champion_promotion_rejects_refresh_from_different_sales_batch():
    manifest = _manifest()
    current = {
        **manifest.metadata[GOVERNED_CHAMPION_LINEAGE_METADATA_KEY],
        "source_sales_batch_id": 102,
    }

    with (
        patch(
            "common.services.forecast_promotion.load_governed_champion_lineage",
            return_value=current,
        ),
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        _validate_governed_champion_source(MagicMock(), manifest=manifest)

    assert exc_info.value.code == "candidate_lineage_mismatch"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"generation_purpose": "snapshot_contender"}, "candidate_run_not_promotable"),
        ({"run_status": "generating"}, "candidate_run_not_promotable"),
        ({"promotion_eligible": False}, "candidate_run_not_promotable"),
        ({"requested_model_id": "lgbm_cluster"}, "candidate_lineage_mismatch"),
        ({"forecast_month_generated": date(2026, 6, 1)}, "stale_candidate_evidence"),
        ({"horizon_months": 5}, "candidate_gate_failed"),
        ({"metadata": {}}, "candidate_lineage_mismatch"),
        (
            {"metadata": {GENERATOR_CONTRACT_METADATA_KEY: "retired-heuristic-v0"}},
            "candidate_lineage_mismatch",
        ),
    ],
)
def test_manifest_validation_fails_closed(overrides, code):
    with pytest.raises(PromotionConflictError) as exc_info:
        validate_generation_manifest(
            _manifest(**overrides),
            model_id="champion",
            planning_month=date(2026, 7, 1),
            required_months=6,
        )
    assert exc_info.value.code == code


def test_promotion_rejects_direct_model_config_changed_after_generation():
    generated_algorithms = {
        "chronos2_enriched": {
            "type": "foundation",
            "params": {"model_name": "amazon/chronos-2", "model_revision": "rev-a"},
        }
    }
    current_algorithms = {
        "chronos2_enriched": {
            "type": "foundation",
            "params": {"model_name": "amazon/chronos-2", "model_revision": "rev-b"},
        }
    }
    metadata = {
        GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
        SOURCE_MODEL_ROSTER_METADATA_KEY: ["chronos2_enriched"],
        DIRECT_MODEL_CONFIG_METADATA_KEY: build_direct_model_config_lineage(
            generated_algorithms,
            {"chronos2_enriched"},
        ),
        GENERATION_CONFIG_METADATA_KEY: build_generation_config_lineage(
            _pipeline_config(generated_algorithms),
            {"chronos2_enriched"},
        ),
    }
    cur = MagicMock()
    cur.fetchall.return_value = [("chronos2_enriched",)]

    with (
        patch(
            "common.services.forecast_promotion.load_forecast_pipeline_config",
            return_value=_pipeline_config(current_algorithms),
        ),
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        _validate_direct_model_lineage(
            cur,
            manifest=_manifest(metadata=metadata),
            model_id="champion",
        )

    assert exc_info.value.code == "candidate_lineage_mismatch"


def test_promotion_validates_direct_members_hidden_by_aggregate_ensemble_label():
    algorithms = {
        "lgbm_cluster": {"type": "tree", "params": {"n_estimators": 100}},
        "mstl": {
            "type": "statistical",
            "params": {"season_length": 12, "min_history": 25},
        },
    }
    metadata = {
        GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
        SOURCE_MODEL_ROSTER_METADATA_KEY: ["lgbm_cluster", "mstl"],
        DIRECT_MODEL_CONFIG_METADATA_KEY: build_direct_model_config_lineage(
            algorithms,
            {"lgbm_cluster", "mstl"},
        ),
        GENERATION_CONFIG_METADATA_KEY: build_generation_config_lineage(
            _pipeline_config(algorithms),
            {"lgbm_cluster", "mstl"},
        ),
    }
    cur = MagicMock()
    # Customer-group aggregation legitimately hides MSTL behind `ensemble`.
    cur.fetchall.return_value = [("ensemble",)]

    with patch(
        "common.services.forecast_promotion.load_forecast_pipeline_config",
        return_value=_pipeline_config(algorithms),
    ):
        _validate_direct_model_lineage(
            cur,
            manifest=_manifest(metadata=metadata),
            model_id="champion",
        )


def test_promotion_rejects_missing_generated_source_model_roster():
    cur = MagicMock()
    cur.fetchall.return_value = [("ensemble",)]

    with (
        patch(
            "common.services.forecast_promotion.load_forecast_pipeline_config",
            return_value={"algorithms": {}},
        ),
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        _validate_direct_model_lineage(
            cur,
            manifest=_manifest(
                metadata={
                    GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
                }
            ),
            model_id="champion",
        )

    assert exc_info.value.code == "candidate_lineage_mismatch"


def test_promotion_rejects_same_experiment_with_changed_cluster_assignments():
    lineage = ProductionTreeArtifactLineage(
        source_sales_batch_id=101,
        data_checksum="a" * 64,
        history_end=date(2026, 6, 1),
        cluster_experiment_id=7,
        cluster_assignment_count=2,
        cluster_assignment_checksum="b" * 64,
    )
    manifest = _manifest(
        metadata={
            GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
            SOURCE_MODEL_ROSTER_METADATA_KEY: ["lgbm_cluster"],
            "tree_artifacts": {
                "lgbm_cluster": {
                    "artifact_set_id": "tree-set",
                    "lineage": lineage.to_metadata(),
                }
            },
        }
    )
    current = SimpleNamespace(
        experiment_id=7,
        assignment_count=2,
        assignment_checksum="c" * 64,
    )

    with (
        patch(
            "common.services.forecast_promotion.load_promoted_cluster_population",
            return_value=current,
        ),
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        _validate_tree_model_lineage(MagicMock(), manifest=manifest)

    assert exc_info.value.code == "candidate_lineage_mismatch"


@pytest.mark.parametrize(
    ("model_id", "metadata_key", "id_key", "reader_name"),
    [
        (
            "lgbm_cluster",
            "tree_artifacts",
            "artifact_set_id",
            "read_active_tree_artifact_ref",
        ),
        (
            "nhits",
            "neural_artifacts",
            "artifact_id",
            "read_active_neural_artifact_ref",
        ),
    ],
)
def test_promotion_rejects_changed_active_model_artifact_after_generation(
    tmp_path,
    model_id: str,
    metadata_key: str,
    id_key: str,
    reader_name: str,
) -> None:
    generated_id = "a" * (32 if model_id == "lgbm_cluster" else 64)
    active_id = "b" * len(generated_id)
    artifact_lineage = {id_key: generated_id}
    if model_id == "nhits":
        artifact_lineage.update(
            {
                "source_sales_batch_id": 91,
                "data_checksum": "c" * 64,
                "history_end": "2026-06-01",
                "training_cohort_checksum": "d" * 64,
            }
        )
    manifest = _manifest(
        requested_model_id=model_id,
        metadata={
            GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
            SOURCE_MODEL_ROSTER_METADATA_KEY: [model_id],
            metadata_key: {model_id: artifact_lineage},
        },
    )
    config = {
        "algorithms": {model_id: {"params": {"h": 6, "min_history": 12}}},
        "production_forecast": {
            "model_registry": {"base_path": str(tmp_path)},
        },
    }
    ref = SimpleNamespace(**{id_key: active_id})

    with (
        patch(
            f"common.services.forecast_promotion.{reader_name}",
            return_value=ref,
        ),
        patch(
            "common.services.forecast_promotion.resolve_forecast_sales_table",
            return_value="fact_sales_monthly_original",
        ),
        patch(
            "common.services.forecast_promotion.load_neural_training_cohort_identity",
            return_value=SimpleNamespace(checksum="d" * 64, dfu_count=2_500),
        ),
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        _validate_active_model_artifacts(
            conn=MagicMock(),
            cur=MagicMock(),
            manifest=manifest,
            pipeline_config=config,
        )

    assert exc_info.value.code == "candidate_lineage_mismatch"


def test_promotion_rejects_current_neural_cohort_drift(tmp_path) -> None:
    manifest = _manifest(
        requested_model_id="nbeats",
        metadata={
            GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
            SOURCE_MODEL_ROSTER_METADATA_KEY: ["nbeats"],
            "neural_artifacts": {
                "nbeats": {
                    "artifact_id": "a" * 64,
                    "source_sales_batch_id": 91,
                    "data_checksum": "c" * 64,
                    "history_end": "2026-06-01",
                    "training_cohort_checksum": "d" * 64,
                }
            },
        },
    )
    config = {
        "algorithms": {"nbeats": {"params": {"h": 6, "min_history": 12}}},
        "production_forecast": {"model_registry": {"base_path": str(tmp_path)}},
    }

    with (
        patch(
            "common.services.forecast_promotion.resolve_forecast_sales_table",
            return_value="fact_sales_monthly_original",
        ),
        patch(
            "common.services.forecast_promotion.load_neural_training_cohort_identity",
            return_value=SimpleNamespace(checksum="e" * 64, dfu_count=2_501),
        ),
        patch(
            "common.services.forecast_promotion.read_active_neural_artifact_ref"
        ) as read_active,
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        _validate_active_model_artifacts(
            conn=MagicMock(),
            cur=MagicMock(),
            manifest=manifest,
            pipeline_config=config,
        )

    assert exc_info.value.code == "candidate_lineage_mismatch"
    read_active.assert_not_called()


@pytest.mark.parametrize(
    "model_id",
    ["lgbm_cluster", "chronos2_enriched", "mstl", "nbeats", "nhits"],
)
def test_single_model_candidates_enter_transactional_promotion(model_id: str) -> None:
    conn = MagicMock()

    with (
        patch(
            "common.services.forecast_promotion._load_manifest",
            side_effect=PromotionConflictError(
                "candidate_run_not_found",
                "The selected forecast generation run does not exist.",
            ),
        ),
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        promote_forecast_run(
            conn,
            model_id=model_id,
            source_run_id=RUN_ID,
            planning_month=date(2026, 7, 1),
            promoted_by="tester",
            notes=None,
            policy={"required_months": 6},
        )

    assert exc_info.value.code == "candidate_run_not_found"
    conn.transaction.assert_called_once()


@pytest.mark.parametrize(
    ("model_id", "source_model_count", "promotion_type"),
    [("champion", 3, "champion"), ("mstl", 1, "single")],
)
def test_promotion_preflights_before_mutation_and_scopes_copy_to_source_run(
    model_id: str,
    source_model_count: int,
    promotion_type: str,
):
    conn = MagicMock()
    tx = conn.transaction.return_value
    tx.__enter__.return_value = tx
    tx.__exit__.return_value = False
    cur = conn.cursor.return_value.__enter__.return_value
    executed: list[tuple[str, object]] = []

    def capture(sql, params=None):
        executed.append((" ".join(sql.split()), params))
        if "INSERT INTO model_promotion_log" in sql:
            cur.fetchone.return_value = (44,)
        cur.rowcount = 1 if "UPDATE forecast_generation_run" in sql else 120

    cur.execute.side_effect = capture

    with (
        patch(
            "common.services.forecast_promotion._load_manifest",
            return_value=_manifest(
                requested_model_id=model_id,
                source_model_count=source_model_count,
            ),
        ),
        patch("common.services.forecast_promotion._validate_candidate_evidence") as validate,
        patch(
            "common.services.forecast_promotion._validate_incoming_snapshot_roster",
            return_value={"roster_models": 4, "ready_contenders": 3},
        ),
        patch(
            "common.services.forecast_promotion._archive_outgoing_release",
            return_value=(None, None, None),
        ),
        patch("common.services.forecast_promotion.compute_staging_payload_stats") as staging_stats,
        patch(
            "common.services.forecast_promotion.compute_production_payload_stats"
        ) as production_stats,
        patch("common.services.forecast_promotion.uuid.uuid4", return_value=PRODUCTION_RUN_ID),
    ):
        stats = MagicMock(
            checksum="c" * 64,
            row_count=120,
            dfu_count=10,
            source_model_count=source_model_count,
        )
        staging_stats.return_value = stats
        production_stats.return_value = stats
        result = promote_forecast_run(
            conn,
            model_id=model_id,
            source_run_id=RUN_ID,
            planning_month=date(2026, 7, 1),
            promoted_by="api",
            notes=None,
            policy={"required_months": 6, "min_coverage_frac": 0.95, "min_ci_coverage_frac": 0.95},
        )

    validate.assert_called_once()
    sqls = [sql for sql, _ in executed]
    delete_index = next(
        i for i, sql in enumerate(sqls) if sql.startswith("DELETE FROM fact_production_forecast")
    )
    insert_index = next(
        i for i, sql in enumerate(sqls) if sql.startswith("INSERT INTO fact_production_forecast ")
    )
    audit_index = next(
        i for i, sql in enumerate(sqls) if sql.startswith("INSERT INTO model_promotion_log")
    )
    assert audit_index < delete_index < insert_index
    insert_sql, insert_params = executed[insert_index]
    assert "s.run_id = %s::uuid" in insert_sql
    assert "s.generation_purpose = 'release_candidate'" in insert_sql
    assert "s.candidate_model_id = %s" in insert_sql
    assert str(RUN_ID) in insert_params
    assert result.source_run_id == RUN_ID
    assert result.production_run_id == PRODUCTION_RUN_ID
    assert result.candidate_checksum == "c" * 64
    assert result.promotion_type == promotion_type


def test_post_copy_checksum_mismatch_rolls_back_transaction():
    conn = MagicMock()
    tx = conn.transaction.return_value
    tx.__enter__.return_value = tx
    tx.__exit__.return_value = False
    cur = conn.cursor.return_value.__enter__.return_value

    def execute(sql, params=None):
        cur.rowcount = 10
        if "INSERT INTO model_promotion_log" in sql:
            cur.fetchone.return_value = (44,)

    cur.execute.side_effect = execute

    with (
        patch(
            "common.services.forecast_promotion._load_manifest",
            return_value=_manifest(row_count=10, dfu_count=2, source_model_count=2),
        ),
        patch("common.services.forecast_promotion._validate_candidate_evidence"),
        patch(
            "common.services.forecast_promotion._validate_incoming_snapshot_roster",
            return_value={"roster_models": 4, "ready_contenders": 3},
        ),
        patch(
            "common.services.forecast_promotion._archive_outgoing_release",
            return_value=(None, None, None),
        ),
        patch("common.services.forecast_promotion.compute_staging_payload_stats") as staging_stats,
        patch(
            "common.services.forecast_promotion.compute_production_payload_stats"
        ) as production_stats,
        patch("common.services.forecast_promotion.uuid.uuid4", return_value=PRODUCTION_RUN_ID),
    ):
        staging_stats.return_value = MagicMock(
            checksum="c" * 64, row_count=10, dfu_count=2, source_model_count=2
        )
        production_stats.return_value = MagicMock(
            checksum="d" * 64, row_count=10, dfu_count=2, source_model_count=2
        )
        with pytest.raises(PromotionConflictError) as exc_info:
            promote_forecast_run(
                conn,
                model_id="champion",
                source_run_id=RUN_ID,
                planning_month=date(2026, 7, 1),
                promoted_by="api",
                notes=None,
                policy={
                    "required_months": 6,
                    "min_coverage_frac": 0.95,
                    "min_ci_coverage_frac": 0.95,
                },
            )

    assert exc_info.value.code == "production_checksum_mismatch"
    assert tx.__exit__.call_args.args[0] is PromotionConflictError


def test_pre_manifest_outgoing_release_is_retired_with_checksum_audit():
    cur = MagicMock()
    production_run_id = UUID("00000000-0000-0000-0000-000000000333")
    cur.fetchone.side_effect = [
        (22, "2026-06", production_run_id, None, 120, 10),
        (120, 10, 120),
    ]
    legacy_stats = MagicMock(
        checksum="d" * 64,
        row_count=120,
        dfu_count=10,
        source_model_count=3,
    )

    with (
        patch(
            "common.services.forecast_promotion.archive_snapshot_in_transaction",
            side_effect=ValueError("missing roster"),
        ),
        patch(
            "common.services.forecast_promotion.compute_production_payload_stats",
            return_value=legacy_stats,
        ),
    ):
        outgoing_id, archive_checksum, report = _archive_outgoing_release(
            cur,
            incoming_planning_month=date(2026, 7, 1),
        )

    assert outgoing_id == 22
    assert archive_checksum is None
    assert report == {
        "promotion_id": 22,
        "plan_version": "2026-06",
        "status": "legacy_retired_unarchived",
        "reason": "pre_manifest_release_without_complete_snapshot_roster",
        "production_run_id": str(production_run_id),
        "row_count": 120,
        "dfu_count": 10,
        "production_checksum": "d" * 64,
    }
    sql_statements = [call.args[0] for call in cur.execute.call_args_list]
    assert any("SAVEPOINT outgoing_forecast_archive" in sql for sql in sql_statements)
    assert any("ROLLBACK TO SAVEPOINT outgoing_forecast_archive" in sql for sql in sql_statements)


def test_modern_outgoing_release_cannot_use_legacy_retirement_path():
    cur = MagicMock()
    cur.fetchone.return_value = (
        23,
        "2026-06",
        PRODUCTION_RUN_ID,
        RUN_ID,
        120,
        10,
    )

    with (
        patch(
            "common.services.forecast_promotion.archive_snapshot_in_transaction",
            side_effect=ValueError("missing roster"),
        ),
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        _archive_outgoing_release(
            cur,
            incoming_planning_month=date(2026, 7, 1),
        )

    assert exc_info.value.code == "outgoing_archive_incomplete"


def test_incoming_release_requires_complete_current_contract_snapshot_roster():
    cur = MagicMock()
    cur.fetchall.return_value = [
        ("champion", "champion", None, None, None),
        ("nhits", "contender", 1, 102, UUID("00000000-0000-0000-0000-000000000121")),
        ("nbeats", "contender", 2, 103, UUID("00000000-0000-0000-0000-000000000122")),
        ("mstl", "contender", 3, 104, UUID("00000000-0000-0000-0000-000000000123")),
    ]

    with patch(
        "common.services.forecast_promotion.validate_ready_snapshot_contender"
    ) as validate:
        _validate_incoming_snapshot_roster(cur, planning_month=date(2026, 7, 1))

    sql, params = cur.execute.call_args.args
    assert "forecast_snapshot_roster" in sql
    assert params == (date(2026, 7, 1),)
    assert validate.call_count == 3


@pytest.mark.parametrize(
    "roster_rows",
    [
        [],
        [
            ("champion", "champion", None, None, None),
            ("nhits", "contender", 1, 102, RUN_ID),
            ("nbeats", "contender", 2, 103, PRODUCTION_RUN_ID),
        ],
        [
            ("champion", "champion", None, None, None),
            ("nhits", "contender", 1, 102, RUN_ID),
            ("nbeats", "contender", 3, 103, PRODUCTION_RUN_ID),
            ("mstl", "contender", 3, 104, UUID("00000000-0000-0000-0000-000000000333")),
        ],
    ],
)
def test_incomplete_or_old_contract_incoming_roster_blocks_publish(roster_rows):
    cur = MagicMock()
    cur.fetchall.return_value = roster_rows

    with pytest.raises(PromotionConflictError) as exc_info:
        _validate_incoming_snapshot_roster(cur, planning_month=date(2026, 7, 1))

    assert exc_info.value.code == "snapshot_roster_not_ready"


def test_stale_ready_contender_blocks_publish():
    cur = MagicMock()
    cur.fetchall.return_value = [
        ("champion", "champion", None, None, None),
        ("nhits", "contender", 1, 102, RUN_ID),
        ("nbeats", "contender", 2, 103, PRODUCTION_RUN_ID),
        ("mstl", "contender", 3, 104, UUID("00000000-0000-0000-0000-000000000333")),
    ]

    with (
        patch(
            "common.services.forecast_promotion.validate_ready_snapshot_contender",
            side_effect=SnapshotContenderStaleError("sales lineage is stale"),
        ),
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        _validate_incoming_snapshot_roster(cur, planning_month=date(2026, 7, 1))

    assert exc_info.value.code == "snapshot_roster_not_ready"


def test_candidate_quality_is_experiment_scoped_and_passes_common_cohort_policy():
    cur = MagicMock()
    cur.fetchone.return_value = (
        6000,
        1200,
        20.0,
        1.0,
        25.0,
        21.0,
        0,
        6200,
        1250,
        6,
        100000.0,
    )

    checks = _candidate_quality_report(
        cur,
        champion_experiment_id=33,
        planning_month=date(2026, 7, 1),
        policy=_quality_policy(),
    )

    sql, params = cur.execute.call_args.args
    assert "f.champion_experiment_id = %s" in sql
    assert "FROM fact_sales_monthly" in sql
    assert "keys.startdate - INTERVAL '12 months'" in sql
    assert "LEFT JOIN sales_by_dfu prior" in sql
    assert "COALESCE(prior.qty, 0)" in sql
    assert sql.count("%s") == len(params)
    assert params[0] == 33
    assert all(check["status"] == "pass" for check in checks)


def test_candidate_quality_blocks_incumbent_regression():
    cur = MagicMock()
    cur.fetchone.return_value = (
        6000,
        1200,
        23.0,
        1.0,
        25.0,
        21.0,
        0,
        6200,
        1250,
        6,
        100000.0,
    )

    with pytest.raises(PromotionConflictError) as exc_info:
        _candidate_quality_report(
            cur,
            champion_experiment_id=33,
            planning_month=date(2026, 7, 1),
            policy=_quality_policy(),
        )

    assert exc_info.value.code == "candidate_quality_failed"


def test_candidate_quality_excludes_external_when_policy_exempts_it():
    cur = MagicMock()
    cur.fetchone.return_value = (
        6200,
        1250,
        20.0,
        1.0,
        25.0,
        None,
        0,
        6200,
        1250,
        6,
        100000.0,
    )
    policy = {**_quality_policy(), "require_external_benchmark": False}

    checks = _candidate_quality_report(
        cur,
        champion_experiment_id=33,
        planning_month=date(2026, 7, 1),
        policy=policy,
    )

    sql, params = cur.execute.call_args.args
    assert "(%s OR f.model_id <> 'external')" in sql
    assert "HAVING COUNT(*) = %s AND COUNT(DISTINCT model_id) = %s" in sql
    assert False in params
    by_id = {check["id"]: check for check in checks}
    assert by_id["delta_vs_external"]["status"] == "pass"
    assert by_id["delta_vs_external"]["threshold"] == "not required"
