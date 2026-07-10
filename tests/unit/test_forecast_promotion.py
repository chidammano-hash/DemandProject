"""Tests for immutable, transaction-safe forecast promotion."""

from datetime import date
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from common.services.forecast_promotion import (
    ForecastGenerationManifest,
    PromotionConflictError,
    _candidate_quality_report,
    promote_forecast_run,
    validate_generation_manifest,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000111")
PRODUCTION_RUN_ID = UUID("00000000-0000-0000-0000-000000000222")


def _quality_policy() -> dict:
    return {
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


def _manifest(**overrides) -> ForecastGenerationManifest:
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
    }
    values.update(overrides)
    return ForecastGenerationManifest(**values)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"generation_purpose": "snapshot_contender"}, "candidate_run_not_promotable"),
        ({"run_status": "generating"}, "candidate_run_not_promotable"),
        ({"promotion_eligible": False}, "candidate_run_not_promotable"),
        ({"requested_model_id": "lgbm_cluster"}, "candidate_lineage_mismatch"),
        ({"forecast_month_generated": date(2026, 6, 1)}, "stale_candidate_evidence"),
        ({"horizon_months": 5}, "candidate_gate_failed"),
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


def test_promotion_preflights_before_mutation_and_scopes_copy_to_source_run():
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
        patch("common.services.forecast_promotion._load_manifest", return_value=_manifest()),
        patch("common.services.forecast_promotion._validate_candidate_evidence") as validate,
        patch(
            "common.services.forecast_promotion._archive_outgoing_release",
            return_value=(None, None),
        ),
        patch("common.services.forecast_promotion.compute_staging_payload_stats") as staging_stats,
        patch(
            "common.services.forecast_promotion.compute_production_payload_stats"
        ) as production_stats,
        patch("common.services.forecast_promotion.uuid.uuid4", return_value=PRODUCTION_RUN_ID),
    ):
        stats = MagicMock(checksum="c" * 64, row_count=120, dfu_count=10, source_model_count=3)
        staging_stats.return_value = stats
        production_stats.return_value = stats
        result = promote_forecast_run(
            conn,
            model_id="champion",
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
            "common.services.forecast_promotion._archive_outgoing_release",
            return_value=(None, None),
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
