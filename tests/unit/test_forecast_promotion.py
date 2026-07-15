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
from common.services.customer_forecast_blend_contract import (
    CUSTOMER_BLEND_CONTRACT_VERSION,
    CUSTOMER_BLEND_LINEAGE_METADATA_KEY,
)
from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from common.services.forecast_promotion import (
    ForecastGenerationManifest,
    ForecastStagingResult,
    PromotionConflictError,
    _candidate_coverage,
    _candidate_quality_report,
    _current_release_evidence,
    _lock_active_release_for_replacement,
    _validate_active_model_artifacts,
    _validate_customer_bottom_up_blend,
    _validate_direct_model_lineage,
    _validate_governed_champion_source,
    _validate_tree_model_lineage,
    promote_forecast_run,
    stage_forecast_run,
    validate_generation_manifest,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000111")
PRODUCTION_RUN_ID = UUID("00000000-0000-0000-0000-000000000222")
CUSTOMER_RUN_ID = UUID("00000000-0000-0000-0000-000000000333")
SOURCE_RUN_ID = UUID("00000000-0000-0000-0000-000000000444")
BACKTEST_RUN_ID = UUID("00000000-0000-0000-0000-000000000555")

_BLEND_CONFIG_CHECKSUM = "blend-config-checksum"
_CUSTOMER_CONFIG_CHECKSUM = "customer-config-checksum"
_CUSTOMER_SOURCE_CHECKSUM = "customer-source-checksum"
_CUSTOMER_OUTPUT_CHECKSUM = "customer-output-checksum"
_SOURCE_PRODUCTION_CHECKSUM = "source-production-checksum"
_BACKTEST_CONFIG_CHECKSUM = "backtest-config-checksum"
_BACKTEST_COMPONENT_CHECKSUM = "backtest-component-checksum"
_BLEND_COMPONENT_CHECKSUM = "blend-component-checksum"
_CUSTOMER_DEMAND_BATCH_ID = 91


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


def _customer_blend_lineage(**overrides) -> dict:
    values = {
        "contract_version": CUSTOMER_BLEND_CONTRACT_VERSION,
        "model_id": "customer_bottom_up_blend",
        "config_checksum": _BLEND_CONFIG_CHECKSUM,
        "customer_run_id": str(CUSTOMER_RUN_ID),
        "customer_config_checksum": _CUSTOMER_CONFIG_CHECKSUM,
        "customer_source_checksum": _CUSTOMER_SOURCE_CHECKSUM,
        "source_customer_demand_batch_id": _CUSTOMER_DEMAND_BATCH_ID,
        "customer_output_checksum": _CUSTOMER_OUTPUT_CHECKSUM,
        "customer_output_row_count": 216,
        "customer_output_series_count": 12,
        "source_promotion_id": 24,
        "source_run_id": str(SOURCE_RUN_ID),
        "source_production_run_id": str(PRODUCTION_RUN_ID),
        "source_production_checksum": _SOURCE_PRODUCTION_CHECKSUM,
        "backtest_run_id": str(BACKTEST_RUN_ID),
        "backtest_config_checksum": _BACKTEST_CONFIG_CHECKSUM,
        "backtest_component_checksum": _BACKTEST_COMPONENT_CHECKSUM,
        "backtest_component_rows": 72,
        "component_checksum": _BLEND_COMPONENT_CHECKSUM,
        "row_count": 288,
        "dfu_count": 12,
        "blended_row_count": 216,
        "fallback_row_count": 72,
        "backtest_gate": {"passed": True, "reason": "passed"},
    }
    values.update(overrides)
    return values


def _customer_blend_db_evidence(**overrides) -> tuple:
    values = {
        "source_run_id": SOURCE_RUN_ID,
        "production_run_id": PRODUCTION_RUN_ID,
        "source_production_checksum": _SOURCE_PRODUCTION_CHECKSUM,
        "customer_config_checksum": _CUSTOMER_CONFIG_CHECKSUM,
        "customer_source_checksum": _CUSTOMER_SOURCE_CHECKSUM,
        "customer_output_row_count": 216,
        "customer_output_series_count": 12,
        "backtest_config_checksum": _BACKTEST_CONFIG_CHECKSUM,
        "backtest_component_checksum": _BACKTEST_COMPONENT_CHECKSUM,
        "backtest_component_rows": 72,
        "gate_passed": True,
        "gate_reason": "passed",
        "source_customer_demand_batch_id": _CUSTOMER_DEMAND_BATCH_ID,
        "current_customer_demand_batch_id": _CUSTOMER_DEMAND_BATCH_ID,
        "profile_customer_demand_batch_id": _CUSTOMER_DEMAND_BATCH_ID,
        "active_customer_demand_loads": 0,
    }
    values.update(overrides)
    return (
        values["source_run_id"],
        values["production_run_id"],
        values["source_production_checksum"],
        values["customer_config_checksum"],
        values["customer_source_checksum"],
        values["customer_output_row_count"],
        values["customer_output_series_count"],
        values["backtest_config_checksum"],
        values["backtest_component_checksum"],
        values["backtest_component_rows"],
        values["gate_passed"],
        values["gate_reason"],
        values["source_customer_demand_batch_id"],
        values["current_customer_demand_batch_id"],
        values["profile_customer_demand_batch_id"],
        values["active_customer_demand_loads"],
    )


@pytest.fixture
def customer_blend_dependencies(monkeypatch):
    blend_settings = SimpleNamespace(
        model_id="customer_bottom_up_blend",
        promotion_enabled=True,
        promotion_reason="backtest_accuracy_gate",
    )
    customer_settings = {"model_id": "croston", "model_params": {"variant": "sba"}}
    backtest_settings = SimpleNamespace(lookback_months=6)
    component_stats = MagicMock(return_value=(_BLEND_COMPONENT_CHECKSUM, 288, 12, 216, 72))
    customer_output_stats = MagicMock(
        return_value=SimpleNamespace(
            checksum=_CUSTOMER_OUTPUT_CHECKSUM,
            row_count=216,
            series_count=12,
        )
    )
    backtest_component_stats = MagicMock(return_value=(_BACKTEST_COMPONENT_CHECKSUM, 72))
    source_production_stats = MagicMock(
        return_value=SimpleNamespace(checksum=_SOURCE_PRODUCTION_CHECKSUM)
    )
    component_lineage = MagicMock(
        return_value=SimpleNamespace(
            customer_run_id=CUSTOMER_RUN_ID,
            backtest_run_id=BACKTEST_RUN_ID,
            source_promotion_id=24,
            source_production_run_id=PRODUCTION_RUN_ID,
        )
    )

    monkeypatch.setattr(
        "common.services.forecast_promotion.validate_blend_settings",
        lambda _config: blend_settings,
    )
    monkeypatch.setattr(
        "common.services.forecast_promotion.customer_blend_config_checksum",
        lambda _settings: _BLEND_CONFIG_CHECKSUM,
    )
    monkeypatch.setattr(
        "common.services.forecast_promotion.compute_customer_blend_component_stats",
        component_stats,
    )
    monkeypatch.setattr(
        "common.services.forecast_promotion.compute_customer_forecast_output_stats",
        customer_output_stats,
    )
    monkeypatch.setattr(
        "common.services.forecast_promotion.compute_production_payload_stats",
        source_production_stats,
    )
    monkeypatch.setattr(
        "common.services.forecast_promotion.load_customer_blend_component_lineage",
        component_lineage,
    )
    monkeypatch.setattr(
        "common.services.customer_forecast.get_customer_forecast_settings",
        lambda: customer_settings,
    )
    monkeypatch.setattr(
        "common.services.customer_forecast.customer_forecast_config_checksum",
        lambda _settings: _CUSTOMER_CONFIG_CHECKSUM,
    )
    monkeypatch.setattr(
        "common.services.customer_forecast_backtest.get_customer_backtest_settings",
        lambda: backtest_settings,
    )
    monkeypatch.setattr(
        "common.services.customer_forecast_backtest.customer_backtest_config_checksum",
        lambda *_args: _BACKTEST_CONFIG_CHECKSUM,
    )
    monkeypatch.setattr(
        "common.services.customer_forecast_backtest.compute_customer_backtest_component_stats",
        backtest_component_stats,
    )
    component_stats.customer_output_stats = customer_output_stats
    component_stats.backtest_component_stats = backtest_component_stats
    component_stats.source_production_stats = source_production_stats
    component_stats.component_lineage = component_lineage
    return component_stats


def test_customer_bottom_up_blend_promotion_accepts_exact_lineage(
    customer_blend_dependencies,
):
    cur = MagicMock()
    cur.fetchone.return_value = _customer_blend_db_evidence()
    backtest_gate = {"passed": True, "reason": "passed"}
    manifest = _manifest(
        metadata={
            CUSTOMER_BLEND_LINEAGE_METADATA_KEY: _customer_blend_lineage(
                backtest_gate=backtest_gate
            )
        }
    )

    result = _validate_customer_bottom_up_blend(
        cur,
        manifest=manifest,
        pipeline_config={"customer_forecast": {}},
    )

    assert result == {
        "customer_run_id": str(CUSTOMER_RUN_ID),
        "backtest_run_id": str(BACKTEST_RUN_ID),
        "source_promotion_id": 24,
        "source_customer_demand_batch_id": _CUSTOMER_DEMAND_BATCH_ID,
        "customer_output_checksum": _CUSTOMER_OUTPUT_CHECKSUM,
        "backtest_component_checksum": _BACKTEST_COMPONENT_CHECKSUM,
        "component_checksum": _BLEND_COMPONENT_CHECKSUM,
        "backtest_gate": backtest_gate,
    }
    assert cur.execute.call_args.args[1] == (
        str(CUSTOMER_RUN_ID),
        str(BACKTEST_RUN_ID),
        24,
    )
    sql = cur.execute.call_args.args[0]
    assert "promotion.is_active = TRUE" in sql
    assert "FOR SHARE OF promotion" in sql
    assert "customer.run_status = 'completed'" in sql
    assert "backtest.run_status = 'completed'" in sql
    assert "customer.source_customer_demand_batch_id" in sql
    assert "domain = 'customer_demand'" in sql
    assert "customer_demand_profile_refresh_state" in sql
    assert "status = 'running'" in sql
    customer_blend_dependencies.customer_output_stats.assert_called_once_with(
        cur, str(CUSTOMER_RUN_ID)
    )
    customer_blend_dependencies.backtest_component_stats.assert_called_once_with(
        cur, str(BACKTEST_RUN_ID)
    )
    customer_blend_dependencies.source_production_stats.assert_called_once_with(
        cur, str(PRODUCTION_RUN_ID)
    )
    customer_blend_dependencies.component_lineage.assert_called_once_with(cur, RUN_ID)
    customer_blend_dependencies.assert_called_once_with(cur, RUN_ID)


def test_customer_bottom_up_blend_promotion_rejects_required_missing_lineage() -> None:
    with pytest.raises(PromotionConflictError) as exc_info:
        _validate_customer_bottom_up_blend(
            MagicMock(),
            manifest=_manifest(metadata={}),
            pipeline_config={"customer_forecast": {}},
            require_lineage=True,
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"


def test_customer_bottom_up_blend_promotion_rejects_current_customer_config_drift(
    customer_blend_dependencies,
    monkeypatch,
):
    cur = MagicMock()
    monkeypatch.setattr(
        "common.services.customer_forecast.customer_forecast_config_checksum",
        lambda _settings: "changed-customer-config",
    )

    with pytest.raises(PromotionConflictError) as exc_info:
        _validate_customer_bottom_up_blend(
            cur,
            manifest=_manifest(
                metadata={CUSTOMER_BLEND_LINEAGE_METADATA_KEY: _customer_blend_lineage()}
            ),
            pipeline_config={"customer_forecast": {}},
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"
    cur.execute.assert_not_called()
    customer_blend_dependencies.assert_not_called()


def test_customer_bottom_up_blend_promotion_rejects_current_backtest_config_drift(
    customer_blend_dependencies,
    monkeypatch,
):
    cur = MagicMock()
    monkeypatch.setattr(
        "common.services.customer_forecast_backtest.customer_backtest_config_checksum",
        lambda *_args: "changed-backtest-config",
    )

    with pytest.raises(PromotionConflictError) as exc_info:
        _validate_customer_bottom_up_blend(
            cur,
            manifest=_manifest(
                metadata={CUSTOMER_BLEND_LINEAGE_METADATA_KEY: _customer_blend_lineage()}
            ),
            pipeline_config={"customer_forecast": {}},
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"
    cur.execute.assert_not_called()
    customer_blend_dependencies.assert_not_called()


@pytest.mark.parametrize(
    "evidence_overrides",
    [
        pytest.param(
            {"source_run_id": UUID("00000000-0000-0000-0000-000000000999")},
            id="source-run",
        ),
        pytest.param(
            {"backtest_config_checksum": "changed-backtest-config"},
            id="backtest-config",
        ),
        pytest.param(
            {"backtest_component_checksum": "changed-backtest-components"},
            id="backtest-components",
        ),
        pytest.param({"customer_output_row_count": 215}, id="customer-row-count"),
        pytest.param({"backtest_component_rows": 71}, id="backtest-row-count"),
        pytest.param({"gate_passed": False}, id="failed-backtest-gate"),
        pytest.param(
            {"current_customer_demand_batch_id": _CUSTOMER_DEMAND_BATCH_ID + 1},
            id="customer-demand-batch",
        ),
        pytest.param(
            {"profile_customer_demand_batch_id": _CUSTOMER_DEMAND_BATCH_ID + 1},
            id="customer-demand-profile-batch",
        ),
        pytest.param(
            {"active_customer_demand_loads": 1},
            id="active-customer-demand-load",
        ),
    ],
)
def test_customer_bottom_up_blend_promotion_rejects_stale_persisted_evidence(
    customer_blend_dependencies,
    evidence_overrides,
):
    cur = MagicMock()
    cur.fetchone.return_value = _customer_blend_db_evidence(**evidence_overrides)

    with pytest.raises(PromotionConflictError) as exc_info:
        _validate_customer_bottom_up_blend(
            cur,
            manifest=_manifest(
                metadata={CUSTOMER_BLEND_LINEAGE_METADATA_KEY: _customer_blend_lineage()}
            ),
            pipeline_config={"customer_forecast": {}},
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"
    customer_blend_dependencies.assert_not_called()


@pytest.mark.parametrize(
    ("stats_name", "changed_stats"),
    [
        pytest.param(
            "customer_output_stats",
            SimpleNamespace(
                checksum="changed-customer-output",
                row_count=216,
                series_count=12,
            ),
            id="customer-output",
        ),
        pytest.param(
            "backtest_component_stats",
            ("changed-backtest-components", 72),
            id="backtest-components",
        ),
        pytest.param(
            "source_production_stats",
            SimpleNamespace(checksum="changed-source-production"),
            id="source-production",
        ),
    ],
)
def test_customer_bottom_up_blend_promotion_recomputes_source_payloads(
    customer_blend_dependencies,
    stats_name,
    changed_stats,
):
    cur = MagicMock()
    cur.fetchone.return_value = _customer_blend_db_evidence()
    getattr(customer_blend_dependencies, stats_name).return_value = changed_stats

    with pytest.raises(PromotionConflictError) as exc_info:
        _validate_customer_bottom_up_blend(
            cur,
            manifest=_manifest(
                metadata={CUSTOMER_BLEND_LINEAGE_METADATA_KEY: _customer_blend_lineage()}
            ),
            pipeline_config={"customer_forecast": {}},
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"
    customer_blend_dependencies.assert_not_called()


def test_customer_bottom_up_blend_promotion_rejects_component_checksum_drift(
    customer_blend_dependencies,
):
    cur = MagicMock()
    cur.fetchone.return_value = _customer_blend_db_evidence()
    customer_blend_dependencies.return_value = (
        "changed-blend-components",
        288,
        12,
        216,
        72,
    )

    with pytest.raises(PromotionConflictError) as exc_info:
        _validate_customer_bottom_up_blend(
            cur,
            manifest=_manifest(
                metadata={CUSTOMER_BLEND_LINEAGE_METADATA_KEY: _customer_blend_lineage()}
            ),
            pipeline_config={"customer_forecast": {}},
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"
    customer_blend_dependencies.assert_called_once_with(cur, RUN_ID)


def test_customer_bottom_up_blend_promotion_rejects_component_lineage_drift(
    customer_blend_dependencies,
):
    cur = MagicMock()
    cur.fetchone.return_value = _customer_blend_db_evidence()
    customer_blend_dependencies.component_lineage.return_value = SimpleNamespace(
        customer_run_id=UUID("00000000-0000-0000-0000-000000000999"),
        backtest_run_id=BACKTEST_RUN_ID,
        source_promotion_id=24,
        source_production_run_id=PRODUCTION_RUN_ID,
    )

    with pytest.raises(PromotionConflictError) as exc_info:
        _validate_customer_bottom_up_blend(
            cur,
            manifest=_manifest(
                metadata={CUSTOMER_BLEND_LINEAGE_METADATA_KEY: _customer_blend_lineage()}
            ),
            pipeline_config={"customer_forecast": {}},
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"
    customer_blend_dependencies.assert_not_called()


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
        ({"generation_purpose": "shadow_candidate"}, "candidate_run_not_promotable"),
        ({"run_status": "generating"}, "candidate_run_not_promotable"),
        ({"promotion_eligible": False}, "candidate_not_staged"),
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
        patch("common.services.forecast_promotion.read_active_neural_artifact_ref") as read_active,
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


def test_champion_routed_customer_blend_holds_demand_lock_through_promotion() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (True,)
    tx = conn.transaction.return_value
    events: list[str] = []

    def execute(sql, *_args):
        if "pg_advisory_lock_shared" in sql:
            events.append("lineage_lock")
        elif "pg_advisory_unlock_shared" in sql:
            events.append("lineage_unlock")
        elif "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE" in sql:
            events.append("snapshot")

    cur.execute.side_effect = execute
    conn.commit.side_effect = lambda: events.append("commit")
    conn.rollback.side_effect = lambda: events.append("rollback")
    tx.__enter__.side_effect = lambda: events.append("transaction_enter") or tx
    tx.__exit__.side_effect = lambda *_args: events.append("transaction_exit") or False

    with (
        patch(
            "common.services.forecast_promotion._load_manifest",
            side_effect=PromotionConflictError(
                "candidate_run_not_found",
                "The selected forecast generation run does not exist.",
            ),
        ),
        pytest.raises(PromotionConflictError),
    ):
        promote_forecast_run(
            conn,
            # Blend manifests deliberately use the champion candidate identity,
            # so lock selection must inspect immutable run lineage rather than
            # relying on this route model id.
            model_id="champion",
            source_run_id=RUN_ID,
            planning_month=date(2026, 7, 1),
            promoted_by="tester",
            notes=None,
            policy={"required_months": 6},
        )

    assert events.index("lineage_lock") < events.index("transaction_enter")
    assert events.index("transaction_enter") < events.index("snapshot")
    assert events.index("transaction_exit") < events.index("lineage_unlock")


def test_generated_candidate_must_be_staged_before_production_promotion() -> None:
    manifest = _manifest(promotion_eligible=False)

    with pytest.raises(PromotionConflictError) as exc_info:
        validate_generation_manifest(
            manifest,
            model_id="champion",
            planning_month=date(2026, 7, 1),
            required_months=6,
        )

    assert exc_info.value.code == "candidate_not_staged"


def test_stage_forecast_run_approves_exact_generated_candidate() -> None:
    conn = MagicMock()
    tx = conn.transaction.return_value
    tx.__enter__.return_value = tx
    tx.__exit__.return_value = False
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 1
    stats = MagicMock(checksum="c" * 64, row_count=120, dfu_count=10, source_model_count=1)

    with (
        patch(
            "common.services.forecast_promotion._load_manifest",
            return_value=_manifest(
                requested_model_id="mstl",
                source_model_count=1,
                promotion_eligible=False,
            ),
        ),
        patch(
            "common.services.forecast_promotion.compute_staging_payload_stats",
            return_value=stats,
        ),
    ):
        result = stage_forecast_run(
            conn,
            model_id="mstl",
            source_run_id=RUN_ID,
            planning_month=date(2026, 7, 1),
        )

    assert result == ForecastStagingResult(
        model_id="mstl",
        source_run_id=RUN_ID,
        status="staged",
        rows_staged=120,
        dfu_count=10,
        candidate_checksum="c" * 64,
    )
    stage_update = next(
        call
        for call in cur.execute.call_args_list
        if "SET promotion_eligible = TRUE" in call.args[0]
    )
    assert "promotion_eligible = FALSE" in stage_update.args[0]
    assert stage_update.args[1] == (str(RUN_ID),)


def test_stage_customer_blend_requires_current_backtest_lineage() -> None:
    conn = MagicMock()
    tx = conn.transaction.return_value
    tx.__enter__.return_value = tx
    tx.__exit__.return_value = False
    cur = conn.cursor.return_value.__enter__.return_value
    manifest = _manifest(
        promotion_eligible=False,
        metadata={
            **_manifest().metadata,
            CUSTOMER_BLEND_LINEAGE_METADATA_KEY: _customer_blend_lineage(),
        },
    )
    stats = MagicMock(
        checksum=manifest.artifact_checksum,
        row_count=manifest.row_count,
        dfu_count=manifest.dfu_count,
        source_model_count=manifest.source_model_count,
    )

    with (
        patch(
            "common.services.forecast_promotion._load_manifest",
            return_value=manifest,
        ),
        patch(
            "common.services.forecast_promotion.compute_staging_payload_stats",
            return_value=stats,
        ),
        patch(
            "common.services.forecast_promotion.load_forecast_pipeline_config",
            return_value={"customer_forecast": {}},
        ),
        patch(
            "common.services.forecast_promotion._validate_customer_bottom_up_blend",
            side_effect=PromotionConflictError(
                "customer_blend_lineage_mismatch",
                "The customer blend backtest is stale.",
            ),
        ) as validate_blend,
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        stage_forecast_run(
            conn,
            model_id="champion",
            source_run_id=RUN_ID,
            planning_month=date(2026, 7, 1),
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"
    validate_blend.assert_called_once()
    assert not any(
        "SET promotion_eligible = TRUE" in call.args[0] for call in cur.execute.call_args_list
    )


def test_stage_detects_customer_blend_payload_when_manifest_lineage_is_missing() -> None:
    conn = MagicMock()
    tx = conn.transaction.return_value
    tx.__enter__.return_value = tx
    tx.__exit__.return_value = False
    manifest = _manifest(promotion_eligible=False)
    stats = MagicMock(
        checksum=manifest.artifact_checksum,
        row_count=manifest.row_count,
        dfu_count=manifest.dfu_count,
        source_model_count=manifest.source_model_count,
    )

    with (
        patch("common.services.forecast_promotion._load_manifest", return_value=manifest),
        patch(
            "common.services.forecast_promotion.compute_staging_payload_stats",
            return_value=stats,
        ),
        patch(
            "common.services.forecast_promotion._is_customer_blend_payload",
            return_value=True,
        ),
        patch(
            "common.services.forecast_promotion._validate_customer_bottom_up_blend",
            side_effect=PromotionConflictError(
                "customer_blend_lineage_mismatch",
                "The customer blend lineage is missing.",
            ),
        ) as validate_blend,
        pytest.raises(PromotionConflictError) as exc_info,
    ):
        stage_forecast_run(
            conn,
            model_id="champion",
            source_run_id=RUN_ID,
            planning_month=date(2026, 7, 1),
        )

    assert exc_info.value.code == "customer_blend_lineage_mismatch"
    assert validate_blend.call_args.kwargs["require_lineage"] is True


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
        cur.rowcount = (
            1
            if "UPDATE forecast_generation_run" in sql or "UPDATE model_promotion_log" in sql
            else 120
        )

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
            "common.services.forecast_promotion._lock_active_release_for_replacement",
            return_value=(
                23,
                {
                    "promotion_id": 23,
                    "plan_version": "2026-07",
                    "model_id": "champion",
                    "status": "replaced",
                },
            ),
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
    assert any(
        sql.startswith("UPDATE model_promotion_log SET is_active = FALSE") and params == (23,)
        for sql, params in executed
    )
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
            "common.services.forecast_promotion._lock_active_release_for_replacement",
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


def test_active_release_lock_allows_same_month_replacement():
    cur = MagicMock()
    cur.fetchone.return_value = (23, "2026-07", "champion", RUN_ID, PRODUCTION_RUN_ID)

    outgoing_id, report = _lock_active_release_for_replacement(cur)

    assert outgoing_id == 23
    assert report == {
        "promotion_id": 23,
        "plan_version": "2026-07",
        "model_id": "champion",
        "source_run_id": str(RUN_ID),
        "production_run_id": str(PRODUCTION_RUN_ID),
        "status": "replaced",
    }
    assert "FOR UPDATE" in cur.execute.call_args.args[0]


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
