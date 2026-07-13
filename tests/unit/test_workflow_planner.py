"""Tests for the cross-domain AI workflow planner."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

from common.ai.workflow_planner import planner
from common.ai.workflow_planner.planner import (
    WorkflowAnswer,
    WorkflowDecision,
    WorkflowState,
    _ai_decision,
    _apply_answer_guardrails,
    _fetch_state,
    _verification_failure,
    run_workflow_planner,
    system_recommendations,
)


def test_fetch_state_scopes_stale_tuning_to_current_cluster_labels(monkeypatch):
    pool = MagicMock()
    cursor = pool.connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (0, None, None, 0, None, None, None, None, None, 0, 0, 0)
    monkeypatch.setattr(planner, "_planning_month", lambda: date(2026, 7, 1))

    _fetch_state(pool)

    state_sql = str(cursor.execute.call_args_list[1].args[0])
    assert "current_sku_cluster_assignment" in state_sql
    assert "assignment.ml_cluster = tuning.cluster_name" in state_sql


def _state(**overrides) -> WorkflowState:
    base = {
        "planning_month": date(2026, 7, 1),
        "active_jobs": [],
        "clustered_skus": 100,
        "latest_feature_refresh": datetime(2026, 6, 30, tzinfo=UTC),
        "latest_cluster_promotion": datetime(2026, 7, 1, tzinfo=UTC),
        "stale_tuning_profiles": 0,
        "latest_sales_load": datetime(2026, 7, 1, tzinfo=UTC),
        "latest_champion_promotion": datetime(2026, 7, 2, tzinfo=UTC),
        "latest_inventory_refresh": datetime(2026, 7, 3, tzinfo=UTC),
        "active_production_month": date(2026, 7, 1),
        "active_production_promoted_at": datetime(2026, 7, 2, tzinfo=UTC),
        "planning_month_production_rows": 100,
        "planning_month_roster_models": 4,
        "planning_month_snapshot_rows": 100,
    }
    base.update(overrides)
    return WorkflowState(**base)


def _scan(*changed_domains: str) -> dict:
    return {
        "changes": [
            {"domain": domain, "changed": True, "reason": "hash mismatch"}
            for domain in changed_domains
        ],
        "proposed_chain": [],
    }


def test_input_changes_put_data_refresh_first():
    recommendations = system_recommendations(_scan("sales"), _state())

    assert recommendations[0].pipeline_name == "data-refresh"
    assert "sales" in recommendations[0].reason


def test_missing_cluster_assignments_recommend_clustering_refresh():
    recommendations = system_recommendations(_scan(), _state(clustered_skus=0))

    assert [item.pipeline_name for item in recommendations] == ["clustering-refresh"]


def test_features_newer_than_cluster_promotion_recommend_clustering_refresh():
    recommendations = system_recommendations(
        _scan(),
        _state(
            latest_feature_refresh=datetime(2026, 7, 2, tzinfo=UTC),
            latest_cluster_promotion=datetime(2026, 7, 1, tzinfo=UTC),
        ),
    )

    assert [item.pipeline_name for item in recommendations] == ["clustering-refresh"]
    assert "newer" in recommendations[0].reason


def test_stale_sales_recommend_model_refresh_before_forecast_publish():
    recommendations = system_recommendations(
        _scan(),
        _state(
            latest_sales_load=datetime(2026, 7, 3, tzinfo=UTC),
            active_production_month=date(2026, 6, 1),
            planning_month_production_rows=0,
        ),
    )

    assert [item.pipeline_name for item in recommendations][:3] == [
        "model-refresh",
        "champion-refresh",
        "forecast-publish",
    ]


def test_ready_release_with_missing_archive_recommends_snapshot_bundle():
    recommendations = system_recommendations(
        _scan(),
        _state(planning_month_snapshot_rows=0),
    )

    assert [item.pipeline_name for item in recommendations] == [
        "forecast-snapshot-bundle",
    ]


def test_inventory_outputs_older_than_champion_recommend_inventory_refresh():
    recommendations = system_recommendations(
        _scan(),
        _state(latest_inventory_refresh=datetime(2026, 7, 1, tzinfo=UTC)),
    )

    assert [item.pipeline_name for item in recommendations] == ["inventory-refresh"]


def test_all_current_returns_no_work():
    assert system_recommendations(_scan(), _state()) == []


def test_wait_answer_blocks_execution_until_active_jobs_finish():
    recommendations = system_recommendations(_scan("sales"), _state())

    guarded = _apply_answer_guardrails(
        recommendations,
        [WorkflowAnswer(question_id="active_job_handling", answer="Wait for active jobs")],
    )

    assert "Wait for active jobs" in guarded[0].blockers[-1]


def test_queue_answer_keeps_recommendation_executable():
    recommendations = system_recommendations(_scan("sales"), _state())

    guarded = _apply_answer_guardrails(
        recommendations,
        [WorkflowAnswer(question_id="active_job_handling", answer="Queue after active jobs")],
    )

    assert guarded == recommendations


def test_usage_limit_has_actionable_safe_failure_flag():
    risk_flag, explanation = _verification_failure(
        RuntimeError("You've hit your usage limit"),
    )

    assert risk_flag == "ai_usage_limit"
    assert "system-safe plan" in explanation


def test_ai_cannot_omit_a_system_required_workflow(monkeypatch):
    stale_state = _state(
        latest_sales_load=datetime(2026, 7, 3, tzinfo=UTC),
        active_production_month=date(2026, 6, 1),
        planning_month_production_rows=0,
    )
    presets = {
        "model-refresh": {
            "description": "Refresh models",
            "steps": [{"job_type": "run_backtest", "params": {}}],
        },
        "champion-refresh": {
            "description": "Assign champion",
            "steps": [{"job_type": "governed_champion_refresh", "params": {}}],
        },
        "forecast-publish": {
            "description": "Publish forecast",
            "steps": [{"job_type": "generate_production_forecast", "params": {}}],
        },
    }
    monkeypatch.setattr(planner, "scan_input_dir", lambda _pool: _scan())
    monkeypatch.setattr(planner, "_fetch_state", lambda _pool: stale_state)
    monkeypatch.setattr(planner, "load_pipeline_presets", lambda: presets)
    monkeypatch.setattr(
        planner,
        "_ai_decision",
        lambda *_args: WorkflowDecision(
            status="planned",
            confidence=0.9,
            explanation="Publish the forecast.",
            recommended_pipeline_names=["forecast-publish"],
        ),
    )

    result = run_workflow_planner(object())

    assert [item["pipeline_name"] for item in result["recommendations"]] == [
        "model-refresh",
        "champion-refresh",
        "forecast-publish",
    ]


def test_production_runtime_uses_openai_client(monkeypatch):
    class _Response:
        parsed = {
            "status": "planned",
            "confidence": 0.95,
            "explanation": "System sequence verified.",
            "risk_flags": [],
            "questions": [],
            "recommended_pipeline_names": [],
        }

    class _Client:
        def chat(self, *_args, **_kwargs):
            return _Response()

    monkeypatch.setenv("INTEGRATION_SCAN_AI_RUNTIME", "openai")
    monkeypatch.setattr(planner, "build_from_config", lambda _cfg: _Client())

    decision = _ai_decision(_scan(), _state(), [], [])

    assert decision.confidence == 0.95
    assert decision.explanation == "System sequence verified."
