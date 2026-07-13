"""Tests for named pipeline presets (common/services/pipeline_presets.py).

The registry check is the mechanical guard: a preset step referencing a job
type that does not exist in JOB_TYPE_REGISTRY fails the suite, not production.
"""

from __future__ import annotations

import pytest

from common.services.job_registry import JOB_TYPE_REGISTRY
from common.services.pipeline_presets import (
    get_pipeline_preset,
    load_pipeline_presets,
    preset_steps,
)


class TestPresetConfig:
    def test_expected_presets_exist(self):
        presets = load_pipeline_presets()
        for name in (
            "data-refresh",
            "clustering-refresh",
            "model-refresh",
            "champion-refresh",
            "forecast-publish",
            "forecast-snapshot-bundle",
            "period-roll",
            "full-refresh",
            "inventory-refresh",
        ):
            assert name in presets, f"missing preset {name}"

    def test_snapshot_bundle_selects_archives_then_cleans(self):
        steps = preset_steps(get_pipeline_preset("forecast-snapshot-bundle"))
        assert [step["job_type"] for step in steps] == [
            "prepare_forecast_snapshot_contenders",
            "archive_forecast_snapshot",
            "cleanup_forecast_staging",
        ]
        assert steps[-1]["params"] == {"dry_run": False}

    def test_period_roll_scores_prior_month_then_archives_current_month(self):
        steps = preset_steps(get_pipeline_preset("period-roll"))
        assert [step["job_type"] for step in steps] == [
            "refresh_forecast_snapshot_kpis",
            "prepare_forecast_snapshot_contenders",
            "archive_forecast_snapshot",
            "cleanup_forecast_staging",
        ]
        assert steps[-1]["params"] == {"dry_run": False}

    def test_generation_pipelines_leave_snapshot_work_to_period_roll(self):
        for name in ("forecast-publish", "full-refresh"):
            steps = preset_steps(get_pipeline_preset(name))
            job_types = [step["job_type"] for step in steps]
            assert "prepare_forecast_snapshot_contenders" not in job_types
            assert "archive_forecast_snapshot" not in job_types
            assert "cleanup_forecast_staging" not in job_types

    def test_model_refresh_stops_after_five_governed_backtests(self):
        steps = preset_steps(get_pipeline_preset("model-refresh"))
        job_types = [step["job_type"] for step in steps]
        assert "champion_select" not in job_types
        assert "governed_champion_refresh" not in job_types
        governed_backtests = [
            step for step in steps if step["job_type"].startswith("backtest_")
        ]
        assert len(governed_backtests) == 5
        assert all(step["params"] == {"governed": True} for step in governed_backtests)

    def test_champion_refresh_is_a_separate_atomic_assignment_pipeline(self):
        steps = preset_steps(get_pipeline_preset("champion-refresh"))
        assert [step["job_type"] for step in steps] == ["governed_champion_refresh"]

    def test_full_refresh_runs_model_evidence_before_champion_assignment(self):
        steps = preset_steps(get_pipeline_preset("full-refresh"))
        job_types = [step["job_type"] for step in steps]
        champion_index = job_types.index("governed_champion_refresh")
        governed_backtests = [
            (index, step)
            for index, step in enumerate(steps)
            if step["job_type"].startswith("backtest_")
        ]
        assert len(governed_backtests) == 5
        assert all(step["params"] == {"governed": True} for _, step in governed_backtests)
        assert all(index < champion_index for index, _ in governed_backtests)

    def test_full_refresh_promotes_clusters_before_tuning(self):
        steps = preset_steps(get_pipeline_preset("full-refresh"))
        job_types = [step["job_type"] for step in steps]
        cluster_index = job_types.index("cluster_pipeline")
        tuning_index = job_types.index("tune_stale_clusters")

        assert steps[cluster_index]["params"] == {"auto_promote": True}
        assert cluster_index < tuning_index

    def test_every_step_job_type_is_registered(self):
        for name, preset in load_pipeline_presets().items():
            for step in preset_steps(preset):
                assert step["job_type"] in JOB_TYPE_REGISTRY, (
                    f"pipeline {name!r} references unregistered job type {step['job_type']!r}"
                )

    def test_every_preset_has_description_and_steps(self):
        for name, preset in load_pipeline_presets().items():
            assert preset.get("description"), f"{name} has no description"
            assert preset_steps(preset), f"{name} has no steps"

    def test_steps_normalized_shape(self):
        steps = preset_steps(get_pipeline_preset("data-refresh"))
        first = steps[0]
        assert first["job_type"] == "etl_pipeline"
        assert first["params"] == {"mode": "refresh", "parallel": True}
        assert set(first) == {"job_type", "params", "label"}

    def test_unknown_preset_raises_keyerror_with_known_names(self):
        with pytest.raises(KeyError, match="data-refresh"):
            get_pipeline_preset("nope")

    def test_malformed_step_raises(self):
        with pytest.raises(ValueError, match="Malformed"):
            preset_steps({"steps": [{"params": {}}]})
