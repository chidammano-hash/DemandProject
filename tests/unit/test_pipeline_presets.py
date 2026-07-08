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
        for name in ("data-refresh", "model-refresh", "forecast-publish", "full-refresh"):
            assert name in presets, f"missing preset {name}"

    def test_every_step_job_type_is_registered(self):
        for name, preset in load_pipeline_presets().items():
            for step in preset_steps(preset):
                assert step["job_type"] in JOB_TYPE_REGISTRY, (
                    f"pipeline {name!r} references unregistered job type "
                    f"{step['job_type']!r}"
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
