"""Period Roll job tests."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_refresh_snapshot_kpis_refreshes_only_governed_snapshot_view():
    from common.services.job_state import _run_refresh_forecast_snapshot_kpis

    expected = {"refreshed": ["agg_accuracy_snapshot"], "failed": [], "missing": []}
    with (
        patch(
            "common.core.mv_refresh.refresh_materialized_views", return_value=expected
        ) as refresh,
        patch("common.services.cache.get_cache") as get_cache,
    ):
        result = _run_refresh_forecast_snapshot_kpis({})

    assert result == expected
    refresh.assert_called_once()
    assert refresh.call_args.args[0] == ["agg_accuracy_snapshot"]
    get_cache.return_value.invalidate.assert_called_once_with("ds:fva_snapshot*")


def test_period_roll_executes_canonical_preset_in_order():
    from common.services.job_state import _run_period_roll

    calls: list[str] = []

    def runner(name: str):
        def _run(params, progress_cb=None, cancel_event=None, job_id=None):
            calls.append(name)
            if progress_cb:
                progress_cb(msg=name)
            return {"output_log": name}

        return _run

    registry = {
        name: SimpleNamespace(callable=runner(name))
        for name in (
            "refresh_forecast_snapshot_kpis",
            "prepare_forecast_snapshot_contenders",
            "archive_forecast_snapshot",
            "cleanup_forecast_staging",
        )
    }
    steps = [{"job_type": name, "params": {}} for name in registry]
    with (
        patch("common.services.pipeline_presets.get_pipeline_preset", return_value={}),
        patch("common.services.pipeline_presets.preset_steps", return_value=steps),
        patch.dict("common.services.job_registry.JOB_TYPE_REGISTRY", registry, clear=True),
    ):
        result = _run_period_roll({}, progress_cb=lambda **_: None)

    assert calls == list(registry)
    assert result["steps_completed"] == 4


def test_period_roll_honors_cancellation_before_starting_a_step():
    from common.services.job_state import JobCancelledError, _run_period_roll

    cancelled = Event()
    cancelled.set()
    with (
        patch("common.services.pipeline_presets.get_pipeline_preset", return_value={}),
        patch(
            "common.services.pipeline_presets.preset_steps",
            return_value=[{"job_type": "refresh_forecast_snapshot_kpis", "params": {}}],
        ),
    ):
        with pytest.raises(JobCancelledError, match="Period Roll cancelled"):
            _run_period_roll({}, cancel_event=cancelled)
