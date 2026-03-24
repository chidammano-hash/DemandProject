"""Tests for scripts/ops/run_perf_analysis.py — performance analysis CLI."""
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import scripts.ops.run_perf_analysis as _perf_mod
from scripts.ops.run_perf_analysis import (
    _run_script_mode,
    _run_report_mode,
    _output_report,
    main,
)
from common.services.perf_profiler import PerfReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_CONFIG: dict = {"thresholds": {}, "script_presets": {}}


def _make_mock_conn():
    """Build a mock psycopg3 connection for _get_connection."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = []
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# test_script_mode_generates_report
# ---------------------------------------------------------------------------


def test_script_mode_generates_report():
    """_run_script_mode profiles a script and returns a PerfReport."""
    # Create a fake module with a run() function
    fake_module = types.ModuleType("fake_script")
    fake_module.run = MagicMock()  # type: ignore[attr-defined]

    mock_conn = _make_mock_conn()

    with (
        patch(
            "scripts.ops.run_perf_analysis._get_connection",
            return_value=(mock_conn, True),
        ),
        patch(
            "scripts.ops.run_perf_analysis.importlib.import_module",
            return_value=fake_module,
        ),
    ):
        report = _run_script_mode("fake_script", _EMPTY_CONFIG)

    assert isinstance(report, PerfReport)
    assert report.script_name == "fake_script"
    assert report.total_wall_time_s >= 0
    # The script run() should have been called
    fake_module.run.assert_called()


# ---------------------------------------------------------------------------
# test_report_mode_combines_results
# ---------------------------------------------------------------------------


def test_report_mode_combines_results():
    """_run_report_mode calls api + pipeline sub-modes and combines them."""
    api_report = PerfReport(
        script_name="api_analysis",
        total_wall_time_s=1.0,
        total_cpu_time_s=0.5,
        peak_memory_mb=100.0,
    )
    pipeline_report = PerfReport(
        script_name="pipeline",
        total_wall_time_s=2.0,
        total_cpu_time_s=1.0,
        peak_memory_mb=200.0,
    )

    with (
        patch(
            "scripts.ops.run_perf_analysis._run_api_mode",
            return_value=api_report,
        ),
        patch(
            "scripts.ops.run_perf_analysis._run_pipeline_mode",
            return_value=pipeline_report,
        ),
    ):
        combined = _run_report_mode(_EMPTY_CONFIG)

    assert isinstance(combined, PerfReport)
    assert combined.script_name == "combined_report"
    # Combined wall time should be sum of sub-reports
    assert combined.total_wall_time_s == pytest.approx(3.0)
    assert combined.total_cpu_time_s == pytest.approx(1.5)
    assert combined.peak_memory_mb == pytest.approx(200.0)
    # Should have 2 wrapper sections (mode_api, mode_pipeline)
    assert len(combined.sections) == 2
    assert combined.sections[0].name == "mode_api"
    assert combined.sections[1].name == "mode_pipeline"


# ---------------------------------------------------------------------------
# test_output_json_written
# ---------------------------------------------------------------------------


def test_output_json_written(tmp_path):
    """_output_report writes valid JSON to the specified path."""
    report = PerfReport(
        script_name="test_output",
        total_wall_time_s=1.23,
        total_cpu_time_s=0.45,
        peak_memory_mb=50.0,
    )

    out_path = tmp_path / "reports" / "test.json"
    _output_report(report, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["script_name"] == "test_output"
    assert data["total_wall_time_s"] == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


def test_main_script_mode_writes_output(tmp_path):
    """main() in script mode writes a JSON report."""
    fake_module = types.ModuleType("fake_script")
    fake_module.run = MagicMock()  # type: ignore[attr-defined]

    mock_conn = _make_mock_conn()
    out_path = tmp_path / "out.json"

    # Patch _load_perf_config (local to run_perf_analysis) plus connection & import
    with (
        patch.object(
            _perf_mod, "_get_connection",
            return_value=(mock_conn, True),
        ),
        patch.object(
            _perf_mod, "importlib",
            **{"import_module.return_value": fake_module},
        ),
        patch.object(
            _perf_mod, "_load_perf_config",
            return_value=_EMPTY_CONFIG,
        ),
    ):
        main(["--mode", "script", "--script", "fake_script", "--output", str(out_path)])

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["script_name"] == "fake_script"
