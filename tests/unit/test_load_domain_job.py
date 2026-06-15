"""US17c — the load_domain JobManager handler.

Drives a single-domain ETL load through the unified engine (scripts.etl.load),
recording parsed row metrics in the job result. Exit codes mirror the legacy
IntegrationRunner: 0 -> loaded, 2 -> skipped (no work), other -> failure.
subprocess.run is mocked — no real subprocess, no DB.
"""
from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from common.services.job_state import _run_load_domain


def _proc(returncode: int, stdout: str = "", stderr: str = "") -> CompletedProcess:
    return CompletedProcess(args=["x"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_handler_invokes_unified_engine() -> None:
    final_json = '{"rows_loaded": 1234, "rows_inserted": 1000, "rows_updated": 234, "rows_deleted": 0}'
    with patch("common.services.job_state.subprocess.run",
               return_value=_proc(0, stdout=final_json)) as run:
        result = _run_load_domain({"domain": "sales", "mode": "delta", "slice": "2026-04"})
    cmd = run.call_args.args[0]
    # shells out to the single unified engine
    assert "scripts.etl.load" in cmd
    assert "--domain" in cmd and "sales" in cmd
    assert "--mode" in cmd and "delta" in cmd
    assert "--slice" in cmd and "2026-04" in cmd
    # result carries the parsed metrics for the unified view to read
    assert result["rows_loaded"] == 1234
    assert result["rows_inserted"] == 1000
    assert result["rows_updated"] == 234
    assert result["skipped"] is False
    assert result["domain"] == "sales"


def test_handler_passes_file_and_reindex() -> None:
    with patch("common.services.job_state.subprocess.run",
               return_value=_proc(0, stdout='{"rows_loaded": 5}')) as run:
        _run_load_domain({"domain": "item", "mode": "file",
                          "file": "data/staged/itemdata_clean.csv", "reindex": True})
    cmd = run.call_args.args[0]
    assert "--file" in cmd and "data/staged/itemdata_clean.csv" in cmd
    assert "--reindex" in cmd


def test_handler_exit_2_is_skipped_not_failure() -> None:
    with patch("common.services.job_state.subprocess.run",
               return_value=_proc(2, stdout='{"rows_loaded": 0}')):
        result = _run_load_domain({"domain": "forecast", "mode": "delta"})
    assert result["skipped"] is True
    assert result["rows_loaded"] == 0


def test_handler_nonzero_exit_raises() -> None:
    with patch("common.services.job_state.subprocess.run",
               return_value=_proc(1, stdout='{"error": "bad partition"}', stderr="trace")):
        with pytest.raises(RuntimeError) as exc:
            _run_load_domain({"domain": "inventory", "mode": "file", "slice": "2026-03"})
    assert "inventory" in str(exc.value)


def test_handler_requires_domain() -> None:
    with pytest.raises(ValueError):
        _run_load_domain({"mode": "delta"})


def test_handler_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        _run_load_domain({"domain": "sales", "mode": "bogus"})
