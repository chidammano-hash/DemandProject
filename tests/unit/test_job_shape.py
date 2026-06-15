"""US17a — pure shape/status adapter between job_history and integration_job.

These tests pin the translation layer that lets a JobManager-backed ingestion
job (etl_pipeline / load_domain) be surfaced through the existing
/integration/jobs API contract. The functions under test must be PURE — no DB,
no network, importable without opening a connection.
"""
from __future__ import annotations

import common.services.job_shape as job_shape
from common.services.job_shape import (
    job_history_to_integration_job,
    to_integration_status,
    to_job_history_status,
)


def test_status_round_trip() -> None:
    # The one divergence: completed <-> success.
    assert to_integration_status("completed") == "success"
    assert to_job_history_status("success") == "completed"
    # Round-trips back to itself.
    assert to_job_history_status(to_integration_status("completed")) == "completed"
    assert to_integration_status(to_job_history_status("success")) == "success"
    # Everything else passes through unchanged, both directions.
    for status in ("queued", "running", "failed", "skipped", "cancelled"):
        assert to_integration_status(status) == status
        assert to_job_history_status(status) == status
    # None passes through.
    assert to_integration_status(None) is None
    assert to_job_history_status(None) is None


def test_maps_etl_pipeline_row_to_integration_shape() -> None:
    row = {
        "job_id": "job-abc",
        "job_type": "etl_pipeline",
        "job_label": "ETL pipeline (refresh)",
        "status": "completed",
        "params": {"mode": "refresh", "domains": ["sales", "forecast"]},
        "result": {"loaded": 7, "skipped": 1},
        "error": None,
        "started_at": "2026-06-01T10:00:00",
        "completed_at": "2026-06-01T10:05:00",
        "triggered_by": "ui",
    }
    out = job_history_to_integration_job(row)
    assert out["id"] == "job-abc"
    assert out["mode"] == "refresh"
    # Multi-domain pipeline collapses to a readable domain label.
    assert out["domain"] == "sales,forecast"
    assert out["status"] == "success"  # completed -> success
    assert out["rows_loaded"] == 7
    assert out["error_message"] is None
    assert out["started_at"] == "2026-06-01T10:00:00"
    assert out["completed_at"] == "2026-06-01T10:05:00"
    assert out["triggered_by"] == "ui"


def test_maps_load_domain_row_to_integration_shape() -> None:
    row = {
        "job_id": "job-xyz",
        "job_type": "load_domain",
        "job_label": "Load sales (delta)",
        "status": "running",
        "params": {"domain": "sales", "mode": "delta", "slice": "2026-04"},
        "result": {"rows_loaded": 1234, "rows_inserted": 1000, "rows_updated": 234},
        "error": None,
        "started_at": "2026-06-01T11:00:00",
        "completed_at": None,
        "triggered_by": "api",
    }
    out = job_history_to_integration_job(row)
    assert out["id"] == "job-xyz"
    assert out["domain"] == "sales"
    assert out["mode"] == "delta"
    assert out["slice"] == "2026-04"
    assert out["status"] == "running"  # passthrough
    assert out["rows_loaded"] == 1234
    assert out["rows_inserted"] == 1000
    assert out["rows_updated"] == 234
    assert out["completed_at"] is None


def test_missing_params_result_defaults() -> None:
    # A non-ETL job_history row (no params/result) must map without raising.
    row = {
        "job_id": "job-empty",
        "job_type": "data_quality",
        "status": "queued",
        "params": None,
        "result": None,
    }
    out = job_history_to_integration_job(row)
    assert out["id"] == "job-empty"
    assert out["rows_loaded"] == 0
    assert out["status"] == "queued"
    assert out["slice"] is None
    assert out["error_message"] is None
    # triggered_by absent -> sensible default
    assert out["triggered_by"] == "api"


def test_failed_row_carries_error() -> None:
    row = {
        "job_id": "job-fail",
        "job_type": "load_domain",
        "status": "failed",
        "params": {"domain": "inventory", "mode": "file", "slice": "2026-03"},
        "result": None,
        "error": "boom: bad partition",
    }
    out = job_history_to_integration_job(row)
    assert out["status"] == "failed"
    assert out["error_message"] == "boom: bad partition"
    assert out["domain"] == "inventory"


def test_output_constructs_integration_job_model() -> None:
    # AC: byte-compatible with the /integration/jobs Job model (US17b relies on this).
    from api.routers.platform.integration import Job

    row = {
        "job_id": "job-model",
        "job_type": "etl_pipeline",
        "status": "completed",
        "params": {"mode": "full", "domains": ["sales"]},
        "result": {"loaded": 3},
        "started_at": "2026-06-01T12:00:00",
        "completed_at": "2026-06-01T12:30:00",
    }
    model = Job(**job_history_to_integration_job(row))
    assert model.id == "job-model"
    assert model.status == "success"
    assert model.rows_loaded == 3


def test_pure_no_db_import() -> None:
    # The module must not pull in DB plumbing — purity guard.
    assert not hasattr(job_shape, "psycopg")
    assert not hasattr(job_shape, "get_db_params")
    assert not hasattr(job_shape, "_get_conn")
    # Calling with a plain dict touches nothing external.
    assert job_history_to_integration_job({"job_id": "x", "status": "queued"})["id"] == "x"
