"""US17b — correctness guards for the integration_job_unified view DDL.

The view normalizes job_history ingestion rows into the integration Job column
set. These tests pin the contract without needing a live Postgres: they assert
the DDL exists, UNIONs both sources, filters to ingestion job types, and uses
the same status vocabulary as the pure Python adapter (common.services.job_shape)
— the single source of truth must agree across SQL and Python.
"""
from __future__ import annotations

from pathlib import Path

from common.core.paths import SQL_DIR
from common.services.job_shape import _JH_TO_INTEGRATION


def _ddl_text() -> str:
    matches = sorted(SQL_DIR.glob("*_create_integration_job_unified*.sql"))
    assert matches, "expected a *_create_integration_job_unified*.sql migration"
    return matches[-1].read_text().lower()


def test_view_ddl_exists_and_creates_view() -> None:
    ddl = _ddl_text()
    assert "create" in ddl and "view" in ddl
    assert "integration_job_unified" in ddl


def test_view_unions_both_sources() -> None:
    ddl = _ddl_text()
    assert "integration_job" in ddl
    assert "job_history" in ddl
    assert "union" in ddl


def test_view_filters_to_ingestion_job_types() -> None:
    ddl = _ddl_text()
    # Only ETL job types from job_history may leak into the integration list.
    assert "etl_pipeline" in ddl
    assert "load_domain" in ddl


def test_view_status_map_agrees_with_python() -> None:
    ddl = _ddl_text()
    # The only divergence is completed -> success; SQL must encode the same map.
    assert _JH_TO_INTEGRATION == {"completed": "success"}
    assert "'completed'" in ddl
    assert "'success'" in ddl


def test_view_exposes_full_job_column_set() -> None:
    ddl = _ddl_text()
    for col in (
        "id", "domain", "mode", "slice", "file_path", "status",
        "rows_loaded", "rows_inserted", "rows_updated", "rows_deleted",
        "error_message", "started_at", "completed_at", "duration_ms",
        "triggered_by",
    ):
        assert col in ddl, f"view must expose column {col!r}"


def test_ddl_in_sql_dir() -> None:
    # Sanity: the migration lives under sql/ with a numeric prefix.
    matches = sorted(Path(SQL_DIR).glob("*_create_integration_job_unified*.sql"))
    assert matches
    assert matches[-1].name[:3].isdigit()
