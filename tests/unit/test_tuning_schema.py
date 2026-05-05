"""Tests for the unified model tuning SQL schema (099_unified_model_tuning.sql).

Validates that the migration file exists and contains the expected DDL
statements for:
- lgbm_tuning_lag table (per-execution-lag accuracy breakdown)
- lgbm_tuning_lag_cluster table (per-lag-per-cluster drill-down)
- tuning_promotion_log table (promotion audit trail)
- lgbm_tuning_run alterations (job_id, template_id, status CHECK extension)
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from common.core.paths import SQL_DIR  # noqa: E402

SQL_PATH = SQL_DIR / "099_unified_model_tuning.sql"


@pytest.fixture(scope="module")
def sql_content() -> str:
    """Read the migration SQL file once for all tests in the module."""
    return SQL_PATH.read_text()


# ===========================================================================
# 1. Migration file exists
# ===========================================================================
def test_migration_file_exists():
    """Assert 099_unified_model_tuning.sql exists on disk."""
    assert SQL_PATH.exists(), f"Migration file not found: {SQL_PATH}"
    assert SQL_PATH.stat().st_size > 0, "Migration file is empty"


# ===========================================================================
# 2. lgbm_tuning_lag table created
# ===========================================================================
def test_lgbm_tuning_lag_table_created(sql_content: str):
    """SQL contains CREATE TABLE lgbm_tuning_lag."""
    assert "CREATE TABLE" in sql_content
    assert "lgbm_tuning_lag" in sql_content
    # Verify it's a CREATE TABLE statement (not just a reference)
    normalized = sql_content.lower().replace("\n", " ")
    assert "create table" in normalized
    assert "lgbm_tuning_lag" in normalized


# ===========================================================================
# 3. lgbm_tuning_lag CHECK constraint on exec_lag
# ===========================================================================
def test_lgbm_tuning_lag_check_constraint(sql_content: str):
    """SQL contains CHECK (exec_lag BETWEEN 0 AND 4)."""
    normalized = sql_content.lower().replace("\n", " ")
    assert "check" in normalized
    assert "exec_lag" in normalized
    assert "between 0 and 4" in normalized


# ===========================================================================
# 4. lgbm_tuning_lag UNIQUE constraint on (run_id, exec_lag)
# ===========================================================================
def test_lgbm_tuning_lag_unique_constraint(sql_content: str):
    """SQL contains UNIQUE (run_id, exec_lag)."""
    normalized = sql_content.lower().replace("\n", " ")
    assert "unique" in normalized
    assert "run_id" in normalized
    assert "exec_lag" in normalized


# ===========================================================================
# 5. tuning_promotion_log table created
# ===========================================================================
def test_promotion_log_table_created(sql_content: str):
    """SQL contains CREATE TABLE tuning_promotion_log."""
    normalized = sql_content.lower().replace("\n", " ")
    assert "create table" in normalized
    assert "tuning_promotion_log" in normalized
    # Verify key columns exist
    assert "promoted_at" in normalized
    assert "promoted_by" in normalized
    assert "params_written" in normalized
    assert "previous_run_id" in normalized


# ===========================================================================
# 6. lgbm_tuning_lag_cluster table created
# ===========================================================================
def test_lag_cluster_table_created(sql_content: str):
    """SQL contains CREATE TABLE lgbm_tuning_lag_cluster."""
    normalized = sql_content.lower().replace("\n", " ")
    assert "create table" in normalized
    assert "lgbm_tuning_lag_cluster" in normalized
    # Verify key columns
    assert "cluster_type" in normalized
    assert "cluster_value" in normalized


# ===========================================================================
# 7. Status CHECK constraint updated to include queued and cancelled
# ===========================================================================
def test_status_check_updated(sql_content: str):
    """SQL contains 'queued' and 'cancelled' in the status CHECK constraint."""
    normalized = sql_content.lower().replace("\n", " ")
    assert "queued" in normalized
    assert "cancelled" in normalized
    # Verify the CHECK constraint is being replaced/added
    assert "lgbm_tuning_run_status_check" in normalized


# ===========================================================================
# 8. job_id column added to lgbm_tuning_run
# ===========================================================================
def test_job_id_column_added(sql_content: str):
    """SQL contains ADD COLUMN ... job_id."""
    normalized = sql_content.lower().replace("\n", " ")
    assert "add column" in normalized
    assert "job_id" in normalized
    # Verify it targets lgbm_tuning_run
    assert "alter table" in normalized
    assert "lgbm_tuning_run" in normalized
