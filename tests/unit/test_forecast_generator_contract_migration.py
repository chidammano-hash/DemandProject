"""Structural guard for retiring pre-canonical release candidates."""

from pathlib import Path

from common.services.forecast_generation import GENERATOR_CONTRACT_VERSION


def test_migration_invalidates_only_ready_old_contract_release_candidates():
    ddl = Path("sql/206_invalidate_pre_canonical_generator_runs.sql").read_text().lower()

    assert "update forecast_generation_run" in ddl
    assert "run_status = 'invalid'" in ddl
    assert "promotion_eligible = false" in ddl
    assert "generation_purpose = 'release_candidate'" in ddl
    assert "run_status = 'ready'" in ddl
    assert "metadata ->> 'generator_contract_version'" in ddl
    assert "is distinct from 'canonical-five-real-adapters-v1'" in ddl
    assert "snapshot_contender" not in ddl


def test_v2_migration_invalidates_ready_candidates_and_contenders():
    ddl = Path(
        "sql/209_invalidate_pre_artifact_lineage_generator_runs.sql"
    ).read_text().lower()

    assert "update forecast_generation_run" in ddl
    assert "run_status = 'invalid'" in ddl
    assert "promotion_eligible = false" in ddl
    assert "generation_purpose in ('release_candidate', 'snapshot_contender')" in ddl
    assert "run_status = 'ready'" in ddl
    assert "metadata ->> 'generator_contract_version'" in ddl
    assert "is distinct from 'canonical-five-artifact-lineage-v2'" in ddl
    assert GENERATOR_CONTRACT_VERSION == "canonical-five-artifact-lineage-v2"
