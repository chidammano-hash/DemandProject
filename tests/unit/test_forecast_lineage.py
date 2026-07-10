"""Unit tests for deterministic forecast lineage evidence."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

from common.services.forecast_lineage import (
    compute_champion_results_stats,
    compute_production_payload_stats,
    compute_staging_payload_stats,
    sha256_file,
)


def test_staging_checksum_is_run_and_window_scoped():
    cur = MagicMock()
    cur.fetchone.return_value = ("a" * 64, 24, 4, 3)

    stats = compute_staging_payload_stats(
        cur,
        "00000000-0000-0000-0000-000000000001",
        start_month=date(2026, 7, 1),
        end_month=date(2027, 7, 1),
    )

    sql, params = cur.execute.call_args.args
    assert "FROM fact_production_forecast_staging" in sql
    assert "WHERE run_id = %s::uuid" in sql
    assert "ORDER BY item_id, loc, forecast_month, source_model_id" in sql
    assert "DIGEST" in sql and "sha256" in sql
    assert params == (
        "00000000-0000-0000-0000-000000000001",
        date(2026, 7, 1),
        date(2026, 7, 1),
        date(2027, 7, 1),
        date(2027, 7, 1),
    )
    assert stats.row_count == 24
    assert stats.dfu_count == 4
    assert stats.source_model_count == 3


def test_production_checksum_uses_preserved_source_model():
    cur = MagicMock()
    cur.fetchone.return_value = ("b" * 64, 12, 2, 2)

    stats = compute_production_payload_stats(
        cur,
        "00000000-0000-0000-0000-000000000002",
    )

    sql = cur.execute.call_args.args[0]
    assert "FROM fact_production_forecast" in sql
    assert "source_model_id" in sql
    assert "model_id AS source_model_id" not in sql
    assert stats.checksum == "b" * 64


def test_sha256_file_hashes_exact_bytes(tmp_path: Path):
    artifact = tmp_path / "winners.csv"
    artifact.write_bytes(b"item_id,loc,model_id\n1,L,A\n")

    assert (
        sha256_file(artifact) == "8108156875e927eba0f15bb3942b48151d8c54a66c49e633df3f98d5a729b280"
    )


def test_champion_results_checksum_is_experiment_scoped_at_full_dfu_lag_grain():
    cur = MagicMock()
    cur.fetchone.return_value = ("e" * 64, 600, 20, 4)

    stats = compute_champion_results_stats(cur, 33)

    sql, params = cur.execute.call_args.args
    assert "champion_experiment_id = %s" in sql
    assert "customer_group" in sql
    assert "startdate" in sql and "lag" in sql
    assert params == (33,)
    assert stats.row_count == 600
