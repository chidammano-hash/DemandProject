"""Run-scoped cleanup guards for forecast snapshot contenders."""

from datetime import date
from unittest.mock import MagicMock, patch

from scripts.forecasting.cleanup_forecast_staging import cleanup_generation


def test_cleanup_deletes_only_reconciled_snapshot_contender_runs():
    cur = MagicMock()
    cur.fetchone.return_value = (180,)
    cur.rowcount = 180
    expected = {
        ("a", "00000000-0000-0000-0000-000000000001"): 60,
        ("b", "00000000-0000-0000-0000-000000000002"): 60,
        ("c", "00000000-0000-0000-0000-000000000003"): 60,
    }

    with (
        patch(
            "scripts.forecasting.cleanup_forecast_staging._expected_contender_counts",
            return_value=expected,
        ),
        patch(
            "scripts.forecasting.cleanup_forecast_staging._archived_contender_counts",
            return_value=expected,
        ),
        patch("scripts.forecasting.cleanup_forecast_staging._validate_lag_coverage"),
        patch(
            "scripts.forecasting.cleanup_forecast_staging._champion_archive_count",
            return_value=60,
        ),
    ):
        deleted = cleanup_generation(cur, date(2026, 6, 1), dry_run=False)

    assert deleted == 180
    delete_sql = next(
        call.args[0]
        for call in cur.execute.call_args_list
        if "DELETE FROM fact_production_forecast_staging" in call.args[0]
    )
    assert "USING forecast_snapshot_roster" in delete_sql
    assert "generation_purpose = 'snapshot_contender'" in delete_sql
    assert "roster.model_id = staging.candidate_model_id" in delete_sql
    assert "forecast_month_generated = %s" not in delete_sql
    update_sql = next(
        call.args[0]
        for call in cur.execute.call_args_list
        if "UPDATE forecast_generation_run" in call.args[0]
    )
    assert "run_status = 'archived'" in update_sql
