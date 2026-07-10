"""Tests for durable forecast-generation run reservations."""

from datetime import date
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from common.services.forecast_generation import (
    invalidate_generation_run,
    reserve_generation_run,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000123")


def test_reserve_inserts_generating_manifest_before_work_starts():
    cur = MagicMock()
    cur.fetchone.return_value = (
        "release_candidate",
        "champion",
        date(2026, 7, 1),
        24,
        "generating",
        0,
    )

    status = reserve_generation_run(
        cur,
        run_id=RUN_ID,
        generation_purpose="release_candidate",
        requested_model_id="champion",
        record_month=date(2026, 7, 1),
        horizon_months=24,
        created_by="api",
    )

    assert status == "generating"
    insert_sql = cur.execute.call_args_list[0].args[0]
    assert "INSERT INTO forecast_generation_run" in insert_sql
    assert "ON CONFLICT (run_id) DO NOTHING" in insert_sql


def test_reserve_rejects_same_uuid_with_different_identity():
    cur = MagicMock()
    cur.fetchone.return_value = (
        "snapshot_contender",
        "lgbm_cluster",
        date(2026, 7, 1),
        6,
        "generating",
        0,
    )

    with pytest.raises(ValueError, match="identity"):
        reserve_generation_run(
            cur,
            run_id=RUN_ID,
            generation_purpose="release_candidate",
            requested_model_id="champion",
            record_month=date(2026, 7, 1),
            horizon_months=24,
            created_by="api",
        )


def test_invalid_run_can_resume_only_without_staging_rows():
    cur = MagicMock()
    cur.fetchone.side_effect = [
        (
            "release_candidate",
            "champion",
            date(2026, 7, 1),
            24,
            "invalid",
            0,
        ),
        (False,),
    ]

    status = reserve_generation_run(
        cur,
        run_id=RUN_ID,
        generation_purpose="release_candidate",
        requested_model_id="champion",
        record_month=date(2026, 7, 1),
        horizon_months=24,
        created_by="retry",
        resume_invalid=True,
    )

    assert status == "generating"
    assert any(
        "SET run_status = 'generating'" in call.args[0]
        for call in cur.execute.call_args_list
    )


def test_invalidate_is_terminal_and_non_destructive_for_ready_run():
    cur = MagicMock()
    cur.rowcount = 1

    changed = invalidate_generation_run(cur, RUN_ID, "subprocess failed")

    sql, params = cur.execute.call_args.args
    assert "run_status IN ('generating', 'invalid')" in sql
    assert params[0] == "subprocess failed"
    assert changed is True
