"""Tests for restart-safe snapshot contender preparation."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from common.services.forecast_generation import (
    GENERATOR_CONTRACT_METADATA_KEY,
    GENERATOR_CONTRACT_VERSION,
)
from common.services.forecast_snapshot_validation import (
    SnapshotContenderIntegrityError,
    SnapshotContenderStaleError,
)
from scripts.forecasting.prepare_forecast_snapshot_contenders import (
    _contender_requires_generation,
    _delete_recoverable_roster,
    _latest_runs,
    _reserve_contender_runs,
    prepare_contenders,
)

RECORD_MONTH = date(2026, 7, 1)
CURRENT_METADATA = {
    GENERATOR_CONTRACT_METADATA_KEY: GENERATOR_CONTRACT_VERSION,
}
CONTENDER = {
    "model_id": "nhits",
    "generation_run_id": "00000000-0000-0000-0000-000000000123",
    "backtest_run_id": 102,
}


def test_missing_manifest_requires_generation():
    cur = MagicMock()
    cur.fetchone.return_value = None

    assert _contender_requires_generation(cur, RECORD_MONTH, CONTENDER) is True


def test_fresh_roster_reserves_all_contender_manifests_before_fk_insert(monkeypatch):
    cur = MagicMock()
    reserve = MagicMock(return_value="generating")
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.reserve_generation_run",
        reserve,
    )
    contenders = [
        {
            "model_id": model_id,
            "generation_run_id": f"00000000-0000-0000-0000-00000000012{rank}",
            "contender_rank": rank,
            "backtest_run_id": rank,
            "wape": 0.1 * rank,
        }
        for rank, model_id in enumerate(("nhits", "nbeats", "mstl"), start=1)
    ]

    _reserve_contender_runs(cur, RECORD_MONTH, contenders)

    assert reserve.call_count == 3
    assert [call.kwargs["requested_model_id"] for call in reserve.call_args_list] == [
        "nhits",
        "nbeats",
        "mstl",
    ]
    assert all(call.kwargs["horizon_months"] == 6 for call in reserve.call_args_list)


def test_prepare_reserves_fk_targets_before_inserting_fresh_roster(monkeypatch):
    connect = MagicMock()
    conn = connect.return_value.__enter__.return_value
    conn.cursor.return_value.__enter__.return_value = MagicMock()
    events: list[str] = []
    selected = [
        {
            "model_id": model_id,
            "contender_rank": rank,
            "backtest_run_id": rank,
            "wape": 0.1 * rank,
        }
        for rank, model_id in enumerate(("nhits", "nbeats", "mstl"), start=1)
    ]
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.psycopg.connect",
        connect,
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._snapshot_config",
        lambda: {},
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._load_existing_roster",
        lambda *_: [],
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._latest_runs",
        lambda *_: selected,
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.select_top_contenders",
        lambda rows: [dict(row) for row in rows],
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._reserve_contender_runs",
        lambda *_: events.append("reserve"),
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._insert_roster",
        lambda *_, **__: events.append("insert"),
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._verify_staged_lags",
        lambda *_: None,
    )
    run = MagicMock()
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.subprocess.run",
        run,
    )

    result = prepare_contenders(RECORD_MONTH)

    assert events == ["reserve", "insert"]
    assert [row["model_id"] for row in result] == ["nhits", "nbeats", "mstl"]
    assert run.call_count == 3
    assert all(
        call.kwargs["env"]["OMP_NUM_THREADS"] == "1"
        for call in run.call_args_list
    )


def test_prepare_replaces_stale_unpublished_roster_before_reserving_new_runs(
    monkeypatch,
):
    connect = MagicMock()
    conn = connect.return_value.__enter__.return_value
    conn.cursor.return_value.__enter__.return_value = MagicMock()
    existing = [
        {
            "model_id": "champion",
            "snapshot_role": "champion",
            "contender_rank": None,
            "generation_run_id": None,
        },
        *[
            {
                "model_id": model_id,
                "snapshot_role": "contender",
                "contender_rank": rank,
                "generation_run_id": f"00000000-0000-0000-0000-00000000011{rank}",
            }
            for rank, model_id in enumerate(("nhits", "nbeats", "mstl"), start=1)
        ],
    ]
    selected = [
        {
            "model_id": model_id,
            "contender_rank": rank,
            "backtest_run_id": rank + 10,
            "wape": 0.1 * rank,
        }
        for rank, model_id in enumerate(("lgbm_cluster", "nhits", "nbeats"), start=1)
    ]
    events: list[str] = []
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.psycopg.connect",
        connect,
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._snapshot_config",
        lambda: {},
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._load_existing_roster",
        lambda *_: existing,
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._contender_requires_generation",
        MagicMock(),
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._validate_existing_roster_backtests",
        MagicMock(side_effect=SnapshotContenderStaleError("sales reload")),
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._delete_recoverable_roster",
        lambda *_: events.append("delete"),
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._latest_runs",
        lambda *_: selected,
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.select_top_contenders",
        lambda rows: [dict(row) for row in rows],
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._reserve_contender_runs",
        lambda *_: events.append("reserve"),
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._insert_roster",
        lambda *_, **__: events.append("insert"),
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._verify_staged_lags",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.subprocess.run",
        MagicMock(),
    )

    result = prepare_contenders(RECORD_MONTH)

    assert events == ["delete", "reserve", "insert"]
    assert [row["model_id"] for row in result] == [
        "lgbm_cluster",
        "nhits",
        "nbeats",
    ]


def test_latest_runs_uses_exact_governed_ids_not_newer_loaded_runs(monkeypatch):
    cur = MagicMock()
    cur.fetchall.return_value = [
        (101, "lgbm_cluster", 0.10, 90.0, None),
        (102, "nhits", 0.20, 80.0, None),
    ]
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._current_governed_lineage",
        lambda _cur: {
            "backtest_run_ids": {"lgbm_cluster": 101, "nhits": 102},
        },
    )

    rows = _latest_runs(cur, ["lgbm_cluster", "nhits"])

    sql, params = cur.execute.call_args.args
    assert "id = ANY" in sql
    assert params[0] == [101, 102]
    assert [row["backtest_run_id"] for row in rows] == [101, 102]


def test_latest_runs_rejects_missing_governed_backtest(monkeypatch):
    cur = MagicMock()
    cur.fetchall.return_value = [(101, "lgbm_cluster", 0.10, 90.0, None)]
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders._current_governed_lineage",
        lambda _cur: {
            "backtest_run_ids": {"lgbm_cluster": 101, "nhits": 102},
        },
    )

    with pytest.raises(SnapshotContenderStaleError, match="exact governed"):
        _latest_runs(cur, ["lgbm_cluster", "nhits"])


def test_ready_complete_manifest_is_reused():
    cur = MagicMock()
    cur.fetchone.return_value = (
        "snapshot_contender",
        "nhits",
        RECORD_MONTH,
        "ready",
        CURRENT_METADATA,
    )
    with patch(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.validate_ready_snapshot_contender"
    ) as validate:
        assert _contender_requires_generation(cur, RECORD_MONTH, CONTENDER) is False

    validate.assert_called_once()


def test_ready_incomplete_manifest_fails_instead_of_overwriting():
    cur = MagicMock()
    cur.fetchone.return_value = (
        "snapshot_contender",
        "nhits",
        RECORD_MONTH,
        "ready",
        CURRENT_METADATA,
    )
    with (
        patch(
            "scripts.forecasting.prepare_forecast_snapshot_contenders."
            "validate_ready_snapshot_contender",
            side_effect=SnapshotContenderIntegrityError("payload mismatch"),
        ),
        pytest.raises(SnapshotContenderIntegrityError, match="payload mismatch"),
    ):
        _contender_requires_generation(cur, RECORD_MONTH, CONTENDER)


def test_manifest_identity_mismatch_is_rejected():
    cur = MagicMock()
    cur.fetchone.return_value = (
        "release_candidate",
        "nhits",
        RECORD_MONTH,
        "ready",
        CURRENT_METADATA,
    )

    with pytest.raises(ValueError, match="identity"):
        _contender_requires_generation(cur, RECORD_MONTH, CONTENDER)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {GENERATOR_CONTRACT_METADATA_KEY: "retired-heuristic-v0"},
    ],
)
def test_ready_manifest_from_old_generator_is_not_reused(metadata):
    cur = MagicMock()
    cur.fetchone.return_value = (
        "snapshot_contender",
        "nhits",
        RECORD_MONTH,
        "ready",
        metadata,
    )

    with (
        patch(
            "scripts.forecasting.prepare_forecast_snapshot_contenders."
            "validate_ready_snapshot_contender",
            side_effect=SnapshotContenderStaleError("outdated generator contract"),
        ),
        pytest.raises(SnapshotContenderStaleError, match="generator contract"),
    ):
        _contender_requires_generation(cur, RECORD_MONTH, CONTENDER)


def test_migration_invalidated_old_contract_with_staging_is_replaceable():
    cur = MagicMock()
    cur.fetchone.return_value = (
        "snapshot_contender",
        "nhits",
        RECORD_MONTH,
        "invalid",
        {GENERATOR_CONTRACT_METADATA_KEY: "canonical-five-real-adapters-v1"},
    )

    with pytest.raises(SnapshotContenderStaleError, match="outdated generator contract"):
        _contender_requires_generation(cur, RECORD_MONTH, CONTENDER)

    assert len(cur.execute.call_args_list) == 1


@pytest.mark.parametrize("status", ["generating", "invalid"])
def test_unfinished_reserved_run_without_staging_rows_is_resumable(status):
    cur = MagicMock()
    cur.fetchone.side_effect = [
        ("snapshot_contender", "nhits", RECORD_MONTH, status, CURRENT_METADATA),
        (False,),
    ]

    assert _contender_requires_generation(cur, RECORD_MONTH, CONTENDER) is True


def test_existing_staging_backfill_requires_an_original_frozen_roster(monkeypatch):
    connect = MagicMock()
    conn = connect.return_value.__enter__.return_value
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.psycopg.connect",
        connect,
    )

    with pytest.raises(ValueError, match="original frozen roster"):
        prepare_contenders(
            RECORD_MONTH,
            from_existing_staging=True,
        )


def test_incomplete_current_roster_can_be_replaced_only_before_archive_or_publish(
    monkeypatch,
):
    cur = MagicMock()
    cur.fetchone.return_value = (False, False)
    cur.rowcount = 2
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.get_planning_date",
        lambda: RECORD_MONTH,
    )

    _delete_recoverable_roster(
        cur,
        RECORD_MONTH,
        [{"model_id": "champion"}, {"model_id": "nhits"}],
    )

    delete_sql, delete_params = cur.execute.call_args_list[-1].args
    assert "DELETE FROM forecast_snapshot_roster" in delete_sql
    assert delete_params == (RECORD_MONTH,)


@pytest.mark.parametrize("safety_state", [(True, False), (False, True)])
def test_incomplete_roster_repair_fails_after_archive_or_publish(
    monkeypatch,
    safety_state,
):
    cur = MagicMock()
    cur.fetchone.return_value = safety_state
    monkeypatch.setattr(
        "scripts.forecasting.prepare_forecast_snapshot_contenders.get_planning_date",
        lambda: RECORD_MONTH,
    )

    with pytest.raises(ValueError, match="cannot be replaced"):
        _delete_recoverable_roster(
            cur,
            RECORD_MONTH,
            [{"model_id": "champion"}],
        )

    assert not any(
        "DELETE FROM forecast_snapshot_roster" in call.args[0]
        for call in cur.execute.call_args_list
    )
