"""Fail-closed evidence tests for governed forecast backtests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from common.ml.backtest_config import BacktestConfigSnapshot
from common.services.governed_backtest_evidence import (
    BACKTEST_EVIDENCE_METADATA_KEY,
    PayloadStats,
    compute_csv_payload_stats,
    compute_model_fact_payload_stats,
    load_current_governed_backtest_runs,
    validate_snapshot_roster_provenance,
)


MODELS = ("lgbm_cluster", "nhits", "nbeats", "mstl", "chronos2_enriched")
LINEAGE = {
    "source_sales_batch_id": 91,
    "data_checksum": "a" * 64,
    "cluster_experiment_id": 35,
    "cluster_assignment_count": 13_968,
    "cluster_assignment_checksum": "b" * 64,
}


def _config_snapshot(model_id: str) -> BacktestConfigSnapshot:
    return BacktestConfigSnapshot(
        model_id=model_id,
        checksum="c" * 64,
        _config_json=(
            '{"cluster_tuning_profiles":{"metadata":{"cluster_experiment_id":35}},'
            f'"model_id":"{model_id}"}}'
        ),
    )


def _metadata(model_id: str, run_id: int, *, fact_checksum: str = "d" * 64) -> dict:
    return {
        "managed_execution": {
            "backtest_run_id": run_id,
            "job_id": f"job-{run_id}",
            "model_id": model_id,
        },
        "governed_lineage": dict(LINEAGE),
        "backtest_config": _config_snapshot(model_id).as_metadata(),
        "accuracy_at_execution_lag": {"wape": 0.2},
        BACKTEST_EVIDENCE_METADATA_KEY: {
            "contract_version": 1,
            "model_id": model_id,
            "artifact_payload": {
                "checksum": "e" * 64,
                "row_count": 2,
                "size_bytes": 300,
            },
            "loaded_fact_payload": {
                "checksum": fact_checksum,
                "row_count": 2,
            },
        },
    }


def _row(
    model_id: str,
    run_id: int,
    *,
    status: str = "completed",
    loaded: bool = True,
    metadata: dict | None = None,
) -> tuple:
    return (
        model_id,
        run_id,
        status,
        loaded,
        Decimal("0.2"),
        Decimal("80.0"),
        datetime(2026, 7, 1, tzinfo=UTC),
        metadata if metadata is not None else _metadata(model_id, run_id),
    )


class _StreamingCursor:
    def __init__(self, batches: list[list[tuple]]) -> None:
        self._batches = iter(batches)
        self.execute_args: tuple | None = None
        self.fetchmany_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, *args) -> None:
        self.execute_args = args

    def fetchmany(self, size: int) -> list[tuple]:
        self.fetchmany_sizes.append(size)
        return next(self._batches, [])


class _StreamingConnection:
    def __init__(self, cursor: _StreamingCursor) -> None:
        self._cursor = cursor
        self.cursor_kwargs: dict | None = None

    def cursor(self, **kwargs):
        self.cursor_kwargs = kwargs
        return self._cursor


def test_csv_payload_stats_seal_exact_bytes_and_logical_row_count(tmp_path: Path) -> None:
    csv_path = tmp_path / "backtest_predictions.csv"
    csv_path.write_text('model_id,value\nmstl,"line one\nline two"\nmstl,2\n', encoding="utf-8")

    stats = compute_csv_payload_stats(csv_path)

    assert stats.row_count == 2
    assert stats.size_bytes == csv_path.stat().st_size
    assert len(stats.checksum) == 64


def test_fact_payload_stats_stream_rows_in_bounded_batches() -> None:
    rows = [
        (
            "ck-1",
            "item-a",
            "group-a",
            "loc-a",
            "2026-01-01",
            "2026-02-01",
            1,
            1,
            "10.0000",
            "9.0000",
            "mstl",
            None,
        ),
        (
            "ck-2",
            "item-b",
            "group-b",
            "loc-b",
            "2026-01-01",
            "2026-02-01",
            1,
            1,
            "11.0000",
            "8.0000",
            "mstl",
            None,
        ),
    ]
    cursor = _StreamingCursor([[rows[0]], [rows[1]], []])
    conn = _StreamingConnection(cursor)

    stats = compute_model_fact_payload_stats(conn, "mstl", batch_size=1)

    assert stats.row_count == 2
    assert len(stats.checksum) == 64
    assert cursor.fetchmany_sizes == [1, 1, 1]
    assert conn.cursor_kwargs and conn.cursor_kwargs.get("name")
    sql, params = cursor.execute_args or (None, None)
    assert "fact_external_forecast_monthly" in sql
    assert "ORDER BY forecast_ck" in sql
    assert params == ("mstl",)


def test_latest_governed_runs_require_exact_current_fact_and_config_evidence() -> None:
    conn = MagicMock()
    query_cur = MagicMock()
    query_cur.__enter__.return_value = query_cur
    query_cur.__exit__.return_value = False
    query_cur.fetchall.return_value = [
        _row(model_id, run_id)
        for run_id, model_id in enumerate(MODELS, start=51)
    ]
    conn.cursor.return_value = query_cur
    current_fact = PayloadStats(checksum="d" * 64, row_count=2)

    with (
        patch(
            "common.services.governed_backtest_evidence.compute_model_fact_payload_stats",
            return_value=current_fact,
        ),
        patch(
            "common.services.governed_backtest_evidence.load_backtest_config_snapshot",
            side_effect=_config_snapshot,
        ),
    ):
        runs = load_current_governed_backtest_runs(
            conn,
            MODELS,
            sales_lineage=SimpleNamespace(batch_id=91, source_hash="a" * 64),
            cluster_population=SimpleNamespace(
                experiment_id=35,
                assignment_count=13_968,
                assignment_checksum="b" * 64,
            ),
        )

    assert [run["model_id"] for run in runs] == list(MODELS)
    assert [run["backtest_run_id"] for run in runs] == list(range(51, 56))
    sql = query_cur.execute.call_args.args[0]
    assert "ORDER BY model_id, id DESC" in sql
    assert "status = 'completed'" not in sql


@pytest.mark.parametrize(
    ("row_override", "error"),
    [
        ({"status": "failed"}, "latest run"),
        ({"loaded": False}, "loaded"),
        ({"metadata": {}}, "governed"),
        (
            {"metadata": _metadata("mstl", 51, fact_checksum="f" * 64)},
            "fact payload",
        ),
    ],
)
def test_latest_failed_unloaded_ungoverned_or_mutated_fact_run_blocks(
    row_override: dict,
    error: str,
) -> None:
    model_id = "mstl"
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    cur.fetchall.return_value = [_row(model_id, 51, **row_override)]
    conn.cursor.return_value = cur

    with (
        patch(
            "common.services.governed_backtest_evidence.compute_model_fact_payload_stats",
            return_value=PayloadStats(checksum="d" * 64, row_count=2),
        ),
        patch(
            "common.services.governed_backtest_evidence.load_backtest_config_snapshot",
            return_value=_config_snapshot(model_id),
        ),
        pytest.raises(RuntimeError, match=error),
    ):
        load_current_governed_backtest_runs(
            conn,
            (model_id,),
            sales_lineage=SimpleNamespace(batch_id=91, source_hash="a" * 64),
            cluster_population=SimpleNamespace(
                experiment_id=35,
                assignment_count=13_968,
                assignment_checksum="b" * 64,
            ),
        )


def test_later_config_or_tuning_profile_change_blocks_governed_evidence() -> None:
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    cur.fetchall.return_value = [_row("lgbm_cluster", 51)]
    conn.cursor.return_value = cur
    changed = BacktestConfigSnapshot(
        model_id="lgbm_cluster",
        checksum="f" * 64,
        _config_json='{"cluster_tuning_profiles":{"metadata":{"cluster_experiment_id":36}}}',
    )

    with (
        patch(
            "common.services.governed_backtest_evidence.compute_model_fact_payload_stats",
            return_value=PayloadStats(checksum="d" * 64, row_count=2),
        ),
        patch(
            "common.services.governed_backtest_evidence.load_backtest_config_snapshot",
            return_value=changed,
        ),
        pytest.raises(RuntimeError, match="configuration"),
    ):
        load_current_governed_backtest_runs(
            conn,
            ("lgbm_cluster",),
            sales_lineage=SimpleNamespace(batch_id=91, source_hash="a" * 64),
            cluster_population=SimpleNamespace(
                experiment_id=35,
                assignment_count=13_968,
                assignment_checksum="b" * 64,
            ),
        )


def test_snapshot_roster_requires_exact_latest_runs_wape_and_deterministic_ranks() -> None:
    current = [
        {
            "model_id": model_id,
            "backtest_run_id": run_id,
            "wape": Decimal(wape),
            "accuracy_pct": Decimal("80"),
            "completed_at": datetime(2026, 7, 1, tzinfo=UTC),
        }
        for model_id, run_id, wape in (
            ("mstl", 51, "0.10"),
            ("nhits", 52, "0.20"),
            ("nbeats", 53, "0.30"),
            ("lgbm_cluster", 54, "0.40"),
            ("chronos2_enriched", 55, "0.50"),
        )
    ]
    roster = [
        {
            "model_id": "mstl",
            "snapshot_role": "contender",
            "contender_rank": 1,
            "backtest_run_id": 51,
            "wape": Decimal("0.10"),
        },
        {
            "model_id": "nhits",
            "snapshot_role": "contender",
            "contender_rank": 2,
            "backtest_run_id": 52,
            "wape": Decimal("0.20"),
        },
        {
            "model_id": "nbeats",
            "snapshot_role": "contender",
            "contender_rank": 3,
            "backtest_run_id": 53,
            "wape": Decimal("0.30"),
        },
    ]

    validate_snapshot_roster_provenance(roster, current)

    changed_wape = [dict(row) for row in roster]
    changed_wape[1]["wape"] = Decimal("0.2001")
    with pytest.raises(RuntimeError, match="WAPE"):
        validate_snapshot_roster_provenance(changed_wape, current)

    newer_run = [dict(run) for run in current]
    newer_run[0]["backtest_run_id"] = 61
    with pytest.raises(RuntimeError, match="latest governed run"):
        validate_snapshot_roster_provenance(roster, newer_run)
