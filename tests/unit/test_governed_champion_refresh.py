"""Governed champion refresh lifecycle tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from common.services.champion_refresh import (
    CANONICAL_CHAMPION_MODELS,
    ChampionRefreshSpec,
    build_refresh_spec,
    finalize_governed_champion_refresh,
    run_governed_champion_refresh,
    validate_backtest_lineage_rows,
)


def _pipeline_config() -> dict:
    return {
        "champion": {
            "strategy": "per_segment",
            "strategy_params": {"segment_strategy_map": {"smooth": {"strategy": "ensemble"}}},
            "meta_learner": {"model_type": "random_forest"},
            "models": list(CANONICAL_CHAMPION_MODELS),
            "metric": "wape",
            "lag": "execution",
            "min_dfu_rows": 3,
            "champion_model_id": "champion",
            "fallback_model_id": "lgbm_cluster",
        }
    }


def _spec(*, cluster_experiment_id: int = 35) -> ChampionRefreshSpec:
    return ChampionRefreshSpec(
        strategy="per_segment",
        strategy_params={"segment_strategy_map": {"smooth": {"strategy": "ensemble"}}},
        meta_learner_params={"model_type": "random_forest"},
        models=CANONICAL_CHAMPION_MODELS,
        metric="wape",
        lag_mode="execution",
        min_dfu_rows=3,
        champion_model_id="champion",
        fallback_model_id="lgbm_cluster",
        cluster_experiment_id=cluster_experiment_id,
        source_sales_batch_id=91,
        data_checksum="c" * 64,
        cluster_assignment_count=13_968,
        cluster_assignment_checksum="d" * 64,
        backtest_run_ids=tuple(
            (model_id, index)
            for index, model_id in enumerate(CANONICAL_CHAMPION_MODELS, start=51)
        ),
    )


def _mock_connection(fetch_rows: list[tuple | None]) -> tuple[MagicMock, MagicMock, MagicMock]:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    transaction = MagicMock()
    transaction.__enter__.return_value = transaction
    transaction.__exit__.return_value = False
    conn.transaction.return_value = transaction
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    cur.fetchone.side_effect = fetch_rows
    cur.rowcount = 1
    conn.cursor.return_value = cur
    return conn, cur, transaction


def test_build_refresh_spec_uses_exact_current_production_contract() -> None:
    spec = build_refresh_spec(
        _pipeline_config(),
        competing_models=list(CANONICAL_CHAMPION_MODELS),
        cluster_experiment_id=35,
    )

    assert spec.strategy == "per_segment"
    assert spec.models == CANONICAL_CHAMPION_MODELS
    assert spec.cluster_experiment_id == 35
    assert spec.metric == "wape"
    assert spec.lag_mode == "execution"
    assert spec.min_dfu_rows == 3


def test_backtest_lineage_requires_one_current_loaded_run_for_each_model() -> None:
    lineage = {
        "source_sales_batch_id": 91,
        "data_checksum": "c" * 64,
        "cluster_experiment_id": 35,
        "cluster_assignment_count": 13_968,
        "cluster_assignment_checksum": "d" * 64,
    }
    rows = [
        (model_id, run_id, {"governed_lineage": lineage})
        for run_id, model_id in enumerate(CANONICAL_CHAMPION_MODELS, start=51)
    ]

    run_ids = validate_backtest_lineage_rows(
        rows,
        models=CANONICAL_CHAMPION_MODELS,
        source_sales_batch_id=91,
        data_checksum="c" * 64,
        cluster_experiment_id=35,
        cluster_assignment_count=13_968,
        cluster_assignment_checksum="d" * 64,
    )

    assert run_ids == tuple(
        (model_id, run_id)
        for run_id, model_id in enumerate(CANONICAL_CHAMPION_MODELS, start=51)
    )


def test_backtest_lineage_rejects_stale_or_missing_model_before_experiment() -> None:
    lineage = {
        "source_sales_batch_id": 90,
        "data_checksum": "b" * 64,
        "cluster_experiment_id": 35,
        "cluster_assignment_count": 13_968,
        "cluster_assignment_checksum": "d" * 64,
    }
    rows = [
        (model_id, run_id, {"governed_lineage": lineage})
        for run_id, model_id in enumerate(CANONICAL_CHAMPION_MODELS[:-1], start=51)
    ]

    with pytest.raises(RuntimeError, match="current governed lineage"):
        validate_backtest_lineage_rows(
            rows,
            models=CANONICAL_CHAMPION_MODELS,
            source_sales_batch_id=91,
            data_checksum="c" * 64,
            cluster_experiment_id=35,
            cluster_assignment_count=13_968,
            cluster_assignment_checksum="d" * 64,
        )


@pytest.mark.parametrize(
    ("configured", "competing"),
    [
        (list(CANONICAL_CHAMPION_MODELS[:-1]), list(CANONICAL_CHAMPION_MODELS)),
        (list(CANONICAL_CHAMPION_MODELS), list(CANONICAL_CHAMPION_MODELS[:-1])),
    ],
)
def test_build_refresh_spec_rejects_partial_five_model_roster(
    configured: list[str],
    competing: list[str],
) -> None:
    cfg = _pipeline_config()
    cfg["champion"]["models"] = configured

    with pytest.raises(ValueError, match="exact canonical five-model roster"):
        build_refresh_spec(cfg, competing_models=competing, cluster_experiment_id=35)


def test_finalize_is_atomic_and_switches_both_promotions_only_after_rows_exist(
    tmp_path: Path,
) -> None:
    winners_csv = tmp_path / "experiment_82_winners.csv"
    winners_csv.write_text("winner evidence", encoding="utf-8")
    conn, cur, transaction = _mock_connection(
        [
            (
                "completed",
                list(CANONICAL_CHAMPION_MODELS),
                "per_segment",
                _spec().strategy_params,
                _spec().meta_learner_params,
                "wape",
                "execution",
                3,
                35,
                False,
                False,
                None,
                None,
                None,
            ),
            (35,),
            (81,),
        ]
    )
    winners_df = pd.DataFrame(
        [
            {
                "item_id": "A",
                "customer_group": "G",
                "loc": "1401-BULK",
                "startdate": pd.Timestamp("2026-01-01"),
                "model_id": "lgbm_cluster",
            }
        ]
    )
    stats = SimpleNamespace(
        checksum="f" * 64,
        row_count=12,
        dfu_count=1,
        source_model_count=1,
    )

    with (
        patch("common.services.champion_refresh._get_conn", return_value=conn),
        patch("common.services.champion_refresh.load_refresh_spec", return_value=_spec()),
        patch(
            "common.services.champion_refresh._load_cached_winners",
            return_value=(winners_df, [("A", "G", "1401-BULK", "2026-01-01", "lgbm_cluster")], False),
        ),
        patch("common.services.champion_refresh.insert_champion_forecasts", return_value=10),
        patch("common.services.champion_refresh.insert_fallback_champions", return_value=2),
        patch("common.services.champion_refresh.compute_ceiling_winners", return_value=[]),
        patch("common.services.champion_refresh.insert_ceiling_forecasts", return_value=0),
        patch("common.services.champion_refresh.compute_champion_results_stats", return_value=stats),
        patch("common.services.champion_refresh.sha256_file", return_value="a" * 64),
    ):
        result = finalize_governed_champion_refresh(
            82,
            job_id="job-82",
            winners_csv=winners_csv,
            expected_spec=_spec(),
            refresh_views=False,
        )

    sql_calls = [str(c.args[0]) for c in cur.execute.call_args_list]
    assert any("is_promoted = FALSE" in sql for sql in sql_calls)
    assert any("is_results_promoted = TRUE" in sql for sql in sql_calls)
    assert any("INSERT INTO champion_promotion_log" in sql for sql in sql_calls)
    assert result == {
        "experiment_id": 82,
        "previous_experiment_id": 81,
        "backtest_run_ids": dict(_spec().backtest_run_ids),
        "source_sales_batch_id": 91,
        "data_checksum": "c" * 64,
        "cluster_experiment_id": 35,
        "cluster_assignment_count": 13_968,
        "cluster_assignment_checksum": "d" * 64,
        "routing_artifact_checksum": "a" * 64,
        "results_forecast_checksum": "f" * 64,
        "results_forecast_row_count": 12,
        "champion_rows": 12,
        "ceiling_rows": 0,
        "already_promoted": False,
        "view_refresh": None,
    }
    transaction.__exit__.assert_called_once_with(None, None, None)


def test_failed_candidate_load_cannot_clear_incumbent_promotion(tmp_path: Path) -> None:
    winners_csv = tmp_path / "experiment_82_winners.csv"
    winners_csv.write_text("winner evidence", encoding="utf-8")
    conn, cur, transaction = _mock_connection(
        [
            (
                "completed",
                list(CANONICAL_CHAMPION_MODELS),
                "per_segment",
                _spec().strategy_params,
                _spec().meta_learner_params,
                "wape",
                "execution",
                3,
                35,
                False,
                False,
                None,
                None,
                None,
            ),
            (35,),
            (81,),
        ]
    )
    winners_df = pd.DataFrame(
        [
            {
                "item_id": "A",
                "customer_group": "G",
                "loc": "1401-BULK",
                "startdate": pd.Timestamp("2026-01-01"),
                "model_id": "mstl",
            }
        ]
    )

    with (
        patch("common.services.champion_refresh._get_conn", return_value=conn),
        patch("common.services.champion_refresh.load_refresh_spec", return_value=_spec()),
        patch(
            "common.services.champion_refresh._load_cached_winners",
            return_value=(winners_df, [("A", "G", "1401-BULK", "2026-01-01", "mstl")], False),
        ),
        patch(
            "common.services.champion_refresh.insert_champion_forecasts",
            side_effect=RuntimeError("candidate insert failed"),
        ),
        patch("common.services.champion_refresh.sha256_file", return_value="a" * 64),
    ):
        with pytest.raises(RuntimeError, match="candidate insert failed"):
            finalize_governed_champion_refresh(
                82,
                job_id="job-82",
                winners_csv=winners_csv,
                expected_spec=_spec(),
                refresh_views=False,
            )

    promotion_sql = "\n".join(str(c.args[0]) for c in cur.execute.call_args_list)
    assert "is_promoted = FALSE" not in promotion_sql
    assert "is_results_promoted = FALSE" not in promotion_sql
    assert transaction.__exit__.call_args.args[0] is RuntimeError


def test_runner_returns_experiment_and_promoted_lineage(tmp_path: Path) -> None:
    winners_csv = tmp_path / "experiment_82_winners.csv"
    final_result = {
        "experiment_id": 82,
        "results_forecast_checksum": "f" * 64,
        "results_forecast_row_count": 12,
    }

    with (
        patch("common.services.champion_refresh.load_refresh_spec", return_value=_spec()),
        patch("common.services.champion_refresh.create_governed_experiment", return_value=82),
        patch("common.services.champion_refresh.persist_job_experiment_id") as persist,
        patch("common.services.champion_refresh.run_champion_experiment_job") as run_experiment,
        patch(
            "common.services.champion_refresh.champion_winners_path",
            return_value=winners_csv,
        ),
        patch(
            "common.services.champion_refresh.finalize_governed_champion_refresh",
            return_value=final_result,
        ) as finalize,
    ):
        result = run_governed_champion_refresh(
            {},
            progress_cb=MagicMock(),
            job_id="job-82",
        )

    persist.assert_called_once_with("job-82", 82, _spec())
    run_experiment.assert_called_once_with(
        82,
        progress_cb=ANY,
        cancel_event=None,
        job_id="job-82",
    )
    finalize.assert_called_once_with(
        82,
        job_id="job-82",
        winners_csv=winners_csv,
        expected_spec=_spec(),
    )
    assert result["experiment_id"] == 82
    assert result["results_forecast_row_count"] == 12


@pytest.mark.parametrize("type_id", ["champion_select", "governed_champion_refresh"])
def test_recovery_finalizes_the_persisted_experiment_and_restores_job_result(
    type_id: str,
) -> None:
    from common.services.job_registry import JobManager
    from common.services.job_state import JobTypeDef

    type_def = JobTypeDef(
        type_id=type_id,
        label="Governed Champion Refresh",
        description="test",
        group="champion",
        callable=MagicMock(),
    )
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    result = {"experiment_id": 82, "results_forecast_row_count": 12}

    with (
        patch(
            "common.services.champion_refresh.finalize_governed_champion_refresh",
            return_value=result,
        ) as finalize,
        patch("common.services.job_registry._get_conn", return_value=conn),
    ):
        JobManager._finalize_recovered_job(
            "job-82",
            type_def,
            {
                "params": {
                    "experiment_id": 82,
                    "governed_spec": {
                        **_spec().__dict__,
                        "models": list(_spec().models),
                        "backtest_run_ids": [list(pair) for pair in _spec().backtest_run_ids],
                    },
                }
            },
        )

    finalize.assert_called_once_with(82, job_id="job-82", expected_spec=_spec())
    conn.execute.assert_called_once_with(
        "UPDATE job_history SET result = %s WHERE job_id = %s",
        (ANY, "job-82"),
    )


def test_standalone_champion_job_uses_only_governed_callable() -> None:
    from common.services.job_registry import JOB_TYPE_REGISTRY

    standalone = JOB_TYPE_REGISTRY["champion_select"]
    governed = JOB_TYPE_REGISTRY["governed_champion_refresh"]

    assert standalone.callable is run_governed_champion_refresh
    assert governed.callable is run_governed_champion_refresh


def test_make_champion_select_has_no_unscoped_mutation_path() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    target = makefile.split("champion-select:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]

    assert "scripts.ml.run_governed_champion_refresh" in target
    assert "run_champion_selection.py" not in target


def test_governed_cli_invokes_the_shared_service() -> None:
    from scripts.ml import run_governed_champion_refresh as cli

    with patch.object(
        cli,
        "run_governed_champion_refresh",
        return_value={"experiment_id": 82},
    ) as run:
        cli.main()

    run.assert_called_once_with({})
