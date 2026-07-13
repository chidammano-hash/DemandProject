"""Fail-closed current-cluster tuning profile contracts."""

from unittest.mock import MagicMock, patch

import psycopg
import pytest


def _connection(*, fetchall: list[list[tuple]], fetchone: list[tuple | None] | None = None):
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchall.side_effect = fetchall
    if fetchone is not None:
        cursor.fetchone.side_effect = fetchone
    return connection, cursor


def _profiles(*, experiment_id: int = 35) -> dict:
    return {
        "enabled": True,
        "metadata": {"cluster_experiment_id": experiment_id},
        "cluster_profiles": {
            "steady_tuned": {
                "match_criteria": {"cluster_name": "steady"},
                "overrides": {"learning_rate": 0.1},
            },
            "intermittent_tuned": {
                "match_criteria": {"cluster_name": "intermittent"},
                "overrides": {"learning_rate": 0.05},
            },
            "default": {"match_criteria": {}, "overrides": {}},
        },
    }


def test_backtest_tuning_profile_validation_fails_on_database_error() -> None:
    from common.ml.backtest_framework import validate_cluster_tuning_profiles

    with (
        patch("common.ml.backtest_framework.load_config", return_value=_profiles()),
        patch(
            "common.ml.backtest_framework.psycopg.connect",
            side_effect=psycopg.OperationalError("database unavailable"),
        ),
        pytest.raises(RuntimeError, match="could not be verified"),
    ):
        validate_cluster_tuning_profiles({})


def test_backtest_tuning_profile_validation_rejects_stale_current_label() -> None:
    from common.ml.backtest_framework import validate_cluster_tuning_profiles

    connection, _cursor = _connection(
        fetchall=[[("intermittent",), ("steady",)], [("steady",)], [(35,)]],
    )
    with (
        patch("common.ml.backtest_framework.load_config", return_value=_profiles()),
        patch("common.ml.backtest_framework.psycopg.connect", return_value=connection),
        pytest.raises(RuntimeError, match=r"stale.*steady"),
    ):
        validate_cluster_tuning_profiles({})


def test_backtest_tuning_profile_validation_rejects_mismatched_current_labels() -> None:
    from common.ml.backtest_framework import validate_cluster_tuning_profiles

    connection, _cursor = _connection(
        fetchall=[[("new_cluster",), ("steady",)], [], [(35,)]],
    )
    with (
        patch("common.ml.backtest_framework.load_config", return_value=_profiles()),
        patch("common.ml.backtest_framework.psycopg.connect", return_value=connection),
        pytest.raises(RuntimeError, match="do not exactly match current cluster labels"),
    ):
        validate_cluster_tuning_profiles({})


def test_backtest_tuning_profile_validation_ignores_retired_extra_labels() -> None:
    from common.ml.backtest_framework import validate_cluster_tuning_profiles

    connection, _cursor = _connection(
        fetchall=[[('steady',)], [], [(35,)]],
    )
    with (
        patch("common.ml.backtest_framework.load_config", return_value=_profiles()),
        patch("common.ml.backtest_framework.psycopg.connect", return_value=connection),
    ):
        validate_cluster_tuning_profiles({})


def test_fetch_stale_clusters_fails_closed_on_query_error() -> None:
    from scripts.ml.tune_cluster_hyperparams import fetch_stale_clusters

    with (
        patch(
            "scripts.ml.tune_cluster_hyperparams.psycopg.connect",
            side_effect=psycopg.OperationalError("database unavailable"),
        ),
        pytest.raises(RuntimeError, match="stale tuning profiles could not be queried"),
    ):
        fetch_stale_clusters({})


def test_clear_stale_flags_fails_when_database_does_not_clear_every_profile() -> None:
    from scripts.ml.tune_cluster_hyperparams import clear_stale_flags

    connection, cursor = _connection(fetchall=[])
    cursor.rowcount = 1
    with (
        patch(
            "scripts.ml.tune_cluster_hyperparams.psycopg.connect",
            return_value=connection,
        ),
        pytest.raises(RuntimeError, match="cleared 1 of 2"),
    ):
        clear_stale_flags({}, ["steady", "intermittent"])
