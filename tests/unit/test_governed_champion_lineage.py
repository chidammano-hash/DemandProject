"""Governed champion promotion lineage contract tests."""

from unittest.mock import MagicMock

import pytest

from common.services.champion_lineage import (
    CANONICAL_CHAMPION_MODELS,
    GovernedChampionLineageError,
    load_active_governed_champion_lineage,
    load_governed_champion_lineage,
)


def _snapshot(**overrides):
    value = {
        "_promotion_mode": "governed_atomic_refresh",
        "models": list(CANONICAL_CHAMPION_MODELS),
        "source_sales_batch_id": 91,
        "data_checksum": "a" * 64,
        "cluster_experiment_id": 35,
        "cluster_assignment_count": 13_968,
        "cluster_assignment_checksum": "b" * 64,
        "backtest_run_ids": [
            [model_id, index]
            for index, model_id in enumerate(CANONICAL_CHAMPION_MODELS, start=101)
        ],
    }
    value.update(overrides)
    return value


def test_loads_exact_governed_champion_lineage_from_promotion_audit():
    cur = MagicMock()
    cur.fetchone.return_value = (_snapshot(),)

    lineage = load_governed_champion_lineage(cur, experiment_id=82)

    assert lineage["experiment_id"] == 82
    assert lineage["source_sales_batch_id"] == 91
    assert lineage["models"] == list(CANONICAL_CHAMPION_MODELS)
    assert set(lineage["backtest_run_ids"]) == set(CANONICAL_CHAMPION_MODELS)


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        _snapshot(_promotion_mode="manual"),
        _snapshot(source_sales_batch_id=0),
        _snapshot(data_checksum="bad"),
        _snapshot(models=["lgbm_cluster"]),
        _snapshot(backtest_run_ids=[["lgbm_cluster", 101]]),
    ],
)
def test_rejects_incomplete_or_unscoped_champion_promotion(snapshot):
    cur = MagicMock()
    cur.fetchone.return_value = (snapshot,)

    with pytest.raises(GovernedChampionLineageError):
        load_governed_champion_lineage(cur, experiment_id=82)


def test_requires_one_promotion_audit_for_the_active_experiment():
    cur = MagicMock()
    cur.fetchone.return_value = None

    with pytest.raises(GovernedChampionLineageError, match="promotion audit"):
        load_governed_champion_lineage(cur, experiment_id=82)


def test_active_lineage_requires_exactly_one_promoted_results_experiment():
    cur = MagicMock()
    cur.fetchall.return_value = [(82,)]
    cur.fetchone.return_value = (_snapshot(),)

    lineage = load_active_governed_champion_lineage(cur)

    assert lineage["experiment_id"] == 82


@pytest.mark.parametrize("rows", [[], [(81,), (82,)]])
def test_active_lineage_rejects_zero_or_multiple_promotions(rows):
    cur = MagicMock()
    cur.fetchall.return_value = rows

    with pytest.raises(GovernedChampionLineageError, match="Exactly one"):
        load_active_governed_champion_lineage(cur)
