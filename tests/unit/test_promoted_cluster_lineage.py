"""Promoted clustering lineage used by production model artifacts."""

from unittest.mock import MagicMock

import pytest


def _connection(rows):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.fetchmany.side_effect = [rows, []]
    cursor.fetchall.side_effect = AssertionError("cluster lineage must stream in bounded batches")
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


def test_load_promoted_cluster_population_requires_one_experiment() -> None:
    from common.services.cluster_lineage import load_promoted_cluster_population

    conn, cursor = _connection(
        [(17, "sku-1", "0"), (17, "sku-2", "high_volume")]
    )

    population = load_promoted_cluster_population(conn)

    assert population.experiment_id == 17
    assert population.cluster_labels == frozenset({"0", "high_volume"})
    assert population.assignment_count == 2
    assert len(population.assignment_checksum) == 64
    assert "current_sku_cluster_assignment" in cursor.execute.call_args.args[0]
    conn.cursor.assert_called_once_with(name="forecast_cluster_lineage")
    cursor.fetchall.assert_not_called()
    assert cursor.fetchmany.call_args.args == (10_000,)


def test_assignment_checksum_changes_when_one_sku_is_reassigned() -> None:
    from common.services.cluster_lineage import load_promoted_cluster_population

    before, _ = _connection(
        [(17, "sku-1", "0"), (17, "sku-2", "1"), (17, "sku-3", "1")]
    )
    after, _ = _connection(
        [(17, "sku-1", "1"), (17, "sku-2", "0"), (17, "sku-3", "1")]
    )

    before_population = load_promoted_cluster_population(before)
    after_population = load_promoted_cluster_population(after)

    assert before_population.cluster_labels == after_population.cluster_labels
    assert before_population.assignment_checksum != after_population.assignment_checksum


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "required"),
        ([(17, "sku-1", "0"), (18, "sku-2", "1")], "exactly one"),
        ([(17, "sku-1", None)], "non-empty"),
        ([(17, "sku-1", "")], "non-empty"),
        ([(17, None, "0")], "sku_ck"),
        ([(17, "sku-1", "0"), (17, "sku-1", "1")], "duplicate sku_ck"),
    ],
)
def test_load_promoted_cluster_population_fails_closed(rows, message: str) -> None:
    from common.services.cluster_lineage import load_promoted_cluster_population

    conn, _cursor = _connection(rows)

    with pytest.raises(RuntimeError, match=message):
        load_promoted_cluster_population(conn)
