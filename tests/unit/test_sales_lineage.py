"""Completed sales-source lineage used by persisted forecast artifacts."""

from unittest.mock import MagicMock

import pytest


def test_completed_sales_lineage_is_source_hash_bound() -> None:
    from common.services.sales_lineage import load_completed_sales_lineage

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.fetchone.return_value = (91, "A" * 64, "sku_lvl2_hist_clean.csv")
    conn = MagicMock()
    conn.cursor.return_value = cursor

    lineage = load_completed_sales_lineage(conn)

    assert lineage.batch_id == 91
    assert lineage.source_hash == "a" * 64
    query = cursor.execute.call_args.args[0]
    assert "domain = 'sales'" in query
    assert "status = 'completed'" in query
    assert "row_count_out > 0" in query


def test_completed_sales_lineage_rejects_unsynchronized_safe_upsert() -> None:
    from common.services.sales_lineage import load_completed_sales_lineage

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.fetchone.return_value = (92, "b" * 64, "safe_upsert")
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with pytest.raises(RuntimeError, match="canonical sales reload"):
        load_completed_sales_lineage(conn)


@pytest.mark.parametrize(
    "row",
    [
        None,
        (91, None, "sku_lvl2_hist_clean.csv"),
        (91, "not-a-sha", "sku_lvl2_hist_clean.csv"),
    ],
)
def test_completed_sales_lineage_fails_closed_without_valid_evidence(row) -> None:
    from common.services.sales_lineage import load_completed_sales_lineage

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.fetchone.return_value = row
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with pytest.raises(RuntimeError, match=r"completed sales load|SHA-256 source hash"):
        load_completed_sales_lineage(conn)
