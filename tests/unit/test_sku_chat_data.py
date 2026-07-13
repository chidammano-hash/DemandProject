"""Unit tests for the read-only SKU data functions (common/ai/sku_chat/sku_data.py)."""
from __future__ import annotations

from unittest.mock import MagicMock

from common.ai.sku_chat import sku_data


def _mock_pool(*, fetchall=None, fetchone=None, description=None):
    """Minimal sync pool mock matching `with pool.connection() as c, c.cursor() as cur`."""
    cursor = MagicMock()
    cursor.description = description if description is not None else [("col",)]
    cursor.fetchall.return_value = fetchall if fetchall is not None else []
    cursor.fetchone.return_value = fetchone

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, cursor


def test_search_skus_returns_results():
    pool, _ = _mock_pool(
        description=[("item_id",), ("customer_group",), ("loc",)],
        fetchall=[("100320", "RETAIL", "DC1"), ("100321", "RETAIL", "DC1")],
    )
    out = sku_data.search_skus(pool, "1003", limit=5)
    assert out["count"] == 2
    assert out["results"][0]["item_id"] == "100320"


def test_fetch_sku_profile_found():
    pool, _ = _mock_pool(
        description=[("item_id",), ("loc",), ("ml_cluster",)],
        fetchone=("100320", "DC1", "C3"),
    )
    out = sku_data.fetch_sku_profile(pool, "100320", "RETAIL", "DC1")
    assert out["found"] is True
    assert out["ml_cluster"] == "C3"


def test_fetch_sku_profile_not_found():
    pool, _ = _mock_pool(fetchone=None)
    out = sku_data.fetch_sku_profile(pool, "999", "RETAIL", "DC1")
    assert out["found"] is False
    assert out["item_id"] == "999"


def test_fetch_sku_sales_history_is_reversed_to_ascending():
    # DB returns DESC; function reverses to ascending.
    pool, _ = _mock_pool(
        description=[("month_start",), ("qty",), ("qty_shipped",), ("qty_ordered",)],
        fetchall=[
            ("2026-03-01", 30, 28, 31),
            ("2026-02-01", 20, 19, 22),
            ("2026-01-01", 10, 9, 11),
        ],
    )
    out = sku_data.fetch_sku_sales_history(pool, "100320", "DC1", months=12)
    assert out["months"] == 3
    assert out["history"][0]["month_start"] == "2026-01-01"
    assert out["history"][-1]["month_start"] == "2026-03-01"


def test_fetch_sku_accuracy_computes_wape_bias_accuracy():
    pool, _ = _mock_pool(
        description=[
            ("model_id",), ("lag",), ("sum_forecast",),
            ("sum_actual",), ("sum_abs_error",), ("row_count",),
        ],
        fetchall=[("lgbm_cluster", 1, 120.0, 100.0, 30.0, 12)],
    )
    out = sku_data.fetch_sku_accuracy(pool, "100320", "RETAIL", "DC1")
    m = out["metrics"][0]
    assert m["wape_pct"] == 30.0
    assert m["bias_pct"] == 20.0
    assert m["accuracy_pct"] == 70.0


def test_fetch_sku_accuracy_handles_zero_actual():
    pool, _ = _mock_pool(
        description=[
            ("model_id",), ("lag",), ("sum_forecast",),
            ("sum_actual",), ("sum_abs_error",), ("row_count",),
        ],
        fetchall=[("lgbm_cluster", 1, 5.0, 0.0, 5.0, 2)],
    )
    out = sku_data.fetch_sku_accuracy(pool, "100320", "RETAIL", "DC1")
    m = out["metrics"][0]
    assert m["wape_pct"] is None
    assert m["bias_pct"] is None
    assert m["accuracy_pct"] is None


def test_fetch_sku_cluster_peers():
    pool, _ = _mock_pool(
        description=[("item_id",), ("customer_group",), ("loc",)],
        fetchall=[("100321", "RETAIL", "DC1")],
    )
    out = sku_data.fetch_sku_cluster_peers(pool, "100320", "RETAIL", "DC1", limit=5)
    assert out["peer_count"] == 1
    assert out["peers"][0]["item_id"] == "100321"
