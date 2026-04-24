"""Tests for the extended make_pool() factory (Gen-4 Stream J).

Verifies that the new ``fetchall_returns`` / ``fetchone_returns`` kwargs wire
through to ``cursor.<x>.side_effect`` while the legacy scalar kwargs still
work unchanged.
"""
from __future__ import annotations

from tests.api.conftest import make_pool


def test_make_pool_legacy_scalar_fetchall():
    """Legacy single-value ``fetchall_return`` still works."""
    pool, conn, cursor = make_pool(fetchall_return=[(1, "a"), (2, "b")])
    assert cursor.fetchall() == [(1, "a"), (2, "b")]
    assert cursor.fetchall() == [(1, "a"), (2, "b")]  # repeatable


def test_make_pool_legacy_scalar_fetchone():
    """Legacy single-value ``fetchone_return`` still works."""
    pool, conn, cursor = make_pool(fetchone_return=(42,))
    assert cursor.fetchone() == (42,)


def test_make_pool_multi_fetchall_side_effect():
    """``fetchall_returns`` wires up sequential per-call returns."""
    pool, conn, cursor = make_pool(
        fetchall_returns=[[("a",)], [("b",)], [("c",)]],
    )
    assert cursor.fetchall() == [("a",)]
    assert cursor.fetchall() == [("b",)]
    assert cursor.fetchall() == [("c",)]


def test_make_pool_multi_fetchone_side_effect():
    """``fetchone_returns`` wires up sequential per-call returns."""
    pool, conn, cursor = make_pool(
        fetchone_returns=[(1,), (2,), (3,)],
    )
    assert cursor.fetchone() == (1,)
    assert cursor.fetchone() == (2,)
    assert cursor.fetchone() == (3,)


def test_make_pool_multi_precedence_over_scalar():
    """``fetchall_returns`` takes precedence over ``fetchall_return``."""
    pool, conn, cursor = make_pool(
        fetchall_return=[("scalar",)],
        fetchall_returns=[[("first",)], [("second",)]],
    )
    assert cursor.fetchall() == [("first",)]
    assert cursor.fetchall() == [("second",)]


def test_make_pool_returns_triple():
    """Signature still returns (pool, conn, cursor)."""
    result = make_pool()
    assert len(result) == 3
