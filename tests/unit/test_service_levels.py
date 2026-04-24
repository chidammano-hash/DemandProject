"""Tests for unified service-level target resolver."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from common.core.service_levels import (
    _HARDCODED_FALLBACK,
    load_sl_targets_by_abc,
    resolve_sl_target,
)


def _cursor_with(rows):
    cur = MagicMock()
    cur.fetchall.return_value = rows
    return cur


def test_load_sl_targets_no_cursor_returns_yaml_defaults():
    """Without a DB cursor, fall back to YAML shared_constants values."""
    targets = load_sl_targets_by_abc(cursor=None)
    assert targets["A"] == pytest.approx(0.98)
    assert targets["B"] == pytest.approx(0.95)
    assert targets["C"] == pytest.approx(0.90)
    assert targets["default"] == pytest.approx(0.95)


def test_load_sl_targets_db_overrides_yaml():
    """DB rows override YAML values for matching ABC classes."""
    cur = _cursor_with([("A", 0.995), ("B", 0.97)])
    targets = load_sl_targets_by_abc(cursor=cur)
    assert targets["A"] == pytest.approx(0.995)
    assert targets["B"] == pytest.approx(0.97)
    # Non-overridden classes retain YAML values
    assert targets["C"] == pytest.approx(0.90)
    assert targets["default"] == pytest.approx(0.95)


def test_load_sl_targets_db_failure_falls_back_to_yaml():
    """Transient DB error must not break the resolver."""
    cur = MagicMock()
    cur.execute.side_effect = RuntimeError("relation does not exist")
    targets = load_sl_targets_by_abc(cursor=cur)
    assert targets["A"] == pytest.approx(0.98)
    assert targets["default"] == pytest.approx(0.95)


def test_load_sl_targets_empty_db_returns_yaml():
    """Empty DB table still yields a complete mapping."""
    cur = _cursor_with([])
    targets = load_sl_targets_by_abc(cursor=cur)
    assert "default" in targets
    assert targets["A"] == pytest.approx(0.98)


def test_resolve_sl_target_class_only():
    """With only ABC class, resolve via pre-loaded targets."""
    targets = {"A": 0.99, "B": 0.95, "default": 0.90}
    assert resolve_sl_target("A", targets_by_abc=targets) == pytest.approx(0.99)
    assert resolve_sl_target("X", targets_by_abc=targets) == pytest.approx(0.90)
    assert resolve_sl_target(None, targets_by_abc=targets) == pytest.approx(0.90)


def test_resolve_sl_target_sku_override_wins():
    """A matching (item_id, loc) row beats class-level defaults."""
    cur = MagicMock()
    cur.fetchone.return_value = (0.9999,)

    target = resolve_sl_target(
        "A",
        item_id="SKU-1",
        loc="DC-1",
        cursor=cur,
        targets_by_abc={"A": 0.98, "default": 0.95},
    )
    assert target == pytest.approx(0.9999)


def test_resolve_sl_target_sku_missing_falls_back_to_class():
    """When SKU row does not exist, use the pre-loaded class default."""
    cur = MagicMock()
    cur.fetchone.return_value = None

    target = resolve_sl_target(
        "B",
        item_id="SKU-99",
        loc="DC-9",
        cursor=cur,
        targets_by_abc={"A": 0.98, "B": 0.95, "default": 0.90},
    )
    assert target == pytest.approx(0.95)


def test_hardcoded_fallback_matches_yaml():
    """Hardcoded last-resort values must match current YAML defaults."""
    assert _HARDCODED_FALLBACK["A"] == pytest.approx(0.98)
    assert _HARDCODED_FALLBACK["B"] == pytest.approx(0.95)
    assert _HARDCODED_FALLBACK["C"] == pytest.approx(0.90)
    assert _HARDCODED_FALLBACK["default"] == pytest.approx(0.95)
