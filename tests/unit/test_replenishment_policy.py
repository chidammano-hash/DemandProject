"""Unit tests for scripts/assign_replenishment_policies.py — IPfeature5."""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from scripts.inventory.assign_replenishment_policies import (
    load_config,
    determine_policy_id,
    upsert_policies,
    auto_assign_dfus,
)


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "replenishment_policy_config.yaml"
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_four_policies(self):
        config = load_config(CONFIG_PATH)
        assert len(config["policies"]) == 4

    def test_policy_ids_present(self):
        config = load_config(CONFIG_PATH)
        ids = [p["id"] for p in config["policies"]]
        assert "A_continuous_v1" in ids
        assert "B_periodic_v1" in ids
        assert "C_min_max_v1" in ids
        assert "lumpy_manual_v1" in ids

    def test_policy_types_are_valid(self):
        config = load_config(CONFIG_PATH)
        valid = {"continuous_rop", "periodic_review", "min_max", "manual"}
        for p in config["policies"]:
            assert p["type"] in valid

    def test_auto_assign_enabled(self):
        config = load_config(CONFIG_PATH)
        assert config["auto_assign"]["enabled"] is True

    def test_variability_override_lumpy(self):
        config = load_config(CONFIG_PATH)
        override = config["auto_assign"]["variability_override"]
        assert "lumpy" in override
        assert override["lumpy"] == "lumpy_manual_v1"


# ---------------------------------------------------------------------------
# determine_policy_id
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg():
    return load_config(CONFIG_PATH)


class TestDeterminePolicy:
    def test_abc_A_low_variability_returns_continuous(self, cfg):
        result = determine_policy_id("A", "low", cfg)
        assert result == "A_continuous_v1"

    def test_abc_B_medium_variability_returns_periodic(self, cfg):
        result = determine_policy_id("B", "medium", cfg)
        assert result == "B_periodic_v1"

    def test_abc_C_high_variability_returns_min_max(self, cfg):
        result = determine_policy_id("C", "high", cfg)
        assert result == "C_min_max_v1"

    def test_lumpy_variability_overrides_abc_A(self, cfg):
        """lumpy variability_class should override ABC=A and return lumpy policy."""
        result = determine_policy_id("A", "lumpy", cfg)
        assert result == "lumpy_manual_v1"

    def test_lumpy_variability_overrides_abc_C(self, cfg):
        """lumpy variability_class should override ABC=C and return lumpy policy."""
        result = determine_policy_id("C", "lumpy", cfg)
        assert result == "lumpy_manual_v1"

    def test_lumpy_variability_with_no_abc(self, cfg):
        """lumpy variability with NULL abc_vol still maps to lumpy policy."""
        result = determine_policy_id(None, "lumpy", cfg)
        assert result == "lumpy_manual_v1"

    def test_unknown_abc_vol_returns_none(self, cfg):
        """Unrecognized ABC class should return None (skip)."""
        result = determine_policy_id("Z", "low", cfg)
        assert result is None

    def test_none_abc_and_none_variability_returns_none(self, cfg):
        result = determine_policy_id(None, None, cfg)
        assert result is None

    def test_abc_lowercase_matches(self, cfg):
        """abc_vol comparison should be case-insensitive."""
        result = determine_policy_id("a", "low", cfg)
        assert result == "A_continuous_v1"

    def test_unknown_variability_falls_through_to_abc(self, cfg):
        """Non-lumpy variability class not in override → falls through to ABC match."""
        result = determine_policy_id("B", "unknown_class", cfg)
        assert result == "B_periodic_v1"


# ---------------------------------------------------------------------------
# upsert_policies (dry-run — no actual DB)
# ---------------------------------------------------------------------------

class TestUpsertPolicies:
    def test_dry_run_returns_count_without_executing(self, cfg):
        conn = MagicMock()
        count = upsert_policies(conn, cfg["policies"], dry_run=True)
        assert count == 4
        # No execute calls in dry-run
        conn.cursor.return_value.__enter__.return_value.execute.assert_not_called()

    def test_upsert_executes_for_each_policy(self, cfg):
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn = MagicMock()
        conn.cursor.return_value = cursor

        upsert_policies(conn, cfg["policies"], dry_run=False)
        assert cursor.execute.call_count == len(cfg["policies"])
        conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# auto_assign_dfus
# ---------------------------------------------------------------------------

class TestAutoAssignDfus:
    def test_dry_run_counts_without_db_writes(self, cfg):
        dfus = [
            {"item_id": "I1", "loc": "L1", "abc_vol": "A", "variability_class": "low"},
            {"item_id": "I2", "loc": "L2", "abc_vol": "B", "variability_class": "medium"},
            {"item_id": "I3", "loc": "L3", "abc_vol": None, "variability_class": None},
        ]
        conn = MagicMock()
        result = auto_assign_dfus(conn, dfus, cfg, dry_run=True)
        assert result["assigned"] == 2
        assert result["skipped"] == 1
        # No DB operations in dry-run
        conn.cursor.assert_not_called()

    def test_unknown_abc_skipped(self, cfg):
        dfus = [
            {"item_id": "I1", "loc": "L1", "abc_vol": "X", "variability_class": None},
        ]
        conn = MagicMock()
        result = auto_assign_dfus(conn, dfus, cfg, dry_run=True)
        assert result["skipped"] == 1
        assert result["assigned"] == 0
