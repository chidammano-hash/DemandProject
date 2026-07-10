"""Unit tests for common/utils.py — shared utility helpers."""

import threading
import time
from unittest.mock import patch

import pytest
import yaml

from common.core.utils import _deep_merge
from common.core.utils import _ts, get_pipeline_config_path, load_config, reset_config

# ---------------------------------------------------------------------------
# _ts() tests
# ---------------------------------------------------------------------------

class TestTs:
    def test_returns_string(self):
        result = _ts()
        assert isinstance(result, str)

    def test_format_hms(self):
        result = _ts()
        # Should be HH:MM:SS
        parts = result.split(":")
        assert len(parts) == 3
        assert all(len(p) == 2 for p in parts)

    def test_matches_time_strftime(self):
        # Allow 1-second tolerance
        expected = time.strftime("%H:%M:%S")
        result = _ts()
        # Either they match or differ by at most 1 second
        assert result[:5] == expected[:5]  # HH:MM must match


# ---------------------------------------------------------------------------
# load_config() tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear config cache before and after each test."""
    reset_config()
    yield
    reset_config()


class TestLoadConfig:
    def test_returns_dict(self, tmp_path):
        cfg_path = tmp_path / "test_config.yaml"
        cfg_path.write_text(yaml.dump({"key": "value"}))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("test_config.yaml")
        assert result == {"key": "value"}

    def test_missing_file_returns_empty_dict(self, tmp_path):
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("nonexistent.yaml")
        assert result == {}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("empty.yaml")
        assert result == {}

    def test_caches_result(self, tmp_path):
        cfg_path = tmp_path / "cached.yaml"
        cfg_path.write_text(yaml.dump({"v": 1}))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            r1 = load_config("cached.yaml")
            # Overwrite file — cached result should NOT change
            cfg_path.write_text(yaml.dump({"v": 2}))
            r2 = load_config("cached.yaml")
        assert r1 == r2 == {"v": 1}

    def test_reset_clears_specific(self, tmp_path):
        cfg_path = tmp_path / "resettable.yaml"
        cfg_path.write_text(yaml.dump({"v": 1}))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            r1 = load_config("resettable.yaml")
            assert r1 == {"v": 1}

            cfg_path.write_text(yaml.dump({"v": 99}))
            reset_config("resettable.yaml")
            r2 = load_config("resettable.yaml")
        assert r2 == {"v": 99}

    def test_reset_all(self, tmp_path):
        cfg_path = tmp_path / "all_reset.yaml"
        cfg_path.write_text(yaml.dump({"v": 1}))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            load_config("all_reset.yaml")
            reset_config()  # reset all
            cfg_path.write_text(yaml.dump({"v": 42}))
            r = load_config("all_reset.yaml")
        assert r == {"v": 42}

    def test_thread_safety(self, tmp_path):
        """Multiple threads loading the same config should not corrupt the cache.

        Note: ``mock.patch`` is not thread-safe — patching from inside a worker
        races with __exit__ unwinding and can leave the module-level value
        pointing at a deleted tmpdir, polluting subsequent tests. We patch
        ONCE on the main thread and let workers race on the same target dir.
        """
        cfg_path = tmp_path / "threaded.yaml"
        cfg_path.write_text(yaml.dump({"threads": "ok"}))

        results = []
        errors = []

        def worker():
            try:
                results.append(load_config("threaded.yaml"))
            except Exception as e:
                errors.append(e)

        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            threads = [threading.Thread(target=worker) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        assert all(r == {"threads": "ok"} for r in results)

    def test_different_configs_independent(self, tmp_path):
        (tmp_path / "a.yaml").write_text(yaml.dump({"name": "a"}))
        (tmp_path / "b.yaml").write_text(yaml.dump({"name": "b"}))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            ra = load_config("a.yaml")
            rb = load_config("b.yaml")
        assert ra == {"name": "a"}
        assert rb == {"name": "b"}

    def test_includes_merges_base_config(self, tmp_path):
        """_includes directive deep-merges referenced configs as base layer."""
        (tmp_path / "shared.yaml").write_text(yaml.dump({
            "service_levels": {"A": 0.98, "B": 0.95},
            "z_table": {0.90: 1.282, 0.95: 1.645},
        }))
        (tmp_path / "consumer.yaml").write_text(yaml.dump({
            "_includes": ["shared"],
            "my_setting": 42,
        }))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("consumer.yaml")
        # Shared values are merged in
        assert result["service_levels"] == {"A": 0.98, "B": 0.95}
        assert result["z_table"] == {0.90: 1.282, 0.95: 1.645}
        # Own values are present
        assert result["my_setting"] == 42
        # _includes key is stripped
        assert "_includes" not in result

    def test_includes_override_takes_precedence(self, tmp_path):
        """Consumer values override included base values."""
        (tmp_path / "base.yaml").write_text(yaml.dump({
            "defaults": {"cost": 50, "moq": 1},
            "shared_val": "from_base",
        }))
        (tmp_path / "override.yaml").write_text(yaml.dump({
            "_includes": ["base"],
            "defaults": {"cost": 100},
            "own_val": "mine",
        }))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("override.yaml")
        # Deep-merged: cost overridden, moq inherited
        assert result["defaults"]["cost"] == 100
        assert result["defaults"]["moq"] == 1
        assert result["shared_val"] == "from_base"
        assert result["own_val"] == "mine"

    def test_includes_multiple_sources(self, tmp_path):
        """Multiple includes are merged left-to-right, consumer wins."""
        (tmp_path / "a.yaml").write_text(yaml.dump({"x": 1, "y": 2}))
        (tmp_path / "b.yaml").write_text(yaml.dump({"y": 3, "z": 4}))
        (tmp_path / "multi.yaml").write_text(yaml.dump({
            "_includes": ["a", "b"],
            "z": 99,
        }))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("multi.yaml")
        assert result["x"] == 1      # from a
        assert result["y"] == 3      # b overrides a
        assert result["z"] == 99     # consumer overrides b

    def test_includes_empty_list_is_noop(self, tmp_path):
        """Empty _includes list has no effect."""
        (tmp_path / "noinc.yaml").write_text(yaml.dump({
            "_includes": [],
            "val": 7,
        }))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("noinc.yaml")
        assert result == {"val": 7}

    def test_includes_missing_config_returns_empty(self, tmp_path):
        """Including a non-existent config merges empty dict (no error)."""
        (tmp_path / "missing_inc.yaml").write_text(yaml.dump({
            "_includes": ["nonexistent"],
            "val": 5,
        }))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("missing_inc.yaml")
        assert result == {"val": 5}

    def test_includes_recursive(self, tmp_path):
        """Included configs can themselves include other configs."""
        (tmp_path / "root.yaml").write_text(yaml.dump({"root_val": 1}))
        (tmp_path / "mid.yaml").write_text(yaml.dump({
            "_includes": ["root"],
            "mid_val": 2,
        }))
        (tmp_path / "leaf.yaml").write_text(yaml.dump({
            "_includes": ["mid"],
            "leaf_val": 3,
        }))
        with patch("common.core.utils._CONFIG_DIR", tmp_path):
            result = load_config("leaf.yaml")
        assert result["root_val"] == 1
        assert result["mid_val"] == 2
        assert result["leaf_val"] == 3

    def test_includes_with_pydantic_defaults_do_not_shadow(self, tmp_path):
        """Pydantic model defaults must not shadow values from _includes.

        Regression test: when a Pydantic model fills in a default for a
        field the consumer YAML omits, that default must not override the
        value provided by the included config.
        """
        from pydantic import BaseModel, Field
        from common.core.config_models import register_config_model, _config_validators

        class _InnerModel(BaseModel):
            service_levels: dict[str, float] = Field(default_factory=dict)
            min_days: int = Field(default=99)  # default differs from shared
            own_field: str = "hello"

        class _TestModel(BaseModel):
            section: _InnerModel = Field(default_factory=_InnerModel)

        (tmp_path / "shared_for_test.yaml").write_text(yaml.dump({
            "section": {"service_levels": {"A": 0.98}, "min_days": 5},
        }))
        (tmp_path / "validated.yaml").write_text(yaml.dump({
            "_includes": ["shared_for_test"],
            "section": {"own_field": "world"},
        }))

        name = "validated.yaml"
        _config_validators[name] = _TestModel
        try:
            with patch("common.core.utils._CONFIG_DIR", tmp_path):
                result = load_config(name)
            # Shared value wins over Pydantic default
            assert result["section"]["service_levels"] == {"A": 0.98}
            assert result["section"]["min_days"] == 5  # NOT the Pydantic default 99
            # Own value from the consumer file is preserved
            assert result["section"]["own_field"] == "world"
        finally:
            del _config_validators[name]

    def test_includes_preserves_extra_keys_with_pydantic(self, tmp_path):
        """Keys from _includes that are not in the Pydantic schema are preserved."""
        from pydantic import BaseModel, Field
        from common.core.config_models import register_config_model, _config_validators

        class _SmallModel(BaseModel):
            known: int = 0

        (tmp_path / "extra_shared.yaml").write_text(yaml.dump({
            "known": 10,
            "extra_from_shared": "preserved",
        }))
        (tmp_path / "with_extra.yaml").write_text(yaml.dump({
            "_includes": ["extra_shared"],
        }))

        name = "with_extra.yaml"
        _config_validators[name] = _SmallModel
        try:
            with patch("common.core.utils._CONFIG_DIR", tmp_path):
                result = load_config(name)
            assert result["known"] == 10
            assert result["extra_from_shared"] == "preserved"
        finally:
            del _config_validators[name]


# ---------------------------------------------------------------------------
# get_pipeline_config_path() tests
# ---------------------------------------------------------------------------

class TestGetPipelineConfigPath:
    def test_points_at_existing_file(self):
        # Regression: the helper once returned config/forecast_pipeline_config.yaml
        # after the file moved to config/forecasting/, breaking every open() caller
        # (tuning launch, promote, tuning-chat backtest) with FileNotFoundError.
        path = get_pipeline_config_path()
        assert path.exists(), f"pipeline config not found at {path}"

    def test_matches_config_dir_layout(self):
        path = get_pipeline_config_path()
        assert path.parts[-3:] == ("config", "forecasting", "forecast_pipeline_config.yaml")


# ---------------------------------------------------------------------------
# _deep_merge() tests
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_empty_base(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self):
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_both_empty(self):
        assert _deep_merge({}, {}) == {}

    def test_override_wins_scalar(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_deeply_nested_merge(self):
        base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
        override = {"l1": {"l2": {"l3": {"b": 99}}}}
        result = _deep_merge(base, override)
        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 99}}}}

    def test_override_replaces_non_dict_with_dict(self):
        result = _deep_merge({"a": 1}, {"a": {"nested": True}})
        assert result == {"a": {"nested": True}}

    def test_override_replaces_dict_with_scalar(self):
        result = _deep_merge({"a": {"nested": True}}, {"a": 42})
        assert result == {"a": 42}

    def test_does_not_mutate_inputs(self):
        base = {"x": {"a": 1}}
        override = {"x": {"b": 2}}
        _deep_merge(base, override)
        assert base == {"x": {"a": 1}}
        assert override == {"x": {"b": 2}}


# ---------------------------------------------------------------------------
# shared_constants integration tests
# ---------------------------------------------------------------------------

class TestSharedConstantsIntegration:
    """Verify that configs using _includes: [shared_constants] actually
    inherit the expected values rather than shadowing them with duplicates."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_safety_stock_inherits_service_levels(self):
        cfg = load_config("safety_stock_config")
        ss = cfg["safety_stock"]
        assert ss["service_levels"]["A"] == 0.98
        assert ss["service_levels"]["default"] == 0.95
        assert ss["z_table"][0.95] == 1.645
        assert ss["min_ss_days"] == 3
        assert ss["max_ss_days"] == 120
        assert ss["lt_std_fallback_pct"] == 0.20

    def test_eoq_inherits_financial_defaults(self):
        cfg = load_config("eoq_config")
        costs = cfg["costs"]
        assert costs["default_ordering_cost"] == 50.0
        assert costs["default_holding_cost_pct"] == 0.25
        assert costs["default_unit_cost"] == 1.0
        assert costs["default_moq"] == 1

    def test_financial_plan_inherits_carrying_cost(self):
        cfg = load_config("financial_plan_config")
        fp = cfg["financial_plan"]
        assert fp["carrying_cost_pct"] == 0.25

    def test_ai_planner_inherits_carrying_cost_rate(self):
        cfg = load_config("ai_planner_config")
        assert cfg["carrying_cost_rate"] == 0.25
        # default_unit_cost is intentionally overridden (10.0 vs shared 1.0)
        assert cfg["default_unit_cost"] == 10.0

    def test_service_level_inherits_plus_override(self):
        cfg = load_config("service_level_config")
        targets = cfg["service_level"]["targets_by_abc"]
        assert targets["A"] == 0.98
        assert targets["B"] == 0.95
        assert targets["C"] == 0.90
        assert targets["default"] == 0.95
        # X is a local override, not from shared constants
        assert targets["X"] == 0.92

    def test_rebalancing_inherits_carrying_cost(self):
        cfg = load_config("rebalancing_config")
        assert cfg["costs"]["carrying_cost_annual_pct"] == 0.25

    def test_replenishment_inherits_all_shared(self):
        cfg = load_config("replenishment_plan_config")
        rp = cfg["replenishment_plan"]
        assert rp["service_levels"]["A"] == 0.98
        assert rp["z_table"][0.95] == 1.645
        assert rp["min_ss_days"] == 3
        assert rp["costs"]["default_ordering_cost"] == 50.0
