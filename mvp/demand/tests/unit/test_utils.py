"""Unit tests for common/utils.py — shared utility helpers."""

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from common.utils import _ts, load_config, reset_config, _config_store, _config_lock


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
        with patch("common.utils._CONFIG_DIR", tmp_path):
            result = load_config("test_config.yaml")
        assert result == {"key": "value"}

    def test_missing_file_returns_empty_dict(self, tmp_path):
        with patch("common.utils._CONFIG_DIR", tmp_path):
            result = load_config("nonexistent.yaml")
        assert result == {}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        with patch("common.utils._CONFIG_DIR", tmp_path):
            result = load_config("empty.yaml")
        assert result == {}

    def test_caches_result(self, tmp_path):
        cfg_path = tmp_path / "cached.yaml"
        cfg_path.write_text(yaml.dump({"v": 1}))
        with patch("common.utils._CONFIG_DIR", tmp_path):
            r1 = load_config("cached.yaml")
            # Overwrite file — cached result should NOT change
            cfg_path.write_text(yaml.dump({"v": 2}))
            r2 = load_config("cached.yaml")
        assert r1 == r2 == {"v": 1}

    def test_reset_clears_specific(self, tmp_path):
        cfg_path = tmp_path / "resettable.yaml"
        cfg_path.write_text(yaml.dump({"v": 1}))
        with patch("common.utils._CONFIG_DIR", tmp_path):
            r1 = load_config("resettable.yaml")
            assert r1 == {"v": 1}

            cfg_path.write_text(yaml.dump({"v": 99}))
            reset_config("resettable.yaml")
            r2 = load_config("resettable.yaml")
        assert r2 == {"v": 99}

    def test_reset_all(self, tmp_path):
        cfg_path = tmp_path / "all_reset.yaml"
        cfg_path.write_text(yaml.dump({"v": 1}))
        with patch("common.utils._CONFIG_DIR", tmp_path):
            load_config("all_reset.yaml")
            reset_config()  # reset all
            cfg_path.write_text(yaml.dump({"v": 42}))
            r = load_config("all_reset.yaml")
        assert r == {"v": 42}

    def test_thread_safety(self, tmp_path):
        """Multiple threads loading the same config should not corrupt the cache."""
        cfg_path = tmp_path / "threaded.yaml"
        cfg_path.write_text(yaml.dump({"threads": "ok"}))

        results = []
        errors = []

        def worker():
            try:
                with patch("common.utils._CONFIG_DIR", tmp_path):
                    r = load_config("threaded.yaml")
                    results.append(r)
            except Exception as e:
                errors.append(e)

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
        with patch("common.utils._CONFIG_DIR", tmp_path):
            ra = load_config("a.yaml")
            rb = load_config("b.yaml")
        assert ra == {"name": "a"}
        assert rb == {"name": "b"}
