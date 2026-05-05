"""Unit tests for common/planning_date.py"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import common.core.planning_date as pd_module
from common.core.planning_date import _reset_cache, _resolve_date, get_planning_date


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the module-level cache before and after every test."""
    _reset_cache()
    yield
    _reset_cache()


@pytest.fixture()
def config_file(tmp_path):
    """Write a planning_config.yaml to a temp path and patch the module's CONFIG_PATH."""

    def _make(planning_date: str, use_system_date: bool = False):
        cfg = {
            "planning": {
                "planning_date": planning_date,
                "use_system_date": use_system_date,
            }
        }
        p = tmp_path / "planning_config.yaml"
        p.write_text(yaml.dump(cfg))
        return p

    return _make


# ---------------------------------------------------------------------------
# 1. Returns configured date when use_system_date is false
# ---------------------------------------------------------------------------

def test_returns_configured_date(config_file):
    cfg_path = config_file("2026-02-24", use_system_date=False)
    with patch.object(pd_module, "_CONFIG_PATH", cfg_path):
        result = get_planning_date()
    assert result == date(2026, 2, 24)


# ---------------------------------------------------------------------------
# 2. Returns date.today() when use_system_date is true in config
# ---------------------------------------------------------------------------

def test_use_system_date_true(config_file):
    cfg_path = config_file("2026-02-24", use_system_date=True)
    with patch.object(pd_module, "_CONFIG_PATH", cfg_path):
        result = get_planning_date()
    assert result == date.today()


# ---------------------------------------------------------------------------
# 3. PLANNING_DATE env var overrides config file
# ---------------------------------------------------------------------------

def test_planning_date_env_overrides_config(config_file, monkeypatch):
    cfg_path = config_file("2026-02-24", use_system_date=False)
    monkeypatch.setenv("PLANNING_DATE", "2025-11-01")
    with patch.object(pd_module, "_CONFIG_PATH", cfg_path):
        result = get_planning_date()
    assert result == date(2025, 11, 1)


# ---------------------------------------------------------------------------
# 4. USE_SYSTEM_DATE env var overrides config file
# ---------------------------------------------------------------------------

def test_use_system_date_env_overrides_config(config_file, monkeypatch):
    cfg_path = config_file("2026-02-24", use_system_date=False)
    monkeypatch.setenv("USE_SYSTEM_DATE", "true")
    with patch.object(pd_module, "_CONFIG_PATH", cfg_path):
        result = get_planning_date()
    assert result == date.today()


# ---------------------------------------------------------------------------
# 5. USE_SYSTEM_DATE env var beats PLANNING_DATE env var
# ---------------------------------------------------------------------------

def test_use_system_date_env_beats_planning_date_env(monkeypatch):
    monkeypatch.setenv("USE_SYSTEM_DATE", "1")
    monkeypatch.setenv("PLANNING_DATE", "2025-01-01")
    result = get_planning_date()
    assert result == date.today()


# ---------------------------------------------------------------------------
# 6. Invalid PLANNING_DATE env var raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_planning_date_env_raises(monkeypatch):
    monkeypatch.setenv("PLANNING_DATE", "not-a-date")
    with pytest.raises(ValueError, match="Invalid PLANNING_DATE env var"):
        get_planning_date()


# ---------------------------------------------------------------------------
# 7. Missing config file falls back to date.today()
# ---------------------------------------------------------------------------

def test_missing_config_falls_back_to_today(tmp_path):
    nonexistent = tmp_path / "nonexistent.yaml"
    with patch.object(pd_module, "_CONFIG_PATH", nonexistent):
        result = get_planning_date()
    assert result == date.today()


# ---------------------------------------------------------------------------
# 8. Config file missing planning_date key falls back to date.today()
# ---------------------------------------------------------------------------

def test_config_missing_planning_date_key(tmp_path):
    cfg_path = tmp_path / "planning_config.yaml"
    cfg_path.write_text(yaml.dump({"planning": {"use_system_date": False}}))
    with patch.object(pd_module, "_CONFIG_PATH", cfg_path):
        result = get_planning_date()
    assert result == date.today()


# ---------------------------------------------------------------------------
# 9. Config is cached (loaded only once)
# ---------------------------------------------------------------------------

def test_config_is_cached(config_file):
    cfg_path = config_file("2026-02-24")
    with patch.object(pd_module, "_CONFIG_PATH", cfg_path):
        r1 = get_planning_date()
        # Overwrite the config — cached result should NOT change
        cfg_path.write_text(yaml.dump({"planning": {"planning_date": "2020-01-01", "use_system_date": False}}))
        r2 = get_planning_date()
    assert r1 == r2 == date(2026, 2, 24)


# ---------------------------------------------------------------------------
# 10. _reset_cache() clears the cache so next call re-reads config
# ---------------------------------------------------------------------------

def test_reset_cache_re_reads_config(config_file):
    cfg_path = config_file("2026-02-24")
    with patch.object(pd_module, "_CONFIG_PATH", cfg_path):
        r1 = get_planning_date()
        assert r1 == date(2026, 2, 24)

        _reset_cache()
        # Now update config to a different date
        cfg_path.write_text(yaml.dump({"planning": {"planning_date": "2025-06-15", "use_system_date": False}}))
        r2 = get_planning_date()
    assert r2 == date(2025, 6, 15)


# ---------------------------------------------------------------------------
# 11. USE_SYSTEM_DATE env var accepts several truthy values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "Yes"])
def test_use_system_date_truthy_values(val, monkeypatch):
    monkeypatch.setenv("USE_SYSTEM_DATE", val)
    assert get_planning_date() == date.today()


# ---------------------------------------------------------------------------
# 12. Invalid planning_date in config raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_planning_date_in_config(tmp_path):
    cfg_path = tmp_path / "planning_config.yaml"
    cfg_path.write_text(yaml.dump({"planning": {"planning_date": "bad-date", "use_system_date": False}}))
    with patch.object(pd_module, "_CONFIG_PATH", cfg_path):
        with pytest.raises(ValueError, match="Invalid planning_date"):
            get_planning_date()
