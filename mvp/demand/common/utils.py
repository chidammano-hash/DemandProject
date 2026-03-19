"""Shared utility helpers for Supply Chain Command Center.

Consolidates commonly duplicated helpers:
- _ts(): Formatted timestamp for console logging
- load_config(): Thread-safe YAML config loader with caching
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Timestamp helper — used by backtest scripts, tuning, feature engineering,
# backtest framework, mlflow_utils, and cleanup scripts for console logging.
# ---------------------------------------------------------------------------


def _ts() -> str:
    """Return current time as HH:MM:SS for console log messages."""
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Thread-safe YAML config loader with per-file caching.
#
# Replaces the duplicated _config_cache / _load_config() / _reset_config()
# pattern found in notification_engine, cache, rate_limiter, dq_engine, and
# auth modules.  Each module used its own module-level `_config_cache: dict | None`
# global without any thread protection — a race condition under concurrent
# API requests.
# ---------------------------------------------------------------------------

_config_store: dict[str, dict] = {}
_config_lock = threading.Lock()

# Base directory for config files: mvp/demand/config/
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(name: str) -> dict:
    """Load and cache a YAML config file by name (thread-safe).

    Parameters
    ----------
    name : str
        Config file name with extension, e.g. ``"notification_config.yaml"``.
        Resolved relative to ``mvp/demand/config/``.

    Returns
    -------
    dict
        Parsed YAML content (empty dict if file is missing or empty).
    """
    cached = _config_store.get(name)
    if cached is not None:
        return cached

    with _config_lock:
        # Double-checked locking
        cached = _config_store.get(name)
        if cached is not None:
            return cached

        cfg_path = _CONFIG_DIR / name
        if cfg_path.exists():
            with open(cfg_path) as f:
                result = yaml.safe_load(f) or {}
        else:
            result = {}

        _config_store[name] = result
        return result


def reset_config(name: str | None = None) -> None:
    """Reset cached config(s). For use in tests only.

    Parameters
    ----------
    name : str or None
        If given, reset only that config file's cache.
        If None, reset all cached configs.
    """
    with _config_lock:
        if name is None:
            _config_store.clear()
        else:
            _config_store.pop(name, None)
