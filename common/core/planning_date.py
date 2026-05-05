"""
common/planning_date.py

Provides get_planning_date() — a single source of truth for "today's date"
across all scripts and API routers.

Priority (highest → lowest):
  1. USE_SYSTEM_DATE env var  — if truthy, returns date.today()
  2. PLANNING_DATE env var    — if set, parses as ISO date (YYYY-MM-DD)
  3. config/planning_config.yaml
       use_system_date: true  → date.today()
       planning_date: "..."   → parsed ISO date
  4. Fallback                 — date.today()

Config is loaded lazily on the first call and cached for the process lifetime.
Call _reset_cache() in tests to clear between test cases.
"""

from __future__ import annotations

import os
import threading
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from common.core.paths import CONFIG_DIR  # noqa: E402

_CONFIG_PATH = CONFIG_DIR / "planning_config.yaml"

# Module-level cache — guarded by _cache_lock for thread safety
_cached_date: Optional[date] = None
_cache_loaded: bool = False
_cache_lock = threading.Lock()


def _load_config() -> dict:
    """Load planning_config.yaml. Returns empty dict if file is missing."""
    try:
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def get_planning_date() -> date:
    """
    Return the configured planning date for date-sensitive operations.

    Respects the priority order documented in the module docstring.
    Result is cached for the process lifetime; use _reset_cache() in tests.
    Thread-safe via double-checked locking.
    """
    if _cache_loaded:
        return _cached_date  # type: ignore[return-value]

    with _cache_lock:
        # Double-checked locking
        if _cache_loaded:
            return _cached_date  # type: ignore[return-value]

        resolved = _resolve_date()
        _set_cache(resolved)
        return resolved


def _set_cache(resolved: date) -> None:
    """Set the cache values. Must be called under _cache_lock."""
    global _cached_date, _cache_loaded
    _cached_date = resolved
    _cache_loaded = True


def _resolve_date() -> date:
    """Resolve the planning date without caching (used internally and in tests)."""
    # 1. USE_SYSTEM_DATE env var
    use_system_env = os.environ.get("USE_SYSTEM_DATE", "").strip().lower()
    if use_system_env in ("1", "true", "yes"):
        return date.today()

    # 2. PLANNING_DATE env var
    planning_date_env = os.environ.get("PLANNING_DATE", "").strip()
    if planning_date_env:
        try:
            return date.fromisoformat(planning_date_env)
        except ValueError as exc:
            raise ValueError(
                f"Invalid PLANNING_DATE env var '{planning_date_env}'. "
                "Expected ISO format YYYY-MM-DD."
            ) from exc

    # 3. Config file
    config = _load_config()
    planning_cfg = config.get("planning", {})

    if planning_cfg.get("use_system_date", False):
        return date.today()

    planning_date_str = planning_cfg.get("planning_date")
    if planning_date_str:
        try:
            return date.fromisoformat(str(planning_date_str))
        except ValueError as exc:
            raise ValueError(
                f"Invalid planning_date '{planning_date_str}' in planning_config.yaml. "
                "Expected ISO format YYYY-MM-DD."
            ) from exc

    # 4. Fallback
    return date.today()


def _reset_cache() -> None:
    """Reset the module-level cache. For use in tests only."""
    global _cached_date, _cache_loaded
    with _cache_lock:
        _cached_date = None
        _cache_loaded = False
