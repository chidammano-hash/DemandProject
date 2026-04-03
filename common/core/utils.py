"""Shared utility helpers for Supply Chain Command Center.

Consolidates commonly duplicated helpers:
- _ts(): Formatted timestamp for console logging
- load_config(): Thread-safe YAML config loader with caching (with optional
  Pydantic validation when a model is registered in common.config_models)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

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

# Base directory for config files: config/
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_config(name: str) -> dict:
    """Load and cache a YAML config file by name (thread-safe).

    Parameters
    ----------
    name : str
        Config file name with extension, e.g. ``"notification_config.yaml"``.
        Resolved relative to ``config/``.

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
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}

        # Optional Pydantic validation — import lazily to avoid circular deps
        try:
            from common.core.config_models import _config_validators

            validator = _config_validators.get(name)
            if validator is not None:
                validated = validator(**raw)
                result = validated.model_dump()
            else:
                result = raw
        except ImportError:
            # config_models not available — skip validation
            result = raw
        except Exception:
            logger.warning(
                "Config validation failed for %s; returning raw dict", name,
                exc_info=True,
            )
            result = raw

        _config_store[name] = result
        return result


# ── Forecast pipeline config convenience helpers ─────────────────────────

_FORECAST_PIPELINE_CFG = "forecast_pipeline_config.yaml"


def load_forecast_pipeline_config() -> dict:
    """Load the master forecast pipeline config (cached)."""
    return load_config(_FORECAST_PIPELINE_CFG)


def get_algorithm_roster(*, stage: str | None = None) -> dict[str, dict]:
    """Return algorithm roster, optionally filtered to a lifecycle stage.

    Parameters
    ----------
    stage : str, optional
        One of ``"tune"``, ``"backtest"``, ``"compete"``, ``"forecast"``,
        ``"expert"``.  When provided, only algorithms where
        ``enabled=True`` AND ``<stage>=True`` are returned.
        When ``None``, returns all enabled algorithms.
    """
    cfg = load_forecast_pipeline_config()
    algorithms = cfg.get("algorithms", {})
    result = {}
    for model_id, entry in algorithms.items():
        if not entry.get("enabled", True):
            continue
        if stage is not None and not entry.get(stage, False):
            continue
        result[model_id] = entry
    return result


def get_competing_model_ids() -> list[str]:
    """Return model_ids that participate in champion selection."""
    return sorted(get_algorithm_roster(stage="compete").keys())


def get_forecastable_model_ids() -> list[str]:
    """Return model_ids eligible for production forecast."""
    return sorted(get_algorithm_roster(stage="forecast").keys())


def is_clustering_enabled() -> bool:
    """Check if clustering is enabled in the pipeline config."""
    cfg = load_forecast_pipeline_config()
    return cfg.get("clustering", {}).get("enabled", True)


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
