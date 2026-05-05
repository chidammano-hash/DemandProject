"""Shared utility helpers for Supply Chain Command Center.

Consolidates commonly duplicated helpers:
- _ts(): Formatted timestamp for console logging
- load_config(): Thread-safe YAML config loader with caching (with optional
  Pydantic validation when a model is registered in common.core.config_models)
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
_config_lock = threading.RLock()

# Base directory for config files: config/
from common.core.paths import CONFIG_DIR as _CONFIG_DIR  # noqa: E402


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge *override* into *base*.  Override values take precedence.

    For nested dicts, merging is recursive.  For all other types (lists,
    scalars) the override value replaces the base value entirely.
    """
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(name: str) -> dict:
    """Load and cache a YAML config file by name (thread-safe).

    Parameters
    ----------
    name : str
        Config file name, e.g. ``"notification_config.yaml"`` or
        ``"notification_config"`` (``.yaml`` is appended automatically
        when no ``.yaml`` / ``.yml`` extension is present).
        Resolved relative to ``config/``.

    Supports an ``_includes`` directive: when the YAML root contains an
    ``_includes`` key whose value is a list of config names, each listed
    config is loaded (recursively) and deep-merged as a base layer.
    The current file's own values take precedence over included values.
    The ``_includes`` key is stripped from the returned dict.

    Returns
    -------
    dict
        Parsed YAML content (empty dict if file is missing or empty).
    """
    # Auto-append .yaml when the caller omits the extension
    if not (name.endswith(".yaml") or name.endswith(".yml")):
        name = f"{name}.yaml"

    cached = _config_store.get(name)
    if cached is not None:
        return cached

    with _config_lock:
        # Double-checked locking
        cached = _config_store.get(name)
        if cached is not None:
            return cached

        cfg_path = _CONFIG_DIR / name
        if not cfg_path.exists():
            # Fallback: search domain subdirs (config/forecasting/, config/inventory/, etc.)
            matches = list(_CONFIG_DIR.rglob(name))
            if len(matches) == 1:
                cfg_path = matches[0]
            elif len(matches) > 1:
                raise ValueError(
                    f"ambiguous config name {name!r}: found in {[str(m) for m in matches]}"
                )
        if cfg_path.exists():
            with open(cfg_path) as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}

        # Extract _includes directive before validation (strip from raw).
        includes = raw.pop("_includes", None)

        # Process _includes FIRST so that inherited values are available
        # during Pydantic validation.  The file's own explicit values
        # override shared constants (deep-merge with raw on top).
        if includes:
            merged_base: dict = {}
            for inc_name in includes:
                inc_cfg = load_config(inc_name)
                merged_base = _deep_merge(merged_base, inc_cfg)
            combined = _deep_merge(merged_base, raw)
        else:
            combined = raw

        # Optional Pydantic validation — import lazily to avoid circular deps
        try:
            from common.core.config_models import _config_validators

            validator = _config_validators.get(name)
            if validator is not None:
                validated = validator(**combined)
                result = validated.model_dump()
                # Re-add keys from the combined dict that Pydantic may
                # have stripped (e.g. top-level shared constant keys not
                # in the model schema).
                for key, val in combined.items():
                    if key not in result:
                        result[key] = val
            else:
                result = combined
        except ImportError:
            # config_models not available — skip validation
            result = combined
        except Exception:
            logger.warning(
                "Config validation failed for %s; returning raw dict", name,
                exc_info=True,
            )
            result = combined

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


def get_algorithm_params(model_id: str) -> dict:
    """Get hyperparameters for a model from the pipeline config.

    Returns the ``params`` sub-dict for the given *model_id* from the
    ``algorithms`` section of ``forecast_pipeline_config.yaml``.

    For backward compatibility, if the algorithm entry has no ``params``
    key, falls back to returning all non-lifecycle keys from the entry
    (i.e., everything except type/enabled/tune/backtest/compete/forecast/
    expert/output_dir/notes/cluster_strategy).
    """
    cfg = load_forecast_pipeline_config()
    algo = cfg.get("algorithms", {}).get(model_id, {})
    params = algo.get("params")
    if params is not None:
        return dict(params)
    # Fallback: return all non-lifecycle keys from the entry itself
    _lifecycle_keys = {
        "type", "enabled", "tune", "backtest", "compete", "forecast",
        "expert", "output_dir", "notes", "cluster_strategy",
    }
    return {k: v for k, v in algo.items() if k not in _lifecycle_keys}


def get_algorithm_config_key(model_id: str) -> str:
    """Get the config key for backward compatibility.

    After consolidation of ``algorithm_config.yaml`` into
    ``forecast_pipeline_config.yaml``, this simply returns *model_id*
    (the pipeline algorithm ID is now the canonical key).
    """
    return model_id


def get_pipeline_config_path() -> Path:
    """Return the absolute path to forecast_pipeline_config.yaml."""
    return _CONFIG_DIR / _FORECAST_PIPELINE_CFG


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
