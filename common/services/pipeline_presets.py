"""Named JobManager pipeline presets.

Loads ``config/forecasting/pipelines.yaml`` — the forecasting lifecycle
expressed as sequential JobManager pipelines (``submit_pipeline``) instead of
manually stepped one-off jobs. Endpoints: ``GET /jobs/pipelines/named`` and
``POST /jobs/pipelines/named/{name}`` in ``api/routers/core/jobs.py``.
"""
from __future__ import annotations

import logging
from typing import Any

from common.core.utils import load_config

logger = logging.getLogger(__name__)


def load_pipeline_presets() -> dict[str, dict[str, Any]]:
    """All named pipeline presets, keyed by name. Empty dict if unconfigured."""
    cfg = load_config("pipelines.yaml") or {}
    presets = cfg.get("pipelines") or {}
    return {str(name): p for name, p in presets.items() if isinstance(p, dict)}


def get_pipeline_preset(name: str) -> dict[str, Any]:
    """One preset by name. Raises KeyError with the known names on a miss."""
    presets = load_pipeline_presets()
    if name not in presets:
        raise KeyError(
            f"Unknown pipeline preset {name!r}. Known: {sorted(presets)}"
        )
    return presets[name]


def preset_steps(preset: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a preset's steps into JobManager.submit_pipeline step dicts."""
    steps: list[dict[str, Any]] = []
    for step in preset.get("steps") or []:
        if not isinstance(step, dict) or not step.get("job_type"):
            raise ValueError(f"Malformed pipeline step: {step!r}")
        steps.append({
            "job_type": str(step["job_type"]),
            "params": step.get("params") or {},
            "label": step.get("label"),
        })
    return steps
