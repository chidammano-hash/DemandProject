"""GET /{model}/templates — list experiment templates from YAML."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response as FastAPIResponse

from api.core import set_cache
from common.core.utils import get_algorithm_params

from ._helpers import _MODEL_PARAM_KEYS, MODEL_ID_MAP, _validate_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


@router.get("/{model}/templates")
def get_templates(model: str, response: FastAPIResponse):
    """Load experiment templates from config/forecasting/tuning_templates.yaml."""
    _validate_model(model)
    set_cache(response, max_age=300)

    try:
        from common.core.utils import load_config
        tmpl_cfg = load_config("tuning_templates.yaml")
    except (FileNotFoundError, OSError):
        logger.exception("Failed to load tuning_templates.yaml")
        raise HTTPException(status_code=500, detail="Failed to load templates") from None

    model_templates = tmpl_cfg.get("templates", {}).get(model, [])

    # Pipeline-backed templates load the current canonical model parameters.
    enriched: list[dict[str, Any]] = []
    live_params: dict[str, Any] | None = None

    for tmpl in model_templates:
        entry = dict(tmpl)
        if entry.get("source") == "pipeline_config":
            # Lazy-load live params from forecast_pipeline_config.yaml
            if live_params is None:
                live_params = _load_live_params(model)
            entry["params"] = live_params
        enriched.append(entry)

    return {"model": model, "templates": enriched}


def _load_live_params(model: str) -> dict[str, Any]:
    """Load the current production params from forecast_pipeline_config.yaml for a model.

    Uses ``get_algorithm_params()`` from ``common.core.utils`` and filters to
    the known hyperparameter keys for the given model.
    """
    try:
        pipeline_key = MODEL_ID_MAP[model]
        all_params = get_algorithm_params(pipeline_key)
        param_keys = _MODEL_PARAM_KEYS[model]
        return {k: v for k, v in all_params.items() if k in param_keys}
    except (OSError, KeyError):
        logger.warning("Failed to load live params for %s", model)
        return {}
