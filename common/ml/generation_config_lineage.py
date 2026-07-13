"""Canonical current-config contract for immutable forecast generation runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from typing import Any

GENERATION_CONFIG_METADATA_KEY = "generation_config"
_GLOBAL_CONFIG_SECTIONS = (
    "production_forecast",
    "forecast_snapshot",
    "backtest",
    "clustering",
)


class GenerationConfigLineageError(RuntimeError):
    """A generation run cannot prove its exact output-affecting configuration."""


def _json_copy(value: object, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise GenerationConfigLineageError(f"{label} is not JSON-serializable") from exc


def build_generation_config_lineage(
    pipeline_config: Mapping[str, object],
    source_model_ids: Collection[str],
) -> dict[str, Any]:
    """Build a checksum over all selected models and global inference policy."""
    roster = sorted(set(source_model_ids))
    if not roster or any(not model_id.strip() for model_id in roster):
        raise GenerationConfigLineageError("Source-model roster must be non-empty")
    algorithms = pipeline_config.get("algorithms")
    if not isinstance(algorithms, Mapping):
        raise GenerationConfigLineageError("Forecast algorithm configuration is unavailable")
    missing = [model_id for model_id in roster if model_id not in algorithms]
    if missing:
        raise GenerationConfigLineageError(
            f"Source-model configuration is missing models: {missing}"
        )
    config: dict[str, Any] = {
        "source_model_ids": roster,
        "algorithms": {
            model_id: _json_copy(algorithms[model_id], label=f"algorithm {model_id}")
            for model_id in roster
        },
    }
    for section in _GLOBAL_CONFIG_SECTIONS:
        if section not in pipeline_config:
            raise GenerationConfigLineageError(
                f"Forecast configuration is missing section {section!r}"
            )
        config[section] = _json_copy(
            pipeline_config[section],
            label=f"forecast section {section}",
        )
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "config_checksum": hashlib.sha256(payload).hexdigest(),
        "config": config,
    }


def validate_generation_config_lineage(
    lineage: object,
    *,
    pipeline_config: Mapping[str, object],
    source_model_ids: Collection[str],
) -> None:
    """Reject a candidate when any selected/global generation config changed."""
    if not isinstance(lineage, Mapping) or set(lineage) != {"config_checksum", "config"}:
        raise GenerationConfigLineageError(
            "Generation config lineage has an invalid metadata schema"
        )
    expected = build_generation_config_lineage(pipeline_config, source_model_ids)
    if dict(lineage) != expected:
        raise GenerationConfigLineageError(
            "Forecast generation configuration changed after candidate creation"
        )
