"""Immutable configuration lineage for direct statistical/foundation adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from typing import Any

DIRECT_MODEL_CONFIG_METADATA_KEY = "direct_model_configs"
SOURCE_MODEL_ROSTER_METADATA_KEY = "source_model_ids"
DIRECT_PRODUCTION_MODELS = frozenset({"mstl", "chronos2_enriched"})


class DirectModelLineageError(RuntimeError):
    """A direct-model generation cannot prove its exact configuration."""


def _canonical_config(
    algorithms: Mapping[str, object],
    model_id: str,
) -> dict[str, Any]:
    raw = algorithms.get(model_id)
    if not isinstance(raw, Mapping):
        raise DirectModelLineageError(
            f"Direct model {model_id!r} is missing its algorithm configuration"
        )
    model_type = raw.get("type")
    params = raw.get("params")
    if not isinstance(model_type, str) or not model_type.strip():
        raise DirectModelLineageError(f"Direct model {model_id!r} has no configured type")
    if not isinstance(params, Mapping):
        raise DirectModelLineageError(
            f"Direct model {model_id!r} has no configured parameters"
        )
    try:
        canonical_params = json.loads(
            json.dumps(dict(params), sort_keys=True, separators=(",", ":"))
        )
    except (TypeError, ValueError) as exc:
        raise DirectModelLineageError(
            f"Direct model {model_id!r} parameters are not JSON-serializable"
        ) from exc
    return {
        "model_id": model_id,
        "type": model_type.strip(),
        "params": canonical_params,
    }


def build_direct_model_config_lineage(
    algorithms: Mapping[str, object],
    model_ids: Collection[str],
) -> dict[str, dict[str, Any]]:
    """Build exact config payloads/checksums for direct models used by a run."""
    required = sorted(set(model_ids) & DIRECT_PRODUCTION_MODELS)
    lineage: dict[str, dict[str, Any]] = {}
    for model_id in required:
        config = _canonical_config(algorithms, model_id)
        payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        lineage[model_id] = {
            "config_checksum": hashlib.sha256(payload).hexdigest(),
            "config": config,
        }
    return lineage


def validate_direct_model_config_lineage(
    lineage: object,
    *,
    algorithms: Mapping[str, object],
    required_model_ids: Collection[str],
) -> None:
    """Require exact generated config lineage for every direct source model."""
    required = set(required_model_ids) & DIRECT_PRODUCTION_MODELS
    if not isinstance(lineage, Mapping) or any(
        not isinstance(key, str) for key in lineage
    ):
        raise DirectModelLineageError("Direct-model config lineage must be a mapping")
    actual = set(lineage)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise DirectModelLineageError(
            "Direct-model config lineage does not match the candidate sources ("
            + ", ".join(details)
            + ")"
        )
    expected = build_direct_model_config_lineage(algorithms, required)
    for model_id in sorted(required):
        if lineage[model_id] != expected[model_id]:
            raise DirectModelLineageError(
                f"Direct model {model_id!r} configuration changed after generation"
            )
