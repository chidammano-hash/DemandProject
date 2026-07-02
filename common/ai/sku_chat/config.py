"""Config loader for the SKU Chatbot.

Thin wrapper over ``load_config`` so callers do not hardcode the file path.
"""
from __future__ import annotations

from common.core.utils import load_config

_CONFIG_NAME = "sku_chat_config"


def get_sku_chat_config() -> dict:
    """Return the SKU chatbot config dict.

    Resolves ``config/ai/sku_chat_config.yaml`` via ``load_config`` (cached,
    thread-safe). Returns an empty dict if the file is missing; every consumer
    falls back to in-code defaults, so an empty dict is non-fatal.
    """
    return load_config(_CONFIG_NAME)
