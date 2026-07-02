"""Unit tests for model-tier routing (common/ai/sku_chat/model_router.py)."""
from __future__ import annotations

from common.ai.sku_chat import model_router

_CFG = {
    "models": {
        "fast": "claude-haiku-4-5",
        "standard": "claude-sonnet-4-6",
        "deep": "claude-opus-4-8",
    },
    "routing": {
        "default_tier": "standard",
        "allow_user_override": True,
        "fast_max_words": 6,
        "deep_keywords": ["why", "compare", "recommend"],
        "fast_keywords": ["what is", "lead time"],
    },
}


def test_deep_keyword_routes_to_opus():
    tier, model = model_router.select_model("Why did the forecast miss in Q3?", _CFG)
    assert tier == "deep"
    assert model == "claude-opus-4-8"


def test_fast_keyword_routes_to_haiku():
    tier, model = model_router.select_model("What is the lead time?", _CFG)
    assert tier == "fast"
    assert model == "claude-haiku-4-5"


def test_short_question_routes_to_fast():
    tier, _ = model_router.select_model("current cluster?", _CFG)
    assert tier == "fast"


def test_default_tier_is_standard():
    tier, model = model_router.select_model(
        "Give me a thorough summary of demand and forecast trends over time", _CFG
    )
    assert tier == "standard"
    assert model == "claude-sonnet-4-6"


def test_explicit_tier_override_wins():
    tier, model = model_router.select_model(
        "What is the lead time?", _CFG, override_tier="deep"
    )
    assert tier == "deep"
    assert model == "claude-opus-4-8"


def test_explicit_model_override_wins():
    tier, model = model_router.select_model(
        "Why did it miss?", _CFG, override_model="claude-haiku-4-5"
    )
    assert tier == "custom"
    assert model == "claude-haiku-4-5"


def test_override_ignored_when_disabled():
    cfg = {**_CFG, "routing": {**_CFG["routing"], "allow_user_override": False}}
    tier, _ = model_router.select_model(
        "What is the lead time?", cfg, override_tier="deep"
    )
    assert tier == "fast"  # heuristic wins; override suppressed


def test_empty_config_falls_back_to_sonnet():
    tier, model = model_router.select_model("a longer question that has no keywords here", {})
    assert tier == "standard"
    assert model == "claude-sonnet-4-6"
