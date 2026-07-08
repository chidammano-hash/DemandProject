"""Per-turn model-tier routing for the SKU Chatbot.

Each chat turn is an independent Agent SDK invocation, so the model is chosen
per turn. A deterministic, config-driven heuristic maps the question to a tier
(``fast``/``standard``/``deep``). An explicit override from the UI wins. The
heuristic is intentionally simple and pure so it is cheap and unit-testable; an
LLM classifier can replace ``classify_tier`` later without changing callers.
"""
from __future__ import annotations

_DEFAULT_CLAUDE_MODELS = {
    "fast": "claude-haiku-4-5",
    "standard": "claude-sonnet-4-6",
    "deep": "claude-opus-4-8",
}
_DEFAULT_CODEX_MODELS = {
    "fast": "gpt-5.4-mini",
    "standard": "gpt-5.5",
    "deep": "gpt-5.5",
}
_DEFAULT_DEEP_KW = (
    "why", "diagnose", "root cause", "compare", "recommend", "explain",
)
_DEFAULT_FAST_KW = (
    "what is", "what's", "define", "lead time", "which cluster", "list",
)


def classify_tier(question: str, routing: dict) -> str:
    """Return ``"fast" | "standard" | "deep"`` for a question via heuristics."""
    q = (question or "").lower()
    deep_kw = routing.get("deep_keywords") or list(_DEFAULT_DEEP_KW)
    if any(k in q for k in deep_kw):
        return "deep"
    fast_kw = routing.get("fast_keywords") or list(_DEFAULT_FAST_KW)
    fast_max = int(routing.get("fast_max_words", 6))
    if any(k in q for k in fast_kw) or len(q.split()) <= fast_max:
        return "fast"
    return str(routing.get("default_tier", "standard"))


def select_model(
    question: str,
    config: dict,
    *,
    provider: str = "claude",
    override_tier: str | None = None,
    override_model: str | None = None,
) -> tuple[str, str]:
    """Return ``(tier, model_id)`` for this turn.

    Explicit overrides win when ``routing.allow_user_override`` is true.
    """
    if provider == "codex":
        models = {**_DEFAULT_CODEX_MODELS, **(config.get("codex_models") or {})}
    else:
        models = {**_DEFAULT_CLAUDE_MODELS, **(config.get("models") or {})}
    routing = config.get("routing") or {}
    allow_override = bool(routing.get("allow_user_override", True))

    if allow_override and override_model:
        return ("custom", override_model)
    if allow_override and override_tier and override_tier in models:
        return (override_tier, models[override_tier])

    tier = classify_tier(question, routing)
    return (tier, models.get(tier, models["standard"]))
