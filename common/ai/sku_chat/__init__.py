"""SKU Chatbot — conversational, tool-using per-SKU assistant.

Built on the Claude Agent SDK (``claude-agent-sdk``). The SDK is an optional
extra (``uv sync --extra agent``) and is imported lazily, so this package — and
the API that mounts its router — load without it installed.

Spec: docs/specs/06-ai-platform/07-sku-chatbot.md
"""
