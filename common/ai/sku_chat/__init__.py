"""SKU Chatbot — conversational, tool-using per-SKU assistant.

The default runtime is the Claude Agent SDK (``claude-agent-sdk``), and an
optional Codex CLI runtime can be selected with ``runtime.provider: codex`` in
``config/ai/sku_chat_config.yaml``. Both paths are lazy so this package — and
the API that mounts its router — load without either local agent dependency.

Spec: docs/specs/06-ai-platform/07-sku-chatbot.md
"""
