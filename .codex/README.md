# Codex Repo Configuration

This directory holds repository-scoped Codex settings.

- `../AGENTS.md` is the Codex-facing project instruction file.
- `config.toml` keeps `AGENTS.md` first and `CLAUDE.md` as a fallback for older/local tooling.
- Repo skills are exposed via `.agents/skills`, which is symlinked to `.claude/skills`.
- Shared lint/test gate scripts live in `scripts/ai_checks/` so Claude hooks, CI, and Codex guidance reuse the same checks.
- Claude-style automatic hooks do not have a one-to-one Codex repo-file equivalent here; Codex should invoke the shared `scripts/ai_checks/` gates directly and through `make ai-sync-check`, `make audit-routers`, and `make test-all`.

Recommended follow-up:

1. Add custom Codex subagent configs under `.codex/agents/` if you want Codex-native reviewer or planner roles.
2. Run `make ai-sync-check` after changing AI guidance or hook wiring.
