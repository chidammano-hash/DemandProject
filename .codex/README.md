# Codex Repo Configuration

This directory holds repository-scoped Codex settings.

- `config.toml` tells Codex to treat `CLAUDE.md` as the project instruction file.
- Repo skills are exposed via `.agents/skills`, which is symlinked to `.claude/skills`.
- Shared lint/test gate scripts live in `scripts/ai_checks/` so Claude hooks, CI, and Codex guidance can reuse the same checks.

Recommended follow-up:

1. Add custom Codex subagent configs under `.codex/agents/` if you want Codex-native reviewer or planner roles.
2. Run `make ai-sync-check` after changing AI guidance or hook wiring.
