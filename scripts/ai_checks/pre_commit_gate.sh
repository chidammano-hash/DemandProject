#!/bin/bash
# Pre-commit quality gate, invoked by .claude/hooks/pre-commit-gate.sh (PreToolUse: Bash).
# The hook passes this script's exit code straight through to Claude Code.
#
# Claude Code hook exit-code semantics:
#   2 = HARD BLOCK — the tool call (git commit) is refused; stderr is shown to Claude.
#   0 = allow.
#
# Design: ONLY the allowlist-based mechanical-rule gate hard-blocks. Its allowlists pin the
# EXISTING violating files, so a clean diff always passes — safe to make absolute. Ruff E,F
# and the full test suite are ADVISORY here: they carry a known pre-existing backlog
# (~1480 ruff E/F, plus known test fails), so hard-blocking them would refuse every commit.
# They are surfaced for awareness but never block. See MEMORY.md "Pre-commit gate".

set -uo pipefail

COMMAND_TEXT="${1:-}"

# Only act on git commit invocations.
echo "$COMMAND_TEXT" | grep -q 'git commit' || exit 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "=== PRE-COMMIT QUALITY GATE ===" >&2

# ---- HARD GATE (exit 2): new violations of the 6 mechanical CLAUDE.md rules ----------
echo "--- Unenforced CLAUDE.md rules (HARD GATE) ---" >&2
if ! bash "$ROOT_DIR/scripts/ai_checks/check_unenforced_rules.sh" >&2; then
  echo "BLOCKED: this commit ADDS a new violation of a mechanically-enforced CLAUDE.md rule." >&2
  echo "         Fix the flagged line before committing — the allowlists pin EXISTING files" >&2
  echo "         only and must not grow. (date.today / parents[N] / bare except / print /" >&2
  echo "         f-string SQL / frontend 'any'.)" >&2
  exit 2
fi

# ---- ADVISORY (never blocks): pre-existing ruff + test backlog, surfaced only --------
echo "--- Ruff E,F (advisory — pre-existing backlog, does not block) ---" >&2
~/.local/bin/uv run ruff check api/ common/ scripts/ --select E,F 2>&1 | tail -5 >&2 || true
echo "--- Backend tests (advisory — does not block) ---" >&2
~/.local/bin/uv run pytest tests/ -q --tb=line 2>&1 | tail -10 >&2 || true

echo "=== GATE PASSED (mechanical rules clean) ===" >&2
exit 0
