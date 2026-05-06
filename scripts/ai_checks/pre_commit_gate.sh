#!/bin/bash
set -euo pipefail

COMMAND_TEXT="${1:-}"

echo "$COMMAND_TEXT" | grep -q 'git commit' || exit 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"

echo "=== PRE-COMMIT QUALITY GATE ==="
echo "--- Ruff ---"
if ! ~/.local/bin/uv run ruff check api/ common/ scripts/ --select E,F 2>&1 | tail -5; then
  echo "BLOCKED: Fix Ruff issues before committing"
  exit 1
fi
echo "--- Unenforced CLAUDE.md rules ---"
if ! bash "$ROOT_DIR/scripts/ai_checks/check_unenforced_rules.sh"; then
  echo "BLOCKED: Fix CLAUDE.md unenforced-rule violations before committing"
  exit 1
fi
echo "--- Backend tests ---"
if ! ~/.local/bin/uv run pytest tests/ -q --tb=line 2>&1 | tail -10; then
  echo "BLOCKED: Fix test failures before committing"
  exit 1
fi
echo "=== GATE PASSED ==="
