#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

pass() {
  echo "OK: $1"
}

[[ -f ".codex/config.toml" ]] || fail "missing .codex/config.toml"
grep -Fq 'project_doc_fallback_filenames = ["CLAUDE.md"]' .codex/config.toml || fail "Codex fallback file is not set to CLAUDE.md"
pass "Codex project doc fallback points at CLAUDE.md"

[[ -L ".agents/skills" ]] || fail ".agents/skills is not a symlink"
[[ "$(readlink .agents/skills)" == "../.claude/skills" ]] || fail ".agents/skills does not point to ../.claude/skills"
pass "Repo skills are shared through .agents/skills"

for script_path in \
  scripts/ai_checks/check_python_edit.sh \
  scripts/ai_checks/check_sql_edit.sh \
  scripts/ai_checks/check_test_edit.sh \
  scripts/ai_checks/pre_commit_gate.sh
do
  [[ -x "$script_path" ]] || fail "$script_path is not executable"
done
pass "Shared AI check scripts are executable"

grep -Fq 'scripts/ai_checks/check_python_edit.sh' .claude/hooks/post-edit-python.sh || fail "Python hook is not wired to shared script"
grep -Fq 'scripts/ai_checks/check_sql_edit.sh' .claude/hooks/post-edit-sql.sh || fail "SQL hook is not wired to shared script"
grep -Fq 'scripts/ai_checks/check_test_edit.sh' .claude/hooks/post-edit-test.sh || fail "Test hook is not wired to shared script"
grep -Fq 'scripts/ai_checks/pre_commit_gate.sh' .claude/hooks/pre-commit-gate.sh || fail "Pre-commit hook is not wired to shared script"
pass "Claude hooks delegate to shared scripts"

echo "AI sync check passed."
