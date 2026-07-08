#!/bin/bash
# Mechanical gates for the 5 CLAUDE.md rules that were previously unenforced.
#
# Each rule has an allowlist at scripts/ai_checks/allowlists/ruleN_*.txt that
# pins the EXISTING violations as of the introduction of these gates. Any NEW
# file that violates a rule will fail this script.
#
# Allowlists are TEMPORARY and meant to shrink, not grow. When a file in an
# allowlist is cleaned up, remove it from the allowlist. When the allowlist is
# empty, the rule has zero existing violations and the gate becomes absolute.
#
# Exit code: 0 = all gates passed (or only allowlisted violations).
#            1 = at least one new (non-allowlisted) violation found.
#
# Run manually with:
#   bash scripts/ai_checks/check_unenforced_rules.sh
#
# Wired into:
#   - .pre-commit-config.yaml as a local hook
#   - .claude/hooks/pre-commit-gate.sh -> scripts/ai_checks/pre_commit_gate.sh

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ALLOW_DIR="scripts/ai_checks/allowlists"
FAIL=0

# ---------------------------------------------------------------------------
# Helper: print a violation in `file:line: msg` form and bump FAIL counter.
# ---------------------------------------------------------------------------
violation() {
  echo "  $1"
  FAIL=$((FAIL + 1))
}

# Normalize a list of files (one per line) by stripping leading `./`.
normalize() { sed 's|^\./||'; }

# Build a fresh allowlist set from a file (one path per line, # for comments).
read_allowlist() {
  local f="$1"
  if [[ -f "$f" ]]; then
    grep -v '^\s*#' "$f" | grep -v '^\s*$' | sort -u
  fi
}

# ---------------------------------------------------------------------------
# Rule 1: date.today() outside common/core/planning_date.py and tests/
# ---------------------------------------------------------------------------
echo "[Rule 1] date.today() must only appear in common/core/planning_date.py"
RULE1_ALLOW=$(read_allowlist "$ALLOW_DIR/rule1_date_today.txt")
RULE1_HITS=$(grep -rln "date\.today()" --include="*.py" . 2>/dev/null \
  | normalize \
  | grep -v "^\.venv/" | grep -v "^archive/" | grep -v "^data/" | grep -v "^node_modules/" | grep -v "^tmp/" \
  | grep -v "^common/core/planning_date\.py$" \
  | grep -v "^tests/" \
  | sort -u || true)

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if ! echo "$RULE1_ALLOW" | grep -Fxq "$f"; then
    LINES=$(grep -nE "date\.today\(\)" "$f" 2>/dev/null | head -3)
    while IFS= read -r ln; do
      violation "$f:$ln  use get_planning_date() from common.core.planning_date"
    done <<< "$LINES"
  fi
done <<< "$RULE1_HITS"

# ---------------------------------------------------------------------------
# Rule 2: Path(__file__).resolve().parents[N] forbidden — use common.core.paths
# Heuristic: any file containing `Path(__file__).resolve().parents[` must
# also import from common.core.paths. Files that only have it inside an
# `if __name__ == "__main__":` block and import paths are fine.
# Simpler heuristic actually used: file must import from common.core.paths.
# ---------------------------------------------------------------------------
echo "[Rule 2] Path(__file__).resolve().parents[ requires common.core.paths import"
RULE2_ALLOW=$(read_allowlist "$ALLOW_DIR/rule2_parents_path.txt")
RULE2_HITS=$(grep -rln "Path(__file__)\.resolve()\.parents\[" --include="*.py" . 2>/dev/null \
  | normalize \
  | grep -v "^\.venv/" | grep -v "^archive/" | grep -v "^data/" | grep -v "^node_modules/" | grep -v "^tmp/" \
  | sort -u || true)

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if ! echo "$RULE2_ALLOW" | grep -Fxq "$f"; then
    # Allow if file imports from common.core.paths (CLI bootstrap exception)
    if ! grep -q "from common\.core\.paths" "$f" 2>/dev/null; then
      LN=$(grep -nE "Path\(__file__\)\.resolve\(\)\.parents\[" "$f" 2>/dev/null | head -1)
      violation "$f:$LN  use 'from common.core.paths import PROJECT_ROOT' instead of parents[N]"
    fi
  fi
done <<< "$RULE2_HITS"

# ---------------------------------------------------------------------------
# Rule 3: bare `except Exception` without `# noqa: BLE001 — <reason>`
# Note: ruff BLE001 is also enabled in pyproject.toml as backup.
# ---------------------------------------------------------------------------
echo "[Rule 3] except Exception requires '# noqa: BLE001 — <reason>'"
RULE3_ALLOW=$(read_allowlist "$ALLOW_DIR/rule3_bare_except.txt")
# Find files with offending lines
RULE3_HITS=$(grep -rlE "except Exception\b" --include="*.py" . 2>/dev/null \
  | normalize \
  | grep -v "^\.venv/" | grep -v "^archive/" | grep -v "^data/" | grep -v "^node_modules/" | grep -v "^tmp/" \
  | sort -u || true)

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  # Has at least one bare instance (no BLE001 escape)?
  if grep -E "except Exception\b" "$f" 2>/dev/null | grep -vq "noqa: BLE001"; then
    if ! echo "$RULE3_ALLOW" | grep -Fxq "$f"; then
      LN=$(grep -nE "except Exception\b" "$f" 2>/dev/null | grep -v "noqa: BLE001" | head -1)
      violation "$f:$LN  catch a specific exception, or add '# noqa: BLE001 — <reason>'"
    fi
  fi
done <<< "$RULE3_HITS"

# ---------------------------------------------------------------------------
# Rule 4: print() in production scripts/etl|ml|inventory|forecasting|ops|ai
# Note: ruff T201 is also enabled in pyproject.toml scoped to those subdirs.
# ---------------------------------------------------------------------------
echo "[Rule 4] print() forbidden in production script subdirs — use logger"
RULE4_ALLOW=$(read_allowlist "$ALLOW_DIR/rule4_print_scripts.txt")
RULE4_HITS=$(for d in scripts/etl scripts/ml scripts/inventory scripts/forecasting scripts/ops scripts/ai; do
  [[ -d "$d" ]] && grep -rln "print(" --include="*.py" "$d" 2>/dev/null
done | sort -u || true)

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if ! echo "$RULE4_ALLOW" | grep -Fxq "$f"; then
    LN=$(grep -nE "\bprint\(" "$f" 2>/dev/null | head -1)
    violation "$f:$LN  use logging.getLogger(__name__) — print() forbidden in production scripts"
  fi
done <<< "$RULE4_HITS"

# ---------------------------------------------------------------------------
# Rule 5: ': any', '<any>', 'as any' in frontend/src/api/queries/
# ---------------------------------------------------------------------------
echo "[Rule 5] no 'any' types in frontend/src/api/queries/"
RULE5_ALLOW=$(read_allowlist "$ALLOW_DIR/rule5_frontend_any.txt")
QDIR="frontend/src/api/queries"
RULE5_HITS=""
if [[ -d "$QDIR" ]]; then
  RULE5_HITS=$(grep -rlE ":\s*any\b|<any>|\bas\s+any\b" "$QDIR" 2>/dev/null | sort -u || true)
fi

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if ! echo "$RULE5_ALLOW" | grep -Fxq "$f"; then
    LN=$(grep -nE ":\s*any\b|<any>|\bas\s+any\b" "$f" 2>/dev/null | head -1)
    violation "$f:$LN  generate concrete types via 'npm run gen:types' — no 'any' allowed in api/queries"
  fi
done <<< "$RULE5_HITS"

# ---------------------------------------------------------------------------
# Rule 6: f-string SQL execute calls — psycopg3 requires %s parameterised queries.
# Detected by scripts/tools/check_fstring_sql.py. Allowlist pins EXISTING files;
# any NEW file with f-string interpolation inside cur.execute() fails the gate.
# ---------------------------------------------------------------------------
echo "[Rule 6] f-string SQL inside cur.execute() — use %s parameterised queries"
RULE6_ALLOW=$(read_allowlist "$ALLOW_DIR/rule6_fstring_sql.txt")
# scripts/tools/check_fstring_sql.py prints lines like 'path/to/file.py:LINE: ...'
# and exits non-zero if any violations exist. We harvest the file paths and
# diff against the allowlist; existing offenders are silenced, new ones fail.
RULE6_OUTPUT=$(python3 scripts/tools/check_fstring_sql.py 2>/dev/null || true)
RULE6_RAW=$(echo "$RULE6_OUTPUT" \
  | grep -E "^[a-zA-Z_/].*\.py:[0-9]+:" \
  | awk -F: '{print $1}' \
  | sort -u)

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if ! echo "$RULE6_ALLOW" | grep -Fxq "$f"; then
    LN=$(echo "$RULE6_OUTPUT" | grep -E "^${f}:[0-9]+:" | head -1)
    violation "$LN  use %s parameterised queries — psycopg3 requires it (see CLAUDE.md Critical Rules)"
  fi
done <<< "$RULE6_RAW"

# ---------------------------------------------------------------------------
# Rule 7: REFRESH MATERIALIZED VIEW in Python outside common/core/mv_refresh.py.
# All MV refreshes go through the central dependency map (refresh_for_tables /
# refresh_materialized_views) — hand-picked inline lists are how MVs silently
# went stale. tests/ excepted (mock SQL assertions).
# ---------------------------------------------------------------------------
echo "[Rule 7] REFRESH MATERIALIZED VIEW only inside common/core/mv_refresh.py"
RULE7_ALLOW=$(read_allowlist "$ALLOW_DIR/rule7_mv_refresh.txt")
RULE7_HITS=$(grep -rln "REFRESH MATERIALIZED VIEW" --include="*.py" . 2>/dev/null \
  | normalize \
  | grep -v "^\.venv/" | grep -v "^archive/" | grep -v "^data/" | grep -v "^node_modules/" | grep -v "^tmp/" \
  | grep -v "^common/core/mv_refresh\.py$" \
  | grep -v "^tests/" \
  | sort -u || true)

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if ! echo "$RULE7_ALLOW" | grep -Fxq "$f"; then
    LN=$(grep -n "REFRESH MATERIALIZED VIEW" "$f" 2>/dev/null | head -1)
    violation "$f:$LN  route through common.core.mv_refresh (refresh_for_tables / refresh_materialized_views)"
  fi
done <<< "$RULE7_HITS"

# ---------------------------------------------------------------------------
echo
if [[ "$FAIL" -gt 0 ]]; then
  echo "BLOCKED: $FAIL new violation(s) of unenforced CLAUDE.md rules."
  echo "         If a violation is intentional, fix it. Allowlists in"
  echo "         $ALLOW_DIR/ pin EXISTING violations only and are not for new code."
  exit 1
fi
echo "OK: no new violations of the 7 CLAUDE.md unenforced rules."
exit 0
