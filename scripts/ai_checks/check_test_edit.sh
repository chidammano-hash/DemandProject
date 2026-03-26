#!/bin/bash
set -euo pipefail

FILE_PATH="${1:-}"

[[ -n "$FILE_PATH" ]] || exit 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"

if [[ "$FILE_PATH" == */tests/*.py ]]; then
  ~/.local/bin/uv run pytest "$FILE_PATH" -q --tb=short 2>&1 | tail -15
elif [[ "$FILE_PATH" == */frontend/src/*test* ]]; then
  cd frontend
  PATH="/opt/homebrew/bin:$PATH" /opt/homebrew/bin/node node_modules/.bin/vitest run "$FILE_PATH" --reporter=dot 2>&1 | tail -15
fi
