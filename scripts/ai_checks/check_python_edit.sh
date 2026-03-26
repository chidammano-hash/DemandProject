#!/bin/bash
set -euo pipefail

FILE_PATH="${1:-}"

[[ -n "$FILE_PATH" ]] || exit 0
[[ "$FILE_PATH" == *.py ]] || exit 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"
~/.local/bin/uv run ruff check --select E,W,F,I "$FILE_PATH" 2>&1 | head -20
