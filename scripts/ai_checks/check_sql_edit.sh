#!/bin/bash
set -euo pipefail

FILE_PATH="${1:-}"

[[ -n "$FILE_PATH" ]] || exit 0
[[ "$FILE_PATH" == *.sql ]] || exit 0

grep -n 'SELECT \*' "$FILE_PATH" 2>/dev/null && echo "WARNING: Use explicit column list, not SELECT *"
grep -n '\$[0-9]' "$FILE_PATH" 2>/dev/null && echo "WARNING: Use %s placeholders (psycopg3), not \$1/\$2"
