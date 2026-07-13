#!/bin/bash
# Post-edit hook for TypeScript/TSX files — runs tsc --noEmit on the frontend
FILE=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('file_path',''))" 2>/dev/null)

# Only check .ts/.tsx files in frontend/
case "$FILE" in
  */frontend/src/*.ts|*/frontend/src/*.tsx)
    ROOT_DIR="/Users/manoharchidambaram/projects/DemandProject"
    cd "$ROOT_DIR/frontend" || exit 0
    # Quick type check — only report errors, don't block
    PATH="/opt/homebrew/bin:$PATH" npx tsc --noEmit --pretty 2>&1 | head -20
    # Exit 0 so we don't block the edit — just show warnings
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
