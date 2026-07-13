#!/bin/bash
FILE=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('file_path',''))" 2>/dev/null)
ROOT_DIR="/Users/manoharchidambaram/projects/DemandProject"
bash "$ROOT_DIR/scripts/ai_checks/check_sql_edit.sh" "$FILE"
