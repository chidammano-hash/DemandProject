#!/bin/bash
CMD=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('command',''))" 2>/dev/null)
ROOT_DIR="/Users/manoharchidambaram/projects/DemandProject"
bash "$ROOT_DIR/scripts/ai_checks/pre_commit_gate.sh" "$CMD"
