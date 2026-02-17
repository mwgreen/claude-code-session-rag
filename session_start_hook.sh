#!/bin/bash
# SessionStart hook: writes CLAUDE_SESSION_ID to CLAUDE_ENV_FILE
# so it's available in Bash and potentially in MCP header expansion.

INPUT=$(cat)

SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)

if [ -n "$SESSION_ID" ] && [ -n "$CLAUDE_ENV_FILE" ]; then
    echo "export CLAUDE_SESSION_ID='$SESSION_ID'" >> "$CLAUDE_ENV_FILE"
fi

exit 0
