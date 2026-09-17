#!/bin/bash
# SessionStart hook: exports CLAUDE_SESSION_ID for the MCP tools and registers
# the project with the session-rag server (which backfills its transcripts).

INPUT=$(cat)
SERVER_URL="${SESSION_RAG_URL:-http://127.0.0.1:7102}"

read -r SESSION_ID CWD < <(printf '%s' "$INPUT" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
print(d.get("session_id", ""), d.get("cwd", ""))
' 2>/dev/null) || true

if [ -n "$SESSION_ID" ] && [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    echo "export CLAUDE_SESSION_ID='$SESSION_ID'" >> "$CLAUDE_ENV_FILE"
fi

PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_ROOT" ] && [ -n "$CWD" ]; then
    PROJECT_ROOT=$(git -C "$CWD" rev-parse --show-toplevel 2>/dev/null || echo "$CWD")
fi

if [ -n "$PROJECT_ROOT" ]; then
    BODY=$(python3 -c 'import json, sys; print(json.dumps({"project_root": sys.argv[1]}))' "$PROJECT_ROOT")
    curl -sf --max-time 5 -X POST "$SERVER_URL/watch" \
        -H "Content-Type: application/json" \
        -H "X-Project-Root: $PROJECT_ROOT" \
        -d "$BODY" >/dev/null 2>&1 || true
fi
exit 0
