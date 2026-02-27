#!/bin/bash
# SessionStart hook: sets CLAUDE_SESSION_ID env var and registers
# the project with the session-rag file watcher for real-time indexing.

INPUT=$(cat)

SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
CWD=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cwd',''))" 2>/dev/null)

# Export session ID for MCP tools
if [ -n "$SESSION_ID" ] && [ -n "$CLAUDE_ENV_FILE" ]; then
    echo "export CLAUDE_SESSION_ID='$SESSION_ID'" >> "$CLAUDE_ENV_FILE"
fi

# Derive project root from git
PROJECT_ROOT=""
if [ -n "$CWD" ]; then
    PROJECT_ROOT=$(git -C "$CWD" rev-parse --show-toplevel 2>/dev/null || echo "$CWD")
fi

# Register project with file watcher (fire-and-forget)
# This starts watching transcripts and backfills any missed sessions.
if [ -n "$PROJECT_ROOT" ]; then
    SERVER_URL="${SESSION_RAG_URL:-http://127.0.0.1:7102}"
    curl -sf --max-time 5 \
        -X POST "$SERVER_URL/watch" \
        -H "Content-Type: application/json" \
        -H "X-Project-Root: $PROJECT_ROOT" \
        -d "{\"project_root\": \"$PROJECT_ROOT\"}" \
        >/dev/null 2>&1 || true
fi

exit 0
