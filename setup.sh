#!/bin/bash
# One-command install for session-rag on Apple Silicon:
# venv + dependencies, embedding model download, self-test, Claude Code hooks
# and the global MCP server entry.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "Session-RAG Setup for Apple Silicon Mac"
echo "======================================================================"
echo ""

# --- Prerequisites ---
echo "Checking prerequisites..."
if [[ $(uname) != "Darwin" ]]; then echo "Error: macOS only"; exit 1; fi
if [[ $(uname -m) != "arm64" ]]; then echo "Error: Apple Silicon required"; exit 1; fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found. Install with: brew install python@3.12"; exit 1
fi
PY_VERSION=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "  Python $PY_VERSION  |  macOS $(sw_vers -productVersion)"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "Error: Python 3.11+ required"; exit 1
fi
echo ""

# --- Virtual environment ---
echo "Creating virtual environment..."
if [ -d venv ]; then echo "  venv already exists, using it"; else python3 -m venv venv; echo "  venv created"; fi
# shellcheck disable=SC1091
source venv/bin/activate

echo ""
echo "Installing dependencies (1-2 minutes on first run)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  Dependencies installed"

# --- Model ---
echo ""
SESSION_RAG_MODEL="${SESSION_RAG_MODEL:-embeddinggemma}"
export SESSION_RAG_MODEL
chmod +x "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR/index_hook.py"
if venv/bin/python -c "import embedder,sys; sys.exit(0 if embedder.get_spec().is_downloaded() else 1)" 2>/dev/null; then
    echo "Embedding model '$SESSION_RAG_MODEL' already downloaded."
else
    echo "Downloading embedding model '$SESSION_RAG_MODEL'..."
    "$SCRIPT_DIR/download-model.sh" "$SESSION_RAG_MODEL"
fi

# --- Self-test ---
echo ""
echo "Testing installation..."
venv/bin/python - << 'PYEOF'
import sys
sys.path.insert(0, '.')
import embedder, rag_engine, transcript_parser, indexer, file_watcher, tools, http_server  # noqa: F401
print("  All imports successful")
spec = rag_engine.model_spec()
vec = rag_engine.embed_texts(["test embedding"])[0]
assert len(vec) == spec.dim, f"expected {spec.dim} dims, got {len(vec)}"
print(f"  Embedding works ({len(vec)} dimensions, model: {spec.name})")
PYEOF
venv/bin/python -m unittest discover -s tests -q 2>&1 | grep -v -iE 'pkg_resources|UserWarning|^\s*from pkg' | tail -1

# --- Hooks in ~/.claude/settings.json ---
echo ""
echo "Installing hooks into ~/.claude/settings.json..."
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
SETTINGS_FILE="$HOME/.claude/settings.json"
SCRIPT_DIR="$SCRIPT_DIR" VENV_PYTHON="$VENV_PYTHON" SETTINGS_FILE="$SETTINGS_FILE" python3 - << 'PYEOF'
import json, os
settings_file = os.environ["SETTINGS_FILE"]
script_dir = os.environ["SCRIPT_DIR"]
venv_python = os.environ["VENV_PYTHON"]

our_hooks = {
    "SessionStart": [
        {"type": "command", "command": f"{script_dir}/session-rag-server.sh start", "timeout": 90000},
        {"type": "command", "command": f"{script_dir}/session_start_hook.sh", "timeout": 10000},
    ],
    "Stop": [{"type": "command", "command": f"{venv_python} {script_dir}/index_hook.py", "timeout": 15000}],
    "PreCompact": [{"type": "command", "command": f"{venv_python} {script_dir}/index_hook.py", "timeout": 30000}],
}

os.makedirs(os.path.dirname(settings_file), exist_ok=True)
settings = {}
if os.path.exists(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
hooks = settings.setdefault("hooks", {})
for event, entries in our_hooks.items():
    groups = hooks.setdefault(event, [])
    if not groups:
        groups.append({"hooks": []})
    existing = groups[0].setdefault("hooks", [])
    changed = 0
    for hook in entries:
        script_name = os.path.basename(hook["command"].split()[-1] if "index_hook" in hook["command"]
                                       else hook["command"].split()[0])
        for i, eh in enumerate(existing):
            if script_name in eh.get("command", ""):
                if eh != hook:
                    existing[i] = hook
                    changed += 1
                break
        else:
            existing.append(hook)
            changed += 1
    print(f"  {event}: {'updated' if changed else 'already configured'}")

tmp = settings_file + ".tmp"
with open(tmp, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")
os.replace(tmp, settings_file)
print("  Settings saved")
PYEOF

# --- Global MCP server (user scope) ---
echo ""
echo "Installing global MCP server..."
MCP_HELPERS_DIR="$HOME/.claude/mcp-helpers"
mkdir -p "$MCP_HELPERS_DIR"
cat > "$MCP_HELPERS_DIR/session-rag-headers.sh" << 'HELPEREOF'
#!/bin/bash
# headersHelper for the session-rag MCP server: tells the server which project
# this Claude Code session belongs to (git repo root, or the working directory).
ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$ROOT" ]; then
    ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
fi
python3 -c 'import json, sys; print(json.dumps({"X-Project-Root": sys.argv[1]}))' "$ROOT" 2>/dev/null \
    || printf '{"X-Project-Root": "%s"}\n' "$ROOT"
HELPEREOF
chmod +x "$MCP_HELPERS_DIR/session-rag-headers.sh"
echo "  Header helper installed: $MCP_HELPERS_DIR/session-rag-headers.sh"

CLAUDE_JSON="$HOME/.claude.json"
CLAUDE_JSON="$CLAUDE_JSON" MCP_HELPERS_DIR="$MCP_HELPERS_DIR" python3 - << 'PYEOF'
import json, os
claude_json = os.environ["CLAUDE_JSON"]
helpers_dir = os.environ["MCP_HELPERS_DIR"]
data = {}
if os.path.exists(claude_json):
    with open(claude_json) as f:
        data = json.load(f)
data.setdefault("mcpServers", {})["session-rag"] = {
    "type": "http",
    "url": "http://127.0.0.1:7102/mcp/",
    "headersHelper": f"{helpers_dir}/session-rag-headers.sh",
}
tmp = claude_json + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
os.replace(tmp, claude_json)
print("  MCP server added to ~/.claude.json (user scope, all projects)")
PYEOF

# --- launchd agent (optional, recommended) ---
echo ""
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/${SESSION_RAG_LAUNCHD_LABEL:-com.mattgreen.session-rag}.plist"
if [ -f "$LAUNCHD_PLIST" ] || [ "${SESSION_RAG_LAUNCHD:-}" = "1" ]; then
    echo "Installing launchd agent (starts at login, restarts on crash)..."
    "$SCRIPT_DIR/session-rag-server.sh" install-launchd
else
    echo "Tip: ./session-rag-server.sh install-launchd makes launchd start the server at login"
    echo "     and keep it alive (recommended). Set SESSION_RAG_LAUNCHD=1 to do it from setup."
fi

# --- Done ---
echo ""
echo "======================================================================"
echo "Installation complete"
echo "======================================================================"
echo ""
echo "Embedding model: $SESSION_RAG_MODEL   (./download-model.sh --list shows all options)"
echo "Hooks: SessionStart (start server + register project), Stop and PreCompact (index)"
echo "MCP server: ~/.claude.json (user scope, available in every project)"
echo ""
echo "Next step: restart Claude Code, or start the server now with ./session-rag-server.sh start"
echo "======================================================================"
