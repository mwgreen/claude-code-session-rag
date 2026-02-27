#!/bin/bash
# Setup script for session-rag on Apple Silicon Mac.
# Single command to install everything: venv, deps, model, and verify.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "Session-RAG Setup for Apple Silicon Mac"
echo "======================================================================"
echo ""

# --- Check prerequisites ---
echo "Checking prerequisites..."

if [[ $(uname) != "Darwin" ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

if [[ $(uname -m) != "arm64" ]]; then
    echo "Error: This script requires Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Install with: brew install python@3.12"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1,2)
echo "  Python $PYTHON_VERSION"

MACOS_VERSION=$(sw_vers -productVersion | cut -d'.' -f1)
echo "  macOS $MACOS_VERSION"
echo ""

# --- Create virtual environment ---
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  venv already exists, using existing"
else
    python3 -m venv venv
    echo "  venv created"
fi

source venv/bin/activate

echo ""
echo "Installing dependencies..."
echo "  This may take 1-2 minutes on first run..."

pip install --quiet --upgrade pip

echo "  Installing Milvus Lite..."
pip install --quiet "setuptools>=70.0,<82.0" "pymilvus[milvus-lite]>=2.6.0"

echo "  Installing MLX embeddings..."
pip install --quiet "mlx>=0.30.0" mlx-embeddings "transformers<5.0"

echo "  Installing MCP + HTTP server..."
pip install --quiet "mcp>=1.0.0" starlette uvicorn httpx "watchdog>=4.0.0"

echo "  All dependencies installed"

# --- Download model ---
echo ""
echo "Downloading ModernBERT Embed Base..."
chmod +x "$SCRIPT_DIR/download-model.sh"
"$SCRIPT_DIR/download-model.sh"

# --- Test installation ---
echo ""
echo "Testing installation..."

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

try:
    import rag_engine
    import transcript_parser
    import file_watcher
    import tools
    print("  All imports successful")
except Exception as e:
    print(f"  Import failed: {e}")
    sys.exit(1)

try:
    emb = rag_engine.embed_texts(["test embedding"])
    dim = len(emb[0])
    assert dim == 768, f"Expected 768 dims, got {dim}"
    print(f"  Embedding works ({dim} dimensions)")
except Exception as e:
    print(f"  Embedding test failed: {e}")
    sys.exit(1)

print("")
print("  All tests passed!")
PYEOF

if [ $? -ne 0 ]; then
    echo "Installation test failed"
    exit 1
fi

# --- Make scripts executable ---
chmod +x "$SCRIPT_DIR/session-rag-server.sh"
chmod +x "$SCRIPT_DIR/download-model.sh"
chmod +x "$SCRIPT_DIR/index_hook.py"

# --- Install hooks into ~/.claude/settings.json ---
echo ""
echo "Installing hooks into ~/.claude/settings.json..."

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
SETTINGS_FILE="$HOME/.claude/settings.json"

python3 << PYEOF
import json
import os
import sys

settings_file = "$SETTINGS_FILE"
script_dir = "$SCRIPT_DIR"
venv_python = "$VENV_PYTHON"

# Our hooks to install
our_hooks = {
    "SessionStart": [
        {
            "type": "command",
            "command": f"{script_dir}/session-rag-server.sh start",
            "timeout": 30000,
        },
        {
            "type": "command",
            "command": f"{script_dir}/session_start_hook.sh",
            "timeout": 5000,
        },
    ],
    "Stop": [
        {
            "type": "command",
            "command": f"{venv_python} {script_dir}/index_hook.py",
            "timeout": 15000,
        },
    ],
    "PreCompact": [
        {
            "type": "command",
            "command": f"{venv_python} {script_dir}/index_hook.py",
            "timeout": 30000,
        },
    ],
}

# Load existing settings
os.makedirs(os.path.dirname(settings_file), exist_ok=True)
if os.path.exists(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
else:
    settings = {}

hooks = settings.setdefault("hooks", {})

for event, new_hook_entries in our_hooks.items():
    # Get or create the event's hook groups array
    groups = hooks.setdefault(event, [])

    # Find or create a hook group (we use the first group)
    if not groups:
        groups.append({"hooks": []})
    group = groups[0]
    existing = group.setdefault("hooks", [])

    # Collect existing commands to avoid duplicates
    existing_commands = {h.get("command", "") for h in existing}

    added = 0
    for hook in new_hook_entries:
        # Check if already installed (match by command containing our script dir)
        cmd = hook["command"]
        if cmd not in existing_commands:
            # Also check if an older version exists (same script name, different path)
            script_name = os.path.basename(cmd.split()[-1])
            replaced = False
            for i, eh in enumerate(existing):
                if script_name in eh.get("command", ""):
                    existing[i] = hook
                    replaced = True
                    break
            if not replaced:
                existing.append(hook)
            added += 1

    if added:
        print(f"  {event}: {added} hook(s) installed")
    else:
        print(f"  {event}: already configured")

with open(settings_file, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")

print("  Settings saved")
PYEOF

# --- Generate MCP config template ---
echo ""
cat > mcp-config-template.json << MCPEOF
{
  "mcpServers": {
    "session-rag": {
      "type": "http",
      "url": "http://127.0.0.1:7102/mcp/",
      "headers": {
        "X-Project-Root": "/path/to/your/project"
      }
    }
  }
}
MCPEOF
echo "  mcp-config-template.json created"

# --- Done ---
echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Embedding model: ModernBERT Embed Base (768 dims, 8192 token ctx, MLX)"
echo ""
echo "Hooks installed in: ~/.claude/settings.json"
echo "  - SessionStart: auto-start server + register file watcher + backfill"
echo "  - Stop: index final turns when session ends"
echo "  - PreCompact: index turns before context compaction"
echo ""
echo "Next steps:"
echo ""
echo "1. Add MCP server to each project's .mcp.json:"
echo '   "session-rag": {'
echo '     "type": "http",'
echo '     "url": "http://127.0.0.1:7102/mcp/",'
echo '     "headers": { "X-Project-Root": "/path/to/your/project" }'
echo '   }'
echo "   (see mcp-config-template.json)"
echo ""
echo "2. Restart Claude Code to activate hooks"
echo ""
echo "See README.md for full documentation."
echo "======================================================================"
