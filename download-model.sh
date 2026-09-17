#!/bin/bash
# Download an embedding model into the Hugging Face cache and smoke-test it.
#
# Usage: ./download-model.sh [model_name]      (default: $SESSION_RAG_MODEL or embeddinggemma)
#        ./download-model.sh --list
#
# Behind a corporate TLS proxy, point SESSION_RAG_CA_BUNDLE (or NODE_EXTRA_CA_CERTS)
# at the proxy's CA certificate; it is merged with the public roots automatically.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "venv not found. Run ./setup.sh first." >&2
    exit 1
fi

cd "$SCRIPT_DIR"
if [ "${1:-}" = "--list" ]; then
    exec "$PYTHON" embedder.py list
fi

MODEL_NAME="${1:-${SESSION_RAG_MODEL:-embeddinggemma}}"
echo "Downloading embedding model '$MODEL_NAME' (cached in ~/.cache/huggingface/hub/) ..."
SESSION_RAG_ALLOW_DOWNLOAD=1 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    "$PYTHON" embedder.py download "$MODEL_NAME"
echo "  Download complete."
