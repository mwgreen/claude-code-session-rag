#!/bin/bash
# Download ModernBERT Embed Base model for MLX.
#
# The model is auto-downloaded from HuggingFace on first use via
# mlx-embeddings, so this script just triggers a test embed
# to pre-cache the model files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/venv/bin/python"

echo "Downloading nomic-ai/modernbert-embed-base..."
echo "  Model will be cached in ~/.cache/huggingface/hub/"

"$PYTHON" -c "
from mlx_embeddings.utils import load, generate
print('  Loading model from HuggingFace...')
model, tokenizer = load('nomic-ai/modernbert-embed-base')
output = generate(model, tokenizer, texts=['search_document: test'])
dims = output.text_embeds.shape[1]
print(f'  Model ready: {dims} dimensions, 8192 token context')
"

echo "  Download complete."
