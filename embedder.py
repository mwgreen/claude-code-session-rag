"""
Embedding backends for session-rag (Apple Silicon / MLX).

A small registry of supported models plus one `Embedder` class that hides the
per-architecture differences (prompt formats, pooling, attention masks).

Why we do not just call `mlx_embeddings.generate()`:
  * EmbeddingGemma is a *bidirectional* encoder. mlx-embeddings 0.0.5 passes the
    tokenizer's 0/1 attention mask straight into the attention layers as an
    additive mask, so padded batches leak padding tokens into every document
    embedding (measured: cos(alone, batched) = 0.90 for the same text). It also
    ignores the 512-token sliding window on local layers. We run the reference
    forward ourselves: bidirectional, padding masked, sliding window on local
    layers, mean pooling, then the two Dense projections, then L2 normalise.
  * Qwen3-Embedding is a causal decoder with last-token pooling and an
    instruction-formatted query. The mlx-embeddings qwen3 module handles the
    masks correctly; we only add the prompt format and length-aware batching.

Models must be downloaded ahead of time (see `python embedder.py download`).
At runtime all Hugging Face network access is disabled.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Runtime is offline unless a download was explicitly requested.
if os.environ.get("SESSION_RAG_ALLOW_DOWNLOAD") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger("session-rag.embedder")

HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    dim: int
    max_tokens: int
    backend: str  # "gemma" | "qwen3" | "generic"
    query_prefix: str = ""
    document_prefix: str = ""
    description: str = ""

    @property
    def cache_dir(self) -> Path:
        return HF_CACHE / ("models--" + self.model_id.replace("/", "--"))

    def is_downloaded(self) -> bool:
        snaps = self.cache_dir / "snapshots"
        if not snaps.is_dir():
            return False
        return any((s / "config.json").exists() for s in snaps.iterdir())


_QWEN_QUERY_INSTRUCTION = (
    "Instruct: Given a question about past coding conversations, retrieve the "
    "conversation turns that answer it\nQuery: "
)

REGISTRY: Dict[str, ModelSpec] = {
    "embeddinggemma": ModelSpec(
        name="embeddinggemma",
        model_id="mlx-community/embeddinggemma-300m-bf16",
        dim=768,
        max_tokens=2048,
        backend="gemma",
        query_prefix="task: search result | query: ",
        document_prefix="title: none | text: ",
        description="Google EmbeddingGemma-300M (bf16). ~600 MB. 2048-token context.",
    ),
    "qwen3": ModelSpec(
        name="qwen3",
        model_id="mlx-community/Qwen3-Embedding-0.6B-8bit",
        dim=1024,
        max_tokens=8192,
        backend="qwen3",
        query_prefix=_QWEN_QUERY_INSTRUCTION,
        document_prefix="",
        description="Qwen3-Embedding-0.6B (8-bit). ~620 MB. Long context, strongest on code.",
    ),
    "qwen3-4bit": ModelSpec(
        name="qwen3-4bit",
        model_id="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        dim=1024,
        max_tokens=8192,
        backend="qwen3",
        query_prefix=_QWEN_QUERY_INSTRUCTION,
        document_prefix="",
        description="Qwen3-Embedding-0.6B (4-bit DWQ). ~350 MB. Smaller/faster, slight quality loss.",
    ),
    "modernbert": ModelSpec(
        name="modernbert",
        model_id="nomic-ai/modernbert-embed-base",
        dim=768,
        max_tokens=8192,
        backend="generic",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        description="Nomic ModernBERT Embed Base. 8192-token context.",
    ),
}

DEFAULT_MODEL = "embeddinggemma"
CONFIG_PATH = Path.home() / ".session-rag" / "config.json"


def read_config() -> Dict:
    """~/.session-rag/config.json: settings shared by the server (however it is
    launched: hook, launchd, or by hand) and the CLI. Currently: {"model": name}."""
    try:
        import json
        data = json.loads(CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_config(**updates) -> None:
    import json
    import tempfile
    data = read_config()
    data.update(updates)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="config.", suffix=".tmp", dir=str(CONFIG_PATH.parent))
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, CONFIG_PATH)


def resolve_model_name(name: Optional[str] = None) -> str:
    """Model short name: explicit argument, then SESSION_RAG_MODEL, then
    ~/.session-rag/config.json, then the default."""
    resolved = (name or os.environ.get("SESSION_RAG_MODEL") or read_config().get("model")
                or DEFAULT_MODEL)
    resolved = str(resolved).strip().lower()
    if resolved not in REGISTRY:
        raise ValueError(
            f"Unknown embedding model '{resolved}'. Valid options: {', '.join(REGISTRY)}"
        )
    return resolved


def get_spec(name: Optional[str] = None) -> ModelSpec:
    return REGISTRY[resolve_model_name(name)]


# --- Batching helpers -------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    # Code/log heavy text tokenises at roughly 3 chars per token.
    return max(8, len(text) // 3)


def plan_batches(texts: Sequence[str], max_tokens: int, token_budget: int = 12288,
                 max_batch: int = 32) -> List[List[int]]:
    """Group indices so that (batch size x longest sequence) stays within a token budget.

    Texts are sorted by length so padding is minimal; the caller re-scatters
    results using the returned indices.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    batches: List[List[int]] = []
    current: List[int] = []
    longest = 0
    for i in order:
        est = min(_estimate_tokens(texts[i]), max_tokens)
        new_longest = max(longest, est)
        if current and ((len(current) + 1) * new_longest > token_budget or len(current) >= max_batch):
            batches.append(current)
            current, longest = [], 0
            new_longest = est
        current.append(i)
        longest = new_longest
    if current:
        batches.append(current)
    return batches


# --- Embedder ---------------------------------------------------------------

class Embedder:
    """Loads one MLX embedding model and produces L2-normalised float vectors."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._model = None
        self._tokenizer = None
        self._custom_forward_ok = True

    # -- loading --
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is not None:
            return
        if not self.spec.is_downloaded():
            raise RuntimeError(
                f"Embedding model '{self.spec.name}' ({self.spec.model_id}) is not downloaded "
                f"(expected under {self.spec.cache_dir}). Run: ./download-model.sh {self.spec.name}"
            )
        from mlx_embeddings.utils import load as mlx_load  # heavy import, deferred

        logger.info("Loading %s via mlx-embeddings...", self.spec.model_id)
        self._model, self._tokenizer = mlx_load(self.spec.model_id)
        import mlx.core as mx
        # Bound MLX's buffer cache instead of clearing it after every call
        # (clearing costs ~60 ms of re-allocation on the next call).
        mx.set_cache_limit(int(os.environ.get("SESSION_RAG_MLX_CACHE_MB", "768")) * 1024 * 1024)
        logger.info("%s ready (%d dims, %d-token context, backend=%s)",
                    self.spec.model_id, self.spec.dim, self.spec.max_tokens, self.spec.backend)

    # -- public API --
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed([self.spec.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._embed([self.spec.query_prefix + text])[0]

    def identity(self) -> Dict:
        return {"model_name": self.spec.name, "model_id": self.spec.model_id, "embed_dim": self.spec.dim}

    # -- internals --
    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        self.load()
        import mlx.core as mx

        out: List[Optional[List[float]]] = [None] * len(texts)
        for idx in plan_batches(texts, self.spec.max_tokens):
            batch = [texts[i] for i in idx]
            vecs = self._embed_batch(batch)
            mx.eval(vecs)
            rows = vecs.astype(mx.float32).tolist()
            for i, row in zip(idx, rows):
                out[i] = row
        return out  # type: ignore[return-value]

    def _encode(self, texts: Sequence[str]):
        enc = self._tokenizer.batch_encode_plus(
            list(texts), return_tensors="mlx", padding=True,
            truncation=True, max_length=self.spec.max_tokens,
        )
        return enc["input_ids"], enc["attention_mask"]

    def _embed_batch(self, texts: Sequence[str]):
        backend = self.spec.backend
        if backend == "gemma":
            return self._embed_gemma(texts)
        if backend == "qwen3":
            ids, mask = self._encode(texts)
            return self._model(ids, attention_mask=mask).text_embeds
        # generic: let mlx-embeddings handle tokenisation + pooling (mean pooling models)
        from mlx_embeddings.utils import generate as mlx_generate
        return mlx_generate(self._model, self._tokenizer, texts=list(texts),
                            max_length=self.spec.max_tokens).text_embeds

    def _embed_gemma(self, texts: Sequence[str]):
        import mlx.core as mx
        from mlx_embeddings.models.base import mean_pooling, normalize_embeddings

        ids, mask = self._encode(texts)
        if self._custom_forward_ok:
            try:
                return self._gemma_reference_forward(ids, mask, mean_pooling, normalize_embeddings)
            except Exception as exc:  # API drift in mlx_lm / mlx_embeddings
                self._custom_forward_ok = False
                logger.warning("EmbeddingGemma reference forward failed (%s); falling back to "
                               "per-text stock forward (slower, no sliding window).", exc)
        # Stock forward is only correct without padding, so embed one text at a time.
        rows = []
        for i in range(ids.shape[0]):
            single = self._encode([texts[i]])
            rows.append(self._model(single[0], attention_mask=single[1]).text_embeds)
        return mx.concatenate(rows, axis=0)

    def _gemma_reference_forward(self, ids, mask, mean_pooling, normalize_embeddings):
        """Bidirectional forward matching the Hugging Face EmbeddingGemma implementation."""
        import mlx.core as mx

        model = self._model
        cfg = model.config
        inner = model.model
        batch, length = ids.shape

        h = inner.embed_tokens(ids)
        h = h * mx.array(cfg.hidden_size ** 0.5, inner.embed_tokens.weight.dtype).astype(h.dtype)

        keys_ok = (mask == 1)[:, None, None, :]                       # (B,1,1,L)
        eye = mx.eye(length, dtype=mx.bool_)[None, None]              # every row may attend to itself
        full = mx.broadcast_to(keys_ok, (batch, 1, length, length)) | eye
        pos = mx.arange(length)
        window = (mx.abs(pos[:, None] - pos[None, :]) < cfg.sliding_window)[None, None]
        local = (full & window) | eye

        zero = mx.array(0, h.dtype)
        neg = mx.array(-mx.inf, h.dtype)
        full_add = mx.where(full, zero, neg)
        local_add = mx.where(local, zero, neg)

        pattern = cfg.sliding_window_pattern
        for i, layer in enumerate(inner.layers):
            is_global = (i % pattern == pattern - 1)
            h = layer(h, full_add if is_global else local_add, None)
        h = inner.norm(h)

        pooled = mean_pooling(h, mask)
        for dense in model.dense:
            pooled = dense(pooled)
        return normalize_embeddings(pooled)


# --- Download support (setup time only) --------------------------------------

def _build_ca_bundle() -> Optional[str]:
    """Return a CA bundle path that trusts both the public roots and any corporate
    proxy CA the machine already trusts for Node (NODE_EXTRA_CA_CERTS) or that the
    user points at with SESSION_RAG_CA_BUNDLE. Returns None to use defaults."""
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return None  # user already configured TLS trust
    extra = os.environ.get("SESSION_RAG_CA_BUNDLE") or os.environ.get("NODE_EXTRA_CA_CERTS")
    if not extra or not Path(extra).is_file():
        return None
    try:
        import certifi
        bundle = Path.home() / ".session-rag" / "ca-bundle.pem"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text(Path(certifi.where()).read_text() + "\n" + Path(extra).read_text())
        return str(bundle)
    except Exception as exc:
        logger.warning("Could not build CA bundle from %s: %s", extra, exc)
        return None


def download(name: Optional[str] = None) -> ModelSpec:
    """Download a model into the Hugging Face cache and smoke-test it."""
    if os.environ.get("SESSION_RAG_ALLOW_DOWNLOAD") != "1":
        raise RuntimeError("Set SESSION_RAG_ALLOW_DOWNLOAD=1 to enable downloads")
    spec = get_spec(name)
    bundle = _build_ca_bundle()
    if bundle:
        os.environ["SSL_CERT_FILE"] = bundle
        os.environ["REQUESTS_CA_BUNDLE"] = bundle
        print(f"  Using CA bundle {bundle}")
    from huggingface_hub import snapshot_download

    print(f"  Downloading {spec.model_id} ...")
    path = snapshot_download(spec.model_id)
    print(f"  Cached at {path}")
    emb = Embedder(spec)
    vec = emb.embed_documents(["smoke test"])[0]
    if len(vec) != spec.dim:
        raise RuntimeError(f"{spec.model_id} produced {len(vec)} dims, registry says {spec.dim}")
    print(f"  Model ready: {spec.dim} dimensions")
    return spec


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="session-rag embedding models")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List supported models")
    p_dl = sub.add_parser("download", help="Download a model into the HF cache")
    p_dl.add_argument("name", nargs="?", help="Model short name (default: SESSION_RAG_MODEL or embeddinggemma)")
    p_chk = sub.add_parser("check", help="Verify a model is downloaded and produces vectors")
    p_chk.add_argument("name", nargs="?")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        active = resolve_model_name()
        for spec in REGISTRY.values():
            flag = "*" if spec.name == active else " "
            state = "downloaded" if spec.is_downloaded() else "not downloaded"
            print(f"{flag} {spec.name:<15} {spec.model_id:<45} {spec.dim}d  {state}\n"
                  f"                  {spec.description}")
        return 0
    if args.cmd == "download":
        download(args.name)
        return 0
    if args.cmd == "check":
        spec = get_spec(args.name)
        emb = Embedder(spec)
        vec = emb.embed_documents(["check"])[0]
        print(f"{spec.name}: OK ({len(vec)} dims)")
        return 0
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(_main(sys.argv[1:]))
