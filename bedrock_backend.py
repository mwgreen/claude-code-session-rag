"""
Amazon Bedrock backend: embeddings (Cohere Embed v4, Titan Text Embeddings V2)
and one-shot text generation through the Converse API.

Used when CODE_RAG_BACKEND=bedrock / SESSION_RAG_BACKEND=bedrock, on machines
without MLX (Linux). The same file lives in code-rag and claude-code-session-rag;
keep the two copies identical rather than adding a dependency between the repos.

Credentials come from boto3's default chain (on an EC2 instance: the instance
role). Region: AWS_REGION, then AWS_DEFAULT_REGION, then us-west-2. boto3 is
imported lazily, so importing this module costs nothing on the MLX path.

Nothing here logs the text being embedded or described: only model IDs,
counts, sizes, timings and error codes.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("bedrock")

DEFAULT_REGION = "us-west-2"

# Embedding models the backend knows how to call. Cohere v4 is only invokable
# through its cross-region inference profile (the "us." ID), not the bare ID.
EMBED_MODELS: Dict[str, dict] = {
    "us.cohere.embed-v4:0": {
        "family": "cohere",
        "dims": (256, 512, 1024, 1536),
        "default_dim": 1024,
        "max_batch": 96,         # texts per InvokeModel call (API limit)
        # Cohere v4 accepts 128k tokens per input; the cap bounds cost and request size.
        # Code tokenizes at roughly 3-4 characters per token.
        "max_chars": 48_000,
    },
    "amazon.titan-embed-text-v2:0": {
        "family": "titan",
        "dims": (256, 512, 1024),
        "default_dim": 1024,
        "max_batch": 1,          # one text per call
        # API limits: 8,192 tokens and 50,000 characters. 24k characters stays
        # under the token limit for code as well as prose.
        "max_chars": 24_000,
    },
}

DEFAULT_EMBED_MODEL = "us.cohere.embed-v4:0"
DEFAULT_DESCRIPTION_MODEL = "us.amazon.nova-micro-v1:0"

# Bare IDs that need the inference-profile prefix to be invokable on demand.
_PROFILE_ONLY = {
    "cohere.embed-v4:0": "us.cohere.embed-v4:0",
    "amazon.nova-micro-v1:0": "us.amazon.nova-micro-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

RETRYABLE_ERROR_CODES = frozenset({
    "ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
    "InternalServerException", "ModelTimeoutException", "TooManyRequestsException",
})

MAX_ATTEMPTS = int(os.getenv("BEDROCK_MAX_ATTEMPTS", "8"))
BACKOFF_BASE = 0.5     # seconds
BACKOFF_CAP = 20.0     # seconds
SLOW_CALL_SECONDS = 5.0

# Patchable in tests.
_sleep: Callable[[float], None] = time.sleep
_random: Callable[[], float] = random.random


def check_model_id(model_id: str) -> str:
    """Reject bare IDs that are only invokable through an inference profile."""
    model_id = (model_id or "").strip()
    if model_id in _PROFILE_ONLY:
        raise ValueError(
            f"Bedrock model {model_id!r} is only invokable through its inference profile; "
            f"use {_PROFILE_ONLY[model_id]!r}."
        )
    return model_id


def region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION


def make_client(max_pool_connections: int = 16):
    """bedrock-runtime client using the default credential chain. botocore's own
    retries are off: call_with_retry() does backoff with jitter and counts retries."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=region(),
        config=Config(
            retries={"max_attempts": 1, "mode": "standard"},
            connect_timeout=10,
            read_timeout=120,
            max_pool_connections=max_pool_connections,
        ),
    )


# --- Usage accounting -------------------------------------------------------

class Usage:
    """Thread-safe per-model counters, reported on the health endpoints."""

    _FIELDS = ("calls", "texts", "input_tokens", "output_tokens", "retries", "errors", "truncated")

    def __init__(self):
        self._lock = threading.Lock()
        self._by_model: Dict[str, Dict[str, int]] = {}

    def add(self, model_id: str, **counts: int) -> None:
        with self._lock:
            row = self._by_model.setdefault(model_id, {f: 0 for f in self._FIELDS})
            for key, value in counts.items():
                row[key] = row.get(key, 0) + int(value or 0)

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {m: dict(row) for m, row in self._by_model.items()}

    def reset(self) -> None:
        with self._lock:
            self._by_model.clear()


USAGE = Usage()


# --- Retry ------------------------------------------------------------------

def _retry_reason(exc: BaseException) -> Optional[str]:
    """Error code if exc is worth retrying, else None."""
    try:
        from botocore.exceptions import (ClientError, ConnectionClosedError, ConnectTimeoutError,
                                         EndpointConnectionError, ReadTimeoutError)
    except ImportError:  # pragma: no cover - boto3 is required for this backend
        return None
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code if code in RETRYABLE_ERROR_CODES else None
    # Network blips, e.g. right after the machine resumes from hibernation.
    if isinstance(exc, (EndpointConnectionError, ConnectionClosedError, ConnectTimeoutError, ReadTimeoutError)):
        return type(exc).__name__
    return None


def backoff_delay(attempt: int) -> float:
    """Full-jitter exponential backoff: uniform in [0, min(cap, base * 2^(attempt-1))]."""
    return _random() * min(BACKOFF_CAP, BACKOFF_BASE * (2 ** (attempt - 1)))


def call_with_retry(fn: Callable[[], object], model_id: str, max_attempts: Optional[int] = None):
    attempts = max_attempts or MAX_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            reason = _retry_reason(exc)
            if reason is None or attempt >= attempts:
                USAGE.add(model_id, errors=1)
                raise
            delay = backoff_delay(attempt)
            USAGE.add(model_id, retries=1)
            logger.warning("Bedrock %s: %s; retry %d/%d in %.1fs", model_id, reason, attempt,
                           attempts - 1, delay)
            _sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


# --- Embeddings -------------------------------------------------------------

def embed_model_spec(model_id: str) -> dict:
    model_id = check_model_id(model_id)
    if model_id not in EMBED_MODELS:
        raise ValueError(f"Unknown Bedrock embedding model {model_id!r}. Choose from: {', '.join(EMBED_MODELS)}")
    return EMBED_MODELS[model_id]


def resolve_embed_dim(model_id: str, requested: Optional[str | int] = None) -> int:
    spec = embed_model_spec(model_id)
    if requested in (None, ""):
        return spec["default_dim"]
    dim = int(requested)
    if dim not in spec["dims"]:
        raise ValueError(f"{model_id} supports output dimensions {spec['dims']}, not {dim}")
    return dim


class BedrockEmbedder:
    """Embeds texts with a Bedrock model.

    input_type is "search_document" when indexing and "search_query" for
    queries (Cohere uses it; Titan V2 is symmetric and ignores it).
    """

    def __init__(self, model_id: str, dim: Optional[int] = None, client=None,
                 concurrency: Optional[int] = None, max_chars: Optional[int] = None):
        self.model_id = check_model_id(model_id)
        self.spec = embed_model_spec(self.model_id)
        self.family = self.spec["family"]
        self.dim = resolve_embed_dim(self.model_id, dim)
        self.max_batch = self.spec["max_batch"]
        self.max_chars = int(max_chars or self.spec["max_chars"])
        self.concurrency = max(1, int(concurrency or os.getenv("BEDROCK_EMBED_CONCURRENCY", "8")))
        self._client = client
        self._client_lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None

    @property
    def client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = make_client(max_pool_connections=max(16, self.concurrency * 2))
        return self._client

    def identity(self) -> Dict:
        return {"backend": "bedrock", "embed_model": self.model_id, "embed_dim": self.dim}

    def _prepare(self, texts: Sequence[str]) -> List[str]:
        out = []
        cut = 0
        for t in texts:
            t = t or " "
            if len(t) > self.max_chars:
                t = t[:self.max_chars]
                cut += 1
            out.append(t)
        if cut:
            logger.warning("Truncated %d of %d text(s) to %d characters for %s", cut, len(texts),
                           self.max_chars, self.model_id)
            USAGE.add(self.model_id, truncated=cut)
        return out

    def embed(self, texts: Sequence[str], input_type: str = "search_document") -> List[List[float]]:
        if not texts:
            return []
        texts = self._prepare(texts)
        t0 = time.perf_counter()
        if self.family == "cohere":
            vectors: List[List[float]] = []
            for start in range(0, len(texts), self.max_batch):
                vectors.extend(self._cohere_batch(texts[start:start + self.max_batch], input_type))
        else:
            if len(texts) == 1 or self.concurrency == 1:
                vectors = [self._titan_one(t) for t in texts]
            else:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=self.concurrency,
                                                    thread_name_prefix="bedrock-embed")
                vectors = list(self._pool.map(self._titan_one, texts))
        elapsed = time.perf_counter() - t0
        log = logger.info if elapsed > SLOW_CALL_SECONDS else logger.debug
        log("Embedded %d text(s) with %s in %.2fs", len(texts), self.model_id, elapsed)
        return vectors

    def _check(self, vec: List[float]) -> List[float]:
        if len(vec) != self.dim:
            raise RuntimeError(f"{self.model_id} returned {len(vec)} dims, expected {self.dim}")
        return vec

    def _cohere_batch(self, batch: List[str], input_type: str) -> List[List[float]]:
        body = json.dumps({
            "texts": batch,
            "input_type": input_type,
            "embedding_types": ["float"],
            "output_dimension": self.dim,
        })

        def call():
            return self.client.invoke_model(modelId=self.model_id, body=body,
                                            contentType="application/json", accept="application/json")

        resp = call_with_retry(call, self.model_id)
        payload = json.loads(resp["body"].read())
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, dict):
            embeddings = embeddings.get("float")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise RuntimeError(f"{self.model_id} returned {len(embeddings or [])} embeddings for "
                               f"{len(batch)} texts")
        headers = (resp.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
        tokens = headers.get("x-amzn-bedrock-input-token-count") or 0
        USAGE.add(self.model_id, calls=1, texts=len(batch), input_tokens=int(tokens))
        return [self._check(list(map(float, v))) for v in embeddings]

    def _titan_one(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})

        def call():
            return self.client.invoke_model(modelId=self.model_id, body=body,
                                            contentType="application/json", accept="application/json")

        resp = call_with_retry(call, self.model_id)
        payload = json.loads(resp["body"].read())
        USAGE.add(self.model_id, calls=1, texts=1, input_tokens=int(payload.get("inputTextTokenCount") or 0))
        return self._check(list(map(float, payload["embedding"])))


# --- Text generation (Converse) ----------------------------------------------

class BedrockGenerator:
    """Single-turn text generation through the Converse API."""

    def __init__(self, model_id: str, client=None, max_tokens: int = 80, temperature: float = 0.0):
        self.model_id = check_model_id(model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client
        self._client_lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = make_client()
        return self._client

    def generate(self, prompt: str) -> str:
        def call():
            return self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
            )

        resp = call_with_retry(call, self.model_id)
        usage = resp.get("usage") or {}
        USAGE.add(self.model_id, calls=1, input_tokens=usage.get("inputTokens", 0),
                  output_tokens=usage.get("outputTokens", 0))
        parts = (resp.get("output") or {}).get("message", {}).get("content") or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
