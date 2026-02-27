"""
RAG engine for session transcripts.

Embeds conversation turns with ModernBERT Embed Base (768 dims, 8192 token context)
via mlx-embeddings. Stores vectors in Milvus Lite, one DB per project.

Full-text search via SQLite FTS5 sidecar for hybrid search (vector + keyword).
Results merged with Reciprocal Rank Fusion (RRF).
"""

import hashlib
import os
from pathlib import Path

# Auto-enable offline mode if model is already cached (avoid network calls)
# Must be set BEFORE importing mlx_embeddings which loads huggingface_hub
_model_cache = Path.home() / ".cache/huggingface/hub/models--nomic-ai--modernbert-embed-base"
if _model_cache.exists() and 'HF_HUB_OFFLINE' not in os.environ:
    os.environ['HF_HUB_OFFLINE'] = '1'

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from mlx_embeddings.utils import load as mlx_load, generate as mlx_generate
import mlx.core as mx
from contextlib import contextmanager
from typing import List, Dict, Optional
import asyncio
import logging
import sys
import time

from fts_hybrid import FTSIndex, rrf_merge

logger = logging.getLogger("session-rag.milvus")

# --- Embedding model ---

_EMBED_DIM = 768
_MODEL_ID = "nomic-ai/modernbert-embed-base"
COLLECTION_NAME = "sessions"

# ModernBERT Embed uses the same Nomic task prefixes
_SEARCH_PREFIX = "search_query: "
_DOCUMENT_PREFIX = "search_document: "

_mlx_model = None
_mlx_tokenizer = None


def get_model():
    """Get or load the MLX embedding model (one-time load)."""
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer

    print(f"Loading {_MODEL_ID} via mlx-embeddings...", file=sys.stderr)
    _mlx_model, _mlx_tokenizer = mlx_load(_MODEL_ID)
    print(f"{_MODEL_ID} ready ({_EMBED_DIM} dims, 8192 token context)", file=sys.stderr)
    return _mlx_model, _mlx_tokenizer


def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Embed texts using ModernBERT. Adds task prefix per Nomic spec."""
    model, tokenizer = get_model()
    prefix = _SEARCH_PREFIX if is_query else _DOCUMENT_PREFIX
    prefixed = [prefix + t for t in texts]
    output = mlx_generate(model, tokenizer, texts=prefixed)
    embeddings = output.text_embeds.tolist()
    mx.clear_cache()
    return embeddings


# --- Milvus client management ---

_persistent_clients: Dict[str, MilvusClient] = {}
_fts = FTSIndex("turns_fts", ["session_id", "git_branch", "turn_index", "timestamp", "chunk_type"])
_write_lock: Optional[asyncio.Lock] = None
_embed_semaphore: Optional[asyncio.Semaphore] = None
_server_mode = False


def init_server_mode():
    """Initialize async concurrency primitives for HTTP server mode."""
    global _write_lock, _embed_semaphore, _server_mode
    _write_lock = asyncio.Lock()
    _embed_semaphore = asyncio.Semaphore(1)
    _server_mode = True
    _fts.set_server_mode(True)
    print("Server mode initialized", file=sys.stderr)


def close_server_mode():
    """Close all persistent clients (Milvus + FTS) and reset server mode."""
    global _write_lock, _embed_semaphore, _server_mode
    for path, client in list(_persistent_clients.items()):
        try:
            client.close()
            logger.info("Closed Milvus client: %s", path)
        except Exception as e:
            logger.warning("Error closing Milvus client %s: %s", path, e)
    _persistent_clients.clear()
    _fts.close_all()
    _write_lock = None
    _embed_semaphore = None
    _server_mode = False


def _get_persistent_client(db_path: str) -> MilvusClient:
    """Get or create a persistent client for the given DB path.
    On failure, evicts the stale client and retries once."""
    if db_path in _persistent_clients:
        try:
            _persistent_clients[db_path].has_collection(COLLECTION_NAME)
            return _persistent_clients[db_path]
        except Exception as e:
            logger.warning("Stale Milvus client for %s: %s — reconnecting", db_path, e)
            try:
                _persistent_clients[db_path].close()
            except Exception:
                pass
            del _persistent_clients[db_path]

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        _persistent_clients[db_path] = MilvusClient(db_path)
        logger.info("Opened client: %s", db_path)
    except Exception as e:
        logger.error("Failed to connect to Milvus at %s: %s", db_path, e)
        raise
    return _persistent_clients[db_path]


def _resolve_db_path(db_path: Optional[str]) -> str:
    if not db_path:
        raise ValueError("db_path is required. Each project stores its index at {project}/.session-rag/milvus.db")
    return db_path


def _ensure_collection(client: MilvusClient):
    """Create the sessions collection with explicit schema if it doesn't exist."""
    if client.has_collection(COLLECTION_NAME):
        return

    print(f"Creating collection: {COLLECTION_NAME} (dim={_EMBED_DIM})", file=sys.stderr)

    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=_EMBED_DIM),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="transcript_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="turn_index", dtype=DataType.INT64),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="git_branch", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=64),
    ])

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    print(f"Collection created: {COLLECTION_NAME}", file=sys.stderr)


@contextmanager
def milvus_client(db_path: Optional[str] = None):
    """Get a Milvus client. In server mode, reuses persistent client."""
    path = _resolve_db_path(db_path)

    if _server_mode:
        client = _get_persistent_client(path)
        _ensure_collection(client)
        yield client
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        client = MilvusClient(path)
        _ensure_collection(client)
        try:
            yield client
        finally:
            client.close()


# --- Core operations ---

def add_turns(turns: List[Dict], db_path: Optional[str] = None) -> int:
    """Insert conversation turn chunks into Milvus. Dedup by doc_id.

    Each turn dict should have:
        text, doc_id, session_id, transcript_file, turn_index,
        timestamp, git_branch, chunk_type
    """
    if not turns:
        return 0

    # Dedup: check which doc_ids already exist
    with milvus_client(db_path) as client:
        existing_ids = set()
        for turn in turns:
            doc_id = turn["doc_id"]
            try:
                results = client.query(
                    collection_name=COLLECTION_NAME,
                    filter=f'doc_id == "{doc_id}"',
                    limit=1,
                    output_fields=["doc_id"],
                )
                if results:
                    existing_ids.add(doc_id)
            except Exception as e:
                logger.warning("Dedup check failed for doc_id %s: %s", doc_id, e)

    new_turns = [t for t in turns if t["doc_id"] not in existing_ids]
    if not new_turns:
        return 0

    # Embed texts
    texts = [t["text"] for t in new_turns]
    embeddings = embed_texts(texts, is_query=False)

    data = []
    for turn, emb in zip(new_turns, embeddings):
        # Stable hash: SHA-256 truncated to int64. Python's hash() is
        # randomized per process, so the same doc_id would get different
        # primary keys across server restarts.
        int_id = int(hashlib.sha256(turn["doc_id"].encode()).hexdigest()[:15], 16)
        data.append({
            "id": int_id,
            "vector": emb,
            "document": turn["text"][:65535],
            "doc_id": turn["doc_id"],
            "session_id": turn.get("session_id", ""),
            "transcript_file": turn.get("transcript_file", ""),
            "turn_index": turn.get("turn_index", 0),
            "timestamp": turn.get("timestamp", ""),
            "git_branch": turn.get("git_branch", ""),
            "chunk_type": turn.get("chunk_type", "turn"),
        })

    with milvus_client(db_path) as client:
        client.insert(collection_name=COLLECTION_NAME, data=data)

    # Dual-write into FTS5 sidecar
    try:
        if db_path:
            fts_conn = _fts.connection(db_path)
            fts_records = [{
                "doc_id": t["doc_id"],
                "content": t["text"],
                "session_id": t.get("session_id", ""),
                "git_branch": t.get("git_branch", ""),
                "turn_index": t.get("turn_index", 0),
                "timestamp": t.get("timestamp", ""),
                "chunk_type": t.get("chunk_type", "turn"),
            } for t in new_turns]
            _fts.insert(fts_conn, fts_records)
            _fts.close_ephemeral(fts_conn)
    except Exception as e:
        logger.warning("FTS insert failed (non-fatal): %s", e)

    return len(data)


def search(query: str, n: int = 5, session_id: Optional[str] = None,
           git_branch: Optional[str] = None, recency_boost: bool = False,
           db_path: Optional[str] = None) -> List[Dict]:
    """Hybrid search: vector similarity + FTS5 keyword search, merged with RRF.

    Both engines run with an expanded candidate pool (n*3), then RRF merges
    the two ranked lists. Recency boost is applied after merging.
    """
    # Expanded candidate pool for both engines
    fetch_n = n * 3

    # --- Vector search ---
    query_embedding = embed_texts([query], is_query=True)[0]

    filters = []
    if session_id:
        filters.append(f'session_id == "{session_id}"')
    if git_branch:
        filters.append(f'git_branch == "{git_branch}"')
    filter_expr = " && ".join(filters) if filters else None

    with milvus_client(db_path) as client:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=fetch_n,
            filter=filter_expr,
            output_fields=["document", "doc_id", "session_id", "transcript_file",
                           "turn_index", "timestamp", "git_branch", "chunk_type"],
        )

    vector_results = []
    if results and results[0]:
        for hit in results[0]:
            entity = hit["entity"]
            vector_results.append({
                "content": entity["document"],
                "doc_id": entity.get("doc_id", ""),
                "session_id": entity.get("session_id", ""),
                "transcript_file": entity.get("transcript_file", ""),
                "turn_index": entity.get("turn_index", 0),
                "timestamp": entity.get("timestamp", ""),
                "git_branch": entity.get("git_branch", ""),
                "chunk_type": entity.get("chunk_type", ""),
                "distance": hit["distance"],
            })

    # --- FTS5 keyword search ---
    fts_filters = {}
    if session_id:
        fts_filters["session_id"] = session_id
    if git_branch:
        fts_filters["git_branch"] = git_branch
    fts_results = _fts.search(query, n=fetch_n, filters=fts_filters or None, db_path=db_path)

    # --- Merge with RRF ---
    if fts_results and vector_results:
        # Both engines returned results — merge
        merged = rrf_merge(vector_results, fts_results, n=fetch_n)
    elif fts_results:
        merged = fts_results
    else:
        merged = vector_results

    if not merged:
        return []

    # Clean up internal RRF score before recency boost
    for r in merged:
        r.pop("_rrf_score", None)

    if recency_boost and merged:
        merged = _apply_recency_boost(merged, n)

    return merged[:n]


def _apply_recency_boost(results: List[Dict], n: int) -> List[Dict]:
    """Re-rank results by combining semantic similarity with recency.

    Score = similarity * (1 + recency_weight * recency_factor)
    where recency_factor is 1.0 for the newest result and 0.0 for the oldest.
    """
    recency_weight = 0.3

    # Parse timestamps and sort to find range
    timestamps = []
    for r in results:
        ts = r.get("timestamp", "")
        if ts:
            try:
                # ISO 8601 strings sort lexicographically
                timestamps.append(ts)
            except Exception:
                timestamps.append("")
        else:
            timestamps.append("")

    if not any(timestamps):
        return results

    valid_ts = [t for t in timestamps if t]
    if len(valid_ts) < 2:
        return results

    ts_min = min(valid_ts)
    ts_max = max(valid_ts)

    for i, r in enumerate(results):
        similarity = 1 - r["distance"]  # COSINE distance → similarity
        ts = timestamps[i]
        if ts and ts_min != ts_max:
            # Normalize timestamp to [0, 1] range
            recency = (valid_ts.index(ts) if ts in valid_ts else 0) / max(len(valid_ts) - 1, 1)
            # Simple linear approach: newer timestamps get higher recency
            # Since ISO strings sort ascending, higher position = newer
            all_sorted = sorted(valid_ts)
            try:
                pos = all_sorted.index(ts)
                recency = pos / max(len(all_sorted) - 1, 1)
            except ValueError:
                recency = 0.5
        else:
            recency = 0.5

        r["_score"] = similarity * (1 + recency_weight * recency)

    results.sort(key=lambda r: r.get("_score", 0), reverse=True)

    # Clean up internal score
    for r in results:
        r.pop("_score", None)

    return results


def get_turns(session_id: str, turn_index: int, context: int = 2,
              db_path: Optional[str] = None) -> List[Dict]:
    """Retrieve turns around a specific turn_index within a session.

    turn_index is a byte offset into the transcript file. context is the
    number of neighboring turns (before and after) to include. We fetch all
    turns for the session, sort by turn_index, find the target, and return
    the surrounding window.

    Returns turns sorted by turn_index ascending, with the same field
    mapping as search() (document → content).
    """
    with milvus_client(db_path) as client:
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'session_id == "{session_id}"',
            output_fields=["document", "doc_id", "session_id", "transcript_file",
                           "turn_index", "timestamp", "git_branch", "chunk_type"],
            limit=16384,
        )

    if not results:
        return []

    # Sort all turns by turn_index (byte offset)
    results.sort(key=lambda r: r.get("turn_index", 0))

    # Find the target turn (closest match to requested turn_index)
    target_idx = 0
    min_dist = float("inf")
    for i, row in enumerate(results):
        dist = abs(row.get("turn_index", 0) - turn_index)
        if dist < min_dist:
            min_dist = dist
            target_idx = i

    # Extract window: context turns before and after
    start = max(0, target_idx - context)
    end = min(len(results), target_idx + context + 1)

    formatted = []
    for row in results[start:end]:
        formatted.append({
            "content": row["document"],
            "doc_id": row.get("doc_id", ""),
            "session_id": row.get("session_id", ""),
            "transcript_file": row.get("transcript_file", ""),
            "turn_index": row.get("turn_index", 0),
            "timestamp": row.get("timestamp", ""),
            "git_branch": row.get("git_branch", ""),
            "chunk_type": row.get("chunk_type", ""),
        })

    return formatted


def get_stats(db_path: Optional[str] = None) -> Dict:
    """Get index statistics."""
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return {"total_turns": 0, "sessions": 0, "by_type": {}}

        stats = client.get_collection_stats(COLLECTION_NAME)
        total = stats["row_count"]

    # Query for breakdowns (capped by Milvus offset limit)
    all_results = _query_all(["session_id", "chunk_type", "git_branch"], db_path=db_path)

    sessions = set(r["session_id"] for r in all_results if r.get("session_id"))
    branches = set(r["git_branch"] for r in all_results if r.get("git_branch"))

    by_type = {}
    for r in all_results:
        t = r.get("chunk_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "total_turns": total,
        "sessions": len(sessions),
        "branches": sorted(branches),
        "by_type": by_type,
    }


def _query_all(output_fields: list, batch_size: int = 1000, db_path: Optional[str] = None) -> list:
    """Query all rows with offset pagination."""
    MILVUS_MAX = 16384
    all_results = []
    offset = 0

    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return []
        while offset < MILVUS_MAX:
            effective_limit = min(batch_size, MILVUS_MAX - offset)
            batch = client.query(
                collection_name=COLLECTION_NAME,
                filter="",
                limit=effective_limit,
                offset=offset,
                output_fields=output_fields,
            )
            if not batch:
                break
            all_results.extend(batch)
            if len(batch) < effective_limit:
                break
            offset += effective_limit

    return all_results


# --- Cleanup operations ---

def delete_by_session(session_id: str, db_path: Optional[str] = None) -> int:
    """Delete all turns for a given session ID."""
    with milvus_client(db_path) as client:
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'session_id == "{session_id}"',
            output_fields=["id"],
        )
        if results:
            client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'session_id == "{session_id}"',
            )

    # Also delete from FTS
    try:
        if db_path:
            conn = _fts.connection(db_path)
            _fts.delete(conn, "session_id", session_id)
            _fts.close_ephemeral(conn)
    except Exception as e:
        logger.warning("FTS delete by session failed (non-fatal): %s", e)

    return len(results)


def delete_by_branch(git_branch: str, db_path: Optional[str] = None) -> int:
    """Delete all turns for a given git branch."""
    with milvus_client(db_path) as client:
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'git_branch == "{git_branch}"',
            output_fields=["id"],
        )
        if results:
            client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'git_branch == "{git_branch}"',
            )

    # Also delete from FTS
    try:
        if db_path:
            conn = _fts.connection(db_path)
            _fts.delete(conn, "git_branch", git_branch)
            _fts.close_ephemeral(conn)
    except Exception as e:
        logger.warning("FTS delete by branch failed (non-fatal): %s", e)

    return len(results)


def delete_older_than(max_age_days: int, db_path: Optional[str] = None) -> int:
    """Delete all turns with timestamps older than max_age_days ago.

    Returns the number of deleted turns.
    """
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

    # Milvus Lite varchar comparison works lexicographically,
    # and ISO 8601 timestamps sort correctly this way.
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return 0

        # Query to count before deleting
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'timestamp < "{cutoff_str}" && timestamp != ""',
            output_fields=["id"],
            limit=16384,
        )
        if results:
            client.delete(
                collection_name=COLLECTION_NAME,
                filter=f'timestamp < "{cutoff_str}" && timestamp != ""',
            )

    # Also delete from FTS
    try:
        if db_path:
            conn = _fts.connection(db_path)
            _fts.delete_where(conn, "timestamp < ? AND timestamp != ''", (cutoff_str,))
            _fts.close_ephemeral(conn)
    except Exception as e:
        logger.warning("FTS delete older_than failed (non-fatal): %s", e)

    return len(results)


def clear_collection(db_path: Optional[str] = None):
    """Drop and recreate the collection (full reset). Also clears FTS."""
    with milvus_client(db_path) as client:
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            print(f"Collection dropped: {COLLECTION_NAME}", file=sys.stderr)

    # Clear FTS database
    if db_path:
        _fts.clear(db_path)


def list_sessions(db_path: Optional[str] = None) -> List[Dict]:
    """List all sessions with turn counts and date ranges."""
    all_results = _query_all(
        ["session_id", "timestamp", "git_branch", "chunk_type"],
        db_path=db_path,
    )

    sessions: Dict[str, Dict] = {}
    for r in all_results:
        sid = r.get("session_id", "")
        if not sid:
            continue
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "turns": 0,
                "branches": set(),
                "min_ts": "",
                "max_ts": "",
            }
        s = sessions[sid]
        s["turns"] += 1
        branch = r.get("git_branch", "")
        if branch:
            s["branches"].add(branch)
        ts = r.get("timestamp", "")
        if ts:
            if not s["min_ts"] or ts < s["min_ts"]:
                s["min_ts"] = ts
            if not s["max_ts"] or ts > s["max_ts"]:
                s["max_ts"] = ts

    result = []
    for s in sessions.values():
        s["branches"] = sorted(s["branches"])
        result.append(s)

    # Sort by most recent first
    result.sort(key=lambda s: s["max_ts"], reverse=True)
    return result


# --- Async wrappers ---

async def search_async(query: str, n: int = 5, session_id: Optional[str] = None,
                       git_branch: Optional[str] = None, recency_boost: bool = False,
                       db_path: Optional[str] = None) -> List[Dict]:
    """Async search with embed semaphore."""
    loop = asyncio.get_event_loop()

    if _embed_semaphore is not None:
        async with _embed_semaphore:
            return await loop.run_in_executor(
                None, lambda: search(query, n, session_id, git_branch, recency_boost, db_path))
    else:
        return await loop.run_in_executor(
            None, lambda: search(query, n, session_id, git_branch, recency_boost, db_path))


async def add_turns_async(turns: List[Dict], db_path: Optional[str] = None) -> int:
    """Async add_turns with embed semaphore + write lock."""
    loop = asyncio.get_event_loop()

    if _embed_semaphore is not None and _write_lock is not None:
        async with _embed_semaphore:
            async with _write_lock:
                return await loop.run_in_executor(None, lambda: add_turns(turns, db_path))
    else:
        return await loop.run_in_executor(None, lambda: add_turns(turns, db_path))
