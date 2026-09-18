"""
RAG engine for session transcripts.

Storage: one global Milvus Lite collection at ~/.session-rag/milvus.db (vectors +
metadata) mirrored by a SQLite FTS5 table (keyword search, and the source text
for re-embedding). Search is hybrid: cosine vector search + BM25, merged with
Reciprocal Rank Fusion, with an optional recency boost.

Concurrency: in server mode every engine call runs on ONE worker thread
(`run()`), which keeps Milvus/SQLite/MLX usage single-threaded and keeps the
asyncio event loop free while embeddings are computed.

Milvus note: for the COSINE metric Milvus reports *similarity* in the `distance`
field (1.0 = identical). We surface it as `similarity`.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# Quiet the gRPC keepalive chatter that milvus-lite writes to stderr.
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GLOG_minloglevel", "2")

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient  # noqa: E402

from embedder import Embedder, ModelSpec, get_spec  # noqa: E402
from fts_hybrid import FTSIndex, rrf_merge  # noqa: E402
from index_state import STATE_DIR, atomic_write_json, read_json  # noqa: E402

logger = logging.getLogger("session-rag.engine")

COLLECTION_NAME = "sessions"
DEFAULT_DB_PATH = str(STATE_DIR / "milvus.db")
IDENTITY_FILE = STATE_DIR / "model_identity.json"
MILVUS_PAGE_MAX = 16384          # Milvus caps offset+limit per query
_ID_BITS = 60                    # primary key = first 15 hex digits of sha256(doc_id)
_SCAN_BUCKETS = 64
RRF_K = 60
RECENCY_WEIGHT = 0.3
MAX_RESULTS = 50

METADATA_FIELDS = ["session_id", "transcript_file", "turn_index", "timestamp",
                   "git_branch", "chunk_type", "project_root"]
OUTPUT_FIELDS = ["document", "doc_id"] + METADATA_FIELDS
FTS_METADATA = ["session_id", "git_branch", "turn_index", "timestamp", "chunk_type",
                "project_root", "transcript_file"]

# Chunks produced by older parser versions that indexed Claude Code's own markup.
NOISE_PREFIXES = ("User: <command-name>", "User: <local-command", "User: <system-reminder>",
                  "User: <task-notification>", "User: <bash-input>", "User: <<")

# --- Module state -------------------------------------------------------------

_spec: ModelSpec = get_spec()
_embedder = Embedder(_spec)
_fts = FTSIndex("turns_fts", FTS_METADATA)
_clients: Dict[str, MilvusClient] = {}
_server_mode = False
_executor: Optional[ThreadPoolExecutor] = None


def use_model(name: Optional[str]) -> ModelSpec:
    """Switch the active embedding model (CLI use, before anything is loaded)."""
    global _spec, _embedder
    _spec = get_spec(name)
    _embedder = Embedder(_spec)
    return _spec


def get_model_name() -> str:
    return _spec.name


def model_spec() -> ModelSpec:
    return _spec


def load_model() -> None:
    _embedder.load()


def model_loaded() -> bool:
    return _embedder.loaded


def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    if is_query:
        return [_embedder.embed_query(t) for t in texts]
    return _embedder.embed_documents(texts)


# --- Server mode / worker thread ----------------------------------------------

def init_server_mode(db_path: Optional[str] = None) -> None:
    """Enable persistent connections and the single engine worker thread."""
    global _server_mode, _executor
    db_path = _resolve_db_path(db_path)
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-engine")
    _server_mode = True
    _fts.set_server_mode(True)
    _check_model_identity(db_path)
    logger.info("Server mode initialised (model=%s, dim=%d)", _spec.name, _spec.dim)


def close_server_mode() -> None:
    global _server_mode, _executor

    def _close():
        for path, client in list(_clients.items()):
            try:
                client.close()
            except Exception as exc:
                logger.warning("Error closing Milvus client %s: %s", path, exc)
        _clients.clear()
        _fts.close_all()

    if _executor is not None:
        try:
            _executor.submit(_close).result(timeout=30)
        except Exception as exc:
            logger.warning("Error during engine shutdown: %s", exc)
        _executor.shutdown(wait=False)
    else:
        _close()
    _executor = None
    _server_mode = False
    _fts.set_server_mode(False)


async def run(fn: Callable, *args, **kwargs):
    """Run a blocking engine function on the engine worker thread."""
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *args, **kwargs)
    if _executor is None:
        return await loop.run_in_executor(None, call)
    return await loop.run_in_executor(_executor, call)


# Backwards-compatible async wrappers
async def search_async(*args, **kwargs):
    return await run(search, *args, **kwargs)


async def add_turns_async(turns: List[Dict], db_path: Optional[str] = None) -> int:
    return await run(add_turns, turns, db_path=db_path)


# --- Model identity -----------------------------------------------------------

def read_identity() -> Dict:
    data = read_json(IDENTITY_FILE, {})
    return data if isinstance(data, dict) else {}


def stamp_identity() -> None:
    atomic_write_json(IDENTITY_FILE, _embedder.identity(), indent=None)


def _collection_dim(client: MilvusClient) -> Optional[int]:
    try:
        info = client.describe_collection(COLLECTION_NAME)
        for field in info.get("fields", []):
            if field.get("name") == "vector":
                return int(field.get("params", {}).get("dim"))
    except Exception as exc:
        logger.debug("describe_collection failed: %s", exc)
    return None


def _check_model_identity(db_path: str) -> None:
    """Refuse to mix vectors from different models. Stamps the identity on first use."""
    stored = read_identity()
    stored_name = stored.get("model_name")
    stored_id = stored.get("model_id")
    if stored_name and (stored_name != _spec.name or (stored_id and stored_id != _spec.model_id)):
        if count(db_path=db_path) > 0:
            raise RuntimeError(
                f"Model mismatch: the index was built with '{stored_name}' ({stored_id or 'unknown id'}) "
                f"but SESSION_RAG_MODEL is '{_spec.name}' ({_spec.model_id}). Stop the server and run "
                f"'python cleanup.py migrate-model --to {_spec.name}' to re-embed the existing index, "
                f"or 'python cleanup.py reset' to start empty."
            )
    with milvus_client(db_path) as client:
        if client.has_collection(COLLECTION_NAME):
            dim = _collection_dim(client)
            if dim and dim != _spec.dim:
                raise RuntimeError(
                    f"Collection vector dimension is {dim} but model '{_spec.name}' produces {_spec.dim}. "
                    f"Run 'python cleanup.py migrate-model --to {_spec.name}'."
                )
    stamp_identity()


# --- Milvus client management ---------------------------------------------------

def _resolve_db_path(db_path: Optional[str]) -> str:
    return db_path or DEFAULT_DB_PATH


def _open_client(db_path: str, attempts: int = 8, delay: float = 1.0) -> MilvusClient:
    """Open Milvus Lite. Retries briefly: right after a restart the previous
    process may still hold the database lock for a second or two."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return MilvusClient(db_path)
        except Exception as exc:  # pymilvus raises ConnectionConfigException("Open local milvus failed")
            last_exc = exc
            if attempt < attempts:
                logger.warning("Milvus database busy (%s); retry %d/%d in %.0fs", exc, attempt, attempts, delay)
                time.sleep(delay)
    raise RuntimeError(
        f"Could not open {db_path}: {last_exc}. Another session-rag process (or cleanup.py) "
        f"is probably using it. Check: ./session-rag-server.sh status"
    ) from last_exc


def _persistent_client(db_path: str) -> MilvusClient:
    client = _clients.get(db_path)
    if client is not None:
        try:
            client.has_collection(COLLECTION_NAME)
            return client
        except Exception as exc:
            logger.warning("Milvus client unusable (%s); reconnecting", exc)
            try:
                client.close()
            except Exception:
                pass
            _clients.pop(db_path, None)
    client = _open_client(db_path)
    _clients[db_path] = client
    return client


def _ensure_loaded(client: MilvusClient) -> None:
    """Milvus Lite 3.x opens an existing collection in the 'released' state, and every
    search/query then fails with "call load() before search/get/query". Loading is
    idempotent and instant for a FLAT index, so do it once per client."""
    if getattr(client, "_session_rag_loaded", False):
        return
    try:
        client.load_collection(COLLECTION_NAME)
    except Exception as exc:
        logger.warning("load_collection(%s) failed: %s", COLLECTION_NAME, exc)
        return
    client._session_rag_loaded = True


def _ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION_NAME):
        _ensure_loaded(client)
        return
    logger.info("Creating collection %s (dim=%d)", COLLECTION_NAME, _spec.dim)
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=_spec.dim),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="transcript_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="turn_index", dtype=DataType.INT64),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="git_branch", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="project_root", dtype=DataType.VARCHAR, max_length=512),
    ])
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
    client._session_rag_loaded = False   # a recreated collection (reembed_all, reindex) must be loaded again
    _ensure_loaded(client)


@contextmanager
def milvus_client(db_path: Optional[str] = None, ensure: bool = True):
    """Yield a Milvus client (persistent in server mode, ephemeral otherwise)."""
    path = _resolve_db_path(db_path)
    if _server_mode:
        client = _persistent_client(path)
        if ensure:
            _ensure_collection(client)
        yield client
    else:
        client = _open_client(path)
        try:
            if ensure:
                _ensure_collection(client)
            yield client
        finally:
            client.close()


# --- Filter helpers -----------------------------------------------------------

def _literal(value: str) -> str:
    """Quote a string for a Milvus filter expression."""
    value = str(value)
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    raise ValueError("filter value may not contain both single and double quotes")


def _project_filter(project_root: str, prefix: bool) -> str:
    exact = f"project_root == {_literal(project_root)}"
    if not prefix:
        return exact
    like = _literal(project_root.rstrip("/") + "/%")
    return f"({exact} or project_root like {like})"


def _build_filter(session_id: Optional[str] = None, git_branch: Optional[str] = None,
                  project_root: Optional[str] = None, project_prefix: bool = True) -> Optional[str]:
    parts = []
    if session_id:
        parts.append(f"session_id == {_literal(session_id)}")
    if git_branch:
        parts.append(f"git_branch == {_literal(git_branch)}")
    if project_root:
        parts.append(_project_filter(project_root, project_prefix))
    return " and ".join(parts) if parts else None


def _primary_key(doc_id: str) -> int:
    return int(hashlib.sha256(doc_id.encode()).hexdigest()[:15], 16)


def _ids_filter(doc_ids: Sequence[str]) -> str:
    return "doc_id in [" + ", ".join(_literal(d) for d in doc_ids) + "]"


# --- Scanning -----------------------------------------------------------------

def _query(client: MilvusClient, filter_expr: str, output_fields: List[str],
           limit: int = MILVUS_PAGE_MAX, offset: int = 0) -> List[Dict]:
    return client.query(collection_name=COLLECTION_NAME, filter=filter_expr or "",
                        output_fields=output_fields, limit=limit, offset=offset)


def _scan_with(client: MilvusClient, output_fields: List[str],
               filter_expr: Optional[str] = None) -> List[Dict]:
    """Every matching row via an open client. Milvus limits one query window to
    16384 rows, so the (uniformly distributed) primary-key space is bucketed."""
    rows: List[Dict] = []
    if not client.has_collection(COLLECTION_NAME):
        return rows
    step = (1 << _ID_BITS) // _SCAN_BUCKETS
    for b in range(_SCAN_BUCKETS):
        range_expr = f"id >= {b * step} and id < {(b + 1) * step}"
        expr = f"({filter_expr}) and {range_expr}" if filter_expr else range_expr
        batch = _query(client, expr, output_fields)
        if len(batch) >= MILVUS_PAGE_MAX:
            logger.warning("Scan bucket %d hit the Milvus page cap; results may be incomplete", b)
        rows.extend(batch)
    return rows


def _count_with(client: MilvusClient, filter_expr: Optional[str] = None) -> int:
    if not client.has_collection(COLLECTION_NAME):
        return 0
    try:
        res = client.query(collection_name=COLLECTION_NAME, filter=filter_expr or "",
                           output_fields=["count(*)"])
        return int(res[0]["count(*)"])
    except Exception as exc:
        logger.debug("count(*) unsupported (%s); falling back to scan", exc)
        return len(_scan_with(client, ["id"], filter_expr))


def scan_all(output_fields: List[str], filter_expr: Optional[str] = None,
             db_path: Optional[str] = None) -> List[Dict]:
    with milvus_client(db_path) as client:
        return _scan_with(client, output_fields, filter_expr)


def count(filter_expr: Optional[str] = None, db_path: Optional[str] = None) -> int:
    with milvus_client(db_path) as client:
        return _count_with(client, filter_expr)


def project_filter(project_root: str, prefix: bool = True) -> str:
    """Public helper: Milvus filter for one project (and its subdirectories)."""
    return _project_filter(project_root, prefix)


# --- Writes -------------------------------------------------------------------

def _fts_record(turn: Dict) -> Dict:
    rec = {"doc_id": turn["doc_id"], "content": turn["text"]}
    for col in FTS_METADATA:
        rec[col] = turn.get(col, 0 if col == "turn_index" else "")
    return rec


def existing_doc_ids(doc_ids: Sequence[str], db_path: Optional[str] = None) -> set:
    found = set()
    with milvus_client(db_path) as client:
        ids = list(dict.fromkeys(doc_ids))
        for i in range(0, len(ids), 200):
            chunk = ids[i:i + 200]
            try:
                rows = _query(client, _ids_filter(chunk), ["doc_id"], limit=len(chunk))
                found.update(r["doc_id"] for r in rows)
            except Exception as exc:
                logger.warning("Dedup query failed (%s); assuming chunk is new", exc)
    return found


def add_turns(turns: List[Dict], db_path: Optional[str] = None) -> int:
    """Embed and insert chunks. Existing doc_ids are skipped. Returns inserted count."""
    if not turns:
        return 0
    db_path = _resolve_db_path(db_path)
    known = existing_doc_ids([t["doc_id"] for t in turns], db_path=db_path)
    seen = set()
    new_turns = []
    for t in turns:
        if t["doc_id"] in known or t["doc_id"] in seen:
            continue
        seen.add(t["doc_id"])
        new_turns.append(t)
    if not new_turns:
        return 0

    embeddings = _embedder.embed_documents([t["text"] for t in new_turns])
    data = []
    for turn, emb in zip(new_turns, embeddings):
        row = {
            "id": _primary_key(turn["doc_id"]),
            "vector": emb,
            "document": turn["text"][:65535],
            "doc_id": turn["doc_id"],
        }
        for col in METADATA_FIELDS:
            row[col] = turn.get(col, 0 if col == "turn_index" else "")
        data.append(row)

    with milvus_client(db_path) as client:
        client.insert(collection_name=COLLECTION_NAME, data=data)

    try:
        conn = _fts.connection(db_path)
        _fts.insert(conn, [_fts_record(t) for t in new_turns])
        _fts.close_ephemeral(conn)
    except Exception as exc:
        logger.warning("FTS insert failed (non-fatal, backfilled at next start): %s", exc)
    return len(data)


# --- Search -------------------------------------------------------------------

def _hit_to_result(entity: Dict, similarity: Optional[float]) -> Dict:
    result = {"content": entity.get("document", ""), "doc_id": entity.get("doc_id", "")}
    for col in METADATA_FIELDS:
        result[col] = entity.get(col, 0 if col == "turn_index" else "")
    result["similarity"] = similarity
    return result


def _vector_search(client: MilvusClient, query_vec: List[float], limit: int,
                   filter_expr: Optional[str]) -> List[Dict]:
    res = client.search(collection_name=COLLECTION_NAME, data=[query_vec], limit=limit,
                        filter=filter_expr, output_fields=OUTPUT_FIELDS)
    hits = res[0] if res else []
    return [_hit_to_result(h["entity"], float(h["distance"])) for h in hits]


def _fill_similarity(client: MilvusClient, query_vec: List[float], results: List[Dict]) -> None:
    """Compute cosine similarity for FTS-only hits so every result has one."""
    missing = [r["doc_id"] for r in results if r.get("similarity") is None and r.get("doc_id")]
    if not missing:
        return
    try:
        rows = _query(client, _ids_filter(missing), ["doc_id", "vector"], limit=len(missing))
    except Exception as exc:
        logger.debug("Could not fetch vectors for FTS hits: %s", exc)
        return
    vectors = {r["doc_id"]: r["vector"] for r in rows}
    for r in results:
        vec = vectors.get(r.get("doc_id"))
        if r.get("similarity") is None and vec is not None:
            r["similarity"] = float(sum(a * b for a, b in zip(query_vec, vec)))


def _apply_recency_boost(results: List[Dict]) -> None:
    """score *= 1 + w * recency, recency = rank of timestamp among candidates in [0,1]."""
    stamps = sorted({r["timestamp"] for r in results if r.get("timestamp")})
    if len(stamps) < 2:
        return
    rank = {ts: i / (len(stamps) - 1) for i, ts in enumerate(stamps)}
    for r in results:
        recency = rank.get(r.get("timestamp"), 0.5)
        r["score"] = r["score"] * (1.0 + RECENCY_WEIGHT * recency)


def search(query: str, n: int = 5, session_id: Optional[str] = None,
           git_branch: Optional[str] = None, project_root: Optional[str] = None,
           recency_boost: bool = False, db_path: Optional[str] = None,
           project_prefix: bool = True) -> List[Dict]:
    """Hybrid search. `project_root=None` searches every project; otherwise the
    project and (with project_prefix) anything launched from a subdirectory of it.

    Each result carries `score` (RRF fusion score normalised to (0, 1], boosted
    by recency when requested) and `similarity` (cosine, 1.0 = identical).
    """
    n = max(1, min(int(n), MAX_RESULTS))
    fetch_n = min(max(n * 3, 10), 60)
    db_path = _resolve_db_path(db_path)
    query_vec = _embedder.embed_query(query)

    with milvus_client(db_path) as client:
        filter_expr = _build_filter(session_id, git_branch, project_root, project_prefix)
        try:
            vector_results = _vector_search(client, query_vec, fetch_n, filter_expr)
        except Exception as exc:
            if project_root and project_prefix:
                logger.warning("Prefix filter failed (%s); retrying with exact project match", exc)
                filter_expr = _build_filter(session_id, git_branch, project_root, False)
                vector_results = _vector_search(client, query_vec, fetch_n, filter_expr)
                project_prefix = False
            else:
                raise

        fts_filters = {k: v for k, v in (("session_id", session_id), ("git_branch", git_branch)) if v}
        prefix_filters = {}
        if project_root:
            if project_prefix:
                prefix_filters["project_root"] = project_root
            else:
                fts_filters["project_root"] = project_root
        fts_results = _fts.search(query, n=fetch_n, filters=fts_filters or None,
                                  prefix_filters=prefix_filters or None, db_path=db_path)

        merged = rrf_merge(vector_results, fts_results, n=fetch_n, k=RRF_K)
        if not merged:
            return []
        _fill_similarity(client, query_vec, merged)

    max_rrf = 2.0 / (RRF_K + 1)
    for r in merged:
        r["score"] = r.pop("_rrf_score", 0.0) / max_rrf
        r.pop("bm25", None)
        r.setdefault("content", "")
    if recency_boost:
        _apply_recency_boost(merged)
    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged[:n]


def get_turns(session_id: str, turn_index: int, context: int = 2,
              db_path: Optional[str] = None, transcript_file: Optional[str] = None) -> List[Dict]:
    """Return the chunks around `turn_index` (a byte offset) in one transcript of a session."""
    context = max(0, min(int(context), 20))
    with milvus_client(db_path) as client:
        rows = _query(client, _build_filter(session_id=session_id), OUTPUT_FIELDS)
    if not rows:
        return []
    if transcript_file:
        rows = [r for r in rows if r.get("transcript_file") == transcript_file] or rows
    rows.sort(key=lambda r: (r.get("transcript_file", ""), r.get("turn_index", 0)))
    target = min(range(len(rows)), key=lambda i: abs(rows[i].get("turn_index", 0) - turn_index))
    target_file = rows[target].get("transcript_file", "")
    rows = [r for r in rows if r.get("transcript_file", "") == target_file]
    target = min(range(len(rows)), key=lambda i: abs(rows[i].get("turn_index", 0) - turn_index))
    window = rows[max(0, target - context): target + context + 1]
    return [_hit_to_result(r, None) for r in window]


# --- Stats ----------------------------------------------------------------------

def get_stats(project_root: Optional[str] = None, db_path: Optional[str] = None,
              project_prefix: bool = True) -> Dict:
    filter_expr = _project_filter(project_root, project_prefix) if project_root else None
    rows = scan_all(["session_id", "chunk_type", "git_branch", "project_root"],
                    filter_expr=filter_expr, db_path=db_path)
    sessions = {r["session_id"] for r in rows if r.get("session_id")}
    branches = {r["git_branch"] for r in rows if r.get("git_branch")}
    projects = {r["project_root"] for r in rows if r.get("project_root")}
    by_type: Dict[str, int] = {}
    for r in rows:
        by_type[r.get("chunk_type") or "unknown"] = by_type.get(r.get("chunk_type") or "unknown", 0) + 1
    return {"total_turns": len(rows), "sessions": len(sessions), "branches": sorted(branches),
            "projects": sorted(projects), "by_type": by_type}


def list_sessions(project_root: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict]:
    filter_expr = _project_filter(project_root, True) if project_root else None
    rows = scan_all(["session_id", "timestamp", "git_branch", "chunk_type", "project_root"],
                    filter_expr=filter_expr, db_path=db_path)
    sessions: Dict[str, Dict] = {}
    for r in rows:
        sid = r.get("session_id")
        if not sid:
            continue
        s = sessions.setdefault(sid, {"session_id": sid, "turns": 0, "branches": set(),
                                      "min_ts": "", "max_ts": "", "project_root": r.get("project_root", "")})
        s["turns"] += 1
        if r.get("git_branch"):
            s["branches"].add(r["git_branch"])
        ts = r.get("timestamp", "")
        if ts:
            s["min_ts"] = min(s["min_ts"] or ts, ts)
            s["max_ts"] = max(s["max_ts"], ts)
    result = []
    for s in sessions.values():
        s["branches"] = sorted(s["branches"])
        result.append(s)
    result.sort(key=lambda s: s["max_ts"], reverse=True)
    return result


# --- Deletes --------------------------------------------------------------------

def _delete_where(filter_expr: str, fts_where: str, fts_params: tuple,
                  db_path: Optional[str] = None) -> int:
    db_path = _resolve_db_path(db_path)
    with milvus_client(db_path) as client:
        if not client.has_collection(COLLECTION_NAME):
            return 0
        before = _count_with(client, filter_expr)
        if before:
            client.delete(collection_name=COLLECTION_NAME, filter=filter_expr)
    try:
        conn = _fts.connection(db_path)
        _fts.delete_where(conn, fts_where, fts_params)
        _fts.close_ephemeral(conn)
    except Exception as exc:
        logger.warning("FTS delete failed (non-fatal): %s", exc)
    return before


def delete_by_session(session_id: str, db_path: Optional[str] = None) -> int:
    return _delete_where(f"session_id == {_literal(session_id)}", "session_id = ?", (session_id,), db_path)


def delete_by_branch(git_branch: str, db_path: Optional[str] = None) -> int:
    return _delete_where(f"git_branch == {_literal(git_branch)}", "git_branch = ?", (git_branch,), db_path)


def delete_older_than(max_age_days: int, db_path: Optional[str] = None) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(max_age_days))).strftime("%Y-%m-%dT%H:%M:%S")
    return _delete_where(f'timestamp < "{cutoff}" and timestamp != ""',
                         "timestamp < ? AND timestamp != ''", (cutoff,), db_path)


def delete_noise(db_path: Optional[str] = None) -> int:
    """Remove chunks that only contain Claude Code command/markup echoes."""
    db_path = _resolve_db_path(db_path)
    total = 0
    for prefix in NOISE_PREFIXES:
        like = prefix.replace("%", "") + "%"
        total += _delete_where(f"document like {_literal(like)}", "content LIKE ? ESCAPE '\\'",
                               (like.replace("_", "\\_"),), db_path)
    return total


def clear_collection(db_path: Optional[str] = None) -> None:
    """Drop the collection and the FTS database (full reset)."""
    db_path = _resolve_db_path(db_path)
    with milvus_client(db_path, ensure=False) as client:
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            logger.info("Collection dropped: %s", COLLECTION_NAME)
    _fts.clear(db_path)
    if IDENTITY_FILE.exists():
        IDENTITY_FILE.unlink()


# --- Maintenance ------------------------------------------------------------------

def backfill_fts(db_path: Optional[str] = None) -> int:
    """Insert into FTS any Milvus rows it is missing (e.g. after an FTS schema rebuild)."""
    db_path = _resolve_db_path(db_path)
    rows = scan_all(["doc_id", "document"] + METADATA_FIELDS, db_path=db_path)
    if not rows:
        return 0
    conn = _fts.connection(db_path)
    try:
        present = _fts.existing_doc_ids(conn, [r["doc_id"] for r in rows if r.get("doc_id")])
        missing = [r for r in rows if r.get("doc_id") and r["doc_id"] not in present]
        if not missing:
            return 0
        records = []
        for r in missing:
            rec = {"doc_id": r["doc_id"], "content": r.get("document", "")}
            for col in FTS_METADATA:
                rec[col] = r.get(col, 0 if col == "turn_index" else "")
            records.append(rec)
        inserted = _fts.insert(conn, records)
    finally:
        _fts.close_ephemeral(conn)
    logger.info("FTS backfill inserted %d records", inserted)
    return inserted


def reembed_all(db_path: Optional[str] = None, batch_size: int = 64,
                progress: Optional[Callable[[str], None]] = None,
                skip_noise: bool = True) -> int:
    """Rebuild every vector with the active model, using the FTS table as the text source.

    All embeddings are computed before the old collection is dropped, so a
    failure part-way leaves the existing index untouched.
    """
    db_path = _resolve_db_path(db_path)
    log = progress or (lambda msg: logger.info(msg))
    conn = _fts.connection(db_path)
    try:
        total = _fts.count(conn)
        log(f"Re-embedding {total} chunks with {_spec.name} ({_spec.model_id}, {_spec.dim}d)...")
        rows_out: List[Dict] = []
        done = 0
        started = time.time()
        for batch in _fts.iter_all(conn, batch_size=batch_size):
            if skip_noise:
                batch = [r for r in batch if not str(r.get("content", "")).startswith(NOISE_PREFIXES)]
            if not batch:
                continue
            vectors = _embedder.embed_documents([r["content"] for r in batch])
            for r, vec in zip(batch, vectors):
                row = {"id": _primary_key(r["doc_id"]), "vector": vec,
                       "document": r["content"][:65535], "doc_id": r["doc_id"]}
                for col in METADATA_FIELDS:
                    row[col] = r.get(col, 0 if col == "turn_index" else "")
                row["turn_index"] = int(row["turn_index"] or 0)
                rows_out.append(row)
            done += len(batch)
            if done % (batch_size * 8) < len(batch):
                rate = done / max(time.time() - started, 1e-6)
                log(f"  {done}/{total} embedded ({rate:.0f}/s)")
        if skip_noise and done < total:
            skipped = [r["doc_id"] for b in _fts.iter_all(conn) for r in b
                       if str(r.get("content", "")).startswith(NOISE_PREFIXES)]
            if skipped:
                _fts.delete_doc_ids(conn, skipped)
                log(f"  dropped {len(skipped)} noise chunks")
    finally:
        _fts.close_ephemeral(conn)

    with milvus_client(db_path, ensure=False) as client:
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
        _ensure_collection(client)
        for i in range(0, len(rows_out), 500):
            client.insert(collection_name=COLLECTION_NAME, data=rows_out[i:i + 500])
    stamp_identity()
    log(f"Done: {len(rows_out)} chunks re-embedded")
    return len(rows_out)
