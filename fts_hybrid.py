"""
SQLite FTS5 keyword-search sidecar for the Milvus vector store.

The FTS table mirrors every indexed chunk (doc_id, content, metadata). Hybrid
search runs a BM25 query here and a vector query in Milvus, then merges both
ranked lists with Reciprocal Rank Fusion (RRF).

Because the FTS table stores the full text of every chunk it is also the
source of truth for re-embedding when the embedding model changes.

Thread-safety: connections are opened with check_same_thread=False and every
operation holds a lock; the server funnels all engine work through one worker
thread anyway.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("session-rag.fts")

# unicode61 + '_' as a token character keeps snake_case identifiers whole.
DEFAULT_TOKENIZER = "unicode61 remove_diacritics 2 tokenchars '_'"

_STOPWORDS = frozenset("""
a an and are as at be by for from has have how i in is it its of on or that the this to
was we what when where which who why will with you your do does did can could should would
me my our us them they their there here about into over under than then so if not no yes
""".split())
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def build_match_query(query: str, prefix_min_len: int = 4) -> Optional[str]:
    """Turn free text into a tolerant FTS5 MATCH expression.

    Terms are quoted (so punctuation like ':' or '-' cannot break the FTS5
    syntax) and OR-ed, letting BM25 rank partial matches. Longer terms also
    match as prefixes ("deploy"* hits "deployment").
    """
    terms: List[str] = []
    seen: Set[str] = set()
    for tok in _TOKEN_RE.findall(query):
        low = tok.lower()
        if len(low) < 2 or low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        quoted = '"' + tok.replace('"', '""') + '"'
        terms.append(quoted + "*" if len(low) >= prefix_min_len else quoted)
        if len(terms) >= 24:
            break
    if not terms:
        return None
    return " OR ".join(terms)


class FTSIndex:
    def __init__(self, table_name: str, metadata_columns: List[str],
                 indexed_metadata: Optional[Set[str]] = None,
                 tokenizer: str = DEFAULT_TOKENIZER):
        self.table_name = table_name
        self.metadata_columns = list(metadata_columns)
        self._indexed_metadata = set(indexed_metadata or ())
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._server_mode = False
        self._lock = threading.RLock()

        meta_defs = ", ".join(
            col if col in self._indexed_metadata else f"{col} UNINDEXED"
            for col in self.metadata_columns
        )
        self._create_sql = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name} USING fts5("
            f"doc_id, content, {meta_defs}, tokenize=\"{tokenizer}\")"
        )
        self.all_columns = ["doc_id", "content"] + self.metadata_columns
        self._insert_sql = (
            f"INSERT INTO {table_name} ({', '.join(self.all_columns)}) "
            f"VALUES ({', '.join('?' for _ in self.all_columns)})"
        )
        weights = ["0", "1"] + ["0.5" if c in self._indexed_metadata else "0" for c in self.metadata_columns]
        self._bm25_call = f"bm25({table_name}, {', '.join(weights)})"

    # -- lifecycle --------------------------------------------------------
    def set_server_mode(self, enabled: bool) -> None:
        self._server_mode = enabled

    @staticmethod
    def db_path(milvus_db_path: str) -> str:
        return str(Path(milvus_db_path).parent / "fts.db")

    def close_all(self) -> None:
        with self._lock:
            for path, conn in list(self._connections.items()):
                try:
                    conn.close()
                except Exception as exc:
                    logger.warning("Error closing FTS connection %s: %s", path, exc)
            self._connections.clear()

    def _check_and_migrate(self, conn: sqlite3.Connection) -> None:
        """Recreate the FTS table when its definition (columns/tokenizer) changed.
        The server repopulates it from Milvus at startup (backfill_fts)."""
        conn.execute("CREATE TABLE IF NOT EXISTS _fts_schema (table_name TEXT PRIMARY KEY, create_sql TEXT NOT NULL)")
        row = conn.execute("SELECT create_sql FROM _fts_schema WHERE table_name = ?", (self.table_name,)).fetchone()
        if row and row[0] == self._create_sql:
            conn.execute(self._create_sql)
            conn.commit()
            return
        if row:
            logger.warning("FTS schema changed for %s; rebuilding table (repopulated from Milvus)", self.table_name)
            conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        conn.execute(self._create_sql)
        conn.execute("INSERT OR REPLACE INTO _fts_schema (table_name, create_sql) VALUES (?, ?)",
                     (self.table_name, self._create_sql))
        conn.commit()

    def connection(self, milvus_db_path: str) -> sqlite3.Connection:
        """Persistent connection in server mode, fresh connection otherwise."""
        fts_path = self.db_path(milvus_db_path)
        with self._lock:
            conn = self._connections.get(fts_path)
            if conn is not None:
                try:
                    conn.execute("SELECT 1")
                    return conn
                except sqlite3.Error as exc:
                    logger.warning("FTS connection unusable (%s); reconnecting", exc)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    del self._connections[fts_path]

            Path(fts_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(fts_path, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._check_and_migrate(conn)
            if self._server_mode:
                self._connections[fts_path] = conn
            return conn

    def close_ephemeral(self, conn: sqlite3.Connection) -> None:
        if not self._server_mode:
            conn.close()

    # -- writes -----------------------------------------------------------
    def existing_doc_ids(self, conn: sqlite3.Connection, doc_ids: Sequence[str]) -> Set[str]:
        found: Set[str] = set()
        ids = list(doc_ids)
        with self._lock:
            for i in range(0, len(ids), 400):
                chunk = ids[i:i + 400]
                rows = conn.execute(
                    f"SELECT doc_id FROM {self.table_name} WHERE doc_id IN ({','.join('?' * len(chunk))})",
                    chunk,
                ).fetchall()
                found.update(r[0] for r in rows)
        return found

    def insert(self, conn: sqlite3.Connection, records: Iterable[Dict]) -> int:
        """Insert records (dicts with doc_id, content and metadata). Skips existing doc_ids."""
        records = list(records)
        if not records:
            return 0
        with self._lock:
            existing = self.existing_doc_ids(conn, [r["doc_id"] for r in records])
            rows = []
            seen: Set[str] = set()
            for rec in records:
                doc_id = rec["doc_id"]
                if doc_id in existing or doc_id in seen:
                    continue
                seen.add(doc_id)
                rows.append([doc_id, (rec.get("content") or "")[:65535]]
                            + [rec.get(col, "") for col in self.metadata_columns])
            if rows:
                conn.executemany(self._insert_sql, rows)
            conn.commit()
        return len(rows)

    def delete(self, conn: sqlite3.Connection, column: str, value) -> int:
        with self._lock:
            cur = conn.execute(f"DELETE FROM {self.table_name} WHERE {column} = ?", (value,))
            conn.commit()
            return cur.rowcount

    def delete_where(self, conn: sqlite3.Connection, where_clause: str, params: tuple) -> int:
        with self._lock:
            cur = conn.execute(f"DELETE FROM {self.table_name} WHERE {where_clause}", params)
            conn.commit()
            return cur.rowcount

    def delete_doc_ids(self, conn: sqlite3.Connection, doc_ids: Sequence[str]) -> int:
        ids = list(doc_ids)
        deleted = 0
        with self._lock:
            for i in range(0, len(ids), 400):
                chunk = ids[i:i + 400]
                cur = conn.execute(
                    f"DELETE FROM {self.table_name} WHERE doc_id IN ({','.join('?' * len(chunk))})", chunk)
                deleted += cur.rowcount
            conn.commit()
        return deleted

    # -- reads ------------------------------------------------------------
    def count(self, conn: sqlite3.Connection) -> int:
        with self._lock:
            return conn.execute(f"SELECT count(*) FROM {self.table_name}").fetchone()[0]

    def iter_all(self, conn: sqlite3.Connection, batch_size: int = 500) -> Iterator[List[Dict]]:
        """Yield all rows in batches (used for re-embedding)."""
        cols = ", ".join(self.all_columns)
        with self._lock:
            cur = conn.execute(f"SELECT {cols} FROM {self.table_name}")
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield [dict(zip(self.all_columns, r)) for r in rows]

    def search(self, query: str, n: int = 15,
               filters: Optional[Dict[str, str]] = None,
               prefix_filters: Optional[Dict[str, str]] = None,
               db_path: Optional[str] = None) -> List[Dict]:
        """BM25 search. `filters` are exact matches; `prefix_filters` match the
        value itself or any path below it (value + '/...')."""
        if not db_path:
            return []
        match = build_match_query(query)
        if not match:
            return []
        try:
            conn = self.connection(db_path)
        except Exception as exc:
            logger.warning("FTS connection failed: %s", exc)
            return []

        where_parts: List[str] = []
        params: list = [match]
        for col, val in (filters or {}).items():
            if val is not None:
                where_parts.append(f"{col} = ?")
                params.append(val)
        for col, val in (prefix_filters or {}).items():
            if val:
                where_parts.append(f"({col} = ? OR {col} LIKE ? ESCAPE '\\')")
                params.append(val)
                params.append(_like_prefix(val.rstrip("/") + "/"))
        where_clause = (" AND " + " AND ".join(where_parts)) if where_parts else ""
        sql = (
            f"SELECT {', '.join(self.all_columns)}, {self._bm25_call} AS rank "
            f"FROM {self.table_name} WHERE {self.table_name} MATCH ?{where_clause} "
            f"ORDER BY rank LIMIT ?"
        )
        params.append(n)
        try:
            with self._lock:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            logger.warning("FTS5 search failed: %s (query=%r)", exc, match)
            rows = []
        finally:
            self.close_ephemeral(conn)

        results = []
        for row in rows:
            rec = dict(zip(self.all_columns, row))
            rec["bm25"] = float(row[len(self.all_columns)])
            results.append(rec)
        return results

    def clear(self, db_path: str) -> None:
        """Delete the FTS database file (and WAL/SHM)."""
        fts_path = self.db_path(db_path)
        with self._lock:
            conn = self._connections.pop(fts_path, None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        for suffix in ("", "-wal", "-shm"):
            p = Path(fts_path + suffix)
            if p.exists():
                p.unlink()
        logger.info("FTS database deleted: %s", fts_path)


def _like_prefix(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def rrf_merge(vector_results: List[Dict], fts_results: List[Dict],
              n: int, k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion of two ranked lists keyed by doc_id.

    score(doc) = sum over lists of 1 / (k + rank). Fields from the vector hit
    (e.g. similarity) are kept when a doc appears in both lists.
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict] = {}
    for ranked in (vector_results, fts_results):
        for rank, r in enumerate(ranked):
            key = r.get("doc_id") or f"_{id(r)}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in docs:
                docs[key] = dict(r)
            else:
                for field, value in r.items():
                    docs[key].setdefault(field, value)
    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    merged = []
    for key in ranked_keys[:n]:
        doc = docs[key]
        doc["_rrf_score"] = scores[key]
        merged.append(doc)
    return merged
