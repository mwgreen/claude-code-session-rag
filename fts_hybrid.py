"""
FTS5 hybrid search sidecar for Milvus-based RAG engines.

Provides a SQLite FTS5 full-text index alongside a Milvus vector store,
with Reciprocal Rank Fusion (RRF) to merge both result sets.

Usage:
    fts = FTSIndex("turns_fts", ["session_id", "git_branch", "turn_index", "timestamp", "chunk_type"])
    fts.set_server_mode(True)

    # Insert
    conn = fts.connection(db_path)
    fts.insert(conn, [{"doc_id": "x", "content": "hello", "session_id": "s1", ...}])

    # Search
    results = fts.search("hello", n=10, filters={"session_id": "s1"}, db_path=db_path)

    # Merge with vector results
    merged = rrf_merge(vector_results, fts_results, n=10)
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("fts-hybrid")


class FTSIndex:
    """SQLite FTS5 full-text search index, designed as a sidecar to Milvus.

    Args:
        table_name: FTS5 virtual table name (e.g. "turns_fts", "chunks_fts")
        metadata_columns: Column names beyond doc_id and content
        indexed_metadata: Set of metadata column names that should be full-text indexed
            (searchable via MATCH). All others are UNINDEXED (stored but not searchable).
    """

    def __init__(self, table_name: str, metadata_columns: List[str],
                 indexed_metadata: Optional[Set[str]] = None):
        self.table_name = table_name
        self.metadata_columns = metadata_columns
        self._indexed_metadata = indexed_metadata or set()
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._server_mode = False

        # Pre-compute SQL fragments
        meta_defs_parts = []
        for col in metadata_columns:
            if col in self._indexed_metadata:
                meta_defs_parts.append(col)  # indexed (searchable)
            else:
                meta_defs_parts.append(f"{col} UNINDEXED")
        meta_defs = ", ".join(meta_defs_parts)
        self._create_sql = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name} USING fts5("
            f"doc_id, content, {meta_defs})"
        )
        all_cols = ["doc_id", "content"] + metadata_columns
        self._insert_cols = ", ".join(all_cols)
        self._insert_placeholders = ", ".join("?" for _ in all_cols)
        # bm25 weights: 0 for doc_id, 1.0 for content, 0.5 for indexed metadata, 0 for unindexed
        bm25_parts = ["0", "1"]  # doc_id, content
        for col in metadata_columns:
            bm25_parts.append("0.5" if col in self._indexed_metadata else "0")
        bm25_weights = ", ".join(bm25_parts)
        self._select_cols = ", ".join(all_cols)
        self._bm25_call = f"bm25({table_name}, {bm25_weights})"

    def set_server_mode(self, enabled: bool):
        self._server_mode = enabled

    def close_all(self):
        """Close all persistent connections."""
        for path, conn in list(self._connections.items()):
            try:
                conn.close()
                logger.info("Closed FTS connection: %s", path)
            except Exception as e:
                logger.warning("Error closing FTS connection %s: %s", path, e)
        self._connections.clear()

    @staticmethod
    def db_path(milvus_db_path: str) -> str:
        """Derive the FTS database path from the Milvus DB path."""
        return str(Path(milvus_db_path).parent / "fts.db")

    def _check_and_migrate(self, conn: sqlite3.Connection):
        """Check schema version and recreate FTS table if needed.

        Stores the CREATE SQL in a _fts_schema table. If the schema changes
        (e.g. column added or UNINDEXED -> indexed), drops and recreates the
        FTS table. Data will be rebuilt organically via file watcher.
        """
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _fts_schema (
                table_name TEXT PRIMARY KEY,
                create_sql TEXT NOT NULL
            )
        """)

        row = conn.execute(
            "SELECT create_sql FROM _fts_schema WHERE table_name = ?",
            (self.table_name,)
        ).fetchone()

        if row and row[0] == self._create_sql:
            # Schema matches — just ensure table exists
            conn.execute(self._create_sql)
            conn.commit()
            return

        if row:
            # Schema changed — drop and recreate
            logger.info("FTS schema changed for %s, rebuilding...", self.table_name)
            conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")

        conn.execute(self._create_sql)
        conn.execute(
            "INSERT OR REPLACE INTO _fts_schema (table_name, create_sql) VALUES (?, ?)",
            (self.table_name, self._create_sql)
        )
        conn.commit()

    def connection(self, milvus_db_path: str) -> sqlite3.Connection:
        """Get or create a connection. Persistent in server mode, ephemeral otherwise."""
        fts_path = self.db_path(milvus_db_path)

        if fts_path in self._connections:
            try:
                self._connections[fts_path].execute("SELECT 1")
                return self._connections[fts_path]
            except Exception:
                logger.warning("Stale FTS connection for %s — reconnecting", fts_path)
                try:
                    self._connections[fts_path].close()
                except Exception:
                    pass
                del self._connections[fts_path]

        Path(fts_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(fts_path)
        conn.execute("PRAGMA journal_mode=WAL")
        self._check_and_migrate(conn)

        if self._server_mode:
            self._connections[fts_path] = conn
            logger.info("Opened FTS connection: %s", fts_path)

        return conn

    def close_ephemeral(self, conn: sqlite3.Connection):
        """Close a connection if not in server mode (i.e. ephemeral)."""
        if not self._server_mode:
            conn.close()

    def insert(self, conn: sqlite3.Connection, records: List[Dict]):
        """Insert records into FTS. Skips duplicates by doc_id.

        Each record must have 'doc_id', 'content', and each metadata column.
        """
        for rec in records:
            doc_id = rec["doc_id"]
            existing = conn.execute(
                f"SELECT doc_id FROM {self.table_name} WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if existing:
                continue
            values = [
                doc_id,
                rec.get("content", "")[:65535],
            ] + [rec.get(col, "") for col in self.metadata_columns]
            conn.execute(
                f"INSERT INTO {self.table_name} ({self._insert_cols}) VALUES ({self._insert_placeholders})",
                values,
            )
        conn.commit()

    def delete(self, conn: sqlite3.Connection, column: str, value):
        """Delete rows where column == value."""
        conn.execute(f"DELETE FROM {self.table_name} WHERE {column} = ?", (value,))
        conn.commit()

    def delete_where(self, conn: sqlite3.Connection, where_clause: str, params: tuple):
        """Delete rows matching an arbitrary WHERE clause."""
        conn.execute(f"DELETE FROM {self.table_name} WHERE {where_clause}", params)
        conn.commit()

    def search(self, query: str, n: int = 15,
               filters: Optional[Dict[str, str]] = None,
               db_path: Optional[str] = None) -> List[Dict]:
        """Full-text search with BM25 ranking.

        Args:
            query: Search text
            n: Max results
            filters: Dict of column -> value for exact-match filtering
            db_path: Milvus DB path (used to derive FTS path)

        Returns:
            List of result dicts with content, doc_id, metadata columns, and distance=0.0
        """
        if not db_path:
            return []

        try:
            conn = self.connection(db_path)
        except Exception as e:
            logger.warning("FTS connection failed: %s", e)
            return []

        safe_query = query.replace('"', '""')

        where_parts = []
        params: list = []
        if filters:
            for col, val in filters.items():
                if val is not None:
                    where_parts.append(f"{col} = ?")
                    params.append(val)

        where_clause = (" AND " + " AND ".join(where_parts)) if where_parts else ""

        sql = (
            f"SELECT {self._select_cols}, {self._bm25_call} as rank "
            f"FROM {self.table_name} "
            f"WHERE {self.table_name} MATCH ?{where_clause} "
            f"ORDER BY rank LIMIT ?"
        )
        params_full = [safe_query] + params + [n]

        try:
            rows = conn.execute(sql, params_full).fetchall()
        except Exception as e:
            logger.debug("FTS5 MATCH failed (%s), trying quoted phrase", e)
            params_full[0] = f'"{safe_query}"'
            try:
                rows = conn.execute(sql, params_full).fetchall()
            except Exception as e2:
                logger.warning("FTS5 search failed: %s", e2)
                return []

        all_cols = ["doc_id", "content"] + self.metadata_columns
        results = []
        for row in rows:
            result = {"distance": 0.0}
            for i, col in enumerate(all_cols):
                result[col] = row[i]
            results.append(result)

        if not self._server_mode:
            conn.close()

        return results

    def clear(self, db_path: str):
        """Delete the FTS database file. Closes persistent connection first."""
        fts_path = self.db_path(db_path)
        if fts_path in self._connections:
            try:
                self._connections[fts_path].close()
            except Exception:
                pass
            del self._connections[fts_path]
        fts_file = Path(fts_path)
        if fts_file.exists():
            fts_file.unlink()
            logger.info("FTS database deleted: %s", fts_path)
        # Also clean up WAL/SHM files
        for suffix in ("-wal", "-shm"):
            wal = Path(fts_path + suffix)
            if wal.exists():
                wal.unlink()


def rrf_merge(vector_results: List[Dict], fts_results: List[Dict],
              n: int, k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion: merge two ranked result lists.

    score(doc) = sum(1 / (k + rank)) across both lists.
    k=60 is the standard constant from the original RRF paper.
    Results are keyed by doc_id (falls back to enumeration index).
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict] = {}

    for rank, r in enumerate(vector_results):
        key = r.get("doc_id", str(rank))
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in docs:
            docs[key] = r

    for rank, r in enumerate(fts_results):
        key = r.get("doc_id", str(rank))
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in docs:
            docs[key] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    merged = []
    for key, score in ranked[:n]:
        result = docs[key].copy()
        result["_rrf_score"] = score
        merged.append(result)

    return merged
