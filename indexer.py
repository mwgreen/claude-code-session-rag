"""
The single indexing pipeline.

Every source of "this transcript changed" (file watcher events, the Stop /
PreCompact hooks, session-start backfills) feeds one queue that is drained by
one loop, one file at a time. That removes the races the old design had between
the watcher, the /index handler and the backfill all reading and writing the
same offset state concurrently.

Layout of ~/.claude/projects/:
    <slug>/<session>.jsonl                      main transcript   -> chunk_type "turn"
    <slug>/<session>/subagents/agent-*.jsonl    subagent work     -> chunk_type "subagent"
    <slug>/<session>/wf_*/...jsonl              workflow agents   -> chunk_type "subagent"
Nested transcripts are attributed to the parent session id so a search scoped
to the current session also finds what its subagents did.

Project root resolution (first match wins): hint from a hook -> previously
recorded root for this file -> slug map -> git toplevel of the `cwd` recorded
inside the transcript -> "".
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rag_engine
import transcript_parser
from index_state import IndexState, SlugMap

logger = logging.getLogger("session-rag.indexer")

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
_STATE_SAVE_INTERVAL = 5.0


def classify_transcript(path: str) -> Tuple[Optional[str], str, str]:
    """Return (slug, session_id, chunk_type) for a transcript path."""
    p = Path(path)
    try:
        rel = p.resolve().relative_to(CLAUDE_PROJECTS.resolve())
    except (ValueError, OSError):
        return None, p.stem, "turn"
    parts = rel.parts
    if len(parts) < 2:
        return None, p.stem, "turn"
    slug = parts[0]
    if len(parts) == 2:
        return slug, p.stem, "turn"
    return slug, parts[1], "subagent"


def _git_toplevel(cwd: str) -> str:
    """Project root for a working directory: git toplevel, else the directory itself."""
    if not cwd:
        return ""
    if os.path.isdir(cwd):
        try:
            res = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                                 capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return cwd.rstrip("/") or cwd


class Indexer:
    def __init__(self, db_path: str, debounce_seconds: float = 2.0):
        self.db_path = db_path
        self.debounce_seconds = debounce_seconds
        self.state = IndexState()
        self.slug_map = SlugMap()
        self._live: Dict[str, dict] = {}        # path -> {"due": t, "project_root": .., "session_id": ..}
        self._backlog: List[str] = []           # backfill queue (paths)
        self._backlog_set: set = set()
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._processing = False
        self._cwd_roots: Dict[str, str] = {}
        self._last_save = time.time()
        self.stats = {
            "turns_indexed": 0, "files_processed": 0, "batches_processed": 0,
            "errors": 0, "backfill_scheduled": 0, "last_indexed_at": None,
        }

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        if self._task is None:
            self._stopped = False
            self._task = asyncio.create_task(self._run(), name="indexer")

    async def stop(self) -> None:
        self._stopped = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self._save_state(force=True)
        logger.info("Indexer stopped (indexed=%d files=%d errors=%d)",
                    self.stats["turns_indexed"], self.stats["files_processed"], self.stats["errors"])

    # -- inputs -----------------------------------------------------------
    def register_project(self, project_root: str, slug: Optional[str] = None) -> bool:
        project_root = (project_root or "").rstrip("/")
        if not project_root:
            return False
        return self.slug_map.register(project_root, slug)

    def enqueue(self, path: str, project_root: str = "", session_id: str = "",
                immediate: bool = False) -> None:
        """Schedule a transcript for (re)indexing after the debounce window."""
        if not path.endswith(".jsonl"):
            return
        due = time.monotonic() + (0.0 if immediate else self.debounce_seconds)
        item = self._live.get(path)
        if item is None:
            self._live[path] = {"due": due, "project_root": project_root, "session_id": session_id}
        else:
            item["due"] = min(item["due"], due) if immediate else due
            item["project_root"] = item["project_root"] or project_root
            item["session_id"] = item["session_id"] or session_id
        self._wake.set()

    async def schedule_backfill(self, slug_filter: Optional[str] = None) -> int:
        """Queue every transcript that has unindexed bytes. Returns the number queued."""
        if not CLAUDE_PROJECTS.is_dir():
            return 0
        root = CLAUDE_PROJECTS / slug_filter if slug_filter else CLAUDE_PROJECTS
        if not root.is_dir():
            return 0

        def _collect() -> List[Tuple[float, str]]:
            found = []
            for p in root.rglob("*.jsonl"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                path = str(p)
                if self.state.get_offset(path) < st.st_size:
                    found.append((st.st_mtime, path))
            found.sort(reverse=True)  # newest sessions first
            return found

        candidates = await asyncio.get_running_loop().run_in_executor(None, _collect)
        if slug_filter is None:
            self._prune_missing()
        added = 0
        for _, path in candidates:
            if path not in self._backlog_set and path not in self._live:
                self._backlog.append(path)
                self._backlog_set.add(path)
                added += 1
        self.stats["backfill_scheduled"] += added
        scope = slug_filter or "all projects"
        logger.info("Backfill (%s): %d transcript(s) queued", scope, added)
        if added:
            self._wake.set()
        return added

    def _prune_missing(self) -> None:
        gone = [p for p in list(self.state.transcripts) if not os.path.exists(p)]
        for p in gone:
            self.state.forget(p)
        if gone:
            logger.info("Forgot offsets for %d deleted transcript(s)", len(gone))

    # -- status -----------------------------------------------------------
    def status(self) -> Dict:
        return {
            "pending": len(self._live),
            "backlog": len(self._backlog),
            "processing": self._processing,
            "registered_projects": len(self.slug_map),
            "stats": dict(self.stats),
        }

    # -- main loop --------------------------------------------------------
    async def _run(self) -> None:
        while not self._stopped:
            now = time.monotonic()
            due = [p for p, item in self._live.items() if item["due"] <= now]
            if due:
                await self._process_live(due)
                continue
            if self._backlog:
                path = self._backlog.pop(0)
                self._backlog_set.discard(path)
                await self._safe_index(path)
                self._save_state()
                await asyncio.sleep(0)
                continue
            timeout = None
            if self._live:
                timeout = max(0.05, min(item["due"] for item in self._live.values()) - now)
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    async def _process_live(self, paths: List[str]) -> None:
        self._processing = True
        try:
            for path in paths:
                item = self._live.pop(path, None) or {}
                await self._safe_index(path, item.get("project_root", ""), item.get("session_id", ""))
            self.stats["batches_processed"] += 1
            self._save_state(force=True)
        finally:
            self._processing = False

    async def _safe_index(self, path: str, project_root: str = "", session_id: str = "") -> int:
        try:
            return await self.index_file(path, project_root, session_id)
        except Exception as exc:
            self.stats["errors"] += 1
            logger.error("Error indexing %s: %s", Path(path).name, exc)
            return 0

    async def index_file(self, path: str, project_root_hint: str = "", session_id_hint: str = "") -> int:
        """Parse new bytes of one transcript and index them. Returns chunks inserted."""
        if not os.path.exists(path):
            self.state.forget(path)
            return 0
        slug, session_id, chunk_type = classify_transcript(path)
        if slug is None and session_id_hint:
            session_id = session_id_hint

        offset = self.state.get_offset(path)
        result = await rag_engine.run(transcript_parser.parse_transcript, path, session_id,
                                      offset, chunk_type=chunk_type)

        project_root = (project_root_hint or self.state.get_project_root(path)
                        or (self.slug_map.get(slug) if slug else ""))
        if not project_root and result.cwd:
            project_root = await self._root_for_cwd(result.cwd)
        if project_root and slug:
            self.slug_map.register(project_root, slug)

        inserted = 0
        if result.turns:
            for t in result.turns:
                t["project_root"] = project_root
            inserted = await rag_engine.run(rag_engine.add_turns, result.turns, db_path=self.db_path)
            if inserted:
                self.stats["turns_indexed"] += inserted
                self.stats["last_indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                logger.info("Indexed %d chunk(s) from %s (session %s%s)", inserted, Path(path).name,
                            session_id[:8], ", subagent" if chunk_type == "subagent" else "")
        self.stats["files_processed"] += 1
        self.state.set_offset(path, result.new_offset, project_root)
        return inserted

    async def _root_for_cwd(self, cwd: str) -> str:
        root = self._cwd_roots.get(cwd)
        if root is None:
            root = await asyncio.get_running_loop().run_in_executor(None, _git_toplevel, cwd)
            self._cwd_roots[cwd] = root
        return root

    def _save_state(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self._last_save >= _STATE_SAVE_INTERVAL:
            try:
                self.state.save()
            except OSError as exc:
                logger.error("Could not save index state: %s", exc)
            self._last_save = now

    # -- maintenance ------------------------------------------------------
    async def expire_old(self, max_age_days: int, interval_seconds: float = 86400) -> int:
        """Prune chunks older than max_age_days, at most once per interval."""
        if max_age_days <= 0:
            return 0
        last = float(self.state.get("last_expire_check", 0) or 0)
        if time.time() - last < interval_seconds:
            return 0
        deleted = await rag_engine.run(rag_engine.delete_older_than, max_age_days, db_path=self.db_path)
        self.state.set("last_expire_check", time.time())
        self._save_state(force=True)
        if deleted:
            logger.info("Expired %d chunk(s) older than %d days", deleted, max_age_days)
        return deleted


# --- Singleton -------------------------------------------------------------------

_indexer: Optional[Indexer] = None


def get_indexer() -> Optional[Indexer]:
    return _indexer


async def start_indexer(db_path: str, debounce_seconds: float) -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer(db_path, debounce_seconds)
        await _indexer.start()
    return _indexer


async def stop_indexer() -> None:
    global _indexer
    if _indexer is not None:
        await _indexer.stop()
        _indexer = None
